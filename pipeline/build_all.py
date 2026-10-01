"""Run the whole world-data pipeline (spec 5, milestone M1).

    python pipeline/build_all.py              # real data: OSM, USGS 3DEP, NAIP, ACS, LODES
    python pipeline/build_all.py --population-source footprints
                                              # real map data; households ESTIMATED from real
                                              # residential footprints (explicit, labeled fallback
                                              # when Census ACS/LODES are unreachable)
    python pipeline/build_all.py --synthetic  # offline FAKE world, same formats (dev/CI only)

Writes data/processed/ (docs/data_contract.md) and client/public/assets/.
Prints counts: buildings, road edges, households, persons, students per school.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pipeline.build_buildings import (
    apply_heroes,
    build_building_tiles,
    flatten_terrain_for_heroes,
    load_heroes,
    prepare_buildings,
    write_buildings_geojson,
)
from pipeline.build_population import (
    WorkModel,
    exit_probs_from_bearings,
    fallback_dist,
    footprint_units,
    resolve_schools,
    students_by_school,
    synthesize_population,
    write_population,
)
from pipeline.build_roads import (
    RoadNetwork,
    build_network,
    local_node_mask,
    write_network,
)
from pipeline.build_terrain import Terrain, TerrainBuild, write_terrain_outputs
from pipeline.common import (
    CONTRACT_VERSION,
    DataSourceUnavailable,
    TileGrid,
    ensure_dirs,
    log,
    now_iso,
    region_extent,
    tile_grid,
    today,
    write_json,
)
from pipeline.config import assumption, assumption_range, load_yaml, region
from pipeline.geo import PROJECTION, scene_origin
from pipeline.sources import attribution_lines

PROCESSED_OUTPUTS = [
    "region_meta.json",
    "network_nodes.parquet",
    "network_edges.parquet",
    "roads_drive.graphml",
    "roads.geojson",
    "buildings.geojson",
    "households.parquet",
    "persons.parquet",
    "schools_resolved.json",
    "exits.json",
    "terrain_meta.json",
]

SYNTHETIC_TEXTURE_PX = 512
LAND_COVER_SOURCE = {
    "name": "Overture Maps base/land_cover (ESA WorldCover 10 m derived), terrain splat-mask prior only",
    "license": "CC BY 4.0 (ESA WorldCover)",
    "kind": "overture",
    "attribution": "(c) ESA WorldCover project / Contains modified Copernicus Sentinel data processed by the ESA WorldCover consortium; Overture Maps Foundation",
}
SPLAT_PX_LIDAR = 1024  # splat mask size per tile when the 0.5 m lidar rasters exist (~1.1 m/px)
SYNTHETIC_TERRAIN_BUDGET = 400_000  # the synthetic DEM is a smooth 10 m grid: no need for 2M
POPULATION_SOURCES = ("acs", "footprints")
POPULATION_SOURCE_LABEL = {"acs": "acs_lodes", "footprints": "footprint_estimate", "synthetic": "synthetic"}


def clean_outputs(processed: Path, assets: Path) -> None:
    for name in PROCESSED_OUTPUTS:
        (processed / name).unlink(missing_ok=True)
    for sub in ("terrain", "buildings", "roads", "ground"):
        d = assets / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    (assets / "manifest.json").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Draco
# ---------------------------------------------------------------------------


def _gltf_transform_cmd() -> list[str] | None:
    """Locate gltf-transform (npx cache) once; None if Node/npm is unavailable."""
    if shutil.which("npx") is None:
        return None
    try:
        r = subprocess.run(["npx", "-y", "@gltf-transform/cli", "--version"], capture_output=True, text=True, timeout=240)
    except (subprocess.SubprocessError, OSError):
        return None
    if r.returncode != 0:
        return None
    for root in (Path.home() / ".npm" / "_npx").glob("*/node_modules/.bin/gltf-transform"):
        return [str(root)]
    return ["npx", "-y", "@gltf-transform/cli"]


def draco_compress(assets: Path, files: list[Path], position_bits: int = 14) -> bool:
    """Compress glbs in place with Draco. Integer attributes (`_BUILDING_ID` uint16, `_MAT`,
    `_VARIANT`, `_FRONT` uint8) are stored losslessly by Draco; float attributes quantize."""
    cmd = _gltf_transform_cmd()
    if cmd is None:
        log("draco: gltf-transform unavailable (no Node/npm); writing uncompressed glb")
        return False

    def one(p: Path) -> bool:
        tmp = p.with_suffix(".draco.glb")
        r = subprocess.run([*cmd, "draco", str(p), str(tmp), "--quantize-position", str(position_bits)], capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not tmp.exists():
            log(f"draco: failed on {p.name}: {r.stderr.strip()[:300]}")
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(p)
        return True

    if not files:
        return True
    with ThreadPoolExecutor(max_workers=8) as ex:
        ok = list(ex.map(one, files))
    log(f"draco: compressed {sum(ok)}/{len(files)} files")
    return all(ok)


# ---------------------------------------------------------------------------
# Shared tail: buildings -> schools -> population -> meta
# ---------------------------------------------------------------------------


def synthetic_work_model(bdf: pd.DataFrame, net: RoadNetwork, source: str = "assumptions (synthetic)") -> WorkModel:
    """Internal jobs at commercial/school buildings (area weighted), the rest via exits
    (assumptions demand.external_job_share, population_synthesis.external_exit_shares).
    Used by the synthetic world and by the footprint population estimate."""
    nodes, edges = net.nodes, net.edges
    loc = nodes[local_node_mask(nodes, edges)]
    tree = cKDTree(loc[["x", "z"]].to_numpy())
    jobs = bdf[bdf["type"].isin(["commercial", "school"])]
    w = jobs["area_m2"].to_numpy() * np.where(jobs["type"] == "school", float(assumption("population_synthesis.internal_job_weight_school")), 1.0)
    _, k = tree.query(jobs[["centroid_x", "centroid_z"]].to_numpy(), k=1)
    agg = pd.Series(w, index=loc["node_id"].to_numpy()[k]).groupby(level=0).sum()
    shares = exit_probs_from_bearings(net.exits)
    log("external jobs by exit (job direction table): " + ", ".join(f"{k} {v:.0%}" for k, v in shares.items()))
    return WorkModel(
        internal_share={"": 1.0 - float(assumption("demand.external_job_share"))},
        internal_nodes={"": agg.index.to_numpy(dtype=np.int64)},
        internal_weights={"": (agg / agg.sum()).to_numpy()},
        exit_probs={"": shares},
        source=source,
    )


def finish(
    *,
    synthetic: bool,
    terrain: Terrain,
    grid: TileGrid,
    net: RoadNetwork,
    tb: TerrainBuild,
    footprints: Any,
    processed: Path,
    assets: Path,
    dists: list[Any] | None,
    work: WorkModel | None,
    school_areas: Any = None,
    block_groups: Any = None,
    osm_schools: Any = None,
    sources: list[dict[str, Any]],
    skip_draco: bool = False,
    population: str = "synthetic",
    extra_meta: dict[str, Any] | None = None,
    raw: Path | None = None,
    texture_px: int = 512,
) -> dict[str, Any]:
    """Shared tail. `population` is 'synthetic', 'acs' (dists + work given) or 'footprints'
    (households estimated from real residential footprints inside the region bbox).
    `raw` (real mode) enables the Overture-based render layers (service roads, pools, bridges,
    land use) and the lidar building attributes."""
    from pipeline.build_buildings import building_shapes, lidar_roof_join, materials_manifest
    from pipeline.build_landcover import (
        fetch_land_cover,
        load_land_cover_scene,
        load_landuse_scene,
        load_water_scene,
        write_splat_masks,
    )
    from pipeline.build_streets import (
        Draper,
        Materials,
        bridge_spans,
        layout_streets,
        overture_missing_roads,
        overture_paths,
        overture_pools,
        overture_service_roads,
        visual_roads,
        write_street_tiles,
    )

    reg = region()
    schools_cfg = load_yaml("schools.yaml")["schools"]
    render = tb.render
    timings: dict[str, float] = {}
    t0 = time.time()
    manifest_mat = materials_manifest(assets)

    bdf = prepare_buildings(footprints, render, grid, school_areas, block_groups)
    if raw is not None:
        bdf = lidar_roof_join(bdf, raw, footprints)
    hero_list = load_heroes()
    bdf = apply_heroes(bdf, hero_list, render, grid)
    shapes = building_shapes(bdf, render, hero_list, manifest_mat)
    hmap = {k: v.height_m for k, v in shapes.items()}
    keep_rule = bdf["height_rule"].isin(["hero", "lidar"]) if "height_rule" in bdf.columns else np.zeros(len(bdf), bool)
    new_h = bdf["id"].map(hmap)
    bdf["height_m"] = np.where(new_h.notna() & ~keep_rule, new_h, bdf["height_m"])
    timings["buildings_prepare"] = time.time() - t0

    # streets / ground layers
    t1 = time.time()
    roads = visual_roads(net.G_visual if net.G_visual is not None else net.G, grid.extent)
    pools: list[Any] = []
    spans: list[Any] = []
    paths: list[Any] = []
    if raw is not None:
        extra = overture_missing_roads(raw, grid.extent, roads)
        if extra:
            log(f"streets: +{len(extra):,} real streets missing from the drive graph (private / gated) rendered from Overture")
        roads += extra
        roads += overture_service_roads(raw, grid.extent)
        paths = overture_paths(raw, grid.extent)
        pools = overture_pools(raw, grid.extent)
        spans = bridge_spans(raw, render, grid.extent)
    sig = net.nodes.loc[net.nodes["signalized"], ["x", "z"]].to_numpy()
    from shapely.geometry import Point

    hero_disks = [Point(h.x, h.z).buffer(h.radius, 32) for h in hero_list]
    layout = layout_streets(roads, grid.extent, sig, shapes, pools, hero_disks, paths)
    draper = Draper(render, spans)
    if spans:
        log(f"streets: {len(spans)} bridge decks (Overture is_bridge) interpolated over the bare-earth DEM")
    # network edge geometry follows the rendered road surface (incl. bridge decks)
    geoms = []
    for g in net.edges["geometry"]:
        xyz = np.asarray(g, dtype=np.float64).reshape(-1, 3)
        xyz[:, 1] = draper.line(xyz[:, [0, 2]])
        geoms.append(xyz.astype(np.float32).reshape(-1))
    net.edges["geometry"] = geoms
    net.nodes["y"] = render.sample(net.nodes["x"].to_numpy(), net.nodes["z"].to_numpy()).astype(np.float32)
    write_network(net, processed)
    mats = Materials(manifest_mat)
    street_tiles, street_tris = write_street_tiles(layout, draper, grid, assets, mats)
    timings["streets"] = time.time() - t1

    t1 = time.time()
    btiles, b_tris, heroes = build_building_tiles(bdf, grid, assets / "buildings", hero_list, render, shapes, layout.frontage, manifest_mat)
    write_buildings_geojson(bdf, processed / "buildings.geojson")
    timings["buildings_tiles"] = time.time() - t1

    # land-cover splat masks
    t1 = time.time()
    from shapely.ops import unary_union

    paved = unary_union([layout.roads_poly, *layout.sidewalks, *[d for d, _ in layout.driveways], *[sh.walls for sh in shapes.values()], *[p for p, k in layout.paths if k == "paved"]])
    water = (load_water_scene(raw) if raw is not None else []) + list(pools)
    landuse = load_landuse_scene(raw) if raw is not None else []
    landcover: list[Any] = []
    lidar_rasters: dict[str, Path] = {}
    if raw is not None:
        fetch_land_cover(raw)
        landcover = load_land_cover_scene(raw)
        if landcover:
            import json as _json

            import pyarrow.parquet as _pq

            rel = _json.loads((_pq.read_schema(raw / "overture" / "land_cover.parquet").metadata or {}).get(b"redraw_key", b"{}").decode() or "{}").get("release", "")
            url = f"https://overturemaps-us-west-2.s3.amazonaws.com/release/{rel}/theme=base/type=land_cover/"
            sources = [*sources, dict(LAND_COVER_SOURCE, url=url, release=rel, retrieved=today())]
        lidar_rasters = {k: raw / "lidar" / f for k, f in (("chm", "chm_0p5m.tif"), ("ndsm", "ndsm_0p5m.tif")) if (raw / "lidar" / f).exists()}
    splat_px = SPLAT_PX_LIDAR if lidar_rasters else min(1024, max(256, texture_px))
    splats = write_splat_masks(grid, tb.albedo, assets / "terrain", splat_px, paved, water, landuse, lawn_extra=layout.medians, landcover=landcover, lidar=lidar_rasters, dirt_extra=[p for p, k in layout.paths if k == "trail"])
    splat_inputs = ["imagery", "vector overrides (roads, sidewalks, driveways, roofs, water, pools)"] + (["overture land_use"] if landuse else []) + (["overture land_cover"] if landcover else []) + ([f"lidar {k}" for k in lidar_rasters])
    timings["splat"] = time.time() - t1

    schools = resolve_schools(schools_cfg, net.nodes, net.edges, bdf, osm_schools)
    pop_bdf = bdf
    if population != "synthetic":
        # people live and (internal) jobs are inside the region bbox; the terrain buffer is scenery
        rext = region_extent()
        pop_bdf = bdf[np.asarray(rext.contains(bdf["centroid_x"].to_numpy(), bdf["centroid_z"].to_numpy()))]
    if population == "footprints":
        units = footprint_units(pop_bdf)
        pop_bdf = pop_bdf.assign(units=units)
        n_hh = int(units.sum())
        by_type = pop_bdf.assign(units=units).groupby("type")["units"].sum().to_dict()
        log("POPULATION SOURCE: FOOTPRINT ESTIMATE (explicit fallback; Census ACS/LODES not used)")
        log(f"  {n_hh:,} households from {int((units > 0).sum()):,} residential footprints inside the region bbox: {by_type}")
        dists = [fallback_dist("", n_hh)]
        work = synthetic_work_model(pop_bdf, net, source="assumptions (footprint estimate: commercial/school buildings by area + exit job-direction table)")
    if dists is None:
        dists = [fallback_dist("", int(assumption("population.target_households_fallback")))]
    if work is None:
        work = synthetic_work_model(pop_bdf, net)
    hh, persons = synthesize_population(pop_bdf, dists, work, net.nodes, net.edges, net.exits, schools, int(assumption("population.seed")))
    write_population(hh, persons, processed)
    sbs = students_by_school(persons, schools)
    for s in schools:
        s["students"] = sbs[s["id"]]
    write_json(processed / "schools_resolved.json", {"schools": schools})

    draco = False
    t1 = time.time()
    if not skip_draco:
        terrain_files = sorted((assets / "terrain").glob("terrain_*.glb"))
        detail_files = sorted((assets / "roads").glob("roads_*.glb")) + sorted((assets / "ground").glob("ground_*.glb")) + sorted((assets / "buildings").glob("buildings_*.glb"))
        draco = draco_compress(assets, terrain_files, position_bits=14) and draco_compress(assets, detail_files, position_bits=16)
    timings["draco"] = time.time() - t1

    tiles = []
    for t in tb.infos:
        b = dict(t["bounds"])
        bt = btiles.get(t["id"], {})
        st = street_tiles.get(t["id"], {})
        for lay in (bt, st):
            if lay.get("max_y") is not None:
                b["min_y"] = min(b["min_y"], lay["min_y"])
                b["max_y"] = max(b["max_y"], lay["max_y"])
        tiles.append(
            {
                "id": t["id"],
                "row": t["row"],
                "col": t["col"],
                "bounds": b,
                "terrain": t["terrain"],
                "terrain_lods": t["terrain_lods"],
                "albedo": t["albedo"],
                "splat": splats.get(t["id"], []),
                "buildings": bt.get("path", f"buildings/buildings_{t['id']}.glb"),
                "roads": st.get("roads"),
                "ground": st.get("ground"),
                "triangles": {"buildings": bt.get("triangles", 0), **st.get("triangles", {})},
            }
        )
    sizes = asset_sizes(assets)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "hd_version": 1,
        "synthetic": synthetic,
        "draco": draco,
        "draco_layers": {"terrain": draco, "roads": draco, "ground": draco, "buildings": draco},
        "grid": {"rows": grid.rows, "cols": grid.cols, **grid.extent.as_dict(), "tile_width_m": grid.extent.width / grid.cols, "tile_depth_m": grid.extent.depth / grid.rows},
        "tiles": tiles,
        "roads": [t["roads"] for t in tiles if t.get("roads")],
        "terrain_meta": "terrain/terrain_meta.json",
        "terrain_lod": {
            "levels": [
                {
                    "lod": lod,
                    "kind": "rtin",
                    "finest_spacing_m": tb.infos[0]["terrain_lods"][lod]["spacing_m"] if tb.infos else None,
                    "max_error_m": round(tb.lod_max_error_m.get(lod, tb.lod0_max_error_m), 4),
                    "triangles": tb.triangles.get(f"lod{lod}", 0),
                    "texture_px": tb.infos[0]["terrain_lods"][lod]["texture_px"] if tb.infos else None,
                }
                for lod in (0, 1, 2)
            ],
            "skirts_m": {"0": 3.0, "1": 6.0, "2": 12.0},
            "suggested_switch_distance_m": {"0": 900.0, "1": 2600.0},
        },
        "splat": {
            "channels": {"a": ["lawn", "chaparral", "dirt"], "b": ["paved", "water", "canopy"]},
            "px": splat_px,
            "encoding": "RGB PNG, 8-bit weights summing to 255 over the 6 channels; north up like the albedo",
            "inputs": splat_inputs,
            "suggested_ground_cells": {"lawn": "grass_lawn", "chaparral": "chaparral", "dirt": "bare_dirt", "paved": "asphalt_worn", "water": "pool_water", "canopy": "mulch"},
        },
        "materials": "materials/materials_manifest.json" if manifest_mat is not None else None,
        "heroes": heroes,
        "triangles": {"terrain": tb.triangles.get("lod0", 0), "terrain_lods": tb.triangles, "buildings": b_tris, "roads": street_tris["roads"], "ground": street_tris["ground"]},
        "street_stats": layout.stats,
        "sizes_mb": sizes,
        "generated_at": now_iso(),
    }
    hd = assets / "buildings_hd" / "manifest_buildings.json"
    if hd.exists():
        manifest["buildings_hd"] = "buildings_hd/manifest_buildings.json"
    write_json(assets / "manifest.json", manifest)
    log("timing: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    log(f"assets: {sizes}")

    o = scene_origin()
    rext = region_extent()
    counts = {
        "buildings": int(len(bdf)),
        "buildings_by_type": {k: int(v) for k, v in bdf["type"].value_counts().items()},
        "road_edges": int(len(net.edges)),
        "road_nodes": int(len(net.nodes)),
        "signals": int(net.nodes["signalized"].sum()),
        "households": int(len(hh)),
        "persons": int(len(persons)),
        "workers": int(persons["is_worker"].sum()),
        "students": int((persons["school_id"] != "").sum()),
        "students_by_school": sbs,
        "exits": len(net.exits),
        "exits_dropped": [e["id"] for e in net.exits_dropped],
        **{f"streets_{k}": v for k, v in layout.stats.items()},
    }
    meta = {
        "contract_version": CONTRACT_VERSION,
        "name": reg["name"],
        "display_name": reg["display_name"],
        "synthetic": synthetic,
        "generated_at": now_iso(),
        "projection": PROJECTION,
        "bbox": reg["bbox"],
        "origin": {"lat": o.lat, "lon": o.lon, "easting": o.easting, "northing": o.northing},
        "extent_scene": rext.as_dict(),
        "terrain_extent_scene": grid.extent.as_dict(),
        "tiles": {"rows": grid.rows, "cols": grid.cols},
        "counts": counts,
        "population_source": POPULATION_SOURCE_LABEL[population],
        "signal_source": "synthetic" if synthetic else net.signal_source,
        "sources": sources,
        "attribution": attribution_lines(sources),
        **(extra_meta or {}),
    }
    if synthetic:
        meta["warning"] = "SYNTHETIC DEV DATA: procedurally generated stand-in world. NOT real geography, buildings, roads or people."
    write_json(processed / "region_meta.json", meta)
    return meta


def asset_sizes(assets: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for sub in ("terrain", "buildings", "roads", "ground", "props", "materials", "buildings_hd"):
        d = assets / sub
        if d.exists():
            out[sub] = round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6, 2)
    out["total"] = round(sum(out.values()), 2)
    return out


def print_counts(meta: dict[str, Any]) -> None:
    c = meta["counts"]
    tag = "SYNTHETIC (fake) " if meta["synthetic"] else ""
    print("\n" + "=" * 60)
    print(f"Redraw {tag}world build complete: {meta['display_name']}")
    print("=" * 60)
    print(f"  buildings      : {c['buildings']:,}  {c['buildings_by_type']}")
    print(f"  road nodes     : {c['road_nodes']:,}")
    print(f"  road edges     : {c['road_edges']:,}")
    print(f"  signals        : {c.get('signals', 0):,}  ({meta.get('signal_source', '')})")
    print(f"  households     : {c['households']:,}  (population source: {meta.get('population_source', '')})")
    print(f"  persons        : {c['persons']:,}  (workers {c['workers']:,}, students {c['students']:,})")
    print("  students per school:")
    for k, v in c["students_by_school"].items():
        print(f"    {k:<36} {v:,}")
    if meta.get("population_source") == "footprint_estimate":
        lo, hi = assumption_range("population.target_households_fallback")
        tgt = int(assumption("population.target_households_fallback"))
        ok = lo <= c["households"] <= hi
        print(f"  sanity check   : {c['households']:,} households vs target_households_fallback {tgt:,} (range {lo:,}-{hi:,}): {'OK' if ok else 'OUTSIDE RANGE'}")
        if not ok:
            print("  WARNING: the footprint estimate is outside the assumed range. The count is NOT scaled to fit;")
            print("           check the region bbox (region.yaml) and footprint_population.* assumptions.")
        print("  NOTE: households are a FOOTPRINT ESTIMATE (no Census data); see region_meta.json sources.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_synthetic(skip_draco: bool = False) -> dict[str, Any]:
    from pipeline.synthetic import make_world

    t0 = time.time()
    dirs = ensure_dirs()
    clean_outputs(dirs["processed"], dirs["assets"])
    grid = tile_grid()
    log("SYNTHETIC MODE: generating a fake stand-in world (not real geography)")
    world = make_world(region_extent(), grid.extent, load_yaml("schools.yaml")["schools"])
    log(f"synthetic world generated in {time.time() - t0:.1f}s")
    flatten_terrain_for_heroes(world.terrain, load_heroes())
    tb, _ = write_terrain_outputs(world.terrain, grid, world.albedo, dirs["processed"], dirs["assets"], texture_px=SYNTHETIC_TEXTURE_PX, budget=SYNTHETIC_TERRAIN_BUDGET)
    net = build_network(world.graph, tb.render, region())
    sources = [
        {"name": "Synthetic procedural world (pipeline/synthetic.py), fixed seeds", "license": "CC BY 4.0 (Redraw project)", "retrieved": today()},
        {"name": "Region config, schools.yaml, assumptions.yaml (unverified placeholders)", "license": "CC BY 4.0 (Redraw project)", "retrieved": today()},
    ]
    meta = finish(
        synthetic=True,
        terrain=world.terrain,
        grid=grid,
        net=net,
        tb=tb,
        footprints=world.footprints,
        processed=dirs["processed"],
        assets=dirs["assets"],
        dists=None,
        work=None,
        sources=sources,
        skip_draco=skip_draco,
        texture_px=SYNTHETIC_TEXTURE_PX,
    )
    log(f"synthetic build finished in {time.time() - t0:.1f}s")
    return meta


def run_real(skip_draco: bool = False, population_source: str = "acs") -> dict[str, Any]:
    from pipeline import (
        build_buildings,
        build_population,
        build_roads,
        build_terrain,
        fetch_census,
        fetch_dem,
        fetch_imagery,
        fetch_osm,
    )
    from pipeline.sources import not_available, population_sources, raw_sources

    if population_source not in POPULATION_SOURCES:
        raise ValueError(f"population_source must be one of {POPULATION_SOURCES}, got {population_source!r}")
    t0 = time.time()
    dirs = ensure_dirs()
    raw = dirs["raw"]
    grid = tile_grid()
    # 1. fetch (each caches to data/raw and raises DataSourceUnavailable with manual steps)
    osm = fetch_osm.fetch_all(raw)
    dem_path = fetch_dem.fetch(raw)
    naip_path = fetch_imagery.fetch(raw)
    census = None
    if population_source == "acs":
        try:
            census = fetch_census.fetch_all(raw)
        except DataSourceUnavailable as e:
            e.args = (
                str(e)
                + "  Alternative: `python pipeline/build_all.py --population-source footprints` estimates households\n"
                "  from the real residential building footprints instead (explicit, labeled fallback; not census data).\n",
            )
            raise
    else:
        log("=" * 72)
        log("POPULATION SOURCE: footprints (explicit fallback). Census ACS/LODES/TIGER are NOT used;")
        log("households will be ESTIMATED from real residential building footprints and assumptions.yaml.")
        log("region_meta.json records population_source = footprint_estimate.")
        log("=" * 72)
    clean_outputs(dirs["processed"], dirs["assets"])
    # 2. build
    t = time.time()
    terrain, tb, terrain_srcs, _alb, tex_px = build_terrain.run_real(dirs["processed"], dirs["assets"], grid, dem_path, naip_path)
    log(f"timing: terrain {time.time() - t:.1f}s")
    t = time.time()
    G, sig = build_roads.load_osm_drive(osm["drive"])
    net = build_network(G, tb.render, region(), sig)
    log(f"timing: roads {time.time() - t:.1f}s")
    footprints = build_buildings.load_osm_buildings(osm["buildings"], build_buildings.load_landuse(raw))
    footprints = build_buildings.add_lidar_missing_buildings(footprints, raw)
    osm_schools, school_areas = fetch_osm.load_schools(osm["schools"], load_yaml("schools.yaml")["schools"])
    bgs = dists = work = None
    if census is not None:
        bgs, frac = fetch_census.load_block_groups(census["tiger_bg"])
        acs = fetch_census.load_acs(census["acs"])
        dists = build_population.dists_from_acs(acs, frac)
        od, xwalk = fetch_census.load_lodes(census["od_main"], census["xwalk"])
        work = build_population.work_model_from_lodes(od, xwalk, set(frac), net.nodes, net.edges, net.exits, region()["bbox"])
    map_srcs = [s for s in raw_sources(raw, dem_path, naip_path) if s.get("kind") not in ("dem", "imagery")]
    sources = [dict(s, retrieved=s.get("retrieved", today())) for s in terrain_srcs + map_srcs] + population_sources(population_source)
    extra: dict[str, Any] = {
        "population_note": (
            "Households ESTIMATED from real residential building footprints (explicit --population-source footprints "
            "fallback; Census ACS/LODES unavailable). Not census data."
            if population_source == "footprints"
            else "Households synthesized to match Census ACS 5-year block group totals; commutes from LEHD LODES."
        ),
        "signal_note": (
            "Traffic signals inferred from road classes (assumptions signal_inference.*); the map source has no signal data."
            if net.signal_source == "inferred"
            else "Traffic signals from OSM highway=traffic_signals."
        ),
    }
    na = not_available(raw)
    if na:
        extra["sources_not_available"] = na
    meta = finish(
        synthetic=False,
        terrain=terrain,
        grid=grid,
        net=net,
        tb=tb,
        footprints=footprints,
        processed=dirs["processed"],
        assets=dirs["assets"],
        dists=dists,
        work=work,
        school_areas=school_areas,
        block_groups=bgs,
        osm_schools=osm_schools,
        sources=sources,
        skip_draco=skip_draco,
        population=population_source,
        extra_meta=extra,
        raw=raw,
        texture_px=tex_px,
    )
    log(f"real build finished in {time.time() - t0:.1f}s")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="build the offline FAKE stand-in world (dev/CI only)")
    ap.add_argument("--no-draco", action="store_true", help="skip Draco compression (manifest.draco=false)")
    ap.add_argument(
        "--population-source",
        choices=POPULATION_SOURCES,
        default="acs",
        help="real mode: 'acs' (Census ACS + LODES, default) or 'footprints' (explicit fallback: households "
        "estimated from real residential footprints; recorded as footprint_estimate in region_meta.json)",
    )
    args = ap.parse_args(argv)
    if args.synthetic and args.population_source != "acs":
        ap.error("--population-source applies to the real build only (the synthetic world has its own population)")
    try:
        meta = run_synthetic(args.no_draco) if args.synthetic else run_real(args.no_draco, args.population_source)
    except DataSourceUnavailable as e:
        print(str(e), file=sys.stderr)
        return 2
    print_counts(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
