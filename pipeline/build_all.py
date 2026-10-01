"""Run the whole world-data pipeline (spec 5, milestone M1).

    python pipeline/build_all.py              # real data: OSM, USGS 3DEP, NAIP, ACS, LODES
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
    fallback_dist,
    resolve_schools,
    students_by_school,
    synthesize_population,
    write_population,
)
from pipeline.build_roads import (
    RoadNetwork,
    build_network,
    build_road_ribbons,
    local_node_mask,
    write_network,
)
from pipeline.build_terrain import Terrain, write_terrain_outputs
from pipeline.common import (
    CONTRACT_VERSION,
    TileGrid,
    ensure_dirs,
    log,
    now_iso,
    region_extent,
    tile_grid,
    today,
    write_json,
)
from pipeline.config import assumption, load_yaml, region
from pipeline.geo import PROJECTION, scene_origin

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

REAL_SOURCES = [
    {"name": "OpenStreetMap (roads, buildings, schools) via Overpass/OSMnx", "license": "ODbL 1.0", "url": "https://www.openstreetmap.org/copyright"},
    {"name": "USGS 3D Elevation Program (3DEP) 1/3 arc-second DEM via py3dep", "license": "Public domain (USGS)", "url": "https://www.usgs.gov/3d-elevation-program"},
    {"name": "USDA NAIP imagery via Microsoft Planetary Computer", "license": "Public domain (USDA FSA)", "url": "https://planetarycomputer.microsoft.com/dataset/naip"},
    {"name": "US Census Bureau ACS 5-year estimates (block groups) and TIGER/Line", "license": "Public domain (US Census Bureau)", "url": "https://www.census.gov/data/developers.html"},
    {"name": "US Census Bureau LEHD LODES 8 (OD, WAC, RAC)", "license": "Public domain (US Census Bureau)", "url": "https://lehd.ces.census.gov/data/"},
]


def clean_outputs(processed: Path, assets: Path) -> None:
    for name in PROCESSED_OUTPUTS:
        (processed / name).unlink(missing_ok=True)
    for sub in ("terrain", "buildings", "roads"):
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


def draco_compress(assets: Path, files: list[Path]) -> bool:
    """Compress terrain and road glbs in place with Draco. Building tiles stay uncompressed so
    `_BUILDING_ID` stays exact (Draco would quantize the generic attribute)."""
    cmd = _gltf_transform_cmd()
    if cmd is None:
        log("draco: gltf-transform unavailable (no Node/npm); writing uncompressed glb")
        return False

    def one(p: Path) -> bool:
        tmp = p.with_suffix(".draco.glb")
        r = subprocess.run([*cmd, "draco", str(p), str(tmp)], capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not tmp.exists():
            log(f"draco: failed on {p.name}: {r.stderr.strip()[:300]}")
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(p)
        return True

    with ThreadPoolExecutor(max_workers=8) as ex:
        ok = list(ex.map(one, files))
    log(f"draco: compressed {sum(ok)}/{len(files)} files")
    return all(ok)


# ---------------------------------------------------------------------------
# Shared tail: buildings -> schools -> population -> meta
# ---------------------------------------------------------------------------


def synthetic_work_model(bdf: pd.DataFrame, net: RoadNetwork) -> WorkModel:
    """Synthetic: internal jobs at commercial/school buildings (area weighted), rest via exits."""
    nodes, edges = net.nodes, net.edges
    loc = nodes[local_node_mask(nodes, edges)]
    tree = cKDTree(loc[["x", "z"]].to_numpy())
    jobs = bdf[bdf["type"].isin(["commercial", "school"])]
    w = jobs["area_m2"].to_numpy() * np.where(jobs["type"] == "school", float(assumption("population_synthesis.internal_job_weight_school")), 1.0)
    _, k = tree.query(jobs[["centroid_x", "centroid_z"]].to_numpy(), k=1)
    agg = pd.Series(w, index=loc["node_id"].to_numpy()[k]).groupby(level=0).sum()
    shares = dict(assumption("population_synthesis.external_exit_shares"))
    exit_ids = {e["id"] for e in net.exits}
    shares = {k2: v for k2, v in shares.items() if k2 in exit_ids}
    return WorkModel(
        internal_share={"": 1.0 - float(assumption("demand.external_job_share"))},
        internal_nodes={"": agg.index.to_numpy(dtype=np.int64)},
        internal_weights={"": (agg / agg.sum()).to_numpy()},
        exit_probs={"": shares},
        source="assumptions (synthetic)",
    )


def finish(
    *,
    synthetic: bool,
    terrain: Terrain,
    grid: TileGrid,
    net: RoadNetwork,
    terrain_tiles: list[dict[str, Any]],
    terrain_tris: int,
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
) -> dict[str, Any]:
    reg = region()
    schools_cfg = load_yaml("schools.yaml")["schools"]
    write_network(net, processed)
    road_tris = build_road_ribbons(net.edges, terrain, assets / "roads" / "roads.glb")

    bdf = prepare_buildings(footprints, terrain, grid, school_areas, block_groups)
    hero_list = load_heroes()
    bdf = apply_heroes(bdf, hero_list, terrain, grid)
    write_buildings_geojson(bdf, processed / "buildings.geojson")
    btiles, b_tris, heroes = build_building_tiles(bdf, grid, assets / "buildings", hero_list)

    schools = resolve_schools(schools_cfg, net.nodes, net.edges, bdf, osm_schools)
    if dists is None:
        dists = [fallback_dist("", int(assumption("population.target_households_fallback")))]
    if work is None:
        work = synthetic_work_model(bdf, net)
    hh, persons = synthesize_population(bdf, dists, work, net.nodes, net.edges, net.exits, schools, int(assumption("population.seed")))
    write_population(hh, persons, processed)
    sbs = students_by_school(persons, schools)
    for s in schools:
        s["students"] = sbs[s["id"]]
    write_json(processed / "schools_resolved.json", {"schools": schools})

    draco = False
    if not skip_draco:
        files = sorted((assets / "terrain").glob("terrain_*.glb")) + [assets / "roads" / "roads.glb"]
        draco = draco_compress(assets, files)

    tiles = []
    for t in terrain_tiles:
        b = dict(t["bounds"])
        bt = btiles.get(t["id"], {})
        if bt.get("max_y") is not None:
            b["min_y"] = min(b["min_y"], bt["min_y"])
            b["max_y"] = max(b["max_y"], bt["max_y"])
        tiles.append({"id": t["id"], "row": t["row"], "col": t["col"], "bounds": b, "terrain": t["terrain"], "buildings": bt.get("path", f"buildings/buildings_{t['id']}.glb")})
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "synthetic": synthetic,
        "draco": draco,
        "draco_layers": {"terrain": draco, "roads": draco, "buildings": False},
        "tiles": tiles,
        "roads": ["roads/roads.glb"],
        "terrain_meta": "terrain/terrain_meta.json",
        "heroes": heroes,
        "triangles": {"terrain": terrain_tris, "buildings": b_tris, "roads": road_tris},
        "generated_at": now_iso(),
    }
    write_json(assets / "manifest.json", manifest)

    o = scene_origin()
    rext = region_extent()
    counts = {
        "buildings": int(len(bdf)),
        "buildings_by_type": {k: int(v) for k, v in bdf["type"].value_counts().items()},
        "road_edges": int(len(net.edges)),
        "road_nodes": int(len(net.nodes)),
        "households": int(len(hh)),
        "persons": int(len(persons)),
        "workers": int(persons["is_worker"].sum()),
        "students": int((persons["school_id"] != "").sum()),
        "students_by_school": sbs,
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
        "sources": sources,
    }
    if synthetic:
        meta["warning"] = "SYNTHETIC DEV DATA: procedurally generated stand-in world. NOT real geography, buildings, roads or people."
    write_json(processed / "region_meta.json", meta)
    return meta


def print_counts(meta: dict[str, Any]) -> None:
    c = meta["counts"]
    tag = "SYNTHETIC (fake) " if meta["synthetic"] else ""
    print("\n" + "=" * 60)
    print(f"Redraw {tag}world build complete: {meta['display_name']}")
    print("=" * 60)
    print(f"  buildings      : {c['buildings']:,}  {c['buildings_by_type']}")
    print(f"  road nodes     : {c['road_nodes']:,}")
    print(f"  road edges     : {c['road_edges']:,}")
    print(f"  households     : {c['households']:,}")
    print(f"  persons        : {c['persons']:,}  (workers {c['workers']:,}, students {c['students']:,})")
    print("  students per school:")
    for k, v in c["students_by_school"].items():
        print(f"    {k:<22} {v:,}")
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
    tiles, tris, _ = write_terrain_outputs(world.terrain, grid, world.albedo, dirs["processed"], dirs["assets"], texture_px=1024)
    net = build_network(world.graph, world.terrain, region())
    sources = [
        {"name": "Synthetic procedural world (pipeline/synthetic.py), fixed seeds", "license": "CC BY 4.0 (Redraw project)", "retrieved": today()},
        {"name": "Region config, schools.yaml, assumptions.yaml (unverified placeholders)", "license": "CC BY 4.0 (Redraw project)", "retrieved": today()},
    ]
    meta = finish(
        synthetic=True,
        terrain=world.terrain,
        grid=grid,
        net=net,
        terrain_tiles=tiles,
        terrain_tris=tris,
        footprints=world.footprints,
        processed=dirs["processed"],
        assets=dirs["assets"],
        dists=None,
        work=None,
        sources=sources,
        skip_draco=skip_draco,
    )
    log(f"synthetic build finished in {time.time() - t0:.1f}s")
    return meta


def run_real(skip_draco: bool = False) -> dict[str, Any]:
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

    t0 = time.time()
    dirs = ensure_dirs()
    raw = dirs["raw"]
    grid = tile_grid()
    # 1. fetch (each caches to data/raw and raises DataSourceUnavailable with manual steps)
    osm = fetch_osm.fetch_all(raw)
    dem_path = fetch_dem.fetch(raw)
    naip_path = fetch_imagery.fetch(raw)
    census = fetch_census.fetch_all(raw)
    clean_outputs(dirs["processed"], dirs["assets"])
    # 2. build
    terrain, tiles, tris = build_terrain.run_real(dirs["processed"], dirs["assets"], grid, dem_path, naip_path)
    G, sig = build_roads.load_osm_drive(osm["drive"])
    net = build_network(G, terrain, region(), sig)
    footprints = build_buildings.load_osm_buildings(osm["buildings"])
    osm_schools, school_areas = fetch_osm.load_schools(osm["schools"], load_yaml("schools.yaml")["schools"])
    bgs, frac = fetch_census.load_block_groups(census["tiger_bg"])
    acs = fetch_census.load_acs(census["acs"])
    dists = build_population.dists_from_acs(acs, frac)
    od, xwalk = fetch_census.load_lodes(census["od_main"], census["xwalk"])
    work = build_population.work_model_from_lodes(od, xwalk, set(frac), net.nodes, net.edges, net.exits, region()["bbox"])
    sources = [dict(s, retrieved=today()) for s in REAL_SOURCES]
    meta = finish(
        synthetic=False,
        terrain=terrain,
        grid=grid,
        net=net,
        terrain_tiles=tiles,
        terrain_tris=tris,
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
    )
    log(f"real build finished in {time.time() - t0:.1f}s")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="build the offline FAKE stand-in world (dev/CI only)")
    ap.add_argument("--no-draco", action="store_true", help="skip Draco compression (manifest.draco=false)")
    args = ap.parse_args(argv)
    from pipeline.common import DataSourceUnavailable

    try:
        meta = run_synthetic(args.no_draco) if args.synthetic else run_real(args.no_draco)
    except DataSourceUnavailable as e:
        print(str(e), file=sys.stderr)
        return 2
    print_counts(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
