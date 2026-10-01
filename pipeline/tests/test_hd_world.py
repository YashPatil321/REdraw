"""HD world build: exits routed via a neighbour, lidar roof join, private streets, splat masks,
terrain LOD tiles and the per-tile manifest (synthetic build into a temp dir)."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from PIL import Image
from shapely.geometry import LineString, box

from pipeline.build_buildings import (
    add_lidar_missing_buildings,
    lidar_roof_join,
    write_buildings_geojson,
)
from pipeline.build_roads import route_unreachable_exits
from pipeline.common import Extent, TileGrid
from pipeline.geo import scene_origin, utm_to_lonlat
from pipeline.glb import read_glb


def _scene_to_lonlat(x: float, z: float) -> tuple[float, float]:
    o = scene_origin()
    return utm_to_lonlat(o.easting + x, o.northing - z)


# ---------------------------------------------------------------- exits


def test_unreachable_exit_is_routed_via_the_nearest_exit() -> None:
    exits = [
        {"id": "a", "label": "A", "node_id": 1, "x": 0.0, "z": 0.0, "bearing_deg": 90.0},
        {"id": "b", "label": "B", "node_id": 2, "x": 1000.0, "z": 0.0, "bearing_deg": 0.0},
    ]
    dropped = [{"id": "c", "label": "C", "bearing_deg": 330.0, "verified": False, "reason": "road 'c' is not connected"}]
    extra = route_unreachable_exits(exits, dropped, {"c": (900.0, -50.0)})
    assert dropped == []
    assert len(extra) == 1
    e = extra[0]
    assert e["id"] == "c" and e["via"] == "b" and e["node_id"] == 2 and e["bearing_deg"] == 330.0
    assert e["snap_method"] == "via_exit"


# ---------------------------------------------------------------- lidar buildings


def test_lidar_roof_join_by_source_id_and_missing_buildings(tmp_path: Path) -> None:
    o = scene_origin()
    raw = tmp_path / "raw"
    (raw / "lidar").mkdir(parents=True)
    polys = [box(o.easting + i * 40, o.northing, o.easting + i * 40 + 12, o.northing + 10) for i in range(3)]
    fp = gpd.GeoDataFrame({"building": ["house"] * 3, "source": ["OpenStreetMap"] * 3, "source_id": ["g0", "g1", "g2"], "osm_sort_key": ["a", "b", "c"]}, geometry=polys, crs="EPSG:32611")
    pd.DataFrame(
        {
            "building_id": ["g0", "g1", "g2"],
            "eave_height_m": [3.1, 2.9, np.nan],
            "ridge_height_m": [6.4, 0.5, np.nan],
            "roof_type": ["gable", "hip", "unknown"],
            "roof_pitch_deg": [24.0, 20.0, np.nan],
            "ridge_azimuth_deg": [90.0, 0.0, np.nan],
            "quality": ["good", "good", "none"],
            "lidar_status": ["present", "present", "absent"],
        }
    ).to_parquet(raw / "lidar" / "buildings_roofs.parquet")
    miss = gpd.GeoDataFrame(
        {"lidar_id": ["lidar_00000"], "eave_height_m": [2.8], "ridge_height_m": [5.5], "roof_type": ["hip"], "quality": ["fair"]},
        geometry=[box(o.easting + 200, o.northing, o.easting + 214, o.northing + 11)],
        crs="EPSG:32611",
    ).to_crs("EPSG:4326")
    miss.to_file(raw / "lidar" / "missing_buildings.geojson", driver="GeoJSON")

    allfp = add_lidar_missing_buildings(fp, raw)
    assert len(allfp) == 4 and allfp["source"].iloc[-1] == "lidar" and allfp["building"].iloc[-1] == "house"
    from pipeline.build_buildings import prepare_buildings
    from pipeline.build_terrain import Terrain

    ext = Extent(-500, 500, -500, 500)
    terr = Terrain(np.zeros((101, 101), np.float32), ext, 10.0)
    bdf = prepare_buildings(allfp, terr, TileGrid(ext, 2, 2))
    assert list(bdf["source_id"]) == ["g0", "g1", "g2", "lidar_00000"]
    bdf = lidar_roof_join(bdf, raw, allfp)
    by = bdf.set_index("source_id")
    assert by.loc["g0", "height_m"] == 6.4 and by.loc["g0", "height_rule"] == "lidar" and by.loc["g0", "roof_shape"] == "gabled"
    assert by.loc["g1", "height_rule"] != "lidar"  # 0.5 m ridge is outside hd_world.lidar_height_*
    assert by.loc["g2", "height_rule"] != "lidar"  # absent from the 2014 flight
    assert by.loc["lidar_00000", "height_m"] == 5.5 and by.loc["lidar_00000", "type"] == "house"
    out = tmp_path / "b.geojson"
    write_buildings_geojson(bdf, out)
    props = {f["properties"]["source_id"]: f["properties"] for f in json.loads(out.read_text())["features"]}
    assert props["g0"]["ridge_height_m"] == 6.4 and props["g0"]["roof_type"] == "gable" and props["g0"]["lidar_quality"] == "good"
    assert props["lidar_00000"]["source"] == "lidar" and props["lidar_00000"]["height_source"] == "lidar"
    assert not any(k.startswith("lidar_") and k not in ("lidar_quality", "lidar_status") for k in props["g0"])


# ---------------------------------------------------------------- private streets / paths


def test_overture_missing_roads_and_paths(tmp_path: Path) -> None:
    from pipeline.build_streets import make_road, overture_missing_roads, overture_paths

    raw = tmp_path / "raw"
    (raw / "overture").mkdir(parents=True)

    def ll(line: list[tuple[float, float]]) -> bytes:
        return shapely.to_wkb(LineString([_scene_to_lonlat(x, z) for x, z in line]))

    rows = [
        ("road", "residential", None, ll([(0, 0), (200, 0)])),  # duplicates the existing road
        ("road", "residential", None, ll([(0, 100), (200, 100)])),  # private street (missing)
        ("road", "footway", None, ll([(0, 300), (150, 300)])),  # paseo
        ("road", "footway", "sidewalk", ll([(0, 5), (200, 5)])),  # mapped sidewalk: dropped
        ("road", "path", None, ll([(0, 400), (100, 450)])),  # trail
    ]
    pq.write_table(pa.table({"subtype": [r[0] for r in rows], "class": [r[1] for r in rows], "subclass": [r[2] for r in rows], "geometry": [r[3] for r in rows]}), raw / "overture" / "segment.parquet")
    ext = Extent(-500, 500, -500, 500)
    existing = [make_road(LineString([(0, 0), (200, 0)]), "residential", 1, 1, "Main", 1, 2)]
    extra = overture_missing_roads(raw, ext, existing)
    assert len(extra) == 1 and abs(extra[0].line.centroid.y - 100) < 0.5 and extra[0].lanes_bwd >= 1
    paths = overture_paths(raw, ext)
    assert sorted(k for _, k in paths) == ["paved", "trail"]


# ---------------------------------------------------------------- splat masks


def test_splat_masks_sum_to_255_and_covered_reservoir_is_paved(tmp_path: Path) -> None:
    from pipeline.build_landcover import write_splat_masks

    grid = TileGrid(Extent(0, 100, 0, 100), 1, 1)
    img = np.full((64, 64, 3), (110, 120, 70), np.uint8)  # dry olive scrub
    img[:32, :32] = (235, 235, 232)  # bright grey covered reservoir
    img[32:, 32:] = (40, 70, 110)  # open water
    water = [box(0, 0, 50, 50), box(50, 50, 100, 100)]
    out = write_splat_masks(grid, {"r0_c0": img}, tmp_path, 64, None, water, [])
    a = np.asarray(Image.open(tmp_path / "splat_r0_c0_a.png")).astype(int)
    b = np.asarray(Image.open(tmp_path / "splat_r0_c0_b.png")).astype(int)
    assert out["r0_c0"] == ["terrain/splat_r0_c0_a.png", "terrain/splat_r0_c0_b.png"]
    assert (a.sum(-1) + b.sum(-1) == 255).all()
    assert b[10, 10, 0] > 200 and b[10, 10, 1] < 30  # covered reservoir -> paved, not water
    assert b[50, 50, 1] > 200  # open water stays water


# ---------------------------------------------------------------- synthetic HD build


def test_synthetic_hd_manifest_and_tiles(synthetic_build) -> None:
    d, meta = synthetic_build
    assets = d["assets"]
    m = json.loads((assets / "manifest.json").read_text())
    assert meta["synthetic"] is True and m["synthetic"] is True and m["hd_version"] == 1
    rows, cols = m["grid"]["rows"], m["grid"]["cols"]
    assert len(m["tiles"]) == rows * cols == 64
    tl = m["triangles"]["terrain_lods"]
    assert tl["lod0"] <= 2_000_000 and tl["lod1"] < tl["lod0"] and tl["lod2"] < tl["lod1"]
    t = m["tiles"][rows * cols // 2]
    assert [x["lod"] for x in t["terrain_lods"]] == [0, 1, 2]
    for x in t["terrain_lods"]:
        assert (assets / x["path"]).exists()
    # skirts hang below the tile's surface
    from pipeline.glb import load_glb_meshes

    lod0 = load_glb_meshes(assets / t["terrain_lods"][0]["path"])[0]
    assert lod0.positions[:, 1].min() < t["bounds"]["min_y"] + 0.01
    a = np.asarray(Image.open(assets / t["splat"][0])).astype(int)
    b = np.asarray(Image.open(assets / t["splat"][1])).astype(int)
    assert a.shape[:2] == (m["splat"]["px"], m["splat"]["px"]) and (a.sum(-1) + b.sum(-1) == 255).all()
    # street layers carry the material convention attributes
    have_roads = [x for x in m["tiles"] if x.get("roads") and (assets / x["roads"]).stat().st_size > 200]
    assert have_roads
    gltf, _ = read_glb(assets / have_roads[0]["roads"])
    names = {tuple(sorted(p["attributes"])) for mm in gltf["meshes"] for p in mm["primitives"]}
    assert all("_MAT" in n and "_VARIANT" in n and "TEXCOORD_0" in n for n in names)
    rm = json.loads((d["processed"] / "region_meta.json").read_text())
    assert rm["synthetic"] is True and rm["tiles"] == {"rows": 8, "cols": 8}


def test_raw_edge_shapes_survive_simplification() -> None:
    import networkx as nx
    import osmnx as ox

    from pipeline.build_roads import explode_edge_geometry

    G = nx.MultiDiGraph(crs="EPSG:4326")
    for n, (x, y) in {1: (-117.10, 33.00), 2: (-117.09, 33.01), 3: (-117.08, 33.01)}.items():
        G.add_node(n, x=x, y=y)
    bend = LineString([(-117.10, 33.00), (-117.10, 33.01), (-117.09, 33.01)])  # L-shaped road 1 -> 2
    G.add_edge(1, 2, osmid="w1", highway="residential", oneway=False, reversed=False, length=2000.0, geometry=bend)
    G.add_edge(2, 1, osmid="w1", highway="residential", oneway=False, reversed=True, length=2000.0, geometry=LineString(bend.coords[::-1]))
    G.add_edge(2, 3, osmid="w2", highway="residential", oneway=False, reversed=False, length=900.0)
    G.add_edge(3, 2, osmid="w2", highway="residential", oneway=False, reversed=True, length=900.0)
    assert explode_edge_geometry(G) == 1  # one shared corner vertex for both directions
    S = ox.simplify_graph(G)
    d = S.get_edge_data(1, 2) or S.get_edge_data(1, 3)
    e = next(iter(d.values()))
    assert any(abs(x + 117.10) < 1e-9 and abs(y - 33.01) < 1e-9 for x, y in e["geometry"].coords)  # corner kept
    assert all(n < 8_000_000_000 for n in S.nodes)


def test_roof_plan_l_shape_and_utm_coordinates() -> None:
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon

    from pipeline.building_geom import roof_plan

    L = Polygon([(0, 0), (20, 0), (20, 8), (10, 8), (10, 15), (0, 15)])
    for p in (L, rotate(L, 23), translate(rotate(L, 23), 488000, 3650000)):
        plan = roof_plan(p)
        assert plan is not None and len(plan.rects) == 2 and plan.iou > 0.99
