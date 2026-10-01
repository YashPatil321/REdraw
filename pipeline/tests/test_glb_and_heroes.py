"""GLB writer/reader round trip, `_BUILDING_ID` attribute, and hero override placement."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pipeline.build_buildings import apply_heroes, build_building_tiles, flatten_terrain_for_heroes, hero_transform, load_heroes, prepare_buildings
from pipeline.build_terrain import Terrain
from pipeline.common import Extent, TileGrid
from pipeline.geo import scene_origin, scene_to_latlon
from pipeline.glb import MeshData, load_glb_meshes, read_accessor, read_glb, write_glb


def _box_mesh(size: float = 10.0, color: tuple[float, float, float, float] = (1, 0, 0, 1)) -> MeshData:
    h = size / 2
    pos = np.array([[-h, 0, -h], [h, 0, -h], [h, 0, h], [-h, 0, h], [-h, size, -h], [h, size, -h], [h, size, h], [-h, size, h]], dtype=np.float32)
    idx = np.array([4, 6, 5, 4, 7, 6, 0, 1, 2, 0, 2, 3, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0], dtype=np.uint32)
    return MeshData(name="box", positions=pos, indices=idx, base_color=color)


def test_glb_roundtrip_with_custom_attribute(tmp_path: Path) -> None:
    m = _box_mesh()
    m.custom = {"_BUILDING_ID": np.full(8, 42.0, dtype=np.float32)}
    m.colors = np.tile([10, 20, 30, 255], (8, 1)).astype(np.uint8)
    p = tmp_path / "b.glb"
    assert write_glb(p, [m]) == 12
    gltf, binb = read_glb(p)
    attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
    assert "_BUILDING_ID" in attrs and "COLOR_0" in attrs
    assert np.all(read_accessor(gltf, binb, attrs["_BUILDING_ID"]) == 42.0)
    back = load_glb_meshes(p)
    assert len(back) == 1 and np.allclose(back[0].positions, m.positions)
    assert back[0].colors is not None and tuple(back[0].colors[0]) == (10, 20, 30, 255)


def test_empty_glb_is_valid(tmp_path: Path) -> None:
    p = tmp_path / "e.glb"
    write_glb(p, [])
    gltf, _ = read_glb(p)
    assert gltf["scenes"] == [{}] and "nodes" not in gltf


def test_node_transforms_are_baked(tmp_path: Path) -> None:
    p = tmp_path / "t.glb"
    write_glb(p, [_box_mesh()])
    gltf, binb = read_glb(p)
    gltf["nodes"][0]["translation"] = [100.0, 0.0, 0.0]
    js = json.dumps(gltf).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    import struct

    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binb), 0x004E4942) + binb
    p.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
    m = load_glb_meshes(p)[0]
    assert m.positions[:, 0].min() == pytest.approx(95.0)


def test_hero_transform_rotation_convention() -> None:
    m = MeshData(name="p", positions=np.array([[10.0, 0.0, 0.0]], dtype=np.float32), indices=np.zeros(3, dtype=np.uint32))
    out = hero_transform([m], 1000.0, 50.0, 2000.0, 90.0)[0]
    # +90 deg about +y turns east (+x) toward north (-z)
    assert np.allclose(out.positions[0], [1000.0, 50.0, 1990.0], atol=1e-4)


def test_hero_override_end_to_end(tmp_path: Path) -> None:
    """hero_overrides.json drops footprints in radius, reuses the school building id, lands in the tile."""
    hero_dir = tmp_path / "hero"
    hero_dir.mkdir()
    write_glb(hero_dir / "campus.glb", [_box_mesh(20.0, (0.2, 0.4, 0.8, 1.0)), _box_mesh(5.0)])
    lat, lon = scene_to_latlon(50.0, 50.0)
    (hero_dir / "hero_overrides.json").write_text(json.dumps([{"school_id": "test_school", "lat": lat, "lon": lon, "rotation_deg": 30, "footprint_radius_m": 40, "glb": "campus.glb", "name": "Test Campus"}]))
    heroes = load_heroes(hero_dir)
    assert len(heroes) == 1 and heroes[0].kind == "school"

    ext = Extent(-200.0, 200.0, -200.0, 200.0)
    elev = np.full((41, 41), 100.0, dtype=np.float32)
    elev[25, 25] = 140.0  # bump under the campus that flattening must remove
    t = Terrain(elev, ext, 10.0)
    flatten_terrain_for_heroes(t, heroes)
    assert t.sample(50.0, 50.0) == pytest.approx(t.sample(55.0, 55.0))
    grid = TileGrid(ext, 2, 2)
    o = scene_origin()

    def utm_box(x0: float, z0: float, x1: float, z1: float):
        return box(x0 + o.easting, o.northing - z1, x1 + o.easting, o.northing - z0)

    fp = gpd.GeoDataFrame(
        {"building": ["school", "school", "house", "house"], "school_id": ["test_school", "test_school", None, None], "osm_sort_key": ["a", "b", "c", "d"]},
        geometry=[utm_box(30, 30, 70, 60), utm_box(40, 65, 60, 75), utm_box(-150, -150, -138, -136), utm_box(55, 30, 60, 35)],
        crs="EPSG:32611",
    )
    bdf = prepare_buildings(fp, t, grid)
    main_id = int(bdf.loc[bdf["area_m2"].idxmax(), "id"])
    bdf2 = apply_heroes(bdf, heroes, t, grid)
    assert heroes[0].building_id == main_id
    assert set(bdf2["id"]) == {main_id, int(bdf.loc[bdf["centroid_x"] < 0, "id"].iloc[0])}
    row = bdf2[bdf2["id"] == main_id].iloc[0]
    assert row["name"] == "Test Campus" and row["type"] == "school" and row["height_rule"] == "hero"
    tiles, tris, hero_out = build_building_tiles(bdf2, grid, tmp_path / "tiles", heroes)
    assert hero_out[0]["tile"] == "r1_c1" and hero_out[0]["building_id"] == main_id
    gltf, binb = read_glb(tmp_path / "tiles" / "buildings_r1_c1.glb")
    prims = [p for m in gltf["meshes"] for p in m["primitives"]]
    assert len(prims) == 2  # two hero primitives keep their own materials
    ids = np.concatenate([read_accessor(gltf, binb, p["attributes"]["_BUILDING_ID"]) for p in prims])
    assert np.all(ids == main_id)
    colors = {tuple(gltf["materials"][p["material"]]["pbrMetallicRoughness"]["baseColorFactor"]) for p in prims}
    assert (0.2, 0.4, 0.8, 1.0) in colors
    ys = np.concatenate([read_accessor(gltf, binb, p["attributes"]["POSITION"])[:, 1] for p in prims])
    assert ys.min() == pytest.approx(t.sample(50.0, 50.0), abs=1e-3)


def test_no_hero_json_means_no_heroes(tmp_path: Path) -> None:
    assert load_heroes(tmp_path) == []
