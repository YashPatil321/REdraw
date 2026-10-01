"""Checks for the Blender-built prop library, hero campuses and prop placements.

Run with the main venv (no bpy needed):  .venv/bin/pytest blender/tests
Built outputs are not committed; tests that need them skip with a hint when absent
(build: .venv-blender/bin/python blender/build_all_assets.py, then .venv/bin/python pipeline/build_props.py).
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "blender"))

PROPS = REPO / "client" / "public" / "assets" / "props"
HERO = REPO / "pipeline" / "hero_overrides"
BUILD_HINT = "build with: .venv-blender/bin/python blender/build_all_assets.py"

trimesh = pytest.importorskip("trimesh")


def _glb_json(path: Path) -> dict:
    d = path.read_bytes()
    magic, version, _length = struct.unpack("<III", d[:12])
    assert magic == 0x46546C67 and version == 2, f"{path} is not glTF 2.0 binary"
    n, kind = struct.unpack("<II", d[12:20])
    assert kind == 0x4E4F534A
    return json.loads(d[20 : 20 + n])


def _manifest() -> list[dict]:
    p = PROPS / "props_manifest.json"
    if not p.exists():
        pytest.skip(f"{p} missing; {BUILD_HINT}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# pure-python modules
# ---------------------------------------------------------------------------


def test_palette_textures_and_uvs(tmp_path: Path) -> None:
    from rdlib import palette

    base, mr = palette.write_textures(tmp_path)
    assert base.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and mr.exists()
    uvs = {palette.cell_uv(k) for k in palette.keys()}
    assert len(uvs) == len(palette.keys()), "palette cells must be unique"
    assert len(palette.PALETTE) <= palette.GRID * palette.GRID


def test_material_library_contract() -> None:
    from rdlib import materials

    m = materials.MATERIALS
    assert m["paint"].tint, "client tints material 'paint' (manifest tintable_region)"
    for k in ("headlight", "taillight"):
        assert m[k].emissive_strength > 0 and m[k].emissive_hex
    assert m["foliage"].alpha_mask and m["foliage"].double_sided
    shares = sum(c["share"] for c in materials.PAINT_COLORS)
    assert abs(shares - 1.0) < 1e-6


def test_paint_shares_match_assumptions() -> None:
    from rdlib import materials

    from pipeline.config import assumption

    a = assumption("props.vehicle_paint_shares")
    assert {c["name"]: c["share"] for c in materials.PAINT_COLORS} == a


# ---------------------------------------------------------------------------
# prop glbs + manifest
# ---------------------------------------------------------------------------

REQUIRED = {"id", "kind", "file", "triangles", "footprint_radius_m", "length_m", "forward_axis",
            "tintable_region", "suggested_scale_range"}
BUDGET = {"vehicle": 1500, "tree": 4000, "shrub": 2000, "lamp": 1000, "groundcover": 200}
KINDS = {"vehicle", "tree", "shrub", "lamp", "groundcover"}
SPECIES = {"tree_oak", "tree_palm_fan", "tree_palm_queen", "tree_pine_canary", "tree_eucalyptus", "tree_jacaranda",
           "tree_ficus", "tree_street", "shrub_bougainvillea", "succulent_agave", "hedge", "grass_tuft", "shrub",
           "grass_ornamental"}


def test_manifest_entries_and_files() -> None:
    man = _manifest()
    assert isinstance(man, list) and man
    ids = [e["id"] for e in man]
    assert len(ids) == len(set(ids))
    kinds = {e["kind"] for e in man}
    assert kinds <= KINDS
    assert SPECIES <= set(ids), f"vegetation species missing: {SPECIES - set(ids)}"
    for need in ("vehicle", "tree", "shrub", "lamp"):
        assert need in kinds
    for e in man:
        assert REQUIRED <= set(e), f"{e['id']} missing {REQUIRED - set(e)}"
        assert (PROPS / e["file"]).exists(), e["file"]
        assert e["forward_axis"] == "-z"
        lo, hi = e["suggested_scale_range"]
        assert 0 < lo <= 1 <= hi


@pytest.mark.parametrize("kind", ["vehicle", "tree", "shrub", "lamp", "groundcover"])
def test_glbs_parse_and_match_manifest(kind: str) -> None:
    for e in [x for x in _manifest() if x["kind"] == kind]:
        path = PROPS / e["file"]
        scene = trimesh.load(path, force="scene")
        tris = sum(len(g.faces) for g in scene.geometry.values())
        assert tris == e["triangles"], f"{e['id']}: glb {tris} tris, manifest {e['triangles']}"
        budget = 3000 if e["id"] == "school_bus" else BUDGET[kind]
        assert tris < budget, f"{e['id']}: {tris} >= {budget}"
        lo, hi = scene.bounds
        assert lo[1] > -0.6 and lo[1] < 0.3, f"{e['id']}: origin must be on the ground (min y {lo[1]:.2f})"
        assert abs((lo[0] + hi[0]) / 2) < max(1.0, 0.25 * (hi[0] - lo[0])), f"{e['id']}: not centered in x"


def test_vehicle_axes_dimensions_and_lights() -> None:
    for e in [x for x in _manifest() if x["kind"] == "vehicle"]:
        path = PROPS / e["file"]
        scene = trimesh.load(path, force="scene")
        lo, hi = scene.bounds
        length, width, height = hi[2] - lo[2], hi[0] - lo[0], hi[1] - lo[1]
        assert length > width, f"{e['id']}: length must run along z"
        assert abs(length - e["length_m"]) < 0.05
        assert 1.3 < height < 3.5
        if e["id"] == "car_sedan":
            assert abs(length - 4.88) < 0.15 and abs(height - 1.44) < 0.05
        js = _glb_json(path)
        assert "KHR_materials_emissive_strength" in js.get("extensionsUsed", [])
        names = {m["name"]: m for m in js["materials"]}
        assert "headlight" in names and "taillight" in names
        attrs = js["meshes"][0]["primitives"][0]["attributes"]
        assert {"_LIGHT", "_TINT", "_ALBEDO"} <= set(attrs)
        # headlights at the front (-z), taillights at the rear (+z)
        g = trimesh.load(path, force="scene")
        cent = {}
        for name, geom in g.geometry.items():
            mat = getattr(geom.visual, "material", None)
            mname = getattr(mat, "name", "") or name
            cent.setdefault(mname, []).append(geom.vertices.mean(axis=0))
        if "headlight" in cent and "taillight" in cent:
            hz = np.mean([c[2] for c in cent["headlight"]])
            tz = np.mean([c[2] for c in cent["taillight"]])
            assert hz < tz, f"{e['id']}: headlights must face -z"


def test_tree_lods() -> None:
    """Every tree: LOD0 alpha cards (2-4k tris for broadleaf / conifer) + LOD1 baked impostor quads."""
    trees = [x for x in _manifest() if x["kind"] == "tree"]
    assert len(trees) >= 8
    for e in trees:
        lods = e["lods"]
        assert [lv["level"] for lv in lods] == [0, 1] and lods[0]["file"] == e["file"]
        assert lods[0]["max_distance_m"] == lods[1]["min_distance_m"] > 0
        assert e["lod1_file"] == lods[1]["file"]
        if not e["id"].startswith("tree_palm"):
            assert 2000 <= e["triangles"] <= 4000, f"{e['id']}: LOD0 {e['triangles']} tris"
        path = PROPS / lods[1]["file"]
        scene = trimesh.load(path, force="scene")
        tris = sum(len(g.faces) for g in scene.geometry.values())
        assert tris == lods[1]["triangles"] and tris <= 96, f"{e['id']} LOD1: {tris} tris"
        lo, hi = scene.bounds
        assert abs(hi[1] - e["height_m"]) < 0.15 * e["height_m"], f"{e['id']}: LOD1 height {hi[1]:.1f} vs {e['height_m']}"
        js = _glb_json(path)
        m = js["materials"][0]
        assert m["alphaMode"] == "MASK" and "baseColorTexture" in m["pbrMetallicRoughness"]
        assert e["crown_radius_m"] > 0.5


def test_trees_use_alpha_mask_atlas() -> None:
    for e in [x for x in _manifest() if x["kind"] in ("tree", "shrub", "groundcover")]:
        js = _glb_json(PROPS / e["file"])
        mats = js["materials"]
        assert len(mats) == 1 and mats[0]["alphaMode"] == "MASK" and mats[0].get("doubleSided")
        assert "baseColorTexture" in mats[0]["pbrMetallicRoughness"]
        assert js["images"][0]["mimeType"] == "image/png"


# ---------------------------------------------------------------------------
# hero campuses
# ---------------------------------------------------------------------------


def test_hero_overrides() -> None:
    p = HERO / "hero_overrides.json"
    if not p.exists():
        pytest.skip(f"{p} missing; {BUILD_HINT}")
    heroes = json.loads(p.read_text())
    assert {h["id"] for h in heroes} >= {"del_norte_hs", "design39", "4s_commons"}
    for h in heroes:
        for k in ("id", "school_id", "name", "lat", "lon", "rotation_deg", "footprint_radius_m", "glb", "replaces", "verified"):
            assert k in h, f"{h.get('id')}: missing {k}"
        assert h["verified"] is False
        assert 32.9 < h["lat"] < 33.1 and -117.2 < h["lon"] < -117.0
        glb = HERO / h["glb"]
        assert glb.exists()
        js = _glb_json(glb)
        assert not ({"KHR_draco_mesh_compression", "EXT_meshopt_compression"} & set(js.get("extensionsUsed", [])))
        scene = trimesh.load(glb, force="scene")
        lo, hi = scene.bounds
        r = h["footprint_radius_m"]
        # local meters, origin = campus center, y up: ground near 0, extent about the footprint radius
        assert -2.0 < lo[1] < 0.5 and hi[1] < 40
        assert max(abs(lo[0]), abs(hi[0]), abs(lo[2]), abs(hi[2])) < r * 2.2
        trees = HERO / f"{h['id']}_trees.json"
        assert trees.exists()
        t = json.loads(trees.read_text())
        assert {"trees", "lamps", "parked_cars"} <= set(t)


def test_hero_glbs_load_in_pipeline() -> None:
    p = HERO / "hero_overrides.json"
    if not p.exists():
        pytest.skip(f"{p} missing; {BUILD_HINT}")
    from pipeline.glb import load_glb_meshes

    for h in json.loads(p.read_text()):
        meshes = load_glb_meshes(HERO / h["glb"])
        assert meshes and len(meshes) <= 6, "few primitives (vertex-colored class materials)"
        assert all(m.colors is not None for m in meshes), "hero colors travel as COLOR_0"


def test_hero_material_atlas_attributes() -> None:
    """Heroes carry _MAT / _VARIANT (materials_manifest ids) + atlas UVs on every vertex."""
    p = HERO / "hero_overrides.json"
    if not p.exists():
        pytest.skip(f"{p} missing; {BUILD_HINT}")
    from pipeline.glb import load_glb_meshes

    man = _materials_manifest()
    n_var = {int(k): len(v["variants"]) for k, v in man["materials"].items()}
    n_var[6] = len(man["atlases"]["ground"]["cells"])
    n_var[7] = 1
    for h in json.loads(p.read_text()):
        seen: set[int] = set()
        for m in load_glb_meshes(HERO / h["glb"]):
            assert {"_MAT", "_VARIANT"} <= set(m.custom), f"{h['id']}: {m.name} lacks _MAT/_VARIANT"
            assert m.uvs is not None and len(m.uvs) == len(m.positions)
            mat = np.asarray(m.custom["_MAT"]).astype(int)
            var = np.asarray(m.custom["_VARIANT"]).astype(int)
            assert mat.min() >= 0 and mat.max() <= 7
            for k in np.unique(mat):
                assert var[mat == k].max() < n_var[int(k)], f"{h['id']}: _VARIANT out of range for _MAT {k}"
            seen |= set(np.unique(mat).tolist())
        assert {0, 2, 6} <= seen, f"{h['id']}: expected stucco walls, flat roofs and ground, got {sorted(seen)}"
        t = json.loads((HERO / f"{h['id']}_trees.json").read_text())
        cls = {k["cls"] for k in t.get("keepout", [])}
        assert "building" in cls, f"{h['id']}: trees json needs keep-out polygons for lidar trees"


# ---------------------------------------------------------------------------
# placements
# ---------------------------------------------------------------------------


def _placements() -> tuple[dict, np.ndarray]:
    hp = PROPS / "placements.json"
    if not hp.exists():
        pytest.skip(f"{hp} missing; run .venv/bin/python pipeline/build_props.py")
    h = json.loads(hp.read_text())
    raw = (PROPS / h["bin"]).read_bytes()
    return h, np.frombuffer(raw, dtype="<f4")


def test_placements_header_and_binary() -> None:
    h, flat = _placements()
    assert h["format"] == "redraw-placements" and h["version"] == 1
    rec = h["record"]
    assert rec["fields"] == ["x", "y", "z", "rot_y", "scale", "prop_index"] == h["fields"]
    assert rec["dtype"] == "float32" and rec["endianness"] == "little" and rec["stride_bytes"] == 24 == h["stride"]
    assert flat.size == h["count"] * 6, "binary size must be count * 6 float32"
    arr = flat.reshape(-1, 6)
    man_ids = {e["id"]: e for e in _manifest()}
    total = 0
    for i, p in enumerate(h["props"]):
        assert p["index"] == i and p["id"] in man_ids and p["file"] == man_ids[p["id"]]["file"]
        assert p["offset"] == p["first_record"] * 24
        seg = arr[p["first_record"] : p["first_record"] + p["count"]]
        assert np.all(seg[:, 5] == i), "sections are contiguous per prop"
        assert sum(c[2] for c in p["cells"]) == p["count"]
        total += p["count"]
        if p["count"]:
            lo, hi = man_ids[p["id"]]["suggested_scale_range"]
            if p["kind"] == "tree" and h.get("trees_source", "procedural") != "procedural":
                lo, hi = 0.2, 2.5  # lidar trees: measured size / model size (props.lidar_trees.scale_clamp), young gap trees
            assert seg[:, 4].min() >= lo - 1e-4 and seg[:, 4].max() <= hi + 1e-4, p["id"]
    assert total == h["count"]
    assert np.all(np.isfinite(arr))
    assert np.all(np.abs(arr[:, 3]) <= 2 * math.pi + 1e-3)
    g = h["cells"]
    assert np.all(arr[:, 0] >= g["min_x"] - 1) and np.all(arr[:, 2] >= g["min_z"] - 1)
    assert np.all(arr[:, 0] <= g["min_x"] + g["cols"] * g["size_m"] + 1)
    assert np.all(arr[:, 2] <= g["min_z"] + g["rows"] * g["size_m"] + 1)


def test_placement_helpers(tmp_path: Path) -> None:
    from PIL import Image

    from pipeline.build_props import Terrain, _rot_toward, cells_index

    # rot_y turns forward (-z) toward the target direction
    for dx, dz in ((1, 0), (0, 1), (-1, 0), (0, -1), (0.6, -0.8)):
        th = _rot_toward(np.array([dx, dz]))
        fwd = (-math.sin(th), -math.cos(th))  # three.js R_y(th) * (0, 0, -1)
        assert abs(fwd[0] - dx) < 1e-9 and abs(fwd[1] - dz) < 1e-9
    # heightmap sampling: pixel (0,0) = (min_x, min_z), +col = +x, +row = +z
    a = (np.arange(4)[None, :] * 100 + np.arange(3)[:, None] * 1000).astype(np.uint16)
    Image.fromarray(a).save(tmp_path / "h.png")
    meta = {"min_x": 0, "max_x": 30, "min_z": 0, "max_z": 20, "elev_scale": 0.01, "elev_offset": 100}
    t = Terrain(meta, tmp_path / "h.png")
    assert abs(t.sample(np.array([10.0]), np.array([0.0]))[0] - 101.0) < 1e-6
    assert abs(t.sample(np.array([0.0]), np.array([10.0]))[0] - 110.0) < 1e-6
    # sections + cells
    rec = np.array([[10, 0, 10, 0, 1, 1], [600, 0, 10, 0, 1, 0], [20, 0, 20, 0, 1, 1], [700, 0, 700, 0, 1, 1]], float)
    arr, grid, sec = cells_index(rec, {"min_x": 0, "max_x": 1000, "min_z": 0, "max_z": 1000}, 2)
    assert list(arr[:, 5]) == [0, 1, 1, 1]
    assert sec[0]["count"] == 1 and sec[1]["first_record"] == 1 and sec[1]["count"] == 3
    assert sec[1]["cells"] == [[0, 1, 2], [3, 3, 1]] and grid["cols"] == 2


def test_placements_inside_region_bbox() -> None:
    h, flat = _placements()
    if "region_bbox" not in h:
        pytest.skip("placements.json predates the region clip; rerun pipeline/build_props.py")
    from shapely import contains_xy

    from pipeline.build_props import region_polygon
    from pipeline.config import region

    assert h["region_bbox"] == region()["bbox"], "placements were built for another bbox"
    arr = flat.reshape(-1, 6)
    poly = region_polygon()
    assert np.all(contains_xy(poly.buffer(1.0), arr[:, 0], arr[:, 2])), "every placement lies inside the region bbox"
    if h.get("trees_source") == "lidar+gaps":
        assert h["stats"]["lidar_trees"] > 0


# ---------------------------------------------------------------------------
# material atlases (client/public/assets/materials)
# ---------------------------------------------------------------------------

MAT_DIR = REPO / "client" / "public" / "assets" / "materials"
# Other agents index these: names and order must never change (append only).
STABLE_VARIANTS = {
    "0": ["stucco_smooth", "stucco_sand", "stucco_lace", "stucco_catface", "stucco_weathered", "stucco_scored", "stone_veneer"],
    "1": ["s_tile_terracotta", "s_tile_blend", "s_tile_brown", "s_tile_aged", "barrel_mission", "flat_tile_brown",
          "flat_tile_grey", "flat_tile_charcoal", "flat_tile_sandstone", "solar_panel"],
    "2": ["flat_tpo", "flat_tpo_grime", "flat_gravel", "flat_modbit", "concrete_deck", "standing_seam"],
    "3": ["glass_curtain"],
    "4": ["stucco_smooth", "stone_veneer"],
    "5": ["garage_2car", "garage_3car"],
}
STABLE_GROUND = ["asphalt_fresh", "asphalt_worn", "asphalt_parking", "concrete_sidewalk", "concrete_driveway", "curb_gutter",
                 "pavers", "concrete_plaza", "grass_lawn", "grass_patchy", "chaparral", "coastal_sage", "decomposed_granite",
                 "bare_dirt", "mulch", "pool_water"]


def _materials_manifest() -> dict:
    p = MAT_DIR / "materials_manifest.json"
    if not p.exists():
        pytest.skip(f"{p} missing; build with: .venv-blender/bin/python blender/build_all_assets.py --only materials")
    return json.loads(p.read_text())


def test_materials_manifest_names_stable() -> None:
    man = _materials_manifest()
    for k, names in STABLE_VARIANTS.items():
        got = man["materials"][k]["variants"]
        assert got[: len(names)] == names, f"_MAT {k} variants changed: {got}"
    ground = [c["name"] for c in man["atlases"]["ground"]["cells"]]
    assert ground[: len(STABLE_GROUND)] == STABLE_GROUND
    assert [c["index"] for c in man["atlases"]["ground"]["cells"]] == list(range(len(ground)))


def test_materials_textures_and_cells() -> None:
    from PIL import Image

    man = _materials_manifest()
    for name, a in man["atlases"].items():
        files = a["files"]
        for kind in ("albedo", "normal", "orm"):
            path = MAT_DIR / files[kind]
            assert path.exists(), path
            with Image.open(path) as im:
                assert list(im.size) == a["size_px"], f"{path.name}: {im.size}"
                assert im.mode == "RGB"
        names = [c["name"] for c in a["cells"]]
        assert len(names) == len(set(names)), f"{name}: duplicate cell names"
        for c in a["cells"]:
            for sc in c["cells"]:
                u0, v0, u1, v1 = sc["uv_inner"]
                assert 0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1, f"{name}/{c['name']}"
            assert len(c["mean_albedo_linear"]) == 3 and all(0 < x < 1 for x in c["mean_albedo_linear"])
    # normal maps: mostly-flat tangent space (mean blue high, mean red/green near 0.5)
    with Image.open(MAT_DIR / man["atlases"]["facade_walls"]["files"]["normal"]) as im:
        n = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    assert abs(n[..., 0].mean() - 0.5) < 0.05 and abs(n[..., 1].mean() - 0.5) < 0.05 and n[..., 2].mean() > 0.8


# ---------------------------------------------------------------------------
# lidar tree placement (synthetic mini world, no built data needed)
# ---------------------------------------------------------------------------


def _mini_world(tmp_path: Path) -> dict:
    import geopandas as gpd
    import pandas as pd
    from PIL import Image
    from shapely.geometry import box

    from pipeline.build_props import Terrain

    man = [
        {"id": "tree_palm_fan", "kind": "tree", "file": "v/a.glb", "height_m": 19.5, "crown_radius_m": 3.0, "suggested_scale_range": [0.75, 1.25]},
        {"id": "tree_palm_queen", "kind": "tree", "file": "v/b.glb", "height_m": 11.8, "crown_radius_m": 3.4, "suggested_scale_range": [0.75, 1.25]},
        {"id": "tree_oak", "kind": "tree", "file": "v/c.glb", "height_m": 9.5, "crown_radius_m": 6.0, "suggested_scale_range": [0.8, 1.2]},
        {"id": "tree_street", "kind": "tree", "file": "v/d.glb", "height_m": 8.8, "crown_radius_m": 3.5, "suggested_scale_range": [0.8, 1.2]},
        {"id": "tree_eucalyptus", "kind": "tree", "file": "v/e.glb", "height_m": 22.6, "crown_radius_m": 5.0, "suggested_scale_range": [0.8, 1.2]},
        {"id": "tree_pine_canary", "kind": "tree", "file": "v/f.glb", "height_m": 20.5, "crown_radius_m": 3.4, "suggested_scale_range": [0.8, 1.2]},
        {"id": "tree_jacaranda", "kind": "tree", "file": "v/g.glb", "height_m": 9.2, "crown_radius_m": 4.5, "suggested_scale_range": [0.8, 1.2]},
        {"id": "tree_ficus", "kind": "tree", "file": "v/h.glb", "height_m": 8.7, "crown_radius_m": 4.0, "suggested_scale_range": [0.8, 1.2]},
    ]
    Image.fromarray(np.zeros((8, 8), np.uint16)).save(tmp_path / "h.png")
    terr = Terrain({"min_x": -4000, "max_x": -2000, "min_z": 2000, "max_z": 4000, "elev_scale": 0.01, "elev_offset": 100.0},
                   tmp_path / "h.png")
    b = gpd.GeoDataFrame({"type": ["house", "commercial"]}, geometry=[box(-3010, 3000, -2990, 3015), box(-3300, 3000, -3200, 3060)])
    # one east-west 2-lane residential street at z = 2950
    edges = pd.DataFrame({"u": [1], "v": [2], "osmid": [7], "lanes": [2], "oneway": [False], "highway": ["residential"],
                          "edge_idx": [0], "geometry": [[-3500.0, 0.0, 2950.0, -2500.0, 0.0, 2950.0]]})
    nodes = pd.DataFrame({"node_id": [1, 2], "x": [-3500.0, -2500.0], "z": [2950.0, 2950.0], "signalized": [False, False]})
    return {"meta": {}, "terrain": terr, "buildings": b, "edges": edges, "nodes": nodes, "manifest": man,
            "region_poly": box(-4000, 2000, -2000, 4000)}


def test_lidar_trees_real_positions_species_and_scale(tmp_path: Path) -> None:
    pytest.importorskip("geopandas")
    import pandas as pd

    from pipeline.build_props import Placer

    inp = _mini_world(tmp_path)
    df = pd.DataFrame({
        "x": [-3000.0, -2800.0, -2980.0, -3250.0, -2600.0, -2400.0, -1000.0],
        "z": [3007.0, 2951.0, 3030.0, 3080.0, 3400.0, 3500.0, 3000.0],
        "h": [9.0, 8.0, 17.0, 9.0, 8.5, 18.0, 9.0],
        "r": [3.0, 3.0, 1.6, 4.0, 3.5, 6.0, 3.0],
        "cls": ["broadleaf", "broadleaf", "palm", "broadleaf", "conifer", "broadleaf", "broadleaf"],
    })
    pl = Placer(inp, 1)
    st = pl.lidar_trees(df)
    assert st["lidar_dropped_building"] == 1  # inside the house footprint
    assert st["lidar_outside_region"] == 1
    assert st["lidar_moved_off_road"] == 1  # top over the travel lane -> back to the curb
    assert st["lidar_trees"] == 5
    recs = {(round(r[0]), round(r[1])): r for r in pl.recs}
    road = [r for r in pl.recs if abs(r[0] + 2800) < 0.01][0]
    assert abs(road[1] - 2950) >= 3.6 + 2.4 + 0.6 - 0.01  # half width (2 lanes + parking) + clearance
    palm = recs[(-2980, 3030)]
    k = st["lidar_crown_scale_palm"]  # measured crowns rescaled to the model crown/height ratio (>= 1)
    assert 1.0 <= k <= 4.0
    assert palm[5] == "tree_palm_fan" and abs(palm[4] - (17 / 19.5) ** 0.7 * (1.6 * k / 3.0) ** 0.3) < 2e-3
    assert recs[(-2600, 3400)][5] == "tree_pine_canary"
    assert recs[(-3250, 3080)][5] in {"tree_ficus", "tree_street", "tree_jacaranda", "tree_eucalyptus"}
    # gap logic: a rule-based tree is refused next to real trees inside the survey
    assert not pl.tree_gap(-2980.0, 3031.0)


def test_lidar_loader_frame_and_stale(tmp_path: Path) -> None:
    import pandas as pd

    from pipeline.build_props import load_lidar_trees
    from pipeline.geo import scene_origin

    o = scene_origin()
    pd.DataFrame({"easting": [o.easting + 10.0, o.easting], "northing": [o.northing + 20.0, o.northing],
                  "height_m": [9.0, 9.0], "crown_radius_m": [3.0, 3.0], "species_guess": ["palm", "weird"],
                  "ground_y": [100.0, 100.0], "ground_lidar_m": [100.2, 104.0]}).to_parquet(tmp_path / "t.parquet")
    df, st = load_lidar_trees(tmp_path / "t.parquet")
    assert st["lidar_trees_stale_regraded"] == 1 and len(df) == 1
    assert abs(df["x"].iloc[0] - 10.0) < 1e-6 and abs(df["z"].iloc[0] + 20.0) < 1e-6  # z = south
    assert load_lidar_trees(tmp_path / "missing.parquet") == (None, {})


def test_region_polygon_and_check() -> None:
    from shapely.geometry import Point

    from pipeline.build_props import Fail, check_region, region_polygon
    from pipeline.config import region
    from pipeline.geo import latlon_to_scene

    poly = region_polygon()
    bb = region()["bbox"]
    assert poly.contains(Point(0, 0))
    assert poly.contains(Point(*latlon_to_scene(33.0136, -117.1221)))  # Del Norte HS
    assert not poly.contains(Point(*latlon_to_scene(bb["north"] + 0.002, -117.12)))
    check_region({"bbox": dict(bb)})
    with pytest.raises(Fail):
        check_region({"bbox": {**bb, "south": bb["south"] - 0.05}})
