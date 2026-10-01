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
BUDGET = {"vehicle": 1500, "tree": 2000, "shrub": 2000, "lamp": 1000}


def test_manifest_entries_and_files() -> None:
    man = _manifest()
    assert isinstance(man, list) and man
    ids = [e["id"] for e in man]
    assert len(ids) == len(set(ids))
    kinds = {e["kind"] for e in man}
    assert kinds <= {"vehicle", "tree", "shrub", "lamp"}
    for need in ("vehicle", "tree", "shrub", "lamp"):
        assert need in kinds
    for e in man:
        assert REQUIRED <= set(e), f"{e['id']} missing {REQUIRED - set(e)}"
        assert (PROPS / e["file"]).exists(), e["file"]
        assert e["forward_axis"] == "-z"
        lo, hi = e["suggested_scale_range"]
        assert 0 < lo <= 1 <= hi


@pytest.mark.parametrize("kind", ["vehicle", "tree", "shrub", "lamp"])
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


def test_trees_use_alpha_mask_atlas() -> None:
    for e in [x for x in _manifest() if x["kind"] in ("tree", "shrub")]:
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
