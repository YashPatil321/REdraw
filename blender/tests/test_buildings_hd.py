"""HD Blender buildings: pure-python model checks + checks of the built tiles.

Run with the main venv (no bpy needed):  .venv/bin/pytest blender/tests/test_buildings_hd.py
Built outputs are not committed; tests that need them skip with a hint when absent. Build:
    .venv/bin/python pipeline/build_buildings_blender.py && .venv-blender/bin/python blender/buildings/build.py
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from blender.buildings import geom, model  # noqa: E402

OUT = REPO / "client" / "public" / "assets" / "buildings_hd"
SPECS = REPO / "blender" / "build" / "buildings_hd" / "specs"
STATS = REPO / "blender" / "build" / "buildings_hd" / "stats"
HINT = "build with: .venv/bin/python pipeline/build_buildings_blender.py && .venv-blender/bin/python blender/buildings/build.py"

# brief: "< ~3M"; ~3.35M after adding the 93 lidar-only houses. The client draws LOD0 only within
# suggested_lod1_distance_m of the camera, so this caps download size (~40 MB), not frame cost.
LOD0_BUDGET = 3_500_000
LOD1_BUDGET = 600_000 * 1.05  # brief: "< ~600k"
REQUIRED_ATTRS = {"POSITION", "NORMAL", "TEXCOORD_0", "COLOR_0", "_BUILDING_ID", "_MAT", "_VARIANT"}


def _glb_json(path: Path) -> dict[str, Any]:
    d = path.read_bytes()
    magic, version, _length = struct.unpack("<III", d[:12])
    assert magic == 0x46546C67 and version == 2, f"{path} is not glTF 2.0 binary"
    n, kind = struct.unpack("<II", d[12:20])
    assert kind == 0x4E4F534A
    return json.loads(d[20 : 20 + n])


def _manifest() -> dict[str, Any]:
    p = OUT / "manifest_buildings.json"
    if not p.exists():
        pytest.skip(f"{p} missing; {HINT}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# synthetic specs (scene coords: x east, z south)
# ---------------------------------------------------------------------------


def _spec(ring: list[tuple[float, float]], **kw: Any) -> dict[str, Any]:
    s = {"id": 7, "type": "house", "ring": [list(p) for p in ring], "holes": [], "floor_y": 100.0, "ground_min_y": 99.6,
         "ground_max_y": 100.2, "levels": 2, "roof_source": "heuristic", "eave_h": 5.9, "ridge_h": 8.3, "roof_type": "hip",
         "pitch_deg": 22.0, "ridge_az_deg": None, "streets": [{"d": 8.0, "cls": "residential", "x": 0.0, "z": 25.0}],
         "driveway": None, "wall_rgb": None, "roof_rgb": None}
    s.update(kw)
    return s


RECT = [(-8.0, -5.0), (8.0, -5.0), (8.0, 5.0), (-8.0, 5.0)]  # street to the south (+z)
ELL = [(-8.0, -5.0), (8.0, -5.0), (8.0, 11.0), (1.0, 11.0), (1.0, 5.0), (-8.0, 5.0)]


def _normals(P: np.ndarray) -> np.ndarray:
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def test_hip_envelope_is_straight_skeleton() -> None:
    pcs = [geom.rect_piece((0, 0, 10, 6), 0.0, 0.5, "hip", 0.0)]
    faces = geom.envelope_faces(pcs)
    assert len(faces) == 4
    assert math.isclose(float(np.nanmax(geom.envelope_height(pcs, np.array([[5.0, 3.0]])))), 1.5, abs_tol=1e-9)
    # L shape: rectangles cover it and the envelope is continuous at the valley
    poly = geom.Polygon([(0, 0), (10, 0), (10, 10), (4, 10), (4, 6), (0, 6)])
    rects = geom.max_rectangles(poly)
    assert len(rects) == 2
    pcs = [geom.rect_piece(r, 0.0, 0.5, "hip", 0.0) for r in rects]
    z = geom.envelope_height(pcs, np.array([[7.0, 3.0], [4.0, 3.0], [7.0, 8.0]]))
    assert np.all(np.isfinite(z)) and np.all(z <= 2.0 + 1e-9)


def test_regularize_snaps_noisy_rectilinear_ring() -> None:
    rng = np.random.default_rng(0)
    ring = np.array(ELL, float) + rng.normal(0, 0.12, (len(ELL), 2))
    ring = np.column_stack([ring[:, 0], -ring[:, 1]])
    ring = ring if geom.poly_area(ring) > 0 else ring[::-1]
    fp = geom.regularize(ring)
    assert fp.rect and fp.iou > 0.9 and len(fp.local) == 6


@pytest.mark.parametrize("ring,rtype", [(RECT, "hip"), (ELL, "hip"), (ELL, "gable"), (RECT, "flat"), (ELL, "complex"), (RECT, "shed")])
def test_building_attributes_and_conventions(ring: list[tuple[float, float]], rtype: str) -> None:
    btype = "commercial" if rtype == "flat" else "house"
    r = model.build_building(_spec(ring, roof_type=rtype, type=btype), 0)
    P, UV, A = r.soup.arrays()
    assert len(P) > 20
    assert set(np.unique(A[:, 0])) <= {0, 1, 2, 3, 4, 5}
    for m, nvar in ((0, 7), (1, 10), (2, 6), (3, 1), (4, 2), (5, 2)):
        v = A[A[:, 0] == m, 1]
        assert np.all((v >= 0) & (v < nvar) & (v == np.round(v)))
    assert np.all(A[:, 5] == 7) and np.all((A[:, 2:5] >= 0) & (A[:, 2:5] <= 1))
    assert np.all(np.isfinite(P)) and np.all(np.isfinite(UV))
    # walls: v = (z - floor) / 3 and outward normals (away from the footprint centroid)
    walls = A[:, 0] == 0
    n = _normals(P)
    vert = walls & (np.abs(n[:, 2]) < 0.05)
    floor = 100.0 + (0.15 if btype == "house" else 0.0)
    assert np.allclose(UV[vert][..., 1], (P[vert][..., 2] - floor) / 3.0, atol=1e-6)
    import shapely

    fp = geom.Polygon([(x, -z) for x, z in ring]).buffer(0.05)
    mid = P[vert].mean(axis=1)[:, :2]
    behind = mid - 0.2 * n[vert][:, :2]  # the side a wall faces away from is inside the building
    assert shapely.contains_xy(fp, behind[:, 0], behind[:, 1]).mean() > 0.97
    # roofs face up; the building stands on the ground and is roughly as tall as asked
    roof = np.isin(A[:, 0], [1, 2])
    assert roof.any() and np.all(n[roof][:, 2] > 0)
    assert P[..., 2].min() <= 99.6 and P[..., 2].max() < 100.0 + 12.0


def test_garage_faces_street_and_no_ground_floor_windows_on_it() -> None:
    b = model.Builder(_spec(ELL), 0, model.DEFAULT_PARAMS)
    soup = b.build()
    assert b.garages, "a two-storey house with a street-facing wall gets a garage door"
    p0, p1, n = b.garages[0]
    assert n[1] < -0.9  # street is to the south (+z) = Blender -Y
    P, _, A = soup.arrays()
    glass = P[A[:, 0] == 3].reshape(-1, 3)
    m = (p0 + p1) / 2
    near = (np.abs((glass[:, :2] - m) @ np.array([-n[1], n[0]])) < np.hypot(*(p1 - p0)) / 2) & (np.abs((glass[:, :2] - m) @ n) < 0.2)
    assert not np.any(near & (glass[:, 2] < b.floor + 2.9)), "ground-floor glass on the garage wall"
    gd = A[:, 0] == 5
    assert gd.any() and P[gd][..., 2].min() >= b.floor - 1e-6


def test_two_level_massing_from_height_grid() -> None:
    # nDSM grid: the north part (two-storey block) is 8 m tall, the garage wing 4 m
    ring = ELL
    xs = np.arange(-7.5, 8.0, 1.0)
    zs = np.arange(-4.5, 11.0, 1.0)  # scene z
    h = np.full((len(zs), len(xs)), -1)
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            inside = (-8 < x < 8 and -5 < z < 5) or (1 < x < 8 and 5 <= z < 11)
            if inside:
                d = min(x + 8, 8 - x, z + 5, 5 - z) if z < 5 else min(x - 1, 8 - x, 11 - z)
                h[i, j] = int(10 * ((5.9 + 0.35 * d) if z < 5 else (3.0 + 0.35 * d)))
    # grid rows go north (+y): flip
    hg = {"x0": float(xs[0]), "y0": float(-zs[-1]), "step": 1.0, "nx": len(xs), "ny": len(zs), "h": h[::-1].ravel().tolist()}
    r = model.build_building(_spec(ring, roof_source="lidar", eave_h=3.0, ridge_h=8.8, hgrid=hg), 0)
    assert r.two_level and r.eave_h > 5.0


def test_lod1_is_much_lighter() -> None:
    s = _spec(ELL)
    n0 = model.build_building(s, 0).soup.ntris
    n1 = model.build_building(s, 1).soup.ntris
    assert n1 < n0 / 3


# ---------------------------------------------------------------------------
# built tiles
# ---------------------------------------------------------------------------


def test_manifest_and_tiles_parse() -> None:
    man = _manifest()
    assert man["format"] == "redraw-buildings-hd" and man["tiles"]
    for t in man["tiles"]:
        for lod in ("lod0", "lod1"):
            if lod not in t:
                continue
            f = OUT.parent / t[lod]["file"]
            assert f.exists(), f
            g = _glb_json(f)
            prims = [p for m in g["meshes"] for p in m["primitives"]]
            assert len(prims) == 1, "one primitive per tile"
            attrs = set(prims[0]["attributes"])
            assert REQUIRED_ATTRS <= attrs, (f.name, REQUIRED_ATTRS - attrs)
            if man["draco"]:
                assert "KHR_draco_mesh_compression" in prims[0].get("extensions", {})
            idx = g["accessors"][prims[0]["indices"]]
            assert idx["count"] // 3 == t[lod]["triangles"] or abs(idx["count"] // 3 - t[lod]["triangles"]) <= 0.01 * t[lod]["triangles"]
            for nm in ("_BUILDING_ID", "_MAT", "_VARIANT"):
                acc = g["accessors"][prims[0]["attributes"][nm]]
                assert acc["type"] == "SCALAR" and acc["componentType"] == 5126  # FLOAT
            assert all(n.get("matrix") is None and n.get("translation") is None for n in g["nodes"])


def test_budgets() -> None:
    man = _manifest()
    assert man["triangles"]["lod0"] < LOD0_BUDGET, man["triangles"]
    assert man["triangles"]["lod1"] < LOD1_BUDGET, man["triangles"]


def test_every_footprint_is_built() -> None:
    man = _manifest()
    idx_p = SPECS / "index.json"
    if not idx_p.exists():
        pytest.skip(f"{idx_p} missing; {HINT}")
    idx = json.loads(idx_p.read_text())
    table = json.loads((OUT / "buildings_hd.json").read_text())["buildings"]
    ids = {b["id"] for b in table}
    want = set()
    for tid in idx["tiles"]:
        want |= {int(b["id"]) for b in json.loads((SPECS / f"{tid}.json").read_text())["buildings"]}
    assert want == ids, f"missing {sorted(want - ids)[:10]}, extra {sorted(ids - want)[:10]}"
    assert not (set(idx["hero_ids"]) & ids), "hero buildings must be skipped"
    built = 0
    for t in man["tiles"]:
        st = json.loads((STATS / f"{t['id']}.json").read_text())
        assert all(b["tris"] > 0 for b in st["buildings"])
        built += len(st["buildings"])
    assert built == len(ids)
    # the pipeline's processed footprints (same bbox) are all present, minus hero campuses
    proc = REPO / "data" / "processed"
    if (proc / "buildings.geojson").exists() and (proc / "region_meta.json").exists():
        from pipeline.config import region

        meta = json.loads((proc / "region_meta.json").read_text())
        if all(abs(float(meta["bbox"][k]) - float(region()["bbox"][k])) < 1e-9 for k in ("south", "north", "west", "east")):
            gj = json.loads((proc / "buildings.geojson").read_text())
            pids = {int(f["properties"]["id"]) for f in gj["features"]}
            hero_like = {int(f["properties"]["id"]) for f in gj["features"] if (f["properties"].get("type") == "hero")}
            missing = pids - ids - set(idx["hero_ids"]) - hero_like
            assert len(missing) <= 0.002 * len(pids), f"{len(missing)} processed footprints not built, e.g. {sorted(missing)[:10]}"


def test_table_fields() -> None:
    _manifest()
    table = json.loads((OUT / "buildings_hd.json").read_text())
    for b in table["buildings"][:2000]:
        assert b["source"] in ("lidar", "tag", "heuristic")
        assert b["roof_type"] in ("flat", "hip", "gable", "shed", "complex")
        assert 1.5 < b["eave_h"] < 60 and b["ridge_h"] >= b["eave_h"] - 1e-6
        for g in b["garages"]:
            assert len(g) == 6 and abs(math.hypot(g[2], g[3]) - 1) < 1e-2
