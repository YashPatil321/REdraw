"""HD buildings: per-tile specs -> Blender meshes -> glTF tiles (LOD0 + LOD1) + manifests.

    .venv/bin/python pipeline/build_buildings_blender.py           # specs (main venv: geopandas, DEM, lidar)
    .venv-blender/bin/python blender/buildings/build.py            # all tiles, 4 parallel bpy workers, resumable
    .venv-blender/bin/python blender/buildings/build.py --tiles r3_c3 r3_c4 --force
    .venv-blender/bin/python blender/buildings/build.py --manifest-only

Every tile is built in its own `bpy` process (`--worker`), so a crash or a container restart
loses at most the tiles in flight. A tile is skipped when its glbs exist and its stamp
(hash of the spec file + generator code + params) matches. Outputs:

    client/public/assets/buildings_hd/r{r}_c{c}_lod0.glb, _lod1.glb   (Draco when gltf-transform is available)
    client/public/assets/buildings_hd/manifest_buildings.json          tiles, bounds, files, triangles
    client/public/assets/buildings_hd/buildings_hd.json                per-building roof model + source
    blender/build/buildings_hd/stats/r{r}_c{c}.json                    per-tile stamp, counts, timings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

SPEC_DIR = REPO / "blender" / "build" / "buildings_hd" / "specs"
STATS_DIR = REPO / "blender" / "build" / "buildings_hd" / "stats"
OUT_DIR = REPO / "client" / "public" / "assets" / "buildings_hd"
CODE_FILES = [HERE / "geom.py", HERE / "model.py", HERE / "build.py"]
FORMAT_VERSION = 1


def log(msg: str) -> None:
    print(f"[buildings_hd] {msg}", flush=True)


def code_hash() -> str:
    h = hashlib.sha1()
    for f in CODE_FILES:
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def tile_stamp(tile: str, params: dict[str, Any]) -> str:
    h = hashlib.sha1()
    h.update((SPEC_DIR / f"{tile}.json").read_bytes())
    h.update(code_hash().encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()[:20]


def load_index() -> dict[str, Any]:
    p = SPEC_DIR / "index.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: run `.venv/bin/python pipeline/build_buildings_blender.py` first")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# soup -> welded arrays (numpy only)
# ---------------------------------------------------------------------------


def weld(P: np.ndarray, UV: np.ndarray, A: np.ndarray) -> dict[str, np.ndarray]:
    """Triangle soup -> indexed vertices (merged where position, uv, normal and attributes agree)."""
    t = len(P)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    pos = P.reshape(-1, 3)
    uv = UV.reshape(-1, 2)
    nn = np.repeat(n, 3, axis=0)
    at = np.repeat(A, 3, axis=0)
    key = np.column_stack([np.round(pos * 1e3), np.round(uv * 1e3), np.round(nn * 1e2), np.round(at[:, :2]),
                           np.round(at[:, 2:5] * 255), at[:, 5]]).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    return {"pos": pos[first], "uv": uv[first], "normal": nn[first], "attr": at[first], "index": inv.reshape(t, 3)}


# ---------------------------------------------------------------------------
# worker (bpy)
# ---------------------------------------------------------------------------


def make_mesh(name: str, W: dict[str, np.ndarray]) -> Any:
    import bpy

    pos, idx = W["pos"], W["index"]
    nv, nt = len(pos), len(idx)
    me = bpy.data.meshes.new(name)
    me.vertices.add(nv)
    me.vertices.foreach_set("co", pos.astype(np.float32).ravel())
    me.loops.add(nt * 3)
    me.loops.foreach_set("vertex_index", idx.astype(np.int32).ravel())
    me.polygons.add(nt)
    me.polygons.foreach_set("loop_start", np.arange(0, nt * 3, 3, dtype=np.int32))
    if hasattr(me.polygons[0] if nt else None, "loop_total"):
        try:
            me.polygons.foreach_set("loop_total", np.full(nt, 3, dtype=np.int32))
        except (AttributeError, TypeError, RuntimeError):
            pass
    me.update(calc_edges=True)
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", W["uv"][idx.ravel()].astype(np.float32).ravel())
    A = W["attr"]
    for nm, col in (("_BUILDING_ID", 5), ("_MAT", 0), ("_VARIANT", 1)):
        a = me.attributes.new(nm, "FLOAT", "POINT")
        a.data.foreach_set("value", A[:, col].astype(np.float32))
    ca = me.color_attributes.new("Col", "BYTE_COLOR", "POINT")
    rgba = np.ones((nv, 4), np.float32)
    rgba[:, :3] = A[:, 2:5]
    ca.data.foreach_set("color_srgb", rgba.ravel())  # sRGB bytes; glTF COLOR_0 is written linear
    me.color_attributes.active_color = ca
    me.polygons.foreach_set("use_smooth", np.zeros(nt, dtype=bool))
    me.validate(clean_customdata=False)
    return me


def tile_material() -> Any:
    import bpy

    m = bpy.data.materials.get("buildings_hd")
    if m:
        return m
    m = bpy.data.materials.new("buildings_hd")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes.get("Principled BSDF")
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "Col"
    nt.links.new(vc.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.85
    m.use_backface_culling = True  # every face is outward oriented: glTF doubleSided = false
    return m


def export_glb(obj: Any, path: Path) -> None:
    import bpy

    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    tmp = path.with_suffix(".tmp.glb")
    bpy.ops.export_scene.gltf(
        filepath=str(tmp), export_format="GLB", use_selection=True, export_apply=False, export_yup=True,
        export_texcoords=True, export_normals=True, export_materials="EXPORT", export_vertex_color="ACTIVE",
        export_attributes=True, export_cameras=False, export_lights=False, export_extras=False,
        export_animations=False, export_draco_mesh_compression_enable=False, export_image_format="NONE",
    )
    tmp.replace(path)


def build_tile_arrays(tile: str, lod: int, params: dict[str, Any]) -> tuple[Any, list[dict[str, Any]], dict[str, float]]:
    from blender.buildings import model

    specs = json.loads((SPEC_DIR / f"{tile}.json").read_text())["buildings"]
    soup = model.Soup()
    meta = []
    timing = {"model_s": 0.0}
    t0 = time.time()
    for sp in specs:
        fallback = False
        try:
            r = model.build_building(sp, lod, params)
            s = r.soup
            if s.ntris == 0:
                raise ValueError("empty")
            info = {"eave_h": r.eave_h, "ridge_h": r.ridge_h, "roof_type": r.roof_type, "two_level": r.two_level}
        except Exception as e:  # noqa: BLE001 - a bad footprint must not kill the tile; logged + counted
            log(f"{tile} building {sp['id']}: {type(e).__name__}: {e} -> box fallback")
            s = model.box_fallback(sp, lod)
            info = {"eave_h": sp.get("eave_h"), "ridge_h": sp.get("eave_h"), "roof_type": "flat", "two_level": False}
            fallback = True
        soup.extend(s)
        meta.append({"id": int(sp["id"]), "type": sp["type"], "source": sp.get("roof_source"), "origin": sp.get("origin", "osm"),
                     "levels": sp.get("levels"), "tris": s.ntris, "fallback": fallback, **info})
    timing["model_s"] = time.time() - t0
    return soup, meta, timing


def worker(tiles: list[str], params: dict[str, Any], stamps: dict[str, str]) -> None:
    import bpy

    for tile in tiles:
        t0 = time.time()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        stats: dict[str, Any] = {"tile": tile, "stamp": stamps[tile], "lods": {}}
        meta0: list[dict[str, Any]] = []
        for lod in (0, 1):
            soup, meta, timing = build_tile_arrays(tile, lod, params)
            if lod == 0:
                meta0 = meta
            P, UV, A = soup.arrays()
            path = OUT_DIR / f"{tile}_lod{lod}.glb"
            if len(P) == 0:
                path.unlink(missing_ok=True)
                stats["lods"][str(lod)] = {"file": None, "triangles": 0}
                continue
            t1 = time.time()
            W = weld(P, UV, A)
            me = make_mesh(f"{tile}_lod{lod}", W)
            me.materials.append(tile_material())
            ob = bpy.data.objects.new(f"buildings_{tile}_lod{lod}", me)
            bpy.context.scene.collection.objects.link(ob)
            export_glb(ob, path)
            bpy.data.objects.remove(ob)
            bpy.data.meshes.remove(me)
            b = {"min_x": float(P[..., 0].min()), "max_x": float(P[..., 0].max()), "min_z": float(-P[..., 1].max()),
                 "max_z": float(-P[..., 1].min()), "min_y": float(P[..., 2].min()), "max_y": float(P[..., 2].max())}
            stats["lods"][str(lod)] = {"file": path.name, "triangles": int(len(P)), "vertices": int(len(W["pos"])),
                                       "bytes_raw": path.stat().st_size, "bounds": b, "model_s": round(timing["model_s"], 2),
                                       "export_s": round(time.time() - t1, 2)}
        stats["buildings"] = meta0
        stats["seconds"] = round(time.time() - t0, 1)
        stats["draco"] = False
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATS_DIR / f"{tile}.json.tmp"
        tmp.write_text(json.dumps(stats))
        tmp.replace(STATS_DIR / f"{tile}.json")
        log(f"{tile}: {len(meta0)} buildings, lod0 {stats['lods']['0']['triangles']:,} tris, "
            f"lod1 {stats['lods'].get('1', {}).get('triangles', 0):,} tris ({stats['seconds']}s)")


# ---------------------------------------------------------------------------
# draco + manifests (no bpy)
# ---------------------------------------------------------------------------


def gltf_transform() -> list[str] | None:
    npx = shutil.which("npx")
    if not npx:
        return None
    try:
        r = subprocess.run([npx, "-y", "@gltf-transform/cli", "--version"], capture_output=True, text=True, timeout=180)
    except (subprocess.SubprocessError, OSError):
        return None
    return [npx, "-y", "@gltf-transform/cli"] if r.returncode == 0 else None


def draco_tile(tile: str, cli: list[str]) -> bool:
    sp = STATS_DIR / f"{tile}.json"
    st = json.loads(sp.read_text())
    if st.get("draco"):
        return True
    ok = True
    for info in st["lods"].values():
        if not info.get("file"):
            continue
        src = OUT_DIR / info["file"]
        tmp = src.with_suffix(".draco.glb")
        r = subprocess.run([*cli, "draco", str(src), str(tmp), "--method", "edgebreaker", "--quantize-position", "20",
                            "--quantize-texcoord", "14", "--quantize-color", "8", "--quantize-generic", "16"],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not tmp.exists():
            log(f"draco failed for {src.name}: {r.stderr.strip()[-300:]}")
            tmp.unlink(missing_ok=True)
            ok = False
            continue
        tmp.replace(src)
        info["bytes"] = src.stat().st_size
    if ok:
        st["draco"] = True
        sp.write_text(json.dumps(st))
    return ok


def write_manifests(index: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    tiles = []
    rows: list[dict[str, Any]] = []
    tri = {"lod0": 0, "lod1": 0}
    by_type: dict[str, int] = {}
    by_src: dict[str, int] = {}
    by_roof: dict[str, int] = {}
    fallbacks = 0
    sizes = {"lod0": 0, "lod1": 0}
    draco_all = True
    secs = 0.0
    g = index["grid"]
    ext = g["extent"]
    w = (ext["max_x"] - ext["min_x"]) / g["cols"]
    d = (ext["max_z"] - ext["min_z"]) / g["rows"]
    for tid in sorted(index["tiles"]):
        sp = STATS_DIR / f"{tid}.json"
        if not sp.exists():
            continue
        st = json.loads(sp.read_text())
        r, c = (int(x[1:]) for x in tid.split("_"))
        ent: dict[str, Any] = {"id": tid, "row": r, "col": c, "buildings": len(st["buildings"]),
                               "cell": {"min_x": ext["min_x"] + c * w, "max_x": ext["min_x"] + (c + 1) * w,
                                        "min_z": ext["min_z"] + r * d, "max_z": ext["min_z"] + (r + 1) * d}}
        bounds = None
        for lod in ("0", "1"):
            info = st["lods"].get(lod) or {}
            if not info.get("file"):
                continue
            f = OUT_DIR / info["file"]
            ent[f"lod{lod}"] = {"file": f"buildings_hd/{info['file']}", "triangles": info["triangles"], "vertices": info.get("vertices"),
                                "bytes": f.stat().st_size if f.exists() else None}
            tri[f"lod{lod}"] += info["triangles"]
            sizes[f"lod{lod}"] += f.stat().st_size if f.exists() else 0
            if lod == "0":
                bounds = info["bounds"]
        ent["bounds"] = bounds
        draco_all &= bool(st.get("draco"))
        secs += st.get("seconds", 0)
        tiles.append(ent)
        for b in st["buildings"]:
            rows.append({"id": b["id"], "tile": tid, "type": b["type"], "eave_h": b["eave_h"], "ridge_h": b["ridge_h"],
                         "roof_type": b["roof_type"], "source": b["source"], "levels": b["levels"], "two_level": b["two_level"],
                         "origin": b["origin"]})
            by_type[b["type"]] = by_type.get(b["type"], 0) + 1
            by_src[str(b["source"])] = by_src.get(str(b["source"]), 0) + 1
            by_roof[str(b["roof_type"])] = by_roof.get(str(b["roof_type"]), 0) + 1
            fallbacks += bool(b["fallback"])
    man = {
        "format": "redraw-buildings-hd",
        "version": FORMAT_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": ".venv-blender/bin/python blender/buildings/build.py (specs: pipeline/build_buildings_blender.py)",
        "synthetic": False,
        "base_path": "assets/",
        "draco": bool(draco_all and tiles),
        "coordinates": "glTF meters, x east, y up, z south, relative to the scene origin (docs/coordinates.md); identity node transforms",
        "origin": index.get("origin"),
        "bbox": index.get("bbox"),
        "grid": g,
        "lods": {
            "lod0": "full detail: real footprint, lidar/tag roof (hip / gable / shed / flat / complex, two roof levels), eaves with fascia + soffit, windows (trim, glass, sill), recessed garage door toward the street, entry door + porch roof, storefronts, parapets, rooftop HVAC, PV arrays",
            "lod1": "coarse footprint, walls to the roof line + roof planes; no overhangs or openings",
        },
        "suggested_lod1_distance_m": 450,
        "materials_manifest": "materials/materials_manifest.json",
        "vertex_attributes": {
            "_BUILDING_ID": "FLOAT scalar (integral) = buildings_hd.json id = buildings.geojson id (lidar-only buildings use ids above the pipeline max)",
            "_MAT": "FLOAT scalar (integral): 0 stucco wall, 1 tile roof, 2 flat roof, 3 glass, 4 trim / door / soffit, 5 garage door (materials_manifest.json materials)",
            "_VARIANT": "FLOAT scalar (integral): index into materials_manifest.json materials[_MAT].variants",
            "COLOR_0": "linear RGB tint: wall color (walls, garage surround), trim / door color (trim), real fallback colors for roofs / glass",
            "TEXCOORD_0": "materials_manifest uv_conventions: facade u = m along wall / 3, v = m above floor / 3; pitched roof u = m along eave / 4, v = m up slope / 4; flat roof u = x / 4, v = -z / 4; garage door uvs map onto the garage cell's door rect",
            "NORMAL": "flat (per face)",
        },
        "tiles": tiles,
        "triangles": tri,
        "bytes": sizes,
        "counts": {"buildings": len(rows), "by_type": by_type, "by_roof_source": by_src, "by_roof_type": by_roof, "box_fallbacks": fallbacks},
        "build_seconds_cpu": round(secs, 1),
        "notes": index.get("notes"),
        "heroes_skipped": index.get("heroes"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest_buildings.json").write_text(json.dumps(man, indent=1))
    rows.sort(key=lambda x: x["id"])
    bj = {"format": "redraw-buildings-hd-table", "version": FORMAT_VERSION, "fields": {
        "eave_h": "m above the finished floor (main / highest roof level)", "ridge_h": "m above the finished floor",
        "roof_type": "flat | hip | gable | shed | complex", "source": "lidar | tag | heuristic",
        "two_level": "one-storey wing + two-storey block", "origin": "osm | lidar_missing"}, "buildings": rows}
    (OUT_DIR / "buildings_hd.json").write_text(json.dumps(bj, separators=(",", ":")))
    return man


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", nargs="*", help="tile ids (default: all)")
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-draco", action="store_true")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--worker", nargs="*", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    index = load_index()
    from blender.buildings import model

    params = {**model.DEFAULT_PARAMS, **(index.get("params") or {})}
    params = json.loads(json.dumps(params))  # tuples -> lists (stable hashing)
    if a.worker is not None:
        stamps = {t: tile_stamp(t, params) for t in a.worker}
        worker(a.worker, params, stamps)
        return 0
    t0 = time.time()
    tiles = a.tiles or sorted(index["tiles"])
    todo = []
    if not a.manifest_only:
        for t in tiles:
            sp = STATS_DIR / f"{t}.json"
            if not a.force and sp.exists():
                st = json.loads(sp.read_text())
                files_ok = all((OUT_DIR / i["file"]).exists() for i in st["lods"].values() if i.get("file"))
                if st.get("stamp") == tile_stamp(t, params) and files_ok:
                    continue
            todo.append(t)
        # biggest tiles first so the pool drains evenly
        todo.sort(key=lambda t: -index["tiles"].get(t, 0))
        log(f"{len(todo)} of {len(tiles)} tiles to build with {a.workers} workers")
        procs: list[tuple[subprocess.Popen[bytes], str]] = []
        queue = list(todo)
        failed = []
        while queue or procs:
            while queue and len(procs) < a.workers:
                t = queue.pop(0)
                p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", t], cwd=str(REPO))
                procs.append((p, t))
            time.sleep(0.5)
            for p, t in list(procs):
                if p.poll() is not None:
                    procs.remove((p, t))
                    if p.returncode != 0:
                        failed.append(t)
                        log(f"tile {t} FAILED (exit {p.returncode})")
        if failed:
            log(f"failed tiles: {failed}")
    if not a.no_draco and not a.manifest_only:
        cli = gltf_transform()
        if cli is None:
            log("gltf-transform (npx @gltf-transform/cli) unavailable: glbs stay uncompressed, manifest draco=false")
        else:
            from concurrent.futures import ThreadPoolExecutor

            need = [t for t in tiles if (STATS_DIR / f"{t}.json").exists() and not json.loads((STATS_DIR / f"{t}.json").read_text()).get("draco")]
            log(f"draco: compressing {len(need)} tiles")
            with ThreadPoolExecutor(a.workers) as ex:
                list(ex.map(lambda t: draco_tile(t, cli), need))
    man = write_manifests(index, params)
    log(f"manifest: {len(man['tiles'])} tiles, {man['counts']['buildings']:,} buildings, lod0 {man['triangles']['lod0']:,} tris, "
        f"lod1 {man['triangles']['lod1']:,} tris, {man['bytes']['lod0'] / 1e6:.1f} + {man['bytes']['lod1'] / 1e6:.1f} MB, "
        f"draco={man['draco']} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
