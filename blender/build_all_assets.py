"""Build every Blender-made asset for Redraw (headless bpy, no Blender UI needed).

    python3.11 -m venv .venv-blender && .venv-blender/bin/pip install bpy==5.0.1
    .venv-blender/bin/python blender/build_all_assets.py                 # everything
    .venv-blender/bin/python blender/build_all_assets.py --only vehicles trees
    .venv-blender/bin/python blender/build_all_assets.py --no-previews

Outputs (regenerated, not committed except the previews):
    client/public/assets/props/vehicles/*.glb, vegetation/*.glb, street/*.glb
    client/public/assets/props/props_manifest.json
    pipeline/hero_overrides/*.glb + hero_overrides.json
    blender/previews/*.png          (committed, < 400 KB each)

Hero campuses need blender/build/hero_sites.json, written by
`.venv/bin/python blender/extract_hero_sites.py` (main venv, geopandas); this script
runs it automatically when the json is missing and the main venv exists.
Instance placements (trees, lamps) are a separate pipeline step: pipeline/build_props.py.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from rdlib import bl, foliage, materials, props, vehicles  # noqa: E402

REPO = HERE.parent
PROPS_DIR = REPO / "client" / "public" / "assets" / "props"
HERO_DIR = REPO / "pipeline" / "hero_overrides"
PREVIEW_DIR = HERE / "previews"
BUILD = HERE / "build"
SITES_JSON = BUILD / "hero_sites.json"

STAGES = ("vehicles", "trees", "street", "heroes", "manifest", "previews")


def log(msg: str) -> None:
    print(f"[assets] {msg}", flush=True)


def _extent(objs: list[bpy.types.Object]) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    for o in objs:
        m = np.array(o.matrix_world)
        v = np.array([vv.co[:] for vv in o.data.vertices])
        pts.append((np.hstack([v, np.ones((len(v), 1))]) @ m.T)[:, :3])
    a = np.vstack(pts)
    return a.min(axis=0), a.max(axis=0)


def _footprint_radius(objs: list[bpy.types.Object]) -> float:
    r = 0.0
    for o in objs:
        v = np.array([vv.co[:] for vv in o.data.vertices])
        r = max(r, float(np.max(np.hypot(v[:, 0], v[:, 1]))))
    return r


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def build_vehicles(entries: list[dict]) -> None:
    for vid, (fn, info) in vehicles.VEHICLES.items():
        bl.reset_scene()
        part = fn()
        obj = bl.part_to_object(part, vid, smooth_angle=40, vehicle_attrs=True)
        out = PROPS_DIR / "vehicles" / f"{vid}.glb"
        bl.export_glb([obj], out, attributes=True)
        tris = bl.tri_count([obj])
        lo, hi = _extent([obj])
        tint = any(materials.MATERIALS[m.name].tint for m in obj.data.materials if m.name in materials.MATERIALS)
        entries.append({
            "id": vid,
            "kind": "vehicle",
            "vehicle_class": "bus" if vid == "school_bus" else ("shuttle" if vid == "shuttle_van" else "car"),
            "file": f"vehicles/{vid}.glb",
            "triangles": tris,
            "footprint_radius_m": round(_footprint_radius([obj]), 3),
            "length_m": round(float(hi[1] - lo[1]), 3),
            "width_m": round(float(hi[0] - lo[0]), 3),
            "height_m": round(float(hi[2] - lo[2]), 3),
            "forward_axis": "-z",
            "up_axis": "+y",
            "origin": "ground, footprint center",
            "tintable_region": "material:paint" if tint else None,
            "suggested_scale_range": [0.97, 1.03],
            "materials": [m.name for m in obj.data.materials],
            "vertex_attributes": {"_LIGHT": "0 body, 1 headlight/DRL, 2 taillight, 3 glass, 4 dark trim/tire",
                                  "_TINT": "1 where the client multiplies the instance paint color"},
            "fleet_share": info["share"],
            "paint_colors": materials.PAINT_COLORS if tint and info["share"] > 0 else None,
            "notes": info["notes"],
        })
        log(f"{vid}: {tris} tris, {hi[1] - lo[1]:.2f} x {hi[0] - lo[0]:.2f} x {hi[2] - lo[2]:.2f} m -> {out.name}")


def build_trees(entries: list[dict]) -> None:
    tex_dir = BUILD / "textures"
    for tid, (fn, info) in foliage.SPECIES.items():
        bl.reset_scene()
        part, normals, tex = fn(tex_dir / f"{tid}.png")
        bl.TEXTURES["foliage_atlas"] = tex
        obj = bl.part_to_object(part, tid)
        bl.set_custom_normals(obj, normals)
        out = PROPS_DIR / "vegetation" / f"{tid}.glb"
        bl.export_glb([obj], out)
        tris = bl.tri_count([obj])
        lo, hi = _extent([obj])
        scale = [0.8, 1.2] if info["kind"] == "tree" else [0.7, 1.3]
        if tid.startswith("tree_palm"):
            scale = [0.75, 1.25]
        entries.append({
            "id": tid,
            "kind": info["kind"],
            "file": f"vegetation/{tid}.glb",
            "triangles": tris,
            "footprint_radius_m": round(_footprint_radius([obj]), 3),
            "length_m": round(float(max(hi[0] - lo[0], hi[1] - lo[1])), 3),
            "height_m": round(float(hi[2]), 3),
            "forward_axis": "-z",
            "up_axis": "+y",
            "origin": "ground, trunk base",
            "tintable_region": None,
            "suggested_scale_range": scale,
            "materials": ["foliage"],
            "alpha_mode": "MASK",
            "notes": info["notes"],
        })
        log(f"{tid}: {tris} tris, h {hi[2]:.1f} m, r {_footprint_radius([obj]):.1f} m -> {out.name}")


def build_street(entries: list[dict]) -> None:
    for sid, (fn, info) in props.STREET.items():
        bl.reset_scene()
        obj = bl.part_to_object(fn(), sid, smooth_angle=45)
        out = PROPS_DIR / "street" / f"{sid}.glb"
        bl.export_glb([obj], out)
        tris = bl.tri_count([obj])
        lo, hi = _extent([obj])
        entries.append({
            "id": sid,
            "kind": info["kind"],
            "file": f"street/{sid}.glb",
            "triangles": tris,
            "footprint_radius_m": info["radius_m"],
            "length_m": round(float(hi[1] - lo[1]), 3),
            "height_m": round(float(hi[2]), 3),
            "reach_m": info["reach_m"],
            "forward_axis": "-z",
            "up_axis": "+y",
            "origin": "ground, pole base; mast arm reaches toward -z (the road)",
            "tintable_region": None,
            "suggested_scale_range": [0.95, 1.05],
            "materials": [m.name for m in obj.data.materials],
            "notes": info["notes"],
        })
        log(f"{sid}: {tris} tris -> {out.name}")


def build_heroes(previews: bool, samples: int) -> None:
    if not SITES_JSON.exists():
        main_py = REPO / ".venv" / "bin" / "python"
        if not main_py.exists():
            raise SystemExit(f"{SITES_JSON} missing and no main venv to run blender/extract_hero_sites.py")
        log("extracting hero footprints with the main venv ...")
        subprocess.run([str(main_py), str(HERE / "extract_hero_sites.py")], check=True)
    from rdlib import heroes

    sites = json.loads(SITES_JSON.read_text())["heroes"]
    overrides = []
    for site in sites:
        bl.reset_scene()
        objs, stats = heroes.build_campus(site)
        out = HERO_DIR / site["glb"]
        bl.export_glb(objs, out)
        tris = bl.tri_count(objs)
        trees = stats.pop("_trees")
        log(f"{site['id']}: {tris} tris, {stats} -> {out}")
        overrides.append({
            "id": site["id"],
            "school_id": site["school_id"],
            "name": site["name"],
            "type": site["type"],
            "lat": site["lat"],
            "lon": site["lon"],
            "rotation_deg": 0.0,
            "footprint_radius_m": site["footprint_radius_m"],
            "glb": site["glb"],
            "replaces": "buildings within footprint_radius_m of lat/lon",
            "triangles": tris,
            "buildings_modeled": site["n_buildings"],
            "trees": f"{site['id']}_trees.json",
            "source": "OSM building footprints + Overture land_use/segments (blender/extract_hero_sites.py)",
            "verified": False,
        })
        (HERO_DIR / f"{site['id']}_trees.json").write_text(json.dumps(trees, separators=(",", ":")))
        if previews:
            heroes.render_preview(site, objs, PREVIEW_DIR / f"hero_{site['id']}.png", samples=samples)
    (HERO_DIR / "hero_overrides.json").write_text(json.dumps(overrides, indent=2) + "\n")
    log(f"wrote {HERO_DIR / 'hero_overrides.json'}")


def write_manifest(entries: list[dict]) -> None:
    path = PROPS_DIR / "props_manifest.json"
    old = []
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except json.JSONDecodeError:
            old = []
    ids = {e["id"] for e in entries}
    merged = [e for e in old if isinstance(e, dict) and e.get("id") not in ids] + entries
    order = {"vehicle": 0, "tree": 1, "shrub": 2, "lamp": 3}
    merged.sort(key=lambda e: (order.get(e["kind"], 9), e["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=1) + "\n")
    log(f"wrote {path} ({len(merged)} props)")


# ---------------------------------------------------------------------------
# previews
# ---------------------------------------------------------------------------


def _paint(obj: bpy.types.Object, hex_: str) -> None:
    for i, slot in enumerate(obj.material_slots):
        if slot.material and slot.material.name.startswith("paint") and slot.material.name != "paint_bus" \
                and slot.material.name != "paint_bus_roof":
            m = slot.material.copy()
            m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*materials.hex_to_linear(hex_), 1.0)
            obj.material_slots[i].material = m


def preview_vehicles(path: Path, samples: int) -> None:
    bl.reset_scene()
    order = ["school_bus", "shuttle_van", "car_pickup", "car_minivan", "car_crossover_ev", "car_suv", "car_sedan"]
    colors = {"car_sedan": "#1F3B66", "car_suv": "#A9ADB1", "car_crossover_ev": "#E9EAEA", "car_minivan": "#16171A",
              "car_pickup": "#8E1B1E", "shuttle_van": "#E9EAEA"}
    x = 0.0
    for i, vid in enumerate(order):
        fn, info = vehicles.VEHICLES[vid]
        o = bl.part_to_object(fn(), vid, smooth_angle=40)
        if vid in colors:
            _paint(o, colors[vid])
        w = info["width_m"]
        x += w / 2 + (1.3 if i else 0.0) + (0.6 if vid in ("shuttle_van", "car_pickup") else 0.0)
        o.location = (x, 0.0, 0.0)
        o.rotation_euler = (0, 0, math.radians(-25))
        x += w / 2
    bl.setup_render(1280, 560, samples=samples)
    bl.setup_world(sun_elev_deg=38, sun_azimuth_deg=210)
    bl.ground_plane(120, "#5E6064", roughness=0.95)
    # parking stall stripes for scale
    for k in range(9):
        me = bpy.data.meshes.new("stripe")
        xs = -1.4 + k * 3.05
        me.from_pydata([(xs, -3.5, 0.005), (xs + 0.1, -3.5, 0.005), (xs + 0.1, 2.5, 0.005), (xs, 2.5, 0.005)], [], [[0, 1, 2, 3]])
        me.materials.append(bl.simple_material("stripe", "#E8E6DF", 0.7))
        ob = bpy.data.objects.new("stripe", me)
        bpy.context.scene.collection.objects.link(ob)
    bl.add_camera((x / 2 - 3.0, 17.5, 4.2), (x / 2 + 0.2, 0, 0.9), lens=34)
    bl.render(path)
    log(f"preview {path} ({path.stat().st_size // 1024} KB)")


def preview_trees(path: Path, samples: int) -> None:
    bl.reset_scene()
    order = ["grass_ornamental", "shrub", "tree_street", "tree_jacaranda", "tree_oak", "tree_palm_queen",
             "tree_eucalyptus", "tree_palm_fan", "street_lamp"]
    x = 0.0
    tex_dir = BUILD / "textures"
    for tid in order:
        if tid == "street_lamp":
            o = bl.part_to_object(props.street_lamp(), tid, smooth_angle=45)
            r = 1.2
        else:
            fn, info = foliage.SPECIES[tid]
            part, normals, tex = fn(tex_dir / f"{tid}.png")
            bl.TEXTURES["foliage_atlas"] = tex
            if "foliage" in bpy.data.materials:
                bpy.data.materials["foliage"].name = "foliage_prev"
            o = bl.part_to_object(part, tid)
            bl.set_custom_normals(o, normals)
            r = info["radius_m"] * (0.75 if tid != "tree_palm_fan" else 0.9)
        x += r
        o.location = (x, 0, 0)
        if tid == "street_lamp":
            o.rotation_euler = (0, 0, math.radians(-90))
        x += r + 0.6
    bl.setup_render(1280, 600, samples=samples)
    bl.setup_world(sun_elev_deg=42, sun_azimuth_deg=200)
    bl.ground_plane(400, "#8E9168", roughness=1.0)
    bl.add_camera((x / 2, -42, 6.5), (x / 2, 0, 7.6), lens=30)
    bl.render(path)
    log(f"preview {path} ({path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=STAGES, help="run only these stages")
    ap.add_argument("--no-previews", action="store_true", help="skip preview renders")
    ap.add_argument("--samples", type=int, default=48, help="Cycles samples for previews")
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    stages = set(a.only or STAGES)
    if a.no_previews:
        stages.discard("previews")
    t0 = time.time()
    entries: list[dict] = []
    if "vehicles" in stages:
        build_vehicles(entries)
    if "trees" in stages:
        build_trees(entries)
    if "street" in stages:
        build_street(entries)
    if entries or "manifest" in stages:
        write_manifest(entries)
    if "heroes" in stages:
        build_heroes("previews" in stages, a.samples)
    if "previews" in stages:
        preview_vehicles(PREVIEW_DIR / "vehicles_lineup.png", a.samples)
        preview_trees(PREVIEW_DIR / "vegetation_lineup.png", a.samples)
    log(f"done in {time.time() - t0:.0f} s ({datetime.now(timezone.utc).isoformat(timespec='seconds')})")


if __name__ == "__main__":
    main()
