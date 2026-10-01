"""Cycles beauty previews of the HD buildings in their real setting (headless bpy).

    .venv/bin/python pipeline/build_buildings_blender.py --preview-only     # terrain / NAIP / roads around the views
    .venv-blender/bin/python blender/buildings/preview.py                   # all views -> blender/previews/buildings_*.png
    .venv-blender/bin/python blender/buildings/preview.py --views 4s_ranch_street --samples 24 --res 960 540

Scene = LOD0 buildings built from the same specs/model as the tiles (materials from the
client's PBR atlases, keyed by _MAT / _VARIANT with COLOR_0 tints), DEM terrain tinted by NAIP
under a lawn texture, asphalt / gutter / sidewalk ribbons from the real road centerlines,
concrete driveways from each garage door to the street, and the client's vegetation, street
lamp and car glbs at their real placements (client/public/assets/props/placements.bin).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "blender") not in sys.path:
    sys.path.insert(0, str(REPO / "blender"))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from rdlib import bl  # noqa: E402
from shapely.geometry import LineString, Point, box  # noqa: E402

from blender.buildings import build as hd  # noqa: E402
from blender.buildings import model  # noqa: E402

MAT_DIR = REPO / "client" / "public" / "assets" / "materials"
PROPS_DIR = REPO / "client" / "public" / "assets" / "props"
PREVIEW_DIR = REPO / "blender" / "build" / "buildings_hd" / "preview"
OUT_DIR = REPO / "blender" / "previews"

# curb-to-curb widths (m) by road class (SD street design manual typical sections)
ROAD_W = {"residential": 11.0, "living_street": 9.0, "unclassified": 10.0, "tertiary": 14.0, "secondary": 20.0, "primary": 26.0,
          "trunk": 28.0, "motorway": 30.0, "service": 7.0, "driveway": 5.0}

# view -> (data name, kind)
VIEWS = {
    "4s_ranch_street": ("4s_ranch", "street"),
    "4s_ranch_oblique": ("4s_ranch", "oblique"),
    "del_sur_street": ("del_sur", "street"),
    "del_sur_oblique": ("del_sur", "oblique"),
    "4s_ranch_closeup": ("4s_ranch", "closeup"),
}


def log(msg: str) -> None:
    print(f"[preview] {msg}", flush=True)


class Terrain:
    def __init__(self, npz: Path):
        d = np.load(npz)
        self.h = d["height"].astype(np.float64)
        self.naip = d["naip"]
        self.x0, self.z0, self.step = float(d["x0"]), float(d["z0"]), float(d["step"])
        self.n = self.h.shape[0]

    def sample(self, x: Any, z: Any) -> np.ndarray:
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        fj = np.clip((x - self.x0) / self.step, 0, self.n - 1.0001)
        fi = np.clip((z - self.z0) / self.step, 0, self.n - 1.0001)
        j0, i0 = np.floor(fj).astype(int), np.floor(fi).astype(int)
        tj, ti = fj - j0, fi - i0
        a = self.h
        return a[i0, j0] * (1 - ti) * (1 - tj) + a[i0, j0 + 1] * (1 - ti) * tj + a[i0 + 1, j0] * ti * (1 - tj) + a[i0 + 1, j0 + 1] * ti * tj


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------


def span_material(name: str, man: dict[str, Any], atlas: str, cell_name: str, tint_attr: str | None) -> bpy.types.Material:
    """Like bl.atlas_material but picks span cell floor(u) (garage doors: 2 / 3 cells wide)."""
    a = man["atlases"][atlas]
    c = next(x for x in a["cells"] if x["name"] == cell_name)
    cells = c["cells"]
    u0, v0, u1, v1 = cells[0]["uv_inner"]
    du = (cells[1]["uv_inner"][0] - u0) if len(cells) > 1 else 0.0
    mat = bl.atlas_material(name, man, atlas, cell_name, MAT_DIR, tint_attr=tint_attr)
    nt = mat.node_tree
    # find the MULTIPLY_ADD node for u and add floor(u) * du
    sep = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeSeparateXYZ")
    ma_u = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeMath" and n.operation == "MULTIPLY_ADD"
                and abs(n.inputs[2].default_value - u0) < 1e-6)
    fl = nt.nodes.new("ShaderNodeMath")
    fl.operation = "FLOOR"
    nt.links.new(sep.outputs[0], fl.inputs[0])
    cl = nt.nodes.new("ShaderNodeMath")
    cl.operation = "MINIMUM"
    nt.links.new(fl.outputs[0], cl.inputs[0])
    cl.inputs[1].default_value = len(cells) - 1
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY_ADD"
    nt.links.new(cl.outputs[0], mul.inputs[0])
    mul.inputs[1].default_value = du
    nt.links.new(ma_u.outputs[0], mul.inputs[2])
    for sock in [lk.to_socket for lk in ma_u.outputs[0].links if lk.to_node is not mul]:
        nt.links.new(mul.outputs[0], sock)  # replaces the existing link into that socket
    return mat


def building_material(man: dict[str, Any], mat: int, var: int) -> bpy.types.Material:
    name = f"hd_{mat}_{var}"
    if name in bl.ATLAS_MATS:
        return bl.ATLAS_MATS[name]
    if mat == model.MAT_WALL:
        m = bl.atlas_material(name, man, "facade_walls", model.WALL_VARIANTS[var], MAT_DIR, tint_attr="Col")
    elif mat == model.MAT_TILE:
        m = bl.atlas_material(name, man, "roofs", model.TILE_VARIANTS[var], MAT_DIR, normal_strength=1.2)
    elif mat == model.MAT_FLAT:
        m = bl.atlas_material(name, man, "roofs", model.FLAT_VARIANTS[var], MAT_DIR)
    elif mat == model.MAT_GLASS:
        m = bl.atlas_material(name, man, "facade_walls", "glass_curtain", MAT_DIR)
        b = m.node_tree.nodes.get("Principled BSDF")
        for link in list(b.inputs["Roughness"].links):
            m.node_tree.links.remove(link)
        b.inputs["Roughness"].default_value = 0.04
        b.inputs["Specular IOR Level"].default_value = 0.6
    elif mat == model.MAT_TRIM:
        m = bl.atlas_material(name, man, "facade_walls", model.TRIM_VARIANTS[var], MAT_DIR, tint_attr="Col", normal_strength=0.4)
    else:
        m = span_material(name, man, "facade_openings", model.GARAGE_VARIANTS[var], tint_attr="Col")
    bl.ATLAS_MATS[name] = m
    return m


def ground_material(man: dict[str, Any], cell: str, tint_attr: str | None = None) -> bpy.types.Material:
    name = f"ground_{cell}_{tint_attr or ''}"
    if name in bl.ATLAS_MATS:
        return bl.ATLAS_MATS[name]
    m = bl.atlas_material(name, man, "ground", cell, MAT_DIR, tint_attr=tint_attr, ao_mix=0.5)
    bl.ATLAS_MATS[name] = m
    return m


def cell_size(man: dict[str, Any], cell: str) -> float:
    c = next(x for x in man["atlases"]["ground"]["cells"] if x["name"] == cell)
    return float(c["world_size_m"][0])


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def soup_object(name: str, soup: model.Soup, man: dict[str, Any]) -> bpy.types.Object | None:
    P, UV, A = soup.arrays()
    if not len(P):
        return None
    W = hd.weld(P, UV, A)
    keys = sorted({(int(m), int(v)) for m, v in A[:, :2]})
    slot = {k: i for i, k in enumerate(keys)}
    me = hd.make_mesh(name, W)
    for k in keys:
        me.materials.append(building_material(man, *k))
    fm = np.array([slot[(int(m), int(v))] for m, v in A[:, :2]], np.int32)
    if len(me.polygons) == len(fm):
        me.polygons.foreach_set("material_index", fm)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def mesh_object(name: str, V: np.ndarray, F: np.ndarray, uv: np.ndarray | None, mat: bpy.types.Material,
                col: np.ndarray | None = None) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.astype(np.int32).ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, F.shape[1], dtype=np.int32))
    me.update(calc_edges=True)
    if uv is not None:
        u = me.uv_layers.new(name="UVMap")
        u.data.foreach_set("uv", uv[F.ravel()].astype(np.float32).ravel())
    if col is not None:
        ca = me.color_attributes.new("Col", "BYTE_COLOR", "POINT")
        rgba = np.ones((len(V), 4), np.float32)
        rgba[:, :3] = col
        ca.data.foreach_set("color_srgb", rgba.ravel())
    me.materials.append(mat)
    me.polygons.foreach_set("use_smooth", np.ones(len(F), dtype=bool))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def terrain_object(T: Terrain, man: dict[str, Any], stride: int = 1) -> None:
    h = T.h[::stride, ::stride]
    nimg = T.naip[::stride, ::stride].astype(np.float32) / 255.0
    n = h.shape[0]
    xs = T.x0 + T.step * stride * np.arange(n)
    zs = T.z0 + T.step * stride * np.arange(n)
    X, Z = np.meshgrid(xs, zs)
    V = np.column_stack([X.ravel(), -Z.ravel(), h.ravel() - 0.04])
    idx = np.arange(n * n).reshape(n, n)
    # rows go south (-Y): (i, j) -> (i, j+1) -> (i+1, j+1) is clockwise seen from above; flip
    F = np.column_stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()])
    size = cell_size(man, "grass_lawn")
    uv = V[:, :2] / size
    # NAIP: brighten / green the lawn a bit (imagery is hazy at 2.5 m), keep dry slopes tan
    c = nimg.reshape(-1, 3)
    gray = c.mean(axis=1, keepdims=True)
    col = np.clip((gray + (c - gray) * 0.55) * 1.05, 0, 1)
    mesh_object("terrain", V, F, uv, ground_material(man, "grass_lawn", "Col"), col)


def drape_polys(g: Any, T: Terrain, dz: float, cell: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate polygons cut into a cell grid and drape on the terrain: (V (k,3) Blender, F (t,3))."""
    from blender.buildings import geom

    if g.is_empty:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    minx, miny, maxx, maxy = g.bounds
    tris = []
    gx = np.arange(math.floor(minx / cell) * cell, maxx + cell, cell)
    gy = np.arange(math.floor(miny / cell) * cell, maxy + cell, cell)
    tree_g = g
    for x in gx:
        for y in gy:
            c = box(x, y, x + cell, y + cell)
            if not tree_g.intersects(c):
                continue
            part = tree_g.intersection(c)
            part = shapely.make_valid(part)
            polys = [p for p in getattr(part, "geoms", [part]) if p.geom_type == "Polygon" and p.area > 1e-3]
            for p in polys:
                t = geom.triangulate(p)
                if len(t):
                    tris.append(t)
    if not tris:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    t2 = np.concatenate(tris)  # (t, 3, 2) in (x, z) scene coords
    xz = t2.reshape(-1, 2)
    y = T.sample(xz[:, 0], xz[:, 1]) + dz
    V = np.column_stack([xz[:, 0], -xz[:, 1], y])
    F = np.arange(len(V)).reshape(-1, 3)[:, ::-1]  # (x, z) CCW -> (x, -z) CW: flip
    return V, F


def roads_and_driveways(T: Terrain, data: dict[str, Any], builders: list[model.Builder], man: dict[str, Any]) -> Any:
    asph, gutter, walk, park, drive, paths = [], [], [], [], [], []
    for r in data["roads"]:
        ln = LineString(r["pts"])
        cls = r["cls"]
        w = ROAD_W.get(cls, 9.0)
        if cls == "driveway":
            drive.append(ln.buffer(w / 2, cap_style=2))
            continue
        asph.append(ln.buffer(w / 2, cap_style=1, quad_segs=6))
        if cls in ("residential", "living_street", "tertiary", "unclassified", "secondary", "primary"):
            gutter.append(ln.buffer(w / 2 + 0.6, quad_segs=6))
            if cls in ("secondary", "primary"):
                park.append(ln.buffer(w / 2 + 3.0, quad_segs=6))
                walk.append(ln.buffer(w / 2 + 4.8, quad_segs=6))
            else:
                park.append(ln.buffer(w / 2 + 2.0, quad_segs=6))
                walk.append(ln.buffer(w / 2 + 3.5, quad_segs=6))
    A = shapely.union_all(asph) if asph else shapely.Polygon()
    G = shapely.union_all(gutter).difference(A) if gutter else shapely.Polygon()
    Pk = shapely.union_all(park) if park else shapely.Polygon()
    Wk = shapely.union_all(walk).difference(Pk) if walk else shapely.Polygon()
    # driveways: from each garage door straight out to the curb
    for b in builders:
        for p0, p1, n in b.garages:
            q0 = np.array([p0[0], -p0[1]])
            q1 = np.array([p1[0], -p1[1]])
            nn = np.array([n[0], -n[1]])
            mid = (q0 + q1) / 2
            ray = LineString([mid, mid + nn * 30])
            hit = ray.intersection(A.boundary) if not A.is_empty else None
            L = 7.0
            if hit is not None and not hit.is_empty:
                L = min(Point(mid).distance(h) for h in getattr(hit, "geoms", [hit]))
            L = float(np.clip(L + 0.3, 1.5, 30.0))
            t = (q1 - q0) / np.linalg.norm(q1 - q0)
            e0, e1 = q0 - t * 0.25, q1 + t * 0.25
            drive.append(shapely.Polygon([e0, e1, e1 + nn * L, e0 + nn * L]))
        for p0, p1, n in b.entries:  # front walk to the driveway / sidewalk
            q0 = np.array([p0[0], -p0[1]])
            q1 = np.array([p1[0], -p1[1]])
            nn = np.array([n[0], -n[1]])
            mid = (q0 + q1) / 2
            ray = LineString([mid, mid + nn * 25])
            hit = ray.intersection(A.boundary) if not A.is_empty else None
            L = 5.0
            if hit is not None and not hit.is_empty:
                L = min(Point(mid).distance(h) for h in getattr(hit, "geoms", [hit]))
            L = float(np.clip(L, 1.5, 25.0))
            paths.append(LineString([mid, mid + nn * L]).buffer(0.65, cap_style=2))
    D = shapely.union_all(drive).difference(A) if drive else shapely.Polygon()
    Wk = shapely.union_all([Wk, *paths]).difference(A).difference(G)
    D = D.difference(G)
    Wk = Wk.difference(D)
    for name, g, cell, dz in (("asphalt", A, "asphalt_worn", 0.03), ("gutter", G, "concrete_sidewalk", 0.06),
                              ("sidewalk", Wk, "concrete_sidewalk", 0.09), ("driveway", D, "concrete_driveway", 0.08)):
        V, F = drape_polys(g, T, dz)
        if len(F):
            size = cell_size(man, cell)
            mesh_object(name, V, F, V[:, :2] / size, ground_material(man, cell))
    return A


def import_proto(path: Path) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    for o in new:
        o.hide_render = True
        o.hide_viewport = True
    root = meshes[0]
    root.parent = None
    root.matrix_world = root.matrix_world.copy()
    return root


def place_props(T: Terrain, center: tuple[float, float], radius: float, building_polys: Any) -> int:
    pj = json.loads((PROPS_DIR / "placements.json").read_text())
    rec = np.fromfile(PROPS_DIR / pj["bin"], dtype="<f4").reshape(-1, 6)
    cx, cz = center
    sel = (np.abs(rec[:, 0] - cx) < radius) & (np.abs(rec[:, 2] - cz) < radius)
    rec = rec[sel]
    props = {p["index"]: p for p in pj["props"]}
    protos: dict[int, bpy.types.Object] = {}
    paints: dict[tuple[int, int], bpy.types.Mesh] = {}
    pm = {p["id"]: p for p in json.loads((PROPS_DIR / "props_manifest.json").read_text())}
    rng = np.random.default_rng(5)
    n = 0
    for x, _y, z, ry, s, pi in rec:
        pr = props[int(pi)]
        f = PROPS_DIR / pr["file"]
        if not f.exists():
            continue
        if building_polys is not None and building_polys.contains(Point(x, z)):
            continue
        if int(pi) not in protos:
            protos[int(pi)] = import_proto(f)
        proto = protos[int(pi)]
        data = proto.data
        if pr.get("kind") == "vehicle" or pr["id"].startswith("car_"):
            cols = pm.get(pr["id"], {}).get("paint_colors") or []
            if cols:
                w = np.array([c["share"] for c in cols])
                k = int(rng.choice(len(cols), p=w / w.sum()))
                if (int(pi), k) not in paints:
                    me = data.copy()
                    for i, m in enumerate(me.materials):
                        if m and m.name.startswith("paint") and "bus" not in m.name:
                            mm = m.copy()
                            bsdf = mm.node_tree.nodes.get("Principled BSDF")
                            if bsdf:
                                from rdlib import materials as mlib

                                bsdf.inputs["Base Color"].default_value = (*mlib.hex_to_linear(cols[k]["hex"]), 1.0)
                            me.materials[i] = mm
                    paints[(int(pi), k)] = me
                data = paints[(int(pi), k)]
        ob = bpy.data.objects.new(pr["id"], data)
        gy = float(T.sample(x, z)) - 0.03
        ob.location = (float(x), float(-z), gy)
        ob.rotation_euler = (math.pi / 2 if proto.rotation_mode == "QUATERNION" and False else 0.0, 0.0, float(ry))
        ob.rotation_mode = "XYZ"
        ob.scale = (float(s), float(s), float(s))
        bpy.context.scene.collection.objects.link(ob)
        n += 1
    return n


def setup_scene(view: str, samples: int, res: tuple[int, int]) -> tuple[Path, dict[str, Any]]:
    data_name, kind = VIEWS[view]
    bl.reset_scene()
    man = json.loads((MAT_DIR / "materials_manifest.json").read_text())
    T = Terrain(PREVIEW_DIR / f"{data_name}.npz")
    data = json.loads((PREVIEW_DIR / f"{data_name}.json").read_text())
    cx, cz = data["center"]
    R = data["radius"] - 20
    index = hd.load_index()
    params = {**model.DEFAULT_PARAMS, **(index.get("params") or {})}
    # buildings near the view (from all tiles touching the square)
    builders = []
    soup = model.Soup()
    polys = []
    t0 = time.time()
    for tid in index["tiles"]:
        f = hd.SPEC_DIR / f"{tid}.json"
        for sp in json.loads(f.read_text())["buildings"]:
            ring = np.asarray(sp["ring"])
            c = ring.mean(axis=0)
            if abs(c[0] - cx) > R or abs(c[1] - cz) > R:
                continue
            try:
                b = model.Builder(sp, 0, params)
                soup.extend(b.build())
                builders.append(b)
                polys.append(shapely.Polygon(ring).buffer(0.5))
            except Exception as e:  # noqa: BLE001
                log(f"building {sp['id']}: {e}")
    log(f"{len(builders)} buildings, {soup.ntris:,} tris ({time.time() - t0:.0f}s)")
    soup_object("buildings", soup, man)
    terrain_object(T, man)
    asphalt = roads_and_driveways(T, data, builders, man)
    bp = shapely.union_all(polys) if polys else None
    if bp is not None and asphalt is not None and not asphalt.is_empty:
        bp = shapely.union_all([bp, asphalt.buffer(-2.0)])
    nprops = place_props(T, (cx, cz), R, bp)
    log(f"{nprops} props placed")
    sc = bl.setup_render(res[0], res[1], samples)
    sc.cycles.max_bounces = 4
    sc.view_settings.exposure = -0.35
    bl.setup_world(sun_elev_deg=36.0, sun_azimuth_deg=200.0, strength=5.2, sky_strength=0.12)
    # camera
    if kind == "street":
        cam, tgt = street_camera(data, T, builders)
        bl.add_camera(cam, tgt, lens=24.0)
    elif kind == "closeup":
        hs = sorted([b for b in builders if b.garages], key=lambda b: np.hypot(*(np.array(b.fp.plan_poly.centroid.coords[0]) - [cx, -cz])))
        p0, p1, n = hs[0].garages[0]
        m = (p0 + p1) / 2
        c = m + n * 15 + np.array([-n[1], n[0]]) * 5.0
        log(f"closeup cam {c} target {m} floor {hs[0].floor}")
        gz = float(T.sample(c[0], -c[1]))
        log(f"closeup ground {gz}")
        bl.add_camera((float(c[0]), float(c[1]), gz + 2.4), (float(m[0]), float(m[1]), hs[0].floor + 2.2), lens=30.0)
    else:
        gz = float(T.sample(cx, cz))
        cam = (cx - 95.0, -(cz + 70.0), gz + 85.0)
        bl.add_camera(cam, (cx + 10, -(cz - 5), gz), lens=32.0)
    sc.camera.data.clip_end = 3000
    return OUT_DIR / f"buildings_{view}.png", {"buildings": len(builders), "tris": soup.ntris, "props": nprops}


def street_camera(data: dict[str, Any], T: Terrain, builders: list[model.Builder]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """On a residential street near the center with house fronts along it, looking down the street."""
    cx, cz = data["center"]
    best = None
    houses = [b for b in builders if b.btype == "house"]
    polys = [shapely.Polygon(np.asarray(b.spec["ring"])) for b in houses]
    tree = shapely.STRtree(polys) if polys else None
    for r in data["roads"]:
        if r["cls"] not in ("residential",):
            continue
        ln = LineString(r["pts"])
        if ln.length < 80 or tree is None:
            continue
        for f in (0.25, 0.4, 0.55, 0.7):
            p = np.asarray(ln.interpolate(f, normalized=True).coords[0])
            q = np.asarray(ln.interpolate(min(1.0, f + 25.0 / ln.length), normalized=True).coords[0])
            d = q - p
            if np.hypot(*d) < 1:
                continue
            d /= np.hypot(*d)
            nrm = np.array([-d[1], d[0]])
            view = LineString([p, p + d * 60])
            near = [polys[i] for i in tree.query(view.buffer(30))]
            if not near:
                continue
            dmin = min(view.distance(g) for g in near)
            if dmin < 8.5:  # alley / private drive with houses on the pavement edge
                continue
            sides = [float((np.asarray(g.centroid.coords[0]) - p) @ nrm) for g in near if view.distance(g) < 22]
            left, right = sum(1 for x in sides if x > 0), sum(1 for x in sides if x < 0)
            score = min(left, right) * 2 + left + right - 0.01 * np.hypot(*(p - [cx, cz]))
            if best is None or score > best[0]:
                best = (score, p, d)
    if best is None:
        g = float(T.sample(cx, cz))
        return (cx, -cz, g + 1.7), (cx + 30, -cz, g + 2)
    _, p, d = best
    nrm = np.array([-d[1], d[0]])
    cam = p + nrm * 2.6
    tgt = p + d * 40 - nrm * 4.0
    gy = float(T.sample(*cam))
    ty = float(T.sample(*tgt))
    return (float(cam[0]), float(-cam[1]), gy + 1.65), (float(tgt[0]), float(-tgt[1]), ty + 3.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", nargs="*", default=list(VIEWS))
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--res", nargs=2, type=int, default=[1600, 900])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    for v in a.views:
        t0 = time.time()
        path, info = setup_scene(v, a.samples, tuple(a.res))
        if a.out:
            path = a.out / path.name
        bl.render(path, max_kb=900)
        log(f"{v}: {path} {info} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]))
