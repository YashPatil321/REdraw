"""Cycles beauty previews of the material atlases: material sheets and a street scene.

Everything is rendered from the built files in client/public/assets/materials (the same
atlases the client loads), so the previews double as a visual test of the manifest contract.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import bpy
import numpy as np

from . import atlas, bl, foliage, houses, materials, props, vehicles
from .mesh import Part, box

WALL_TINTS = ["#E8D9BC", "#DCC5A0", "#CDB08A", "#EFE9DE", "#BDB394", "#D9C3A2", "#C9B7A0", "#E2CFB0"]


def ensure_materials(part: Part, man: dict[str, Any], mat_dir: Path) -> None:
    for key in sorted(set(part.M)):
        if key in bl.ATLAS_MATS or key in materials.MATERIALS:
            continue
        kind, _, rest = key.partition(":")
        if kind == "wall":
            variant, _, tint = rest.partition(":")
            tint_hex = None if tint in ("", "#FFFFFF") else tint
            bl.atlas_material(key, man, "facade_walls", variant, mat_dir, tint_hex=tint_hex)
        elif kind == "open":
            cell, k, tint = rest.split(":")
            bl.atlas_material(key, man, "facade_openings", cell, mat_dir, span_k=int(k), tint_hex=tint)
        elif kind == "roof":
            bl.atlas_material(key, man, "roofs", rest, mat_dir)
        elif kind == "ground":
            bl.atlas_material(key, man, "ground", rest, mat_dir, ao_mix=0.5)
        elif kind == "decal":
            col = next(c for c in man["markings"]["columns"] if c["name"] == rest)
            bl.decal_material(key, mat_dir / man["markings"]["file"], tuple(col["uv"]))
        else:
            raise KeyError(f"unknown preview material key {key}")


def add_part(part: Part, name: str, man: dict[str, Any], mat_dir: Path, smooth: float | None = None) -> bpy.types.Object:
    for i, u in enumerate(part.U):  # faces built without UVs (boxes): facade convention
        if u is None:
            part.U[i] = [tuple(x) for x in atlas.face_uv(part.V[part.F[i]], atlas.MAT_WALL)]
    ensure_materials(part, man, mat_dir)
    return bl.part_to_object(part, name, smooth_angle=smooth)


def label(text: str, x: float, y: float, z: float, size: float = 0.32, rot_x: float = 90.0) -> None:
    cu = bpy.data.curves.new("lbl", "FONT")
    cu.body = text
    cu.size = size
    cu.align_x = "CENTER"
    ob = bpy.data.objects.new("lbl", cu)
    ob.location = (x, y, z)
    ob.rotation_euler = (math.radians(rot_x), 0, 0)
    ob.data.materials.append(bl.simple_material("label", "#1C1C1C", 0.9))
    bpy.context.scene.collection.objects.link(ob)


# ---------------------------------------------------------------------------
# material sheets
# ---------------------------------------------------------------------------


def _panel(x: float, z: float, w: float, h: float, key: str, uv: tuple[float, float, float, float]) -> Part:
    u0, v0, u1, v1 = uv
    P = np.array([(x, 0, z), (x + w, 0, z), (x + w, 0, z + h), (x, 0, z + h)], float)
    return Part(P, [[0, 1, 2, 3]], [key], [[(u0, v0), (u1, v0), (u1, v1), (u0, v1)]])


def sheet(man: dict[str, Any], mat_dir: Path, path: Path, rows: list[list[tuple[str, str, float, int]]], samples: int,
          width_px: int = 1600) -> None:
    """rows of (material key, label, width in panel units, uv repeats); panels are 3 x 3 m, vertical, raking sun."""
    bl.reset_scene()
    S, gap = 3.0, 0.45
    p = Part()
    maxw = 0.0
    for r, row in enumerate(rows):
        x = 0.0
        z = -r * (S + 0.9)
        for key, text, wu, rep in row:
            w = S * wu + gap * (wu - 1)
            p += _panel(x, z, w, S, key, (0.0, 0.0, wu * rep if wu > 1 else rep, rep))
            label(text, x + w / 2, -0.01, z - 0.42, size=0.3)
            x += w + gap
        maxw = max(maxw, x - gap)
    add_part(p, "sheet", man, mat_dir)
    bl.setup_render(width_px, int(width_px * (S + 0.25 + (len(rows) - 1) * (S + 0.9) + 0.75) / (maxw + 0.8)), samples=samples)
    bl.setup_world(sun_elev_deg=32, sun_azimuth_deg=215, strength=4.5, sky_strength=0.12)
    bpy.context.scene.view_settings.exposure = -0.25
    bl.ground_plane(200, "#E9E7E2", z=-60, roughness=1.0)
    back = bpy.data.meshes.new("back")
    back.from_pydata([(-50, 3, -80), (100, 3, -80), (100, 3, 40), (-50, 3, 40)], [], [[0, 1, 2, 3]])
    back.materials.append(bl.simple_material("backdrop", "#ECEAE5", 1.0))
    bo = bpy.data.objects.new("back", back)
    bpy.context.scene.collection.objects.link(bo)
    top, bottom = S + 0.25, -(len(rows) - 1) * (S + 0.9) - 0.75
    cx, cz = maxw / 2, (top + bottom) / 2
    bl.add_camera((cx, -60, cz), (cx, 0, cz), ortho_scale=maxw + 0.8)
    bl.render(path)


def materials_sheets(man: dict[str, Any], mat_dir: Path, out_dir: Path, samples: int) -> list[Path]:
    walls = [c["name"] for c in man["atlases"]["facade_walls"]["cells"]]
    wrow = [(f"wall:{n}:{WALL_TINTS[i] if man['atlases']['facade_walls']['cells'][i]['tintable'] else '#FFFFFF'}", n, 1, 1)
            for i, n in enumerate(walls)]
    t = "#E3D2B2"
    singles = ["window_slider", "window_pair", "window_picture", "window_small", "window_arched", "door_front", "door_slider", "storefront"]
    orow = [(f"open:{n}:0:{t}", n, 1, 1) for n in singles]
    grow = [(f"open:garage_2car:0:{t}", "garage_2car", 2, 1), (f"open:garage_3car:0:{t}", "garage_3car", 3, 1),
            (f"open:storefront_sign:0:{t}", "storefront_sign", 1, 1), ("open:school_window_band:0:#D8D2C4", "school_window_band", 1, 1),
            ("open:school_door:0:#D8D2C4", "school_door", 1, 1)]
    p1 = out_dir / "materials_facade.jpg"
    # span panels: build each span cell as its own key so the preview uses the real per-cell lookup
    rows = [wrow, orow, grow]
    sheet_spans(man, mat_dir, p1, rows, samples)
    roofs = [(f"roof:{c['name']}", c["name"], 1, 2) for c in man["atlases"]["roofs"]["cells"]]
    ground = [(f"ground:{c['name']}", c["name"], 1, 2) for c in man["atlases"]["ground"]["cells"]]
    p2 = out_dir / "materials_roofs_ground.jpg"
    sheet(man, mat_dir, p2, [roofs[:8], roofs[8:], ground[:8], ground[8:]], samples)
    return [p1, p2]


def sheet_spans(man: dict[str, Any], mat_dir: Path, path: Path, rows: list, samples: int) -> None:
    """Like sheet(), but multi-cell spans are built from one panel per span cell (span cell k material)."""
    bl.reset_scene()
    S, gap = 3.0, 0.45
    p = Part()
    maxw = 0.0
    for r, row in enumerate(rows):
        x = 0.0
        z = -r * (S + 0.9)
        for key, text, wu, rep in row:
            if wu > 1:
                base = key.split(":")
                for k in range(wu):
                    kk = f"open:{base[1]}:{k}:{base[3]}"
                    p += _panel(x + k * S, z, S, S, kk, (k, 0, k + 1, 1))
                w = S * wu
            else:
                w = S
                p += _panel(x, z, w, S, key, (0, 0, rep, rep))
            label(text, x + w / 2, -0.01, z - 0.42, size=0.3)
            x += w + gap
        maxw = max(maxw, x - gap)
    add_part(p, "sheet", man, mat_dir)
    bl.setup_render(1600, int(1600 * (S + 0.25 + (len(rows) - 1) * (S + 0.9) + 0.75) / (maxw + 0.8)), samples=samples)
    bl.setup_world(sun_elev_deg=30, sun_azimuth_deg=215, strength=4.5, sky_strength=0.12)
    bpy.context.scene.view_settings.exposure = -0.25
    back = bpy.data.meshes.new("back")
    back.from_pydata([(-50, 3, -80), (100, 3, -80), (100, 3, 40), (-50, 3, 40)], [], [[0, 1, 2, 3]])
    back.materials.append(bl.simple_material("backdrop", "#ECEAE5", 1.0))
    bo = bpy.data.objects.new("back", back)
    bpy.context.scene.collection.objects.link(bo)
    top, bottom = S + 0.25, -(len(rows) - 1) * (S + 0.9) - 0.75
    cx, cz = maxw / 2, (top + bottom) / 2
    bl.add_camera((cx, -60, cz), (cx, 0, cz), ortho_scale=maxw + 0.8)
    bl.render(path)


# ---------------------------------------------------------------------------
# street scene
# ---------------------------------------------------------------------------


class Instancer:
    """Hidden prototype objects + linked duplicates (cheap instancing in Cycles)."""

    def __init__(self, tex_dir: Path):
        self.tex_dir = tex_dir
        self.protos: dict[str, bpy.types.Object] = {}
        self.painted: dict[tuple[str, str], bpy.types.Mesh] = {}

    def proto(self, key: str) -> bpy.types.Object:
        if key in self.protos:
            return self.protos[key]
        if key in foliage.SPECIES:
            built = foliage.build(key, self.tex_dir / f"{key}.png")
            bl.TEXTURES["foliage_atlas"] = built.tex
            if "foliage" in bpy.data.materials:
                bpy.data.materials["foliage"].name = f"foliage_{len(self.protos)}"
            o = bl.part_to_object(built.part, f"proto_{key}")
            bl.set_custom_normals(o, built.normals)
            bl.set_vertex_ao(o, built.ao)
        elif key == "street_lamp":
            o = bl.part_to_object(props.street_lamp(), "proto_lamp", smooth_angle=45)
        else:
            o = bl.part_to_object(vehicles.VEHICLES[key][0](), f"proto_{key}", smooth_angle=40)
        o.hide_render = True
        o.hide_viewport = True
        self.protos[key] = o
        return o

    def place(self, key: str, x: float, y: float, z: float = 0.0, rot_deg: float = 0.0, s: float = 1.0,
              paint: str | None = None) -> bpy.types.Object:
        pr = self.proto(key)
        data = pr.data
        if paint:
            if (key, paint) not in self.painted:
                me = pr.data.copy()
                for i, m in enumerate(me.materials):
                    if m and m.name.startswith("paint") and not m.name.startswith("paint_bus"):
                        mm = m.copy()
                        mm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*materials.hex_to_linear(paint), 1.0)
                        me.materials[i] = mm
                self.painted[(key, paint)] = me
            data = self.painted[(key, paint)]
        ob = bpy.data.objects.new(key, data)
        ob.location = (x, y, z)
        ob.rotation_euler = (0, 0, math.radians(rot_deg))
        ob.scale = (s, s, s)
        bpy.context.scene.collection.objects.link(ob)
        return ob


def street_scene(man: dict[str, Any], mat_dir: Path, tex_dir: Path, path: Path, samples: int,
                 res: tuple[int, int] = (1600, 900)) -> None:
    bl.reset_scene()
    rng = np.random.default_rng(7)
    inst = Instancer(tex_dir)
    world = Part()
    lot = 19.0
    xs = [-38.0, -19.0, 0.0, 19.0, 38.0, 57.0, 76.0]
    roofs = ["s_tile_blend", "s_tile_brown", "flat_tile_grey", "s_tile_terracotta", "flat_tile_brown", "s_tile_blend", "barrel_mission"]
    tints = ["#E3D0AE", "#D6BE98", "#EDE6D8", "#CDB592", "#E6D9C0", "#C9B48E", "#E8DCC4"]
    trims = ["#F1EEE6", "#EAE3D2", "#F3F1EC", "#E9E1CF", "#F2EFE7", "#EFE9DA", "#F0ECE2"]
    variants = ["stucco_sand", "stucco_lace", "stucco_sand", "stucco_smooth", "stucco_sand", "stucco_catface", "stucco_sand"]
    z0 = 0.17
    for i, x in enumerate(xs):
        st = houses.HouseStyle(tints[i], trims[i], roofs[i], wall_variant=variants[i],
                               garage="garage_3car" if i in (2, 5) else "garage_2car", garage_side=-1 if i % 2 == 0 else 1, seed=i + 3)
        hp, lay = houses.house(x, 23.0, st)
        world += hp.moved(0, 0, z0)
        gx0, gx1, gy = lay["driveway"]
        bx0, by0, bw, bd = lay["body"]
        # yard: lawn, driveway, walk, planting beds
        world += houses.ground_rect(x - lot / 2, 10.1, x + lot / 2, 23.0 + bd + 6, "grass_lawn", 2.0, z0)
        world += houses.ground_rect(gx0, 10.1, gx1, gy, "concrete_driveway", 6.0, z0 + 0.012)
        world += houses.ground_rect(gx0, 6.0, gx1, 10.1, "concrete_driveway", 6.0, z0 + 0.012)  # apron across the parkway
        dx = lay["door"][0]
        world += houses.ground_rect(dx - 0.7, gy + 0.6, dx + 0.7, 23.0, "pavers", 3.0, z0 + 0.012)
        bed_x0, bed_x1 = (gx1 + 0.3, bx0 + bw) if st.garage_side < 0 else (bx0, gx0 - 0.3)
        world += houses.ground_rect(bed_x0, 21.6, bed_x1, 23.0, "mulch", 2.0, z0 + 0.008)
        bed_plant = "succulent_agave" if i % 3 == 2 else "shrub"  # xeriscape beds on some lots
        for k in range(int((bed_x1 - bed_x0) / 1.6)):
            sx = bed_x0 + 0.8 + k * 1.6
            if abs(sx - dx) > 1.2:
                kind = bed_plant if k % 2 == 0 else "shrub"
                inst.place(kind, sx, 22.3 + rng.uniform(-0.2, 0.2), z0, rng.uniform(0, 360),
                           rng.uniform(0.55, 0.75) if kind == "succulent_agave" else rng.uniform(0.8, 1.1))
        if i in (0, 3, 5):  # clipped hedge along the lot line (2 m segments, local x along the run)
            for k in range(5):
                inst.place("hedge", x + lot / 2 - 0.5, 12.0 + 2.0 * k, z0, 90.0, 1.0)
        inst.place("grass_ornamental", gx0 - 0.6 if st.garage_side > 0 else gx1 + 0.6, gy + 0.6, z0, rng.uniform(0, 360))
        # yard tree / palm
        if i % 3 == 0:
            inst.place("tree_palm_queen", x + (5.5 if st.garage_side < 0 else -5.5), 14.5, z0, rng.uniform(0, 360), 0.9)
        elif i % 3 == 1:
            inst.place("tree_jacaranda", x + (5.0 if st.garage_side < 0 else -5.0), 15.0, z0, rng.uniform(0, 360), 0.8)
        # street trees in the parkway (not in front of the driveway apron)
        for tx in (x - 6.5, x + 3.5):
            if not (gx0 - 1.5 < tx < gx1 + 1.5):
                inst.place("tree_street", tx, 7.6, z0, rng.uniform(0, 360), rng.uniform(0.85, 1.05))
        # side-yard return wall with a gate pier (stucco, 1.8 m)
        wx = bx0 - 0.9 if st.garage_side > 0 else bx0 + bw + 0.9
        world += box(0.2, 4.0, 1.8, f"wall:stucco_sand:{tints[i]}").moved(wx, 25.0, z0)
        if i in (1, 4):
            inst.place("shrub_bougainvillea", wx, 23.2, z0, rng.uniform(0, 360), 0.8)
    # backyard fences / second row backdrop (simple massing: two-storey boxes with hip roofs)
    for i, x in enumerate(np.arange(-60, 110, 17.0)):
        st = houses.HouseStyle(tints[i % 7], trims[i % 7], roofs[(i + 3) % 7], floors=2, seed=40 + i)
        hp, _ = houses.house(float(x), 62.0, st)
        world += hp.rotated_z(0).moved(0, 0, z0 + 1.0)
    # street, curbs, parkways, sidewalks
    x0, x1 = -90.0, 140.0
    world += houses.ground_rect(x0, -5.4, x1, 5.4, "asphalt_worn", 4.0, 0.0)
    world += houses.curb(x0, x1, 5.4, +1)
    world += houses.curb(x0, x1, -5.4, -1)
    for y0, y1, cell, size in ((6.0, 8.6, "grass_lawn", 2.0), (8.6, 10.1, "concrete_sidewalk", 3.0),
                               (-8.6, -6.0, "grass_lawn", 2.0), (-10.1, -8.6, "concrete_sidewalk", 3.0),
                               (-40.0, -10.1, "grass_lawn", 2.0)):
        world += houses.ground_rect(x0, y0, x1, y1, cell, size, z0)
    world += houses.ground_rect(x0, 10.1, x1, 120.0, "grass_lawn", 2.0, z0 - 0.02)
    # lane markings: continental crosswalk and a stop bar near the camera, faded bike/parking edge line
    for k in range(6):
        bx = -16.0 + k * 1.5
        P = [(bx - 0.5, -5.2, 0.004), (bx + 0.5, -5.2, 0.004), (bx + 0.5, 5.2, 0.004), (bx - 0.5, 5.2, 0.004)]
        world += Part(np.array(P), [[0, 1, 2, 3]], ["decal:crosswalk_bar_24in"], [[(0, -5.2 / 3), (1, -5.2 / 3), (1, 5.2 / 3), (0, 5.2 / 3)]])
    for yl in (-3.4, 3.4):
        P = [(x0, yl - 0.15, 0.003), (x1, yl - 0.15, 0.003), (x1, yl + 0.15, 0.003), (x0, yl + 0.15, 0.003)]
        world += Part(np.array(P), [[0, 1, 2, 3]], ["decal:white_solid_4in"], [[(0, x0 / 8), (0, x1 / 8), (1, x1 / 8), (1, x0 / 8)]])
    P = [(-5.0, -0.15, 0.003), (x1, -0.15, 0.003), (x1, 0.15, 0.003), (-5.0, 0.15, 0.003)]
    world += Part(np.array(P), [[0, 1, 2, 3]], ["decal:yellow_dashed_4in"], [[(0, -5 / 12), (0, x1 / 12), (1, x1 / 12), (1, -5 / 12)]])
    add_part(world, "street", man, mat_dir)
    # near side: parkway trees, lamp, hedge line
    for tx, sp in ((-24.0, "tree_street"), (6.0, "tree_jacaranda"), (30.0, "tree_street")):
        inst.place(sp, tx, -7.4, z0, rng.uniform(0, 360), 1.0)
    inst.place("tree_ficus", -34.0, -16.0, z0, 20.0, 1.0)
    for k in range(10):
        inst.place("grass_tuft", -9.0 + k * 0.9 + rng.uniform(-0.3, 0.3), -7.0 + rng.uniform(-0.6, 0.6), z0, rng.uniform(0, 360), 1.2)
    # Canary Island pines behind the first row
    for px in (-12.0, 66.0):
        inst.place("tree_pine_canary", px, 47.0, z0 + 1.0, rng.uniform(0, 360), 0.95)
    inst.place("street_lamp", 12.0, -6.4, z0, 0.0)
    inst.place("street_lamp", 45.0, 6.4, z0, 180.0)
    # vehicles
    pal = ["#E9EAEA", "#16171A", "#5E6267", "#A9ADB1", "#1F3B66", "#8E1B1E", "#F1EFE8"]
    inst.place("car_suv", -19.0 + 2.6, 14.0, z0 + 0.012, 0.0, paint=pal[3])  # in a driveway, nose to the garage
    inst.place("car_crossover_ev", 0.0 - 4.6 + 2.2, 13.5, z0 + 0.012, 180.0, paint=pal[0])
    inst.place("car_minivan", 38.0 - 3.0, 14.2, z0 + 0.012, 0.0, paint=pal[2])
    inst.place("car_sedan", 9.0, 4.15, 0.0, -90.0, paint=pal[4])  # parked at the north curb, facing east
    inst.place("car_pickup", 26.0, 4.2, 0.0, -90.0, paint=pal[1])
    inst.place("car_sedan", 3.0, -1.9, 0.0, 90.0, paint=pal[5])  # driving west in the near lane
    inst.place("car_suv", 52.0, -4.2, 0.0, 90.0, paint=pal[0])
    # skyline: eucalyptus windbreak and chaparral hills
    for k in range(14):
        inst.place("tree_eucalyptus", -60 + k * 13 + rng.uniform(-4, 4), 95 + rng.uniform(-6, 6), z0, rng.uniform(0, 360), rng.uniform(0.9, 1.2))
    for k in range(10):
        inst.place("tree_palm_fan", -30 + k * 17 + rng.uniform(-3, 3), 75 + rng.uniform(-4, 4), z0, rng.uniform(0, 360), rng.uniform(0.9, 1.15))
    hill = Part()
    n = 40
    for i in range(n):
        xa, xb = -400 + 1000 * i / n, -400 + 1000 * (i + 1) / n
        ha = 55 + 25 * math.sin(xa / 90) + 12 * math.sin(xa / 31)
        hb = 55 + 25 * math.sin(xb / 90) + 12 * math.sin(xb / 31)
        P = np.array([(xa, 260, -5), (xb, 260, -5), (xb, 420, hb), (xa, 420, ha)], float)
        hill += Part(P, [[0, 1, 2, 3]], ["ground:chaparral"], [[(p[0] / 12, p[1] / 12 + p[2] / 12) for p in P]])
    add_part(hill, "hills", man, mat_dir)
    bl.ground_plane(600, "#8E8A66", z=-0.5, roughness=1.0)
    bl.setup_render(*res, samples=samples)
    bl.setup_world(sun_elev_deg=34, sun_azimuth_deg=238, strength=5.0, sky_strength=0.09)
    sc = bpy.context.scene
    sc.view_settings.exposure = -0.35
    sc.cycles.max_bounces = 4
    bl.add_camera((-8.0, -9.0, 1.62), (14.0, 16.0, 3.2), lens=26)
    bl.render(path)
