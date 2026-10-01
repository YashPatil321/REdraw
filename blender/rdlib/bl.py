"""Blender (bpy) side: Part -> object with materials, glTF export, preview renders.

Two material modes, chosen per face by its key:
- keys in materials.MATERIALS -> real PBR materials (props: vehicles, lamps, trees)
- keys in palette.PALETTE     -> per-face vertex colors (COLOR_0) on a few shared
  "class" materials (hero campuses: hero_matte / hero_glass / hero_metal). The
  pipeline's hero loader keeps baseColorFactor, roughness and COLOR_0, so heroes
  bake into building tiles with very few primitives.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np

from . import materials as mlib
from . import palette
from .mesh import Part

REPO = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO / "blender" / "build"  # generated textures (regenerated, not committed)

# texture key -> png path (registered by foliage.py before building trees)
TEXTURES: dict[str, Path] = {}

HERO_CLASSES = {
    # name: (roughness, metallic)
    "hero_matte": (0.9, 0.0),
    "hero_glass": (0.08, 0.0),
    "hero_metal": (0.4, 0.7),
    "hero_glow": (0.3, 0.0),
}


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.objects, bpy.data.lights,
                 bpy.data.cameras, bpy.data.worlds):
        for block in list(coll):
            coll.remove(block)


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------


def _bsdf(mat: bpy.types.Material) -> bpy.types.Node:
    mat.use_nodes = True
    return mat.node_tree.nodes.get("Principled BSDF")


def pbr_material(key: str) -> bpy.types.Material:
    mat = bpy.data.materials.get(key)
    if mat is not None:
        return mat
    spec = mlib.get(key)
    mat = bpy.data.materials.new(key)
    b = _bsdf(mat)
    nt = mat.node_tree
    mat.use_backface_culling = not spec.double_sided
    b.inputs["Base Color"].default_value = (*mlib.hex_to_linear(spec.hex), 1.0)
    b.inputs["Roughness"].default_value = spec.roughness
    b.inputs["Metallic"].default_value = spec.metallic
    if spec.emissive_hex:
        b.inputs["Emission Color"].default_value = (*mlib.hex_to_linear(spec.emissive_hex), 1.0)
        b.inputs["Emission Strength"].default_value = spec.emissive_strength
    if spec.clearcoat > 0:
        b.inputs["Coat Weight"].default_value = spec.clearcoat
        b.inputs["Coat Roughness"].default_value = spec.clearcoat_roughness
    if spec.texture:
        img = nt.nodes.new("ShaderNodeTexImage")
        img.image = bpy.data.images.load(str(TEXTURES[spec.texture]), check_existing=True)
        img.image.colorspace_settings.name = "sRGB"
        img.image.alpha_mode = "STRAIGHT"
        nt.links.new(img.outputs["Color"], b.inputs["Base Color"])
        if spec.alpha_mask:
            # glTF exporter: alpha through a "Round" math node -> alphaMode MASK.
            rnd = nt.nodes.new("ShaderNodeMath")
            rnd.operation = "ROUND"
            nt.links.new(img.outputs["Alpha"], rnd.inputs[0])
            nt.links.new(rnd.outputs[0], b.inputs["Alpha"])
    return mat


def hero_material(cls: str) -> bpy.types.Material:
    mat = bpy.data.materials.get(cls)
    if mat is not None:
        return mat
    rough, metal = HERO_CLASSES[cls]
    mat = bpy.data.materials.new(cls)
    b = _bsdf(mat)
    nt = mat.node_tree
    mat.use_backface_culling = False  # glTF doubleSided: awnings, canopies, shade sails are single sheets
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "Col"
    nt.links.new(vc.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if cls == "hero_glow":
        nt.links.new(vc.outputs["Color"], b.inputs["Emission Color"])
        b.inputs["Emission Strength"].default_value = 1.5
    return mat


def _hero_class_of(sw: palette.Swatch, key: str) -> str:
    if key in ("lamp_lens",):
        return "hero_glow"
    if sw.roughness < 0.2:
        return "hero_glass"
    if sw.metallic >= 0.5:
        return "hero_metal"
    return "hero_matte"


def _srgb8_to_float(h: str) -> tuple[float, float, float]:
    r, g, b = palette.hex_to_rgb8(h)
    return r / 255.0, g / 255.0, b / 255.0


# ---------------------------------------------------------------------------
# Part -> object
# ---------------------------------------------------------------------------


def part_to_object(part: Part, name: str, collection: bpy.types.Collection | None = None, *,
                   smooth_angle: float | None = None, split_materials: bool = False,
                   vehicle_attrs: bool = False, color_jitter: float = 0.0, seed: int = 0,
                   ao_ground: float = 0.0) -> bpy.types.Object:
    """Build a Blender mesh object from a Part.

    smooth_angle: shade smooth with sharp edges above this angle (deg); None = flat.
    split_materials: split edges on material boundaries (crisp normals; also makes the
        point-domain `_LIGHT` / `_TINT` attributes unambiguous).
    vehicle_attrs: add `_LIGHT` and `_TINT` float vertex attributes (see materials.Mat).
    color_jitter / ao_ground: vertex-color mode only; per-face brightness jitter and a
        darkening of faces near the ground (cheap ambient occlusion baked into COLOR_0).
    """
    me = bpy.data.meshes.new(name)
    me.from_pydata(part.V.tolist(), [], part.F)
    me.validate(clean_customdata=False)
    if len(me.polygons) != len(part.F):
        raise RuntimeError(f"{name}: mesh.validate removed {len(part.F) - len(me.polygons)} degenerate faces")

    slot_of: dict[str, int] = {}
    face_slot = np.zeros(len(part.F), dtype=np.int32)
    vcol_faces: list[tuple[int, str]] = []
    for i, key in enumerate(part.M):
        if key in mlib.MATERIALS:
            mname = key
            mat_fn = pbr_material
        elif key in palette.PALETTE:
            mname = _hero_class_of(palette.PALETTE[key], key)
            mat_fn = hero_material
            vcol_faces.append((i, key))
        else:
            raise KeyError(f"{name}: unknown material/palette key '{key}'")
        if mname not in slot_of:
            slot_of[mname] = len(me.materials)
            me.materials.append(mat_fn(mname))
        face_slot[i] = slot_of[mname]
    me.polygons.foreach_set("material_index", face_slot)

    if any(u is not None for u in part.U):
        uv = me.uv_layers.new(name="UVMap")
        uvs = np.zeros((len(me.loops), 2), dtype=np.float32)
        for poly, u in zip(me.polygons, part.U, strict=True):
            if u is not None:
                uvs[poly.loop_start : poly.loop_start + poly.loop_total] = u
        uv.data.foreach_set("uv", uvs.ravel())

    if vcol_faces:
        ca = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
        cols = np.ones((len(me.loops), 4), dtype=np.float32)
        rng = np.random.default_rng(seed)
        cache: dict[str, tuple[float, float, float]] = {}
        zmin = float(part.V[:, 2].min()) if len(part.V) else 0.0
        for i, key in vcol_faces:
            if key not in cache:
                cache[key] = _srgb8_to_float(palette.PALETTE[key].hex)
            c = np.array(cache[key])
            if color_jitter > 0:
                c = c * (1.0 + rng.uniform(-color_jitter, color_jitter))
            poly = me.polygons[i]
            if ao_ground > 0:
                zs = part.V[part.F[i], 2]
                for k, z in enumerate(zs):
                    shade = 1.0 - ao_ground * max(0.0, 1.0 - (z - zmin) / 1.2)
                    cols[poly.loop_start + k, :3] = np.clip(c * shade, 0, 1)
            else:
                cols[poly.loop_start : poly.loop_start + poly.loop_total, :3] = np.clip(c, 0, 1)
        ca.data.foreach_set("color", cols.ravel())
        me.color_attributes.active_color = ca

    if smooth_angle is None:
        me.polygons.foreach_set("use_smooth", [False] * len(me.polygons))
    else:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
        me.set_sharp_from_angle(angle=math.radians(smooth_angle))

    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(obj)

    if split_materials or vehicle_attrs:
        import bmesh

        bm = bmesh.new()
        bm.from_mesh(me)
        edges = [e for e in bm.edges if len(e.link_faces) == 2 and e.link_faces[0].material_index != e.link_faces[1].material_index]
        bmesh.ops.split_edges(bm, edges=edges)
        bm.to_mesh(me)
        bm.free()
    if vehicle_attrs:
        n = len(me.vertices)
        light = np.zeros(n, dtype=np.float32)
        tint = np.zeros(n, dtype=np.float32)
        mats = [me.materials[i].name for i in range(len(me.materials))]
        for poly in me.polygons:
            spec = mlib.MATERIALS.get(mats[poly.material_index])
            if spec is None:
                continue
            for vi in poly.vertices:
                light[vi] = spec.light
                tint[vi] = 1.0 if spec.tint else 0.0
        a = me.attributes.new("_LIGHT", "FLOAT", "POINT")
        a.data.foreach_set("value", light)
        t = me.attributes.new("_TINT", "FLOAT", "POINT")
        t.data.foreach_set("value", tint)
        # linear base color per vertex, so a client can merge all primitives into ONE draw call
        alb = np.zeros((n, 3), dtype=np.float32)
        for poly in me.polygons:
            spec = mlib.MATERIALS.get(mats[poly.material_index])
            if spec is not None:
                c = mlib.hex_to_linear(spec.hex)
                for vi in poly.vertices:
                    alb[vi] = c
        al = me.attributes.new("_ALBEDO", "FLOAT_VECTOR", "POINT")
        al.data.foreach_set("vector", alb.ravel())
    return obj


def set_custom_normals(obj: bpy.types.Object, normals: np.ndarray) -> None:
    """Per-vertex custom normals (zero vectors keep the automatic normal). Exported as glTF NORMAL."""
    me = obj.data
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    n = np.asarray(normals, dtype=np.float64)
    if len(n) != len(me.vertices):
        raise ValueError(f"{obj.name}: {len(n)} normals for {len(me.vertices)} vertices")
    me.normals_split_custom_set_from_vertices([tuple(v) for v in n])


def tri_count(objs: list[bpy.types.Object]) -> int:
    n = 0
    for o in objs:
        for p in o.data.polygons:
            n += len(p.vertices) - 2
    return n


def export_glb(objs: list[bpy.types.Object], path: Path, attributes: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_vertex_color="ACTIVE",
        export_attributes=attributes,
        export_cameras=False,
        export_lights=False,
        export_extras=False,
        export_animations=False,
        export_draco_mesh_compression_enable=False,
        export_image_format="AUTO",
    )


# ---------------------------------------------------------------------------
# preview rendering (Cycles CPU; EEVEE needs a GPU context that headless bpy lacks)
# ---------------------------------------------------------------------------


def setup_render(width: int, height: int, samples: int = 48) -> bpy.types.Scene:
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.03
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    sc.cycles.max_bounces = 6
    sc.cycles.transparent_max_bounces = 16
    sc.render.resolution_x = width
    sc.render.resolution_y = height
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.compression = 100
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        sc.view_settings.view_transform = "Filmic"
    sc.view_settings.exposure = -0.45
    return sc


def setup_world(sun_elev_deg: float = 32.0, sun_azimuth_deg: float = 125.0, strength: float = 5.5,
                sky_strength: float = 0.07) -> None:
    """SoCal morning light: warm low sun from the south-east plus a hazy physical sky."""
    sc = bpy.context.scene
    world = bpy.data.worlds.new("world")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    sky = nt.nodes.new("ShaderNodeTexSky")
    for t in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "NISHITA", "HOSEK_WILKIE"):
        try:
            sky.sky_type = t
            break
        except TypeError:
            continue
    if hasattr(sky, "sun_elevation"):
        sky.sun_elevation = math.radians(sun_elev_deg)
        sky.sun_rotation = math.radians(sun_azimuth_deg)
        if hasattr(sky, "sun_disc"):
            sky.sun_disc = False
        if hasattr(sky, "air_density"):
            sky.air_density = 1.2
        if hasattr(sky, "aerosol_density"):
            sky.aerosol_density = 2.0
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = sky_strength
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = strength
    sun.angle = math.radians(1.5)
    sun.color = (1.0, 0.94, 0.86)
    so = bpy.data.objects.new("sun", sun)
    sc.collection.objects.link(so)
    # Blender sun points down -Z; tilt by (90 - elevation) and spin to azimuth (from north, clockwise).
    so.rotation_euler = (math.radians(90 - sun_elev_deg), 0.0, math.radians(180 - sun_azimuth_deg))


def add_camera(location: tuple[float, float, float], target: tuple[float, float, float], lens: float = 50.0,
               ortho_scale: float | None = None) -> bpy.types.Object:
    from mathutils import Vector

    sc = bpy.context.scene
    cam = bpy.data.cameras.new("cam")
    cam.lens = lens
    cam.clip_start = 0.1
    cam.clip_end = 5000
    if ortho_scale:
        cam.type = "ORTHO"
        cam.ortho_scale = ortho_scale
    co = bpy.data.objects.new("cam", cam)
    sc.collection.objects.link(co)
    co.location = location
    d = Vector(target) - Vector(location)
    co.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    sc.camera = co
    return co


def simple_material(name: str, hex_: str, roughness: float = 0.9, metallic: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    b = _bsdf(mat)
    b.inputs["Base Color"].default_value = (*mlib.hex_to_linear(hex_), 1.0)
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Metallic"].default_value = metallic
    return mat


def ground_plane(size: float, hex_: str = "#55585C", z: float = 0.0, roughness: float = 0.9) -> bpy.types.Object:
    """Preview-only ground (asphalt by default)."""
    me = bpy.data.meshes.new("ground")
    s = size
    me.from_pydata([(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)], [], [[0, 1, 2, 3]])
    me.materials.append(simple_material(f"ground_{hex_}", hex_, roughness))
    o = bpy.data.objects.new("ground", me)
    bpy.context.scene.collection.objects.link(o)
    return o


def import_glb(path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [o for o in bpy.data.objects if o not in before]


def render(path: Path, max_kb: int = 400) -> None:
    """Render to PNG; re-encode as a palettized PNG if it exceeds max_kb (previews are committed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    if path.stat().st_size > max_kb * 1024:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        for colors in (256, 192, 128):
            q = im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
            q.save(path, optimize=True)
            if path.stat().st_size <= max_kb * 1024:
                break
        else:
            w, h = im.size
            im.resize((int(w * 0.8), int(h * 0.8)), Image.LANCZOS).quantize(colors=128).save(path, optimize=True)
