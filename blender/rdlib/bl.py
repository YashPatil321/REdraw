"""Blender (bpy) side: palette material, Part -> object, glTF export, preview renders."""

from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np

from . import palette
from .mesh import Part

REPO = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO / "blender" / "build"  # palette textures (regenerated, not committed)
MATERIAL_NAME = "redraw_palette"


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.objects):
        for block in list(coll):
            coll.remove(block)


def palette_material() -> bpy.types.Material:
    mat = bpy.data.materials.get(MATERIAL_NAME)
    if mat is not None:
        return mat
    base_png, mr_png = palette.write_textures(BUILD_DIR)
    mat = bpy.data.materials.new(MATERIAL_NAME)
    mat.use_nodes = True
    mat.use_backface_culling = False  # glTF doubleSided: fronds, sails, awnings are single sheets
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    img_b = nt.nodes.new("ShaderNodeTexImage")
    img_b.image = bpy.data.images.load(str(base_png), check_existing=True)
    img_b.image.colorspace_settings.name = "sRGB"
    img_b.interpolation = "Closest"
    img_mr = nt.nodes.new("ShaderNodeTexImage")
    img_mr.image = bpy.data.images.load(str(mr_png), check_existing=True)
    img_mr.image.colorspace_settings.name = "Non-Color"
    img_mr.interpolation = "Closest"
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(img_b.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(img_mr.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.5
    return mat


def part_to_object(part: Part, name: str, collection: bpy.types.Collection | None = None) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    me.from_pydata(part.V.tolist(), [], part.F)
    me.validate(clean_customdata=False)
    if len(me.polygons) != len(part.F):
        raise RuntimeError(f"{name}: mesh.validate removed {len(part.F) - len(me.polygons)} degenerate faces")
    me.materials.append(palette_material())
    uv = me.uv_layers.new(name="UVMap")
    cache: dict[str, tuple[float, float]] = {}
    uvs = np.zeros((len(me.loops), 2), dtype=np.float32)
    for poly, key in zip(me.polygons, part.M, strict=True):
        if key not in cache:
            cache[key] = palette.cell_uv(key)
        uvs[poly.loop_start : poly.loop_start + poly.loop_total] = cache[key]
    uv.data.foreach_set("uv", uvs.ravel())
    me.polygons.foreach_set("use_smooth", [False] * len(me.polygons))
    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def export_glb(objs: list[bpy.types.Object], path: Path, draco: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    kw = dict(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_cameras=False,
        export_lights=False,
        export_extras=False,
        export_animations=False,
        export_draco_mesh_compression_enable=draco,
    )
    if draco:
        kw.update(export_draco_mesh_compression_level=6, export_draco_position_quantization=14,
                  export_draco_normal_quantization=8, export_draco_texcoord_quantization=12)
    bpy.ops.export_scene.gltf(**kw)


# ---------------------------------------------------------------------------
# preview rendering (Cycles CPU; EEVEE needs a GPU context that headless bpy lacks)
# ---------------------------------------------------------------------------


def setup_render(width: int, height: int, samples: int = 64) -> bpy.types.Scene:
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
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
    sc.view_settings.exposure = 0.0
    return sc


def setup_world(sun_elev_deg: float = 32.0, sun_azimuth_deg: float = 125.0, strength: float = 3.2) -> None:
    """Morning SoCal light: warm low sun from the south-east plus a soft sky."""
    sc = bpy.context.scene
    world = bpy.data.worlds.new("world")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    sky = nt.nodes.new("ShaderNodeTexSky")
    ok = False
    for t in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "NISHITA", "HOSEK_WILKIE"):
        try:
            sky.sky_type = t
            ok = True
            break
        except TypeError:
            continue
    if ok and hasattr(sky, "sun_elevation"):
        sky.sun_elevation = math.radians(sun_elev_deg)
        sky.sun_rotation = math.radians(sun_azimuth_deg)
        if hasattr(sky, "sun_disc"):
            sky.sun_disc = False
        if hasattr(sky, "air_density"):
            sky.air_density = 1.2
        if hasattr(sky, "aerosol_density"):
            sky.aerosol_density = 1.8
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.22
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = strength
    sun.angle = math.radians(2.5)
    sun.color = (1.0, 0.93, 0.83)
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


def ground_plane(size: float, key: str, z: float = -0.02) -> bpy.types.Object:
    from .mesh import quad

    return part_to_object(quad(-size, -size, size, size, z, key), "ground")


def render(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
