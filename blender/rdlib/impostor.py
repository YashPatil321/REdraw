"""LOD1 impostors for trees: Cycles-baked side + top views on crossed quads (8 triangles).

The LOD0 plant is rendered orthographically under a uniform white sky (Standard view
transform, no sun), so the texture holds albedo x ambient occlusion and the client's sun
lighting stays consistent with LOD0. Atlas: 2*res x res RGBA, left square = side view
(aspect-fit, centered), right square = top view. Geometry: three vertical quads at 0/60/120
degrees through the trunk plus one horizontal quad at crown height, normals bent outward from
the crown center like the LOD0 cards.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np

from .mesh import Part


def _bounds(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([vv.co[:] for vv in obj.data.vertices])
    return v.min(axis=0), v.max(axis=0)


def _setup(res_x: int, res_y: int, samples: int) -> bpy.types.Scene:
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    sc.cycles.transparent_max_bounces = 48
    sc.render.film_transparent = True
    sc.render.resolution_x = res_x
    sc.render.resolution_y = res_y
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    world = bpy.data.worlds.get("impostor_world") or bpy.data.worlds.new("impostor_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    sc.world = world
    return sc


def _camera(loc: tuple[float, float, float], rot: tuple[float, float, float], scale: float) -> bpy.types.Object:
    cam = bpy.data.cameras.new("imp_cam")
    cam.type = "ORTHO"
    cam.ortho_scale = scale
    cam.clip_start = 0.1
    cam.clip_end = 1000
    co = bpy.data.objects.new("imp_cam", cam)
    bpy.context.scene.collection.objects.link(co)
    co.location = loc
    co.rotation_euler = rot
    bpy.context.scene.camera = co
    return co


def bake(obj: bpy.types.Object, out_png: Path, res: int = 512, samples: int = 32) -> dict:
    """Render side + top views of `obj` (Blender +Z up, trunk at the origin) into one RGBA atlas."""
    from PIL import Image

    lo, hi = _bounds(obj)
    half_w = float(max(np.abs(lo[:2]).max(), np.abs(hi[:2]).max())) * 1.02
    zb = min(float(lo[2]), 0.0)
    H = float(hi[2]) - zb + 0.02
    W = 2 * half_w
    tmp = out_png.with_suffix(".side.png")
    # side view (camera on -Y looking +Y)
    if H >= W:
        rx, ry = max(8, int(round(res * W / H))), res
    else:
        rx, ry = res, max(8, int(round(res * H / W)))
    sc = _setup(rx, ry, samples)
    cam = _camera((0.0, -200.0, zb + H / 2), (math.radians(90), 0, 0), max(W, H))
    sc.render.filepath = str(tmp)
    bpy.ops.render.render(write_still=True)
    side = Image.open(tmp).convert("RGBA")
    bpy.data.objects.remove(cam)
    # top view
    sc = _setup(res, res, samples)
    cam = _camera((0.0, 0.0, float(hi[2]) + 50.0), (0, 0, 0), W)
    sc.render.filepath = str(tmp)
    bpy.ops.render.render(write_still=True)
    top = Image.open(tmp).convert("RGBA")
    bpy.data.objects.remove(cam)
    tmp.unlink()
    atlas = Image.new("RGBA", (2 * res, res), (0, 0, 0, 0))
    ox, oy = (res - rx) // 2, (res - ry) // 2
    atlas.paste(side, (ox, oy))
    atlas.paste(top, (res, 0))
    atlas = _bleed(atlas)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    atlas.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(out_png, optimize=True)
    T = 2.0 * res
    side_uv = ((ox + 0.5) / T, 1 - (oy + ry - 0.5) / res, (ox + rx - 0.5) / T, 1 - (oy + 0.5) / res)
    top_uv = ((res + 0.5) / T, 0.5 / res, (2 * res - 0.5) / T, 1 - 0.5 / res)
    return {"W": W, "H": H, "zb": zb, "side_uv": side_uv, "top_uv": top_uv, "res": res}


def _bleed(im):  # noqa: ANN001, ANN202
    """Fill transparent texels with nearby leaf colors (no dark halos under mipmapping / alpha test)."""
    from PIL import Image, ImageFilter

    a = np.asarray(im).astype(np.float32)
    rgb, al = a[..., :3], a[..., 3:4] / 255.0
    prem = Image.fromarray(np.clip(rgb * al, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))
    wgt = Image.fromarray(np.clip(al[..., 0] * 255, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))
    fill = np.asarray(prem).astype(np.float32) / np.maximum(np.asarray(wgt).astype(np.float32)[..., None] / 255.0, 1e-3)
    rgb = np.where(al > 0.02, rgb, np.clip(fill, 0, 255))
    return Image.fromarray(np.concatenate([rgb, a[..., 3:4]], axis=-1).astype(np.uint8), "RGBA")


def lod1_part(info: dict, crown_z: float) -> tuple[Part, np.ndarray]:
    """Crossed quads (3 vertical + 1 horizontal) with bent normals; material key 'foliage_impostor'.
    Every quad is doubled back to back (single-sided material): a double-sided quad would flip the
    bent normal on its back face and shade half the impostor black."""
    W, H, zb = info["W"], info["H"], info["zb"]
    su0, sv0, su1, sv1 = info["side_uv"]
    tu0, tv0, tu1, tv1 = info["top_uv"]
    p = Part()
    for k in range(3):
        a = math.pi * k / 3
        dx, dy = math.cos(a) * W / 2, math.sin(a) * W / 2
        V = np.array([(-dx, -dy, zb), (dx, dy, zb), (dx, dy, zb + H), (-dx, -dy, zb + H)])
        uv = [(su0, sv0), (su1, sv0), (su1, sv1), (su0, sv1)]
        p += Part(V, [[0, 1, 2, 3]], ["foliage_impostor"], [uv])
        p += Part(V[[1, 0, 3, 2]].copy(), [[0, 1, 2, 3]], ["foliage_impostor"], [[uv[1], uv[0], uv[3], uv[2]]])
    h = W / 2
    V = np.array([(-h, -h, crown_z), (h, -h, crown_z), (h, h, crown_z), (-h, h, crown_z)])
    uv = [(tu0, tv0), (tu1, tv0), (tu1, tv1), (tu0, tv1)]
    p += Part(V, [[0, 1, 2, 3]], ["foliage_impostor"], [uv])
    p += Part(V[[3, 2, 1, 0]].copy(), [[0, 1, 2, 3]], ["foliage_impostor"], [[uv[3], uv[2], uv[1], uv[0]]])
    c = np.array([0.0, 0.0, crown_z])
    N = []
    for v in p.V:
        r = v - c
        r[2] *= 0.7
        r = r / (np.linalg.norm(r) + 1e-9)
        m = 0.65 * r + 0.35 * np.array([0, 0, 1.0])
        N.append(m / np.linalg.norm(m))
    return p, np.array(N)
