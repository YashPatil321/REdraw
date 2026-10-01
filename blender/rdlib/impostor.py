"""LOD1 impostors for trees: Cycles-baked side + top views on crossed quads (8 triangles).

The LOD0 plant is rendered orthographically under a uniform white sky (Standard view
transform, no sun), so the texture holds albedo x ambient occlusion and the client's sun
lighting stays consistent with LOD0. Atlas: 2*res x res RGBA, left square = side view
(aspect-fit, centered), right square = top view. Geometry: three vertical quads at 0/60/120
degrees through the trunk plus one horizontal quad at crown height, each doubled back to back and
split into a few cells carrying ellipsoid-crown normals (see lod1_part).
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


def lod1_part(info: dict, crown_z: float, top_z: float | None = None, cols: int = 2, rows: int = 3) -> tuple[Part, np.ndarray]:
    """Crossed quads (3 vertical + 1 horizontal) with sphere normals; material key 'foliage_impostor'.

    Every quad is doubled back to back (single-sided material) so each side carries its own normal
    field: the normal of an ellipsoidal crown seen from that side (lateral offset -> sideways,
    height -> up / down, the rest toward the viewer). Quads are split into cols x rows cells so the
    field interpolates smoothly. The impostor then shades like the LOD0 card cloud from any view and
    sun direction. The horizontal top-view quad sits high in the crown (top_z), is 80 % of the crown
    width (hidden inside the side silhouettes when seen edge-on) and faces up only, so from street
    level (backfaces culled) it never shows as a dark plate. 80 triangles.

    Real-time use: let impostors cast shadows but not receive them (crossed opaque silhouettes would
    shadow each other into dark quadrants)."""
    W, H, zb = info["W"], info["H"], info["zb"]
    su0, sv0, su1, sv1 = info["side_uv"]
    tu0, tv0, tu1, tv1 = info["top_uv"]
    top = zb + H
    hz = max(0.25 * H, top - crown_z)  # crown half height (ellipsoid around crown_z)
    up = np.array([0.0, 0.0, 1.0])
    V: list[np.ndarray] = []
    N: list[np.ndarray] = []
    F: list[list[int]] = []
    U: list[list[tuple[float, float]]] = []

    def grid(origin: np.ndarray, ax_u: np.ndarray, ax_v: np.ndarray, uv_rect: tuple[float, float, float, float],
             nrm_fn, nu: int, nv: int, flip: bool) -> None:  # noqa: ANN001
        u0, v0, u1, v1 = uv_rect
        base = len(V)
        for j in range(nv + 1):
            for i in range(nu + 1):
                a, b = i / nu, j / nv
                V.append(origin + ax_u * a + ax_v * b)
                N.append(nrm_fn(a, b))
        for j in range(nv):
            for i in range(nu):
                q = [base + j * (nu + 1) + i, base + j * (nu + 1) + i + 1, base + (j + 1) * (nu + 1) + i + 1,
                     base + (j + 1) * (nu + 1) + i]
                uv = [(u0 + (u1 - u0) * (i + di) / nu, v0 + (v1 - v0) * (j + dj) / nv) for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1))]
                if flip:
                    q, uv = q[::-1], uv[::-1]
                F.append(q)
                U.append(uv)

    for k in range(3):
        ang = math.pi * k / 3
        ax = np.array([math.cos(ang), math.sin(ang), 0.0])
        face = np.array([-math.sin(ang), math.cos(ang), 0.0])
        for side in (1.0, -1.0):
            def nf(a: float, b: float, ax: np.ndarray = ax, face: np.ndarray = face, side: float = side) -> np.ndarray:
                uu = 2.0 * a - 1.0
                z = zb + H * b
                vv = float(np.clip((z - crown_z) / hz, -0.45, 1.0))
                w = math.sqrt(max(0.08, 1.0 - uu * uu - vv * vv))
                n = uu * ax + vv * up + w * side * face
                n = n / np.linalg.norm(n) + 0.25 * up
                return n / np.linalg.norm(n)

            # unflipped cells wind along ax then up: geometric normal ax x up = -face, so the +face copy flips
            grid(np.array([0.0, 0.0, zb]) - ax * W / 2, ax * W, up * H, (su0, sv0, su1, sv1), nf, cols, rows, side > 0)
    h = 0.4 * W
    zt = crown_z if top_z is None else top_z
    du, dv = (tu1 - tu0) * 0.1, (tv1 - tv0) * 0.1  # inner 80 % of the top view

    def nt(a: float, b: float) -> np.ndarray:
        n = np.array([2 * a - 1, 2 * b - 1, 0.0]) * 0.45 + up
        return n / np.linalg.norm(n)

    grid(np.array([-h, -h, zt]), np.array([2 * h, 0.0, 0.0]), np.array([0.0, 2 * h, 0.0]),
         (tu0 + du, tv0 + dv, tu1 - du, tv1 - dv), nt, 2, 2, False)
    p = Part(np.array(V, float), F, ["foliage_impostor"] * len(F), U)
    return p, np.array(N)
