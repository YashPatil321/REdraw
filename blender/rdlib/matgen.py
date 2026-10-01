"""PBR material atlases for Redraw: facades, roofs, ground (tileable, SoCal suburban look).

Pure numpy + PIL (texgen.py), run as the `materials` stage of blender/build_all_assets.py
(the Blender venv); the Cycles previews in build_all_assets.py render them on real geometry.

Every atlas is a grid of 512 px cells. A cell holds a 480 px TILEABLE inner square plus a
16 px gutter of wrapped content on each side, so a shader can do
    atlas_uv = cell.uv_inner.xy + fract(mesh_uv) * (cell.uv_inner.zw - cell.uv_inner.xy)
with bilinear filtering and the first few mip levels without seams.

Outputs (client/public/assets/materials/, regenerated, not committed):
    <atlas>_albedo.jpg   sRGB base color
    <atlas>_normal.jpg   tangent-space normal, OpenGL/glTF convention (+Y up)
    <atlas>_orm.jpg      R ambient occlusion, G roughness, B metalness (glTF packing)
    facade_openings_mask.png   R wall-tint mask (1 wall, 0.5 paintable, 0 fixed), G night-window light mask
    ground_markings.png  RGBA lane-marking decals (alpha = paint coverage with wear)
    *_1k.jpg             half-resolution copies (low quality preset / far LOD)
    materials_manifest.json    layout, conventions, per-cell metadata (see blender/README.md)
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import texgen as tg
from .texgen import Canvas, hex_rgb, smoothstep

CELL = 512
GUTTER = 16
INNER = CELL - 2 * GUTTER  # 480

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

STUCCO_GRAY = 0.80  # neutral sRGB level of tintable stucco (the shader multiplies the wall color)


def _palette_pick(pal: list[str], r: np.ndarray) -> np.ndarray:
    cols = np.array([hex_rgb(c) for c in pal])
    idx = np.clip((r * len(pal)).astype(int), 0, len(pal) - 1)
    return cols[idx]


def stucco(cv: Canvas, kind: str, seed: int) -> None:
    """Neutral stucco over the whole canvas (mask = 1, wall). Height in meters (sub-millimeter relief)."""
    sh = cv.shape
    g = tg.fbm(sh, seed + 1, beta=0.4, fmin=1)
    med = cv.noise(seed + 3, 0.6, beta=2.2, octaves_m=0.03)
    mottle = cv.noise(seed + 4, 2.5, beta=2.6, octaves_m=0.15)
    h = np.zeros(sh)
    alb = np.full(sh, STUCCO_GRAY)
    rough = np.full(sh, 0.88)
    if kind == "smooth":  # Santa Barbara smooth: burnished trowel arcs
        n = cv.noise(seed + 5, 0.9, beta=2.4, octaves_m=0.05)
        arcs = np.abs(np.sin(n * 5.0))
        h = 0.00025 * arcs + 0.00008 * g + 0.0002 * med
        burn = tg.smoothstep(0.75, 1.0, arcs)
        alb = alb * (1 - 0.035 * burn + 0.02 * med * 0.5) + 0.01 * g
        rough = 0.8 - 0.12 * burn
    elif kind == "sand":  # 20/30 sand finish (the common SoCal tract finish)
        grain = tg.blur(tg.fbm(sh, seed + 6, beta=0.0, fmin=1), 0.6)
        grain /= grain.std()
        h = 0.00035 * grain + 0.0002 * med
        alb = alb * (1 + 0.035 * grain * 0.5 + 0.015 * med)
        rough = 0.9 + 0.04 * grain * 0.3
    elif kind == "lace":  # skip trowel / lace: flattened raised islands
        n = cv.noise(seed + 7, 0.18, beta=2.0, octaves_m=0.012)
        isl = tg.smoothstep(0.15, 0.55, n)
        grain = tg.blur(tg.fbm(sh, seed + 8, beta=0.0, fmin=1), 0.6)
        grain /= grain.std()
        h = 0.0016 * isl + 0.00025 * grain * (1 - isl) + 0.00006 * g
        alb = alb * (1 + 0.03 * isl + 0.025 * grain * 0.5 * (1 - isl))
        rough = 0.92 - 0.12 * isl
    elif kind == "catface":  # smooth with rough recessed "cat faces"
        n = cv.noise(seed + 9, 0.35, beta=2.2, octaves_m=0.02)
        face = tg.smoothstep(0.9, 1.3, n)
        grain = tg.blur(tg.fbm(sh, seed + 10, beta=0.0, fmin=1), 0.7)
        grain /= grain.std()
        h = -0.0012 * face + 0.0003 * grain * face + 0.0001 * med
        alb = alb * (1 - 0.06 * face + 0.03 * grain * 0.5 * face + 0.015 * med)
        rough = 0.78 + 0.15 * face
    elif kind in ("weathered", "scored"):
        grain = tg.blur(tg.fbm(sh, seed + 11, beta=0.0, fmin=1), 0.6)
        grain /= grain.std()
        h = 0.00035 * grain + 0.0002 * med
        alb = alb * (1 + 0.03 * grain * 0.5 + 0.015 * med)
        rough = 0.9 + 0.03 * grain * 0.3
        if kind == "weathered":
            blotch = cv.noise(seed + 12, 1.5, beta=2.8, octaves_m=0.1)
            alb = alb * (1 - 0.06 * tg.smoothstep(0.3, 1.6, blotch) + 0.03 * tg.smoothstep(0.5, 1.8, -blotch))
            c = tg.worley(sh, (5, 5), seed + 13)
            crack = tg.smoothstep(2.0, 0.4, c.f2 - c.f1) * tg.smoothstep(0.6, 1.2, cv.noise(seed + 14, 1.0, octaves_m=0.05))
            h = h - 0.0008 * crack
            alb = alb * (1 - 0.18 * crack)
            spots = tg.smoothstep(2.6, 3.4, cv.noise(seed + 15, 0.08, beta=1.5))
            alb = alb * (1 - 0.05 * spots)
        else:  # reveals: horizontal at each floor line, vertical control joint at each bay line
            dh = np.minimum(np.abs(cv.Y - np.round(cv.Y / cv.h_m) * cv.h_m), 99)
            dv = np.minimum(np.abs(cv.X - np.round(cv.X / cv.w_m) * cv.w_m), 99)
            rev = np.maximum(tg.smoothstep(0.016, 0.009, dh), tg.smoothstep(0.012, 0.006, dv))
            h = h - 0.012 * rev
            alb = alb * (1 - 0.1 * rev)
    alb = alb * (1 + 0.012 * mottle)
    cv.alb = np.repeat(alb[..., None], 3, axis=-1)
    cv.h = h
    cv.rough = np.clip(rough, 0.5, 1.0)
    cv.metal[:] = 0
    cv.mask[:] = 1.0


def glass(cv: Canvas, cov: np.ndarray, x0: float, y0: float, x1: float, y1: float, seed: int, h: float = -0.07,
          interior: str | None = None, blind: float = 0.0, light: float = 1.0) -> None:
    """Dark tinted double glazing with a subtle baked sky / tree reflection and optional interior hints."""
    rng = np.random.default_rng(seed)
    t = np.clip((cv.Y - y0) / max(y1 - y0, 1e-6), 0, 1)
    s = np.clip((cv.X - x0) / max(x1 - x0, 1e-6), 0, 1)
    sky = hex_rgb("#26303A") * (1 - t[..., None]) + hex_rgb("#3A4856") * t[..., None]
    ph = rng.uniform(0, 6, 3)
    sil = 0.3 + 0.07 * np.sin(s * 5.0 + ph[0]) + 0.04 * np.sin(s * 13.0 + ph[1]) + 0.025 * np.sin(s * 31.0 + ph[2])
    trees = smoothstep(0.06, -0.06, t - sil)
    col = sky * (1 - 0.45 * trees[..., None]) + hex_rgb("#1E2420") * 0.45 * trees[..., None]
    sheen = np.exp(-(((s - (1 - t) * 0.5) - rng.uniform(0.2, 0.7)) / 0.18) ** 2) * 0.035
    col = col + sheen[..., None]
    if interior == "blinds" and blind > 0:  # white horizontal mini-blinds lowered from the head
        slat = 0.5 + 0.5 * np.cos(2 * np.pi * cv.Y / 0.025)
        down = smoothstep(1 - blind - 0.01, 1 - blind + 0.01, t)
        col = col * (1 - 0.55 * down[..., None]) + (hex_rgb("#C9C6BC") * (0.7 + 0.3 * slat[..., None])) * 0.55 * down[..., None]
    elif interior == "vblinds" and blind > 0:  # vertical blinds (patio sliders)
        slat = 0.5 + 0.5 * np.cos(2 * np.pi * cv.X / 0.09)
        drawn = smoothstep(blind + 0.01, blind - 0.01, s)
        col = col * (1 - 0.6 * drawn[..., None]) + hex_rgb("#D3CCBC") * (0.75 + 0.25 * slat[..., None]) * 0.6 * drawn[..., None]
    elif interior == "curtain":
        fold = 0.5 + 0.5 * np.sin(2 * np.pi * cv.X / 0.12 + 0.6 * np.sin(cv.Y * 3))
        side = np.maximum(smoothstep(0.22, 0.18, s), smoothstep(0.78, 0.82, s))
        col = col * (1 - 0.5 * side[..., None]) + hex_rgb("#B8A88E") * (0.6 + 0.4 * fold[..., None]) * 0.5 * side[..., None]
    elif interior == "store":  # lit retail interior: ceiling, shelving, floor; seen through 35 % reflection
        ncol = max(2, int((x1 - x0) / 0.35))
        tab = rng.uniform(0.25, 0.7, (ncol + 1, 3)) * np.array([1.0, 0.92, 0.8])
        k = np.clip(((cv.X - x0) / (x1 - x0) * ncol).astype(int), 0, ncol)
        shelf = tg.blur(tab[k] * smoothstep(0.08, 0.14, t)[..., None] * smoothstep(0.7, 0.62, t)[..., None], 2.0)
        rows_ = 0.85 + 0.15 * (np.cos(2 * np.pi * cv.Y / 0.45) > 0.6)
        ceil = smoothstep(0.72, 0.8, t)[..., None] * hex_rgb("#D8CDB6") * 0.75
        lamp = smoothstep(0.5, 0.0, np.abs(((cv.X - x0) % 1.2) - 0.6)) * smoothstep(0.84, 0.88, t) * smoothstep(0.96, 0.92, t)
        floor = smoothstep(0.1, 0.0, t)[..., None] * hex_rgb("#6E675C")
        back = hex_rgb("#8E8576") * 0.55
        inside = back + shelf * rows_[..., None] * 0.8 + ceil + floor * 0.5 + lamp[..., None] * 0.5
        col = col * 0.35 + np.clip(inside, 0, 1) * 0.65 * light
    elif interior == "classroom":
        q = tg.norm01(tg.blur(rng.random((cv.py, cv.px)), (10, 8)))
        posters = smoothstep(0.62, 0.66, q) * smoothstep(0.35, 0.45, t) * smoothstep(0.8, 0.7, t)
        col = col * 0.7 + hex_rgb("#C9C2B0") * 0.12 + posters[..., None] * np.array([0.35, 0.22, 0.12])
    cv.put(cov, color=None, h=h, rough=0.05, metal=0.0, mask=0.0, emit=1.0 if interior else 0.6)
    cv.alb = cv.alb * (1 - cov[..., None]) + np.clip(col, 0, 1) * cov[..., None]


def frame(cv: Canvas, x0: float, y0: float, x1: float, y1: float, w: float, color: str = "#ECEBE6",
          h: float = -0.035, rough: float = 0.45, metal: float = 0.0, mask: float = 0.0) -> None:
    outer = cv.rect(x0, y0, x1, y1)
    inner = cv.rect(x0 + w, y0 + w, x1 - w, y1 - w)
    ring = np.clip(outer - inner, 0, 1)
    prof = cv.bevel(x0, y0, x1, y1, w * 0.5) * (1 - cv.bevel(x0 + w * 0.5, y0 + w * 0.5, x1 - w * 0.5, y1 - w * 0.5, w * 0.5))
    cv.put(ring, color=color, h=h + 0.006 * prof, rough=rough, metal=metal, mask=mask, emit=0.0)


def bar(cv: Canvas, x0: float, y0: float, x1: float, y1: float, color: str = "#ECEBE6", h: float = -0.04,
        rough: float = 0.45, metal: float = 0.0, mask: float = 0.0) -> None:
    cv.put(cv.rect(x0, y0, x1, y1), color=color, h=h + 0.004 * cv.bevel(x0, y0, x1, y1, 0.01), rough=rough, metal=metal,
           mask=mask, emit=0.0)


def recess_shadow(cv: Canvas, x0: float, y0: float, x1: float, y1: float, depth: float = 0.6) -> None:
    """Sky-occlusion falloff inside an opening (top and sides darker): extra AO the heightfield AO misses."""
    d_top = np.clip((y1 - cv.Y) / 0.25, 0, 1)
    d_side = np.clip(np.minimum(cv.X - x0, x1 - cv.X) / 0.12, 0, 1)
    occ = 1 - depth * (1 - d_top * 0.6 - 0.4) * 0.5 - depth * 0.25 * (1 - d_side)
    cv.ao = cv.ao * np.where(cv.rect(x0, y0, x1, y1) > 0.5, np.clip(occ, 0.3, 1), 1.0)


def foam_trim(cv: Canvas, x0: float, y0: float, x1: float, y1: float, w: float = 0.1, proj: float = 0.025,
              sill: bool = True) -> None:
    """Stucco-wrapped foam surround (very common on SoCal tract windows); keeps the wall tint (mask 1)."""
    outer = cv.rect(x0 - w, y0 - w, x1 + w, y1 + w)
    prof = cv.bevel(x0 - w, y0 - w, x1 + w, y1 + w, 0.015)
    cv.h = cv.h + proj * prof * outer
    if sill:
        s = cv.rect(x0 - w - 0.04, y0 - w - 0.05, x1 + w + 0.04, y0 - w + 0.02)
        cv.h = np.maximum(cv.h, (proj + 0.02) * cv.bevel(x0 - w - 0.04, y0 - w - 0.05, x1 + w + 0.04, y0 - w + 0.02, 0.01) * s)


def coach_light(cv: Canvas, cx: float, cy: float, s: float = 1.0) -> None:
    w, hh = 0.2 * s, 0.36 * s
    cv.put(cv.rect(cx - 0.05, cy - hh * 0.62, cx + 0.05, cy + hh * 0.55), color="#1C1C1C", h=0.03, rough=0.5, metal=0.6, mask=0.0)
    cv.put(cv.rect(cx - w / 2, cy - hh / 2, cx + w / 2, cy + hh / 2), color="#2A2620", h=0.1, rough=0.5, metal=0.5, mask=0.0)
    g = cv.rect(cx - w / 2 + 0.025, cy - hh / 2 + 0.04, cx + w / 2 - 0.025, cy + hh / 2 - 0.05)
    cv.put(g, color="#D8C9A0", h=0.1, rough=0.15, mask=0.0, emit=1.0)
    cv.put(cv.rect(cx - w / 2 - 0.02, cy + hh / 2 - 0.02, cx + w / 2 + 0.02, cy + hh / 2 + 0.03), color="#1A1A1A", h=0.12, metal=0.6, rough=0.45)


def grime_bottom(cv: Canvas, x0: float, x1: float, y0: float, height: float, amt: float = 0.15) -> None:
    n = cv.noise(77, 0.3, octaves_m=0.02)
    g = cv.vband(x0, x1) * np.clip(1 - (cv.Y - y0) / height, 0, 1) ** 1.5 * (cv.Y >= y0)
    cv.alb = cv.alb * (1 - amt * np.clip(g * (0.8 + 0.3 * n), 0, 1))[..., None]


# ---------------------------------------------------------------------------
# facade: walls (tileable 3 m x 3 m)
# ---------------------------------------------------------------------------


def wall_stucco(kind: str, seed: int) -> Callable[[Canvas], None]:
    def f(cv: Canvas) -> None:
        stucco(cv, kind, seed)
    return f


def wall_stone(cv: Canvas) -> None:
    """Stacked ledgestone veneer (wainscots, columns, monument walls)."""
    rng = np.random.default_rng(41)
    pal = ["#B9A88C", "#A38F74", "#C8B89B", "#8E7C66", "#B49A78", "#9C8B7B", "#A9916F", "#7E7062"]
    H, W = cv.shape
    sid = np.zeros((H, W), int)
    edge = np.zeros((H, W))
    y = 0.0
    k = 0
    while y < cv.h_m - 1e-6:
        rh = float(rng.uniform(0.045, 0.13))
        if cv.h_m - y - rh < 0.05:
            rh = cv.h_m - y
        rows = (cv.Y >= y) & (cv.Y < y + rh)
        x = float(rng.uniform(0, 0.3))
        starts = [x]
        while True:
            x += float(rng.uniform(0.18, 0.62))
            if x >= starts[0] + cv.w_m - 0.12:
                break
            starts.append(x)
        st = np.array(starts)
        X = (cv.X - st[0]) % cv.w_m + st[0]
        j = np.searchsorted(st, X, side="right") - 1
        x0 = st[j]
        x1 = np.where(j + 1 < len(st), st[np.minimum(j + 1, len(st) - 1)], st[0] + cv.w_m)
        de = np.minimum(np.minimum(X - x0, x1 - X), np.minimum(cv.Y - y, y + rh - cv.Y))
        sid = np.where(rows, k * 100 + j, sid)
        edge = np.where(rows, de, edge)
        y += rh
        k += 1
    r = (np.sin(sid * 12.9898) * 43758.5453) % 1.0
    r2 = (np.sin(sid * 78.233) * 12543.123) % 1.0
    col = _palette_pick(pal, r)
    n = cv.noise(42, 0.15, beta=1.6, octaves_m=0.004)
    gap = tg.smoothstep(0.004, 0.012 + 0.004 * np.clip(n, -1, 2), edge)
    proj = 0.012 + 0.03 * r2
    hh = proj * gap + 0.004 * n * gap + 0.006 * cv.noise(43, 0.4, octaves_m=0.02)
    cv.alb = col * (0.85 + 0.12 * n[..., None] * 0.4 + 0.1 * (r2[..., None] - 0.5)) * gap[..., None] + hex_rgb("#2E2822") * (1 - gap[..., None])
    cv.h = hh
    cv.rough = 0.92 - 0.05 * gap
    cv.metal[:] = 0
    cv.mask[:] = 0
    cv.ao = 0.55 + 0.45 * gap


def wall_glass_curtain(cv: Canvas) -> None:
    """Storefront / curtain wall glazing for _MAT 3: mullions every 1.5 m, spandrel band at each floor line."""
    cv.alb[:] = hex_rgb("#2A3036")
    cv.h[:] = -0.06
    glass(cv, cv.rect(0, 0.75, 3, 3), 0, 0.75, 3, 3, 51, interior=None)
    sp = cv.rect(0, 0, 3, 0.75)
    cv.put(sp, color="#3A4148", h=-0.05, rough=0.12, metal=0.3, mask=0.0)
    for x in (0.0, 1.5, 3.0):
        bar(cv, x - 0.035, 0, x + 0.035, 3, color="#5B6166", h=0.0, rough=0.35, metal=0.85)
    bar(cv, 0, 0.72, 3, 0.8, color="#5B6166", h=0.0, rough=0.35, metal=0.85)
    bar(cv, 0, -0.03, 3, 0.03, color="#5B6166", h=0.0, rough=0.35, metal=0.85)
    bar(cv, 0, 2.97, 3, 3.03, color="#5B6166", h=0.0, rough=0.35, metal=0.85)


# ---------------------------------------------------------------------------
# facade: openings (3 m x 3 m floor-bays; wall areas carry mask 1)
# ---------------------------------------------------------------------------


def _wall(cv: Canvas, seed: int = 101) -> None:
    stucco(cv, "sand", seed)


def win_slider(cv: Canvas, x0: float = 0.6, w: float = 1.8, sill: float = 1.0, hgt: float = 1.2, seed: int = 1,
               blind: float = 0.45, trim: bool = True) -> None:
    """White vinyl horizontal slider (6040 class): fixed lite + sliding sash, meeting rail."""
    x1, y0, y1 = x0 + w, sill, sill + hgt
    if trim:
        foam_trim(cv, x0, y0, x1, y1)
    cv.put(cv.rect(x0, y0, x1, y1), h=-0.06, mask=0.0)
    recess_shadow(cv, x0, y0, x1, y1)
    glass(cv, cv.rect(x0, y0, x1, y1), x0, y0, x1, y1, seed, interior="blinds", blind=blind)
    frame(cv, x0, y0, x1, y1, 0.065)
    xm = x0 + w / 2
    frame(cv, xm - 0.02, y0 + 0.05, x1 - 0.05, y1 - 0.05, 0.05, h=-0.045)
    bar(cv, xm - 0.03, y0, xm + 0.03, y1, h=-0.04)


def win_single_hung(cv: Canvas, x0: float, w: float, sill: float, hgt: float, seed: int, grids: bool = True,
                    blind: float = 0.0) -> None:
    x1, y0, y1 = x0 + w, sill, sill + hgt
    cv.put(cv.rect(x0, y0, x1, y1), h=-0.06, mask=0.0)
    recess_shadow(cv, x0, y0, x1, y1)
    glass(cv, cv.rect(x0, y0, x1, y1), x0, y0, x1, y1, seed, interior="curtain" if blind == 0 else "blinds", blind=blind)
    frame(cv, x0, y0, x1, y1, 0.06)
    ym = y0 + hgt / 2
    bar(cv, x0, ym - 0.035, x1, ym + 0.035, h=-0.038)
    if grids:  # colonial grids between the glass: thin, slightly muted
        for k in (1, 2):
            gx = x0 + w * k / 3
            for ya, yb in ((y0 + 0.06, ym - 0.035), (ym + 0.035, y1 - 0.06)):
                cv.put(cv.rect(gx - 0.011, ya, gx + 0.011, yb), color="#D9D8D2", h=-0.064, rough=0.4)
        for gy in (y0 + hgt * 0.25, y0 + hgt * 0.75):
            cv.put(cv.rect(x0 + 0.06, gy - 0.011, x1 - 0.06, gy + 0.011), color="#D9D8D2", h=-0.064, rough=0.4)


def op_window_slider(cv: Canvas) -> None:
    _wall(cv)
    win_slider(cv, 0.6, 1.8, 1.0, 1.2, seed=11, blind=0.45)


def op_window_pair(cv: Canvas) -> None:
    _wall(cv)
    foam_trim(cv, 0.45, 0.85, 2.55, 2.4)
    win_single_hung(cv, 0.45, 0.98, 0.85, 1.55, 12, blind=0.0)
    win_single_hung(cv, 1.57, 0.98, 0.85, 1.55, 13, blind=0.3)
    cv.put(cv.rect(1.43, 0.85, 1.57, 2.4), color="#ECEBE6", h=-0.03, rough=0.45, mask=0.0)


def op_window_picture(cv: Canvas) -> None:
    _wall(cv)
    x0, x1, y0, y1 = 0.3, 2.7, 0.75, 2.4
    foam_trim(cv, x0, y0, x1, y1, w=0.12)
    cv.put(cv.rect(x0, y0, x1, y1), h=-0.06, mask=0.0)
    recess_shadow(cv, x0, y0, x1, y1)
    glass(cv, cv.rect(x0, y0, x1, y1), x0, y0, x1, y1, 14, interior="curtain")
    frame(cv, x0, y0, x1, y1, 0.07)
    for xm in (0.3 + 0.6, 2.7 - 0.6):
        bar(cv, xm - 0.03, y0, xm + 0.03, y1)
    for gx in (0.3 + 0.3, 2.7 - 0.3, 1.5 - 0.3, 1.5 + 0.3):  # grids in the side lites and center
        cv.put(cv.rect(gx - 0.011, y0 + 0.07, gx + 0.011, y1 - 0.07), color="#D9D8D2", h=-0.064)
    for gy in (y0 + (y1 - y0) / 3, y0 + 2 * (y1 - y0) / 3):
        cv.put(cv.rect(x0 + 0.07, gy - 0.011, x1 - 0.07, gy + 0.011), color="#D9D8D2", h=-0.064)


def op_window_small(cv: Canvas) -> None:
    _wall(cv)
    x0, x1, y0, y1 = 1.05, 1.95, 1.65, 2.3
    foam_trim(cv, x0, y0, x1, y1, w=0.08)
    cv.put(cv.rect(x0, y0, x1, y1), h=-0.06, mask=0.0)
    recess_shadow(cv, x0, y0, x1, y1)
    glass(cv, cv.rect(x0, y0, x1, y1), x0, y0, x1, y1, 15)
    obs = cv.noise(16, 0.02, beta=1.0) * 0.04  # obscure (bathroom) glass: light, rippled
    cv.put(cv.rect(x0, y0, x1, y1), color="#9AA3A6", h=-0.06, rough=0.25)
    cv.alb = cv.alb + (obs[..., None] * cv.rect(x0, y0, x1, y1)[..., None])
    cv.h = cv.h + 0.002 * cv.noise(17, 0.03) * cv.rect(x0, y0, x1, y1)
    frame(cv, x0, y0, x1, y1, 0.055)
    xm = (x0 + x1) / 2
    bar(cv, xm - 0.025, y0, xm + 0.025, y1)


def op_window_arched(cv: Canvas) -> None:
    """Mediterranean arched-top window with grids and a stucco arch surround."""
    _wall(cv)
    x0, x1, y0, ys = 0.85, 2.15, 0.8, 2.0
    r = (x1 - x0) / 2
    cx = (x0 + x1) / 2
    surround = cv.arch(x0 - 0.12, y0 - 0.12, x1 + 0.12, ys)
    cv.h = cv.h + 0.025 * surround
    op = cv.arch(x0, y0, x1, ys)
    cv.put(op, h=-0.06, mask=0.0)
    glass(cv, op, x0, y0, x1, ys + r, 18, interior="curtain")
    inner = cv.arch(x0 + 0.065, y0 + 0.065, x1 - 0.065, ys)
    ring = np.clip(op - inner, 0, 1)
    cv.put(ring, color="#ECEBE6", h=-0.03, rough=0.45, mask=0.0)
    bar(cv, x0, ys - 0.03, x1, ys + 0.03)
    for gx in (cx - r / 3, cx + r / 3):
        cv.put(cv.rect(gx - 0.011, y0 + 0.06, gx + 0.011, ys) * inner, color="#D9D8D2", h=-0.064)
    for gy in (y0 + 0.4, y0 + 0.8):
        cv.put(cv.rect(x0 + 0.06, gy - 0.011, x1 - 0.06, gy + 0.011), color="#D9D8D2", h=-0.064)
    ang = np.arctan2(cv.Y - ys, cv.X - cx)
    spokes = np.zeros_like(ang)
    for a in (math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        spokes = np.maximum(spokes, np.clip(1 - np.abs(ang - a) * np.hypot(cv.X - cx, cv.Y - ys) / 0.011, 0, 1))
    cv.put(spokes * inner * (cv.Y > ys), color="#D9D8D2", h=-0.064)
    s = cv.rect(x0 - 0.18, y0 - 0.2, x1 + 0.18, y0 - 0.1)
    cv.h = np.maximum(cv.h, 0.045 * s)


def op_door_front(cv: Canvas) -> None:
    """Recessed entry: 8 ft stained plank-panel door, sidelite, coach light, concrete step."""
    _wall(cv)
    dx0, dx1, dy1 = 0.95, 1.9, 2.44
    sx0, sx1 = 1.98, 2.36
    foam_trim(cv, dx0 - 0.08, 0.0, sx1 + 0.08, dy1 + 0.08, w=0.14, sill=False)
    cv.put(cv.rect(dx0 - 0.08, 0.0, sx1 + 0.08, dy1 + 0.08), color="#E4E1D9", h=-0.12, rough=0.5, mask=0.0)
    door = cv.rect(dx0, 0.05, dx1, dy1)
    wood = cv.noise(21, 0.6, beta=1.4, octaves_m=0.003, aniso=(0.08, 1.0))
    planks = 0.5 + 0.5 * np.cos(2 * np.pi * (cv.X - dx0) / ((dx1 - dx0) / 5))
    col = hex_rgb("#4A3122") * (0.9 + 0.08 * wood[..., None] + 0.05 * planks[..., None])
    cv.put(door, h=-0.13, rough=0.55, mask=0.0)
    cv.alb = cv.alb * (1 - door[..., None]) + col * door[..., None]
    cv.h = cv.h - 0.003 * np.clip(planks - 0.92, 0, 1) * 10 * door + 0.0004 * wood * door
    lite = cv.rect(dx0 + 0.25, 1.75, dx1 - 0.25, 2.2)  # small speakeasy lite
    glass(cv, lite, dx0 + 0.25, 1.75, dx1 - 0.25, 2.2, 22, h=-0.14)
    for yy in (0.45, 1.5):  # dark iron straps
        cv.put(cv.rect(dx0 + 0.04, yy, dx1 - 0.25, yy + 0.05), color="#1B1A19", h=-0.12, metal=0.7, rough=0.5)
    cv.put(cv.rect(dx1 - 0.14, 0.95, dx1 - 0.1, 1.2), color="#2A2A28", h=-0.11, metal=0.8, rough=0.35)
    side = cv.rect(sx0, 0.05, sx1, dy1)
    glass(cv, side, sx0, 0.05, sx1, dy1, 23, h=-0.14, interior="curtain")
    frame(cv, sx0, 0.05, sx1, dy1, 0.05, color="#3E2A1E", h=-0.12, rough=0.55)
    frame(cv, dx0 - 0.05, 0.0, dx1 + 0.05, dy1 + 0.05, 0.05, color="#E8E6E0", h=-0.1)
    cv.put(cv.rect(dx0 - 0.25, 0.0, sx1 + 0.25, 0.07), color="#B7B2A8", h=0.02, rough=0.85, mask=0.0)
    recess_shadow(cv, dx0 - 0.08, 0.0, sx1 + 0.08, dy1 + 0.08, depth=0.8)
    coach_light(cv, 0.6, 1.85)
    grime_bottom(cv, 0, 3, 0.0, 0.25, 0.08)


def op_door_slider(cv: Canvas) -> None:
    """White vinyl patio sliding door 8 ft x 6 ft 8 in, screen on the sliding panel, vertical blinds."""
    _wall(cv)
    x0, x1, y0, y1 = 0.3, 2.7, 0.04, 2.1
    foam_trim(cv, x0, y0, x1, y1, w=0.1, sill=False)
    cv.put(cv.rect(x0, y0, x1, y1), h=-0.07, mask=0.0)
    recess_shadow(cv, x0, y0, x1, y1)
    glass(cv, cv.rect(x0, y0, x1, y1), x0, y0, x1, y1, 24, interior="vblinds", blind=0.35)
    frame(cv, x0, y0, x1, y1, 0.07)
    xm = (x0 + x1) / 2
    frame(cv, x0 + 0.05, y0 + 0.05, xm + 0.03, y1 - 0.05, 0.075, h=-0.045)
    frame(cv, xm - 0.03, y0 + 0.05, x1 - 0.05, y1 - 0.05, 0.075, h=-0.05)
    scr = cv.rect(xm + 0.04, y0 + 0.12, x1 - 0.12, y1 - 0.12)  # screen door: gray mesh over the glass
    mesh = 0.5 + 0.5 * np.cos(2 * np.pi * cv.X / 0.004) * np.cos(2 * np.pi * cv.Y / 0.004)
    cv.alb = cv.alb * (1 - 0.45 * scr[..., None]) + (hex_rgb("#5C6166") * (0.9 + 0.1 * mesh[..., None])) * 0.45 * scr[..., None]
    cv.put(cv.rect(xm + 0.04, 1.0, x1 - 0.12, 1.04), color="#D5D3CC", h=-0.02)
    cv.put(cv.rect(x0 - 0.1, 0.0, x1 + 0.1, 0.04), color="#B9B6AE", h=0.0, rough=0.8, mask=0.0)
    grime_bottom(cv, 0, 3, 0.0, 0.2, 0.06)


def garage_door(cv: Canvas, x0: float, x1: float, seed: int, panels: int, windows: bool = False,
                long_panel: bool = False) -> None:
    """Sectional steel raised-panel garage door, 7 ft (2.13 m), embossed wood grain, near-white paint (mask 0.5)."""
    y0, y1 = 0.0, 2.13
    foam_trim(cv, x0, y0, x1, y1, w=0.12, sill=False)
    cv.put(cv.rect(x0 - 0.02, y0, x1 + 0.02, y1 + 0.02), color="#E3E0D8", h=-0.1, rough=0.55, mask=0.5)
    grain = cv.noise(seed, 0.5, beta=1.2, octaves_m=0.002, aniso=(0.05, 1.0))
    base = hex_rgb("#E7E4DC")
    door = cv.rect(x0, y0, x1, y1)
    col = base * (0.97 + 0.012 * grain[..., None])
    cv.alb = cv.alb * (1 - door[..., None]) + col * door[..., None]
    hh = -0.08 + 0.0003 * grain
    sec_h = (y1 - y0) / 4
    n_col = panels if not long_panel else max(2, panels // 2)
    pw = (x1 - x0) / n_col
    for r in range(4):
        sy0, sy1 = y0 + r * sec_h, y0 + (r + 1) * sec_h
        seam = cv.rect(x0, sy0 - 0.005, x1, sy0 + 0.006) if r else np.zeros(cv.shape)
        hh = hh - 0.008 * seam
        cv.ao = cv.ao * (1 - 0.35 * seam)
        for c in range(n_col):
            px0, px1 = x0 + c * pw + 0.07, x0 + (c + 1) * pw - 0.07
            py0, py1 = sy0 + 0.09, sy1 - 0.09
            if windows and r == 3:
                g = cv.rect(px0 + 0.03, py0 + 0.02, px1 - 0.03, py1 - 0.02)
                glass(cv, g, px0, py0, px1, py1, seed + c, h=-0.09)
                cv.put(g, mask=0.0)
                continue
            raised = cv.bevel(px0, py0, px1, py1, 0.035) * cv.rect(px0, py0, px1, py1)
            hh = hh + 0.011 * raised
    cv.h = cv.h * (1 - door) + hh * door
    cv.rough = cv.rough * (1 - door) + 0.5 * door
    cv.mask = cv.mask * (1 - door) + 0.5 * door
    seal = cv.rect(x0, 0.0, x1, 0.025)
    cv.put(seal, color="#1E1E1E", h=-0.08, rough=0.9, mask=0.0)
    grime_bottom(cv, x0, x1, 0.02, 0.35, 0.1)
    cv.put(cv.rect(x0 + (x1 - x0) / 2 - 0.12, 0.45, x0 + (x1 - x0) / 2 + 0.12, 0.49), color="#9A9A96", h=-0.06, metal=0.8)


def op_garage_2car(cv: Canvas) -> None:  # canvas 6 m x 3 m
    _wall(cv, 102)
    garage_door(cv, 0.56, 5.44, 31, panels=8)
    coach_light(cv, 0.25, 1.95, 0.85)
    coach_light(cv, 5.75, 1.95, 0.85)


def op_garage_3car(cv: Canvas) -> None:  # canvas 9 m x 3 m: 16 ft door + pier + 8 ft door
    _wall(cv, 103)
    garage_door(cv, 0.45, 5.33, 32, panels=8, windows=True, long_panel=True)
    garage_door(cv, 6.13, 8.57, 33, panels=4, windows=True, long_panel=True)
    coach_light(cv, 5.73, 1.95, 0.85)
    coach_light(cv, 8.82, 1.95, 0.8)


def op_storefront(cv: Canvas) -> None:
    """Dark bronze aluminum storefront bay: kick plate, tall glass with lit interior, transom."""
    cv.alb[:] = hex_rgb("#30302E")
    cv.h[:] = -0.05
    cv.rough[:] = 0.4
    cv.mask[:] = 0.0
    glass(cv, cv.rect(0, 0.38, 3, 2.4), 0, 0.38, 3, 2.4, 41, h=-0.06, interior="store")
    glass(cv, cv.rect(0, 2.5, 3, 2.94), 0, 2.5, 3, 2.94, 42, h=-0.06)
    bronze = "#3A3129"
    cv.put(cv.rect(0, 0, 3, 0.38), color="#3C342C", h=-0.03, rough=0.35, metal=0.85, mask=0.0)
    for x in (0.0, 3.0):
        bar(cv, x - 0.045, 0, x + 0.045, 3, color=bronze, h=0.0, rough=0.35, metal=0.85)
    bar(cv, 0, 0.35, 3, 0.42, color=bronze, h=0.0, rough=0.35, metal=0.85)
    bar(cv, 0, 2.4, 3, 2.5, color=bronze, h=0.0, rough=0.35, metal=0.85)
    bar(cv, 0, 2.94, 3, 3.0, color=bronze, h=0.0, rough=0.35, metal=0.85)
    grime_bottom(cv, 0, 3, 0.0, 0.3, 0.1)


def op_storefront_sign(cv: Canvas) -> None:
    """Upper commercial band: stucco with a projecting cornice and a recessed sign field (mask 1)."""
    stucco(cv, "scored", 104)
    cv.h = cv.h + 0.0 * cv.X
    sign = cv.rect(0.15, 0.55, 2.85, 1.85)
    cv.h = cv.h - 0.02 * sign
    cv.ao = cv.ao * (1 - 0.15 * cv.rect(0.15, 1.75, 2.85, 1.85))
    for (y0, y1, p) in ((2.35, 2.55, 0.05), (2.55, 2.7, 0.09), (2.7, 2.85, 0.13), (2.85, 3.0, 0.16)):
        cv.h = np.maximum(cv.h, p * cv.hband(y0, y1) * (0.85 + 0.15 * cv.bevel(-1, y0, 4, y1, 0.02)))
    cv.ao = cv.ao * (1 - 0.35 * np.clip(1 - (2.35 - cv.Y) / 0.25, 0, 1) * (cv.Y < 2.35))
    for x in (0.4, 2.6):  # gooseneck sign lights
        cv.put(cv.rect(x - 0.02, 1.95, x + 0.02, 2.3), color="#202020", h=0.05, metal=0.7, rough=0.4)
        cv.put(cv.ellipse(x, 1.95, 0.09, 0.05), color="#262626", h=0.12, metal=0.7, rough=0.4)


def op_school_window_band(cv: Canvas) -> None:
    """Classroom ribbon window (clear anodized aluminum, mullions every 1.5 m) with a sunshade louver."""
    stucco(cv, "scored", 105)
    y0, y1 = 0.95, 2.4
    cv.put(cv.rect(0, y0, 3, y1), h=-0.08, mask=0.0)
    glass(cv, cv.rect(0, y0, 3, y1), 0, y0, 3, y1, 43, h=-0.08, interior="classroom")
    alu = "#A9ADB0"
    for x in (0.0, 1.5, 3.0):
        bar(cv, x - 0.04, y0, x + 0.04, y1, color=alu, h=-0.05, rough=0.35, metal=0.9)
    bar(cv, 0, y0, 3, y0 + 0.06, color=alu, h=-0.05, rough=0.35, metal=0.9)
    bar(cv, 0, y1 - 0.06, 3, y1, color=alu, h=-0.05, rough=0.35, metal=0.9)
    bar(cv, 0, 1.75, 3, 1.8, color=alu, h=-0.05, rough=0.35, metal=0.9)
    cv.put(cv.rect(0, y0 - 0.08, 3, y0), color="#B8B4AA", h=0.02, rough=0.7, mask=0.0)  # precast sill
    lv = cv.rect(0, 2.55, 3, 2.7)  # horizontal sunshade louver
    cv.put(lv, color="#BFC3C5", h=0.12, rough=0.4, metal=0.8, mask=0.0)
    for k in range(10):
        xx = 0.15 + k * 0.3
        cv.put(cv.rect(xx - 0.01, 2.55, xx + 0.01, 2.7), color="#8E9396", h=0.12)
    cv.ao = cv.ao * (1 - 0.3 * np.clip(1 - (2.55 - cv.Y) / 0.5, 0, 1) * (cv.Y < 2.55) * (cv.Y > y1 - 0.4))
    recess_shadow(cv, 0, y0, 3, y1, depth=0.4)


def op_school_door(cv: Canvas) -> None:
    """Pair of hollow-metal doors with vision lites and push bars, transom (paintable, mask 0.5)."""
    stucco(cv, "scored", 106)
    x0, x1, y1 = 0.58, 2.42, 2.13
    frame(cv, x0 - 0.06, 0.0, x1 + 0.06, 2.75, 0.06, color="#8E9295", h=-0.04, rough=0.4, metal=0.7)
    t = cv.rect(x0, y1 + 0.06, x1, 2.69)
    glass(cv, t, x0, y1, x1, 2.69, 44, h=-0.08)
    bar(cv, x0, y1, x1, y1 + 0.06, color="#8E9295", h=-0.04, metal=0.7, rough=0.4)
    xm = (x0 + x1) / 2
    for a, b in ((x0, xm - 0.004), (xm + 0.004, x1)):
        d = cv.rect(a, 0.01, b, y1)
        cv.put(d, color="#D9D8D2", h=-0.06 + 0.002 * cv.bevel(a, 0.01, b, y1, 0.02), rough=0.45, metal=0.2, mask=0.5)
        v0 = a + (b - a) * 0.62
        g = cv.rect(v0, 1.2, v0 + 0.13, 1.95)
        glass(cv, g, v0, 1.2, v0 + 0.13, 1.95, 45, h=-0.07)
        cv.put(cv.rect(a + 0.08, 0.95, b - 0.08, 1.02), color="#B5B8BA", h=-0.03, metal=0.9, rough=0.3)
        cv.put(cv.rect(a + 0.02, 0.01, b - 0.02, 0.27), color="#A8ACAE", h=-0.058, metal=0.9, rough=0.35)
    cv.put(cv.rect(xm - 0.006, 0.0, xm + 0.006, y1), color="#2A2A2A", h=-0.07)
    cv.put(cv.rect(x0 - 0.3, 0, x1 + 0.3, 0.03), color="#B4B0A6", h=0.0, mask=0.0, rough=0.85)
    recess_shadow(cv, x0, 0.0, x1, 2.69, depth=0.4)
    grime_bottom(cv, x0, x1, 0.0, 0.3, 0.08)


# ---------------------------------------------------------------------------
# roofs (tileable 4 m x 4 m: u along the eave, v up the slope)
# ---------------------------------------------------------------------------


def _tile_grid(cv: Canvas, n_cols: int, n_rows: int, stagger: bool) -> tuple[np.ndarray, ...]:
    tw, tl = cv.w_m / n_cols, cv.h_m / n_rows
    row = np.floor(cv.Y / tl)
    xx = cv.X / tw + (0.5 * (row % 2) if stagger else 0.0)
    col = np.floor(xx)
    lx = xx - col
    ly = cv.Y / tl - row
    tid = (row.astype(int) * 997 + (col.astype(int) % n_cols) * 31)
    r = (np.sin(tid * 12.9898 + 1.7) * 43758.5453) % 1.0
    r2 = (np.sin(tid * 4.1414 + 0.3) * 24634.6345) % 1.0
    return lx, ly, r, r2, tw, tl


def roof_s_tile(pal: list[str], seed: int, aged: float = 0.0, flash: float = 0.25, var: float = 0.7) -> Callable[[Canvas], None]:
    """Spanish S tile (one-piece concrete/clay 'S'): straight columns, ~0.33 m wide, 0.33 m exposure."""
    def f(cv: Canvas) -> None:
        lx, ly, r, r2, tw, tl = _tile_grid(cv, 12, 12, stagger=False)
        warp = lx + 0.1 * np.sin(2 * np.pi * lx)
        across = 0.5 + 0.5 * np.sin(2 * np.pi * (warp - 0.25))  # 0 pan .. 1 cap
        A = 0.06
        butt = 0.028 * (1 - ly) ** 0.8
        jit = (r2 - 0.5) * 0.006
        hh = A * across + butt + jit + 0.004 * (r - 0.5) * (ly - 0.5)
        n = cv.noise(seed, 0.4, beta=1.8, octaves_m=0.005)
        hh = hh + 0.0008 * n
        pc = np.array([hex_rgb(c) for c in pal])
        col = pc.mean(axis=0) + (_palette_pick(pal, r) - pc.mean(axis=0)) * var
        fl = 1 - flash * smoothstep(0.2, 1.0, ly + 0.25 * np.sin(lx * 3 + r2 * 6)) * (r2 > 0.35)
        shade = (0.92 + 0.1 * across)[..., None]
        col = col * fl[..., None] * shade * (1 + 0.05 * n[..., None])
        mouth = smoothstep(0.07, 0.0, ly) * smoothstep(0.45, 0.85, across)  # dark open ends under the caps
        lock = smoothstep(0.035, 0.0, np.abs(lx - 0.03)) * 0.5
        dirt = smoothstep(0.35, 0.0, across) * (0.5 + 0.5 * cv.noise(seed + 1, 1.0))
        col = col * (1 - 0.18 * dirt * (0.4 + aged))[..., None] * (1 - 0.25 * lock)[..., None]
        if aged > 0:
            lich = smoothstep(1.4, 2.2, cv.noise(seed + 2, 0.12, beta=1.5)) * aged
            col = col * (1 - 0.35 * aged) + hex_rgb("#8D8B79") * 0.35 * aged
            col = col * (1 - lich[..., None]) + hex_rgb("#B8B49E")[None, None] * lich[..., None]
            efflo = smoothstep(1.2, 2.0, cv.noise(seed + 3, 0.6, beta=2.0)) * 0.25 * aged
            col = col + efflo[..., None] * 0.3
        cv.alb = np.clip(col, 0, 1)
        cv.h = hh
        cv.rough = 0.72 + 0.12 * dirt + 0.1 * aged
        cv.ao = 1 - 0.55 * mouth
        cv.mask[:] = 0
    return f


def roof_barrel(cv: Canvas) -> None:
    """Two-piece mission barrel clay tile: convex caps over concave pans, staggered butts, clay blend."""
    pal = ["#B4552F", "#A2482A", "#C26739", "#94402A", "#B85E38", "#A85130"]
    lx, ly, r, r2, tw, tl = _tile_grid(cv, 16, 9, stagger=False)
    cap = (np.floor(cv.X / (cv.w_m / 16)) % 2) == 1
    prof = np.sin(np.pi * lx)
    hh = np.where(cap, 0.03 + 0.07 * prof, 0.03 * (1 - prof) * 0.4)
    hh = hh + 0.025 * (1 - ly) ** 0.7 + (r2 - 0.5) * 0.008
    n = cv.noise(61, 0.3, beta=1.8, octaves_m=0.005)
    col = _palette_pick(pal, r) * (0.9 + 0.12 * prof[..., None] * cap[..., None]) * (1 + 0.06 * n[..., None])
    col = col * (1 - 0.2 * smoothstep(0.3, 1.0, ly) * (r2 > 0.5))[..., None]
    mortar = smoothstep(0.04, 0.0, ly) * cap * (r > 0.8) * 0.5
    col = col * (1 - mortar[..., None]) + hex_rgb("#CFC6B4") * mortar[..., None]
    cv.alb = np.clip(col, 0, 1)
    cv.h = hh + 0.0008 * n
    cv.rough = 0.75 + 0.1 * (~cap)
    cv.ao = 1 - 0.45 * smoothstep(0.08, 0.0, ly) * cap
    cv.mask[:] = 0


def roof_flat_tile(pal: list[str], seed: int, slate: bool = True) -> Callable[[Canvas], None]:
    """Flat concrete tile (slate / shake look), half-staggered courses, ~0.33 m exposure."""
    def f(cv: Canvas) -> None:
        lx, ly, r, r2, tw, tl = _tile_grid(cv, 12, 12, stagger=True)
        gap = smoothstep(0.0, 0.025, np.minimum(lx, 1 - lx))
        butt = 0.022 * (1 - ly) + 0.002
        n = cv.noise(seed, 0.3, beta=1.6, octaves_m=0.004)
        streak = cv.noise(seed + 1, 0.4, beta=1.4, octaves_m=0.004, aniso=(1.0, 0.12)) if slate else n
        hh = butt * gap + 0.0012 * streak + (r2 - 0.5) * 0.004 - 0.01 * (1 - gap)
        pc = np.array([hex_rgb(c) for c in pal])
        col = pc.mean(axis=0) + (_palette_pick(pal, r) - pc.mean(axis=0)) * 0.55
        col = col * (1 + 0.06 * streak[..., None] + 0.04 * (r2[..., None] - 0.5))
        col = col * (0.55 + 0.45 * gap[..., None])
        edge = smoothstep(0.06, 0.0, ly)
        dirt = (0.5 + 0.5 * cv.noise(seed + 2, 1.5)) * 0.08
        cv.alb = np.clip(col * (1 - dirt[..., None]), 0, 1)
        cv.h = hh
        cv.rough = 0.8 + 0.08 * (1 - gap)
        cv.ao = 1 - 0.35 * edge
        cv.mask[:] = 0
    return f


def roof_tpo(grime: float, seed: int) -> Callable[[Canvas], None]:
    """White TPO membrane (2 m sheets, welded seams) with dust; grime > 0 adds HVAC soot, rust and ponding."""
    def f(cv: Canvas) -> None:
        sh = cv.shape
        n = cv.noise(seed, 2.0, beta=2.4, octaves_m=0.02)
        fine = tg.fbm(sh, seed + 1, beta=0.5, fmin=40)
        col = np.full((*sh, 3), 0.86) * (1 - 0.05 * tg.smoothstep(-0.5, 1.5, n))[..., None] * hex_rgb("#F2F1EC")
        dx = np.abs(((cv.X + 1.0) % 2.0) - 1.0)
        seam = smoothstep(0.03, 0.015, dx)
        wrinkle = cv.noise(seed + 2, 0.6, beta=2.2, octaves_m=0.03, aniso=(1.0, 0.3)) * 0.0015
        hh = 0.0025 * seam + wrinkle + 0.0002 * fine
        col = col * (1 + 0.03 * seam)[..., None]
        if grime > 0:
            pond = smoothstep(0.8, 1.6, cv.noise(seed + 3, 1.5, beta=2.6))
            ring = smoothstep(0.08, 0.0, np.abs(cv.noise(seed + 3, 1.5, beta=2.6) - 0.8))
            soot = smoothstep(0.2, 2.2, tg.blur(cv.noise(seed + 4, 1.2, beta=2.6), 3)) * grime * 0.6
            rust = smoothstep(2.3, 2.9, cv.noise(seed + 5, 0.15, beta=2.0)) * grime * 0.7
            path = smoothstep(0.35, 0.0, np.abs(cv.X - 1.4 - 0.2 * np.sin(cv.Y * 1.57))) * 0.5 * grime
            col = col * (1 - 0.22 * pond * grime - 0.25 * ring * grime)[..., None] + hex_rgb("#8B7A60") * (0.12 * pond * grime)[..., None]
            col = col * (1 - 0.45 * soot - 0.15 * path)[..., None]
            col = col * (1 - rust[..., None]) + hex_rgb("#7A4A2A") * rust[..., None]
        cv.alb = np.clip(col, 0, 1)
        cv.h = hh
        cv.rough = 0.62 + 0.15 * grime
        cv.mask[:] = 0
    return f


def roof_gravel(cv: Canvas) -> None:
    """Ballasted roof: 2-4 cm river gravel over the membrane."""
    sh = cv.shape
    c1 = tg.worley(sh, (130, 130), 71, jitter=1.0)
    c2 = tg.worley(sh, (90, 90), 72, jitter=1.0)
    st1 = np.clip(1 - c1.f1 / (0.75 * c1.f2 + 1e-6), 0, 1)
    st2 = np.clip(1 - c2.f1 / (0.75 * c2.f2 + 1e-6), 0, 1)
    hh = 0.012 * np.maximum(st1 ** 0.5, st2 ** 0.5 * 1.1)
    pal = ["#A9A196", "#8E877D", "#BDB6A9", "#7C756C", "#C8C0B1", "#9B8F80"]
    col = np.where((st2 ** 0.5 * 1.1 > st1 ** 0.5)[..., None], _palette_pick(pal, c2.rnd), _palette_pick(pal, c1.rnd))
    dirt = cv.noise(73, 2.0, beta=2.4)
    col = col * (0.85 + 0.2 * np.maximum(st1, st2)[..., None]) * (1 - 0.06 * dirt[..., None])
    cv.alb = np.clip(col, 0, 1)
    cv.h = hh
    cv.rough[:] = 0.93
    cv.mask[:] = 0


def roof_modbit(cv: Canvas) -> None:
    """Gray granulated modified-bitumen cap sheet, 1 m laps."""
    sh = cv.shape
    gr = tg.fbm(sh, 81, beta=0.0, fmin=1)
    sheet = np.floor(cv.Y / 1.0)
    shade = 1 + 0.04 * np.sin(sheet * 7.3)
    lap = smoothstep(0.04, 0.0, cv.Y % 1.0)
    col = hex_rgb("#7E7D79") * (shade * (1 + 0.07 * gr) * (1 - 0.15 * lap))[..., None]
    n = cv.noise(82, 1.5, beta=2.4)
    col = col * (1 - 0.06 * n)[..., None]
    cv.alb = np.clip(col, 0, 1)
    cv.h = 0.0006 * gr + 0.003 * smoothstep(0.0, 0.05, cv.Y % 1.0) * (1 - smoothstep(0.05, 0.12, cv.Y % 1.0))
    cv.rough[:] = 0.95
    cv.mask[:] = 0


def roof_solar(cv: Canvas) -> None:
    """Residential PV array: 1 x 2 m mono panels, 6 x 12 cells, silver frames, 2 cm gaps."""
    pw, ph = 1.0, 2.0
    lx = cv.X % pw
    ly = cv.Y % ph
    inside = cv.rect(0, 0, cv.w_m, cv.h_m)
    fr = np.minimum(np.minimum(lx, pw - lx), np.minimum(ly, ph - ly))
    gap = fr < 0.01
    frame_ = (fr >= 0.01) & (fr < 0.04)
    cx = (lx - 0.04) / ((pw - 0.08) / 6)
    cy = (ly - 0.04) / ((ph - 0.08) / 12)
    grid = (np.abs(cx - np.round(cx)) < 0.025) | (np.abs(cy - np.round(cy)) < 0.012)
    busbar = np.abs(((cx % 1.0) - 0.5)) < 0.01
    col = np.where(gap[..., None], hex_rgb("#1A1A1A"), np.where(frame_[..., None], hex_rgb("#B7BCC0"),
                   np.where(grid[..., None], hex_rgb("#2E3644"), hex_rgb("#121722"))))
    col = np.where((busbar & ~grid & ~frame_ & ~gap)[..., None], hex_rgb("#3A4250"), col)
    sky = (0.04 * (cv.Y / cv.h_m))[..., None]
    cv.alb = np.clip(col + sky * (~gap & ~frame_)[..., None], 0, 1) * inside[..., None] + 0 * col
    cv.h = np.where(gap, -0.04, np.where(frame_, 0.0, -0.004))
    cv.rough = np.where(frame_, 0.35, np.where(gap, 0.9, 0.08))
    cv.metal = np.where(frame_, 0.85, 0.0)
    cv.mask[:] = 0


def roof_standing_seam(cv: Canvas) -> None:
    """Standing-seam metal (canopies, covered walks, towers): 0.45 m pans, 4 cm ribs."""
    pw = cv.w_m / 9
    lx = cv.X % pw
    rib = smoothstep(0.03, 0.012, np.minimum(lx, pw - lx))
    n = cv.noise(91, 1.5, beta=2.4)
    cv.alb = np.clip(hex_rgb("#7E868B") * (1 + 0.04 * n + 0.05 * rib)[..., None], 0, 1)
    cv.h = 0.04 * rib + 0.0008 * cv.noise(92, 0.5, aniso=(0.2, 1.0))
    cv.rough = 0.38 + 0.05 * n
    cv.metal[:] = 0.75
    cv.mask[:] = 0


def roof_concrete_deck(cv: Canvas) -> None:
    concrete(cv, 93, joint=2.0, broom=False, tone="#A8A49C")


# ---------------------------------------------------------------------------
# ground
# ---------------------------------------------------------------------------


def asphalt(cv: Canvas, seed: int, wear: float, oil: float = 0.0, cracks: float = 0.0, seal: float = 0.0) -> None:
    sh = cv.shape
    c = tg.worley(sh, (int(cv.w_m * 55), int(cv.h_m * 55)), seed, jitter=1.0)
    agg = np.clip(1 - c.f1 / (0.8 * c.f2 + 1e-6), 0, 1)
    expo = 0.6 * smoothstep(0.62 - 0.25 * wear, 0.9, agg) * smoothstep(-0.3, 0.8, tg.fbm(sh, seed + 1, beta=1.0, fmin=3))
    binder = 0.16 + 0.16 * wear
    stone = 0.3 + 0.25 * c.rnd
    mot = cv.noise(seed + 2, 2.0, beta=2.4, octaves_m=0.05)
    v = binder * (1 + 0.15 * mot) * (1 - expo) + stone * expo
    fine = tg.fbm(sh, seed + 3, beta=0.0, fmin=1)
    v = v * (1 + 0.08 * fine)
    col = np.stack([v, v * 1.0, v * 1.03], axis=-1)
    hh = 0.003 * agg + 0.0006 * fine
    rough = 0.9 - 0.05 * expo
    if oil > 0:  # oil / coolant drips concentrated along the wheel paths' center
        sp = smoothstep(1.2, 2.4, cv.noise(seed + 4, 0.5, beta=1.8, octaves_m=0.03)) * oil
        band = smoothstep(0.9, 0.3, np.abs(((cv.X % 4.0) - 2.0))) * 0.7 + 0.3
        stain = sp * band
        col = col * (1 - 0.45 * stain)[..., None]
        rough = rough - 0.35 * stain
    if cracks > 0:
        cw = tg.worley(sh, (3, 3), seed + 5)
        n = cv.noise(seed + 6, 0.3, octaves_m=0.01)
        cr = smoothstep(1.6, 0.3, cw.f2 - cw.f1 + 0.9 * n) * smoothstep(0.0, 0.8, cv.noise(seed + 7, 2.0))
        lon = smoothstep(0.012, 0.0, np.abs(cv.X - 1.3 - 0.06 * np.sin(cv.Y * 2.1) - 0.02 * n)) * cracks
        crack = np.clip(cr * cracks + lon, 0, 1)
        if seal > 0:  # rubberized crack sealant (very common on SD streets): glossy black ribbons
            s = tg.blur(crack, 2.2)
            s = smoothstep(0.08, 0.2, s) * seal
            col = col * (1 - s[..., None]) + np.array([0.05, 0.05, 0.055]) * s[..., None]
            rough = rough * (1 - s) + 0.45 * s
            hh = hh + 0.0015 * s
        col = col * (1 - 0.6 * crack)[..., None]
        hh = hh - 0.004 * crack
        pn = 0.08 * cv.noise(seed + 8, 0.3, octaves_m=0.02)
        patch = (np.abs((cv.X - 2.6) % 4.0 - 1.0) + pn < 0.7) & (np.abs((cv.Y - 0.6) % 4.0 - 1.3) + pn < 0.55)
        pm = tg.blur(patch.astype(float), 1.0) * cracks
        col = col * (1 - 0.35 * pm)[..., None]
        hh = hh + 0.002 * pm
    cv.alb = np.clip(col, 0, 1)
    cv.h = hh
    cv.rough = np.clip(rough, 0.3, 1)
    cv.mask[:] = 0


def concrete(cv: Canvas, seed: int, joint: float = 1.5, broom: bool = True, tone: str = "#B4B0A8",
             tracks: bool = False, joint_x: float | None = None) -> None:
    sh = cv.shape
    jx = joint_x or joint
    fine = tg.fbm(sh, seed, beta=0.2, fmin=1)
    mot = cv.noise(seed + 1, 1.5, beta=2.6, octaves_m=0.04)
    sx = np.floor(cv.X / jx)
    sy = np.floor(cv.Y / joint)
    slab = (np.sin((sx * 13 + sy * 7 + seed) * 12.9898) * 43758.5453) % 1.0
    col = hex_rgb(tone) * (1 + 0.04 * (slab - 0.5) + 0.04 * mot + 0.03 * fine)[..., None]
    hh = 0.0003 * fine
    if broom:
        br = tg.fbm(sh, seed + 2, beta=0.5, fmin=8, aniso=(0.04, 1.0))
        hh = hh + 0.0005 * br
        col = col * (1 + 0.02 * br)[..., None]
    dx = np.abs(cv.X - np.round(cv.X / jx) * jx)
    dy = np.abs(cv.Y - np.round(cv.Y / joint) * joint)
    j = np.maximum(smoothstep(0.008, 0.003, dx), smoothstep(0.008, 0.003, dy))
    tool = np.maximum(smoothstep(0.06, 0.04, dx), smoothstep(0.06, 0.04, dy)) * (1 - j)
    col = col * (1 - 0.55 * j)[..., None] * (1 + 0.03 * tool)[..., None]
    hh = hh - 0.008 * j
    spots = smoothstep(2.4, 3.0, cv.noise(seed + 3, 0.05, beta=1.0))
    col = col * (1 - 0.3 * spots)[..., None]  # gum / drip spots
    stain = smoothstep(0.8, 2.0, cv.noise(seed + 4, 1.5, beta=2.4)) * 0.12
    col = col * (1 - stain)[..., None] + hex_rgb("#8C7458") * (stain * 0.5)[..., None]
    if tracks:  # tire paths along v on driveways
        tp = np.zeros(sh)
        for xc in (0.95, 2.55):
            tp = np.maximum(tp, smoothstep(0.2, 0.05, np.abs((cv.X % 3.0) - xc)))
        col = col * (1 - 0.05 * tp * (0.6 + 0.4 * mot))[..., None]
        oil = smoothstep(1.6, 2.4, cv.noise(seed + 5, 0.4)) * smoothstep(0.5, 0.2, np.abs((cv.X % 3.0) - 1.75))
        col = col * (1 - 0.35 * oil)[..., None]
    cv.alb = np.clip(col, 0, 1)
    cv.h = hh
    cv.rough = 0.88 - 0.05 * tool
    cv.mask[:] = 0


def ground_curb(cv: Canvas) -> None:
    """Curb and gutter profile strip (canvas 3 m along x, 0.75 m across y): gutter pan 0-0.45, face 0.45-0.6, top 0.6-0.75."""
    concrete(cv, 101, joint=3.0, joint_x=3.0, broom=False, tone="#B9B5AC")
    y = cv.Y
    pan = y < 0.45
    face = (y >= 0.45) & (y < 0.6)
    top = y >= 0.6
    hh = np.where(pan, 0.02 * y / 0.45, np.where(face, 0.02 + 0.15 * (y - 0.45) / 0.15, 0.17 + 0.008 * np.sin(np.pi * (y - 0.6) / 0.15)))
    flow = smoothstep(0.12, 0.0, y) * (0.6 + 0.4 * cv.noise(102, 0.6))
    col = cv.alb * (1 - 0.25 * flow)[..., None] + hex_rgb("#6E6455") * (0.12 * flow)[..., None]
    col = col * np.where(face, 0.88, 1.0)[..., None]
    leaves = smoothstep(2.2, 2.8, cv.noise(103, 0.05, beta=1.2)) * pan * smoothstep(0.25, 0.05, y)
    col = col * (1 - leaves[..., None]) + hex_rgb("#7A6A4A") * leaves[..., None]
    cv.alb = np.clip(col, 0, 1)
    cv.h = hh + cv.h
    cv.ao = cv.ao * (1 - 0.3 * smoothstep(0.08, 0.0, np.abs(y - 0.45)))


def ground_pavers(cv: Canvas) -> None:
    """Interlocking concrete pavers, running bond 10 x 20 cm, tan / charcoal blend (driveways, plazas)."""
    lx, ly, r, r2, tw, tl = _tile_grid(cv, 15, 30, stagger=True)
    pal = ["#B49E86", "#9C8670", "#C7B39A", "#857464", "#A58F78", "#BCA68A", "#8F7C69"]
    e = np.minimum(np.minimum(lx * tw, (1 - lx) * tw), np.minimum(ly * tl, (1 - ly) * tl))
    j = smoothstep(0.002, 0.008, e)
    bev = smoothstep(0.002, 0.012, e)
    n = cv.noise(111, 0.1, beta=1.4, octaves_m=0.004)
    col = _palette_pick(pal, r) * (1 + 0.06 * n[..., None] + 0.05 * (r2[..., None] - 0.5))
    col = col * j[..., None] + hex_rgb("#8A7E6C") * (1 - j[..., None])
    stain = smoothstep(0.6, 2.0, cv.noise(112, 1.2, beta=2.4)) * 0.12
    cv.alb = np.clip(col * (1 - stain[..., None]), 0, 1)
    cv.h = 0.006 * bev + 0.0005 * n
    cv.rough = 0.85 + 0.08 * (1 - j)
    cv.mask[:] = 0


def grass(cv: Canvas, seed: int, dry: float = 0.0) -> None:
    """Top-down irrigated lawn (tall fescue / St. Augustine mix): thousands of thin blade strokes in
    several directions over a darker thatch, clump-scale and mowing-scale color variation."""
    sh = cv.shape
    rng = np.random.default_rng(seed)
    top = np.zeros(sh)
    hh = np.zeros(sh)
    blade_col = np.zeros(sh)
    for k in range(10):
        ang = rng.uniform(0, math.pi)
        n = tg.fbm(sh, seed + k, beta=0.0, fmin=1)
        bl = tg.blur_oriented(n, rng.uniform(4.0, 7.0), 0.55, ang)  # ~2-3 cm long, ~2 mm wide strokes
        bl /= bl.std() + 1e-9
        b = smoothstep(1.1, 2.0, bl)
        z = b * (0.004 + 0.0015 * k)
        sel = z > hh
        hh = np.where(sel, z, hh)
        top = np.maximum(top, b)
        blade_col = np.where(sel, np.clip(rng.uniform(0.2, 0.8) + 0.25 * (bl - 1.5), 0, 1), blade_col)
    clump = cv.noise(seed + 10, 0.4, beta=2.0, octaves_m=0.03)
    big = cv.noise(seed + 11, 2.0, beta=2.6)
    thatch = hex_rgb("#2B3A1C") * (0.9 + 0.1 * tg.fbm(sh, seed + 12, beta=0.5, fmin=1))[..., None]
    g1, g2 = hex_rgb("#3F6127"), hex_rgb("#6E8C3E")
    blade = g1 * (1 - blade_col[..., None]) + g2 * blade_col[..., None]
    t = top[..., None]
    col = thatch * (1 - t) + blade * t
    col = col * (1 + 0.07 * clump[..., None] + 0.05 * big[..., None])
    if dry > 0:
        d = smoothstep(0.2, 1.4, cv.noise(seed + 13, 1.0, beta=2.4, octaves_m=0.03)) * dry
        straw = hex_rgb("#9C8858") * (0.75 + 0.35 * t)
        col = col * (1 - d[..., None]) + straw * d[..., None]
        dz = smoothstep(1.6, 2.3, cv.noise(seed + 14, 0.3, beta=1.6)) * dry
        col = col * (1 - dz[..., None]) + hex_rgb("#7C6447") * dz[..., None]
    cv.alb = np.clip(col, 0, 1)
    cv.h = hh
    cv.rough = 0.95 - 0.12 * top
    cv.mask[:] = 0


def scrub(cv: Canvas, seed: int, pal: list[str], soil: str, cover: float, size: tuple[int, int], crown_r: float = 0.62) -> None:
    """Top-down chaparral / coastal sage scrub: leafy shrub crowns (Worley blobs with twig/leaf texture,
    sunlit tops, dark undersides) over dry soil and leaf litter."""
    sh = cv.shape
    c1 = tg.worley(sh, size, seed, jitter=1.0)
    c2 = tg.worley(sh, (size[0] * 2, size[1] * 2), seed + 1, jitter=1.0)
    leaf = tg.fbm(sh, seed + 2, beta=0.3, fmin=40)
    leaf2 = tg.fbm(sh, seed + 5, beta=1.2, fmin=10)
    wob = 0.18 * tg.fbm(sh, seed + 6, beta=1.6, fmin=6)

    def crown(c: tg.Cells, sc: float, ok: np.ndarray) -> np.ndarray:
        rr = (0.6 + 0.5 * c.rnd) * sc
        return np.clip(1 - c.f1 / rr + wob, 0, 1) * ok

    k1 = crown(c1, cv.px / size[0] * crown_r, c1.rnd < cover)
    k2 = crown(c2, cv.px / size[0] * crown_r * 0.55, c2.rnd < cover * 0.9)
    dome = np.maximum(np.sqrt(k1), 0.8 * np.sqrt(k2))
    holes = smoothstep(-0.2, 0.6, leaf + 0.5 * leaf2 + 1.4 * dome - 0.9)
    m = smoothstep(0.03, 0.2, np.maximum(k1, k2)) * holes
    which = np.where(np.sqrt(k1) >= 0.8 * np.sqrt(k2), c1.rnd, c2.rnd)
    base = _palette_pick(pal, (which * 7.31) % 1.0)
    shade = (0.5 + 0.6 * dome) * (0.65 + 0.5 * smoothstep(-1.5, 1.5, leaf))
    col = base * shade[..., None]
    soilc = hex_rgb(soil) * (1 + 0.08 * cv.noise(seed + 3, 0.5, beta=1.6, octaves_m=0.01))[..., None]
    litter = smoothstep(0.3, 1.4, cv.noise(seed + 4, 0.12, beta=1.2)) * 0.5
    soilc = soilc * (1 - litter[..., None]) + hex_rgb("#6E5E48") * litter[..., None]
    near = smoothstep(0.0, 0.08, np.maximum(k1, k2) + 0.05)  # shadow skirt around each crown
    soilc = soilc * (1 - 0.35 * near * (1 - m))[..., None]
    mm = m[..., None]
    cv.alb = np.clip(col * mm + soilc * (1 - mm), 0, 1)
    cv.h = (0.35 * np.sqrt(k1) + 0.2 * np.sqrt(k2)) * (0.6 + 0.4 * holes) + 0.01 * leaf * m
    cv.rough = 0.95 - 0.07 * m
    cv.mask[:] = 0


def dirt(cv: Canvas, seed: int, tone: str, pebbles: float, grain: float = 1.0) -> None:
    sh = cv.shape
    fine = tg.fbm(sh, seed, beta=0.4, fmin=1)
    mot = cv.noise(seed + 1, 1.5, beta=2.6, octaves_m=0.03)
    c = tg.worley(sh, (int(cv.w_m * 18), int(cv.h_m * 18)), seed + 2)
    peb = np.clip(1 - c.f1 / (cv.px / (cv.w_m * 18) * (0.25 + 0.3 * c.rnd)), 0, 1) * (c.rnd < pebbles)
    clod = cv.noise(seed + 3, 0.12, beta=1.6, octaves_m=0.01)
    col = hex_rgb(tone) * (1 + 0.05 * grain * fine + 0.08 * mot)[..., None]
    pc = _palette_pick(["#C9C2B4", "#8F857A", "#A99B88", "#6E655C", "#BFAE93"], c.rnd)
    pm = smoothstep(0.0, 0.25, peb)[..., None]
    col = col * (1 - 0.12 * smoothstep(0.5, 1.5, -clod))[..., None]
    col = col * (1 - pm) + pc * (0.85 + 0.2 * peb[..., None]) * pm
    cv.alb = np.clip(col, 0, 1)
    cv.h = 0.0008 * fine * grain + 0.008 * peb ** 0.5 + 0.003 * clod
    cv.rough = 0.95 - 0.1 * pm[..., 0]
    cv.mask[:] = 0


def ground_mulch(cv: Canvas) -> None:
    """Shredded bark mulch (planting beds): random elongated chips with deep gaps."""
    sh = cv.shape
    hh = np.zeros(sh)
    col = np.zeros((*sh, 3))
    pal = ["#5E4130", "#7A5538", "#4A3326", "#6B4A33", "#8A6444", "#3E2B20"]
    for k in range(6):
        ang = k * 1.05
        c = tg.worley(sh, (60, 60), 120 + k, jitter=1.0)
        ca, sa = math.cos(ang), math.sin(ang)
        u = c.dx * ca + c.dy * sa
        v = -c.dx * sa + c.dy * ca
        d = np.hypot(u / 2.6, v / 0.8)
        chip = np.clip(1 - d / (cv.px / 60 * 0.42), 0, 1) * (c.rnd < 0.7)
        z = chip ** 0.5 * (0.01 + 0.006 * k)
        sel = z > hh
        hh = np.where(sel, z, hh)
        col = np.where(sel[..., None], _palette_pick(pal, (c.rnd * 3.7) % 1.0) * (0.75 + 0.35 * chip[..., None]), col)
    gaps = hh <= 1e-4
    col = np.where(gaps[..., None], hex_rgb("#2A1E16"), col)
    fib = tg.fbm(sh, 130, beta=0.3, fmin=60)
    cv.alb = np.clip(col * (1 + 0.06 * fib[..., None]), 0, 1)
    cv.h = hh + 0.0005 * fib
    cv.rough[:] = 0.93
    cv.mask[:] = 0


def ground_pool(cv: Canvas) -> None:
    """Pool water over light plaster: caustic web in the base color, gentle ripple normals, glossy."""
    sh = cv.shape
    c = tg.worley(sh, (9, 9), 141, jitter=1.0)
    c2 = tg.worley(sh, (13, 13), 142, jitter=1.0)
    web = smoothstep(9.0, 0.0, c.f2 - c.f1) * 0.6 + smoothstep(7.0, 0.0, c2.f2 - c2.f1) * 0.4
    deep = cv.noise(143, 2.0, beta=2.6)
    col = hex_rgb("#3FA6C4") * (1 + 0.06 * deep[..., None]) + np.array([0.25, 0.3, 0.3]) * web[..., None] * 0.55
    rip = cv.noise(144, 0.5, beta=2.6, octaves_m=0.02)
    cv.alb = np.clip(col, 0, 1)
    cv.h = 0.004 * rip
    cv.rough[:] = 0.03
    cv.mask[:] = 0


# ---------------------------------------------------------------------------
# lane markings (RGBA decal sheet)
# ---------------------------------------------------------------------------

MARKINGS = [
    # name, color, column width m, period m, [(x0, x1) line edges m], dash (on m) or None
    ("white_solid_4in", "#ECECE6", 0.3, 8.0, [(0.1, 0.2)], None),
    ("white_dashed_4in", "#ECECE6", 0.3, 12.0, [(0.1, 0.2)], 3.0),
    ("yellow_solid_4in", "#E3B23C", 0.3, 8.0, [(0.1, 0.2)], None),
    ("yellow_double_4in", "#E3B23C", 0.5, 8.0, [(0.12, 0.22), (0.3, 0.4)], None),
    ("yellow_dashed_4in", "#E3B23C", 0.3, 12.0, [(0.1, 0.2)], 3.0),
    ("white_solid_8in", "#ECECE6", 0.4, 8.0, [(0.1, 0.3)], None),
    ("white_bar_12in", "#ECECE6", 0.5, 4.0, [(0.1, 0.4)], None),
    ("crosswalk_bar_24in", "#ECECE6", 1.0, 3.0, [(0.2, 0.8)], None),
]


def markings_sheet(size: int = 1024) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cols = len(MARKINGS)
    cw = size // cols
    out = np.zeros((size, size, 4))
    meta = []
    for i, (name, hexc, width, period, lines, dash) in enumerate(MARKINGS):
        cv = Canvas(width, period, cw)
        yy = np.linspace(0, period, size, endpoint=False)[::-1][:, None] * np.ones((1, cw))
        xx = (np.arange(cw)[None, :] + 0.5) * (width / cw) * np.ones((size, 1))
        cov = np.zeros((size, cw))
        mpp = width / cw
        for x0, x1 in lines:
            cov = np.maximum(cov, np.clip((xx - x0) / mpp + 0.5, 0, 1) * np.clip((x1 - xx) / mpp + 0.5, 0, 1))
        if dash:
            ymod = yy % period
            cov = cov * np.clip(np.minimum(ymod - 0.0, dash - ymod) / (period / size) + 0.5, 0, 1)
        n = tg.fbm((size, cw), 150 + i, beta=1.0, fmin=2)
        speck = tg.fbm((size, cw), 160 + i, beta=0.0, fmin=1)
        wear = np.clip(0.9 + 0.2 * n - 0.25 * smoothstep(1.2, 2.2, speck), 0, 1)
        a = cov * wear
        col = hex_rgb(hexc) * (0.93 + 0.05 * n[..., None])
        out[:, i * cw : (i + 1) * cw, :3] = col
        out[:, i * cw : (i + 1) * cw, 3] = a
        meta.append({"name": name, "column": i, "px": [i * cw, 0, cw, size], "uv": [i * cw / size, 0.0, (i + 1) * cw / size, 1.0],
                     "world_width_m": width, "period_m": period, "line_edges_m": lines, "dash_m": dash, "color_srgb": hexc,
                     "mapping": "u = (meters across the line from the column's left edge) / world_width_m; v = meters along / period_m"})
        del cv
    return out, meta


# ---------------------------------------------------------------------------
# atlas definitions
# ---------------------------------------------------------------------------


@dataclass
class CellDef:
    name: str
    col: int
    row: int
    fn: Callable[[Canvas], None]
    world: tuple[float, float]  # meters covered by ONE cell's inner square (x, y)
    span: int = 1  # cells wide (garage doors)
    desc: str = ""
    tint: bool = False
    ao_r: float = 0.08


FACADE_WALLS = [
    CellDef("stucco_smooth", 0, 0, wall_stucco("smooth", 1), (3, 3), desc="Santa Barbara smooth trowel stucco", tint=True),
    CellDef("stucco_sand", 1, 0, wall_stucco("sand", 2), (3, 3), desc="20/30 sand finish stucco (most tract homes)", tint=True),
    CellDef("stucco_lace", 2, 0, wall_stucco("lace", 3), (3, 3), desc="lace / skip-trowel stucco", tint=True),
    CellDef("stucco_catface", 3, 0, wall_stucco("catface", 4), (3, 3), desc="smooth stucco with recessed cat faces", tint=True),
    CellDef("stucco_weathered", 0, 1, wall_stucco("weathered", 5), (3, 3), desc="sand finish, mottled, hairline cracks", tint=True),
    CellDef("stucco_scored", 1, 1, wall_stucco("scored", 6), (3, 3),
            desc="sand finish with reveals at every floor line (v=0) and bay line (u=0): schools, commercial", tint=True),
    CellDef("stone_veneer", 2, 1, wall_stone, (3, 3), desc="stacked ledgestone veneer (wainscots, columns, monuments)", ao_r=0.05),
    CellDef("glass_curtain", 3, 1, wall_glass_curtain, (3, 3), desc="curtain / storefront glazing, mullions every 1.5 m, spandrel 0-0.75 m", ao_r=0.1),
]

FACADE_OPENINGS = [
    CellDef("window_slider", 0, 0, op_window_slider, (3, 3), desc="white vinyl 6040 slider (1.8 x 1.2 m, sill 1.0 m), foam trim, mini-blinds", ao_r=0.15),
    CellDef("window_pair", 1, 0, op_window_pair, (3, 3), desc="pair of single-hung windows with colonial grids, sill 0.85 m", ao_r=0.15),
    CellDef("window_picture", 2, 0, op_window_picture, (3, 3), desc="2.4 m picture window with side lites and grids", ao_r=0.15),
    CellDef("window_small", 3, 0, op_window_small, (3, 3), desc="small obscure-glass bath window, high sill 1.65 m", ao_r=0.15),
    CellDef("door_front", 0, 1, op_door_front, (3, 3), desc="recessed 8 ft entry door (stained), sidelite, coach light, step", ao_r=0.2),
    CellDef("door_slider", 1, 1, op_door_slider, (3, 3), desc="8 ft white vinyl patio slider, screen, vertical blinds", ao_r=0.15),
    CellDef("garage_2car", 2, 1, op_garage_2car, (3, 3), span=2, desc="16 x 7 ft raised-panel steel garage door in a 6 m wide 2-cell span", ao_r=0.15),
    CellDef("garage_3car", 0, 2, op_garage_3car, (3, 3), span=3, desc="16 ft + 8 ft carriage doors with top lites in a 9 m wide 3-cell span", ao_r=0.15),
    CellDef("window_arched", 3, 2, op_window_arched, (3, 3), desc="Mediterranean arched-top window with grids and stucco surround", ao_r=0.15),
    CellDef("storefront", 0, 3, op_storefront, (3, 3), desc="dark bronze aluminum storefront bay with lit interior + transom", ao_r=0.1),
    CellDef("storefront_sign", 1, 3, op_storefront_sign, (3, 3), desc="upper commercial band: cornice, sign field, gooseneck lights (wall tint)", ao_r=0.2),
    CellDef("school_window_band", 2, 3, op_school_window_band, (3, 3), desc="classroom ribbon window, clear anodized, sunshade louver", ao_r=0.2),
    CellDef("school_door", 3, 3, op_school_door, (3, 3), desc="pair of hollow-metal doors with vision lites and transom", ao_r=0.12),
]

SPANISH = {
    "terracotta": ["#B4532E", "#A84B2A", "#BE5C33", "#AE5030", "#9F4628"],
    "blend": ["#A24A2A", "#B85F36", "#8A3E25", "#C2703F", "#6E3524", "#9C5233", "#B0663E"],
    "brown": ["#6B4A35", "#7D5639", "#5A3D2D", "#8B6443", "#4E3628", "#76513A"],
}

ROOFS = [
    CellDef("s_tile_terracotta", 0, 0, roof_s_tile(SPANISH["terracotta"], 201, flash=0.12), (4, 4), desc="Spanish S tile, terracotta"),
    CellDef("s_tile_blend", 1, 0, roof_s_tile(SPANISH["blend"], 202, flash=0.3), (4, 4), desc="Spanish S tile, mixed red/brown/orange flashed blend"),
    CellDef("s_tile_brown", 2, 0, roof_s_tile(SPANISH["brown"], 203, flash=0.25), (4, 4), desc="Spanish S tile, brown blend"),
    CellDef("s_tile_aged", 3, 0, roof_s_tile(SPANISH["terracotta"], 204, aged=0.5, flash=0.2), (4, 4), desc="Spanish S tile, sun-faded with lichen / efflorescence"),
    CellDef("flat_tile_brown", 0, 1, roof_flat_tile(["#5E4636", "#6F5341", "#4F3A2E", "#7A5C46", "#664B39"], 211), (4, 4), desc="flat concrete tile, brown blend"),
    CellDef("flat_tile_grey", 1, 1, roof_flat_tile(["#6D6E6C", "#7C7D7A", "#5E5F5E", "#888884", "#727370"], 212), (4, 4), desc="flat concrete tile, grey blend"),
    CellDef("flat_tile_charcoal", 2, 1, roof_flat_tile(["#3E3F40", "#4A4A4B", "#353637", "#555557"], 213), (4, 4), desc="flat concrete tile, charcoal slate"),
    CellDef("flat_tile_sandstone", 3, 1, roof_flat_tile(["#B39C7E", "#A68D70", "#C2AC8F", "#9A8266"], 214, slate=False), (4, 4), desc="flat concrete tile, sandstone shake"),
    CellDef("flat_tpo", 0, 2, roof_tpo(0.0, 221), (4, 4), desc="white TPO membrane, 2 m sheets"),
    CellDef("flat_tpo_grime", 1, 2, roof_tpo(1.0, 222), (4, 4), desc="TPO with HVAC soot, rust spots, ponding stains, foot path"),
    CellDef("flat_gravel", 2, 2, roof_gravel, (4, 4), desc="ballasted gravel roof"),
    CellDef("flat_modbit", 3, 2, roof_modbit, (4, 4), desc="grey granulated modified bitumen, 1 m laps"),
    CellDef("solar_panel", 0, 3, roof_solar, (4, 4), desc="residential PV array, 1 x 2 m mono panels (portrait, v up the slope)"),
    CellDef("barrel_mission", 1, 3, roof_barrel, (4, 4), desc="two-piece mission barrel clay tile"),
    CellDef("standing_seam", 2, 3, roof_standing_seam, (4, 4), desc="standing-seam metal, ribs every 0.44 m along v"),
    CellDef("concrete_deck", 3, 3, roof_concrete_deck, (4, 4), desc="concrete roof / parking deck with 2 m joints"),
]

GROUND = [
    CellDef("asphalt_fresh", 0, 0, lambda cv: asphalt(cv, 301, 0.0), (4, 4), desc="fresh dark asphalt"),
    CellDef("asphalt_worn", 1, 0, lambda cv: asphalt(cv, 302, 0.8, oil=0.5, cracks=0.6, seal=1.0), (4, 4),
            desc="sun-faded asphalt, exposed aggregate, crack sealant, patches, oil (v along the lane, wheel paths at u 0.25/0.75)"),
    CellDef("asphalt_parking", 2, 0, lambda cv: asphalt(cv, 303, 0.5, oil=1.0, cracks=0.25, seal=0.6), (4, 4),
            desc="parking-lot asphalt, oil drips centered in u (stall center)"),
    CellDef("concrete_sidewalk", 3, 0, lambda cv: concrete(cv, 304, joint=1.5), (3, 3), desc="broom-finish sidewalk, tooled joints every 1.5 m"),
    CellDef("concrete_driveway", 0, 1, lambda cv: concrete(cv, 305, joint=3.0, tone="#B8B1A6", tracks=True), (6, 6),
            desc="driveway concrete, joints every 3 m, tire paths along v (u across the driveway)"),
    CellDef("curb_gutter", 1, 1, ground_curb, (3, 0.75), desc="curb & gutter profile: u along the curb (3 m), v across: 0-0.6 gutter pan, 0.6-0.8 face, 0.8-1 top"),
    CellDef("pavers", 2, 1, ground_pavers, (3, 3), desc="interlocking concrete pavers, running bond, tan blend"),
    CellDef("concrete_plaza", 3, 1, lambda cv: concrete(cv, 306, joint=3.0, broom=False, tone="#C2BDB3"), (6, 6), desc="smooth plaza / school hardscape, 3 m joints"),
    CellDef("grass_lawn", 0, 2, lambda cv: grass(cv, 311), (2, 2), desc="irrigated lawn"),
    CellDef("grass_patchy", 1, 2, lambda cv: grass(cv, 312, dry=0.7), (2, 2), desc="summer-stressed lawn with dormant patches"),
    CellDef("chaparral", 2, 2, lambda cv: scrub(cv, 313, ["#4F5A3A", "#5F6745", "#3F4A2F", "#6E7650", "#565E3E"], "#8A7456", 0.95, (6, 6), 0.85), (4, 4),
            desc="dense chaparral (chamise / scrub oak) seen from above", ao_r=0.6),
    CellDef("coastal_sage", 3, 2, lambda cv: scrub(cv, 314, ["#8C9474", "#A0A585", "#7A8064", "#93977A", "#6E7358"], "#B49B78", 0.7, (7, 7), 0.6), (4, 4),
            desc="coastal sage scrub: grey-green clumps over tan soil", ao_r=0.5),
    CellDef("decomposed_granite", 0, 3, lambda cv: dirt(cv, 321, "#C2A47C", 0.35, grain=1.4), (2, 2), desc="decomposed granite paths / xeriscape"),
    CellDef("bare_dirt", 1, 3, lambda cv: dirt(cv, 322, "#9C7B58", 0.2), (4, 4), desc="bare graded soil / trail"),
    CellDef("mulch", 2, 3, ground_mulch, (2, 2), desc="shredded bark mulch (planting beds)"),
    CellDef("pool_water", 3, 3, ground_pool, (2, 2), desc="pool water over light plaster (caustics in base color, glossy)"),
]


@dataclass
class AtlasDef:
    name: str
    size: tuple[int, int]  # px (w, h)
    cells: list[CellDef]
    mask: bool = False
    desc: str = ""


ATLASES = [
    AtlasDef("facade_walls", (2048, 1024), FACADE_WALLS, desc="tileable wall finishes, 3 m x 3 m per cell (one floor-bay)"),
    AtlasDef("facade_openings", (2048, 2048), FACADE_OPENINGS, mask=True,
             desc="floor-bay tiles with openings, 3 m x 3 m per cell; wall areas are stucco_sand with mask 1"),
    AtlasDef("roofs", (2048, 2048), ROOFS, desc="roof coverings, 4 m x 4 m per cell (u along the eave, v up the slope)"),
    AtlasDef("ground", (2048, 2048), GROUND, desc="ground / paving / landscape, world size per cell in world_size_m"),
]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def render_cell(cd: CellDef) -> dict[str, np.ndarray]:
    cv = Canvas(cd.world[0] * cd.span, cd.world[1], INNER * cd.span, py_px=INNER)
    cd.fn(cv)
    return cv.maps(ao_radius_m=cd.ao_r)


def _uv(px: list[int], size: tuple[int, int]) -> list[float]:
    """Pixel rect [x, y, w, h] (top-left origin) -> [u0, v0, u1, v1] with v measured from the BOTTOM."""
    x, y, w, h = px
    W, H = size
    return [round(x / W, 6), round(1 - (y + h) / H, 6), round((x + w) / W, 6), round(1 - y / H, 6)]


def build_atlas(ad: AtlasDef, out_dir: Path, log: Callable[[str], None] = print) -> dict[str, Any]:
    W, H = ad.size
    alb = np.zeros((H, W, 3))
    nrm = np.tile(np.array([0.5, 0.5, 1.0]), (H, W, 1))
    orm = np.zeros((H, W, 3))
    orm[..., 0] = 1.0
    orm[..., 1] = 0.8
    mask = np.zeros((H, W, 3))
    cells_meta = []
    for cd in ad.cells:
        m = render_cell(cd)
        stack = {
            "albedo": m["albedo"], "normal": m["normal"],
            "orm": np.stack([m["ao"], m["rough"], m["metal"]], axis=-1),
            "mask": np.stack([m["mask"], m["emit"], np.zeros_like(m["mask"])], axis=-1),
        }
        padded = {k: tg.pad_wrap(v, GUTTER) for k, v in stack.items()}
        for k in range(cd.span):
            x0 = (cd.col + k) * CELL
            y0 = cd.row * CELL
            sl = (slice(y0, y0 + CELL), slice(x0, x0 + CELL))
            src = (slice(0, CELL), slice(k * INNER, k * INNER + CELL))
            alb[sl] = padded["albedo"][src]
            nrm[sl] = padded["normal"][src]
            orm[sl] = padded["orm"][src]
            mask[sl] = padded["mask"][src]
        lin = tg.srgb_to_linear(m["albedo"])
        wall = m["mask"] > 0.99
        entry: dict[str, Any] = {
            "name": cd.name,
            "index": len(cells_meta),
            "description": cd.desc,
            "span_cells": cd.span,
            "world_size_m": [cd.world[0], cd.world[1]],
            "texels_per_m": round(INNER / cd.world[0], 1),
            "cells": [],
            "tintable": bool(cd.tint),
            "mean_albedo_linear": [round(float(v), 4) for v in lin.reshape(-1, 3).mean(axis=0)],
            "mean_roughness": round(float(m["rough"].mean()), 3),
        }
        if ad.mask:
            entry["wall_fraction"] = round(float(wall.mean()), 3)
        for k in range(cd.span):
            px = [(cd.col + k) * CELL, cd.row * CELL, CELL, CELL]
            inner = [px[0] + GUTTER, px[1] + GUTTER, INNER, INNER]
            entry["cells"].append({"px": px, "inner_px": inner, "uv_inner": _uv(inner, ad.size)})
        cells_meta.append(entry)
        log(f"  {ad.name}/{cd.name}")
    files = {}
    tg.save_jpg(out_dir / f"{ad.name}_albedo.jpg", alb, 90)
    tg.save_jpg(out_dir / f"{ad.name}_normal.jpg", nrm, 93)
    tg.save_jpg(out_dir / f"{ad.name}_orm.jpg", orm, 90)
    files = {"albedo": f"{ad.name}_albedo.jpg", "normal": f"{ad.name}_normal.jpg", "orm": f"{ad.name}_orm.jpg"}
    if ad.mask:
        tg.save_png(out_dir / f"{ad.name}_mask.png", mask, "RGB")
        files["mask"] = f"{ad.name}_mask.png"
    half = {}
    from PIL import Image

    for k in ("albedo", "normal", "orm"):
        im = Image.open(out_dir / files[k])
        im = im.resize((W // 2, H // 2), Image.LANCZOS)
        p = out_dir / f"{ad.name}_{k}_1k.jpg"
        im.save(p, quality=88, optimize=True, progressive=True)
        half[k] = p.name
    return {"name": ad.name, "description": ad.desc, "size_px": [W, H], "cell_px": CELL, "gutter_px": GUTTER, "inner_px": INNER,
            "files": files, "files_half_res": half, "cells": cells_meta}


MAT_IDS = {
    0: {"name": "stucco_wall", "atlas": "facade_walls",
        "variants": ["stucco_smooth", "stucco_sand", "stucco_lace", "stucco_catface", "stucco_weathered", "stucco_scored", "stone_veneer"],
        "tint": "COLOR_0 (building wall color, sRGB vertex color): base = COLOR_0_linear * texel_linear / cell.mean_albedo_linear",
        "openings": "optional bay tiles from facade_openings chosen per (bay, floor) cell; see bay_grammar"},
    1: {"name": "tile_roof", "atlas": "roofs",
        "variants": ["s_tile_terracotta", "s_tile_blend", "s_tile_brown", "s_tile_aged", "barrel_mission",
                     "flat_tile_brown", "flat_tile_grey", "flat_tile_charcoal", "flat_tile_sandstone", "solar_panel"],
        "tint": "none (texel colors are real tile blends); optionally lerp 25% toward COLOR_0 for per-house variety"},
    2: {"name": "flat_roof", "atlas": "roofs",
        "variants": ["flat_tpo", "flat_tpo_grime", "flat_gravel", "flat_modbit", "concrete_deck", "standing_seam"],
        "tint": "none"},
    3: {"name": "glass", "atlas": "facade_walls", "variants": ["glass_curtain"], "tint": "none",
        "notes": "roughness 0.05: let the env map supply reflections; G of facade_openings_mask marks glass that can glow at night"},
    4: {"name": "trim", "atlas": "facade_walls", "variants": ["stucco_smooth", "stone_veneer"],
        "tint": "COLOR_0 as for walls (pass a trim color: white / cream / dark bronze)"},
    5: {"name": "garage_door", "atlas": "facade_openings", "variants": ["garage_2car", "garage_3car"],
        "notes": "either map whole garage bays with facade UVs (span cells), or a door quad onto the cell's opening_rects_m.door"},
}


# openings inside facade_openings cells, meters from the cell's (span's) bottom-left corner: [x0, y0, x1, y1]
OPENING_RECTS: dict[str, dict[str, list[list[float]]]] = {
    "window_slider": {"glass": [[0.6, 1.0, 2.4, 2.2]]},
    "window_pair": {"glass": [[0.45, 0.85, 1.43, 2.4], [1.57, 0.85, 2.55, 2.4]]},
    "window_picture": {"glass": [[0.3, 0.75, 2.7, 2.4]]},
    "window_small": {"glass": [[1.05, 1.65, 1.95, 2.3]]},
    "window_arched": {"glass": [[0.85, 0.8, 2.15, 2.65]]},
    "door_front": {"door": [[0.95, 0.0, 1.9, 2.44]], "glass": [[1.98, 0.05, 2.36, 2.44]]},
    "door_slider": {"door": [[0.3, 0.04, 2.7, 2.1]]},
    "garage_2car": {"door": [[0.56, 0.0, 5.44, 2.13]]},
    "garage_3car": {"door": [[0.45, 0.0, 5.33, 2.13], [6.13, 0.0, 8.57, 2.13]]},
    "storefront": {"glass": [[0.0, 0.38, 3.0, 2.94]]},
    "storefront_sign": {},
    "school_window_band": {"glass": [[0.0, 0.95, 3.0, 2.4]]},
    "school_door": {"door": [[0.58, 0.0, 2.42, 2.13]], "glass": [[0.58, 2.19, 2.42, 2.69]]},
}


def manifest(atlases: list[dict[str, Any]], markings: list[dict[str, Any]]) -> dict[str, Any]:
    by = {a["name"]: a for a in atlases}
    for c in by["facade_openings"]["cells"]:
        c["opening_rects_m"] = OPENING_RECTS.get(c["name"], {})
    return {
        "format": "redraw-materials",
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": ".venv-blender/bin/python blender/build_all_assets.py --only materials (blender/rdlib/matgen.py, texgen.py)",
        "base_path": "assets/materials/",
        "conventions": {
            "uv_origin": "uv rects use v measured from the BOTTOM of the image (three.js TextureLoader flipY=true, OpenGL, Blender). "
                         "px rects are [x, y, w, h] from the TOP-left.",
            "albedo": "sRGB JPEG (texture.colorSpace = SRGBColorSpace)",
            "normal": "tangent-space, OpenGL/glTF convention (+X right, +Y up = toward v=1); linear (NoColorSpace)",
            "orm": "R ambient occlusion, G roughness, B metalness (glTF metallicRoughness packing; aoMap reads R); linear",
            "mask": "facade_openings_mask.png R: 1 = wall (stucco: tint with the wall color), 0.5 = paintable near-white element "
                    "(garage / service doors: optional accent tint), 0 = fixed colors; G: glass that can glow at night (emissive "
                    "mask); B unused",
            "tileable": "every cell's inner square tiles seamlessly; cells carry a 16 px gutter of wrapped content",
            "atlas_lookup": "local = fract(mesh_uv); atlas_uv = mix(cell.uv_inner.xy, cell.uv_inner.zw, local) "
                            "(use textureGrad with the derivatives of mesh_uv * inner_size to avoid the fract seam at mip edges)",
            "spans": "garage_2car / garage_3car cover 2 / 3 consecutive cells in a row: span cell k holds local x in [k, k+1) of the "
                     "span (meters x / 3)",
        },
        "vertex_attributes": {
            "_MAT": {str(k): v["name"] for k, v in MAT_IDS.items()},
            "_VARIANT": "integer index into materials[_MAT].variants (float attribute, integral)",
            "COLOR_0": "tint for tintable cells (walls, trim); roofs ignore it unless the client wants variety",
            "TEXCOORD_0": "see uv_conventions",
        },
        "uv_conventions": {
            "facade": "walls (_MAT 0, 3, 4, 5): u = meters along the wall from its start corner / 3.0, v = meters above the wall base / 3.0. "
                      "One UV unit = one 3 m x 3 m floor-bay = one atlas cell. floor = floor(v), bay = floor(u).",
            "roof_pitched": "_MAT 1: u = meters along the eave direction / 4.0, v = meters up the slope (in the roof plane, from the "
                            "eave) / 4.0. One UV unit = one 4 m cell. S tiles run in columns along v.",
            "roof_flat": "_MAT 2: planar u = scene x / 4.0, v = -scene z / 4.0",
            "ground": "planar u = scene x / world_size_m[0], v = -scene z / world_size_m[1] unless the cell says otherwise "
                      "(curb_gutter, concrete_driveway, asphalt_worn are oriented: u across / v along as described)",
            "markings": "ground_markings.png columns: u across the line (world_width_m), v along (period_m)",
        },
        "materials": {str(k): v for k, v in MAT_IDS.items()},
        "bay_grammar": {
            "description": "Suggested assignment of facade_openings cells to (bay, floor) cells of _MAT 0 walls; a deterministic hash "
                           "of (building id, wall index, bay, floor) picks among the options. Bays at a wall's ends that are "
                           "narrower than 1.2 m stay plain wall.",
            "house": {"street_wall_floor0": ["garage_2car|garage_3car at one end (span)", "door_front next to it",
                                             "window_slider / window_picture / window_arched elsewhere"],
                      "other_walls_floor0": {"window_slider": 0.3, "door_slider": 0.15, "window_pair": 0.15, "plain": 0.4},
                      "upper_floors": {"window_slider": 0.3, "window_pair": 0.2, "window_arched": 0.08, "window_small": 0.12, "plain": 0.3}},
            "commercial": {"floor0_front": ["storefront"], "floor0_other": {"plain": 0.8, "school_door": 0.2},
                           "floor1_front": ["storefront_sign"], "upper": {"glass_curtain(_MAT 3)": 0.5, "window_pair": 0.2, "plain": 0.3}},
            "school": {"floor0": {"school_window_band": 0.7, "school_door": 0.15, "plain": 0.15},
                       "upper": {"school_window_band": 0.8, "plain": 0.2}, "wall_variant": "stucco_scored"},
            "apartments": {"floor0": {"window_slider": 0.35, "door_slider": 0.25, "door_front": 0.1, "plain": 0.3},
                           "upper": {"window_slider": 0.4, "door_slider": 0.25, "window_small": 0.1, "plain": 0.25}},
        },
        "atlases": by,
        "markings": {"file": "ground_markings.png", "size_px": [1024, 1024], "format": "RGBA sRGB, alpha = paint coverage with wear",
                     "columns": markings, "verified": False,
                     "notes": "dash 3 m / gap 9 m approximates the 10/30 ft local-street pattern (verified: false)"},
        "pipeline_notes": [
            "Hero campus glbs (pipeline/hero_overrides) carry _MAT and _VARIANT vertex attributes and facade/roof UVs in this "
            "convention plus COLOR_0 tints; pipeline.glb.load_glb_meshes currently drops custom attributes, so copy accessors "
            "named _MAT / _VARIANT into MeshData.custom to keep them in the building tiles.",
            "Vertex colors stay a valid fallback: a client without atlases renders COLOR_0 as before.",
        ],
    }


def build_all(out_dir: Path, log: Callable[[str], None] = print) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    atl = []
    for ad in ATLASES:
        log(f"materials: atlas {ad.name} {ad.size[0]}x{ad.size[1]}")
        atl.append(build_atlas(ad, out_dir, log))
    sheet, meta = markings_sheet()
    tg.save_png(out_dir / "ground_markings.png", sheet, "RGBA")
    man = manifest(atl, meta)
    p = out_dir / "materials_manifest.json"
    p.write_text(json.dumps(man, indent=1) + "\n")
    log(f"materials: wrote {p}")
    return p
