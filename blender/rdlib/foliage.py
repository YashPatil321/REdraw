"""Procedural foliage textures and alpha-card vegetation (trees, palms, shrubs, grass).

Every plant is ONE mesh with ONE material (`foliage`: glTF alphaMode MASK,
doubleSided) whose base color texture is a small per-species atlas generated here:

    512 x 512 RGBA PNG
    x in [0, 448):   leaf cells, 2 columns x 2 rows of 224 x 256 px (a cell may span both columns)
    x in [448, 512): bark strip, full height (opaque), mapped onto the trunk tube

Leaf cards are quads scattered through the canopy volume. Their vertex normals are
bent outward from the canopy center ("spherical normals"), so a cloud of flat cards
shades like a soft rounded crown in three.js and Cycles alike.

Species are the common trees of 4S Ranch / Del Sur streetscapes and canyons:
coast live oak (native canyons and slopes), Mexican fan palm and queen palm
(arterials, commercial centers), eucalyptus (windbreaks along older edges),
jacaranda and a broadleaf parkway tree (residential streets), plus irrigated
shrubs and ornamental grass. Sizes are typical mature suburban specimens.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh import Part

TEX = 1024
LEAF_W = 896
CELL_W, CELL_H = 448, 512
RES = 2  # drawing functions work in "design" pixels (224 x 256 per cell) and output RES x that
SS = 4  # canvas scale relative to design pixels (2x supersampling of the RES output)


# ---------------------------------------------------------------------------
# texture drawing (PIL)
# ---------------------------------------------------------------------------


def _hex(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _jit(c: np.ndarray, rng: np.random.Generator, amt: float = 0.12) -> tuple[int, int, int, int]:
    k = 1.0 + rng.uniform(-amt, amt)
    hue = rng.uniform(-amt, amt) * 0.4
    col = c * k + np.array([hue * 40, 0, -hue * 30])
    col = np.clip(col, 0, 255)
    return int(col[0]), int(col[1]), int(col[2]), 255


def _canvas(w: int, h: int):  # noqa: ANN202
    from PIL import Image, ImageDraw

    im = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def _finish(im, w: int, h: int):  # noqa: ANN001, ANN202
    """Downsample and bleed leaf colors into transparent texels (no dark fringes in mip levels)."""
    from PIL import Image, ImageFilter

    w, h = w * RES, h * RES
    im = im.resize((w, h), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    rgb, al = a[..., :3], a[..., 3:4] / 255.0
    if al.max() > 0:
        prem = Image.fromarray(np.clip(rgb * al, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5))
        wgt = Image.fromarray(np.clip(al[..., 0] * 255, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5))
        fill = np.asarray(prem).astype(np.float32) / np.maximum(np.asarray(wgt).astype(np.float32)[..., None] / 255.0, 1e-3)
        rgb = np.where(al > 0.05, rgb, np.clip(fill, 0, 255))
    out = np.concatenate([np.clip(rgb, 0, 255), a[..., 3:4]], axis=-1).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _leaf_poly(cx: float, cy: float, length: float, width: float, ang: float, n: int = 6) -> list[tuple[float, float]]:
    pts = []
    for i in range(n + 1):
        t = i / n
        r = width * math.sin(math.pi * t) ** 0.8
        pts.append((t * length, r))
    pts += [(t, -r) for t, r in reversed(pts[1:-1])]
    c, s = math.cos(ang), math.sin(ang)
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]


def _leaf(d, x: float, y: float, ln: float, wd: float, ang: float, c: np.ndarray, rng: np.random.Generator) -> None:  # noqa: ANN001
    """One leaf: base color, a slightly darker half (folded blade), a lighter midrib."""
    pts = _leaf_poly(x, y, ln, wd, ang)
    d.polygon(pts, fill=_jit(c, rng, 0.12))
    half = len(pts) // 2
    d.polygon(pts[: half + 1], fill=_jit(c * 0.86, rng, 0.06))
    if ln > 6 * SS:
        tip = (x + math.cos(ang) * ln * 0.92, y + math.sin(ang) * ln * 0.92)
        d.line([(x, y), tip], fill=_jit(np.minimum(c * 1.25 + 12, 255), rng, 0.05), width=max(1, int(0.5 * SS)))


def draw_broadleaf(w: int, h: int, seed: int, colors: Sequence[str], leaf_len: tuple[float, float],
                   leaf_w: float, n_leaves: int, twig: str = "#5A4A3A", clumps: int = 7, spread: float = 0.42,
                   flowers: str | None = None, flower_frac: float = 0.0, droop: float = 0.0,
                   sun_bias: float = 0.25):  # noqa: ANN201
    """A branchlet seen side-on: twigs radiating from the bottom with leaf clusters."""
    rng = np.random.default_rng(seed)
    im, d = _canvas(w, h)
    W, H = w * SS, h * SS
    base = (W / 2, H * 0.98)
    cols = [_hex(c) for c in colors]
    centers = []
    for _ in range(clumps):
        a = rng.uniform(-1.15, 1.15)
        r = rng.uniform(0.35, 0.8) * H
        cx = base[0] + math.sin(a) * r * 0.62
        cy = base[1] - math.cos(a) * r
        centers.append((cx, cy))
        mx = (base[0] + cx) / 2 + rng.normal(0, 0.08) * W
        my = (base[1] + cy) / 2 + rng.normal(0, 0.05) * H
        pts = [((1 - t) ** 2 * base[0] + 2 * (1 - t) * t * mx + t * t * cx, (1 - t) ** 2 * base[1] + 2 * (1 - t) * t * my + t * t * cy)
               for t in np.linspace(0, 1, 9)]
        tw = _jit(_hex(twig), rng, 0.1)
        for j in range(8):
            d.line([pts[j], pts[j + 1]], fill=tw, width=max(1, int((3.2 - 2.4 * j / 8) * SS)))
    centers.append((W / 2, H * 0.35))
    for _i in range(n_leaves):
        cx, cy = centers[rng.integers(len(centers))]
        rr = rng.normal(0, spread * 0.5) * W * 0.5
        aa = rng.uniform(0, 2 * math.pi)
        x = float(np.clip(cx + math.cos(aa) * rr, 8, W - 8))
        y = float(np.clip(cy + math.sin(aa) * rr * 0.9, 8, H - 8))
        ln = rng.uniform(*leaf_len) * SS
        ang = rng.uniform(0, 2 * math.pi) if droop == 0 else math.pi / 2 + rng.normal(0, droop)
        top = 1.0 - y / H  # higher leaves are sunlit
        c = cols[min(len(cols) - 1, int(rng.random() * len(cols)))]
        c = c * (1 - sun_bias + 2 * sun_bias * top)
        if flowers and rng.random() < flower_frac:
            c = _hex(flowers)
            ln *= 0.6
        depth = 0.72 + 0.28 * _i / n_leaves  # leaves drawn first sit deeper in the cluster: darker
        _leaf(d, x, y, ln, ln * leaf_w, ang, c * depth, rng)
    return _finish(im, w, h)


def draw_hanging(w: int, h: int, seed: int, colors: Sequence[str], n_strands: int = 26, leaves: int = 16,
                 stem: str = "#7A6250"):  # noqa: ANN201
    """Eucalyptus: drooping strands of long sickle leaves."""
    rng = np.random.default_rng(seed)
    im, d = _canvas(w, h)
    W, H = w * SS, h * SS
    cols = [_hex(c) for c in colors]
    for _ in range(n_strands):
        x0 = rng.uniform(0.08, 0.92) * W
        y0 = rng.uniform(0.02, 0.45) * H
        length = rng.uniform(0.35, 0.6) * H
        sway = rng.uniform(-0.15, 0.15) * W
        pts = [(x0 + sway * (t**2), y0 + length * t) for t in np.linspace(0, 1, 6)]
        d.line(pts, fill=_jit(_hex(stem), rng), width=int(2 * SS))
        for _k in range(leaves):
            t = rng.uniform(0.05, 1.0)
            px, py = x0 + sway * t**2, y0 + length * t
            ang = math.pi / 2 + rng.normal(0, 0.45)
            ln = rng.uniform(22, 38) * SS
            c = cols[rng.integers(len(cols))] * (1.05 - 0.25 * (py / H))
            d.polygon(_leaf_poly(px, py, ln, 0.13 * ln, ang, n=5), fill=_jit(c, rng, 0.1))
    return _finish(im, w, h)


def draw_fan_frond(w: int, h: int, seed: int, color: str, tip: str):  # noqa: ANN201
    """Washingtonia fan: ~60 pleated segments radiating from the hastula (bottom center)."""
    rng = np.random.default_rng(seed)
    im, d = _canvas(w, h)
    W, H = w * SS, h * SS
    hub = (W / 2, H * 0.92)
    R = H * 0.86
    n = 64
    c0, ct = _hex(color), _hex(tip)
    for i in range(n):
        a0 = -math.pi * 0.92 + math.pi * 0.84 * i / n
        a1 = a0 + math.pi * 0.84 / n * 0.92
        r = R * rng.uniform(0.86, 1.0)
        split = r * rng.uniform(0.62, 0.78)
        c = c0 * rng.uniform(0.82, 1.12) * (0.92 if i % 2 else 1.06)
        p = [hub, (hub[0] + math.cos(a0) * split, hub[1] + math.sin(a0) * split),
             (hub[0] + math.cos(a1) * split, hub[1] + math.sin(a1) * split)]
        d.polygon(p, fill=_jit(c, rng, 0.05))
        # free drooping tips (filaments) beyond the split
        am = (a0 + a1) / 2
        for off in (-0.012, 0.012):
            q0 = (hub[0] + math.cos(am + off) * split, hub[1] + math.sin(am + off) * split)
            q1 = (hub[0] + math.cos(am + off * 2) * r, hub[1] + math.sin(am + off * 2) * r + (r - split) * 0.35)
            d.line([q0, q1], fill=_jit(ct * rng.uniform(0.85, 1.1), rng, 0.05), width=int(2.2 * SS))
    d.ellipse([hub[0] - 9 * SS, hub[1] - 7 * SS, hub[0] + 9 * SS, hub[1] + 7 * SS], fill=_jit(_hex("#8A7A52"), rng))
    return _finish(im, w, h)


def draw_pinnate(w: int, h: int, seed: int, color: str, rachis: str = "#9C9A62"):  # noqa: ANN201
    """Queen palm frond: rachis along x with long soft leaflets drooping on both sides."""
    rng = np.random.default_rng(seed)
    im, d = _canvas(w, h)
    W, H = w * SS, h * SS
    c0 = _hex(color)
    y_r = lambda x: H * 0.32 + (x / W) ** 2 * H * 0.18  # noqa: E731  slight arch of the rachis
    n = 120
    for i in range(n):
        x = W * (0.02 + 0.96 * i / n)
        t = i / n
        ln = H * (0.32 + 0.28 * math.sin(math.pi * min(1.0, t * 1.15))) * rng.uniform(0.8, 1.05)
        for side in (-1, 1):
            ang = math.pi / 2 + side * rng.uniform(0.35, 0.9)  # droop down, both planes collapse to below
            ex = x + math.cos(ang) * ln * 0.45
            ey = y_r(x) + math.sin(ang) * ln
            mid = ((x + ex) / 2 + side * 6 * SS, (y_r(x) + ey) / 2 - 8 * SS)
            c = c0 * rng.uniform(0.78, 1.12) * (1.0 - 0.15 * t)
            d.line([(x, y_r(x)), mid, (ex, ey)], fill=_jit(c, rng, 0.05), width=int(rng.uniform(2.0, 3.2) * SS))
    xs = np.linspace(0, W, 24)
    d.line([(x, y_r(x)) for x in xs], fill=_jit(_hex(rachis), rng), width=int(4 * SS))
    return _finish(im, w, h)


def draw_grass(w: int, h: int, seed: int, colors: Sequence[str], n: int = 140, plume: str | None = None):  # noqa: ANN201
    rng = np.random.default_rng(seed)
    im, d = _canvas(w, h)
    W, H = w * SS, h * SS
    cols = [_hex(c) for c in colors]
    for _ in range(n):
        x0 = W / 2 + rng.normal(0, W * 0.06)
        ang = rng.normal(0, 0.45)
        ln = rng.uniform(0.55, 0.97) * H
        bend = rng.uniform(0.1, 0.45) * np.sign(ang + 1e-6)
        pts = []
        for t in np.linspace(0, 1, 7):
            a = ang + bend * t * t * 2.2
            pts.append((x0 + math.sin(a) * ln * t, H - math.cos(ang) * ln * t + ln * 0.1 * (t**2) * abs(bend)))
        c = cols[rng.integers(len(cols))]
        d.line(pts, fill=_jit(c, rng, 0.12), width=int(rng.uniform(1.6, 2.8) * SS))
        if plume and rng.random() < 0.35:
            px, py = pts[-1]
            d.ellipse([px - 5 * SS, py - 14 * SS, px + 5 * SS, py + 4 * SS], fill=_jit(_hex(plume), rng, 0.08))
    return _finish(im, w, h)


def draw_skirt(w: int, h: int, seed: int):  # noqa: ANN201
    """Dead hanging Washingtonia fronds (the 'skirt' / petticoat below the crown)."""
    rng = np.random.default_rng(seed)
    im, d = _canvas(w, h)
    W, H = w * SS, h * SS
    for _ in range(170):
        x = rng.uniform(0.02, 0.98) * W
        y0 = rng.uniform(0, 0.12) * H
        ln = rng.uniform(0.55, 0.98) * H
        c = _hex("#9A825C") * rng.uniform(0.55, 1.15)
        d.line([(x, y0), (x + rng.normal(0, 8 * SS), y0 + ln)], fill=_jit(c, rng, 0.06), width=int(rng.uniform(3, 7) * SS))
    return _finish(im, w, h)


def draw_bark(w: int, h: int, seed: int, base: str, dark: str, light: str, kind: str):  # noqa: ANN201
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(seed)
    b, dk, lt = _hex(base), _hex(dark), _hex(light)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    noise = rng.random((h // 4 + 1, w // 4 + 1)).astype(np.float32)
    n_img = Image.fromarray((noise * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(1.5))
    n = np.asarray(n_img).astype(np.float32) / 255.0
    if kind == "fissured":  # oak / street tree: vertical fissures
        fis = np.abs(np.sin(xx / w * math.pi * 6 + n * 3.0 + np.sin(yy / 23.0) * 0.6))
        t = np.clip((fis - 0.75) * 4, 0, 1)
        img = b[None, None] * (0.85 + 0.3 * n[..., None]) * (1 - t[..., None]) + dk[None, None] * t[..., None]
    elif kind == "patchy":  # eucalyptus: smooth pale bark shedding in patches
        blobs = rng.random((h // 24 + 1, w // 16 + 1)).astype(np.float32)
        bl = np.asarray(Image.fromarray((blobs * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3))).astype(np.float32) / 255
        t = np.clip((bl - 0.55) * 5, 0, 1)
        img = lt[None, None] * (0.9 + 0.15 * n[..., None]) * (1 - t[..., None]) + b[None, None] * t[..., None]
        streak = np.clip(np.sin(xx / w * math.pi * 9 + n * 2) * 0.5 + 0.5, 0, 1)
        img = img * (0.93 + 0.07 * streak[..., None])
    else:  # ringed palm trunk: horizontal leaf-scar rings
        ring = np.abs(np.sin(yy / h * math.pi * 90 + n * 2.5 + np.sin(xx / w * math.pi * 2) * 0.8))
        t = np.clip((ring - 0.9) * 4, 0, 1) * (0.35 + 0.3 * n)
        img = b[None, None] * (0.85 + 0.3 * n[..., None]) * (1 - t[..., None]) + dk[None, None] * t[..., None]
    out = np.concatenate([np.clip(img, 0, 255), np.full((h, w, 1), 255.0)], axis=-1).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


# ---------------------------------------------------------------------------
# atlas assembly + UV lookup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    name: str
    col: int
    row: int
    colspan: int = 1


def build_atlas(path: Path, cells: Sequence[tuple[Cell, Callable]], bark: Callable | None) -> Path:  # type: ignore[type-arg]
    from PIL import Image

    atlas = Image.new("RGBA", (TEX, TEX), (90, 100, 60, 0))
    for cell, fn in cells:
        w = CELL_W // RES * cell.colspan
        img = fn(w, CELL_H // RES)
        atlas.paste(img, (cell.col * CELL_W, cell.row * CELL_H))
    if bark is not None:
        atlas.paste(bark(TEX - LEAF_W, TEX), (LEAF_W, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    # 256-color palette PNG with alpha: ~3x smaller, indistinguishable for alpha-tested leaves
    atlas.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(path, optimize=True)
    return path


def cell_uv(cell: Cell) -> tuple[float, float, float, float]:
    """(u0, v0, u1, v1) in Blender/glTF-after-export convention (v up from the image bottom)."""
    u0 = cell.col * CELL_W / TEX
    u1 = (cell.col + cell.colspan) * CELL_W / TEX
    v1 = 1.0 - cell.row * CELL_H / TEX
    v0 = 1.0 - (cell.row + 1) * CELL_H / TEX
    pad = 1.5 / TEX
    return u0 + pad, v0 + pad, u1 - pad, v1 - pad


BARK_U = (LEAF_W + 2) / TEX, (TEX - 2) / TEX


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


class Plant:
    """Accumulates trunk tubes and leaf cards with UVs and (optional) custom normals."""

    def __init__(self, cells: dict[str, Cell]):
        self.part = Part()
        self.normals: list[np.ndarray | None] = []  # per vertex (None = automatic)
        self.kinds: list[int] = []  # per vertex: 0 bark / solid, 1 foliage card
        self.cells = cells

    def _add(self, V: np.ndarray, F: list[list[int]], U: list, N: np.ndarray | None, kind: int = 1) -> None:
        self.part.add(Part(np.asarray(V, dtype=float), F, ["foliage"] * len(F), U))
        if N is None:
            self.normals.extend([None] * len(V))
        else:
            self.normals.extend(list(N))
        self.kinds.extend([kind] * len(V))

    def tube(self, pts: Sequence[Sequence[float]], radii: Sequence[float], n: int = 6, v_scale: float = 0.08,
             cap: bool = False) -> None:
        """Bark tube along a polyline. v runs along the atlas bark strip (v_scale per meter, clamped)."""
        from mathutils import Vector

        P = [Vector(p) for p in pts]
        V, F, U = [], [], []
        prev = None
        L = 0.0
        vs = []
        for i, p in enumerate(P):
            if i:
                L += (P[i] - P[i - 1]).length
            vs.append(min(0.99, 0.01 + L * v_scale))
            t = (P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]).normalized()
            ref = Vector((1, 0, 0)) if abs(t.x) < 0.9 else Vector((0, 1, 0))
            x = (prev - t * prev.dot(t)).normalized() if prev is not None else t.cross(ref).normalized()
            prev = x
            y = t.cross(x)
            for k in range(n + 1):  # duplicate seam column for clean UVs
                a = 2 * math.pi * k / n
                V.append(tuple(p + (x * math.cos(a) + y * math.sin(a)) * radii[i]))
        u0, u1 = BARK_U
        for i in range(len(P) - 1):
            for k in range(n):
                a, b = i * (n + 1) + k, i * (n + 1) + k + 1
                F.append([a, b, b + n + 1, a + n + 1])
                uk0 = u0 + (u1 - u0) * k / n
                uk1 = u0 + (u1 - u0) * (k + 1) / n
                U.append([(uk0, vs[i]), (uk1, vs[i]), (uk1, vs[i + 1]), (uk0, vs[i + 1])])
        if cap:
            last = (len(P) - 1) * (n + 1)
            F.append([last + k for k in range(n)])
            U.append([(u0 + 0.01, vs[-1])] * n)
        # outward fix: tubes built with x,y frame are CCW around t -> faces point outward already
        self._add(np.array(V), F, U, None, kind=0)

    def card(self, center: Sequence[float], normal: Sequence[float], width: float, height: float, cell: str,
             roll: float = 0.0, canopy_center: Sequence[float] | None = None, bend: float = 0.7,
             anchor: str = "center", fold: float = 0.0) -> None:
        """One textured quad (optionally V-folded along its vertical axis: 2 quads)."""
        c = np.asarray(center, float)
        n = np.asarray(normal, float)
        n /= np.linalg.norm(n) + 1e-12
        up0 = np.array([0.0, 0.0, 1.0])
        if abs(n @ up0) > 0.97:
            up0 = np.array([0.0, 1.0, 0.0])
        right = np.cross(up0, n)
        right /= np.linalg.norm(right)
        up = np.cross(n, right)
        if roll:
            cr, sr = math.cos(roll), math.sin(roll)
            right, up = right * cr + up * sr, -right * sr + up * cr
        hw, hh = width / 2, height / 2
        if anchor == "bottom":
            c = c + up * hh
        u0, v0, u1, v1 = cell_uv(self.cells[cell])
        if fold > 0:
            mid_t = c - n * fold * hw
            V = np.array([c - right * hw - up * hh, mid_t - up * hh, mid_t + up * hh, c - right * hw + up * hh,
                          c + right * hw - up * hh, c + right * hw + up * hh])
            um = (u0 + u1) / 2
            F = [[0, 1, 2, 3], [1, 4, 5, 2]]
            U = [[(u0, v0), (um, v0), (um, v1), (u0, v1)], [(um, v0), (u1, v0), (u1, v1), (um, v1)]]
        else:
            V = np.array([c - right * hw - up * hh, c + right * hw - up * hh, c + right * hw + up * hh, c - right * hw + up * hh])
            F = [[0, 1, 2, 3]]
            U = [[(u0, v0), (u1, v0), (u1, v1), (u0, v1)]]
        N = None
        if canopy_center is not None:
            cc = np.asarray(canopy_center, float)
            N = []
            for v in V:
                r = v - cc
                r /= np.linalg.norm(r) + 1e-9
                m = bend * r + (1 - bend) * n
                N.append(m / (np.linalg.norm(m) + 1e-9))
            N = np.array(N)
        self._add(V, F, U, N)

    def strip(self, pts: Sequence[Sequence[float]], side: Sequence[float], width: float, cell: str,
              canopy_center: Sequence[float] | None = None, bend: float = 0.5, fold: float = 0.25, mid: float = 0.32,
              taper: bool = False) -> None:
        """A V-folded ribbon along a 3D polyline (pinnate palm fronds). Texture u along the ribbon."""
        P = [np.asarray(p, float) for p in pts]
        s = np.asarray(side, float)
        s /= np.linalg.norm(s)
        u0, v0, u1, v1 = cell_uv(self.cells[cell])
        V, F, U, N = [], [], [], []
        L = [0.0]
        for i in range(1, len(P)):
            L.append(L[-1] + float(np.linalg.norm(P[i] - P[i - 1])))
        for i, p in enumerate(P):
            t = (P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)])
            t /= np.linalg.norm(t)
            up = np.cross(t, s)
            up /= np.linalg.norm(up) + 1e-9
            w = width if taper else width * (0.55 + 0.45 * math.sin(math.pi * min(1.0, 0.15 + 0.85 * L[i] / L[-1])))
            V += [p + s * w / 2 - up * fold * w / 2, p + up * 0.02, p - s * w / 2 - up * fold * w / 2]
            uu = u0 + (u1 - u0) * L[i] / L[-1]
            U.append(uu)
        for i in range(len(P) - 1):
            a = i * 3
            F.append([a, a + 3, a + 4, a + 1])
            F.append([a + 1, a + 4, a + 5, a + 2])
        UV = []
        vm = (v0 + v1) / 2
        for i in range(len(P) - 1):
            ua, ub = U[i], U[i + 1]
            # the texture's rachis sits at 32% from the top of the cell: map it to the strip center
            vr = v1 - (v1 - v0) * mid
            UV.append([(ua, v1), (ub, v1), (ub, vr), (ua, vr)])
            UV.append([(ua, vr), (ub, vr), (ub, v0), (ua, v0)])
        del vm
        Varr = np.array(V)
        if canopy_center is not None:
            cc = np.asarray(canopy_center, float)
            for v in Varr:
                r = v - cc
                r /= np.linalg.norm(r) + 1e-9
                m = bend * r + (1 - bend) * np.array([0, 0, 1.0])
                N.append(m / np.linalg.norm(m))
        self._add(Varr, F, UV, np.array(N) if N else None)

    def canopy(self, rng: np.random.Generator, clumps: Sequence[tuple[Sequence[float], Sequence[float]]],
               n_cards: int, cells: Sequence[str], size: tuple[float, float], center: Sequence[float],
               weights: Sequence[float] | None = None, bend: float = 0.75, outward: float = 0.75,
               shell: float = 0.55, aspect: float = 1.14) -> None:
        """Scatter cards through ellipsoid clumps (center, radii), biased toward the clump shells."""
        cen = np.asarray(center, float)
        cw = np.asarray(weights, float) / np.sum(weights) if weights is not None else None
        vols = np.array([r[0] * r[1] * r[2] for _, r in clumps], float)
        vols /= vols.sum()
        for _ in range(n_cards):
            k = rng.choice(len(clumps), p=vols)
            c, r = np.asarray(clumps[k][0], float), np.asarray(clumps[k][1], float)
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            rad = rng.uniform(shell, 1.0) ** 0.5
            p = c + d * r * rad
            radial = p - cen
            radial[2] *= 0.6
            radial /= np.linalg.norm(radial) + 1e-9
            nrm = outward * radial + (1 - outward) * rng.normal(size=3)
            nrm[2] += 0.25  # cards tilt up toward the sky a bit
            sz = rng.uniform(*size)
            cell = cells[rng.choice(len(cells), p=cw)] if cw is not None else cells[rng.integers(len(cells))]
            self.card(p, nrm, sz, sz * aspect, cell, roll=rng.uniform(-0.6, 0.6), canopy_center=cen, bend=bend)

    def build(self) -> tuple[Part, np.ndarray]:
        N = np.array([n if n is not None else np.zeros(3) for n in self.normals], dtype=float)
        return self.part, N

    def ao(self, inner: float = 0.45, bottom: float = 0.3) -> np.ndarray:
        """Per-vertex ambient occlusion baked into COLOR_0: cards deep inside the crown and on its
        underside are darker (self-shadowing a flat card cloud cannot produce); bark under the crown too."""
        V = self.part.V
        k = np.array(self.kinds)
        ao = np.ones(len(V))
        if not (k == 1).any():
            return ao
        C = V[k == 1]
        lo, hi = np.percentile(C, 2, axis=0), np.percentile(C, 98, axis=0)
        c = (lo + hi) / 2
        R = np.maximum((hi - lo) / 2, 0.05)
        r = np.linalg.norm((V - c) / R, axis=1)
        zt = np.clip((V[:, 2] - lo[2]) / max(hi[2] - lo[2], 1e-3), 0, 1)
        t = np.clip((r - 0.15) / 0.8, 0, 1)
        card = (1 - inner) + inner * t * t * (3 - 2 * t)
        card *= (1 - bottom) + bottom * zt
        bark = np.where(V[:, 2] > lo[2] - 0.5, 0.62, 0.85)
        ao = np.where(k == 1, card, bark)
        return np.clip(ao, 0.25, 1.0)


# ---------------------------------------------------------------------------
# species
# ---------------------------------------------------------------------------


def _limb(pl: Plant, p0: Sequence[float], p1: Sequence[float], r0: float, r1: float, n: int = 5, segs: int = 3,
          bend: float = 0.25, v_scale: float = 0.08, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    a, b = np.asarray(p0, float), np.asarray(p1, float)
    pts, radii = [], []
    for i in range(segs + 1):
        t = i / segs
        q = a + (b - a) * t
        q[2] += math.sin(t * math.pi) * bend * np.linalg.norm(b - a) * 0.3
        if rng is not None and 0 < i < segs:
            q[:2] += rng.normal(0, 0.12, 2)
        pts.append(q)
        radii.append(r0 + (r1 - r0) * t)
    pl.tube(pts, radii, n=n, v_scale=v_scale)
    return pts


OAK_CELLS = {"a": Cell("a", 0, 0), "b": Cell("b", 1, 0), "c": Cell("c", 0, 1), "d": Cell("d", 1, 1)}


def coast_live_oak(tex: Path) -> tuple[Part, np.ndarray, Path]:
    """Quercus agrifolia: ~9 m tall, 12 m wide; short stout trunk, sinuous low limbs, dense dark dome."""
    cols_dark = ["#2F4A1E", "#3A5524", "#43602A", "#2A4219"]
    cols_light = ["#4F6B2C", "#5B7833", "#46632A"]
    path = build_atlas(tex, [
        (OAK_CELLS["a"], lambda w, h: draw_broadleaf(w, h, 11, cols_dark, (9, 14), 0.62, 900, twig="#4A3B2E")),
        (OAK_CELLS["b"], lambda w, h: draw_broadleaf(w, h, 12, cols_dark + cols_light[:1], (9, 14), 0.62, 850, twig="#4A3B2E")),
        (OAK_CELLS["c"], lambda w, h: draw_broadleaf(w, h, 13, cols_light, (9, 14), 0.62, 800, twig="#4A3B2E", sun_bias=0.3)),
        (OAK_CELLS["d"], lambda w, h: draw_broadleaf(w, h, 14, cols_dark, (8, 12), 0.6, 650, clumps=5, spread=0.5)),
    ], lambda w, h: draw_bark(w, h, 15, "#5C5045", "#2E2620", "#7A6E62", "fissured"))
    rng = np.random.default_rng(101)
    pl = Plant(OAK_CELLS)
    pl.tube([(0, 0, -0.3), (0.1, 0.05, 1.0), (0.25, 0.1, 2.0)], [0.48, 0.42, 0.36], n=7, v_scale=0.1)
    limbs = [((0.25, 0.1, 1.9), (3.6, 1.4, 4.6)), ((0.25, 0.1, 1.9), (-3.0, 2.2, 4.9)), ((0.25, 0.1, 1.9), (-1.0, -3.4, 4.4)),
             ((0.25, 0.1, 1.9), (1.2, -0.3, 6.2)), ((0.25, 0.1, 2.0), (2.6, -2.6, 4.8))]
    for a, b in limbs:
        pts = _limb(pl, a, b, 0.26, 0.09, n=5, segs=3, bend=0.6, rng=rng)
        tip = pts[-1]
        for _ in range(2):  # secondary branches into the crown
            d = rng.normal(size=3) * np.array([1.6, 1.6, 0.6]) + np.array([0, 0, 1.2])
            _limb(pl, tip, tip + d, 0.08, 0.03, n=4, segs=1, bend=0.0)
    cen = (0.0, 0.0, 4.6)
    clumps = [((0.0, 0.0, 6.4), (3.6, 3.6, 2.4)), ((3.6, 1.2, 5.4), (2.8, 2.6, 2.0)), ((-3.2, 2.0, 5.6), (2.9, 2.7, 2.0)),
              ((-1.0, -3.5, 5.2), (2.8, 2.6, 1.9)), ((2.6, -2.6, 5.0), (2.4, 2.4, 1.8)), ((-1.8, 3.8, 4.9), (2.3, 2.2, 1.7)),
              ((1.6, 3.4, 6.1), (2.4, 2.3, 1.8)), ((-3.8, -1.4, 4.8), (2.2, 2.3, 1.7))]
    pl.canopy(rng, clumps, 1150, ["a", "b", "c", "d"], (1.4, 2.3), cen, weights=[0.32, 0.28, 0.25, 0.15])
    p, N = pl.build()
    return p, N, path, pl.ao()


FAN_CELLS = {"frond": Cell("frond", 0, 0), "old": Cell("old", 1, 0), "skirt": Cell("skirt", 0, 1), "skirt2": Cell("skirt2", 1, 1)}


def mexican_fan_palm(tex: Path) -> tuple[Part, np.ndarray, Path]:
    """Washingtonia robusta: ~17 m, slender gray trunk, compact crown (~4 m) of fans, trimmed skirt."""
    path = build_atlas(tex, [
        (FAN_CELLS["frond"], lambda w, h: draw_fan_frond(w, h, 21, "#5F7D2E", "#6E8434")),
        (FAN_CELLS["old"], lambda w, h: draw_fan_frond(w, h, 22, "#7D8B3A", "#9A9248")),
        (FAN_CELLS["skirt"], lambda w, h: draw_skirt(w, h, 23)),
        (FAN_CELLS["skirt2"], lambda w, h: draw_skirt(w, h, 24)),
    ], lambda w, h: draw_bark(w, h, 25, "#857868", "#5A4F44", "#9C9080", "ringed"))
    rng = np.random.default_rng(202)
    pl = Plant(FAN_CELLS)
    H = 16.5
    lean = np.array([0.55, 0.2])
    pts, radii = [], []
    for i in range(9):
        t = i / 8
        off = lean * t**1.7
        pts.append((off[0], off[1], -0.3 + (H + 0.3) * t))
        radii.append(0.45 if i == 0 else 0.33 - 0.1 * t)
    pl.tube(pts, radii, n=6, v_scale=1.0 / (H + 1))
    top = np.array(pts[-1])
    # a few dead fronds hanging flat against the trunk under the crown (light "skirt")
    for k in range(7):
        az = 2 * math.pi * k / 7 + rng.normal(0, 0.15)
        dvec = np.array([math.cos(az), math.sin(az), 0.0])
        drop = rng.uniform(0.9, 1.5)
        pl.card(top + dvec * 0.36 + np.array([0, 0, -drop]), dvec + np.array([0, 0, 0.25]), rng.uniform(0.9, 1.2),
                rng.uniform(1.5, 2.0), "skirt" if k % 2 else "skirt2", roll=rng.normal(0, 0.12))
    # crown: ~28 fan fronds on long petioles; blades V-folded and rolled about the petiole
    cen = top + np.array([0, 0, 0.5])
    n_f = 34
    for k in range(n_f):
        az = k * 2.39996 + rng.normal(0, 0.08)  # golden-angle phyllotaxis
        elev = math.radians(float(np.interp(k / n_f, [0, 0.35, 1], [72, 30, -55])) + rng.normal(0, 6))
        dvec = np.array([math.cos(az) * math.cos(elev), math.sin(az) * math.cos(elev), math.sin(elev)])
        pet = rng.uniform(1.0, 1.5)
        base = top + np.array([0, 0, 0.25])
        hub = base + dvec * pet
        pl.tube([base, hub], [0.05, 0.03], n=3, v_scale=0.02)
        old = elev < math.radians(-25)
        size = rng.uniform(1.5, 1.9)
        side = np.array([-math.sin(az), math.cos(az), 0.0])
        nrm = np.cross(side, dvec)
        if nrm[2] < 0:
            nrm = -nrm
        twist = rng.normal(0, 0.6)  # roll the blade about the petiole
        nrm = nrm * math.cos(twist) + np.cross(dvec, nrm) * math.sin(twist)
        upv = dvec - nrm * (dvec @ nrm)
        rightv = np.cross(upv, nrm)
        # card() builds its frame from world up; pass roll so the card's up axis follows the petiole
        up0 = np.array([0.0, 0.0, 1.0]) if abs(nrm[2]) < 0.97 else np.array([0.0, 1.0, 0.0])
        r0 = np.cross(up0, nrm)
        r0 /= np.linalg.norm(r0)
        u0 = np.cross(nrm, r0)
        roll = math.atan2(upv @ (-r0), upv @ u0)
        del rightv
        pl.card(hub + dvec * size * 0.42, nrm, size, size * 1.0, "old" if old else "frond", roll=roll,
                canopy_center=cen, bend=0.4, fold=0.3)
    p, N = pl.build()
    return p, N, path, pl.ao()


QUEEN_CELLS = {"frond": Cell("frond", 0, 0, 2), "frond2": Cell("frond2", 0, 1, 2)}


def queen_palm(tex: Path) -> tuple[Part, np.ndarray, Path]:
    """Syagrus romanzoffiana: ~11 m, smooth gray ringed trunk, arching feathery fronds 3-4 m long."""
    path = build_atlas(tex, [
        (QUEEN_CELLS["frond"], lambda w, h: draw_pinnate(w, h, 31, "#6B8E36")),
        (QUEEN_CELLS["frond2"], lambda w, h: draw_pinnate(w, h, 32, "#7C9C40")),
    ], lambda w, h: draw_bark(w, h, 33, "#9A968C", "#6E6A62", "#B5B0A6", "ringed"))
    rng = np.random.default_rng(303)
    pl = Plant(QUEEN_CELLS)
    H = 10.5
    pts = [(0.0, 0.0, -0.3 + (H + 0.3) * i / 6) for i in range(7)]
    pts = [(0.25 * (i / 6) ** 2, -0.1 * (i / 6) ** 2, z) for i, (_, _, z) in enumerate(pts)]
    pl.tube(pts, [0.24, 0.22, 0.2, 0.19, 0.19, 0.2, 0.22], n=6, v_scale=1.0 / (H + 1))
    top = np.array(pts[-1])
    pl.tube([top, top + np.array([0, 0, 0.9])], [0.24, 0.14], n=6, v_scale=0.05)  # green crownshaft
    cen = top + np.array([0, 0, 0.8])
    n_f = 30
    for k in range(n_f):
        az = k * 2.39996 + rng.normal(0, 0.1)
        elev = math.radians(float(np.interp(k / n_f, [0, 0.3, 1], [80, 50, 5])) + rng.normal(0, 7))
        L = rng.uniform(3.9, 4.8) * (0.8 if k < 4 else 1.0)  # youngest spear fronds shorter
        d0 = np.array([math.cos(az), math.sin(az), 0.0])
        frond = []
        base = top + np.array([0, 0, 0.8])
        droop = math.radians(rng.uniform(70, 105))
        for i in range(8):
            t = i / 7
            ang = elev - t * droop  # arches over and droops
            frond.append(base + d0 * L * 0.85 * t * math.cos(ang * 0.5) + np.array([0, 0, L * 0.5 * math.sin(elev) * t - L * 0.55 * t * t]))
        side = np.array([-math.sin(az), math.cos(az), 0.0])
        pl.strip(frond, side, rng.uniform(2.1, 2.6), "frond" if k % 2 else "frond2", canopy_center=cen, bend=0.5, fold=0.3)
    p, N = pl.build()
    return p, N, path, pl.ao()


EUC_CELLS = {"a": Cell("a", 0, 0), "b": Cell("b", 1, 0), "c": Cell("c", 0, 1), "d": Cell("d", 1, 1)}


def eucalyptus(tex: Path) -> tuple[Part, np.ndarray, Path]:
    """Eucalyptus (river red gum / blue gum windbreak type): ~20 m, pale leaning trunk, open weeping crown."""
    cols = ["#6E8462", "#7D9270", "#5F7656", "#8A9C7A"]
    path = build_atlas(tex, [
        (EUC_CELLS["a"], lambda w, h: draw_hanging(w, h, 41, cols)),
        (EUC_CELLS["b"], lambda w, h: draw_hanging(w, h, 42, cols[1:])),
        (EUC_CELLS["c"], lambda w, h: draw_hanging(w, h, 43, cols[:3], n_strands=30)),
        (EUC_CELLS["d"], lambda w, h: draw_broadleaf(w, h, 44, cols, (20, 30), 0.18, 420, twig="#8A7460", droop=0.5)),
    ], lambda w, h: draw_bark(w, h, 45, "#B8A890", "#7D7060", "#E2DACB", "patchy"))
    rng = np.random.default_rng(404)
    pl = Plant(EUC_CELLS)
    trunk = [(0, 0, -0.3), (0.3, 0.1, 4.0), (0.9, 0.4, 8.5), (1.3, 0.5, 12.0)]
    pl.tube(trunk, [0.55, 0.45, 0.36, 0.28], n=7, v_scale=0.05)
    forks = [((1.3, 0.5, 11.5), (3.6, 1.9, 17.0)), ((1.3, 0.5, 11.5), (-1.4, -0.6, 18.0)),
             ((0.9, 0.4, 8.6), (-2.9, 1.6, 12.8)), ((1.3, 0.5, 11.8), (1.9, -1.7, 19.5)), ((0.6, 0.3, 6.5), (3.2, -1.0, 9.8))]
    for a, b in forks:
        pts = _limb(pl, a, b, 0.2, 0.07, n=4, segs=2, bend=0.3, v_scale=0.05)
        _limb(pl, pts[-1], np.asarray(pts[-1]) + rng.normal(size=3) * np.array([1.4, 1.4, 0.5]) + np.array([0, 0, 1.2]), 0.06, 0.02, n=3, segs=1, bend=0)
    cen = (0.8, 0.4, 15.0)
    clumps = [((3.6, 1.9, 17.4), (2.6, 2.4, 2.4)), ((-1.5, -0.6, 18.6), (2.8, 2.6, 2.4)), ((-3.0, 1.7, 13.4), (2.3, 2.2, 2.1)),
              ((1.9, -1.8, 19.8), (2.3, 2.2, 2.0)), ((0.6, 0.8, 15.6), (2.4, 2.4, 2.2)), ((3.3, -1.0, 10.6), (2.0, 1.9, 1.8)),
              ((-0.4, 2.4, 16.8), (2.0, 2.0, 2.0))]
    pl.canopy(rng, clumps, 1000, ["a", "b", "c", "d"], (1.6, 2.5), cen, bend=0.7, shell=0.35, aspect=1.2)
    p, N = pl.build()
    return p, N, path, pl.ao()


JAC_CELLS = {"a": Cell("a", 0, 0), "b": Cell("b", 1, 0), "flower": Cell("flower", 0, 1), "c": Cell("c", 1, 1)}


def jacaranda(tex: Path) -> tuple[Part, np.ndarray, Path]:
    """Jacaranda mimosifolia: ~8.5 m, 9 m wide umbrella crown of fine fern-like leaves; sparse
    autumn re-bloom (main purple bloom is May-June)."""
    cols = ["#5E7F33", "#6B8C3A", "#557530", "#78964A"]
    path = build_atlas(tex, [
        (JAC_CELLS["a"], lambda w, h: draw_broadleaf(w, h, 51, cols, (7, 10), 0.35, 1100, twig="#5E5246", spread=0.5)),
        (JAC_CELLS["b"], lambda w, h: draw_broadleaf(w, h, 52, cols, (7, 10), 0.35, 1000, twig="#5E5246", spread=0.5)),
        (JAC_CELLS["flower"], lambda w, h: draw_broadleaf(w, h, 53, cols, (7, 10), 0.35, 900, flowers="#8E6BC8", flower_frac=0.45)),
        (JAC_CELLS["c"], lambda w, h: draw_broadleaf(w, h, 54, cols[1:], (7, 10), 0.35, 800, spread=0.55)),
    ], lambda w, h: draw_bark(w, h, 55, "#7A6F64", "#4E463E", "#958A7E", "fissured"))
    rng = np.random.default_rng(505)
    pl = Plant(JAC_CELLS)
    pl.tube([(0, 0, -0.3), (0.05, 0.0, 1.2), (0.1, 0, 2.3)], [0.3, 0.27, 0.24], n=6, v_scale=0.1)
    for a, b in [((0.1, 0, 2.2), (2.4, 1.1, 5.4)), ((0.1, 0, 2.2), (-2.2, 1.5, 5.3)), ((0.1, 0, 2.2), (-0.4, -2.5, 5.4)),
                 ((0.1, 0, 2.2), (1.6, -1.8, 5.8))]:
        pts = _limb(pl, a, b, 0.18, 0.07, n=5, segs=2, bend=0.3, rng=rng)
        _limb(pl, pts[-1], np.asarray(pts[-1]) + np.array([0, 0, 1.0]) + rng.normal(size=3) * 0.8, 0.06, 0.02, n=3, segs=1, bend=0)
    cen = (0.0, 0.0, 5.0)
    clumps = [((0, 0, 6.8), (3.2, 3.2, 1.5)), ((2.7, 1.2, 6.1), (2.4, 2.3, 1.4)), ((-2.5, 1.7, 6.0), (2.4, 2.3, 1.4)),
              ((-0.4, -2.8, 6.1), (2.4, 2.3, 1.4)), ((2.1, -1.9, 6.0), (2.0, 2.0, 1.3)), ((-2.3, -1.3, 5.8), (2.0, 2.0, 1.3))]
    pl.canopy(rng, clumps, 950, ["a", "b", "flower", "c"], (1.25, 2.0), cen, weights=[0.33, 0.3, 0.12, 0.25])
    p, N = pl.build()
    return p, N, path, pl.ao()


ST_CELLS = {"a": Cell("a", 0, 0), "b": Cell("b", 1, 0), "c": Cell("c", 0, 1), "d": Cell("d", 1, 1)}


def street_tree(tex: Path) -> tuple[Part, np.ndarray, Path]:
    """Parkway street tree (Brisbane box / Chinese elm / Chinese pistache class): ~8 m, rounded crown."""
    cols = ["#4F7A2E", "#5C8A36", "#476E29", "#6A9640"]
    path = build_atlas(tex, [
        (ST_CELLS["a"], lambda w, h: draw_broadleaf(w, h, 61, cols, (11, 17), 0.5, 760, twig="#5A4E43")),
        (ST_CELLS["b"], lambda w, h: draw_broadleaf(w, h, 62, cols, (11, 17), 0.5, 720, twig="#5A4E43")),
        (ST_CELLS["c"], lambda w, h: draw_broadleaf(w, h, 63, cols[1:], (11, 17), 0.5, 700, sun_bias=0.35)),
        (ST_CELLS["d"], lambda w, h: draw_broadleaf(w, h, 64, cols[:3], (10, 15), 0.5, 600, clumps=5)),
    ], lambda w, h: draw_bark(w, h, 65, "#7B7065", "#4D453D", "#978C80", "fissured"))
    rng = np.random.default_rng(606)
    pl = Plant(ST_CELLS)
    pl.tube([(0, 0, -0.3), (0, 0.03, 1.3), (0.02, 0.05, 2.5)], [0.2, 0.17, 0.15], n=6, v_scale=0.12)
    for a, b in [((0.02, 0.05, 2.4), (1.4, 0.6, 4.4)), ((0.02, 0.05, 2.4), (-1.2, 0.8, 4.6)), ((0.02, 0.05, 2.4), (-0.2, -1.4, 4.5)),
                 ((0.02, 0.05, 2.4), (0.3, 0.2, 5.6))]:
        _limb(pl, a, b, 0.11, 0.05, n=4, segs=2, bend=0.2, rng=rng)
    cen = (0.0, 0.0, 5.0)
    clumps = [((0, 0, 5.6), (2.7, 2.7, 2.4)), ((1.2, 0.7, 4.8), (1.9, 1.8, 1.7)), ((-1.2, 0.6, 5.0), (1.9, 1.8, 1.7)),
              ((0.1, -1.3, 4.8), (1.8, 1.8, 1.6)), ((0.2, 0.3, 6.8), (1.7, 1.7, 1.4))]
    pl.canopy(rng, clumps, 950, ["a", "b", "c", "d"], (1.15, 1.8), cen, shell=0.6)
    p, N = pl.build()
    return p, N, path, pl.ao()


SHRUB_CELLS = {"a": Cell("a", 0, 0), "b": Cell("b", 1, 0), "flower": Cell("flower", 0, 1), "c": Cell("c", 1, 1)}


def shrub(tex: Path) -> tuple[Part, np.ndarray, Path]:
    """Irrigated landscape shrub mound (~1.3 m tall, 1.8 m wide; pittosporum / bougainvillea accents)."""
    cols = ["#43642A", "#4F7230", "#3B5A25", "#5C7E38"]
    path = build_atlas(tex, [
        (SHRUB_CELLS["a"], lambda w, h: draw_broadleaf(w, h, 71, cols, (8, 12), 0.55, 1300, spread=0.6, clumps=9)),
        (SHRUB_CELLS["b"], lambda w, h: draw_broadleaf(w, h, 72, cols, (8, 12), 0.55, 1200, spread=0.6, clumps=9)),
        (SHRUB_CELLS["flower"], lambda w, h: draw_broadleaf(w, h, 73, cols, (8, 12), 0.6, 1200, spread=0.6, clumps=9,
                                                            flowers="#C8417F", flower_frac=0.35)),
        (SHRUB_CELLS["c"], lambda w, h: draw_broadleaf(w, h, 74, cols[1:], (8, 12), 0.55, 1100, spread=0.6, clumps=9)),
    ], None)
    rng = np.random.default_rng(707)
    pl = Plant(SHRUB_CELLS)
    cen = (0.0, 0.0, 0.3)
    clumps = [((0, 0, 0.65), (0.9, 0.9, 0.6)), ((0.45, 0.25, 0.5), (0.6, 0.6, 0.45)), ((-0.4, -0.3, 0.5), (0.6, 0.55, 0.45))]
    pl.canopy(rng, clumps, 170, ["a", "b", "flower", "c"], (0.6, 0.9), cen, weights=[0.35, 0.3, 0.1, 0.25], shell=0.4)
    p, N = pl.build()
    p.V[:, 2] = np.maximum(p.V[:, 2], 0.0)
    return p, N, path, pl.ao()


GRASS_CELLS = {"a": Cell("a", 0, 0), "b": Cell("b", 1, 0), "c": Cell("c", 0, 1), "d": Cell("d", 1, 1)}


def ornamental_grass(tex: Path) -> tuple[Part, np.ndarray, Path]:
    """Ornamental bunch grass (Mexican feather grass / deer grass / fountain grass): ~0.8 m clump."""
    path = build_atlas(tex, [
        (GRASS_CELLS["a"], lambda w, h: draw_grass(w, h, 81, ["#8C9A55", "#A3A865", "#7B8A4A"], plume="#D8CDA0")),
        (GRASS_CELLS["b"], lambda w, h: draw_grass(w, h, 82, ["#9AA35E", "#B3B074", "#86934F"])),
        (GRASS_CELLS["c"], lambda w, h: draw_grass(w, h, 83, ["#6E8A45", "#7F9A50", "#5F7A3C"], plume="#C9B98E")),
        (GRASS_CELLS["d"], lambda w, h: draw_grass(w, h, 84, ["#A8A06A", "#BDB37E", "#958F5B"])),
    ], None)
    rng = np.random.default_rng(808)
    pl = Plant(GRASS_CELLS)
    for k in range(7):
        az = math.pi * k / 7 + rng.normal(0, 0.1)
        nrm = np.array([math.cos(az), math.sin(az), 0.15])
        cell = "abcd"[k % 4]
        pl.card((rng.normal(0, 0.05), rng.normal(0, 0.05), 0.0), nrm, 1.0, 0.85, cell, roll=rng.normal(0, 0.08),
                canopy_center=(0, 0, -0.3), bend=0.6, anchor="bottom")
    p, N = pl.build()
    return p, N, path, pl.ao()



# ---------------------------------------------------------------------------
# additional species (2026-10 upgrade): pine, ficus, bougainvillea, agave, hedge, lawn tuft
# ---------------------------------------------------------------------------


def draw_pine_tuft(w: int, h: int, seed: int, colors: Sequence[str], n_tufts: int = 5):  # noqa: ANN201
    """Canary Island pine: long (25-30 cm) needles in drooping bundles radiating from twig tips."""
    rng = np.random.default_rng(seed)
    im, d = _canvas(w, h)
    W, H = w * SS, h * SS
    cols = [_hex(c) for c in colors]
    stem = _hex("#6E5A44")
    for _t in range(n_tufts):
        cx, cy = rng.uniform(0.25, 0.75) * W, rng.uniform(0.3, 0.7) * H
        ang0 = rng.uniform(-0.6, 0.6) - math.pi / 2
        L = rng.uniform(0.18, 0.3) * H
        tip = (cx + math.cos(ang0) * L * 0.4, cy + math.sin(ang0) * L * 0.4)
        d.line([(cx, cy + L * 0.3), tip], fill=_jit(stem, rng, 0.1), width=int(2.4 * SS))
        for _k in range(rng.integers(140, 200)):
            a = rng.uniform(0, 2 * math.pi)
            ln = rng.uniform(0.45, 1.0) * L
            droop = 0.35 * ln
            p0 = (tip[0] + rng.normal(0, 2 * SS), tip[1] + rng.normal(0, 2 * SS))
            p1 = (p0[0] + math.cos(a) * ln * 0.55, p0[1] + math.sin(a) * ln * 0.55 + droop * 0.2)
            p2 = (p0[0] + math.cos(a) * ln, p0[1] + math.sin(a) * ln * 0.85 + droop)
            c = cols[rng.integers(len(cols))] * (0.8 + 0.35 * (1 - (p2[1] / H)))
            d.line([p0, p1, p2], fill=_jit(c, rng, 0.1), width=max(1, int(0.9 * SS)))
    return _finish(im, w, h)


def draw_bracts(w: int, h: int, seed: int, leaf_cols: Sequence[str], bract: str, frac: float):  # noqa: ANN201
    """Bougainvillea: arching canes with heart-shaped leaves and papery magenta bract clusters."""
    rng = np.random.default_rng(seed)
    im, d = _canvas(w, h)
    W, H = w * SS, h * SS
    lc = [_hex(c) for c in leaf_cols]
    bc = _hex(bract)
    for _c in range(9):
        x0 = rng.uniform(0.1, 0.9) * W
        pts = [(x0 + math.sin(t * 2.2 + rng.uniform(0, 3)) * W * 0.2 * t, H * (0.98 - 0.9 * t)) for t in np.linspace(0, 1, 10)]
        d.line(pts, fill=_jit(_hex("#5C4A35"), rng, 0.1), width=int(1.6 * SS))
        for _k in range(70):
            px, py = pts[rng.integers(1, len(pts))]
            px += rng.normal(0, 12 * SS)
            py += rng.normal(0, 12 * SS)
            if rng.random() < frac:  # bract triplet
                for _b in range(3):
                    a = rng.uniform(0, 2 * math.pi)
                    r = rng.uniform(3.5, 5.5) * SS
                    q = (px + math.cos(a) * r * 0.6, py + math.sin(a) * r * 0.6)
                    d.polygon(_leaf_poly(q[0], q[1], r * 1.6, r * 0.9, a, n=6), fill=_jit(bc * rng.uniform(0.75, 1.1), rng, 0.08))
            else:
                ln = rng.uniform(6, 10) * SS
                _leaf(d, px, py, ln, ln * 0.6, rng.uniform(0, 2 * math.pi), lc[rng.integers(len(lc))], rng)
    return _finish(im, w, h)


def draw_agave_leaf(w: int, h: int, seed: int, color: str, edge: str):  # noqa: ANN201
    """Agave americana leaf laid along x (base at x=0, tip at x=w, midline at h/2): thick tapered blade,
    marginal teeth, terminal spine, pale bud imprints. Mapped by Plant.strip(mid=0.5)."""
    rng = np.random.default_rng(seed)
    im, d = _canvas(w, h)
    W, H = w * SS, h * SS
    c0, ce = _hex(color), _hex(edge)
    top, bot = [], []
    n = 48
    for i in range(n + 1):
        t = i / n
        x = W * (0.01 + 0.95 * t)
        hw = H * 0.46 * (1 - t) ** 0.85 * (0.7 + 0.3 * math.sin(math.pi * min(1.0, t * 1.7)))
        tooth = (2.2 * SS) * (i % 3 == 0) * (1 - t) ** 0.5
        top.append((x, H / 2 - hw - tooth))
        bot.append((x, H / 2 + hw + tooth))
    poly = top + bot[::-1]
    d.polygon(poly, fill=_jit(ce, rng, 0.03))
    inner = [(x, H / 2 + (y - H / 2) * 0.84) for x, y in poly]
    d.polygon(inner, fill=_jit(c0, rng, 0.03))
    for k in range(4):  # bud imprints echo the toothed margin
        f = 0.5 + 0.11 * k
        band = [(x, H / 2 + (y - H / 2) * f) for x, y in top]
        d.line(band, fill=_jit(c0 * 1.12 + 8, rng, 0.02), width=max(1, int(1.0 * SS)))
    d.line([(W * 0.95, H / 2), (W * 0.999, H / 2)], fill=(60, 45, 32, 255), width=int(2.5 * SS))
    return _finish(im, w, h)


def draw_dense(w: int, h: int, seed: int, colors: Sequence[str], leaf_len: tuple[float, float], n: int = 2600):  # noqa: ANN201
    """Opaque clipped-hedge surface: a dark leafy background fully covered with small leaves."""
    rng = np.random.default_rng(seed)
    im, d = _canvas(w, h)
    W, H = w * SS, h * SS
    cols = [_hex(c) for c in colors]
    d.rectangle([0, 0, W, H], fill=tuple(int(v) for v in cols[0] * 0.55) + (255,))
    for i in range(n):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        ln = rng.uniform(*leaf_len) * SS
        _leaf(d, x, y, ln, ln * 0.55, rng.uniform(0, 2 * math.pi), cols[rng.integers(len(cols))] * (0.7 + 0.3 * i / n), rng)
    return _finish(im, w, h)


def draw_bark_plated(w: int, h: int, seed: int):  # noqa: ANN201
    """Canary Island pine: thick reddish-brown bark broken into irregular plates with dark fissures."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = rng.random((h // 6 + 2, w // 6 + 2)).astype(np.float32)
    nimg = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)).astype(np.float32) / 255
    plates = np.abs(np.sin(xx / w * math.pi * 7 + nimg * 4)) * np.abs(np.sin(yy / 37.0 + nimg * 3 + (xx // 18) * 1.7))
    t = np.clip((0.18 - plates) * 6, 0, 1)
    base = _hex("#7A5440") * (0.8 + 0.35 * nimg[..., None])
    img = base * (1 - t[..., None]) + _hex("#2E211A") * t[..., None]
    out = np.concatenate([np.clip(img, 0, 255), np.full((h, w, 1), 255.0)], axis=-1).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


PINE_CELLS = {"a": Cell("a", 0, 0), "b": Cell("b", 1, 0), "c": Cell("c", 0, 1), "d": Cell("d", 1, 1)}


def canary_pine(tex: Path) -> tuple[Part, np.ndarray, Path, np.ndarray]:
    """Pinus canariensis: ~20 m, straight trunk, short tiered branches with long drooping needle tufts,
    columnar-irregular crown ~6 m wide, epicormic tufts on the trunk."""
    cols = ["#4A6A3E", "#587A4A", "#64845A", "#405E36"]
    path = build_atlas(tex, [
        (PINE_CELLS["a"], lambda w, h: draw_pine_tuft(w, h, 91, cols, 7)),
        (PINE_CELLS["b"], lambda w, h: draw_pine_tuft(w, h, 92, cols, 6)),
        (PINE_CELLS["c"], lambda w, h: draw_pine_tuft(w, h, 93, cols[1:], 8)),
        (PINE_CELLS["d"], lambda w, h: draw_pine_tuft(w, h, 94, cols[:3], 5)),
    ], lambda w, h: draw_bark_plated(w, h, 95))
    rng = np.random.default_rng(909)
    pl = Plant(PINE_CELLS)
    H = 19.5
    trunk = [(0.25 * (i / 8) ** 2, 0.1 * (i / 8) ** 2, -0.3 + (H + 0.3) * i / 8) for i in range(9)]
    pl.tube(trunk, [0.42, 0.36, 0.32, 0.28, 0.24, 0.2, 0.16, 0.12, 0.08], n=7, v_scale=0.05)
    tip_cards: list[tuple[np.ndarray, np.ndarray]] = []
    z = 4.5
    k = 0
    while z < H - 0.6:
        t = (z - 4.5) / (H - 4.5)
        nb = int(rng.integers(4, 6))
        L = 2.9 * (1 - t) ** 0.6 + 0.7
        az0 = rng.uniform(0, 2 * math.pi)
        tz = np.interp(z, [p[2] for p in trunk], [p[0] for p in trunk]), np.interp(z, [p[2] for p in trunk], [p[1] for p in trunk])
        for b in range(nb):
            az = az0 + 2 * math.pi * b / nb + rng.normal(0, 0.3)
            d0 = np.array([math.cos(az), math.sin(az), 0.0])
            base = np.array([tz[0], tz[1], z])
            Lb = L * rng.uniform(0.7, 1.15)
            end = base + d0 * Lb + np.array([0, 0, rng.uniform(0.2, 0.9)])
            pts = _limb(pl, base, end, 0.09 * (1 - t) + 0.04, 0.025, n=4, segs=2, bend=0.25, v_scale=0.05)
            for s_ in np.linspace(0.3, 1.0, max(2, int(Lb / 0.42))):
                q = base + (end - base) * s_
                tip_cards.append((q, d0))
            k += 1
        z += rng.uniform(0.8, 1.2)
    cen = (0.2, 0.1, H * 0.62)
    for q, d0 in tip_cards:
        for _ in range(3):
            nrm = d0 * 0.7 + rng.normal(size=3) * 0.6 + np.array([0, 0, 0.3])
            sz = rng.uniform(1.1, 1.6)
            pl.card(q + rng.normal(size=3) * 0.25, nrm, sz, sz * 1.14, "abcd"[rng.integers(4)], roll=rng.uniform(-0.8, 0.8),
                    canopy_center=cen, bend=0.6)
    for _ in range(26):  # top tuft cluster and epicormic tufts on the trunk
        zz = rng.uniform(H - 1.2, H + 0.6) if rng.random() < 0.6 else rng.uniform(2.5, H - 2)
        tx = np.interp(zz, [p[2] for p in trunk], [p[0] for p in trunk])
        ty = np.interp(zz, [p[2] for p in trunk], [p[1] for p in trunk])
        sz = rng.uniform(0.9, 1.4) if zz > H - 2 else rng.uniform(0.6, 0.9)
        pl.card((tx + rng.normal(0, 0.3), ty + rng.normal(0, 0.3), zz), rng.normal(size=3) + np.array([0, 0, 0.4]), sz, sz * 1.14,
                "abcd"[rng.integers(4)], roll=rng.uniform(-1, 1), canopy_center=cen, bend=0.5)
    p, N = pl.build()
    return p, N, path, pl.ao(inner=0.35)


FICUS_CELLS = {"a": Cell("a", 0, 0), "b": Cell("b", 1, 0), "c": Cell("c", 0, 1), "d": Cell("d", 1, 1)}


def ficus(tex: Path) -> tuple[Part, np.ndarray, Path, np.ndarray]:
    """Indian laurel fig (Ficus microcarpa 'Nitida'): commercial / parking-lot tree, ~8 m, dense clipped
    dome of small glossy leaves, smooth pale gray trunk branching low."""
    cols = ["#2F4F22", "#3A5E28", "#467030", "#2A4720"]
    path = build_atlas(tex, [
        (FICUS_CELLS["a"], lambda w, h: draw_broadleaf(w, h, 101, cols, (6, 9), 0.5, 1700, twig="#3E3A30", clumps=8, spread=0.55)),
        (FICUS_CELLS["b"], lambda w, h: draw_broadleaf(w, h, 102, cols, (6, 9), 0.5, 1600, twig="#3E3A30", clumps=8, spread=0.55)),
        (FICUS_CELLS["c"], lambda w, h: draw_broadleaf(w, h, 103, cols[1:], (6, 9), 0.5, 1500, twig="#3E3A30", spread=0.6, sun_bias=0.35)),
        (FICUS_CELLS["d"], lambda w, h: draw_broadleaf(w, h, 104, cols[:3], (5, 8), 0.5, 1400, clumps=9, spread=0.6)),
    ], lambda w, h: draw_bark(w, h, 105, "#9A968C", "#6F6B63", "#B4B0A6", "patchy"))
    rng = np.random.default_rng(1010)
    pl = Plant(FICUS_CELLS)
    pl.tube([(0, 0, -0.3), (0.02, 0.0, 0.9), (0.05, 0.02, 1.8)], [0.36, 0.3, 0.26], n=7, v_scale=0.1)
    for a, b in [((0.05, 0.02, 1.7), (1.8, 0.7, 3.6)), ((0.05, 0.02, 1.7), (-1.6, 1.0, 3.8)), ((0.05, 0.02, 1.7), (-0.3, -1.9, 3.7)),
                 ((0.05, 0.02, 1.7), (1.2, -1.4, 4.1)), ((0.05, 0.02, 1.8), (0.1, 0.3, 4.6))]:
        pts = _limb(pl, a, b, 0.15, 0.06, n=5, segs=2, bend=0.25, rng=rng)
        _limb(pl, pts[-1], np.asarray(pts[-1]) + np.array([0, 0, 1.0]) + rng.normal(size=3) * 0.7, 0.05, 0.02, n=3, segs=1, bend=0)
    cen = (0.0, 0.0, 4.8)
    clumps = [((0, 0, 5.3), (3.6, 3.6, 2.6)), ((1.6, 0.8, 4.9), (2.4, 2.3, 2.0)), ((-1.5, 0.9, 5.0), (2.4, 2.3, 2.0)),
              ((0.0, -1.7, 4.9), (2.3, 2.3, 1.9)), ((0.3, 0.3, 6.6), (2.4, 2.4, 1.6))]
    pl.canopy(rng, clumps, 1350, ["a", "b", "c", "d"], (1.0, 1.6), cen, shell=0.8, outward=0.85, bend=0.8)
    p, N = pl.build()
    return p, N, path, pl.ao(inner=0.55)


BOUG_CELLS = {"a": Cell("a", 0, 0), "b": Cell("b", 1, 0), "c": Cell("c", 0, 1), "d": Cell("d", 1, 1)}


def bougainvillea(tex: Path) -> tuple[Part, np.ndarray, Path, np.ndarray]:
    """Bougainvillea mound (walls, slopes, entries): ~2.2 m tall, 3 m wide, arching canes, magenta bracts."""
    lc = ["#3E6127", "#4A6F2E", "#355421"]
    path = build_atlas(tex, [
        (BOUG_CELLS["a"], lambda w, h: draw_bracts(w, h, 111, lc, "#C81E6E", 0.55)),
        (BOUG_CELLS["b"], lambda w, h: draw_bracts(w, h, 112, lc, "#D42A7E", 0.45)),
        (BOUG_CELLS["c"], lambda w, h: draw_bracts(w, h, 113, lc, "#B5185F", 0.7)),
        (BOUG_CELLS["d"], lambda w, h: draw_bracts(w, h, 114, lc, "#D23F86", 0.3)),
    ], lambda w, h: draw_bark(w, h, 115, "#6B5A48", "#3E332A", "#857260", "fissured"))
    rng = np.random.default_rng(1111)
    pl = Plant(BOUG_CELLS)
    for k in range(7):  # arching canes
        az = 2 * math.pi * k / 7 + rng.normal(0, 0.2)
        d0 = np.array([math.cos(az), math.sin(az), 0.0])
        pts = [d0 * 1.4 * t + np.array([0, 0, 2.0 * math.sin(math.pi * 0.7 * t) * (1 - 0.3 * t)]) for t in np.linspace(0, 1, 5)]
        pl.tube(pts, [0.05, 0.04, 0.03, 0.02, 0.015], n=4, v_scale=0.1)
    cen = (0.0, 0.0, 0.7)
    clumps = [((0, 0, 1.2), (1.3, 1.3, 0.9)), ((0.8, 0.4, 0.9), (0.9, 0.9, 0.7)), ((-0.8, -0.3, 0.8), (0.9, 0.9, 0.7)),
              ((0.2, -0.8, 1.5), (0.8, 0.8, 0.6))]
    pl.canopy(rng, clumps, 300, ["a", "b", "c", "d"], (0.8, 1.2), cen, shell=0.45)
    p, N = pl.build()
    p.V[:, 2] = np.maximum(p.V[:, 2], 0.0)
    return p, N, path, pl.ao(inner=0.4)


AGAVE_CELLS = {"leaf": Cell("leaf", 0, 0), "leaf2": Cell("leaf2", 1, 0), "small": Cell("small", 0, 1), "gravel": Cell("gravel", 1, 1)}


def agave_cluster(tex: Path) -> tuple[Part, np.ndarray, Path, np.ndarray]:
    """Agave americana rosettes with pups and small echeveria (front-yard xeriscape): leaves are
    V-folded tapered blades (geometry), alpha-cut by the drawn teeth."""
    path = build_atlas(tex, [
        (AGAVE_CELLS["leaf"], lambda w, h: draw_agave_leaf(w, h, 121, "#7E9C90", "#B9B48A")),
        (AGAVE_CELLS["leaf2"], lambda w, h: draw_agave_leaf(w, h, 122, "#6F8E86", "#A9A47C")),
        (AGAVE_CELLS["small"], lambda w, h: draw_agave_leaf(w, h, 123, "#8DA79A", "#C4A6A0")),
        (AGAVE_CELLS["gravel"], lambda w, h: draw_dense(w, h, 124, ["#9C9284", "#B3A996", "#857B6E"], (3, 5), 1800)),
    ], None)
    rng = np.random.default_rng(1212)
    pl = Plant(AGAVE_CELLS)

    def rosette(cx: float, cy: float, scale: float, n: int, cell: str) -> None:
        for i in range(n):
            az = i * 2.39996 + rng.normal(0, 0.05)
            t = i / n
            elev = math.radians(80 - 65 * t + rng.normal(0, 4))  # inner leaves upright, outer leaves spread
            L = scale * (0.55 + 0.45 * math.sin(math.pi * min(1, 0.25 + t)))
            dvec = np.array([math.cos(az) * math.cos(elev), math.sin(az) * math.cos(elev), math.sin(elev)])
            side = np.array([-math.sin(az), math.cos(az), 0.0])
            pts = []
            for k in range(6):
                s_ = k / 5
                curl = np.array([0, 0, -0.18 * L * s_ * s_]) if elev < math.radians(45) else np.zeros(3)
                pts.append(np.array([cx, cy, 0.05]) + dvec * L * s_ + curl)
            pl.strip(pts, side, L * 0.3, cell, canopy_center=(cx, cy, -0.3 * scale), bend=0.35, fold=0.45, mid=0.5, taper=True)

    rosette(0.0, 0.0, 0.95, 22, "leaf")
    rosette(0.95, 0.35, 0.55, 15, "leaf2")
    rosette(-0.7, 0.75, 0.45, 13, "leaf2")
    for _ in range(5):  # echeveria-like small rosettes
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.9, 1.4)
        rosette(math.cos(a) * r, math.sin(a) * r, 0.16, 9, "small")
    p, N = pl.build()
    return p, N, path, pl.ao(inner=0.3, bottom=0.25)


HEDGE_CELLS = {"dense": Cell("dense", 0, 0), "dense2": Cell("dense2", 1, 0), "tuft": Cell("tuft", 0, 1), "tuft2": Cell("tuft2", 1, 1)}


def hedge(tex: Path) -> tuple[Part, np.ndarray, Path, np.ndarray]:
    """Clipped hedge segment (pittosporum / podocarpus / boxwood): 2.0 m long along the prop's x axis,
    0.9 m wide, 1.3 m tall. An opaque leafy box plus fuzzy cards on its faces; tile segments end to end."""
    cols = ["#2F4D22", "#3B5D28", "#476C30", "#2A451E"]
    path = build_atlas(tex, [
        (HEDGE_CELLS["dense"], lambda w, h: draw_dense(w, h, 131, cols, (5, 8))),
        (HEDGE_CELLS["dense2"], lambda w, h: draw_dense(w, h, 132, cols[1:] + cols[:1], (5, 8))),
        (HEDGE_CELLS["tuft"], lambda w, h: draw_broadleaf(w, h, 133, cols, (5, 8), 0.55, 1200, clumps=9, spread=0.7)),
        (HEDGE_CELLS["tuft2"], lambda w, h: draw_broadleaf(w, h, 134, cols[1:], (5, 8), 0.55, 1100, clumps=9, spread=0.7)),
    ], None)
    rng = np.random.default_rng(1313)
    pl = Plant(HEDGE_CELLS)
    L, W, H = 2.0, 0.9, 1.3
    hx, hy = L / 2, W / 2
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    u0, v0, u1, v1 = cell_uv(HEDGE_CELLS["dense"])
    for i in range(4):  # sides
        (ax, ay), (bx, by) = corners[i], corners[(i + 1) % 4]
        V = np.array([(ax, ay, 0.0), (bx, by, 0.0), (bx, by, H), (ax, ay, H)])
        seg = math.hypot(bx - ax, by - ay)
        uu = u0 + (u1 - u0) * min(1.0, seg / 2.0)
        pl._add(V, [[0, 1, 2, 3]], [[(u0, v0), (uu, v0), (uu, v1), (u0, v1)]], None, kind=0)
    V = np.array([(-hx, -hy, H), (hx, -hy, H), (hx, hy, H), (-hx, hy, H)])
    pl._add(V, [[0, 1, 2, 3]], [[(u0, v0), (u1, v0), (u1, v0 + (v1 - v0) * 0.45), (u0, v0 + (v1 - v0) * 0.45)]], None, kind=0)
    cen = (0.0, 0.0, H * 0.45)
    for _ in range(170):  # fuzz: cards just outside the faces, leaning outward
        face = rng.integers(5)
        if face == 4:
            pt = np.array([rng.uniform(-hx, hx), rng.uniform(-hy, hy), H - 0.02])
            nrm = np.array([0, 0, 1.0]) + rng.normal(size=3) * 0.2
        else:
            (ax, ay), (bx, by) = corners[face], corners[(face + 1) % 4]
            t = rng.uniform(0, 1)
            out = np.array([by - ay, -(bx - ax), 0.0])
            out /= np.linalg.norm(out)
            pt = np.array([ax + (bx - ax) * t, ay + (by - ay) * t, rng.uniform(0.15, H - 0.05)]) + out * 0.04
            nrm = out + rng.normal(size=3) * 0.5
        sz = rng.uniform(0.35, 0.55)
        pl.card(pt, nrm, sz, sz * 1.14, "tuft" if rng.random() < 0.5 else "tuft2", roll=rng.uniform(-1, 1), canopy_center=cen, bend=0.5)
    p, N = pl.build()
    return p, N, path, pl.ao(inner=0.25, bottom=0.35)


TUFT_CELLS = {"a": Cell("a", 0, 0), "b": Cell("b", 1, 0), "c": Cell("c", 0, 1), "d": Cell("d", 1, 1)}


def lawn_tuft(tex: Path) -> tuple[Part, np.ndarray, Path, np.ndarray]:
    """Lawn grass tuft (3 crossed cards, ~0.22 m) for procedural near-camera scattering over lawns."""
    path = build_atlas(tex, [
        (TUFT_CELLS["a"], lambda w, h: draw_grass(w, h, 141, ["#4E7A2E", "#5E8A36", "#3F6A26"], n=220)),
        (TUFT_CELLS["b"], lambda w, h: draw_grass(w, h, 142, ["#557F31", "#6B9440", "#46702A"], n=200)),
        (TUFT_CELLS["c"], lambda w, h: draw_grass(w, h, 143, ["#5A8034", "#77984A", "#4A7029"], n=180)),
        (TUFT_CELLS["d"], lambda w, h: draw_grass(w, h, 144, ["#62853A", "#85A050", "#557A30"], n=160)),
    ], None)
    rng = np.random.default_rng(1414)
    pl = Plant(TUFT_CELLS)
    for k in range(3):
        az = math.pi * k / 3 + rng.normal(0, 0.1)
        nrm = np.array([math.cos(az), math.sin(az), 0.1])
        pl.card((0, 0, 0.0), nrm, 0.32, 0.22, "abcd"[k], canopy_center=(0, 0, -0.3), bend=0.7, anchor="bottom")
    p, N = pl.build()
    return p, N, path, np.clip(0.55 + 0.45 * p.V[:, 2] / 0.22, 0, 1)


@dataclass
class Built:
    part: Part
    normals: np.ndarray
    tex: Path
    ao: np.ndarray


def build(tid: str, tex: Path) -> Built:
    fn, _ = SPECIES[tid]
    part, normals, path, ao = fn(tex)
    return Built(part, normals, path, ao)

SPECIES: dict[str, tuple[Callable[[Path], tuple[Part, np.ndarray, Path, np.ndarray]], dict]] = {
    "tree_oak": (coast_live_oak, {"kind": "tree", "height_m": 9.0, "radius_m": 6.0,
                                  "notes": "Coast live oak (Quercus agrifolia): canyons, slopes, open space edges, large yards"}),
    "tree_palm_fan": (mexican_fan_palm, {"kind": "tree", "height_m": 17.5, "radius_m": 2.4,
                                         "notes": "Mexican fan palm (Washingtonia robusta): arterial medians, commercial centers"}),
    "tree_palm_queen": (queen_palm, {"kind": "tree", "height_m": 13.0, "radius_m": 3.6,
                                     "notes": "Queen palm (Syagrus romanzoffiana): entries, yards, commercial landscaping"}),
    "tree_eucalyptus": (eucalyptus, {"kind": "tree", "height_m": 21.5, "radius_m": 5.5,
                                     "notes": "Eucalyptus windbreak: older edges, slopes, canyon rims"}),
    "tree_jacaranda": (jacaranda, {"kind": "tree", "height_m": 8.3, "radius_m": 5.0,
                                   "notes": "Jacaranda (green in autumn, sparse late bloom): residential streets and yards"}),
    "tree_street": (street_tree, {"kind": "tree", "height_m": 8.2, "radius_m": 3.6,
                                  "notes": "Brisbane box (Lophostemon confertus) parkway tree; stands in for elm / pistache"}),
    "tree_ficus": (ficus, {"kind": "tree", "height_m": 8.0, "radius_m": 4.0,
                           "notes": "Indian laurel fig (Ficus microcarpa): commercial centers, parking lots, clipped domes"}),
    "tree_pine_canary": (canary_pine, {"kind": "tree", "height_m": 20.0, "radius_m": 3.5,
                                       "notes": "Canary Island pine (Pinus canariensis): parks, slopes, school edges, older streets"}),
    "shrub": (shrub, {"kind": "shrub", "height_m": 1.3, "radius_m": 1.2,
                      "notes": "Irrigated landscape shrub mound with bougainvillea-pink accents"}),
    "grass_ornamental": (ornamental_grass, {"kind": "shrub", "height_m": 0.9, "radius_m": 0.6,
                                            "notes": "Ornamental bunch grass (drought-tolerant front-yard planting)"}),
    "shrub_bougainvillea": (bougainvillea, {"kind": "shrub", "height_m": 2.2, "radius_m": 1.6,
                                            "notes": "Bougainvillea mound (magenta bracts): walls, entries, slopes"}),
    "succulent_agave": (agave_cluster, {"kind": "shrub", "height_m": 1.1, "radius_m": 1.5,
                                        "notes": "Agave americana rosettes + echeveria (xeriscape front yards, medians)"}),
    "hedge": (hedge, {"kind": "shrub", "height_m": 1.3, "radius_m": 1.1,
                      "notes": "Clipped hedge SEGMENT 2.0 m along local x (0.9 m wide): tile end to end along lot lines"}),
    "grass_tuft": (lawn_tuft, {"kind": "groundcover", "height_m": 0.22, "radius_m": 0.17,
                               "notes": "Lawn grass tuft for procedural near-camera scattering over lawn ground (not placed)"}),
}
