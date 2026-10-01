"""Tileable PBR texture synthesis (pure numpy + PIL; runs in the Blender venv and the main venv).

Everything here is periodic: noise comes from FFT-filtered white noise, Worley cells wrap
around the edges and every blur / derivative uses wrap-around, so a generated tile repeats
seamlessly. Heights are in METERS, so normal maps and ambient occlusion come out with real
strengths for the tile's physical size.

Image convention: arrays are (rows, cols) with row 0 at the TOP of the image. `Canvas.Y`
is meters measured UP from the bottom edge (so facade code reads like elevation drawings).
Normal maps follow glTF / OpenGL / three.js / Blender: +X right, +Y up (toward row 0), +Z out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# periodic noise
# ---------------------------------------------------------------------------


def _freqs(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    fy = np.fft.fftfreq(shape[0])[:, None] * shape[0]
    fx = np.fft.fftfreq(shape[1])[None, :] * shape[1]
    return fy, fx


def fbm(shape: tuple[int, int], seed: int, beta: float = 2.0, fmin: float = 1.0, fmax: float | None = None,
        aniso: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Periodic fractal noise (power spectrum 1/f^beta between fmin and fmax cycles per tile).
    Returns zero mean, unit std. aniso=(sx, sy) stretches features (sx > 1: longer along x)."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape)
    fy, fx = _freqs(shape)
    f = np.hypot(fx * aniso[0], fy * aniso[1])
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[f < fmin] = 0.0
    if fmax is not None:
        amp *= np.exp(-((f / fmax) ** 4))
    amp[0, 0] = 0.0
    out = np.real(np.fft.ifft2(np.fft.fft2(w) * amp))
    s = out.std()
    return out / s if s > 0 else out


def blur(a: np.ndarray, sigma_px: float | tuple[float, float]) -> np.ndarray:
    """Periodic gaussian blur (sigma in pixels; tuple = (sigma_x, sigma_y))."""
    if np.isscalar(sigma_px):
        sx = sy = float(sigma_px)
    else:
        sx, sy = sigma_px  # type: ignore[misc]
    if sx <= 0 and sy <= 0:
        return a
    fy, fx = np.fft.fftfreq(a.shape[0])[:, None], np.fft.fftfreq(a.shape[1])[None, :]
    g = np.exp(-2 * math.pi**2 * ((fx * sx) ** 2 + (fy * sy) ** 2))
    if a.ndim == 3:
        return np.stack([np.real(np.fft.ifft2(np.fft.fft2(a[..., c]) * g)) for c in range(a.shape[2])], axis=-1)
    return np.real(np.fft.ifft2(np.fft.fft2(a) * g))


def blur_oriented(a: np.ndarray, along: float, across: float, angle: float) -> np.ndarray:
    """Periodic anisotropic gaussian blur: sigma `along` (px) in direction `angle` (radians from +x), `across` normal to it."""
    fy, fx = np.fft.fftfreq(a.shape[0])[:, None], np.fft.fftfreq(a.shape[1])[None, :]
    c, s = math.cos(angle), math.sin(angle)
    fu = fx * c + fy * s
    fv = -fx * s + fy * c
    g = np.exp(-2 * math.pi**2 * ((fu * along) ** 2 + (fv * across) ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(a) * g))


def norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


@dataclass
class Cells:
    f1: np.ndarray  # distance to the nearest feature point (pixels)
    f2: np.ndarray  # distance to the second nearest
    cid: np.ndarray  # integer id of the nearest cell
    rnd: np.ndarray  # per-cell uniform random [0,1) (same for every pixel of a cell)
    dx: np.ndarray  # offset to the nearest point (pixels, x)
    dy: np.ndarray


def worley(shape: tuple[int, int], cells: tuple[int, int], seed: int, jitter: float = 0.9) -> Cells:
    """Periodic Worley / Voronoi noise with cells=(nx, ny) jittered points over the tile."""
    H, W = shape
    nx, ny = cells
    rng = np.random.default_rng(seed)
    px = (np.arange(nx)[None, :] + 0.5 + rng.uniform(-0.5, 0.5, (ny, nx)) * jitter) * (W / nx)
    py = (np.arange(ny)[:, None] + 0.5 + rng.uniform(-0.5, 0.5, (ny, nx)) * jitter) * (H / ny)
    rnd_tab = rng.random((ny, nx))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64) + 0.5
    cx = np.floor(xx / (W / nx)).astype(int)
    cy = np.floor(yy / (H / ny)).astype(int)
    f1 = np.full(shape, 1e9)
    f2 = np.full(shape, 1e9)
    cid = np.zeros(shape, int)
    dxo = np.zeros(shape)
    dyo = np.zeros(shape)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            jx, jy = cx + ox, cy + oy
            wx, wy = jx % nx, jy % ny
            qx = px[wy, wx] + (jx - wx) * (W / nx)
            qy = py[wy, wx] + (jy - wy) * (H / ny)
            ddx, ddy = qx - xx, qy - yy
            d = np.hypot(ddx, ddy)
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            f1 = np.where(closer, d, f1)
            cid = np.where(closer, wy * nx + wx, cid)
            dxo = np.where(closer, ddx, dxo)
            dyo = np.where(closer, ddy, dyo)
    return Cells(f1, f2, cid, rnd_tab.ravel()[cid], dxo, dyo)


# ---------------------------------------------------------------------------
# derived maps
# ---------------------------------------------------------------------------


def gradient(h: np.ndarray, px_m: float, py_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(dh/dx, dh/dy_up) with wrap-around central differences (meters per meter)."""
    py_m = py_m or px_m
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * px_m)
    dy = (np.roll(h, 1, axis=0) - np.roll(h, -1, axis=0)) / (2 * py_m)  # row 0 is the top: up = -row
    return dx, dy


def normal_map(h: np.ndarray, px_m: float, strength: float = 1.0, py_m: float | None = None) -> np.ndarray:
    dx, dy = gradient(h, px_m, py_m)
    n = np.stack([-dx * strength, -dy * strength, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def horizon_ao(h: np.ndarray, px_m: float, radius_m: float, dirs: int = 8, steps: int = 10, power: float = 1.0,
               py_m: float | None = None) -> np.ndarray:
    """Horizon-based ambient occlusion of a periodic heightfield (1 = open sky)."""
    py_m = py_m or px_m
    rad_px = max(1.0, radius_m / px_m)
    acc = np.zeros_like(h)
    for k in range(dirs):
        a = 2 * math.pi * (k + 0.5) / dirs
        ux, uy = math.cos(a), math.sin(a)
        best = np.zeros_like(h)
        for s in range(1, steps + 1):
            r = rad_px * (s / steps) ** 1.5
            ox, oy = int(round(ux * r)), int(round(uy * r))
            if ox == 0 and oy == 0:
                continue
            dist = math.hypot(ox * px_m, oy * py_m)
            dh = np.roll(np.roll(h, -oy, axis=0), -ox, axis=1) - h
            best = np.maximum(best, np.arctan2(dh, dist))
        acc += np.sin(best)
    ao = 1.0 - acc / dirs
    return np.clip(ao, 0, 1) ** power


def cavity(h: np.ndarray, px_m: float, sigma_m: float) -> np.ndarray:
    """Signed local relief: >0 on bumps, <0 in crevices (meters)."""
    return h - blur(h, sigma_m / px_m)


# ---------------------------------------------------------------------------
# canvas with analytic antialiased shapes (meters, y up)
# ---------------------------------------------------------------------------


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


@dataclass
class Canvas:
    """One material tile: base color (sRGB 0..1), height (m), roughness, metalness, extra AO, tint mask.

    mask: 1 = wall (the shader tints with the building's wall color), 0.5 = paintable near-white
    element (garage doors, service doors: optional accent tint), 0 = fixed colors.
    """

    w_m: float
    h_m: float
    px: int  # pixels along x; y pixels follow from the aspect unless py_px is given
    py_px: int | None = None
    alb: np.ndarray = field(init=False)
    h: np.ndarray = field(init=False)
    rough: np.ndarray = field(init=False)
    metal: np.ndarray = field(init=False)
    ao: np.ndarray = field(init=False)
    mask: np.ndarray = field(init=False)
    emit: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.py = self.py_px or int(round(self.px * self.h_m / self.w_m))
        self.m = self.w_m / self.px  # meters per pixel along x
        self.my = self.h_m / self.py  # meters per pixel along y (differs only for non-square strips)
        shape = (self.py, self.px)
        self.alb = np.full((*shape, 3), 0.5)
        self.h = np.zeros(shape)
        self.rough = np.full(shape, 0.8)
        self.metal = np.zeros(shape)
        self.ao = np.ones(shape)
        self.mask = np.zeros(shape)
        self.emit = np.zeros(shape)  # night-light mask (windows / lamps), optional
        xs = (np.arange(self.px) + 0.5) * self.m
        ys = self.h_m - (np.arange(self.py) + 0.5) * self.my
        self.X, self.Y = np.meshgrid(xs, ys)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.py, self.px)

    # -- coverage primitives (0..1, antialiased) --------------------------------
    def _ramp(self, d: np.ndarray) -> np.ndarray:
        return np.clip(d / self.m + 0.5, 0.0, 1.0)

    def rect(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        X, Y = self.X, self.Y
        return self._ramp(X - x0) * self._ramp(x1 - X) * self._ramp(Y - y0) * self._ramp(y1 - Y)

    def hband(self, y0: float, y1: float) -> np.ndarray:
        return self._ramp(self.Y - y0) * self._ramp(y1 - self.Y)

    def vband(self, x0: float, x1: float) -> np.ndarray:
        return self._ramp(self.X - x0) * self._ramp(x1 - self.X)

    def ellipse(self, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
        d = np.hypot((self.X - cx) / rx, (self.Y - cy) / ry)
        return np.clip((1 - d) * min(rx, ry) / self.m + 0.5, 0, 1)

    def arch(self, x0: float, y0: float, x1: float, y_spring: float) -> np.ndarray:
        """Rectangle x0..x1 from y0 up to the springline, topped by a half circle."""
        r = (x1 - x0) / 2
        cx = (x0 + x1) / 2
        box = self.rect(x0, y0, x1, y_spring)
        circ = np.clip((r - np.hypot(self.X - cx, self.Y - y_spring)) / self.m + 0.5, 0, 1) * self._ramp(self.Y - y_spring)
        return np.maximum(box, circ)

    def bevel(self, x0: float, y0: float, x1: float, y1: float, b: float) -> np.ndarray:
        """0 at the rect border rising to 1 at distance b inside (pyramid / chamfer profile)."""
        d = np.minimum(np.minimum(self.X - x0, x1 - self.X), np.minimum(self.Y - y0, y1 - self.Y))
        return np.clip(d / b, 0.0, 1.0)

    # -- painting ------------------------------------------------------------
    def put(self, cov: np.ndarray, color: np.ndarray | str | None = None, h: float | np.ndarray | None = None,
            dh: float | np.ndarray | None = None, rough: float | np.ndarray | None = None,
            metal: float | np.ndarray | None = None, mask: float | None = None, ao: float | np.ndarray | None = None,
            emit: float | None = None) -> None:
        c = cov[..., None]
        if color is not None:
            col = hex_rgb(color) if isinstance(color, str) else np.asarray(color, float)
            self.alb = self.alb * (1 - c) + col * c
        if h is not None:
            self.h = self.h * (1 - cov) + np.asarray(h, float) * cov
        if dh is not None:
            self.h = self.h + np.asarray(dh, float) * cov
        if rough is not None:
            self.rough = self.rough * (1 - cov) + np.asarray(rough, float) * cov
        if metal is not None:
            self.metal = self.metal * (1 - cov) + np.asarray(metal, float) * cov
        if mask is not None:
            self.mask = self.mask * (1 - cov) + mask * cov
        if ao is not None:
            self.ao = self.ao * (1 - cov) + np.asarray(ao, float) * cov
        if emit is not None:
            self.emit = self.emit * (1 - cov) + emit * cov

    def noise(self, seed: int, scale_m: float, beta: float = 2.0, octaves_m: float | None = None,
              aniso: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """Periodic fbm with features from scale_m down to octaves_m (meters); zero mean, unit std."""
        fmin = max(1.0, self.w_m / scale_m * 0.5)
        fmax = self.w_m / octaves_m if octaves_m else None
        return fbm(self.shape, seed, beta, fmin=fmin, fmax=fmax, aniso=aniso)

    # -- finishing -------------------------------------------------------------
    def maps(self, ao_radius_m: float = 0.05, normal_strength: float = 1.0, ao_power: float = 1.0) -> dict[str, np.ndarray]:
        sh = self.shape
        for k in ("h", "rough", "metal", "ao", "mask", "emit"):
            setattr(self, k, np.broadcast_to(np.asarray(getattr(self, k), float), sh).copy())
        self.alb = np.broadcast_to(np.asarray(self.alb, float), (*sh, 3)).copy()
        ao = horizon_ao(self.h, self.m, ao_radius_m, power=ao_power, py_m=self.my) * self.ao
        return {
            "albedo": np.clip(self.alb, 0, 1),
            "normal": normal_map(self.h, self.m, normal_strength, py_m=self.my),
            "ao": np.clip(ao, 0, 1),
            "rough": np.clip(self.rough, 0.02, 1),
            "metal": np.clip(self.metal, 0, 1),
            "mask": np.clip(self.mask, 0, 1),
            "emit": np.clip(self.emit, 0, 1),
            "height": self.h,
        }


# ---------------------------------------------------------------------------
# atlas assembly and file output
# ---------------------------------------------------------------------------


def pad_wrap(a: np.ndarray, g: int) -> np.ndarray:
    pad = ((g, g), (g, g)) + (((0, 0),) if a.ndim == 3 else ())
    return np.pad(a, pad, mode="wrap")


def to_u8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)


def save_jpg(path: Path, rgb: np.ndarray, quality: int = 90) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_u8(rgb), "RGB").save(path, quality=quality, optimize=True, progressive=True, subsampling=0)


def save_png(path: Path, a: np.ndarray, mode: str) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_u8(a), mode).save(path, optimize=True)
