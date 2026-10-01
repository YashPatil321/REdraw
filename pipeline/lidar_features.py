"""Lidar features: DSM / DTM / nDSM rasters, per-building roof models, lidar-only buildings and
individual trees, from the EPSG:32611 tiles written by pipeline/fetch_lidar.py.

Outputs (data/raw/lidar/, documented in docs/data_contract.md "Lidar features"):

    dsm_0p5m.tif            max surface of all non-noise returns, NAVD88 m, 0.5 m
    dtm_0p5m.tif            lidar bare earth (ground classes 2 + 22), Delaunay-interpolated
    ndsm_0p5m.tif           dsm - dtm (height above ground), m, >= 0
    chm_0p5m.tif            canopy height model (nDSM on vegetation cells, else 0)
    ground_change_0p5m.tif  3DEP 2024 DEM (dem_3dep_10m.tif) - lidar 2014 DTM, m
    buildings_roofs.parquet roof model for every footprint of osm_buildings.geojson in the bbox
    missing_buildings.geojson lidar buildings without a footprint (WGS84), with roof models
    trees.parquet           individual trees (tree tops + watershed crowns)
    lidar_validation.json   stats (lidar vs Overture heights, registration, timings)

Class codes in CA_SanDiegoQL2_2014 (checked on the data): 1 unclassified, 2 ground,
7 noise, 21 unclassified in swath overlap, 22 ground in swath overlap. There is no building
or vegetation class, so buildings and trees are told apart geometrically: roofs are
planar (small local plane-fit residual) and opaque (few multi-return pulses), canopy is
rough and penetrable.

    .venv/bin/python -m pipeline.lidar_features            # resume (per-tile work cached)
    .venv/bin/python -m pipeline.lidar_features --force    # recompute every tile
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import ndimage as ndi

from pipeline.common import log, now_iso, write_json
from pipeline.config import assumption, raw_dir
from pipeline.fetch_lidar import TILE_M, Box, lidar_dir, tile_keys_for, tile_name, utm_bbox
from pipeline.geo import PROJECTION, scene_origin

GROUND_CLASSES = (2, 22)  # 22 = ground in swath overlap (project convention, see module doc)
NOISE_CLASSES = (7, 18)
HALO_M = 100.0  # points read around each 1 km tile so edge buildings / crowns are complete
SEED = 20140901

# Geometric building / vegetation discrimination (calibrated on footprints vs. non-footprint
# canopy in this dataset, see lidar_validation.json "feature_calibration").
PLANE_RESID_MAX_M = 0.20  # 1.5 m window plane-fit residual std on roofs
PEN_MAX = 0.30  # fraction of elevated returns that come from multi-return pulses
VEG_MIN_H_M = 2.0


# ---------------------------------------------------------------------------
# Raster grid helpers (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """North-up raster grid: (x0, y1) is the NW corner, rows go south, cols go east."""

    x0: float
    y1: float
    res: float
    nrows: int
    ncols: int

    @staticmethod
    def from_box(b: Box, res: float) -> Grid:
        return Grid(b.minx, b.maxy, res, int(round((b.maxy - b.miny) / res)), int(round((b.maxx - b.minx) / res)))

    @property
    def box(self) -> Box:
        return Box(self.x0, self.y1 - self.nrows * self.res, self.x0 + self.ncols * self.res, self.y1)

    @property
    def shape(self) -> tuple[int, int]:
        return self.nrows, self.ncols

    def transform(self) -> Any:
        from rasterio.transform import Affine

        return Affine(self.res, 0.0, self.x0, 0.0, -self.res, self.y1)

    def rc(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = np.floor((x - self.x0) / self.res).astype(np.int64)
        r = np.floor((self.y1 - y) / self.res).astype(np.int64)
        ok = (r >= 0) & (r < self.nrows) & (c >= 0) & (c < self.ncols)
        return r, c, ok

    def frac_rc(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fractional (row, col) of points in pixel-center coordinates (for map_coordinates)."""
        return (self.y1 - y) / self.res - 0.5, (x - self.x0) / self.res - 0.5

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x0 + (np.arange(self.ncols) + 0.5) * self.res
        ys = self.y1 - (np.arange(self.nrows) + 0.5) * self.res
        return xs, ys


def rasterize_stat(grid: Grid, x: np.ndarray, y: np.ndarray, v: np.ndarray, how: str) -> np.ndarray:
    """Per-cell max / min / mean / sum / count of point values; NaN (0 for count/sum) if empty."""
    r, c, ok = grid.rc(x, y)
    idx = r[ok] * grid.ncols + c[ok]
    vv = np.asarray(v, dtype=np.float64)[ok]
    n = grid.nrows * grid.ncols
    if how == "count":
        return np.bincount(idx, minlength=n).reshape(grid.shape).astype(np.float32)
    if how == "sum":
        return np.bincount(idx, weights=vv, minlength=n).reshape(grid.shape).astype(np.float32)
    if how == "mean":
        s = np.bincount(idx, weights=vv, minlength=n)
        k = np.bincount(idx, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(k > 0, s / np.maximum(k, 1), np.nan)
        return out.reshape(grid.shape).astype(np.float32)
    if how in ("max", "min"):
        fill = -np.inf if how == "max" else np.inf
        out = np.full(n, fill)
        (np.maximum if how == "max" else np.minimum).at(out, idx, vv)
        out[~np.isfinite(out)] = np.nan
        return out.reshape(grid.shape).astype(np.float32)
    raise ValueError(how)


def fill_nan(a: np.ndarray, smooth_iters: int = 3) -> np.ndarray:
    """Fill NaN cells: a few passes of 3x3 normalized averaging, then nearest valid cell."""
    a = np.array(a, dtype=np.float32, copy=True)
    for _ in range(smooth_iters):
        m = np.isfinite(a)
        if m.all() or not m.any():
            break
        s = ndi.uniform_filter(np.where(m, a, 0.0).astype(np.float64), 3)
        w = ndi.uniform_filter(m.astype(np.float64), 3)
        f = ~m & (w > 1e-9)
        a[f] = (s[f] / w[f]).astype(np.float32)
    m = np.isfinite(a)
    if not m.all() and m.any():
        _, (ri, ci) = ndi.distance_transform_edt(~m, return_indices=True)
        a = a[ri, ci]
    return a


def sample_bilinear(grid: Grid, a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r, c = grid.frac_rc(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return ndi.map_coordinates(a, [r, c], order=1, mode="nearest").astype(np.float64)


def interpolate_ground(grid: Grid, x: np.ndarray, y: np.ndarray, z: np.ndarray, cell: float = 1.0) -> np.ndarray:
    """Bare-earth surface on `grid` from ground points: per-cell mean at `cell` m, then linear
    (Delaunay) interpolation across gaps (under buildings, dense canopy), nearest outside hull."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    coarse = Grid(grid.x0, grid.y1, cell, int(math.ceil(grid.nrows * grid.res / cell)), int(math.ceil(grid.ncols * grid.res / cell)))
    zc = rasterize_stat(coarse, x, y, z, "mean")
    ok = np.isfinite(zc)
    if ok.sum() < 3:
        return np.full(grid.shape, np.nan, dtype=np.float32)
    xs, ys = coarse.centers()
    full = zc.astype(np.float64)
    miss = ~ok
    if miss.any():
        # triangulate only the valid cells that border a gap (same result inside the gaps as a
        # full Delaunay, a fraction of the memory)
        edge = ok & ndi.binary_dilation(miss, structure=np.ones((3, 3)), iterations=2)
        rr, cc = np.nonzero(edge)
        pts = np.column_stack([xs[cc], ys[rr]])
        vals = zc[edge].astype(np.float64)
        mr, mc = np.nonzero(miss)
        q = np.column_stack([xs[mc], ys[mr]])
        est = LinearNDInterpolator(pts, vals)(q) if len(pts) >= 3 else np.full(len(q), np.nan)
        bad = ~np.isfinite(est)
        if bad.any():
            est[bad] = NearestNDInterpolator(pts, vals)(q[bad])
        full[mr, mc] = est
    fr, fc = np.meshgrid((np.arange(grid.nrows) + 0.5) * grid.res / cell - 0.5,
                         (np.arange(grid.ncols) + 0.5) * grid.res / cell - 0.5, indexing="ij")
    return ndi.map_coordinates(full, [fr, fc], order=1, mode="nearest").astype(np.float32)


def plane_residual_std(dsm: np.ndarray, size: int = 3) -> np.ndarray:
    """Std of the residual of a least-squares plane z = a + b*dx + c*dy fitted in each
    size x size window (closed form via box filters). Roofs ~0.1 m, canopy ~0.5 m+."""
    z = dsm.astype(np.float64)
    h = size // 2
    off = np.arange(-h, h + 1, dtype=np.float64)
    kx = np.tile(off, (size, 1))  # column offset (east)
    ky = kx.T  # row offset (south)
    n = float(size * size)
    sxx = float((kx**2).sum())
    mz = ndi.uniform_filter(z, size, mode="nearest")
    mzz = ndi.uniform_filter(z * z, size, mode="nearest")
    # correlate (not convolve) so kernel offsets match neighbour positions
    szx = ndi.correlate(z, kx, mode="nearest")
    szy = ndi.correlate(z, ky, mode="nearest")
    var = mzz - mz**2 - (szx**2 + szy**2) / (sxx * n)
    return np.sqrt(np.clip(var * n / max(n - 3.0, 1.0), 0.0, None)).astype(np.float32)


def disk(radius_px: float) -> np.ndarray:
    r = int(math.ceil(radius_px))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (xx**2 + yy**2) <= radius_px**2 + 1e-9


# ---------------------------------------------------------------------------
# Roof planes (pure)
# ---------------------------------------------------------------------------


@dataclass
class Plane:
    normal: np.ndarray  # unit (e, n, up), up component >= 0
    d: float  # normal . p = d
    inliers: np.ndarray  # indices into the building's point array
    rmse: float = 0.0

    @property
    def slope_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.normal[2], -1.0, 1.0))))

    @property
    def aspect_deg(self) -> float:
        """Compass azimuth (deg, clockwise from north) the plane faces / drains toward."""
        return float(np.degrees(np.arctan2(self.normal[0], self.normal[1])) % 360.0)


def fit_plane_lsq(p: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Total least squares plane through points (N,3) -> (unit normal with up >= 0, d, rmse)."""
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    d = float(n @ c)
    rmse = float(np.sqrt(np.mean((p @ n - d) ** 2)))
    return n, d, rmse


def point_normals(p: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-point unit normals (up >= 0) from PCA of the k nearest neighbours in 3D (3D, not
    plan-view, neighbours keep a low roof next to a high wall from mixing with the upper roof)."""
    from scipy.spatial import cKDTree

    n = len(p)
    if n < 4:
        return np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    k = min(k, n)
    _, idx = cKDTree(p).query(p, k=k)
    nb = p[idx]
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / k
    _, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0].copy()
    nrm[nrm[:, 2] < 0] *= -1.0
    return nrm


def largest_component(xy: np.ndarray, cell: float) -> np.ndarray:
    """Mask of the points in the largest 8-connected cluster of occupied `cell` m cells."""
    if len(xy) == 0:
        return np.zeros(0, dtype=bool)
    ij = np.floor((xy - xy.min(axis=0)) / cell).astype(np.int64)
    occ = np.zeros(tuple(ij.max(axis=0) + 1), dtype=bool)
    occ[ij[:, 0], ij[:, 1]] = True
    lab, n = ndi.label(occ, structure=np.ones((3, 3)))
    if n <= 1:
        return np.ones(len(xy), dtype=bool)
    lp = lab[ij[:, 0], ij[:, 1]]
    return lp == int(np.argmax(np.bincount(lp)))


def ransac_planes(
    p: np.ndarray,
    threshold: float,
    min_inliers: int,
    max_planes: int = 8,
    n_hyp: int = 300,
    max_slope_deg: float = 65.0,
    rng: np.random.Generator | None = None,
    normals: np.ndarray | None = None,
    normal_tol_deg: float = 30.0,
    cell: float | None = None,
) -> list[Plane]:
    """Sequential RANSAC: repeatedly take the plane with most inliers, refine by least squares,
    remove its inliers. An inlier must lie within `threshold` of the plane and, when per-point
    `normals` are given, have a normal within `normal_tol_deg` of the plane's (so planes cannot
    cut across ridges); with `cell`, only the largest spatially connected inlier patch is kept.
    Planes steeper than max_slope_deg (walls, tree flanks) are rejected."""
    rng = rng or np.random.default_rng(SEED)
    remaining = np.arange(len(p))
    planes: list[Plane] = []
    cos_max = math.cos(math.radians(max_slope_deg))
    cos_n = math.cos(math.radians(normal_tol_deg))

    def inliers(q: np.ndarray, qn: np.ndarray | None, nb: np.ndarray, db: float) -> np.ndarray:
        m = np.abs(q @ nb - db) < threshold
        if qn is not None:
            m &= np.abs(qn @ nb) >= cos_n
        return m

    fails = 0
    while len(remaining) >= max(min_inliers, 3) and len(planes) < max_planes and fails < 3:
        q = p[remaining]
        qn = normals[remaining] if normals is not None else None
        score_idx = rng.choice(len(q), size=1500, replace=False) if len(q) > 1500 else np.arange(len(q))
        s = q[score_idx]
        if qn is not None:
            # 1-point hypotheses: seed point + its local PCA normal (finds small facets that
            # random global triples almost never land on)
            seeds = rng.choice(len(q), size=min(n_hyp // 3, len(q)), replace=False)
            a, n = q[seeds], qn[seeds].copy()
        else:
            tri = rng.integers(0, len(q), size=(n_hyp, 3))
            a, b, c = q[tri[:, 0]], q[tri[:, 1]], q[tri[:, 2]]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n, axis=1)
            good = norm > 1e-6
            n = n[good] / norm[good, None]
            a = a[good]
        n[n[:, 2] < 0] *= -1.0
        keep = n[:, 2] >= cos_max
        n, a = n[keep], a[keep]
        if len(n) == 0:
            break
        d = np.einsum("ij,ij->i", n, a)
        ok = np.abs(s @ n.T - d[None, :]) < threshold
        if qn is not None:
            ok &= np.abs(qn[score_idx] @ n.T) >= cos_n
        counts = ok.sum(axis=0)
        best = int(np.argmax(counts))
        scale = len(q) / len(s)
        if counts[best] * scale < min_inliers:
            fails += 1
            continue
        nb, db = n[best], float(d[best])
        inl = inliers(q, qn, nb, db)
        for _ in range(2):  # refine
            if inl.sum() < 3:
                break
            nb, db, _ = fit_plane_lsq(q[inl])
            inl = inliers(q, qn, nb, db)
        if cell is not None and inl.sum() >= 3:
            ii = np.nonzero(inl)[0]
            inl = np.zeros(len(q), dtype=bool)
            inl[ii[largest_component(q[ii, :2], cell)]] = True
            if inl.sum() >= 3:
                nb, db, _ = fit_plane_lsq(q[inl])
        if inl.sum() < min_inliers or nb[2] < cos_max:
            fails += 1
            continue
        rmse = float(np.sqrt(np.mean((q[inl] @ nb - db) ** 2)))
        planes.append(Plane(normal=nb, d=db, inliers=remaining[inl], rmse=rmse))
        remaining = remaining[~inl]
    return planes


def merge_planes(planes: list[Plane], p: np.ndarray, angle_deg: float = 8.0, offset_m: float = 0.3) -> list[Plane]:
    """Merge near-identical planes (same orientation and offset), refitting on the union."""
    out: list[Plane] = []
    for pl in sorted(planes, key=lambda q: -len(q.inliers)):
        for o in out:
            if np.degrees(np.arccos(np.clip(o.normal @ pl.normal, -1, 1))) < angle_deg and abs(o.d - pl.d) < offset_m:
                idx = np.concatenate([o.inliers, pl.inliers])
                n, d, r = fit_plane_lsq(p[idx])
                o.normal, o.d, o.inliers, o.rmse = n, d, idx, r
                break
        else:
            out.append(Plane(pl.normal.copy(), pl.d, pl.inliers.copy(), pl.rmse))
    return sorted(out, key=lambda q: -len(q.inliers))


def absorb_points(planes: list[Plane], p: np.ndarray, threshold: float, radius: float = 1.5) -> list[Plane]:
    """Give each point left unassigned by RANSAC to the closest plane (within `threshold`)
    among the planes of its labelled neighbours (within `radius` m in plan view)."""
    from scipy.spatial import cKDTree

    if not planes:
        return planes
    lab = np.full(len(p), -1, dtype=np.int64)
    for i, pl in enumerate(planes):
        lab[pl.inliers] = i
    free = np.nonzero(lab < 0)[0]
    if not len(free) or len(free) == len(p):
        return planes
    done = np.nonzero(lab >= 0)[0]
    tree = cKDTree(p[done, :2])
    new = lab.copy()
    for j, nb in zip(free, tree.query_ball_point(p[free, :2], radius), strict=True):
        cand = np.unique(lab[done[nb]])
        if not len(cand):
            continue
        dist = np.array([abs(float(p[j] @ planes[c].normal) - planes[c].d) for c in cand])
        k = int(np.argmin(dist))
        if dist[k] < threshold:
            new[j] = cand[k]
    return [Plane(pl.normal, pl.d, np.nonzero(new == i)[0], pl.rmse) for i, pl in enumerate(planes)]


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two compass angles (deg, 0..180)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def aspect_clusters(planes: list[Plane], tol_deg: float = 35.0) -> list[tuple[float, float]]:
    """Group sloped planes by facing direction -> [(mean aspect deg, weight = inlier count)]."""
    cl: list[list[float]] = []  # [sum_sin*w, sum_cos*w, w]
    for pl in sorted(planes, key=lambda q: -len(q.inliers)):
        a, w = pl.aspect_deg, float(len(pl.inliers))
        for c in cl:
            ca = math.degrees(math.atan2(c[0], c[1])) % 360.0
            if angle_diff(ca, a) <= tol_deg:
                c[0] += w * math.sin(math.radians(a))
                c[1] += w * math.cos(math.radians(a))
                c[2] += w
                break
        else:
            cl.append([w * math.sin(math.radians(a)), w * math.cos(math.radians(a)), w])
    return sorted(((math.degrees(math.atan2(s, c)) % 360.0, w) for s, c, w in cl), key=lambda t: -t[1])


def ridge_azimuth(n1: np.ndarray, n2: np.ndarray) -> float:
    """Azimuth (deg in [0, 180), clockwise from north) of the intersection line of two planes."""
    line = np.cross(n1, n2)
    return float(np.degrees(np.arctan2(line[0], line[1])) % 180.0)


def outline_samples(ring_xy: np.ndarray, step: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points every ~`step` m along a closed ring -> (xy (M,2), outward unit normals (M,2),
    sample lengths (M,)). Works for either ring orientation."""
    ring = np.asarray(ring_xy, dtype=np.float64)[:, :2]
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[:1]])
    seg = np.diff(ring, axis=0)
    lens = np.hypot(seg[:, 0], seg[:, 1])
    area2 = float(np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1]))
    sign = 1.0 if area2 > 0 else -1.0  # CCW: outward normal of edge (dx, dy) is (dy, -dx)
    xy, nrm, w = [], [], []
    for (x0, y0), (dx, dy), length in zip(ring[:-1], seg, lens, strict=True):
        if length < 1e-6:
            continue
        k = max(int(round(length / step)), 1)
        t = (np.arange(k) + 0.5) / k
        xy.append(np.column_stack([x0 + t * dx, y0 + t * dy]))
        nrm.append(np.tile(sign * np.array([dy, -dx]) / length, (k, 1)))
        w.append(np.full(k, length / k))
    if not xy:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    return np.vstack(xy), np.vstack(nrm), np.concatenate(w)


def eave_perimeter_fraction(
    footprint_xy: np.ndarray, pts: np.ndarray, planes: list[Plane], flat_max_slope: float,
    radius: float = 1.5, max_eave_angle_deg: float = 50.0,
) -> float:
    """Share of the outline (next to sloped roof planes) that is an eave: the adjacent plane
    drains outward across that edge (plane aspect within `max_eave_angle_deg` of the edge's
    outward normal). Rakes (gable ends) and high sides have the plane aspect across or against
    the edge. Hip roofs ~0.9-1.0; gables ~L/(L+W) ~0.6-0.7. NaN if undetermined."""
    from scipy.spatial import cKDTree

    if len(pts) < 10 or footprint_xy is None or len(footprint_xy) < 3 or not planes:
        return float("nan")
    lab = np.full(len(pts), -1, dtype=np.int64)
    for i, pl in enumerate(planes):
        lab[pl.inliers] = i
    sloped = np.array([pl.slope_deg >= flat_max_slope for pl in planes])
    asp = np.array([pl.normal[:2] / max(float(np.hypot(*pl.normal[:2])), 1e-9) for pl in planes])
    s, o, w = outline_samples(footprint_xy)
    if not len(s):
        return float("nan")
    tree = cKDTree(pts[:, :2])
    cos_e = math.cos(math.radians(max_eave_angle_deg))
    eave = tot = 0.0
    for j, ix in enumerate(tree.query_ball_point(s, radius)):
        ll = lab[ix]
        ll = ll[ll >= 0]
        if len(ll) < 2:
            continue
        top = int(np.bincount(ll).argmax())
        if not sloped[top]:
            continue
        tot += w[j]
        if float(asp[top] @ o[j]) >= cos_e:
            eave += w[j]
    return eave / tot if tot > 0 else float("nan")


@dataclass
class RoofModel:
    roof_type: str
    eave_height_m: float
    ridge_height_m: float
    height_p50_m: float
    height_max_m: float
    roof_pitch_deg: float
    ridge_azimuth_deg: float
    n_planes: int
    inlier_frac: float
    plane_rmse_m: float
    eave_perimeter_frac: float
    n_points: int
    planes: list[dict[str, Any]] = field(default_factory=list)


ROOF_TYPES = ("flat", "gable", "hip", "complex", "shed", "unknown")


def classify_roof(
    planes: list[Plane], n_points: int, eave_frac: float, flat_max_slope: float, hip_min_eave_frac: float = 0.75
) -> tuple[str, float, float]:
    """(roof_type, pitch_deg, ridge_azimuth_deg or NaN) from fitted planes.

    flat: low-slope planes hold >= 60 % of plane points. shed: one facing direction.
    gable: two opposite facing directions hold >= 85 % of the sloped area. hip: >= 3 facing
    directions and eaves on (nearly) the whole outline (no gable ends). complex: anything else
    (cross gables, hip + gable mixes, multi-level, non-orthogonal).
    """
    if not planes:
        return "unknown", float("nan"), float("nan")
    total = float(sum(len(p.inliers) for p in planes))
    major = [p for p in planes if len(p.inliers) >= max(0.06 * total, 6)]
    flat = [p for p in major if p.slope_deg < flat_max_slope]
    sloped = [p for p in major if p.slope_deg >= flat_max_slope]
    flat_share = sum(len(p.inliers) for p in flat) / total
    sw = float(sum(len(p.inliers) for p in sloped))
    pitch = float(np.average([p.slope_deg for p in sloped], weights=[len(p.inliers) for p in sloped])) if sloped else (
        float(np.average([p.slope_deg for p in flat], weights=[len(p.inliers) for p in flat])) if flat else float("nan"))
    # ridge from the strongest pair of opposite-facing planes
    az = float("nan")
    pair_share = 0.0
    for i, a in enumerate(sloped):
        for b in sloped[i + 1 :]:
            if angle_diff(a.aspect_deg, b.aspect_deg) >= 145.0:
                share = (len(a.inliers) + len(b.inliers)) / max(sw, 1.0)
                if share > pair_share:
                    pair_share, az = share, ridge_azimuth(a.normal, b.normal)
    if flat_share >= 0.6 or not sloped:
        return "flat", pitch, float("nan")
    dirs = [c for c in aspect_clusters(sloped) if c[1] >= 0.07 * sw]
    if math.isnan(az):
        az = float((sloped[0].aspect_deg + 90.0) % 180.0)
    if len(dirs) <= 1:
        return "shed", pitch, az
    two_opposite = angle_diff(dirs[0][0], dirs[1][0]) >= 145.0
    top2 = (dirs[0][1] + dirs[1][1]) / sw
    if len(dirs) >= 3 and not math.isnan(eave_frac) and eave_frac >= hip_min_eave_frac:
        return "hip", pitch, az
    if two_opposite and (len(dirs) == 2 or top2 >= 0.85):
        return "gable", pitch, az
    return "complex", pitch, az


def analyze_roof(
    pts: np.ndarray,
    footprint_xy: np.ndarray | None,
    *,
    threshold: float = 0.18,
    flat_max_slope: float = 8.0,
    min_points: int = 12,
    hip_min_eave_frac: float = 0.75,
    normal_tol_deg: float = 15.0,
    rng: np.random.Generator | None = None,
) -> RoofModel:
    """Roof model from building points in a local frame: pts (N,3) = (east m, north m, height
    above ground m), already restricted to the (shrunk) footprint and to roof heights.
    `footprint_xy` is the shrunk outline in the same local frame (for the eave test).

    Plane dicts use the same frame: normal = (e, n, up) unit vector, `d` with normal . p = d,
    so the roof height at (e, n) is (d - ne*e - nn*n) / nup.
    """
    n = len(pts)
    if n < min_points:
        hs = pts[:, 2] if n else np.array([np.nan])
        return RoofModel("unknown", float(np.nanpercentile(hs, 5)) if n else float("nan"),
                         float(np.nanpercentile(hs, 99)) if n else float("nan"),
                         float(np.nanmedian(hs)) if n else float("nan"), float(np.nanmax(hs)) if n else float("nan"),
                         float("nan"), float("nan"), 0, 0.0, float("nan"), float("nan"), n)
    h = pts[:, 2]
    eave_h = float(np.percentile(h, 5))
    ridge_h = float(np.percentile(h, 99))
    min_inl = max(6, int(0.05 * n))
    area = None
    if footprint_xy is not None and len(footprint_xy) >= 3:
        from shapely.geometry import Polygon

        area = float(Polygon(footprint_xy).area)
    density = n / area if area else 4.0
    cell = max(1.0, 1.8 / math.sqrt(max(density, 0.5)))
    normals = point_normals(pts, k=10)
    planes = merge_planes(ransac_planes(pts, threshold, min_inl, rng=rng, normals=normals, normal_tol_deg=normal_tol_deg, cell=cell), pts)
    planes = absorb_points(planes, pts, threshold)
    inl = int(sum(len(p.inliers) for p in planes))
    rmse = float(np.sqrt(sum(p.rmse**2 * len(p.inliers) for p in planes) / inl)) if inl else float("nan")
    ef = eave_perimeter_fraction(footprint_xy, pts, planes, flat_max_slope) if footprint_xy is not None else float("nan")
    rtype, pitch, az = classify_roof(planes, n, ef, flat_max_slope, hip_min_eave_frac)
    if rtype == "flat":
        eave_h = ridge_h = float(np.percentile(h, 90))
    area_per_pt = area / n if area else None
    plist = [
        {
            "normal": [round(float(v), 5) for v in p.normal],
            "d": round(p.d, 4),
            "slope_deg": round(p.slope_deg, 2),
            "aspect_deg": round(p.aspect_deg, 1),
            "n_points": int(len(p.inliers)),
            "area_m2": round(len(p.inliers) * area_per_pt, 1) if area_per_pt else None,
            "rmse_m": round(p.rmse, 3),
        }
        for p in planes
    ]
    return RoofModel(rtype, eave_h, ridge_h, float(np.median(h)), float(h.max()), pitch, az, len(planes),
                     inl / n, rmse, ef, n, plist)




# ---------------------------------------------------------------------------
# Trees (pure)
# ---------------------------------------------------------------------------


def crown_window_radius_m(h: np.ndarray | float) -> np.ndarray:
    """Local-maximum search radius from tree height: half the Popescu & Wynne (2004)
    crown width model CW = 3.09632 + 0.00895 h^2 (m), clipped to [1, 5] m."""
    return np.clip(0.5 * (3.09632 + 0.00895 * np.asarray(h, dtype=np.float64) ** 2), 1.0, 5.0)


def tree_tops(s: np.ndarray, cand: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Variable-window local maxima: 3x3 peaks of the smoothed CHM `s` (plateaus merged to
    their centroid) that have no higher peak within crown_window_radius_m(height)."""
    from scipy.spatial import cKDTree

    peak = cand & (s >= ndi.maximum_filter(s, size=3, mode="nearest") - 1e-4)
    lab, n = ndi.label(peak, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    cen = np.array(ndi.center_of_mass(peak, lab, np.arange(1, n + 1)))
    r = np.clip(np.round(cen[:, 0]).astype(int), 0, s.shape[0] - 1)
    c = np.clip(np.round(cen[:, 1]).astype(int), 0, s.shape[1] - 1)
    h = np.asarray(ndi.maximum(s, lab, np.arange(1, n + 1)), dtype=np.float64)
    win = crown_window_radius_m(h) / res
    tree = cKDTree(np.column_stack([r, c]).astype(np.float64))
    keep = np.ones(n, dtype=bool)
    order = np.argsort(-h, kind="stable")  # tallest first; ties: earlier label wins
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    for i, nb in enumerate(tree.query_ball_point(np.column_stack([r, c]).astype(np.float64), win)):
        if any(rank[j] < rank[i] for j in nb if j != i):
            keep[i] = False
    return r[keep], c[keep]


@dataclass
class TreeSet:
    rows: np.ndarray
    cols: np.ndarray
    height: np.ndarray
    crown_radius: np.ndarray
    crown_mean_ratio: np.ndarray  # mean crown CHM / top height (cone ~0.5, dome ~0.8)
    labels: np.ndarray  # crown label raster (0 = none)


def detect_trees(chm: np.ndarray, res: float, min_h: float, veg_mask: np.ndarray | None = None) -> TreeSet:
    """Tree tops = local maxima of the smoothed CHM within a height-dependent circular window;
    crowns = marker watershed of the inverted CHM restricted to canopy cells."""
    chm = np.nan_to_num(chm.astype(np.float32), nan=0.0)
    s = ndi.gaussian_filter(chm, sigma=0.6 / res)
    canopy = (chm >= VEG_MIN_H_M) if veg_mask is None else (veg_mask & (chm >= VEG_MIN_H_M))
    tr, tc = tree_tops(s, canopy & (s >= min_h), res)
    n = len(tr)
    if n == 0:
        z = np.zeros(0)
        return TreeSet(z.astype(int), z.astype(int), z, z, z, np.zeros(chm.shape, dtype=np.int32))
    markers = np.zeros(chm.shape, dtype=np.int32)
    markers[tr, tc] = np.arange(1, n + 1)
    markers[~canopy & (markers == 0)] = -1  # background marker floods non-canopy
    inv = (np.clip(s.max() - s, 0, None) / max(float(s.max()), 1e-6) * 65000).astype(np.uint16)
    inv[~canopy] = 65535  # background floods last (smoothing bleeds s outside the canopy)
    ws = ndi.watershed_ift(inv, markers).astype(np.int32)
    ws[~canopy] = 0
    ws[ws < 0] = 0
    # crown cells must stay above 40 % of their own top height (trims flooding into low canopy)
    tops_h = np.concatenate([[0.0], s[tr, tc]])
    ws[(ws > 0) & (chm < 0.4 * tops_h[ws])] = 0
    idx = np.arange(1, n + 1)
    area = ndi.sum_labels(np.ones_like(chm), ws, idx) * res * res
    mean_h = ndi.mean(chm, ws, idx)
    h = np.maximum(chm[tr, tc], np.nan_to_num(np.asarray(ndi.maximum(chm, ws, idx), dtype=np.float64)))
    radius = np.sqrt(np.maximum(area, res * res) / math.pi)
    ratio = np.where(h > 0, np.nan_to_num(mean_h) / np.maximum(h, 1e-6), np.nan)
    return TreeSet(tr, tc, h.astype(np.float64), radius, ratio, ws)


def species_guess(
    height: np.ndarray, radius: np.ndarray, mean_ratio: np.ndarray, *, palm_min_h: float = 7.0,
    palm_max_r: float = 2.4, palm_max_rh: float = 0.2, conifer_max_fill: float = 0.62, conifer_min_h: float = 6.0,
    canopy_cover: np.ndarray | None = None, palm_max_cover: float = 0.35,
) -> np.ndarray:
    """CRUDE GUESS from crown geometry only (no spectral data): 'palm' = tall with a small
    crown (fan / queen palms), 'conifer' = conical crown (mean/top < conifer_max_fill, narrow),
    'broadleaf' = everything else (oaks, sycamores, eucalyptus, pepper trees). With
    `canopy_cover` (share of canopy within 20 m), palms must also stand fairly isolated
    (cover < palm_max_cover): tall narrow crowns inside continuous groves are eucalyptus /
    riparian trees split by the watershed. Defaults mirror lidar.* in assumptions.yaml."""
    h, r, q = (np.asarray(a, dtype=np.float64) for a in (height, radius, mean_ratio))
    out = np.full(h.shape, "broadleaf", dtype=object)
    conifer = (q < conifer_max_fill) & (r / np.maximum(h, 1e-6) < 0.35) & (h >= conifer_min_h)
    palm = (h >= palm_min_h) & (r <= palm_max_r) & (r / np.maximum(h, 1e-6) <= palm_max_rh)
    if canopy_cover is not None:
        palm &= np.nan_to_num(np.asarray(canopy_cover, dtype=np.float64), nan=0.0) < palm_max_cover
    out[conifer] = "conifer"
    out[palm] = "palm"
    return out


def canopy_cover_at(chm: np.ndarray, core: np.ndarray, res: float, e: np.ndarray, n: np.ndarray, window_m: float = 20.0) -> np.ndarray:
    """Share of canopy cells (CHM >= VEG_MIN_H_M) in a window_m square around each point, from
    one tile's CHM window (core = [minx, miny, maxx, maxy])."""
    k = max(int(round(window_m / res)) | 1, 3)
    cov = ndi.uniform_filter((np.nan_to_num(chm) >= VEG_MIN_H_M).astype(np.float32), size=k, mode="nearest")
    r = np.clip(((core[3] - np.asarray(n)) / res).astype(int), 0, chm.shape[0] - 1)
    c = np.clip(((np.asarray(e) - core[0]) / res).astype(int), 0, chm.shape[1] - 1)
    return cov[r, c].astype(np.float64)


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


def utm_to_scene_arrays(e: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o = scene_origin()
    return np.asarray(e) - o.easting, -(np.asarray(n) - o.northing)


def utm_to_lonlat_arrays(e: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    t = Transformer.from_crs(PROJECTION, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(np.asarray(e, dtype=np.float64), np.asarray(n, dtype=np.float64))
    return np.asarray(lon), np.asarray(lat)


def in_bbox(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Mask of points inside the region bbox (region.yaml, WGS84)."""
    from pipeline.config import region

    b = region()["bbox"]
    lon, lat = np.asarray(lon), np.asarray(lat)
    return (lon >= b["west"]) & (lon <= b["east"]) & (lat >= b["south"]) & (lat <= b["north"])


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def work_dir() -> Path:
    return lidar_dir() / "work"


def raster_extent() -> Box:
    b = utm_bbox(float(assumption("lidar.fetch_buffer_m")))
    return Box(math.floor(b.minx / 10) * 10, math.floor(b.miny / 10) * 10, math.ceil(b.maxx / 10) * 10, math.ceil(b.maxy / 10) * 10)


def load_points(box: Box) -> dict[str, np.ndarray]:
    """Points of every 1 km tile intersecting `box`, cropped to it (EPSG:32611)."""
    import laspy

    parts: dict[str, list[np.ndarray]] = {}
    for ckm, rkm in tile_keys_for(box):
        f = lidar_dir() / "tiles" / tile_name(ckm, rkm)
        if not f.exists():
            continue
        las = laspy.read(str(f))
        x, y = np.asarray(las.x), np.asarray(las.y)
        m = (x >= box.minx) & (x < box.maxx) & (y >= box.miny) & (y < box.maxy)
        cls = np.asarray(las.classification)
        m &= ~np.isin(cls, NOISE_CLASSES)
        for k, v in (("x", x), ("y", y), ("z", np.asarray(las.z)), ("cls", cls),
                     ("ret", np.asarray(las.return_number)), ("nret", np.asarray(las.number_of_returns))):
            parts.setdefault(k, []).append(v[m])
    if not parts:
        return {k: np.zeros(0) for k in ("x", "y", "z", "cls", "ret", "nret")}
    return {k: np.concatenate(v) for k, v in parts.items()}


def load_footprints(clip: bool = True) -> Any:
    """osm_buildings.geojson footprints (EPSG:32611); with `clip`, only those whose
    representative point is inside the region bbox."""
    import geopandas as gpd

    from pipeline.config import region

    g = gpd.read_file(raw_dir() / "osm_buildings.geojson")
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g["geometry"] = g.geometry.make_valid()
    b = region()["bbox"]
    c = g.geometry.representative_point()
    if clip:
        g = g[(c.x >= b["west"]) & (c.x <= b["east"]) & (c.y >= b["south"]) & (c.y <= b["north"])].copy()
    g = g.to_crs(PROJECTION)
    # keep polygonal parts only
    g["geometry"] = g.geometry.apply(_polygonal)
    g = g[~g.geometry.is_empty].reset_index(drop=True)
    return g


def _polygonal(geom: Any) -> Any:
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(geom, Polygon | MultiPolygon):
        return geom
    polys = [p for p in getattr(geom, "geoms", []) if isinstance(p, Polygon | MultiPolygon)]
    if not polys:
        return Polygon()
    from shapely.ops import unary_union

    return unary_union(polys)


def load_road_lines() -> list[Any]:
    """Drive-graph edge polylines (EPSG:32611) for masking bridges/overpasses from the
    missing-building search."""
    import networkx as nx
    from pyproj import Transformer
    from shapely import wkt
    from shapely.geometry import LineString

    f = raw_dir() / "osm_drive_raw.graphml"
    if not f.exists():
        return []
    g = nx.read_graphml(f)
    t = Transformer.from_crs("EPSG:4326", PROJECTION, always_xy=True)
    lines = []
    for u, v, data in g.edges(data=True):
        hw = str(data.get("highway", ""))
        if "geometry" in data:
            ls = wkt.loads(data["geometry"])
            xy = np.asarray(ls.coords)
        else:
            nu, nv = g.nodes[u], g.nodes[v]
            xy = np.array([[float(nu["x"]), float(nu["y"])], [float(nv["x"]), float(nv["y"])]])
        ex, ny = t.transform(xy[:, 0], xy[:, 1])
        lines.append((hw, LineString(np.column_stack([ex, ny]))))
    return lines


# ---------------------------------------------------------------------------
# Per-tile processing
# ---------------------------------------------------------------------------


@dataclass
class TileJob:
    ckm: int
    rkm: int
    core: Box
    footprints: list[tuple[int, Any]]  # (row index in footprint table, shapely polygon UTM)
    other_fp: list[Any]  # footprints near the tile (for masks)
    road_lines: list[Any]
    shift: tuple[float, float]  # (dx, dy) applied to lidar x/y to register onto footprints
    force: bool


def _roof_record(model: RoofModel) -> dict[str, Any]:
    return {
        "roof_type": model.roof_type,
        "eave_height_m": model.eave_height_m,
        "ridge_height_m": model.ridge_height_m,
        "height_p50_m": model.height_p50_m,
        "height_max_m": model.height_max_m,
        "roof_pitch_deg": model.roof_pitch_deg,
        "ridge_azimuth_deg": model.ridge_azimuth_deg,
        "n_planes": model.n_planes,
        "inlier_frac": model.inlier_frac,
        "plane_rmse_m": model.plane_rmse_m,
        "eave_perimeter_frac": model.eave_perimeter_frac,
        "n_roof_points": model.n_points,
        "planes_json": json.dumps(model.planes),
    }


def _local_xy(geom: Any, cx: float, cy: float) -> np.ndarray | None:
    from shapely.geometry import MultiPolygon

    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    return np.asarray(geom.exterior.coords) - np.array([cx, cy])


def building_roof(
    poly: Any, px: np.ndarray, py: np.ndarray, ph: np.ndarray, shrink: float, min_h: float, cfg: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Roof record for one footprint polygon (UTM) from candidate points (UTM x, y, height)."""
    import shapely

    shrunk = poly.buffer(-shrink)
    if shrunk.is_empty or shrunk.area < 4.0:
        shrunk = poly.buffer(-min(shrink, 0.15))
    if shrunk.is_empty:
        shrunk = poly
    inside = shapely.contains_xy(shrunk, px, py)
    n_in = int(inside.sum())
    h = ph[inside]
    roof = h >= min_h
    cx, cy = poly.centroid.x, poly.centroid.y
    pts = np.column_stack([px[inside][roof] - cx, py[inside][roof] - cy, h[roof]])
    model = analyze_roof(pts, _local_xy(shrunk, cx, cy), threshold=cfg["thr"], flat_max_slope=cfg["flat"],
                         hip_min_eave_frac=cfg.get("hip", 0.75), rng=rng)
    rec = _roof_record(model)
    rec["n_points_footprint"] = n_in
    rec["roof_cover_frac"] = float(roof.mean()) if n_in else float("nan")
    rec["point_density_m2"] = n_in / max(shrunk.area, 1e-6)
    return rec


def _mask_from_geoms(grid: Grid, geoms: list[Any], buffer_m: float = 0.0) -> np.ndarray:
    from rasterio.features import rasterize

    shapes = [(g.buffer(buffer_m) if buffer_m else g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(grid.shape, dtype=bool)
    return rasterize(shapes, out_shape=grid.shape, transform=grid.transform(), fill=0, dtype="uint8").astype(bool)


def surface_features(grid: Grid, P: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """(dtm, dsm, ndsm, point heights above dtm, penetration, plane residual) on `grid`.

    penetration = share of elevated (> 1 m) returns that belong to multi-return pulses,
    averaged over ~1.5 m; plane residual = 3x3 (1.5 m) plane-fit residual std of the DSM,
    median-filtered."""
    g = np.isin(P["cls"], GROUND_CLASSES)
    dtm = interpolate_ground(grid, P["x"][g], P["y"][g], P["z"][g])
    dsm = np.maximum(fill_nan(rasterize_stat(grid, P["x"], P["y"], P["z"], "max")), dtm)
    ndsm = np.clip(dsm - dtm, 0.0, None)
    ph = P["z"] - sample_bilinear(grid, dtm, P["x"], P["y"])
    elev = (ph > 1.0) & ~g
    multi = elev & (P["nret"] > 1)
    n_el = rasterize_stat(grid, P["x"][elev], P["y"][elev], np.ones(int(elev.sum())), "count").astype(np.float64)
    n_mu = rasterize_stat(grid, P["x"][multi], P["y"][multi], np.ones(int(multi.sum())), "count").astype(np.float64)
    k = max(int(round(1.5 / grid.res)), 1) * 2 + 1
    den = ndi.uniform_filter(n_el, k)
    pen = np.where(den > 1e-9, ndi.uniform_filter(n_mu, k) / np.maximum(den, 1e-9), 0.0).astype(np.float32)
    resid = ndi.median_filter(plane_residual_std(dsm, 3), 3)
    return dtm, dsm, ndsm, ph, pen, resid


def process_tile(job: TileJob) -> dict[str, Any]:
    """Compute rasters, lidar-only building candidates and trees for one tile (roofs are a
    separate pass, process_roofs, that reads the mosaicked DTM)."""
    import pandas as pd
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape

    out_npz = work_dir() / f"tile_{job.ckm}_{job.rkm}.npz"
    out_t = work_dir() / f"trees_{job.ckm}_{job.rkm}.parquet"
    out_m = work_dir() / f"missing_{job.ckm}_{job.rkm}.parquet"
    if not job.force and all(p.exists() for p in (out_npz, out_t, out_m)):
        return {"tile": (job.ckm, job.rkm), "cached": True}
    t0 = time.time()
    res = float(assumption("lidar.raster_res_m"))
    min_h = float(assumption("lidar.roof_min_height_m"))
    cfg = {"thr": float(assumption("lidar.ransac_threshold_m")), "flat": float(assumption("lidar.flat_roof_max_slope_deg"))}
    mb_h = float(assumption("lidar.missing_building_min_height_m"))
    mb_a = float(assumption("lidar.missing_building_min_area_m2"))
    tree_h = float(assumption("lidar.tree_min_height_m"))
    chg_thr = float(assumption("lidar.ground_change_m"))
    rng = np.random.default_rng(SEED + job.ckm * 1000 + job.rkm)

    work = job.core.buffered(HALO_M)
    P = load_points(work.buffered(5.0))
    P["x"] = P["x"] + job.shift[0]
    P["y"] = P["y"] + job.shift[1]
    grid = Grid.from_box(work, res)
    dtm, dsm, ndsm, ph, pen, resid_s = surface_features(grid, P)

    # 2024 DEM vs 2014 lidar ground
    dem = _sample_scene_dem(grid)
    change = (dem - dtm).astype(np.float32) if dem is not None else np.zeros(grid.shape, dtype=np.float32)
    changed = np.abs(change) > chg_thr

    fp_mask = _mask_from_geoms(grid, job.other_fp)
    fp_dil = ndi.binary_dilation(fp_mask, structure=disk(1.5 / res))
    planar = (resid_s < PLANE_RESID_MAX_M) & (pen < PEN_MAX)
    bld_lidar = (ndsm > mb_h) & planar
    bld_lidar = ndi.binary_opening(bld_lidar, structure=disk(1.0 / res))
    veg = (ndsm >= VEG_MIN_H_M) & ~(bld_lidar | fp_mask) & ~((resid_s < PLANE_RESID_MAX_M * 0.75) & (pen < 0.1))
    chm = np.where(veg, ndsm, 0.0).astype(np.float32)


    # --- lidar-only buildings ---
    core_r0 = int(round((work.maxy - job.core.maxy) / res))
    core_c0 = int(round((job.core.minx - work.minx) / res))
    core_nr = int(round((job.core.maxy - job.core.miny) / res))
    core_nc = int(round((job.core.maxx - job.core.minx) / res))
    cand = bld_lidar & ~fp_dil & ~changed
    if job.road_lines:
        roads = _mask_from_geoms(grid, [ls for _, ls in job.road_lines], buffer_m=6.0)
        cand &= ~roads
    lab, n = ndi.label(cand)
    mrecs = []
    if n:
        idxs = np.arange(1, n + 1)
        areas = ndi.sum_labels(np.ones(grid.shape), lab, idxs) * res * res
        coms = np.array(ndi.center_of_mass(cand, lab, idxs))
        objs = ndi.find_objects(lab)
        for i in np.nonzero(areas >= mb_a)[0]:
            cr, cc = coms[i]
            if not (core_r0 <= cr < core_r0 + core_nr and core_c0 <= cc < core_c0 + core_nc):
                continue
            sl = objs[i]
            sub = (lab[sl] == i + 1)
            # thin structures (walls, hedges): require a 2.5 m wide core
            if not ndi.binary_erosion(sub, structure=disk(1.25 / res)).any():
                continue
            x0 = grid.x0 + sl[1].start * res
            y1 = grid.y1 - sl[0].start * res
            from rasterio.transform import Affine

            tr = Affine(res, 0, x0, 0, -res, y1)
            geoms = [shape(gj) for gj, v in rio_shapes(sub.astype(np.uint8), mask=sub, transform=tr) if v == 1]
            if not geoms:
                continue
            poly = max(geoms, key=lambda q: q.area).simplify(0.5, preserve_topology=True).buffer(0)
            if poly.is_empty or poly.area < mb_a:
                continue
            bx0, by0, bx1, by1 = poly.bounds
            m = (P["x"] >= bx0) & (P["x"] <= bx1) & (P["y"] >= by0) & (P["y"] <= by1)
            idx = np.nonzero(m)[0]
            rec = building_roof(poly, P["x"][idx], P["y"][idx], ph[idx], 0.25, min_h, cfg, rng)
            win = _poly_window_values(grid, poly, [dtm, ndsm, pen] + ([dem] if dem is not None else []), p90=1)
            rec.update({
                "wkt": poly.wkt, "area_m2": float(poly.area), "ground_elev_lidar_m": float(win[0]),
                "ndsm_p90_m": float(win[1]), "penetration": float(win[2]),
                "ground_elev_m": float(win[3]) if dem is not None else float(win[0]),
            })
            mrecs.append(rec)
    pd.DataFrame(mrecs).to_parquet(out_m, index=False)

    # --- trees ---
    ts = detect_trees(chm, res, tree_h, veg)
    if len(ts.rows):
        xs, ys = grid.centers()
        tx, ty = xs[ts.cols], ys[ts.rows]
        keep = (tx >= job.core.minx) & (tx < job.core.maxx) & (ty >= job.core.miny) & (ty < job.core.maxy)
        keep &= ~fp_dil[ts.rows, ts.cols] & ~changed[ts.rows, ts.cols]
        tdf = pd.DataFrame({
            "easting": tx[keep], "northing": ty[keep], "height_m": ts.height[keep],
            "crown_radius_m": ts.crown_radius[keep], "crown_mean_ratio": ts.crown_mean_ratio[keep],
            "ground_lidar_m": dtm[ts.rows[keep], ts.cols[keep]].astype(np.float64),
            "ground_y": (dem if dem is not None else dtm)[ts.rows[keep], ts.cols[keep]].astype(np.float64),
            "penetration": pen[ts.rows[keep], ts.cols[keep]].astype(np.float64),
        })
        n_dropped = int((~keep).sum())
    else:
        tdf = pd.DataFrame(columns=["easting", "northing", "height_m", "crown_radius_m", "crown_mean_ratio",
                                    "ground_lidar_m", "ground_y", "penetration"])
        n_dropped = 0
    tdf.to_parquet(out_t, index=False)

    # --- rasters for the core window ---
    sl = (slice(core_r0, core_r0 + core_nr), slice(core_c0, core_c0 + core_nc))
    cls_map = np.zeros(grid.shape, dtype=np.uint8)  # 0 ground, 1 building-like, 2 vegetation, 3 other elevated
    cls_map[ndsm > VEG_MIN_H_M] = 3
    cls_map[veg] = 2
    cls_map[bld_lidar] = 1
    tmp_npz = out_npz.with_suffix(".part")
    with open(tmp_npz, "wb") as fh:
        np.savez(fh, dsm=dsm[sl], dtm=dtm[sl], ndsm=ndsm[sl], chm=chm[sl], change=change[sl],
                 cls=cls_map[sl], pen=pen[sl], resid=resid_s[sl], count=rasterize_stat(grid, P["x"], P["y"], P["z"], "count")[sl].astype(np.uint16),
                 core=np.array([job.core.minx, job.core.miny, job.core.maxx, job.core.maxy]))
    tmp_npz.replace(out_npz)
    return {"tile": (job.ckm, job.rkm), "cached": False, "points": len(P["x"]),
            "missing": len(mrecs), "trees": len(tdf), "trees_dropped": n_dropped, "seconds": time.time() - t0}


def _read_mosaic(name: str, grid: Grid) -> np.ndarray:
    """Window of a mosaicked lidar GeoTIFF on `grid` (NaN outside the raster)."""
    import rasterio
    from rasterio.windows import from_bounds

    b = grid.box
    with rasterio.open(lidar_dir() / name) as src:
        w = from_bounds(b.minx, b.miny, b.maxx, b.maxy, transform=src.transform)
        a = src.read(1, window=w, boundless=True, fill_value=np.nan, out_shape=grid.shape)
    return a.astype(np.float32)


def process_roofs(job: TileJob) -> dict[str, Any]:
    """Roof models for the footprints whose centroid is in this tile, from the tile points and
    the mosaicked DTM / nDSM / ground-change rasters (so roofs can be re-run on their own)."""
    import pandas as pd

    out_b = work_dir() / f"roofs_{job.ckm}_{job.rkm}.parquet"
    if not job.force and out_b.exists():
        return {"tile": (job.ckm, job.rkm), "cached": True}
    t0 = time.time()
    res = float(assumption("lidar.raster_res_m"))
    shrink = float(assumption("lidar.footprint_shrink_m"))
    min_h = float(assumption("lidar.roof_min_height_m"))
    cfg = {"thr": float(assumption("lidar.ransac_threshold_m")), "flat": float(assumption("lidar.flat_roof_max_slope_deg")),
           "hip": float(assumption("lidar.hip_min_eave_frac"))}
    rng = np.random.default_rng(SEED + job.ckm * 1000 + job.rkm)
    recs = []
    if job.footprints:
        bnds = np.array([p.bounds for _, p in job.footprints])
        b = Box(float(bnds[:, 0].min()), float(bnds[:, 1].min()), float(bnds[:, 2].max()), float(bnds[:, 3].max()))
        work = Box(math.floor(b.minx) - 2, math.floor(b.miny) - 2, math.ceil(b.maxx) + 2, math.ceil(b.maxy) + 2)
        P = load_points(Box(work.minx - job.shift[0], work.miny - job.shift[1], work.maxx - job.shift[0], work.maxy - job.shift[1]))
        P["x"] = P["x"] + job.shift[0]
        P["y"] = P["y"] + job.shift[1]
        grid = Grid.from_box(work, res)
        dtm = fill_nan(_read_mosaic("dtm_0p5m.tif", grid))
        ndsm = _read_mosaic("ndsm_0p5m.tif", grid)
        change = _read_mosaic("ground_change_0p5m.tif", grid)
        ph = P["z"] - sample_bilinear(grid, dtm, P["x"], P["y"])
        lab_grid = Grid.from_box(work, 1.0)
        r, c, ok = lab_grid.rc(P["x"], P["y"])
        cell = np.where(ok, r * lab_grid.ncols + c, -1)
        order = np.argsort(cell, kind="stable")
        cs = cell[order]
        for row_idx, poly in job.footprints:
            minx, miny, maxx, maxy = poly.bounds
            r0, c0 = max(int((lab_grid.y1 - maxy) // 1), 0), max(int((minx - lab_grid.x0) // 1), 0)
            r1 = min(int((lab_grid.y1 - miny) // 1), lab_grid.nrows - 1)
            c1 = min(int((maxx - lab_grid.x0) // 1), lab_grid.ncols - 1)
            sel = []
            for rr in range(r0, r1 + 1):
                a = np.searchsorted(cs, rr * lab_grid.ncols + c0, "left")
                bb = np.searchsorted(cs, rr * lab_grid.ncols + c1, "right")
                sel.append(order[a:bb])
            idx = np.concatenate(sel) if sel else np.zeros(0, dtype=np.int64)
            rec = building_roof(poly, P["x"][idx], P["y"][idx], ph[idx], shrink, min_h, cfg, rng)
            win = _poly_window_values(grid, poly, [dtm, change, ndsm], p90=2)
            rec["ground_elev_lidar_m"] = float(win[0])
            rec["ground_change_m"] = float(win[1]) if np.isfinite(win[1]) else 0.0
            rec["ndsm_p90_m"] = float(win[2])
            rec["ground_elev_m"] = rec["ground_elev_lidar_m"] + rec["ground_change_m"]  # = 3DEP 2024 DEM
            rec["row"] = row_idx
            recs.append(rec)
    tmp = out_b.with_suffix(".part")
    pd.DataFrame(recs).to_parquet(tmp, index=False)
    tmp.replace(out_b)
    return {"tile": (job.ckm, job.rkm), "cached": False, "buildings": len(recs), "seconds": time.time() - t0}


def _poly_window_values(grid: Grid, poly: Any, arrays: list[np.ndarray], p90: int | None = None) -> list[float]:
    """Median of each array over the cells covered by `poly` (centroid cell if none); the
    array at index `p90` gets its 90th percentile instead."""
    from rasterio.features import geometry_mask

    minx, miny, maxx, maxy = poly.bounds
    r0 = max(int((grid.y1 - maxy) / grid.res) - 1, 0)
    r1 = min(int((grid.y1 - miny) / grid.res) + 2, grid.nrows)
    c0 = max(int((minx - grid.x0) / grid.res) - 1, 0)
    c1 = min(int((maxx - grid.x0) / grid.res) + 2, grid.ncols)
    if r1 <= r0 or c1 <= c0:
        return [float("nan")] * len(arrays)
    from rasterio.transform import Affine

    tr = Affine(grid.res, 0, grid.x0 + c0 * grid.res, 0, -grid.res, grid.y1 - r0 * grid.res)
    m = geometry_mask([poly], out_shape=(r1 - r0, c1 - c0), transform=tr, invert=True)
    if not m.any():
        rr, cc, ok = grid.rc(np.array([poly.centroid.x]), np.array([poly.centroid.y]))
        if not ok[0]:
            return [float("nan")] * len(arrays)
        return [float(a[rr[0], cc[0]]) for a in arrays]
    out = []
    for a in arrays:
        v = a[r0:r1, c0:c1][m]
        v = v[np.isfinite(v)]
        out.append(float(np.median(v)) if len(v) else float("nan"))
    if p90 is not None:
        v = arrays[p90][r0:r1, c0:c1][m]
        v = v[np.isfinite(v)]
        out[p90] = float(np.percentile(v, 90)) if len(v) else float("nan")
    return out


_DEM_CACHE: dict[str, Any] = {}


def _sample_scene_dem(grid: Grid) -> np.ndarray | None:
    """Scene terrain DEM (data/raw/dem_3dep_10m.tif, 3DEP 2024 1 m read at 2 m) bilinearly
    resampled onto `grid`."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    f = raw_dir() / "dem_3dep_10m.tif"
    if not f.exists():
        return None
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    with rasterio.open(f) as src:
        reproject(rasterio.band(src, 1), out, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=grid.transform(), dst_crs=PROJECTION, resampling=Resampling.bilinear,
                  dst_nodata=np.nan)
    return out


# ---------------------------------------------------------------------------
# Registration (lidar vs footprints)
# ---------------------------------------------------------------------------


def estimate_shift(ndsm: np.ndarray, fp_mask: np.ndarray, res: float, max_shift_m: float = 3.0) -> tuple[float, float, float, float]:
    """(dx, dy, iou_best, iou_zero): integer-pixel shift of the lidar building mask
    (ndsm > 2.5) that maximizes IoU with the footprint mask. dx east, dy north (meters)."""
    b = ndsm > 2.5
    k = int(round(max_shift_m / res))
    best = (0.0, 0.0, -1.0)
    iou0 = 0.0
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            sh = np.roll(np.roll(b, dr, axis=0), dc, axis=1)
            sl = (slice(k, -k), slice(k, -k))
            inter = np.logical_and(sh[sl], fp_mask[sl]).sum()
            uni = np.logical_or(sh[sl], fp_mask[sl]).sum()
            iou = inter / max(uni, 1)
            if dr == 0 and dc == 0:
                iou0 = iou
            if iou > best[2]:
                best = (dc * res, -dr * res, iou)
    return best[0], best[1], float(best[2]), float(iou0)


def registration(footprints: Any) -> dict[str, Any]:
    """Estimate the lidar -> footprint horizontal offset on a few dense residential tiles."""
    res = 0.5
    results = []
    ext = raster_extent()
    cx, cy = (ext.minx + ext.maxx) / 2, (ext.miny + ext.maxy) / 2
    from shapely.geometry import box as sbox

    sindex = footprints.sindex
    cands = []
    for dx in (-2500, -1200, 0, 1200, 2500):
        for dy in (-1500, 0, 1500):
            b = Box(cx + dx - 250, cy + dy - 250, cx + dx + 250, cy + dy + 250)
            hits = sindex.query(sbox(b.minx, b.miny, b.maxx, b.maxy))
            cands.append((len(hits), b, hits))
    cands.sort(key=lambda t: -t[0])
    for _, b, hits in cands[:5]:
        P = load_points(b.buffered(5))
        grid = Grid.from_box(b, res)
        _, _, ndsm, _, pen, resid = surface_features(grid, P)
        fp = _mask_from_geoms(grid, list(footprints.geometry.iloc[hits]))
        planar = (resid < PLANE_RESID_MAX_M) & (pen < PEN_MAX)
        dx, dy, iou, iou0 = estimate_shift(np.where(planar, ndsm, 0.0), fp, res)
        results.append({"box": b.__dict__, "dx": dx, "dy": dy, "iou": iou, "iou_zero": iou0, "n_footprints": int(len(hits))})
    dxs = np.array([r["dx"] for r in results])
    dys = np.array([r["dy"] for r in results])
    return {"samples": results, "dx_m": float(np.median(dxs)), "dy_m": float(np.median(dys))}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def mosaic(tiles: list[Path], ext: Box, res: float, outputs: dict[str, str]) -> None:
    """Write tile npz windows into GeoTIFFs (tiled, deflate)."""
    import rasterio

    grid = Grid.from_box(ext, res)
    profiles = {}
    dsts = {}
    try:
        for key, name in outputs.items():
            dtype = "uint8" if key == "cls" else "float32"
            prof = {"driver": "GTiff", "width": grid.ncols, "height": grid.nrows, "count": 1, "dtype": dtype,
                    "crs": PROJECTION, "transform": grid.transform(), "tiled": True, "blockxsize": 512,
                    "blockysize": 512, "compress": "deflate", "predictor": 2 if dtype == "uint8" else 3,
                    "nodata": None if key == "cls" else np.nan, "BIGTIFF": "IF_SAFER"}
            profiles[key] = prof
            dsts[key] = rasterio.open(lidar_dir() / name, "w", **prof)
        for t in tiles:
            z = np.load(t)
            core = z["core"]
            c0 = int(round((core[0] - grid.x0) / res))
            r0 = int(round((grid.y1 - core[3]) / res))
            for key, dst in dsts.items():
                a = z[key]
                if key != "cls":
                    a = np.round(a.astype(np.float32), 2)
                from rasterio.windows import Window

                dst.write(a, 1, window=Window(c0, r0, a.shape[1], a.shape[0]))
    finally:
        for d in dsts.values():
            d.close()


def build_jobs(fp: Any, roads: list[Any], shift: tuple[float, float], force: bool, fp_all: Any = None) -> list[TileJob]:
    """One job per 1 km tile: roofs for `fp` (bbox footprints) whose centroid is in the tile;
    masks from `fp_all` (every footprint, also outside the bbox) near the tile."""
    from shapely.geometry import box as sbox

    ext = raster_extent()
    cent = fp.geometry.representative_point()
    cx, cy = cent.x.to_numpy(), cent.y.to_numpy()
    fp_all = fp if fp_all is None else fp_all
    sindex = fp_all.sindex
    road_geoms = [ls for _, ls in roads]
    from shapely import STRtree

    rtree = STRtree(road_geoms) if road_geoms else None
    jobs = []
    for ckm, rkm in tile_keys_for(ext):
        tb = Box(ckm * TILE_M, rkm * TILE_M, (ckm + 1) * TILE_M, (rkm + 1) * TILE_M)
        core = Box(max(tb.minx, ext.minx), max(tb.miny, ext.miny), min(tb.maxx, ext.maxx), min(tb.maxy, ext.maxy))
        if core.maxx <= core.minx or core.maxy <= core.miny:
            continue
        sel = np.nonzero((cx >= core.minx) & (cx < core.maxx) & (cy >= core.miny) & (cy < core.maxy))[0]
        w = core.buffered(HALO_M)
        near = sindex.query(sbox(w.minx, w.miny, w.maxx, w.maxy))
        rl = []
        if rtree is not None:
            for i in rtree.query(sbox(w.minx, w.miny, w.maxx, w.maxy)):
                rl.append(roads[int(i)])
        jobs.append(TileJob(ckm, rkm, core, [(int(i), fp.geometry.iloc[int(i)]) for i in sel],
                            list(fp_all.geometry.iloc[near]), rl, shift, force))
    return jobs


def _pool(workers: int) -> ProcessPoolExecutor:
    """Process pool with the 'spawn' start method: forking after GDAL / BLAS threads have run
    in the parent (registration) deadlocks the workers."""
    import multiprocessing as mp
    import os

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")  # one BLAS thread per worker (inherited by spawned workers)
    return ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))


def run_parallel(fn: Any, jobs: list[TileJob], workers: int, what: str) -> list[dict[str, Any]]:
    stats = []
    if workers <= 1:
        results: Any = map(fn, jobs)
        for i, r in enumerate(results, 1):
            stats.append(r)
            _log_tile(i, len(jobs), what, r)
        return stats
    from concurrent.futures.process import BrokenProcessPool

    try:
        with _pool(workers) as ex:
            for i, r in enumerate(ex.map(fn, jobs), 1):
                stats.append(r)
                _log_tile(i, len(jobs), what, r)
    except BrokenProcessPool:
        # a worker died (usually the OOM killer on a shared machine): finished tiles are cached,
        # so resume with one worker fewer
        log(f"WARNING {what}: a worker died; resuming with {max(workers - 1, 1)} workers")
        return run_parallel(fn, jobs, max(workers - 1, 1), what)
    return stats


def _log_tile(i: int, n: int, what: str, r: dict[str, Any]) -> None:
    if r.get("cached"):
        return
    extra = ", ".join(f"{k} {v:,}" if isinstance(v, int) else f"{k} {v:.0f}" for k, v in r.items()
                      if k not in ("tile", "cached") and isinstance(v, int | float))
    log(f"  {what} {i}/{n} {r['tile']}: {extra}")


def write_missing(jobs: list[TileJob]) -> Any:
    import geopandas as gpd
    import pandas as pd
    from shapely import wkt as swkt

    miss = pd.concat([pd.read_parquet(work_dir() / f"missing_{j.ckm}_{j.rkm}.parquet") for j in jobs], ignore_index=True)
    if len(miss):
        geoms = [swkt.loads(w) for w in miss["wkt"]]
        mg = gpd.GeoDataFrame(miss.drop(columns=["wkt"]), geometry=geoms, crs=PROJECTION)
        mg = mg[mg["roof_type"] != "unknown"].reset_index(drop=True)
        c = mg.geometry.centroid
        mg["centroid_x"], mg["centroid_z"] = utm_to_scene_arrays(c.x.to_numpy(), c.y.to_numpy())
        lon, lat = utm_to_lonlat_arrays(c.x.to_numpy(), c.y.to_numpy())
        mg["centroid_lat"], mg["centroid_lon"] = lat, lon
        mg = mg[in_bbox(lon, lat)].reset_index(drop=True)
        mg.insert(0, "lidar_id", [f"lidar_{i:05d}" for i in range(len(mg))])
        mg["height_m"] = mg["ridge_height_m"].round(2)
        mg["quality"] = np.where(mg["n_roof_points"] >= 40, "good", np.where(mg["n_roof_points"] >= 12, "fair", "poor"))
        mg["lidar_status"] = "present"
        mg = gpd.GeoDataFrame(add_spec_aliases(mg), geometry="geometry", crs=PROJECTION)
        mg["planes"] = [json.dumps(v) for v in mg["planes"]]  # GeoJSON properties: JSON string
        mg = mg.to_crs("EPSG:4326")
    else:
        mg = gpd.GeoDataFrame({"lidar_id": []}, geometry=[], crs="EPSG:4326")
    p = lidar_dir() / "missing_buildings.geojson"
    tmp = p.with_suffix(".part")
    tmp.unlink(missing_ok=True)
    mg.to_file(tmp, driver="GeoJSON")
    tmp.replace(p)
    return mg


def write_trees(jobs: list[TileJob]) -> Any:
    import pandas as pd

    parts = []
    for j in jobs:
        t = pd.read_parquet(work_dir() / f"trees_{j.ckm}_{j.rkm}.parquet")
        if len(t):
            z = np.load(work_dir() / f"tile_{j.ckm}_{j.rkm}.npz")
            t["canopy_cover_20m"] = canopy_cover_at(z["chm"], z["core"], float(assumption("lidar.raster_res_m")),
                                                    t["easting"].to_numpy(), t["northing"].to_numpy())
        parts.append(t)
    trees = assemble_trees(pd.concat(parts, ignore_index=True))
    _write_parquet_atomic(trees, lidar_dir() / "trees.parquet")
    return trees


def _write_parquet_atomic(df: Any, path: Path) -> None:
    tmp = path.with_suffix(".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="recompute every tile and every roof")
    ap.add_argument("--force-roofs", action="store_true", help="recompute roofs only (tiles stay cached)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tiles-only", action="store_true", help="stop after rasters, trees and lidar-only buildings")
    ap.add_argument("--no-shift", action="store_true", help="do not apply the measured registration shift")
    args = ap.parse_args(argv)
    t0 = time.time()
    timings: dict[str, float] = {}
    tiles_dir = lidar_dir() / "tiles"
    if not tiles_dir.exists() or not any(tiles_dir.glob("*.laz")):
        raise SystemExit("no lidar tiles: run `.venv/bin/python -m pipeline.fetch_lidar` first")
    work_dir().mkdir(parents=True, exist_ok=True)
    res = float(assumption("lidar.raster_res_m"))

    fp = load_footprints()
    log(f"lidar_features: {len(fp)} footprints in bbox")
    timings["load_footprints_s"] = time.time() - t0

    reg_path = work_dir() / "registration.json"
    if reg_path.exists() and not args.force:
        reg = json.loads(reg_path.read_text())
    else:
        t = time.time()
        reg = registration(fp)
        write_json(reg_path, reg)
        timings["registration_s"] = time.time() - t
    shift = (0.0, 0.0)
    mag = math.hypot(reg["dx_m"], reg["dy_m"])
    if not args.no_shift and mag >= 0.75:
        shift = (reg["dx_m"], reg["dy_m"])
    log(f"lidar_features: registration lidar->footprints dx={reg['dx_m']:+.1f} dy={reg['dy_m']:+.1f} m (applied {shift})")

    t = time.time()
    roads = load_road_lines()
    timings["load_roads_s"] = time.time() - t
    jobs = build_jobs(fp, roads, shift, args.force, fp_all=load_footprints(clip=False))

    # 1. per-tile rasters, trees, lidar-only buildings
    t = time.time()
    stats = run_parallel(process_tile, jobs, args.workers, "tile")
    timings["tiles_s"] = time.time() - t
    trees = write_trees(jobs)
    mg = write_missing(jobs)
    log(f"lidar_features: {len(trees):,} trees, {len(mg):,} lidar-only buildings written")

    # 2. rasters
    t = time.time()
    npzs = [work_dir() / f"tile_{j.ckm}_{j.rkm}.npz" for j in jobs]
    rasters = {"dsm": "dsm_0p5m.tif", "dtm": "dtm_0p5m.tif", "ndsm": "ndsm_0p5m.tif", "chm": "chm_0p5m.tif",
               "change": "ground_change_0p5m.tif", "cls": "surface_class_0p5m.tif"}
    if args.force or any(s.get("cached") is False for s in stats) or not all((lidar_dir() / v).exists() for v in rasters.values()):
        mosaic(npzs, raster_extent(), res, rasters)
    timings["mosaic_s"] = time.time() - t
    if args.tiles_only:
        log(f"lidar_features: tiles done in {time.time() - t0:.0f}s (--tiles-only)")
        return 0

    # 3. roofs (need the mosaicked DTM)
    t = time.time()
    rjobs = [TileJob(j.ckm, j.rkm, j.core, j.footprints, [], [], j.shift, args.force or args.force_roofs) for j in jobs]
    rstats = run_parallel(process_roofs, rjobs, args.workers, "roofs")
    timings["roofs_s"] = time.time() - t
    roofs = pd.concat([pd.read_parquet(work_dir() / f"roofs_{j.ckm}_{j.rkm}.parquet") for j in jobs], ignore_index=True)
    out = assemble_buildings(fp, roofs)
    _write_parquet_atomic(out, lidar_dir() / "buildings_roofs.parquet")
    log(f"lidar_features: {len(out):,} roof records written")

    timings["total_s"] = time.time() - t0
    val = validation(out, mg, trees, reg, shift, timings, stats + rstats)
    write_json(lidar_dir() / "lidar_validation.json", val)
    log(json.dumps(val["summary"], indent=1))
    log(f"lidar_features: done in {timings['total_s']:.0f}s")
    return 0


def assemble_buildings(fp: Any, roofs: Any) -> Any:
    import pandas as pd

    from pipeline.build_buildings import parse_height_m

    chg_thr = float(assumption("lidar.ground_change_m"))
    roofs = roofs.sort_values("row").reset_index(drop=True)
    sub = fp.iloc[roofs["row"].to_numpy()].reset_index(drop=True)
    c = sub.geometry.centroid
    cx, cy = c.x.to_numpy(), c.y.to_numpy()
    sx, sz = utm_to_scene_arrays(cx, cy)
    lon, lat = utm_to_lonlat_arrays(cx, cy)
    rp = sub.geometry.representative_point()
    rlon, rlat = utm_to_lonlat_arrays(rp.x.to_numpy(), rp.y.to_numpy())
    # NB: scene centroid columns are scene_x / scene_z (not centroid_x / centroid_z): consumers
    # treat building_id + centroid_x as integer ids of a processed build.
    df = pd.DataFrame({
        "building_id": sub["id"].astype(str),
        "osm_way_id": sub.get("osm_way_id", pd.Series([None] * len(sub))).astype("string"),
        "footprint_source": sub.get("source", pd.Series([None] * len(sub))).astype("string"),
        "centroid_lat": lat, "centroid_lon": lon, "scene_x": sx, "scene_z": sz,
        "lat": rlat, "lon": rlon,
        "easting": cx, "northing": cy,
        "footprint_area_m2": sub.geometry.area.to_numpy(),
        "overture_height_m": [parse_height_m(v) for v in sub["height"]],
    })
    df = pd.concat([df, roofs.drop(columns=["row"])], axis=1)
    expected = df["footprint_area_m2"] * df["point_density_m2"].where(df["point_density_m2"] > 0)
    present = (df["n_roof_points"] >= np.maximum(6, 0.2 * df["n_points_footprint"])) & (df["roof_cover_frac"] >= 0.3)
    regraded = df["ground_change_m"].abs() > chg_thr
    status = np.where(present, "present", np.where(regraded, "absent_regraded_after_2014", "absent"))
    df["lidar_status"] = status
    q = np.full(len(df), "none", dtype=object)
    pres = df["lidar_status"] == "present"
    good = pres & (df["n_roof_points"] >= 40) & (df["inlier_frac"] >= 0.7) & (df["roof_type"] != "unknown")
    fair = pres & ~good & (df["n_roof_points"] >= 12) & (df["roof_type"] != "unknown")
    q[pres] = "poor"
    q[fair] = "fair"
    q[good] = "good"
    df["quality"] = q
    del expected
    # buildings absent from the 2014 lidar keep NaN roof fields (built later or footprint error)
    for col in ("eave_height_m", "ridge_height_m", "height_p50_m", "height_max_m", "roof_pitch_deg", "ridge_azimuth_deg"):
        df.loc[~pres, col] = np.nan
    df.loc[~pres, "roof_type"] = "unknown"
    df.loc[~pres, "planes_json"] = "[]"
    df.loc[~pres, "n_planes"] = 0
    return add_spec_aliases(df)


def add_spec_aliases(df: Any) -> Any:
    """Short field names of the task spec next to the descriptive ones (same values):
    ground_elev, eave_h, ridge_h, pitch_deg, and `planes` as a list of
    {normal, offset, area_m2, slope_deg, aspect_deg} (from planes_json)."""
    df = df.copy()
    df["ground_elev"] = df["ground_elev_m"]
    df["eave_h"] = df["eave_height_m"]
    df["ridge_h"] = df["ridge_height_m"]
    df["pitch_deg"] = df["roof_pitch_deg"]
    df["planes"] = [
        [{"normal": [float(v) for v in q["normal"]], "offset": float(q["d"]), "area_m2": float(q["area_m2"] or 0.0),
          "slope_deg": float(q["slope_deg"]), "aspect_deg": float(q["aspect_deg"])} for q in json.loads(js)]
        for js in df["planes_json"]
    ]
    for c in df.columns:
        if df[c].dtype == np.float64 and c not in ("easting", "northing", "lat", "lon", "centroid_lat", "centroid_lon"):
            df[c] = df[c].round(3)
    return df


def assemble_trees(t: Any) -> Any:
    import pandas as pd

    if not len(t):
        return t
    lon, lat = utm_to_lonlat_arrays(t["easting"].to_numpy(), t["northing"].to_numpy())
    inside = in_bbox(lon, lat)
    t, lon, lat = t[inside].reset_index(drop=True), lon[inside], lat[inside]
    x, z = utm_to_scene_arrays(t["easting"].to_numpy(), t["northing"].to_numpy())
    out = pd.DataFrame({
        "tree_id": np.arange(len(t), dtype=np.int64),
        "x": x, "z": z, "lat": lat, "lon": lon,
        "ground_y": t["ground_y"].to_numpy(), "height_m": t["height_m"].to_numpy(),
        "crown_radius_m": t["crown_radius_m"].to_numpy(),
        "species_guess": species_guess(
            t["height_m"].to_numpy(), t["crown_radius_m"].to_numpy(), t["crown_mean_ratio"].to_numpy(),
            palm_min_h=float(assumption("lidar.palm_min_height_m")),
            palm_max_r=float(assumption("lidar.palm_max_crown_radius_m")),
            palm_max_rh=float(assumption("lidar.palm_max_radius_height_ratio")),
            conifer_max_fill=float(assumption("lidar.conifer_max_crown_fill")),
            conifer_min_h=float(assumption("lidar.conifer_min_height_m")),
            canopy_cover=t["canopy_cover_20m"].to_numpy() if "canopy_cover_20m" in t else None,
            palm_max_cover=float(assumption("lidar.palm_max_canopy_cover")),
        ).astype(str),
        "canopy_cover_20m": t["canopy_cover_20m"].to_numpy() if "canopy_cover_20m" in t else np.nan,
        "crown_mean_ratio": t["crown_mean_ratio"].to_numpy(),
        "ground_lidar_m": t["ground_lidar_m"].to_numpy(),
        "easting": t["easting"].to_numpy(), "northing": t["northing"].to_numpy(),
    })
    for c in ("x", "z", "ground_y", "height_m", "crown_radius_m", "crown_mean_ratio", "ground_lidar_m", "canopy_cover_20m"):
        out[c] = out[c].round(2)
    return out


def validation(b: Any, mg: Any, trees: Any, reg: dict[str, Any], shift: tuple[float, float],
               timings: dict[str, float], stats: list[dict[str, Any]]) -> dict[str, Any]:
    pres = b[b["lidar_status"] == "present"]
    m = pres[pres["overture_height_m"].notna() & pres["ridge_height_m"].notna() & (pres["ground_change_m"].abs() < 1.0)]

    def st(d: np.ndarray) -> dict[str, float]:
        d = d[np.isfinite(d)]
        if not len(d):
            return {}
        return {"n": int(len(d)), "median": float(np.median(d)), "mean": float(np.mean(d)),
                "mae": float(np.mean(np.abs(d))), "rmse": float(np.sqrt(np.mean(d**2))),
                "within_1m": float(np.mean(np.abs(d) <= 1.0)), "within_2m": float(np.mean(np.abs(d) <= 2.0)),
                "p05": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95))}

    m = m[m["quality"].isin(["good", "fair"])]
    ridge_vs = {"all": st((m["overture_height_m"] - m["ridge_height_m"]).to_numpy())}
    for src, grp in m.groupby("footprint_source"):
        ridge_vs[str(src)] = st((grp["overture_height_m"] - grp["ridge_height_m"]).to_numpy())
    for rt, grp in m.groupby("roof_type"):
        ridge_vs[f"roof_{rt}"] = st((grp["overture_height_m"] - grp["ridge_height_m"]).to_numpy())
    other_vs = {f: st((m["overture_height_m"] - m[f]).to_numpy()) for f in ("height_p50_m", "eave_height_m", "height_max_m")}

    def corr_of(f: str) -> float:
        ok = m["overture_height_m"].notna() & m[f].notna()
        return float(np.corrcoef(m.loc[ok, "overture_height_m"], m.loc[ok, f])[0, 1]) if ok.sum() > 2 else float("nan")

    corr = corr_of("ridge_height_m")
    summary = {
        "footprints": int(len(b)),
        "roof_models_by_type": {str(k): int(v) for k, v in pres["roof_type"].value_counts().items()},
        "lidar_status": {str(k): int(v) for k, v in b["lidar_status"].value_counts().items()},
        "quality": {str(k): int(v) for k, v in b["quality"].value_counts().items()},
        "missing_buildings": int(len(mg)),
        "trees": int(len(trees)),
        "trees_by_species_guess": {str(k): int(v) for k, v in trees["species_guess"].value_counts().items()} if len(trees) else {},
        "overture_minus_lidar_ridge_m": ridge_vs,
        "overture_vs_ridge_corr": corr,
        "overture_minus_lidar_other_m": other_vs,
        "overture_vs_other_corr": {f: corr_of(f) for f in ("height_p50_m", "eave_height_m")},
        "ridge_height_m_percentiles": {str(q): float(np.nanpercentile(pres["ridge_height_m"], q)) for q in (5, 25, 50, 75, 95)} if len(pres) else {},
        "pitch_deg_percentiles_sloped": {str(q): float(np.nanpercentile(pres.loc[pres["roof_type"] != "flat", "roof_pitch_deg"], q)) for q in (5, 25, 50, 75, 95)} if len(pres) else {},
        "tree_height_m_percentiles": {str(q): float(np.percentile(trees["height_m"], q)) for q in (5, 25, 50, 75, 95)} if len(trees) else {},
        "tree_crown_radius_m_percentiles": {str(q): float(np.percentile(trees["crown_radius_m"], q)) for q in (5, 25, 50, 75, 95)} if len(trees) else {},
        "missing_buildings_by_roof_type": {str(k): int(v) for k, v in mg["roof_type"].value_counts().items()} if len(mg) and "roof_type" in mg else {},
        "missing_buildings_area_m2_total": float(mg["area_m2"].sum()) if len(mg) and "area_m2" in mg else 0.0,
        "registration_shift_applied_m": list(shift),
    }
    return {"generated_at": now_iso(), "summary": summary, "registration": reg, "timings_s": timings,
            "tiles": [s for s in stats if not s.get("cached")]}


if __name__ == "__main__":
    raise SystemExit(main())
