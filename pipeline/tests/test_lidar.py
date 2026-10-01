"""Pure-function tests for pipeline/fetch_lidar.py and pipeline/lidar_features.py (no data)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.fetch_lidar import Box, ept_children, ept_node_box, tile_keys_for, tile_name
from pipeline.lidar_features import (
    Grid,
    analyze_roof,
    angle_diff,
    crown_window_radius_m,
    detect_trees,
    estimate_shift,
    fill_nan,
    fit_plane_lsq,
    interpolate_ground,
    largest_component,
    outline_samples,
    plane_residual_std,
    point_normals,
    rasterize_stat,
    ridge_azimuth,
    species_guess,
)

# ---------------------------------------------------------------------------
# fetch_lidar helpers
# ---------------------------------------------------------------------------


def test_tile_keys_cover_box() -> None:
    keys = tile_keys_for(Box(484385.2, 3650313.8, 486100.0, 3651000.0))
    assert keys == [(484, 3650), (485, 3650), (486, 3650)]
    assert tile_name(484, 3650) == "utm_484_3650.laz"


def test_box_intersects_and_buffer() -> None:
    a = Box(0, 0, 10, 10)
    assert a.intersects(Box(9, 9, 20, 20))
    assert not a.intersects(Box(10, 0, 20, 10))  # touching edges do not intersect
    assert a.buffered(2) == Box(-2, -2, 12, 12)


def test_ept_node_box_and_children() -> None:
    cube = [0.0, 0.0, 0.0, 1024.0, 1024.0, 1024.0]
    assert ept_node_box("0-0-0-0", cube) == Box(0, 0, 1024, 1024)
    assert ept_node_box("2-1-3-0", cube) == Box(256, 768, 512, 1024)
    kids = ept_children("1-1-0-1")
    assert len(kids) == 8 and "2-2-0-2" in kids and "2-3-1-3" in kids


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------


def test_grid_rc_and_rasterize_stat() -> None:
    g = Grid.from_box(Box(0, 0, 4, 2), 1.0)
    assert g.shape == (2, 4)
    x = np.array([0.5, 0.6, 3.5, 10.0])
    y = np.array([1.5, 1.2, 0.5, 0.5])  # last point is outside
    v = np.array([1.0, 3.0, 5.0, 7.0])
    mx = rasterize_stat(g, x, y, v, "max")
    assert mx[0, 0] == 3.0 and mx[1, 3] == 5.0 and np.isnan(mx[1, 0])
    assert rasterize_stat(g, x, y, v, "count").sum() == 3
    assert rasterize_stat(g, x, y, v, "mean")[0, 0] == pytest.approx(2.0)


def test_fill_nan_fills_everything() -> None:
    a = np.arange(25, dtype=np.float32).reshape(5, 5)
    a[1:4, 1:4] = np.nan
    f = fill_nan(a)
    assert np.isfinite(f).all()
    assert f[0, 0] == 0.0  # valid cells untouched


def test_plane_residual_std_flat_vs_rough() -> None:
    yy, xx = np.mgrid[0:20, 0:20].astype(np.float64)
    plane = 0.3 * xx - 0.2 * yy + 5.0
    assert plane_residual_std(plane)[2:-2, 2:-2].max() < 1e-6
    rough = plane + np.random.default_rng(0).normal(0, 0.5, plane.shape)
    assert np.median(plane_residual_std(rough)) > 0.3


def test_interpolate_ground_recovers_slope_under_gap() -> None:
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 50, 5000), rng.uniform(0, 50, 5000)
    keep = ~((np.abs(x - 25) < 8) & (np.abs(y - 25) < 8))  # building-sized hole
    z = 100 + 0.1 * x + 0.05 * y
    g = Grid.from_box(Box(0, 0, 50, 50), 0.5)
    dtm = interpolate_ground(g, x[keep], y[keep], z[keep])
    xs, ys = g.centers()
    truth = 100 + 0.1 * xs[None, :] + 0.05 * ys[:, None]
    assert np.abs(dtm - truth)[10:-10, 10:-10].max() < 0.15


def test_estimate_shift_finds_offset() -> None:
    fp = np.zeros((80, 80), dtype=bool)
    fp[20:40, 30:50] = True
    fp[50:60, 10:25] = True
    lidar = np.roll(np.roll(fp, 2, axis=0), -3, axis=1).astype(np.float32) * 6.0  # 2 rows S, 3 cols W
    dx, dy, iou, iou0 = estimate_shift(lidar, fp, res=0.5, max_shift_m=3.0)
    assert (dx, dy) == (1.5, 1.0)  # shift that moves lidar back: +3 cols (E), -2 rows (N)
    assert iou == pytest.approx(1.0) and iou0 < 0.6


# ---------------------------------------------------------------------------
# Roofs
# ---------------------------------------------------------------------------


def _roof_points(kind: str, L: float = 16.0, W: float = 10.0, theta_deg: float = 30.0, pitch_deg: float = 22.0,
                 eave: float = 3.0, density: float = 4.5, noise: float = 0.04, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic roof point cloud (local e, n, h) and its outline. Ridge along the long axis,
    which points to compass azimuth `theta_deg`."""
    rng = np.random.default_rng(seed)
    n = int(L * W * density)
    u = rng.uniform(-L / 2, L / 2, n)  # along the long axis
    v = rng.uniform(-W / 2, W / 2, n)
    t = math.tan(math.radians(pitch_deg))
    if kind == "gable":
        h = eave + t * (W / 2 - np.abs(v))
    elif kind == "hip":
        h = eave + t * np.minimum(W / 2 - np.abs(v), L / 2 - np.abs(u))
    elif kind == "shed":
        h = eave + t * (v + W / 2)
    else:
        h = np.full(n, eave + 1.0)
    h = h + rng.normal(0, noise, n)
    a = math.radians(theta_deg)
    ax, ay = math.sin(a), math.cos(a)  # long axis unit vector (east, north)
    px, py = -ay, ax
    e = u * ax + v * px
    nn = u * ay + v * py
    corners = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2]])
    ring = np.column_stack([corners[:, 0] * ax + corners[:, 1] * px, corners[:, 0] * ay + corners[:, 1] * py])
    return np.column_stack([e, nn, h]), ring


@pytest.mark.parametrize("kind", ["gable", "hip", "flat", "shed"])
def test_analyze_roof_types(kind: str) -> None:
    pts, ring = _roof_points(kind)
    m = analyze_roof(pts, ring, rng=np.random.default_rng(3))
    assert m.roof_type == kind
    if kind != "flat":
        assert m.roof_pitch_deg == pytest.approx(22.0, abs=2.0)
    if kind in ("gable", "hip"):
        assert angle_diff(m.ridge_azimuth_deg, 30.0) < 5.0 or angle_diff(m.ridge_azimuth_deg, 210.0) < 5.0
        assert m.ridge_height_m == pytest.approx(3.0 + 5.0 * math.tan(math.radians(22.0)), abs=0.3)
        assert m.eave_height_m < 3.6
    if kind == "flat":
        assert m.eave_height_m == m.ridge_height_m == pytest.approx(4.0, abs=0.15)


def test_eave_fraction_separates_hip_and_gable() -> None:
    hip = analyze_roof(*_roof_points("hip"), rng=np.random.default_rng(1))
    gab = analyze_roof(*_roof_points("gable"), rng=np.random.default_rng(1))
    assert hip.eave_perimeter_frac > 0.85
    assert 0.45 < gab.eave_perimeter_frac < 0.75  # ~L/(L+W) = 0.62


def test_analyze_roof_too_few_points_is_unknown() -> None:
    m = analyze_roof(np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 3.1]]), None)
    assert m.roof_type == "unknown" and m.n_planes == 0


def test_fit_plane_and_ridge_azimuth() -> None:
    rng = np.random.default_rng(0)
    xy = rng.uniform(-5, 5, (200, 2))
    p = np.column_stack([xy, 2.0 + 0.5 * xy[:, 0]])  # rises to the east -> faces west
    n, d, rmse = fit_plane_lsq(p)
    assert n[2] > 0 and rmse < 1e-9
    assert np.degrees(np.arctan2(n[0], n[1])) % 360 == pytest.approx(270.0)
    # planes facing east and west meet in a north-south ridge
    assert ridge_azimuth(np.array([0.4, 0.0, 0.9]), np.array([-0.4, 0.0, 0.9])) == pytest.approx(0.0, abs=1e-9)
    assert angle_diff(350.0, 10.0) == pytest.approx(20.0)


def test_point_normals_on_tilted_plane() -> None:
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10, (400, 2))
    p = np.column_stack([xy, 0.4 * xy[:, 1]])
    nrm = point_normals(p, k=10)
    expect = np.array([0.0, -0.4, 1.0]) / math.hypot(0.4, 1.0)
    assert np.median(np.abs(nrm @ expect)) > 0.999


def test_largest_component() -> None:
    xy = np.vstack([np.random.default_rng(0).uniform(0, 5, (100, 2)), np.random.default_rng(1).uniform(20, 22, (30, 2))])
    m = largest_component(xy, 1.0)
    assert m[:100].all() and not m[100:].any()


def test_outline_samples_outward_normals_both_orientations() -> None:
    sq = np.array([[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]], dtype=float)
    for ring in (sq, sq[::-1]):
        xy, nrm, w = outline_samples(ring, step=1.0)
        assert w.sum() == pytest.approx(16.0)
        # outward normals point away from the centre
        assert np.all(np.einsum("ij,ij->i", xy - 2.0, nrm) > 0)


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


def test_detect_trees_on_synthetic_chm() -> None:
    res = 0.5
    yy, xx = np.mgrid[0:120, 0:120] * res
    chm = np.zeros(xx.shape, dtype=np.float32)
    trees = [(10.0, 10.0, 12.0, 3.0), (30.0, 12.0, 8.0, 2.0), (20.0, 40.0, 15.0, 4.0)]  # x, y, h, crown r
    for tx, ty, h, r in trees:
        d2 = (xx - tx) ** 2 + (yy - ty) ** 2
        chm = np.maximum(chm, (h * np.clip(1 - d2 / r**2, 0, None) ** 0.5).astype(np.float32))
    ts = detect_trees(chm, res, min_h=4.0)
    assert len(ts.rows) == 3
    order = np.argsort(ts.height)
    assert np.allclose(ts.height[order], [8.0, 12.0, 15.0], atol=0.6)
    assert np.allclose(ts.crown_radius[order], [2.0, 3.0, 4.0], atol=0.8)


def test_crown_window_and_species_guess() -> None:
    assert crown_window_radius_m(0.0) == pytest.approx(1.54816, abs=1e-4)
    assert crown_window_radius_m(40.0) == 5.0
    g = species_guess(np.array([12.0, 12.0, 15.0]), np.array([1.5, 4.0, 2.5]), np.array([0.9, 0.9, 0.5]))
    assert list(g) == ["palm", "broadleaf", "conifer"]
