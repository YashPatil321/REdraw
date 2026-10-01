"""Right-triangulated irregular network (RTIN) terrain meshing, vectorized with numpy.

The algorithm is the one used by Mapbox Martini (Evans, Kirkpatrick & Townsend 1997,
"Right-triangulated irregular networks"): a (2^k + 1)^2 height grid is split recursively
along hypotenuses; every split point carries the maximum interpolation error of the whole
subtree, so extracting the mesh for a given `max_error` keeps a triangle only when its
hypotenuse midpoint error is below the threshold. The result is a crack-free (inside one
grid) adaptive mesh: flat ground gets big triangles, canyon walls, berms and graded pads get
the full 2 m lidar detail.

Here the per-level recursion runs over all triangles of a level at once (numpy), so a 513^2
grid takes well under a second.
"""

from __future__ import annotations

import numpy as np


def _level_coords(level: int, tile: int) -> np.ndarray:
    """(n, 4) int array of (ax, ay, bx, by) for all triangles with heap ids in
    [2^level, 2^(level+1)) (level >= 1). Triangle c is implied (right angle vertex)."""
    ids = np.arange(1 << level, 1 << (level + 1), dtype=np.int64)
    n = len(ids)
    ax = np.zeros(n, np.int64)
    ay = np.zeros(n, np.int64)
    bx = np.zeros(n, np.int64)
    by = np.zeros(n, np.int64)
    cx = np.zeros(n, np.int64)
    cy = np.zeros(n, np.int64)
    odd = (ids & 1) == 1  # bit 0 picks the root triangle
    # root triangles (ids 2 and 3), martini convention
    bx[odd] = tile
    by[odd] = tile
    cx[odd] = tile
    ax[~odd] = tile
    ay[~odd] = tile
    cy[~odd] = tile
    # martini consumes the remaining path bits from the least significant upward
    for bit in range(1, level):
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        left = ((ids >> bit) & 1) == 1
        nax = np.where(left, cx, bx)
        nay = np.where(left, cy, by)
        nbx = np.where(left, ax, cx)
        nby = np.where(left, ay, cy)
        ax, ay, bx, by = nax, nay, nbx, nby
        cx, cy = mx, my
    return np.stack([ax, ay, bx, by], axis=1)


def rtin_errors(heights: np.ndarray) -> np.ndarray:
    """Error map for a (2^k + 1)^2 height grid (heights[y, x]). errors[y, x] is the max
    vertical error (same units as heights) of not splitting at that grid point."""
    size = heights.shape[0]
    tile = size - 1
    if heights.shape != (size, size) or tile & (tile - 1):
        raise ValueError(f"RTIN needs a (2^k+1)^2 grid, got {heights.shape}")
    h = heights.astype(np.float64).ravel()
    err = np.zeros(size * size, dtype=np.float64)
    max_level = int(np.log2(tile)) * 2  # deepest triangles: legs of length 1
    for level in range(max_level, 0, -1):
        c = _level_coords(level, tile)
        ax, ay, bx, by = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        legs = np.abs(ax - bx) + np.abs(ay - by)
        keep = legs > 1  # hypotenuse must have a midpoint
        if not keep.any():
            continue
        ax, ay, bx, by, mx, my = ax[keep], ay[keep], bx[keep], by[keep], mx[keep], my[keep]
        mid = my * size + mx
        e = np.abs((h[ay * size + ax] + h[by * size + bx]) / 2.0 - h[mid])
        np.maximum.at(err, mid, e)
        if level < max_level:  # parent triangles also carry their children's errors
            cx = mx + my - ay
            cy = my + ax - mx
            lc = ((ay + cy) >> 1) * size + ((ax + cx) >> 1)
            rc = ((by + cy) >> 1) * size + ((bx + cx) >> 1)
            np.maximum.at(err, mid, np.maximum(err[lc], err[rc]))
    return err.reshape(size, size)


def rtin_mesh(errors: np.ndarray, max_error: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract the adaptive mesh. Returns (vertices (n, 2) int grid coords [x, y],
    triangles (m, 3) indices into vertices). Winding: as produced (callers fix it)."""
    size = errors.shape[0]
    tile = size - 1
    err = errors.ravel()
    # (ax, ay, bx, by, cx, cy)
    cur = np.array([[0, 0, tile, tile, tile, 0], [tile, tile, 0, 0, 0, tile]], dtype=np.int64)
    out = []
    while len(cur):
        ax, ay, bx, by, cx, cy = cur.T
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        split = ((np.abs(ax - cx) + np.abs(ay - cy)) > 1) & (err[my * size + mx] > max_error)
        out.append(cur[~split])
        s = cur[split]
        if not len(s):
            break
        ax, ay, bx, by, cx, cy = s.T
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        left = np.stack([cx, cy, ax, ay, mx, my], axis=1)
        right = np.stack([bx, by, cx, cy, mx, my], axis=1)
        cur = np.concatenate([left, right])
    tris = np.concatenate(out) if out else np.zeros((0, 6), np.int64)
    pts = tris.reshape(-1, 2)
    keys = pts[:, 1] * size + pts[:, 0]
    uniq, inv = np.unique(keys, return_inverse=True)
    verts = np.stack([uniq % size, uniq // size], axis=1)
    return verts, inv.reshape(-1, 3)


def rtin_count(errors: np.ndarray, max_error: float) -> int:
    """Triangle count rtin_mesh would produce (cheap: no vertex dedup)."""
    size = errors.shape[0]
    tile = size - 1
    err = errors.ravel()
    cur = np.array([[0, 0, tile, tile, tile, 0], [tile, tile, 0, 0, 0, tile]], dtype=np.int64)
    n = 0
    while len(cur):
        ax, ay, bx, by, cx, cy = cur.T
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        split = ((np.abs(ax - cx) + np.abs(ay - cy)) > 1) & (err[my * size + mx] > max_error)
        n += int((~split).sum())
        s = cur[split]
        if not len(s):
            break
        ax, ay, bx, by, cx, cy = s.T
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        cur = np.concatenate([np.stack([cx, cy, ax, ay, mx, my], axis=1), np.stack([bx, by, cx, cy, mx, my], axis=1)])
    return n
