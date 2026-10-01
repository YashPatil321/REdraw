"""Detailed building geometry: per-house pitched roofs, parapets, facade UVs and material ids.

Conventions (docs/data_contract.md "HD building tiles"; shared with the Blender agent's
client/public/assets/materials/materials_manifest.json):

- `_MAT` (uint8): 0 stucco wall, 1 tile roof, 2 flat roof, 3 glass, 4 trim, 5 garage door.
- `_VARIANT` (uint8): index into materials_manifest materials[_MAT].variants.
- `_FRONT` (uint8): 1 on wall quads that face the street the house is addressed from
  (the wall that holds the garage door), else 0. Lets the client's bay grammar put the
  entry / garage openings on the right wall.
- TEXCOORD_0: walls, glass, trim, garage (facade): u = meters along the wall / 3, v = meters
  above the building base (finished floor) / 3; pitched roofs: u = meters along the eave / 4,
  v = meters up the slope / 4; flat roofs: u = x / 4, v = -z / 4.
- COLOR_0: wall / roof / trim tint (also the color-only fallback).

Everything works in scene coordinates (x east, z south, y up).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mapbox_earcut as earcut
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

MAT_WALL, MAT_TILE_ROOF, MAT_FLAT_ROOF, MAT_GLASS, MAT_TRIM, MAT_GARAGE = range(6)
FACADE_M = 3.0
ROOF_M = 4.0


@dataclass
class MeshAcc:
    """Growable triangle soup with the building tile attributes."""

    pos: list[np.ndarray] = field(default_factory=list)
    nrm: list[np.ndarray] = field(default_factory=list)
    uv: list[np.ndarray] = field(default_factory=list)
    col: list[np.ndarray] = field(default_factory=list)
    mat: list[np.ndarray] = field(default_factory=list)
    var: list[np.ndarray] = field(default_factory=list)
    front: list[np.ndarray] = field(default_factory=list)
    bid: list[np.ndarray] = field(default_factory=list)
    tri: list[np.ndarray] = field(default_factory=list)
    n: int = 0
    # current building attributes (set per building)
    cur_bid: int = 0

    def add(
        self,
        pos: np.ndarray,
        nrm: np.ndarray,
        uv: np.ndarray,
        tri: np.ndarray,
        mat: int,
        var: int,
        color: tuple[int, int, int] | np.ndarray,
        front: int = 0,
    ) -> None:
        k = len(pos)
        if k == 0 or len(tri) == 0:
            return
        self.pos.append(np.asarray(pos, dtype=np.float64))
        self.nrm.append(np.asarray(nrm, dtype=np.float64))
        self.uv.append(np.asarray(uv, dtype=np.float64))
        self.tri.append(np.asarray(tri, dtype=np.int64) + self.n)
        c = np.asarray(color, dtype=np.float64)
        self.col.append(np.tile(c[:3], (k, 1)) if c.ndim == 1 else c[:, :3])
        self.mat.append(np.full(k, mat, np.uint8))
        self.var.append(np.full(k, var, np.uint8))
        self.front.append(np.full(k, front, np.uint8))
        self.bid.append(np.full(k, self.cur_bid, np.int64))
        self.n += k

    @property
    def triangles(self) -> int:
        return int(sum(len(t) for t in self.tri))

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.pos:
            return {}
        return {
            "pos": np.concatenate(self.pos).astype(np.float32),
            "nrm": np.concatenate(self.nrm).astype(np.float32),
            "uv": np.concatenate(self.uv).astype(np.float32),
            "col": np.clip(np.concatenate(self.col), 0, 255).astype(np.uint8),
            "mat": np.concatenate(self.mat),
            "var": np.concatenate(self.var),
            "front": np.concatenate(self.front),
            "bid": np.concatenate(self.bid),
            "tri": np.concatenate(self.tri).reshape(-1).astype(np.uint32),
        }


# ---------------------------------------------------------------------------
# Footprint analysis
# ---------------------------------------------------------------------------


def dominant_angle(poly: Polygon) -> float:
    """Angle (radians) of the longest edge of the minimum rotated rectangle."""
    r = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:4]
    e0 = r[1] - r[0]
    e1 = r[2] - r[1]
    e = e0 if np.hypot(*e0) >= np.hypot(*e1) else e1
    return math.atan2(e[1], e[0])


def _cluster(vals: np.ndarray, tol: float) -> list[float]:
    v = np.sort(vals)
    out: list[list[float]] = [[float(v[0])]]
    for x in v[1:]:
        if x - out[-1][-1] <= tol:
            out[-1].append(float(x))
        else:
            out.append([float(x)])
    return [float(np.mean(g)) for g in out]


def _slabs(pr: Polygon, axis: int, tol: float, min_w: float) -> list[tuple[float, float, float, float]]:
    """Axis-aligned rectangles (x0, x1, z0, z1) from slabs along `axis` (0 = x) of a polygon
    already rotated to its dominant orientation."""
    coords = np.asarray(pr.exterior.coords)[:, :2]
    cuts = _cluster(coords[:, axis], tol)
    lo_all, hi_all = coords[:, axis].min(), coords[:, axis].max()
    cuts[0], cuts[-1] = float(lo_all), float(hi_all)
    bx0, bz0, bx1, bz1 = pr.bounds
    raw: list[list[float]] = []
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        if b - a < 0.05:
            continue
        slab = box(a, bz0 - 1.0, b, bz1 + 1.0) if axis == 0 else box(bx0 - 1.0, a, bx1 + 1.0, b)
        inter = pr.intersection(slab)
        # polygonal parts only: edges lying on the slab boundary come back as zero-area lines
        # whose bounds would stretch the slab across the neighbouring wing
        parts = [g for g in getattr(inter, "geoms", [inter]) if g.geom_type == "Polygon" and g.area > 1e-6]
        if not parts or sum(g.area for g in parts) < 0.05 * (b - a):
            continue
        x0, z0, x1, z1 = unary_union(parts).bounds
        lo, hi = (z0, z1) if axis == 0 else (x0, x1)
        raw.append([a, b, lo, hi])
    if not raw:
        return []
    # merge neighbours with (nearly) the same cross extent
    merged = [raw[0]]
    for s in raw[1:]:
        m = merged[-1]
        if abs(s[2] - m[2]) <= tol and abs(s[3] - m[3]) <= tol and abs(s[0] - m[1]) < 1e-6:
            m[1] = s[1]
            m[2] = min(m[2], s[2])
            m[3] = max(m[3], s[3])
        else:
            merged.append(s)
    # absorb slivers into the neighbour with the closest cross extent
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, s in enumerate(merged):
            if s[1] - s[0] >= min_w:
                continue
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(merged)]
            j = min(nb, key=lambda q: abs(merged[q][2] - s[2]) + abs(merged[q][3] - s[3]))
            t = merged[j]
            t[0], t[1] = min(t[0], s[0]), max(t[1], s[1])
            merged.pop(i)
            changed = True
            break
    out = []
    for a, b, lo, hi in merged:
        if hi - lo < min_w * 0.75:
            continue
        out.append((a, b, lo, hi) if axis == 0 else (lo, hi, a, b))
    return out


@dataclass
class RoofPlan:
    """Pitched-roof plan of a footprint: wall outline + roof rectangles (scene coordinates)."""

    walls: Polygon  # orthogonal outline the walls follow
    rects: list[np.ndarray]  # (4, 2) corners of each roof rectangle (CCW), may overlap (wings extend into mains)
    iou: float


def roof_plan(poly: Polygon, max_rects: int = 4, tol: float = 1.0, min_w: float = 2.5, min_iou: float = 0.78) -> RoofPlan | None:
    """Decompose a (roughly orthogonal) footprint into up to `max_rects` rectangles.

    Slabs along both principal axes are tried; the better one wins. Narrow wings are
    extended into the neighbouring larger rectangle by half its depth so their roofs
    intersect the main roof the way real cross-hips / cross-gables do. Returns None when
    the footprint is not well approximated (caller falls back to a flat roof)."""
    if poly.is_empty or poly.area < 12.0:
        return None
    c0 = poly.centroid
    if abs(c0.x) > 1e4 or abs(c0.y) > 1e4:  # work near the origin (float precision), shift back
        from shapely.affinity import translate

        plan = roof_plan(translate(poly, -c0.x, -c0.y), max_rects, tol, min_w, min_iou)
        if plan is None:
            return None
        return RoofPlan(walls=translate(plan.walls, c0.x, c0.y), rects=[r + np.array([c0.x, c0.y]) for r in plan.rects], iou=plan.iou)
    ang = dominant_angle(poly)
    c = poly.centroid
    pr = rotate(poly, -ang, origin=c, use_radians=True)
    best: tuple[float, list[tuple[float, float, float, float]]] | None = None
    for axis in (0, 1):
        rs = _slabs(pr, axis, tol, min_w)
        if not rs or len(rs) > max_rects:
            continue
        u = unary_union([box(r[0], r[2], r[1], r[3]) for r in rs])
        inter = u.intersection(pr).area
        iou = inter / max(u.union(pr).area, 1e-9)
        score = iou - 0.015 * len(rs)
        if best is None or score > best[0]:
            best = (score, rs)
    if best is None:
        return None
    rs = best[1]
    u = unary_union([box(r[0], r[2], r[1], r[3]) for r in rs])
    iou = u.intersection(pr).area / max(u.union(pr).area, 1e-9)
    if iou < min_iou or u.geom_type != "Polygon":
        return None
    # wings: extend the narrower of two touching rectangles into the wider one
    ext = [list(r) for r in rs]
    for i in range(len(rs)):
        for j in range(len(rs)):
            if i == j:
                continue
            a, b = rs[i], rs[j]
            wa_x, wa_z = a[1] - a[0], a[3] - a[2]
            if abs(a[1] - b[0]) < 1e-6 or abs(a[0] - b[1]) < 1e-6:  # touch along x
                if wa_z < (b[3] - b[2]) - 0.5:
                    d = (b[1] - b[0]) / 2.0
                    if abs(a[1] - b[0]) < 1e-6:
                        ext[i][1] = max(ext[i][1], a[1] + d)
                    else:
                        ext[i][0] = min(ext[i][0], a[0] - d)
            if abs(a[3] - b[2]) < 1e-6 or abs(a[2] - b[3]) < 1e-6:  # touch along z
                if wa_x < (b[1] - b[0]) - 0.5:
                    d = (b[3] - b[2]) / 2.0
                    if abs(a[3] - b[2]) < 1e-6:
                        ext[i][3] = max(ext[i][3], a[3] + d)
                    else:
                        ext[i][2] = min(ext[i][2], a[2] - d)
    walls = rotate(orient(u.simplify(0.05), 1.0), ang, origin=c, use_radians=True)
    rects = []
    for x0, x1, z0, z1 in ext:
        rp = rotate(box(x0, z0, x1, z1), ang, origin=c, use_radians=True)
        rects.append(np.asarray(orient(rp, 1.0).exterior.coords)[:4, :2])
    return RoofPlan(walls=orient(walls, 1.0), rects=rects, iou=float(iou))


# ---------------------------------------------------------------------------
# Mesh pieces
# ---------------------------------------------------------------------------


def ring_coords(ring: np.ndarray) -> np.ndarray:
    r = np.asarray(ring, dtype=np.float64)[:, :2]
    if len(r) > 1 and np.allclose(r[0], r[-1]):
        r = r[:-1]
    return r


def add_walls(
    acc: MeshAcc,
    ring: np.ndarray,
    y0: float,
    y1: float,
    base_y: float,
    mat: int,
    var: int,
    color: tuple[int, int, int],
    outward: bool = True,
    front_mask: np.ndarray | None = None,
) -> None:
    """Vertical quads for each segment of a ring (x, z). For a CCW ring (seen from above, in
    x/z math orientation) the faces point outward when `outward`."""
    r = ring_coords(ring)
    if len(r) < 2:
        return
    p0 = r
    p1 = np.roll(r, -1, axis=0)
    d = p1 - p0
    ln = np.hypot(d[:, 0], d[:, 1])
    keep = ln > 1e-3
    p0, p1, d, ln = p0[keep], p1[keep], d[keep], ln[keep]
    fm = np.zeros(len(r), np.uint8) if front_mask is None else np.asarray(front_mask, np.uint8)
    fm = fm[keep]
    k = len(p0)
    if k == 0:
        return
    n2 = np.column_stack([d[:, 1], -d[:, 0]]) / ln[:, None]
    if not outward:
        n2 = -n2
    q = np.empty((k, 4, 3))
    q[:, 0] = np.column_stack([p0[:, 0], np.full(k, y0), p0[:, 1]])
    q[:, 1] = np.column_stack([p1[:, 0], np.full(k, y0), p1[:, 1]])
    q[:, 2] = np.column_stack([p1[:, 0], np.full(k, y1), p1[:, 1]])
    q[:, 3] = np.column_stack([p0[:, 0], np.full(k, y1), p0[:, 1]])
    nrm = np.repeat(np.column_stack([n2[:, 0], np.zeros(k), n2[:, 1]])[:, None, :], 4, axis=1)
    uv = np.empty((k, 4, 2))
    uv[:, 0] = np.column_stack([np.zeros(k), np.full(k, (y0 - base_y) / FACADE_M)])
    uv[:, 1] = np.column_stack([ln / FACADE_M, np.full(k, (y0 - base_y) / FACADE_M)])
    uv[:, 2] = np.column_stack([ln / FACADE_M, np.full(k, (y1 - base_y) / FACADE_M)])
    uv[:, 3] = np.column_stack([np.zeros(k), np.full(k, (y1 - base_y) / FACADE_M)])
    base = np.arange(k)[:, None] * 4
    tri = np.concatenate([np.hstack([base, base + 1, base + 2]), np.hstack([base, base + 2, base + 3])])
    pos = q.reshape(-1, 3)
    nr = nrm.reshape(-1, 3)
    tri = fix_winding(pos, tri, (nr[tri[:, 0]]))
    if fm.any():
        for flag in (0, 1):
            sel = np.where(fm == flag)[0]
            if not len(sel):
                continue
            vi = (sel[:, None] * 4 + np.arange(4)[None, :]).reshape(-1)
            remap = -np.ones(len(pos), np.int64)
            remap[vi] = np.arange(len(vi))
            tsel = tri[np.all(np.isin(tri, vi), axis=1)]
            acc.add(pos[vi], nr[vi], uv.reshape(-1, 2)[vi], remap[tsel], mat, var, color, front=flag)
        return
    acc.add(pos, nr, uv.reshape(-1, 2), tri, mat, var, color)


def fix_winding(pos: np.ndarray, tri: np.ndarray, desired: np.ndarray) -> np.ndarray:
    v0, v1, v2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    flip = np.einsum("ij,ij->i", n, desired) < 0
    tri = tri.copy()
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri


def triangulate(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Earcut a polygon with holes. Returns (verts (n, 2), tri (m, 3))."""
    rings = [ring_coords(np.asarray(poly.exterior.coords))] + [ring_coords(np.asarray(r.coords)) for r in poly.interiors]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return np.zeros((0, 2)), np.zeros((0, 3), np.int64)
    verts = np.concatenate(rings).astype(np.float64)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    idx = earcut.triangulate_float64(verts, ends).reshape(-1, 3).astype(np.int64)
    return verts, idx


def add_flat(acc: MeshAcc, poly: Polygon, y: float, mat: int, var: int, color: tuple[int, int, int], up: bool = True, uv_m: float = ROOF_M) -> None:
    v, t = triangulate(poly)
    if not len(t):
        return
    pos = np.column_stack([v[:, 0], np.full(len(v), y), v[:, 1]])
    n = np.array([0.0, 1.0 if up else -1.0, 0.0])
    nrm = np.tile(n, (len(v), 1))
    uv = np.column_stack([v[:, 0] / uv_m, -v[:, 1] / uv_m])
    acc.add(pos, nrm, uv, fix_winding(pos, t, np.tile(n, (len(t), 1))), mat, var, color)


def add_quad(
    acc: MeshAcc, corners: np.ndarray, normal: np.ndarray, uv: np.ndarray, mat: int, var: int, color: tuple[int, int, int], front: int = 0
) -> None:
    """One quad (4, 3) corners in order, explicit (4, 2) uvs; winding follows `normal`."""
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    nrm = np.tile(normal, (4, 1))
    acc.add(corners, nrm, uv, fix_winding(corners, tri, np.tile(normal, (2, 1))), mat, var, color, front)


def pitched_roof(
    acc: MeshAcc,
    rect: np.ndarray,
    wall_top: float,
    pitch_deg: float,
    overhang: float,
    gabled: bool,
    roof_var: int,
    roof_color: tuple[int, int, int],
    wall_var: int,
    wall_color: tuple[int, int, int],
    trim_color: tuple[int, int, int],
    base_y: float,
) -> float:
    """Hip or gable roof over one rectangle (4, 2 corners) whose roof plane passes through the
    wall top line; eaves overhang by `overhang` with a soffit and fascia (trim). Gable ends get
    triangular stucco walls. Returns the ridge height."""
    c = np.asarray(rect, dtype=np.float64)
    L = float(np.hypot(*(c[1] - c[0])))
    W = float(np.hypot(*(c[2] - c[1])))
    if L < W:
        c = np.roll(c, -1, axis=0)
        L, W = W, L
    a = (c[1] - c[0]) / max(L, 1e-9)  # along the ridge
    b = (c[3] - c[0]) / max(W, 1e-9)  # across (from edge c0-c1 toward c3-c2)
    m = c.mean(axis=0)
    t = math.tan(math.radians(pitch_deg))
    o = overhang
    # expanded rectangle (eaves). Gables overhang at the rake too.
    hl = L / 2 + o
    hw = W / 2 + o
    eave_y = wall_top - o * t
    ridge_y = wall_top + (W / 2) * t
    r_half = hl if gabled else max(0.0, hl - hw)
    R0 = m - a * r_half
    R1 = m + a * r_half
    E = [m - a * hl - b * hw, m + a * hl - b * hw, m + a * hl + b * hw, m - a * hl + b * hw]

    def p3(p: np.ndarray, y: float) -> np.ndarray:
        return np.array([p[0], y, p[1]])

    slope = math.hypot(hw, ridge_y - eave_y)

    def face(pts2: list[np.ndarray], ys: list[float], eave_dir: np.ndarray, eave_origin: np.ndarray, down: np.ndarray) -> None:
        pts = np.array([p3(p, y) for p, y in zip(pts2, ys, strict=True)])
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        n /= max(np.linalg.norm(n), 1e-9)
        if n[1] < 0:
            n = -n
        # uv: u along the eave, v up the slope (meters / ROOF_M)
        rel = np.array(pts2) - eave_origin
        u = rel @ eave_dir
        dist_in = np.abs(rel @ down)  # horizontal distance from the eave line
        v = dist_in / max(hw, 1e-9) * slope
        uv = np.column_stack([u / ROOF_M, v / ROOF_M])
        tri = np.array([[0, 1, 2]] if len(pts) == 3 else [[0, 1, 2], [0, 2, 3]])
        acc.add(pts, np.tile(n, (len(pts), 1)), uv, fix_winding(pts, tri, np.tile(n, (len(tri), 1))), MAT_TILE_ROOF, roof_var, roof_color)

    # long faces
    face([E[0], E[1], R1, R0], [eave_y, eave_y, ridge_y, ridge_y], a, E[0], b)
    face([E[2], E[3], R0, R1], [eave_y, eave_y, ridge_y, ridge_y], -a, E[2], -b)
    if gabled:
        # gable triangles in the wall planes (stucco), rake overhang already in the long faces
        for sgn, corner_a, corner_b in ((-1, c[0], c[3]), (1, c[1], c[2])):
            apex = m + a * sgn * (L / 2)
            pts = np.array([p3(corner_a, wall_top), p3(corner_b, wall_top), p3(apex, ridge_y)])
            n = np.array([a[0] * sgn, 0.0, a[1] * sgn])
            rel = np.array([corner_a, corner_b, apex]) - corner_a
            u = rel @ b
            uv = np.column_stack([u / 3.0, (pts[:, 1] - base_y) / 3.0])
            acc.add(pts, np.tile(n, (3, 1)), uv, fix_winding(pts, np.array([[0, 1, 2]]), n[None, :]), MAT_WALL, wall_var, wall_color)
    else:
        face([E[1], E[2], R1], [eave_y, eave_y, ridge_y], b, E[1], -a)
        face([E[3], E[0], R0], [eave_y, eave_y, ridge_y], -b, E[3], a)
    # soffit (under the overhang) and fascia (vertical board at the eave), trim material
    wall = [c[0], c[1], c[2], c[3]]
    sides = [(0, 1), (2, 3)] if gabled else [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in sides:
        e0, e1 = E[i], E[j]
        w0, w1 = wall[i], wall[j]
        n_out = np.array([e0[0] - w0[0] + e1[0] - w1[0], 0.0, e0[1] - w0[1] + e1[1] - w1[1]])
        n_out2 = n_out / max(np.linalg.norm(n_out), 1e-9)
        # project the out direction to be perpendicular to the edge
        ed = (e1 - e0) / max(np.hypot(*(e1 - e0)), 1e-9)
        perp = np.array([ed[1], -ed[0]])
        if perp @ np.array([n_out2[0], n_out2[2]]) < 0:
            perp = -perp
        nrm = np.array([perp[0], 0.0, perp[1]])
        ln = float(np.hypot(*(e1 - e0)))
        fascia = np.array([p3(e0, eave_y - 0.2), p3(e1, eave_y - 0.2), p3(e1, eave_y + 0.02), p3(e0, eave_y + 0.02)])
        uv = np.array([[0, (eave_y - 0.2 - base_y) / 3.0], [ln / 3.0, (eave_y - 0.2 - base_y) / 3.0], [ln / 3.0, (eave_y - base_y) / 3.0], [0, (eave_y - base_y) / 3.0]])
        add_quad(acc, fascia, nrm, uv, MAT_TRIM, 0, trim_color)
        soffit = np.array([p3(w0, wall_top - 0.01), p3(w1, wall_top - 0.01), p3(e1, eave_y - 0.2), p3(e0, eave_y - 0.2)])
        uvs = np.column_stack([np.array([0, ln, ln, 0]) / 3.0, np.array([0, 0, o, o]) / 3.0])
        add_quad(acc, soffit, np.array([0.0, -1.0, 0.0]), uvs, MAT_TRIM, 0, trim_color)
    return ridge_y


def parapet_roof(
    acc: MeshAcc,
    poly: Polygon,
    roof_y: float,
    parapet_h: float,
    inset: float,
    roof_var: int,
    roof_color: tuple[int, int, int],
    wall_var: int,
    wall_color: tuple[int, int, int],
    trim_color: tuple[int, int, int],
    base_y: float,
) -> None:
    """Flat roof with a parapet: inner parapet walls, a cap (trim) and the roof deck inset."""
    inner = poly.buffer(-inset, join_style=2)
    if parapet_h <= 0 or inner.is_empty or inner.geom_type != "Polygon" or inner.area < 0.3 * poly.area:
        add_flat(acc, poly, roof_y, MAT_FLAT_ROOF, roof_var, roof_color)
        return
    inner = orient(inner, 1.0)
    top = roof_y + parapet_h
    # inner faces look inward: ring CCW, so flip
    add_walls(acc, np.asarray(inner.exterior.coords), roof_y, top, base_y, MAT_WALL, wall_var, wall_color, outward=False)
    for h in inner.interiors:
        add_walls(acc, np.asarray(h.coords), roof_y, top, base_y, MAT_WALL, wall_var, wall_color, outward=False)
    cap = poly.difference(inner)
    for g in getattr(cap, "geoms", [cap]):
        if g.geom_type == "Polygon" and g.area > 0:
            add_flat(acc, g, top, MAT_TRIM, 0, trim_color, uv_m=FACADE_M)
    add_flat(acc, inner, roof_y, MAT_FLAT_ROOF, roof_var, roof_color)
