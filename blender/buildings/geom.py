"""Pure geometry for the HD building generator (numpy + shapely, no bpy).

Plan coordinates are (x, y) = (scene x east, -scene z north): CCW rings are CCW here and in
Blender. Heights are absolute elevations (scene y / Blender Z).

Pieces of the pipeline:
- `regularize`: real footprint -> dominant axis + rectilinear ring in the local frame
  (OSM / ML traces are near-orthogonal with 0.1-0.5 m noise; real offsets above `step_m` survive).
- `max_rectangles`: maximal axis-aligned rectangles of a rectilinear polygon.
- Roofs are upper envelopes of convex "pieces" (expanded rectangle, height = min of planes).
  For hip pieces on the maximal rectangles of a rectilinear polygon the envelope IS the
  straight-skeleton hip roof (the L-infinity distance to the outline), so valleys, hips and
  ridges of L / T / U / stepped plans come out exact; gable pieces give cross gables.
- `envelope_faces`: visible planar faces of the envelope (2D polygons + plane);
  `envelope_profile`: piecewise-linear envelope height along a segment (wall tops, fascia).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient

EPS = 1e-9


# ---------------------------------------------------------------------------
# footprint regularization
# ---------------------------------------------------------------------------


def dominant_angle(ring: np.ndarray) -> float:
    """Length-weighted 4-fold circular mean of edge directions (radians, in [-pi/4, pi/4))."""
    e = np.roll(ring, -1, axis=0) - ring
    L = np.hypot(e[:, 0], e[:, 1])
    a = np.arctan2(e[:, 1], e[:, 0])
    z = np.sum(L * np.exp(4j * a))
    if abs(z) < 1e-9:
        return 0.0
    return float(np.angle(z) / 4.0)


def rot(pts: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    return np.column_stack([pts[:, 0] * c - pts[:, 1] * s, pts[:, 0] * s + pts[:, 1] * c])


def clean_ring(ring: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """Drop duplicate and collinear vertices (open ring, CCW)."""
    p = Polygon(ring)
    if not p.is_valid:
        p = shapely.make_valid(p)
        if isinstance(p, MultiPolygon) or p.geom_type != "Polygon":
            polys = [g for g in getattr(p, "geoms", [p]) if g.geom_type == "Polygon"]
            p = max(polys, key=lambda g: g.area)
    p = orient(p.simplify(tol, preserve_topology=True), 1.0)
    return np.asarray(p.exterior.coords)[:-1]


def off_axis_share(local: np.ndarray, tol_deg: float = 12.0) -> float:
    e = np.roll(local, -1, axis=0) - local
    L = np.hypot(e[:, 0], e[:, 1])
    a = np.degrees(np.arctan2(e[:, 1], e[:, 0])) % 90.0
    dev = np.minimum(a, 90.0 - a)
    return float(np.sum(L * (dev > tol_deg)) / max(np.sum(L), EPS))


def rectilinearize(local: np.ndarray, step_m: float = 0.6) -> np.ndarray | None:
    """Snap a near-orthogonal ring (local frame, CCW) to an axis-aligned ring.

    Each edge is H or V by its nearest axis; consecutive edges of one class form a run whose
    coordinate is the length-weighted mean; runs shorter than step_m are removed (their
    neighbours merge). Returns the CCW vertex ring or None."""
    n = len(local)
    if n < 4:
        return None
    e = np.roll(local, -1, axis=0) - local
    L = np.hypot(e[:, 0], e[:, 1])
    cls = (np.abs(e[:, 1]) > np.abs(e[:, 0])).astype(int)  # 0 = H (coord y), 1 = V (coord x)
    if cls.min() == cls.max():
        return None
    start = int(np.argmax(cls != np.roll(cls, 1)))
    order = [(start + k) % n for k in range(n)]
    runs: list[list[float]] = []  # [cls, coord_sum_weighted, weight]
    for i in order:
        c = int(cls[i])
        mid = (local[i] + local[(i + 1) % n]) / 2
        coord = mid[1] if c == 0 else mid[0]
        w = max(L[i], 1e-3)
        if runs and runs[-1][0] == c:
            runs[-1][1] += coord * w
            runs[-1][2] += w
        else:
            runs.append([c, coord * w, w])
    if runs[0][0] == runs[-1][0] and len(runs) > 1:
        runs[0][1] += runs[-1][1]
        runs[0][2] += runs[-1][2]
        runs.pop()
    if len(runs) % 2 or len(runs) < 4:
        return None

    def verts(rs: list[list[float]]) -> np.ndarray:
        m = len(rs)
        out = np.zeros((m, 2))
        for k in range(m):
            a, b = rs[k - 1], rs[k]
            ca, cb = a[1] / a[2], b[1] / b[2]
            # vertex k joins run k-1 and run k
            out[k] = (ca, cb) if a[0] == 1 else (cb, ca)
        return out

    for _ in range(200):
        if len(runs) <= 4:
            break
        v = verts(runs)
        m = len(runs)
        ext = np.array([np.hypot(*(v[(k + 1) % m] - v[k])) for k in range(m)])
        k = int(np.argmin(ext))
        if ext[k] >= step_m:
            break
        # remove run k: runs k-1 and k+1 (same class) merge
        a, b = (k - 1) % m, (k + 1) % m
        runs[a][1] += runs[b][1]
        runs[a][2] += runs[b][2]
        for idx in sorted({k, b}, reverse=True):
            runs.pop(idx)
        # keep alternation (a merge can leave two equal neighbours at the seam)
        merged = True
        while merged and len(runs) > 4:
            merged = False
            for j in range(len(runs)):
                if runs[j][0] == runs[j - 1][0]:
                    runs[j - 1][1] += runs[j][1]
                    runs[j - 1][2] += runs[j][2]
                    runs.pop(j)
                    merged = True
                    break
    if len(runs) % 2 or len(runs) < 4:
        return None
    v = verts(runs)
    p = Polygon(v)
    if not p.is_valid or p.area < 1.0:
        return None
    p = orient(p.simplify(0.01), 1.0)
    return np.asarray(p.exterior.coords)[:-1]


@dataclass
class Footprint:
    theta: float  # local -> plan rotation (radians)
    center: np.ndarray  # plan centroid (rotation origin)
    local: np.ndarray  # CCW ring in the local frame (rectilinear when rect=True)
    rect: bool
    iou: float
    plan_poly: Polygon  # real (cleaned) footprint in plan coords
    holes_local: list[np.ndarray] = field(default_factory=list)

    def to_plan(self, pts: np.ndarray) -> np.ndarray:
        return rot(np.asarray(pts, float).reshape(-1, 2), self.theta) + self.center

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        return rot(np.asarray(pts, float).reshape(-1, 2) - self.center, -self.theta)

    @property
    def local_poly(self) -> Polygon:
        return Polygon(self.local, self.holes_local)


def regularize(ring_plan: np.ndarray, holes_plan: list[np.ndarray] | None = None, step_m: float = 0.6,
               min_iou: float = 0.86) -> Footprint:
    ring = clean_ring(np.asarray(ring_plan, float))
    poly = Polygon(ring, [h for h in (holes_plan or []) if len(h) >= 3])
    if not poly.is_valid:
        poly = Polygon(ring)
    center = np.array(poly.centroid.coords[0])
    theta = dominant_angle(ring)
    local = rot(ring - center, -theta)
    holes_local = [rot(np.asarray(h, float) - center, -theta) for h in (holes_plan or []) if len(h) >= 3]
    lp = Polygon(local)
    best = None
    if off_axis_share(local) <= 0.35:
        for st in (step_m, step_m * 1.5, step_m * 2.5):
            r = rectilinearize(local, st)
            if r is None:
                continue
            rp = Polygon(r)
            inter = rp.intersection(lp).area
            iou = inter / max(rp.union(lp).area, EPS)
            if iou >= min_iou:
                best = (r, iou)
                break
    if best is None:
        return Footprint(theta, center, local, False, 1.0, poly, [])
    hl = []
    for h in holes_local:
        hp = Polygon(h)
        if not hp.is_valid or hp.area < 4.0:
            continue
        r = rectilinearize(np.asarray(orient(hp, 1.0).exterior.coords)[:-1], step_m)
        if r is not None and Polygon(best[0]).buffer(-0.3).contains(Polygon(r)):
            hl.append(r[::-1])
    return Footprint(theta, center, best[0], True, best[1], poly, hl)


# ---------------------------------------------------------------------------
# rectangles
# ---------------------------------------------------------------------------


def _uniq(v: np.ndarray, tol: float = 0.02) -> np.ndarray:
    v = np.sort(v)
    out = [v[0]]
    for x in v[1:]:
        if x - out[-1] > tol:
            out.append(x)
    return np.array(out)


def max_rectangles(poly: Polygon, max_n: int = 10, min_w: float = 0.0) -> list[tuple[float, float, float, float]]:
    """Maximal axis-aligned rectangles (x0, y0, x1, y1) of a rectilinear polygon, largest first.

    If there are more than max_n, a greedy cover (largest new area first) is returned."""
    coords = np.vstack([np.asarray(poly.exterior.coords)] + [np.asarray(r.coords) for r in poly.interiors])
    xs, ys = _uniq(coords[:, 0]), _uniq(coords[:, 1])
    nx, ny = len(xs) - 1, len(ys) - 1
    if nx < 1 or ny < 1:
        return []
    cx = (xs[:-1] + xs[1:]) / 2
    cy = (ys[:-1] + ys[1:]) / 2
    gx, gy = np.meshgrid(cx, cy)
    inside = shapely.contains_xy(poly, gx, gy)  # (ny, nx)
    rects: list[tuple[int, int, int, int]] = []
    for j0 in range(ny):
        acc = np.ones(nx, bool)
        for j1 in range(j0, ny):
            acc &= inside[j1]
            if not acc.any():
                break
            # runs of True in acc
            i = 0
            while i < nx:
                if not acc[i]:
                    i += 1
                    continue
                i0 = i
                while i < nx and acc[i]:
                    i += 1
                i1 = i - 1
                # maximal vertically?
                down = j0 > 0 and inside[j0 - 1, i0 : i1 + 1].all()
                up = j1 < ny - 1 and inside[j1 + 1, i0 : i1 + 1].all()
                if not down and not up:
                    rects.append((i0, j0, i1, j1))
    out = [(float(xs[a]), float(ys[b]), float(xs[c + 1]), float(ys[d + 1])) for a, b, c, d in rects]
    out = [r for r in out if min(r[2] - r[0], r[3] - r[1]) >= min_w] or out
    out.sort(key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
    if len(out) <= max_n:
        return out
    # greedy cover
    chosen: list[tuple[float, float, float, float]] = []
    covered = Polygon()
    target = poly.area
    pool = list(out)
    while pool and covered.area < target * 0.995 and len(chosen) < max_n:
        best = max(pool, key=lambda r: box(*r).difference(covered).area)
        gain = box(*best).difference(covered).area
        if gain < 0.5:
            break
        chosen.append(best)
        covered = covered.union(box(*best))
        pool.remove(best)
    return chosen


# ---------------------------------------------------------------------------
# roof pieces and the envelope
# ---------------------------------------------------------------------------


@dataclass
class Piece:
    """Convex region with height = min over planes z = a x + b y + c (local frame)."""

    region: np.ndarray  # (k, 2) CCW convex polygon (local frame)
    planes: np.ndarray  # (m, 3) rows (a, b, c)
    kind: str = "hip"
    tag: int = 0  # free use (e.g. which volume)

    def height(self, pts: np.ndarray) -> np.ndarray:
        return np.min(pts @ self.planes[:, :2].T + self.planes[:, 2], axis=1)


def rect_piece(r: tuple[float, float, float, float], z_wall: float, tanp: float, kind: str, ov: float,
               ridge_axis: int | None = None) -> Piece:
    """Roof piece over rectangle r (local). z_wall = roof plane height at the wall line.

    hip: 4 planes; gable: 2 planes with the ridge parallel to ridge_axis (0 = x, 1 = y; default
    the long side); shed: one plane rising from y0 to y1 (long side low); flat: one level plane."""
    x0, y0, x1, y1 = r
    if ridge_axis is None:
        ridge_axis = 0 if (x1 - x0) >= (y1 - y0) else 1
    P = []
    if kind == "flat":
        P = [(0.0, 0.0, z_wall)]
    elif kind == "shed":
        if ridge_axis == 0:
            P = [(0.0, tanp, z_wall - tanp * y0)]
        else:
            P = [(tanp, 0.0, z_wall - tanp * x0)]
    else:
        if kind == "hip" or ridge_axis == 1:
            P += [(tanp, 0.0, z_wall - tanp * x0), (-tanp, 0.0, z_wall + tanp * x1)]
        if kind == "hip" or ridge_axis == 0:
            P += [(0.0, tanp, z_wall - tanp * y0), (0.0, -tanp, z_wall + tanp * y1)]
    reg = np.array([[x0 - ov, y0 - ov], [x1 + ov, y0 - ov], [x1 + ov, y1 + ov], [x0 - ov, y1 + ov]], float)
    return Piece(reg, np.array(P, float), kind)


def clip_halfplane(poly: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Keep a*x + b*y + c >= 0 (Sutherland-Hodgman on a convex polygon)."""
    if len(poly) == 0:
        return poly
    d = poly[:, 0] * a + poly[:, 1] * b + c
    if np.all(d >= -1e-12):
        return poly
    if np.all(d < 1e-12):
        return poly[:0]
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp, dq = d[i], d[(i + 1) % n]
        if dp >= 0:
            out.append(p)
        if (dp >= 0) != (dq >= 0):
            t = dp / (dp - dq)
            out.append(p + t * (q - p))
    return np.array(out) if len(out) >= 3 else poly[:0]


def poly_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


@dataclass
class Face:
    poly: Polygon | MultiPolygon  # visible part (local 2D)
    plane: np.ndarray  # (a, b, c)
    piece: int
    kind: str


def envelope_faces(pieces: list[Piece], eps: float = 2e-3, min_area: float = 0.02) -> list[Face]:
    """Visible faces of max_i(min_j plane_ij) over the union of piece regions."""
    out: list[Face] = []
    for i, pc in enumerate(pieces):
        for j, pl in enumerate(pc.planes):
            F = pc.region.copy()
            for k, pk in enumerate(pc.planes):
                if k == j:
                    continue
                F = clip_halfplane(F, *(pk - pl))
                if len(F) == 0:
                    break
            if poly_area(F) < min_area:
                continue
            occ = []
            for i2, p2 in enumerate(pieces):
                if i2 == i:
                    continue
                C = p2.region.copy()
                # quick reject: bounding boxes
                if C[:, 0].max() < F[:, 0].min() or C[:, 0].min() > F[:, 0].max() or C[:, 1].max() < F[:, 1].min() or C[:, 1].min() > F[:, 1].max():
                    continue
                off = eps if i2 > i else -eps  # earlier pieces win ties
                for pk in p2.planes:
                    d = pk - pl
                    C = clip_halfplane(C, d[0], d[1], d[2] - off)
                    if len(C) == 0:
                        break
                if poly_area(C) > 1e-4:
                    occ.append(Polygon(C))
            fp = Polygon(F)
            if occ:
                fp = fp.difference(shapely.union_all(occ))
                fp = _clean_poly(fp, min_area)
                if fp is None:
                    continue
            out.append(Face(fp, pl, i, pc.kind))
    return out


def _clean_poly(g: Polygon | MultiPolygon, min_area: float) -> Polygon | MultiPolygon | None:
    if g.is_empty:
        return None
    parts = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area >= min_area]
    if not parts:
        return None
    parts = [p.buffer(0) for p in parts]
    parts = [p for p in parts if not p.is_empty and p.geom_type == "Polygon"]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def envelope_height(pieces: list[Piece], pts: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Envelope height at local points (nan outside every region)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    best = np.full(len(pts), -np.inf)
    for pc in pieces:
        R = pc.region
        inside = np.ones(len(pts), bool)
        n = len(R)
        for k in range(n):
            p, q = R[k], R[(k + 1) % n]
            cr = (q[0] - p[0]) * (pts[:, 1] - p[1]) - (q[1] - p[1]) * (pts[:, 0] - p[0])
            inside &= cr >= -tol * max(1.0, float(np.hypot(*(q - p))))
        if inside.any():
            h = pc.height(pts[inside])
            best[inside] = np.maximum(best[inside], h)
    best[~np.isfinite(best)] = np.nan
    return best


def envelope_profile(pieces: list[Piece], a: np.ndarray, b: np.ndarray, inward: np.ndarray | None = None,
                     probe: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear envelope height along segment a->b: (t in [0,1], z), jumps as repeated t.

    inward: unit vector; the envelope is probed `probe` m to that side (fascia on the outline)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    L = float(np.hypot(*d))
    if L < 1e-6:
        return np.array([0.0, 1.0]), envelope_height(pieces, np.vstack([a, b]))
    sh = (inward * probe) if inward is not None else np.zeros(2)
    ts = [0.0, 1.0]
    lines = []  # (alpha, beta) of each plane along the segment, restricted to the piece interval
    for pc in pieces:
        # interval of the segment inside the convex region
        t0, t1 = 0.0, 1.0
        R = pc.region
        n = len(R)
        ok = True
        for k in range(n):
            p, q = R[k], R[(k + 1) % n]
            # inside: cross(q-p, x-p) >= 0 ; x = a + sh + t d
            e = q - p
            f0 = e[0] * (a[1] + sh[1] - p[1]) - e[1] * (a[0] + sh[0] - p[0])
            f1 = e[0] * d[1] - e[1] * d[0]
            if abs(f1) < 1e-12:
                if f0 < -1e-9:
                    ok = False
                    break
                continue
            tc = -f0 / f1
            if f1 > 0:
                t0 = max(t0, tc)
            else:
                t1 = min(t1, tc)
        if not ok or t1 <= t0:
            continue
        ts += [t0, t1]
        for pl in pc.planes:
            alpha = pl[0] * (a[0] + sh[0]) + pl[1] * (a[1] + sh[1]) + pl[2]
            beta = pl[0] * d[0] + pl[1] * d[1]
            lines.append((alpha, beta))
    m = len(lines)
    if m > 1:
        A = np.array(lines)
        for i in range(m):
            db = A[i, 1] - A[i + 1 :, 1]
            da = A[i + 1 :, 0] - A[i, 0]
            ok = np.abs(db) > 1e-12
            tt = da[ok] / db[ok]
            ts += [t for t in tt if 0.0 < t < 1.0]
    ts = np.unique(np.clip(np.array(ts), 0.0, 1.0))
    # merge near-duplicates
    keep = [ts[0]]
    for t in ts[1:]:
        if (t - keep[-1]) * L > 1e-4:
            keep.append(t)
    ts = np.array(keep)
    dt = 1e-6 / max(L, 1e-6)
    left = envelope_height(pieces, a + sh + np.outer(np.clip(ts - dt, 0, 1), d))
    right = envelope_height(pieces, a + sh + np.outer(np.clip(ts + dt, 0, 1), d))
    at = envelope_height(pieces, a + sh + np.outer(ts, d))
    T, Z = [], []
    for k, t in enumerate(ts):
        zl = left[k] if k > 0 else at[k]
        zr = right[k] if k < len(ts) - 1 else at[k]
        if np.isnan(zl):
            zl = at[k]
        if np.isnan(zr):
            zr = at[k]
        if abs(zl - zr) > 1e-4:
            T += [t, t]
            Z += [zl, zr]
        else:
            T.append(t)
            Z.append(zl if not np.isnan(zl) else zr)
    T_, Z_ = np.array(T), np.array(Z)
    # drop collinear interior points
    if len(T_) > 2:
        keepm = [0]
        for k in range(1, len(T_) - 1):
            t0, z0 = T_[keepm[-1]], Z_[keepm[-1]]
            t1, z1 = T_[k], Z_[k]
            t2, z2 = T_[k + 1], Z_[k + 1]
            if t1 == t0 or t2 == t1:
                keepm.append(k)
                continue
            if abs((z1 - z0) / (t1 - t0) - (z2 - z1) / (t2 - t1)) * L > 1e-4 * L * L:
                keepm.append(k)
        keepm.append(len(T_) - 1)
        T_, Z_ = T_[keepm], Z_[keepm]
    return T_, Z_


# ---------------------------------------------------------------------------
# triangulation
# ---------------------------------------------------------------------------


def triangulate(g: Polygon | MultiPolygon) -> np.ndarray:
    """(t, 3, 2) CCW triangles of a polygon (holes ok)."""
    tris = []
    for p in getattr(g, "geoms", [g]):
        if p.is_empty or p.area < 1e-6:
            continue
        ext = np.asarray(p.exterior.coords)[:-1]
        if not p.interiors and len(ext) <= 8 and _is_convex(ext):
            ext = ext if poly_area(ext) > 0 else ext[::-1]
            for k in range(1, len(ext) - 1):
                tris.append([ext[0], ext[k], ext[k + 1]])
            continue
        cdt = shapely.constrained_delaunay_triangles(p)
        for t in getattr(cdt, "geoms", []):
            c = np.asarray(t.exterior.coords)[:3]
            if abs(poly_area(c)) < 1e-9:
                continue
            if poly_area(c) < 0:
                c = c[::-1]
            tris.append(c)
    return np.array(tris, float).reshape(-1, 3, 2)


def _is_convex(p: np.ndarray) -> bool:
    n = len(p)
    if n <= 3:
        return True
    s = 0
    for i in range(n):
        a, b, c = p[i], p[(i + 1) % n], p[(i + 2) % n]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cr) < 1e-9:
            continue
        sg = 1 if cr > 0 else -1
        if s == 0:
            s = sg
        elif sg != s:
            return False
    return True


def rect_union_outline(rects: list[tuple[float, float, float, float]], ov: float) -> Polygon | MultiPolygon:
    return shapely.union_all([box(r[0] - ov, r[1] - ov, r[2] + ov, r[3] + ov) for r in rects]).simplify(1e-4)
