"""Tiny procedural modeling kit on top of numpy + bmesh.

Geometry is accumulated as `Part`s (vertices, polygon index lists, palette key
per polygon). Primitives that need bevels are built in a throwaway bmesh and
converted to a Part. Everything ends up as ONE Blender mesh per asset with one
shared palette material (see palette.py and bl.py).

Blender axes are used throughout: +X east, +Y north, +Z up. The glTF exporter
(export_yup) maps that to glTF x east, y up, z south, which is the scene
convention in docs/coordinates.md.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import bmesh
import numpy as np
from mathutils import Matrix, Vector

Vec2 = tuple[float, float]


@dataclass
class Part:
    V: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    F: list[list[int]] = field(default_factory=list)
    M: list[str] = field(default_factory=list)
    # Optional per-face UVs (one (u, v) per face corner) or None (palette / flat material).
    U: list[list[tuple[float, float]] | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.U) < len(self.F):
            self.U = list(self.U) + [None] * (len(self.F) - len(self.U))

    # ---- composition ------------------------------------------------------
    def add(self, other: Part) -> Part:
        off = len(self.V)
        self.V = np.vstack([self.V, other.V]) if len(self.V) else other.V.copy()
        self.F.extend([[i + off for i in f] for f in other.F])
        self.M.extend(other.M)
        self.U.extend(other.U if len(other.U) == len(other.F) else [None] * len(other.F))
        return self

    def __iadd__(self, other: Part) -> Part:
        return self.add(other)

    def copy(self) -> Part:
        return Part(self.V.copy(), [list(f) for f in self.F], list(self.M), [None if u is None else list(u) for u in self.U])

    # ---- transforms (return new parts) ---------------------------------------
    def moved(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Part:
        p = self.copy()
        p.V = p.V + np.array([x, y, z])
        return p

    def scaled(self, sx: float, sy: float | None = None, sz: float | None = None) -> Part:
        p = self.copy()
        p.V = p.V * np.array([sx, sx if sy is None else sy, sx if sz is None else sz])
        if (sx * (sx if sy is None else sy) * (sx if sz is None else sz)) < 0:
            p.F = [list(reversed(f)) for f in p.F]
            p.U = [None if u is None else list(reversed(u)) for u in p.U]
        return p

    def rotated_z(self, deg: float) -> Part:
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        p = self.copy()
        p.V = p.V @ R.T
        return p

    def rotated_x(self, deg: float) -> Part:
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
        p = self.copy()
        p.V = p.V @ R.T
        return p

    def rotated_y(self, deg: float) -> Part:
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
        p = self.copy()
        p.V = p.V @ R.T
        return p

    def placed(self, x: float, y: float, z: float = 0.0, rot: float = 0.0, s: float = 1.0) -> Part:
        p = self
        if s != 1.0:
            p = p.scaled(s)
        if rot:
            p = p.rotated_z(rot)
        return p.moved(x, y, z)

    def recolor(self, mapping: dict[str, str]) -> Part:
        p = self.copy()
        p.M = [mapping.get(m, m) for m in p.M]
        return p

    def tri_count(self) -> int:
        return sum(len(f) - 2 for f in self.F)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.V.min(axis=0), self.V.max(axis=0)


def merge(parts: Iterable[Part]) -> Part:
    out = Part()
    for p in parts:
        out.add(p)
    return out


# ---------------------------------------------------------------------------
# bmesh helpers
# ---------------------------------------------------------------------------


def _bm_to_part(bm: bmesh.types.BMesh, mat_of_face) -> Part:  # noqa: ANN001
    bm.verts.index_update()
    V = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    F, M = [], []
    for f in bm.faces:
        F.append([v.index for v in f.verts])
        M.append(mat_of_face(f))
    bm.free()
    return Part(V, F, M)


def _bevel(bm: bmesh.types.BMesh, width: float, segments: int = 1, edges=None) -> None:  # noqa: ANN001
    if width <= 0:
        return
    es = list(bm.edges) if edges is None else edges
    bmesh.ops.bevel(
        bm,
        geom=es,
        offset=width,
        offset_type="OFFSET",
        segments=segments,
        profile=0.5,
        affect="EDGES",
        clamp_overlap=True,
    )


def _classify(f, top: str, side: str, bottom: str | None, z_top: float, z_bot: float) -> str:  # noqa: ANN001
    n = f.normal
    c = f.calc_center_median()
    if n.z > 0.95 and abs(c.z - z_top) < 1e-3:
        return top
    if bottom is not None and n.z < -0.95 and abs(c.z - z_bot) < 1e-3:
        return bottom
    return side


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def box(
    sx: float,
    sy: float,
    sz: float,
    mat: str,
    top: str | None = None,
    bevel: float = 0.0,
    segments: int = 1,
    bottom: bool = False,
) -> Part:
    """Axis aligned box, bottom-center at the origin. `bottom=False` drops the bottom face."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    bmesh.ops.translate(bm, vec=(0, 0, sz / 2), verts=bm.verts)
    if bevel > 0:
        _bevel(bm, min(bevel, 0.45 * min(sx, sy, sz)), segments)
    bm.normal_update()
    if not bottom:
        kill = [f for f in bm.faces if f.normal.z < -0.95 and f.calc_center_median().z < 1e-4]
        bmesh.ops.delete(bm, geom=kill, context="FACES_ONLY")
    t = top or mat
    return _bm_to_part(bm, lambda f: _classify(f, t, mat, None, sz, 0.0))


def slab(x0: float, y0: float, x1: float, y1: float, z0: float, z1: float, mat: str, side: str | None = None) -> Part:
    """Axis aligned slab between corners, no bottom face, no bevel (ground layers, stripes)."""
    p = box(abs(x1 - x0), abs(y1 - y0), z1 - z0, side or mat, top=mat)
    return p.moved((x0 + x1) / 2, (y0 + y1) / 2, z0)


def quad(x0: float, y0: float, x1: float, y1: float, z: float, mat: str) -> Part:
    V = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], dtype=float)
    return Part(V, [[0, 1, 2, 3]], [mat])


def prism(
    poly: Sequence[Vec2],
    h: float,
    mat: str,
    top: str | None = None,
    bevel: float = 0.0,
    z0: float = 0.0,
    bottom: bool = False,
) -> Part:
    """Extrude a 2D polygon (CCW, any simple shape) from z0 to z0+h."""
    bm = bmesh.new()
    vs = [bm.verts.new((x, y, 0.0)) for x, y in poly]
    f = bm.faces.new(vs)
    f.normal_update()
    if f.normal.z < 0:
        f.normal_flip()
    res = bmesh.ops.extrude_face_region(bm, geom=[f])
    nv = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=(0, 0, h), verts=nv)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if bevel > 0:
        _bevel(bm, bevel)
    bm.normal_update()
    if not bottom:
        kill = [fc for fc in bm.faces if fc.normal.z < -0.95 and fc.calc_center_median().z < 1e-4]
        bmesh.ops.delete(bm, geom=kill, context="FACES_ONLY")
    t = top or mat
    return _bm_to_part(bm, lambda fc: _classify(fc, t, mat, None, h, 0.0)).moved(0, 0, z0)


def polygon(poly: Sequence[Vec2], z: float, mat: str) -> Part:
    V = np.array([[x, y, z] for x, y in poly], dtype=float)
    return Part(V, [list(range(len(poly)))], [mat])


def cylinder(
    r: float,
    h: float,
    mat: str,
    n: int = 8,
    top: str | None = None,
    r_top: float | None = None,
    caps: bool = True,
    bevel: float = 0.0,
) -> Part:
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=caps, cap_tris=False, segments=n, radius1=r, radius2=r if r_top is None else r_top, depth=h
    )
    bmesh.ops.translate(bm, vec=(0, 0, h / 2), verts=bm.verts)
    if bevel > 0:
        rim = [e for e in bm.edges if all(abs(v.co.z - h) < 1e-4 or abs(v.co.z) < 1e-4 for v in e.verts) and abs(e.verts[0].co.z - e.verts[1].co.z) < 1e-5]
        _bevel(bm, bevel, 1, rim)
    bm.normal_update()
    if caps:
        kill = [f for f in bm.faces if f.normal.z < -0.95 and f.calc_center_median().z < 1e-4]
        bmesh.ops.delete(bm, geom=kill, context="FACES_ONLY")
    t = top or mat
    return _bm_to_part(bm, lambda f: _classify(f, t, mat, None, h, 0.0))


def cylinder_x(r: float, length: float, mat: str, n: int = 8, side_mat: str | None = None) -> Part:
    """Cylinder along X (wheels), centered at the origin. `side_mat` colors the caps."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=n, radius1=r, radius2=r, depth=length)
    bm.normal_update()
    caps = side_mat or mat
    p = _bm_to_part(bm, lambda f: caps if abs(f.normal.z) > 0.95 else mat)
    return p.rotated_y(90)


def ico(r: float, mat: str, subdiv: int = 1, jitter: float = 0.0, rng: np.random.Generator | None = None,
        squash: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Part:
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=r)
    p = _bm_to_part(bm, lambda f: mat)
    if jitter > 0:
        g = rng or np.random.default_rng(0)
        p.V = p.V + g.normal(0, jitter * r, p.V.shape)
    p.V = p.V * np.array(squash)
    return p


def cone(r: float, h: float, mat: str, n: int = 7, base: bool = True) -> Part:
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=base, cap_tris=False, segments=n, radius1=r, radius2=0.0, depth=h)
    bmesh.ops.translate(bm, vec=(0, 0, h / 2), verts=bm.verts)
    return _bm_to_part(bm, lambda f: mat)


def gable_roof(w: float, d: float, h: float, mat: str, gable: str, overhang: float = 0.4, thick: float = 0.2,
               fascia: str = "trim_white") -> Part:
    """Gable roof shell, ridge along X, eaves at z=0, centered on the origin. Includes gable walls."""
    W, D = w / 2 + overhang, d / 2 + overhang
    k = thick * math.hypot(D, h) / D  # vertical thickness at the ridge
    V = [
        (-W, -D, 0), (W, -D, 0), (W, 0, h), (-W, 0, h), (W, D, 0), (-W, D, 0),  # 0-5 outer
        (-W, -D, -thick), (W, -D, -thick), (W, 0, h - k), (-W, 0, h - k), (W, D, -thick), (-W, D, -thick),  # 6-11
    ]
    F = [
        [0, 1, 2, 3], [3, 2, 4, 5],  # outer slopes
        [9, 8, 7, 6], [11, 10, 8, 9],  # undersides
        [6, 7, 1, 0], [4, 10, 11, 5],  # eave fascia
        [6, 0, 3, 5, 11, 9][::-1], [7, 8, 10, 4, 2, 1][::-1],  # end caps (fixed below)
    ]
    M = [mat, mat, fascia, fascia, fascia, fascia, fascia, fascia]
    p = Part(np.array(V, dtype=float), F, M)
    _orient_outward(p, [6, 7])
    hw, hd = w / 2, d / 2
    zt = h - k
    west = Part(np.array([(-hw, -hd, 0), (-hw, 0, zt), (-hw, hd, 0)], dtype=float), [[0, 1, 2]], [gable])
    p.add(west)
    p.add(Part(np.array([(hw, -hd, 0), (hw, hd, 0), (hw, 0, zt)], dtype=float), [[0, 1, 2]], [gable]))
    return p


def _orient_outward(p: Part, faces: Sequence[int]) -> None:
    """Flip the given faces if their normal points toward the part's centroid."""
    c = p.V.mean(axis=0)
    for i in faces:
        f = p.F[i]
        pts = p.V[f]
        n = np.zeros(3)
        for a in range(len(f)):  # Newell normal
            u, v = pts[a], pts[(a + 1) % len(f)]
            n += np.array([(u[1] - v[1]) * (u[2] + v[2]), (u[2] - v[2]) * (u[0] + v[0]), (u[0] - v[0]) * (u[1] + v[1])])
        if np.dot(n, pts.mean(axis=0) - c) < 0:
            p.F[i] = list(reversed(f))


def hip_roof(w: float, d: float, h: float, mat: str, overhang: float = 0.5, fascia: str = "trim_white",
             fascia_h: float = 0.25) -> Part:
    """Hip roof over a w x d rectangle (ridge along the longer axis), eaves at z=0."""
    W, D = w / 2 + overhang, d / 2 + overhang
    if W >= D:
        r = W - D  # ridge half length
        top = [(-r, 0, h), (r, 0, h)]
        V = [(-W, -D, 0), (W, -D, 0), (W, D, 0), (-W, D, 0), *top]
        F = [[0, 1, 5, 4], [1, 2, 5], [2, 3, 4, 5], [3, 0, 4]]
    else:
        r = D - W
        top = [(0, -r, h), (0, r, h)]
        V = [(-W, -D, 0), (W, -D, 0), (W, D, 0), (-W, D, 0), *top]
        F = [[0, 1, 4], [1, 2, 5, 4], [2, 3, 5], [3, 0, 4, 5]]
    p = Part(np.array(V, dtype=float), F, [mat] * 4)
    # fascia band
    ring = [(-W, -D), (W, -D), (W, D), (-W, D)]
    p.add(band(ring, 0.0 - fascia_h, fascia_h, fascia))
    p.add(polygon(ring[::-1], -fascia_h, fascia))
    return p


def band(ring: Sequence[Vec2], z0: float, h: float, mat: str) -> Part:
    """Vertical outward-facing wall strip along a closed CCW ring (no caps)."""
    n = len(ring)
    V = [(x, y, z0) for x, y in ring] + [(x, y, z0 + h) for x, y in ring]
    F = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return Part(np.array(V, dtype=float), F, [mat] * n)


def annulus(outer: Sequence[Vec2], inner: Sequence[Vec2], z: float, mat: str) -> Part:
    """Flat ring between two closed CCW polylines with equal vertex counts."""
    n = len(outer)
    assert len(inner) == n
    V = [(x, y, z) for x, y in outer] + [(x, y, z) for x, y in inner]
    F = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return Part(np.array(V, dtype=float), F, [mat] * n)


def ring_slab(outer: Sequence[Vec2], inner: Sequence[Vec2], z0: float, z1: float, mat: str, side: str | None = None) -> Part:
    p = annulus(outer, inner, z1, mat)
    p.add(band(outer, z0, z1 - z0, side or mat))
    inner_band = band(inner, z0, z1 - z0, side or mat)
    inner_band.F = [list(reversed(f)) for f in inner_band.F]
    p.add(inner_band)
    return p


def stadium(straight: float, radius: float, n_arc: int = 12) -> list[Vec2]:
    """CCW stadium (running track) outline centered at origin, straights along X."""
    pts: list[Vec2] = []
    hs = straight / 2
    for i in range(n_arc + 1):  # east arc, -90 -> 90 deg
        a = -math.pi / 2 + math.pi * i / n_arc
        pts.append((hs + radius * math.cos(a), radius * math.sin(a)))
    for i in range(n_arc + 1):  # west arc, 90 -> 270
        a = math.pi / 2 + math.pi * i / n_arc
        pts.append((-hs + radius * math.cos(a), radius * math.sin(a)))
    return pts


def rounded_rect(w: float, d: float, r: float, n: int = 4) -> list[Vec2]:
    pts: list[Vec2] = []
    cs = [(w / 2 - r, d / 2 - r, 0), (-w / 2 + r, d / 2 - r, 90), (-w / 2 + r, -d / 2 + r, 180), (w / 2 - r, -d / 2 + r, 270)]
    for cx, cy, a0 in cs:
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def circle(r: float, n: int = 16, cx: float = 0.0, cy: float = 0.0) -> list[Vec2]:
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def strip(points: Sequence[Vec2], width: float, z: float, mat: str, closed: bool = False) -> Part:
    """Flat ribbon along a 2D polyline (paths, painted lines)."""
    pts = [np.array(p, dtype=float) for p in points]
    n = len(pts)
    left, right = [], []
    for i in range(n):
        if closed:
            a, b = pts[i - 1], pts[(i + 1) % n]
        else:
            a, b = pts[max(i - 1, 0)], pts[min(i + 1, n - 1)]
        t = b - a
        t = t / (np.linalg.norm(t) + 1e-9)
        nrm = np.array([-t[1], t[0]])
        left.append(pts[i] + nrm * width / 2)
        right.append(pts[i] - nrm * width / 2)
    V = [(p[0], p[1], z) for p in right] + [(p[0], p[1], z) for p in left]
    m = n if closed else n - 1
    F = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(m)]
    return Part(np.array(V, dtype=float), F, [mat] * m)


def tube(points: Sequence[tuple[float, float, float]], radii: Sequence[float], mat: str, n: int = 5,
         cap: bool = True) -> Part:
    """Tapered tube along a 3D polyline (trunks, branches, palm fronds' rachis)."""
    P = [Vector(p) for p in points]
    V: list[tuple[float, float, float]] = []
    F: list[list[int]] = []
    prev_x = None
    for i, p in enumerate(P):
        t = (P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]).normalized()
        ref = Vector((1, 0, 0)) if abs(t.x) < 0.9 else Vector((0, 1, 0))
        x = (prev_x - t * prev_x.dot(t)).normalized() if prev_x is not None else t.cross(ref).normalized()
        prev_x = x
        y = t.cross(x)
        for k in range(n):
            a = 2 * math.pi * k / n
            q = p + (x * math.cos(a) + y * math.sin(a)) * radii[i]
            V.append(q[:])
    for i in range(len(P) - 1):
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            F.append([a, b, b + n, a + n])
    if cap:
        last = len(P) - 1
        F.append([last * n + k for k in range(n)])
    return Part(np.array(V, dtype=float), F, [mat] * len(F))


def foliage_blob(r: float, mat: str, rng: np.random.Generator, squash=(1.0, 1.0, 0.8), jitter: float = 0.12,
                 subdiv: int = 1, flat_bottom: float = 0.35) -> Part:
    """Faceted canopy clump: jittered icosphere with a slightly flattened underside."""
    p = ico(r, mat, subdiv=subdiv, jitter=jitter, rng=rng, squash=squash)
    zmin = -r * squash[2] * (1 - flat_bottom)
    p.V[:, 2] = np.maximum(p.V[:, 2], zmin + (p.V[:, 2] - zmin) * 0.25)
    return p


def matrix_part(p: Part, m: Matrix) -> Part:
    q = p.copy()
    M = np.array(m)
    hom = np.hstack([q.V, np.ones((len(q.V), 1))])
    q.V = (hom @ M.T)[:, :3]
    return q
