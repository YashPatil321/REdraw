"""Low-poly but realistic vehicles: lofted bodies, projected light/grille decals, lathed wheels.

Blender frame: +Y forward, +X right (curb side), +Z up, origin on the ground at the
center of the footprint. The glTF exporter (export_yup) turns this into
**forward = -Z, up = +Y, right = +X**, i.e. a three.js object with rotation.y = 0
faces north (scene -z) and rotation.y = atan2(-dx, -dz) faces direction (dx, dz).

Bodies are built as a loft of cross-section rings along the length. Each ring is a
half profile (bottom center -> rocker -> side -> shoulder -> glass -> roof rail ->
roof center) mirrored to the left side, so the body is one closed smooth shell whose
faces get paint / glass / trim materials by zone (hood, windshield, cabin, pillar,
rear glass, deck, bed) and by ring column. Wheel arches are cut by raising the
bottom of the rings around each axle. Lights, grilles and plates are small grids
projected onto the body with a BVH ray cast, so they hug the curved nose and tail.

Real dimensions (m) are noted on each builder; triangle budgets: cars < 1500,
bus < 3000 (checked by tests).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .mesh import Part, box

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _interp(pts: Sequence[tuple[float, float]], y: float) -> float:
    """Piecewise-linear lookup in (y, value) points listed with y DESCENDING (front to rear)."""
    ys = [p[0] for p in pts][::-1]
    vs = [p[1] for p in pts][::-1]
    return float(np.interp(y, ys, vs))


def _smooth(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lathe_x(profile: Sequence[tuple[float, float]], mats: Sequence[str], n: int, cap_mat: str | None = None,
            outward_seg: int = 0) -> Part:
    """Surface of revolution about the X axis. profile = [(radius, x), ...]; mats[i] for segment i.

    The profile must walk along the surface with the solid always on the same side, so
    one winding fits every face; segment `outward_seg` (a tread-like segment) is used to
    pick it. cap_mat closes the last ring with a center fan (rim hub).
    """
    V: list[tuple[float, float, float]] = []
    for r, x in profile:
        for k in range(n):
            a = 2 * math.pi * (k + 0.5) / n
            V.append((x, r * math.cos(a), r * math.sin(a)))
    F: list[list[int]] = []
    M: list[str] = []
    for i in range(len(profile) - 1):
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            F.append([a, b, b + n, a + n])
            M.append(mats[i])
    Va = np.array(V, dtype=float)
    f0 = F[outward_seg * n]
    pts = Va[f0]
    nrm = np.cross(pts[1] - pts[0], pts[3] - pts[0])
    radial = pts.mean(axis=0) * np.array([0.0, 1.0, 1.0])
    if np.dot(nrm, radial) < 0:
        F = [list(reversed(f)) for f in F]
    if cap_mat is not None:
        r_last, x_last = profile[-1]
        sgn = 1.0 if x_last >= 0 else -1.0
        c = len(V)
        V.append((x_last + 0.012 * sgn, 0.0, 0.0))
        last = (len(profile) - 1) * n
        for k in range(n):
            f = [last + k, last + (k + 1) % n, c]
            q = np.array([V[j] for j in f])
            if np.cross(q[1] - q[0], q[2] - q[0])[0] * sgn < 0:
                f = f[::-1]
            F.append(f)
            M.append(cap_mat)
    return Part(np.array(V, dtype=float), F, M)


def wheel(r: float, w: float, rim_r: float, outward: int, rim: str = "rim", n: int = 12) -> Part:
    """Tire + dished rim, axis along X, centered at the origin. outward=+1 rim faces +X."""
    s = float(outward)
    h = w / 2
    # tire: inner sidewall -> tread -> outer sidewall -> rim lip -> dish -> hub cap
    prof = [
        (r * 0.72, -s * h),
        (r, -s * (h - 0.04)),
        (r, s * (h - 0.04)),
        (rim_r * 1.06, s * h),
        (rim_r * 0.92, s * (h - 0.012)),
        (rim_r * 0.3, s * (h - 0.05)),
    ]
    mats = ["tire", "tire", "tire", rim, rim]
    p = lathe_x(prof, mats, n, cap_mat=rim, outward_seg=1)
    # dark gaps between spokes on the dish ring (spoke look)
    dish0 = 4 * n
    for k in range(n):
        if k % 2 == 1:
            p.M[dish0 + k] = "rim_dark" if rim == "rim" else "trim_black"
    return p


# ---------------------------------------------------------------------------
# lofted body
# ---------------------------------------------------------------------------

# half-ring column indices (between ring points i and i+1)
COL_UNDER, COL_ROCKER, COL_LOWER, COL_SIDE, COL_SHOULDER, COL_GLASS, COL_FRAME, COL_ROOF = range(8)


@dataclass
class BodySpec:
    length: float
    width: float  # body width without mirrors
    wheelbase: float
    front_overhang: float
    wheel_r: float
    tire_w: float
    clearance: float  # body bottom z between the wheels
    top: list[tuple[float, float]]  # upper silhouette (y, z), front -> rear
    belt: list[tuple[float, float]]  # beltline / window sill (y, z)
    zones: list[tuple[float, float, str]]  # (y_front, y_rear, zone) zones: hood windshield cabin pillar rearglass deck bed
    glass_top_drop: float = 0.07  # window top below roof rail (painted frame strip)
    tumble: float = 0.2  # greenhouse inset at the roof (tumblehome)
    plan_front: list[tuple[float, float]] = field(default_factory=list)  # (y, half-width scale) at the nose
    plan_rear: list[tuple[float, float]] = field(default_factory=list)
    nose_lift: float = 0.1  # bottom rises toward the bumper ends (approach angle)
    tail_lift: float = 0.12
    rocker: str = "paint"  # material of the rocker / arch band (SUVs: trim_black cladding)
    roof: str = "paint"
    pillar: str = "trim_black"
    bed_floor: float | None = None  # pickups: z of the bed floor in the 'bed' zone
    extra_y: list[float] = field(default_factory=list)  # extra stations

    @property
    def axles(self) -> tuple[float, float]:
        yf = self.length / 2 - self.front_overhang
        return yf, yf - self.wheelbase


def _zone(spec: BodySpec, y: float) -> str:
    for y0, y1, z in spec.zones:
        if y1 <= y <= y0:
            return z
    return "deck"


def _stations(spec: BodySpec) -> list[float]:
    L2 = spec.length / 2
    R = spec.wheel_r + 0.06
    ys = {L2, -L2}
    for y, _ in spec.top:
        ys.add(y)
    for ax in spec.axles:
        for d in (1.025, 1.0, 0.45, -0.45, -1.0, -1.025):
            ys.add(ax + d * R)
    ys.update(spec.extra_y)
    ys.update(y for y, _ in spec.plan_front)
    ys.update(y for y, _ in spec.plan_rear)
    # zone boundaries snap to an existing station within 12 cm (saves rings)
    base = sorted(ys)
    for y0, y1, _ in spec.zones:
        for yz in (y0, y1):
            if min(abs(yz - q) for q in base) > 0.12:
                ys.add(yz)
    ys = sorted((y for y in ys if -L2 - 1e-6 <= y <= L2 + 1e-6), reverse=True)
    out: list[float] = []
    for y in ys:  # drop near-duplicates
        if not out or abs(out[-1] - y) > 0.008:
            out.append(y)
    return out


def _half_ring(spec: BodySpec, y: float) -> list[tuple[float, float]]:
    L2 = spec.length / 2
    R = spec.wheel_r + 0.06
    hw = spec.width / 2
    # plan rounding at the ends
    s = 1.0
    if spec.plan_front:
        s = min(s, _interp([(L2 + 1, spec.plan_front[0][1])] + spec.plan_front + [(-L2 - 1, 1.0)], y))
    if spec.plan_rear:
        s = min(s, _interp([(L2 + 1, 1.0)] + spec.plan_rear + [(-L2 - 1, spec.plan_rear[-1][1])], y))
    hw *= s
    ztop = _interp(spec.top, y)
    zbelt = min(_interp(spec.belt, y), ztop - 0.02)
    # bottom: arches around axles, lift toward the ends
    zb = spec.clearance
    zb += spec.nose_lift * _smooth((y - (L2 - spec.front_overhang * 0.55)) / (spec.front_overhang * 0.55))
    rear_oh = L2 + spec.axles[1]
    zb += spec.tail_lift * _smooth((-(y) - (L2 - rear_oh * 0.55)) / (rear_oh * 0.55))
    zarch = zb
    for ax in spec.axles:
        d = y - ax
        if abs(d) <= R + 1e-6:
            zarch = max(zarch, spec.wheel_r + math.sqrt(max(R * R - d * d, 0.0)))
    zone = _zone(spec, y)
    z_rocker = max(zarch + 0.05, zb + 0.16)
    z_side = max(zbelt - 0.16, z_rocker + 0.04)
    pts = [
        (0.0, zb),
        (hw - 0.05, zarch),
        (hw - 0.005, z_rocker),
        (hw, z_side),
        (hw - 0.035, zbelt),
    ]
    gh = ztop - zbelt  # greenhouse height (0 on hood / deck)
    if spec.bed_floor is not None and zone == "bed":
        f = spec.bed_floor
        pts += [(hw - 0.05, zbelt + 0.012), (hw - 0.11, zbelt + 0.012), (hw - 0.13, f), (0.0, f)]
        return pts
    t = _smooth((gh - 0.06) / 0.35)  # 0 = hood/deck ring, 1 = full greenhouse ring
    tum = spec.tumble * t
    gt = ztop - spec.glass_top_drop * t - 0.02 * t
    pts += [
        (hw - 0.06 - 0.01 * t, zbelt + 0.012 + 0.01 * t),  # window sill / hood edge
        (hw - 0.07 - tum * 0.85, max(zbelt + 0.02, gt - 0.012)),  # top of glass
        (hw - 0.08 - tum, max(zbelt + 0.025, ztop - 0.03 * t - 0.012)),  # roof rail
        (0.0, ztop),
    ]
    return pts


def _col_material(spec: BodySpec, zone: str, col: int) -> str:
    if col == COL_UNDER:
        return "underbody"
    if col == COL_ROCKER:
        return spec.rocker
    if col in (COL_LOWER, COL_SIDE, COL_SHOULDER):
        return "paint"
    if zone == "bed":
        return "paint" if col <= COL_GLASS else "bed_liner"
    if col == COL_GLASS:
        if zone == "cabin":
            return "glass"
        if zone == "pillar":
            return spec.pillar
        if zone in ("windshield", "rearglass"):
            return spec.pillar
        return "paint"
    if col == COL_FRAME:
        if zone in ("windshield", "rearglass"):
            return "glass_clear" if zone == "windshield" else "glass"
        return spec.roof if zone in ("cabin", "pillar") else "paint"
    # roof
    if zone in ("windshield",):
        return "glass_clear"
    if zone == "rearglass":
        return "glass"
    if zone in ("cabin", "pillar"):
        return spec.roof
    return "paint"


def loft_body(spec: BodySpec) -> Part:
    ys = _stations(spec)
    rings: list[list[tuple[float, float, float]]] = []
    for y in ys:
        h = _half_ring(spec, y)
        right = [(x, y, z) for x, z in h]
        left = [(-x, y, z) for x, z in h[1:-1]][::-1]
        rings.append(right + left)
    nh = len(_half_ring(spec, ys[0]))
    n = len(rings[0])
    if any(len(r) != n for r in rings):
        raise ValueError("ring sizes differ (bed zone must not be at the very ends)")
    V = np.array([p for r in rings for p in r], dtype=float)
    F: list[list[int]] = []
    M: list[str] = []
    for i in range(len(rings) - 1):
        zone = _zone(spec, (ys[i] + ys[i + 1]) / 2)
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            F.append([a, b, b + n, a + n])
            col = k if k < nh - 1 else (n - 1 - k)  # mirror column index for the left half
            M.append(_col_material(spec, zone, col))
    # end caps (n-gons; the exporter triangulates)
    F.append(list(range(n)))
    M.append("paint")
    last = (len(rings) - 1) * n
    F.append([last + k for k in range(n)][::-1])
    M.append("paint")
    p = Part(V, F, M)
    return _orient_closed(p)


def _orient_closed(p: Part) -> Part:
    """Consistent outward normals for a closed shell (bmesh recalc), keeping per-face materials."""
    bm = bmesh.new()
    vs = [bm.verts.new(v) for v in p.V]
    faces = []
    for f, m in zip(p.F, p.M, strict=True):
        fc = bm.faces.new([vs[i] for i in f])
        faces.append((fc, m))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.verts.index_update()
    F = [[v.index for v in fc.verts] for fc, _ in faces]
    M = [m for _, m in faces]
    V = np.array([v.co[:] for v in bm.verts])
    bm.free()
    return Part(V, F, M)


# ---------------------------------------------------------------------------
# projected decals (lights, grilles, plates)
# ---------------------------------------------------------------------------


class Projector:
    def __init__(self, body: Part):
        self.tree = BVHTree.FromPolygons([tuple(v) for v in body.V], body.F, all_triangles=False)

    def decal(self, quad: Sequence[tuple[float, float]], axis: str, mat: str, nu: int = 3, nv: int = 2,
              offset: float = 0.008, dirn: tuple[float, float, float] | None = None) -> Part:
        """Project a quad given in the plane perpendicular to `axis` onto the body.

        axis: '+y' (front, looking backward), '-y' (rear), '+x'/'-x' (sides), '+z' (top).
        quad: 4 corners (a, b) in the plane coords: for y axes (x, z); x axes (y, z); z axis (x, y).
        """
        sign = 1.0 if axis[0] == "+" else -1.0
        ax = axis[1]
        q = [np.array(c, dtype=float) for c in quad]
        grid = []
        for j in range(nv + 1):
            for i in range(nu + 1):
                u, v = i / nu, j / nv
                a = q[0] * (1 - u) * (1 - v) + q[1] * u * (1 - v) + q[2] * u * v + q[3] * (1 - u) * v
                grid.append(a)
        V = []
        far = 20.0
        for a, b in grid:
            if ax == "y":
                o, d = Vector((a, sign * far, b)), Vector((0, -sign, 0))
            elif ax == "x":
                o, d = Vector((sign * far, a, b)), Vector((-sign, 0, 0))
            else:
                o, d = Vector((a, b, sign * far)), Vector((0, 0, -sign))
            if dirn is not None:
                d = Vector(dirn).normalized()
                o = Vector(o) - d * 0.0  # origin already far along the axis
            hit, nrm, _, _ = self.tree.ray_cast(o, d, 2 * far)
            if hit is None:
                return Part()
            if nrm.dot(d) > 0:
                nrm = -nrm
            V.append(tuple(hit + nrm * offset))
        F = []
        for j in range(nv):
            for i in range(nu):
                a = j * (nu + 1) + i
                F.append([a, a + 1, a + nu + 2, a + nu + 1])
        p = Part(np.array(V), F, [mat] * len(F))
        # face winding toward the viewer side
        out = np.zeros(3)
        out["xyz".index(ax)] = sign
        for k, f in enumerate(p.F):
            pts = p.V[f]
            nn = np.cross(pts[1] - pts[0], pts[3] - pts[0])
            if np.dot(nn, out) < 0:
                p.F[k] = list(reversed(f))
        return p

    def pair(self, quad: Sequence[tuple[float, float]], axis: str, mat: str, **kw) -> Part:  # noqa: ANN003
        """Decal and its mirror image across x = 0 (quads given for the right side, x > 0)."""
        p = self.decal(quad, axis, mat, **kw)
        if axis[1] in "yz":
            mq = [(-a, b) for a, b in quad]
            p += self.decal(mq, axis, mat, **kw)
        else:
            p += self.decal(quad, ("-" if axis[0] == "+" else "+") + "x", mat, **kw)
        return p


def _wheels(spec: BodySpec, rim: str = "rim", dual_rear: bool = False, n: int = 10) -> Part:
    p = Part()
    hw = spec.width / 2
    r, w = spec.wheel_r, spec.tire_w
    for k, ax in enumerate(spec.axles):
        ww = w * (1.9 if (dual_rear and k == 1) else 1.0)
        for side in (1, -1):
            x = side * (hw - ww / 2 - 0.025)
            p += wheel(r, ww, r * 0.62, side, rim=rim, n=n).moved(x, ax, r)
    return p


def _mirrors(spec: BodySpec, y: float, z: float, mat: str = "paint", size: tuple[float, float, float] = (0.2, 0.09, 0.13)) -> Part:
    p = Part()
    hw = spec.width / 2
    for side in (1, -1):
        m = box(size[0], size[1], size[2], mat, bottom=True)
        p += m.moved(side * (hw + size[0] / 2 - 0.02), y, z)
        p += box(0.12, 0.05, 0.04, "trim_black").moved(side * (hw + 0.02), y + 0.01, z + 0.03)  # stalk
    return p


def _car_common(spec: BodySpec, front: dict, rear: dict, mirror_y: float, mirror_z: float,
                rim: str = "rim") -> Part:
    """Body + wheels + standard decal set. front/rear: dicts of decal quads (right side, x>0)."""
    body = loft_body(spec)
    pr = Projector(body)
    p = body
    for key, mat, kw in (("headlight", "headlight", {"nu": 3, "nv": 1}), ("drl", "drl", {"nu": 3, "nv": 1}),
                         ("fog", "trim_black", {"nu": 1, "nv": 1})):
        if key in front:
            p += pr.pair(front[key], "+y", mat, **kw)
    for key, mat in (("grille", "trim_black"), ("intake", "trim_black")):
        if key in front:
            p += pr.decal(front[key], "+y", mat, nu=4, nv=1)
    if "plate" in front:
        p += pr.decal(front["plate"], "+y", "plate", nu=1, nv=1, offset=0.012)
    if "taillight" in rear:
        p += pr.pair(rear["taillight"], "-y", "taillight", nu=3, nv=1)
    if "bar" in rear:  # full-width light bar (modern SUVs / Teslas)
        p += pr.decal(rear["bar"], "-y", "taillight", nu=4, nv=1)
    if "bumper" in rear:
        p += pr.decal(rear["bumper"], "-y", "trim_black", nu=4, nv=1)
    if "plate" in rear:
        p += pr.decal(rear["plate"], "-y", "plate", nu=1, nv=1, offset=0.012)
    if "front_bumper" in front:
        p += pr.decal(front["front_bumper"], "+y", "trim_black", nu=4, nv=1)
    p += _mirrors(spec, mirror_y, mirror_z)
    p += _wheels(spec, rim=rim)
    return p


# ---------------------------------------------------------------------------
# vehicles
# ---------------------------------------------------------------------------


def sedan() -> Part:
    """Mid-size sedan (Camry / Accord / Model 3 class): 4.88 x 1.84 x 1.44 m, wheelbase 2.83 m."""
    L2 = 2.44
    s = BodySpec(
        length=4.88, width=1.84, wheelbase=2.83, front_overhang=0.97, wheel_r=0.335, tire_w=0.225, clearance=0.17,
        top=[(L2, 0.68), (2.38, 0.78), (2.0, 0.86), (1.15, 0.95), (0.2, 1.40), (-0.25, 1.44), (-0.8, 1.40),
             (-1.7, 1.05), (-2.32, 1.03), (-L2, 0.93)],
        belt=[(L2, 0.64), (2.38, 0.74), (1.15, 0.92), (-1.7, 1.01), (-2.32, 1.0), (-L2, 0.89)],
        zones=[(1.15, 0.2, "windshield"), (0.2, -0.12, "cabin"), (-0.12, -0.24, "pillar"), (-0.24, -0.8, "cabin"),
               (-0.8, -1.7, "rearglass")],
        plan_front=[(L2, 0.8), (L2 - 0.07, 0.91), (L2 - 0.22, 0.975), (L2 - 0.5, 1.0)],
        plan_rear=[(-L2 + 0.3, 1.0), (-L2 + 0.08, 0.95), (-L2, 0.86)],
        tumble=0.21, nose_lift=0.12, tail_lift=0.16,
    )
    front = {
        "headlight": [(0.5, 0.70), (0.82, 0.715), (0.84, 0.64), (0.55, 0.625)],
        "drl": [(0.53, 0.73), (0.80, 0.745), (0.81, 0.725), (0.55, 0.715)],
        "grille": [(-0.42, 0.62), (0.42, 0.62), (0.36, 0.46), (-0.36, 0.46)],
        "intake": [(-0.55, 0.34), (0.55, 0.34), (0.5, 0.26), (-0.5, 0.26)],
        "plate": [(-0.16, 0.44), (0.16, 0.44), (0.16, 0.36), (-0.16, 0.36)],
    }
    rear = {
        "taillight": [(0.36, 0.99), (0.89, 0.97), (0.88, 0.84), (0.40, 0.88)],
        "bumper": [(-0.8, 0.42), (0.8, 0.42), (0.78, 0.33), (-0.78, 0.33)],
        "plate": [(-0.16, 0.80), (0.16, 0.80), (0.16, 0.72), (-0.16, 0.72)],
    }
    return _car_common(s, front, rear, mirror_y=1.0, mirror_z=0.9)


def suv() -> Part:
    """Compact / mid-size crossover SUV (RAV4 / CR-V / Model Y class): 4.75 x 1.88 x 1.70 m, wheelbase 2.72 m."""
    L2 = 2.375
    s = BodySpec(
        length=4.75, width=1.88, wheelbase=2.72, front_overhang=0.95, wheel_r=0.365, tire_w=0.235, clearance=0.24,
        top=[(L2, 0.84), (2.26, 1.0), (1.3, 1.12), (0.45, 1.62), (0.1, 1.70), (-1.85, 1.69),
             (-2.2, 1.6), (-2.33, 1.12), (-L2, 1.02)],
        belt=[(L2, 0.80), (2.26, 0.96), (1.3, 1.08), (-1.9, 1.15), (-2.22, 1.15), (-L2, 0.98)],
        zones=[(1.3, 0.45, "windshield"), (0.45, -0.25, "cabin"), (-0.25, -0.37, "pillar"), (-0.37, -1.35, "cabin"),
               (-1.35, -1.55, "pillar"), (-1.55, -2.2, "cabin"), (-2.2, -2.33, "rearglass")],
        plan_front=[(L2, 0.8), (L2 - 0.07, 0.91), (L2 - 0.22, 0.975), (L2 - 0.5, 1.0)],
        plan_rear=[(-L2 + 0.25, 1.0), (-L2 + 0.07, 0.95), (-L2, 0.88)],
        tumble=0.17, rocker="trim_black", nose_lift=0.12, tail_lift=0.14, glass_top_drop=0.08,
    )
    front = {
        "headlight": [(0.52, 0.93), (0.86, 0.95), (0.87, 0.87), (0.56, 0.85)],
        "grille": [(-0.48, 0.86), (0.48, 0.86), (0.42, 0.6), (-0.42, 0.6)],
        "front_bumper": [(-0.85, 0.48), (0.85, 0.48), (0.8, 0.30), (-0.8, 0.30)],
        "plate": [(-0.16, 0.58), (0.16, 0.58), (0.16, 0.5), (-0.16, 0.5)],
    }
    rear = {
        "taillight": [(0.55, 1.12), (0.89, 1.1), (0.88, 0.98), (0.6, 1.0)],
        "bumper": [(-0.86, 0.52), (0.86, 0.52), (0.84, 0.34), (-0.84, 0.34)],
        "plate": [(-0.16, 0.86), (0.16, 0.86), (0.16, 0.78), (-0.16, 0.78)],
    }
    p = _car_common(s, front, rear, mirror_y=1.15, mirror_z=1.06)
    for side in (1, -1):  # roof rails
        p += box(0.05, 1.9, 0.05, "trim_black").moved(side * 0.66, -0.8, 1.685)
    return p


def minivan() -> Part:
    """Minivan (Odyssey / Sienna / Pacifica class): 5.16 x 1.99 x 1.74 m, wheelbase 3.0 m."""
    L2 = 2.58
    s = BodySpec(
        length=5.16, width=1.99, wheelbase=3.0, front_overhang=0.98, wheel_r=0.36, tire_w=0.235, clearance=0.2,
        top=[(L2, 0.70), (2.45, 0.9), (1.65, 1.04), (0.75, 1.68), (0.4, 1.74), (-2.0, 1.74),
             (-2.44, 1.64), (-2.55, 1.08), (-L2, 0.98)],
        belt=[(L2, 0.66), (2.45, 0.86), (1.65, 1.0), (-2.1, 1.1), (-2.48, 1.1), (-L2, 0.94)],
        zones=[(1.65, 0.75, "windshield"), (0.75, 0.0, "cabin"), (0.0, -0.1, "pillar"), (-0.1, -1.25, "cabin"),
               (-1.25, -1.45, "pillar"), (-1.45, -2.44, "cabin"), (-2.44, -2.55, "rearglass")],
        plan_front=[(L2, 0.8), (L2 - 0.07, 0.91), (L2 - 0.22, 0.975), (L2 - 0.5, 1.0)],
        plan_rear=[(-L2 + 0.25, 1.0), (-L2 + 0.07, 0.95), (-L2, 0.88)],
        tumble=0.2, nose_lift=0.12, tail_lift=0.12,
    )
    front = {
        "headlight": [(0.52, 0.84), (0.9, 0.86), (0.91, 0.77), (0.58, 0.76)],
        "grille": [(-0.46, 0.74), (0.46, 0.74), (0.4, 0.56), (-0.4, 0.56)],
        "intake": [(-0.6, 0.38), (0.6, 0.38), (0.55, 0.28), (-0.55, 0.28)],
        "plate": [(-0.16, 0.52), (0.16, 0.52), (0.16, 0.44), (-0.16, 0.44)],
    }
    rear = {
        "taillight": [(0.6, 1.2), (0.94, 1.18), (0.93, 0.95), (0.7, 0.98)],
        "bumper": [(-0.9, 0.46), (0.9, 0.46), (0.88, 0.34), (-0.88, 0.34)],
        "plate": [(-0.16, 0.92), (0.16, 0.92), (0.16, 0.84), (-0.16, 0.84)],
    }
    p = _car_common(s, front, rear, mirror_y=1.45, mirror_z=0.98)
    # sliding door track (dark line along the rear quarter glass bottom)
    for side in (1, -1):
        p += box(0.02, 1.1, 0.025, "trim_black").moved(side * 0.985, -1.55, 1.08)
    return p


def pickup() -> Part:
    """Full-size crew-cab pickup (F-150 SuperCrew 5.5 ft bed class): 5.89 x 2.03 x 1.96 m, wheelbase 3.68 m."""
    L2 = 2.945
    s = BodySpec(
        length=5.89, width=2.03, wheelbase=3.68, front_overhang=1.0, wheel_r=0.42, tire_w=0.275, clearance=0.42,
        top=[(L2, 1.08), (2.84, 1.28), (2.4, 1.32), (1.9, 1.36), (1.2, 1.9), (0.95, 1.95), (-0.82, 1.94),
             (-0.9, 1.42), (-0.96, 1.31), (-L2, 1.31)],
        belt=[(L2, 1.04), (2.84, 1.25), (1.9, 1.32), (-0.9, 1.34), (-0.96, 1.30), (-L2, 1.30)],
        zones=[(1.9, 1.2, "windshield"), (1.2, 0.0, "cabin"), (0.0, -0.12, "pillar"), (-0.12, -0.82, "cabin"),
               (-0.82, -0.9, "rearglass"), (-0.97, -2.885, "bed")],
        plan_front=[(L2, 0.9), (L2 - 0.1, 0.97), (L2 - 0.3, 1.0)],
        plan_rear=[(-L2 + 0.05, 1.0), (-L2, 0.985)],
        tumble=0.16, nose_lift=0.05, tail_lift=0.08, bed_floor=0.92, extra_y=[-2.89],
    )
    body = loft_body(s)
    pr = Projector(body)
    p = body
    p += pr.pair([(0.6, 1.2), (0.95, 1.21), (0.96, 1.07), (0.62, 1.05)], "+y", "headlight", nu=3, nv=1)
    p += pr.decal([(-0.6, 1.22), (0.6, 1.22), (0.6, 0.78), (-0.6, 0.78)], "+y", "trim_black", nu=4, nv=2)  # big grille
    p += pr.decal([(-0.98, 0.72), (0.98, 0.72), (0.96, 0.5), (-0.96, 0.5)], "+y", "chrome", nu=4, nv=1)  # bumper
    p += pr.decal([(-0.16, 0.7), (0.16, 0.7), (0.16, 0.62), (-0.16, 0.62)], "+y", "plate", nu=1, nv=1, offset=0.014)
    # tail lights + bumper (the last ring closes the U-shaped bed: tailgate)
    p += box(0.12, 0.05, 0.36, "taillight").moved(0.96, -L2 + 0.005, 0.88)
    p += box(0.12, 0.05, 0.36, "taillight").moved(-0.96, -L2 + 0.005, 0.88)
    p += box(2.0, 0.16, 0.2, "chrome", bevel=0.03, segments=1).moved(0, -L2 - 0.02, 0.45)
    p += box(0.32, 0.03, 0.16, "plate").moved(0, -L2 - 0.11, 0.48)
    p += _mirrors(s, 1.6, 1.3, mat="trim_black", size=(0.22, 0.1, 0.26))
    p += _wheels(s, rim="rim")
    return p


def shuttle_van() -> Part:
    """High-roof passenger van (Ford Transit 350 HD / Sprinter class): 5.98 x 2.06 x 2.78 m, wheelbase 3.75 m.

    White body (paint tints to the fleet color), passenger window band, no badges.
    """
    L2 = 2.99
    s = BodySpec(
        length=5.98, width=2.06, wheelbase=3.75, front_overhang=0.95, wheel_r=0.36, tire_w=0.235, clearance=0.33,
        top=[(L2, 0.86), (2.9, 1.08), (2.6, 1.16), (2.2, 1.24), (1.45, 2.08), (1.2, 2.62), (0.85, 2.76),
             (-2.85, 2.78), (-2.96, 2.68), (-L2, 2.5)],
        belt=[(L2, 0.82), (2.9, 1.04), (2.2, 1.22), (1.45, 1.26), (0.9, 1.28), (-2.92, 1.32), (-L2, 1.25)],
        zones=[(2.2, 1.45, "windshield"), (1.45, 0.95, "cabin"), (0.95, 0.8, "pillar"), (0.8, -2.3, "cabin"),
               (-2.3, -2.5, "pillar"), (-2.5, -2.97, "deck")],
        plan_front=[(L2, 0.88), (L2 - 0.12, 0.96), (L2 - 0.35, 1.0)],
        plan_rear=[(-L2 + 0.06, 1.0), (-L2, 0.97)],
        tumble=0.12, glass_top_drop=0.72, nose_lift=0.08, tail_lift=0.06, extra_y=[1.2, 0.85],
    )
    front = {
        "headlight": [(0.62, 1.05), (0.96, 1.07), (0.97, 0.94), (0.66, 0.92)],
        "grille": [(-0.55, 1.02), (0.55, 1.02), (0.5, 0.7), (-0.5, 0.7)],
        "front_bumper": [(-0.98, 0.62), (0.98, 0.62), (0.95, 0.38), (-0.95, 0.38)],
        "plate": [(-0.16, 0.6), (0.16, 0.6), (0.16, 0.52), (-0.16, 0.52)],
    }
    rear = {
        "taillight": [(0.92, 1.6), (1.01, 1.6), (1.01, 0.85), (0.92, 0.85)],
        "bumper": [(-1.0, 0.62), (1.0, 0.62), (1.0, 0.4), (-1.0, 0.4)],
        "plate": [(-0.16, 0.8), (0.16, 0.8), (0.16, 0.72), (-0.16, 0.72)],
    }
    p = _car_common(s, front, rear, mirror_y=1.62, mirror_z=1.22)
    pr = Projector(loft_body(s))  # rear door glass + split line
    p += pr.pair([(0.15, 2.05), (0.85, 2.05), (0.85, 1.45), (0.15, 1.45)], "-y", "glass", nu=1, nv=1)
    p += pr.decal([(-0.012, 2.6), (0.012, 2.6), (0.012, 0.65), (-0.012, 0.65)], "-y", "trim_black", nu=1, nv=2)
    return p


def school_bus() -> Part:
    """Type C conventional school bus (Blue Bird Vision / Thomas C2 class), 72-77 passengers.

    11.6 x 2.44 x 3.2 m (body 96 in wide), wheelbase ~6.1 m, 22.5 in wheels (dual rear).
    National School Bus Glossy Yellow, black rub rails, white roof, roof warning lamps.
    """
    L, W, H = 11.6, 2.44, 3.2
    hw = W / 2
    yb_front = L / 2 - 2.05  # front of the passenger body (windshield base line)
    yb_rear = -L / 2
    zb0 = 0.62  # body bottom (skirt)
    p = Part()
    # passenger body: beveled box (rounded roof edges)
    body = box(W, yb_front - yb_rear, H - zb0, "paint_bus", top="paint_bus_roof", bevel=0.16, segments=2)
    body = body.moved(0, (yb_front + yb_rear) / 2, zb0)
    p += body
    # cowl + sloped hood (Vision style) as a small loft
    hood = BodySpec(
        length=2.2, width=2.2, wheelbase=0.1, front_overhang=0.05, wheel_r=0.05, tire_w=0.1, clearance=0.55,
        top=[(1.1, 1.05), (1.0, 1.32), (0.4, 1.55), (-0.5, 1.78), (-1.1, 1.85)],
        belt=[(1.1, 1.0), (1.0, 1.28), (0.4, 1.5), (-0.5, 1.72), (-1.1, 1.8)],
        zones=[], plan_front=[(1.1, 0.82), (0.95, 0.94), (0.7, 1.0)], nose_lift=0.0, tail_lift=0.0,
    )
    hp = loft_body(hood).recolor({"paint": "paint_bus", "underbody": "trim_black"})
    p += hp.moved(0, yb_front + 1.05, 0)
    pr = Projector(hp.moved(0, yb_front + 1.05, 0))
    yf = yb_front + 2.15
    p += pr.decal([(-0.62, 1.22), (0.62, 1.22), (0.58, 0.72), (-0.58, 0.72)], "+y", "trim_black", nu=3, nv=1)  # grille
    p += pr.pair([(0.68, 1.12), (0.98, 1.12), (0.98, 0.92), (0.68, 0.92)], "+y", "headlight", nu=1, nv=1)
    p += box(W - 0.1, 0.22, 0.3, "trim_black", bevel=0.05).moved(0, yf + 0.02, 0.42)  # bumper
    p += box(0.32, 0.03, 0.16, "plate").moved(0, yf + 0.14, 0.48)
    # windshield (two panes) and front cap above it
    p += box(W - 0.22, 0.04, 1.05, "glass_clear").moved(0, yb_front + 0.005, 1.88)
    p += box(0.08, 0.05, 1.05, "trim_black").moved(0, yb_front + 0.02, 1.88)
    # side window band: dark glass with yellow posts and a silver sash line
    y_w0, y_w1 = yb_front - 0.9, yb_rear + 0.75
    z_w0, z_w1 = 1.72, 2.55
    for side in (1, -1):
        p += box(0.03, y_w0 - y_w1, z_w1 - z_w0, "glass").moved(side * (hw + 0.005), (y_w0 + y_w1) / 2, z_w0)
        p += box(0.034, y_w0 - y_w1, 0.035, "chrome").moved(side * (hw + 0.006), (y_w0 + y_w1) / 2, (z_w0 + z_w1) / 2)
        nwin = 11
        step = (y_w0 - y_w1) / nwin
        for k in range(nwin + 1):
            p += box(0.04, 0.09, z_w1 - z_w0, "paint_bus").moved(side * (hw + 0.008), y_w1 + k * step, z_w0)
        # rub rails
        for z in (0.95, 1.35, 1.66):
            p += box(0.05, yb_front - yb_rear - 0.3, 0.07, "trim_black").moved(side * (hw + 0.01), (yb_front + yb_rear) / 2 - 0.05, z)
        # wheel well skirts (dark arches)
        for ya in (yb_front + 0.95, -2.95):
            p += box(0.03, 1.25, 0.45, "trim_black").moved(side * (hw + 0.004), ya, 0.62)
    # entrance door on the curb side (+x) behind the front wheel: glass panels
    p += box(0.03, 0.85, 1.75, "glass").moved(hw + 0.012, yb_front - 0.5, 0.75)
    p += box(0.035, 0.04, 1.75, "trim_black").moved(hw + 0.014, yb_front - 0.5, 0.75)
    # driver window (-x)
    p += box(0.03, 0.75, 0.85, "glass").moved(-hw - 0.006, yb_front - 0.45, 1.72)
    # stop arm (folded, driver side) and crossing arm
    p += box(0.05, 0.45, 0.45, "stop_arm", bevel=0.08).moved(-hw - 0.05, yb_front - 1.35, 1.75)
    p += box(0.1, 0.06, 0.06, "stop_arm").moved(0.95, yf + 0.18, 0.45)
    # rear: emergency door glass, lamps
    p += box(1.0, 0.03, 0.9, "glass").moved(0, yb_rear - 0.006, 1.65)
    p += box(0.035, 0.035, 2.2, "trim_black").moved(0, yb_rear - 0.012, 0.75)
    for sx in (-1, 1):
        p += box(0.24, 0.04, 0.24, "taillight", bevel=0.05).moved(sx * 0.95, yb_rear - 0.012, 1.15)
        p += box(0.24, 0.04, 0.24, "amber", bevel=0.05).moved(sx * 0.95, yb_rear - 0.012, 0.86)
    p += box(W - 0.2, 0.2, 0.28, "trim_black", bevel=0.05).moved(0, yb_rear - 0.08, 0.5)
    p += box(0.32, 0.03, 0.16, "plate").moved(0, yb_rear - 0.19, 0.55)
    # roof warning lamps (amber inboard, red outboard) and SCHOOL BUS sign boards
    for yy, sgn in ((yb_front + 0.02, 1), (yb_rear - 0.02, -1)):
        p += box(1.55, 0.06, 0.3, "paint_bus").moved(0, yy + sgn * 0.015, H - 0.42)
        p += box(1.2, 0.07, 0.12, "sign_black").moved(0, yy + sgn * 0.02, H - 0.33)
        for sx, m in ((-0.98, "bus_red_lamp"), (-0.78, "amber"), (0.78, "amber"), (0.98, "bus_red_lamp")):
            p += box(0.17, 0.08, 0.17, m, bevel=0.03).moved(sx, yy + sgn * 0.02, H - 0.42)
    # mirrors: side mirrors on arms + crossview mirrors on the front fenders
    for side in (1, -1):
        p += box(0.1, 0.12, 0.42, "trim_black", bevel=0.03).moved(side * (hw + 0.3), yb_front + 0.1, 1.95)
        p += box(0.3, 0.04, 0.04, "trim_black").moved(side * (hw + 0.13), yb_front + 0.1, 2.1)
        p += box(0.2, 0.2, 0.2, "trim_black", bevel=0.08).moved(side * 1.0, yf - 0.25, 1.32)
    # wheels: 22.5 in, front single, rear dual
    wr = 0.5
    spec = BodySpec(length=L, width=W, wheelbase=6.1, front_overhang=1.15, wheel_r=wr, tire_w=0.28, clearance=0.6,
                    top=[], belt=[], zones=[])
    yfa = yb_front + 1.0
    for k, ya in enumerate((yfa, yfa - 6.1)):
        ww = 0.28 * (1.9 if k == 1 else 1.0)
        for side in (1, -1):
            x = side * (hw - ww / 2 - 0.04)
            p += wheel(wr, ww, wr * 0.6, side, rim="rim", n=14).moved(x, ya, wr)
    del spec
    return p


def crossover_ev() -> Part:
    """Electric crossover (Tesla Model Y class, very common in 4S Ranch / Del Sur): 4.75 x 1.92 x 1.62 m."""
    L2 = 2.375
    s = BodySpec(
        length=4.75, width=1.92, wheelbase=2.89, front_overhang=0.86, wheel_r=0.36, tire_w=0.255, clearance=0.2,
        top=[(L2, 0.72), (2.3, 0.82), (1.9, 0.9), (1.25, 0.98), (0.35, 1.56), (-0.2, 1.62), (-1.2, 1.58),
             (-2.05, 1.2), (-2.3, 1.12), (-L2, 1.0)],
        belt=[(L2, 0.68), (2.3, 0.78), (1.25, 0.96), (-2.05, 1.08), (-2.3, 1.06), (-L2, 0.96)],
        zones=[(1.25, 0.35, "windshield"), (0.35, -0.12, "cabin"), (-0.12, -0.24, "pillar"), (-0.24, -1.2, "cabin"),
               (-1.2, -2.05, "rearglass")],
        plan_front=[(L2, 0.8), (L2 - 0.07, 0.91), (L2 - 0.22, 0.975), (L2 - 0.5, 1.0)],
        plan_rear=[(-L2 + 0.3, 1.0), (-L2 + 0.08, 0.95), (-L2, 0.86)],
        tumble=0.22, roof="glass", nose_lift=0.1, tail_lift=0.14,
    )
    front = {
        "headlight": [(0.55, 0.80), (0.86, 0.83), (0.87, 0.77), (0.6, 0.75)],
        "intake": [(-0.5, 0.36), (0.5, 0.36), (0.45, 0.28), (-0.45, 0.28)],
    }
    rear = {
        "taillight": [(0.45, 1.1), (0.92, 1.06), (0.9, 0.98), (0.5, 1.02)],
        "bumper": [(-0.85, 0.44), (0.85, 0.44), (0.83, 0.34), (-0.83, 0.34)],
        "plate": [(-0.16, 0.9), (0.16, 0.9), (0.16, 0.82), (-0.16, 0.82)],
    }
    return _car_common(s, front, rear, mirror_y=1.1, mirror_z=0.92, rim="rim_dark")


VEHICLES: dict[str, tuple[Callable[[], Part], dict]] = {
    # id: (builder, manifest info)
    "car_sedan": (sedan, {"length_m": 4.88, "width_m": 1.84, "height_m": 1.44, "share": 0.30,
                          "notes": "Mid-size sedan (Camry / Accord / Model 3 class)"}),
    "car_suv": (suv, {"length_m": 4.75, "width_m": 1.88, "height_m": 1.70, "share": 0.30,
                      "notes": "Crossover SUV (RAV4 / CR-V class)"}),
    "car_crossover_ev": (crossover_ev, {"length_m": 4.75, "width_m": 1.92, "height_m": 1.62, "share": 0.12,
                                        "notes": "Electric crossover (Model Y class), glass roof"}),
    "car_minivan": (minivan, {"length_m": 5.16, "width_m": 1.99, "height_m": 1.74, "share": 0.12,
                              "notes": "Minivan (Odyssey / Sienna class)"}),
    "car_pickup": (pickup, {"length_m": 5.89, "width_m": 2.03, "height_m": 1.96, "share": 0.16,
                            "notes": "Full-size crew-cab pickup (F-150 class)"}),
    "school_bus": (school_bus, {"length_m": 11.6, "width_m": 2.44, "height_m": 3.2, "share": 0.0,
                                "notes": "Type C conventional yellow school bus (Blue Bird Vision class)"}),
    "shuttle_van": (shuttle_van, {"length_m": 5.98, "width_m": 2.06, "height_m": 2.78, "share": 0.0,
                                  "notes": "High-roof passenger shuttle van (Transit / Sprinter class), white"}),
}
