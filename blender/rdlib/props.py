"""Instanced prop library: trees, shrubs, street lamp, vehicles.

Every builder returns a `Part` with its origin on the ground at the footprint
center. Vehicles face +Y in Blender (= -z in glTF/three.js = north at
rot_y = 0). Seeds are fixed so rebuilding gives identical meshes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import bmesh
import numpy as np

from .mesh import (
    Part,
    box,
    cylinder,
    cylinder_x,
    foliage_blob,
    merge,
    tube,
)

# ---------------------------------------------------------------------------
# trees
# ---------------------------------------------------------------------------


def _limb(p0: Sequence[float], p1: Sequence[float], r0: float, r1: float, mat: str, bend: float = 0.25,
          n: int = 5, segs: int = 3) -> Part:
    p0a, p1a = np.array(p0, float), np.array(p1, float)
    pts, radii = [], []
    for i in range(segs + 1):
        t = i / segs
        q = p0a + (p1a - p0a) * t
        q[2] += math.sin(t * math.pi) * bend * np.linalg.norm(p1a - p0a) * 0.3
        pts.append(tuple(q))
        radii.append(r0 + (r1 - r0) * t)
    return tube(pts, radii, mat, n=n, cap=True)


def coast_live_oak() -> Part:
    """Quercus agrifolia: short stout trunk, sinuous low limbs, broad dark dome (~9 m tall, 13 m wide)."""
    rng = np.random.default_rng(101)
    p = Part()
    p += _limb((0, 0, -0.3), (0.2, 0.1, 2.0), 0.5, 0.38, "bark_brown", bend=0.1, n=6, segs=2)
    limbs = [((0.2, 0.1, 1.9), (3.2, 1.2, 4.6)), ((0.2, 0.1, 1.9), (-2.6, 2.0, 5.0)),
             ((0.2, 0.1, 1.9), (-0.8, -3.0, 4.4)), ((0.2, 0.1, 1.9), (1.0, -0.2, 6.0))]
    for a, b in limbs:
        p += _limb(a, b, 0.3, 0.12, "bark_brown", bend=0.6, n=5, segs=3)
    blobs = [((0.0, 0.0, 6.6), 3.4), ((3.4, 1.1, 5.6), 2.8), ((-3.0, 1.9, 5.8), 2.9), ((-0.9, -3.3, 5.4), 2.7),
             ((2.2, -2.6, 5.2), 2.4), ((-1.6, 3.6, 5.0), 2.3), ((1.6, 3.2, 6.3), 2.4), ((-3.6, -1.3, 5.0), 2.2)]
    for i, (c, r) in enumerate(blobs):
        mat = "leaf_oak_light" if i % 3 == 0 else "leaf_oak"
        p += foliage_blob(r, mat, rng, squash=(1.0, 1.0, 0.72), jitter=0.13).moved(*c)
    return p


def mexican_fan_palm() -> Part:
    """Washingtonia robusta: very tall slender trunk, trimmed skirt, small crown of pleated fans (~17 m)."""
    rng = np.random.default_rng(202)
    H = 16.0
    lean = np.array([0.5, 0.25])
    pts, radii = [], []
    for i in range(7):
        t = i / 6
        off = lean * t**1.6
        pts.append((off[0], off[1], -0.2 + (H + 0.2) * t))
        radii.append(0.42 if i == 0 else 0.3 - 0.08 * t)
    p = tube(pts, radii, "palm_trunk", n=6, cap=False)
    top = np.array(pts[-1])
    # trimmed skirt of dead fronds (an inverted, jagged cone)
    sk = tube([tuple(top + (0, 0, -2.6)), tuple(top + (0, 0, -1.2)), tuple(top + (0, 0, 0.3))],
              [0.42, 0.85, 0.55], "palm_dead", n=8, cap=True)
    sk.V[: 8, :2] += rng.normal(0, 0.08, (8, 2))
    p += sk
    # crown: pleated fan fronds on short petioles
    n_fronds = 13
    for k in range(n_fronds):
        az = 2 * math.pi * k / n_fronds + rng.normal(0, 0.12)
        tilt = math.radians(rng.uniform(-35, 25))  # droop below / above horizontal
        p += _palm_frond(top + (0, 0, 0.25), az, tilt, rng)
    p += foliage_blob(0.55, "leaf_palm_dark", rng, squash=(1, 1, 1.1), jitter=0.05).moved(*(top + (0, 0, 0.4)))
    return p


def _palm_frond(base: np.ndarray, az: float, tilt: float, rng: np.random.Generator) -> Part:
    d = np.array([math.cos(az) * math.cos(tilt), math.sin(az) * math.cos(tilt), math.sin(tilt)])
    side = np.array([-math.sin(az), math.cos(az), 0.0])
    up = np.cross(side, d)
    pet_len = 1.1
    hub = base + d * pet_len
    p = tube([tuple(base), tuple(hub)], [0.05, 0.035], "palm_trunk", n=3, cap=False)
    # fan: half disc of radius R in the plane (d, side), pleated along `up`
    R = 1.35 + rng.uniform(-0.15, 0.15)
    n = 8
    V = [hub]
    for i in range(n + 1):
        a = -math.radians(80) + math.radians(160) * i / n
        q = hub + (d * math.cos(a) + side * math.sin(a)) * R
        q = q + up * (0.12 if i % 2 else -0.06) - np.array([0, 0, 0.25 * R * abs(math.cos(a))])
        V.append(q)
    F = [[0, i, i + 1] for i in range(1, n + 1)]
    mat = "leaf_palm" if rng.random() < 0.7 else "leaf_palm_dark"
    p += Part(np.array(V), F, [mat] * len(F))
    return p


def eucalyptus() -> Part:
    """Eucalyptus (blue gum windbreak type): tall pale leaning trunk, open irregular blue-green crown (~22 m)."""
    rng = np.random.default_rng(303)
    p = Part()
    trunk = [(0, 0, -0.3), (0.3, 0.1, 4.0), (0.9, 0.4, 8.5), (1.2, 0.5, 12.5)]
    p += tube(trunk, [0.55, 0.45, 0.36, 0.28], "bark_euc", n=6, cap=True)
    forks = [((1.2, 0.5, 12.0), (3.4, 1.8, 17.5)), ((1.2, 0.5, 12.0), (-1.5, -0.6, 18.5)),
             ((0.9, 0.4, 9.0), (-2.8, 1.5, 13.0)), ((1.2, 0.5, 12.0), (1.8, -1.6, 20.5))]
    for a, b in forks:
        p += _limb(a, b, 0.22, 0.08, "bark_euc", bend=0.3, n=4, segs=2)
    blobs = [((3.5, 1.9, 18.0), 2.6, 1.3), ((-1.6, -0.6, 19.5), 2.9, 1.25), ((-2.9, 1.6, 13.8), 2.2, 1.3),
             ((1.9, -1.7, 21.0), 2.3, 1.2), ((0.6, 0.8, 16.0), 2.5, 1.2), ((-0.4, 2.4, 17.4), 2.0, 1.3),
             ((2.4, -0.4, 14.4), 1.9, 1.3)]
    for i, (c, r, sz) in enumerate(blobs):
        mat = "leaf_euc" if i % 2 == 0 else "leaf_euc_dark"
        p += foliage_blob(r, mat, rng, squash=(0.9, 0.9, sz), jitter=0.16).moved(*c)
    return p


def jacaranda() -> Part:
    """Jacaranda mimosifolia in bloom: forked trunk, flat umbrella crown of purple (~8.5 m, 9 m wide)."""
    rng = np.random.default_rng(404)
    p = Part()
    p += _limb((0, 0, -0.3), (0.1, 0, 2.4), 0.3, 0.24, "bark_gray", bend=0.0, n=6, segs=1)
    for a, b in [((0.1, 0, 2.3), (2.2, 1.0, 5.6)), ((0.1, 0, 2.3), (-2.0, 1.4, 5.4)),
                 ((0.1, 0, 2.3), (-0.4, -2.3, 5.5))]:
        p += _limb(a, b, 0.19, 0.08, "bark_gray", bend=0.3, n=5, segs=2)
    blobs = [((0, 0, 7.0), 2.9), ((2.6, 1.2, 6.2), 2.4), ((-2.4, 1.6, 6.1), 2.4), ((-0.4, -2.7, 6.2), 2.4),
             ((2.0, -1.8, 6.0), 2.0), ((-2.2, -1.2, 5.8), 2.0), ((0.5, 2.8, 6.5), 2.0)]
    for i, (c, r) in enumerate(blobs):
        mat = "leaf_jacaranda" if i % 3 != 2 else "leaf_jacaranda_dark"
        p += foliage_blob(r, mat, rng, squash=(1.0, 1.0, 0.55), jitter=0.12).moved(*c)
    return p


def street_tree() -> Part:
    """Generic parkway street tree (Chinese elm / Brisbane box type): clean trunk, rounded crown (~7.5 m)."""
    rng = np.random.default_rng(505)
    p = Part()
    p += _limb((0, 0, -0.3), (0, 0.05, 2.6), 0.2, 0.15, "bark_gray", bend=0.0, n=5, segs=1)
    p += _limb((0, 0.05, 2.5), (0.9, 0.4, 4.0), 0.12, 0.06, "bark_gray", bend=0.2, n=4, segs=1)
    p += _limb((0, 0.05, 2.5), (-0.8, -0.5, 4.2), 0.12, 0.06, "bark_gray", bend=0.2, n=4, segs=1)
    blobs = [((0, 0, 5.4), 2.3, "leaf_street"), ((1.1, 0.6, 4.6), 1.7, "leaf_street_light"),
             ((-1.1, 0.5, 4.8), 1.7, "leaf_street"), ((0.1, -1.2, 4.6), 1.6, "leaf_street_light"),
             ((0.2, 0.3, 6.6), 1.5, "leaf_street_light")]
    for c, r, mat in blobs:
        p += foliage_blob(r, mat, rng, squash=(1.0, 1.0, 0.9), jitter=0.12).moved(*c)
    return p


def shrub() -> Part:
    """Irrigated landscape shrub mound (~1.2 m) with a few flowers."""
    rng = np.random.default_rng(606)
    p = foliage_blob(0.8, "shrub", rng, squash=(1.0, 1.0, 0.8), jitter=0.12, flat_bottom=0.6).moved(0, 0, 0.55)
    p += foliage_blob(0.55, "shrub", rng, squash=(1.0, 1.0, 0.85), jitter=0.12, flat_bottom=0.6).moved(0.55, 0.25, 0.4)
    p += foliage_blob(0.25, "shrub_flower", rng, jitter=0.1).moved(0.2, -0.45, 0.95)
    p += foliage_blob(0.2, "shrub_flower", rng, jitter=0.1).moved(-0.45, 0.25, 0.9)
    return p


def chaparral() -> Part:
    """Native coastal sage / chaparral clump for undeveloped slopes (~1.6 m, dusty sage green)."""
    rng = np.random.default_rng(707)
    p = Part()
    for c, r, m in [((0, 0, 0.7), 1.0, "chaparral"), ((0.9, 0.5, 0.5), 0.7, "chaparral_dark"),
                    ((-0.8, 0.4, 0.45), 0.65, "chaparral"), ((0.1, -0.8, 0.45), 0.6, "chaparral_dark")]:
        p += foliage_blob(r, m, rng, squash=(1.0, 1.0, 0.75), jitter=0.2, flat_bottom=0.55).moved(*c)
    return p


# ---------------------------------------------------------------------------
# street furniture
# ---------------------------------------------------------------------------


def street_lamp() -> Part:
    """Cobra-head street light, 9 m pole; the mast arm reaches +Y (forward)."""
    p = cylinder(0.32, 0.6, "concrete", n=8, bevel=0.04).moved(0, 0, -0.1)
    p += cylinder(0.13, 8.8, "pole_gray", n=6, r_top=0.08).moved(0, 0, 0.45)
    arm = tube([(0, 0, 8.3), (0, 0.8, 8.85), (0, 2.2, 9.05)], [0.06, 0.05, 0.045], "pole_gray", n=4, cap=True)
    p += arm
    head = box(0.42, 0.85, 0.2, "pole_gray", top="pole_gray", bevel=0.07).moved(0, 2.45, 8.88)
    p += head
    p += box(0.3, 0.6, 0.04, "lamp_lens").moved(0, 2.48, 8.85)
    return p


# ---------------------------------------------------------------------------
# vehicles
# ---------------------------------------------------------------------------


def _profile_body(profile: Sequence[tuple[float, float]], width: float, mat: str, bevel: float) -> Part:
    """Extrude a side silhouette (y, z) across X, centered, with softened edges."""
    bm = bmesh.new()
    vs = [bm.verts.new((-width / 2, y, z)) for y, z in profile]
    f = bm.faces.new(vs)
    f.normal_update()
    res = bmesh.ops.extrude_face_region(bm, geom=[f])
    nv = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=(width, 0, 0), verts=nv)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if bevel > 0:
        long_edges = [e for e in bm.edges if abs(e.verts[0].co.x - e.verts[1].co.x) > width * 0.9]
        side_edges = [e for e in bm.edges if e not in long_edges]
        bmesh.ops.bevel(bm, geom=side_edges, offset=bevel, offset_type="OFFSET", segments=1, profile=0.5,
                        affect="EDGES", clamp_overlap=True)
    bm.verts.index_update()
    V = np.array([v.co[:] for v in bm.verts])
    F = [[v.index for v in fc.verts] for fc in bm.faces]
    bm.free()
    return Part(V, F, [mat] * len(F))


def _greenhouse(profile: Sequence[tuple[float, float]], w_base: float, w_top: float, roof: str,
                glass: str = "car_glass") -> Part:
    """Cabin from a side trapezoid; side/front/rear faces glass, roof painted, tumblehome taper."""
    bm = bmesh.new()
    vs = [bm.verts.new((-w_base / 2, y, z)) for y, z in profile]
    f = bm.faces.new(vs)
    res = bmesh.ops.extrude_face_region(bm, geom=[f])
    nv = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=(w_base, 0, 0), verts=nv)
    zmin = min(z for _, z in profile)
    for v in bm.verts:
        if v.co.z > zmin + 1e-3:
            v.co.x *= w_top / w_base
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    bm.verts.index_update()
    V = np.array([v.co[:] for v in bm.verts])
    F, M = [], []
    for fc in bm.faces:
        if fc.normal.z < -0.9:
            continue  # hidden bottom
        F.append([v.index for v in fc.verts])
        M.append(roof if fc.normal.z > 0.85 else glass)
    bm.free()
    return Part(V, F, M)


def _wheel(r: float, w: float, x: float, y: float) -> Part:
    return cylinder_x(r, w, "tire", n=8, side_mat="rim").moved(x, y, r)


def _arch(cy: float, r: float, z0: float, n: int = 5) -> list[tuple[float, float]]:
    """Wheel arch points from front to back (y descending) over an axle at y=cy."""
    pts = []
    for i in range(n + 1):
        a = math.pi * i / n
        pts.append((cy + r * math.cos(a), z0 + r * math.sin(a)))
    return pts


@dataclass
class CarSpec:
    length: float
    width: float
    wheelbase: float
    wheel_r: float
    clearance: float  # body bottom height
    belt: float  # beltline (top of lower body)
    hood_front: float  # z of the hood's front edge
    roof: float  # roof height
    cabin: tuple[float, float, float, float]  # y: windshield base, roof front, roof rear, rear glass base
    paint: str = "paint_tint"
    nose_slope: float = 0.12
    tail_drop: float = 0.06


def _car(s: CarSpec, extras: bool = True) -> Part:
    L2 = s.length / 2
    ax = s.wheelbase / 2
    ar = s.wheel_r + 0.07
    z0 = s.clearance
    prof: list[tuple[float, float]] = []
    prof.append((L2 - 0.08, z0 + 0.02))
    prof += [(y, max(z, z0)) for y, z in _arch(ax, ar, s.wheel_r)]
    prof += [(y, max(z, z0)) for y, z in _arch(-ax, ar, s.wheel_r)]
    prof += [(-L2 + 0.08, z0 + 0.02), (-L2, z0 + 0.18), (-L2 + 0.02, s.belt - s.tail_drop),
             (-L2 + 0.18, s.belt)]
    prof += [(s.cabin[0] + 0.05, s.belt + 0.02), (L2 - 0.35, s.hood_front + 0.03), (L2 - 0.02, s.hood_front - s.nose_slope * 0.4),
             (L2, z0 + 0.25)]
    # profile must be CCW in the (y, z) plane seen from -X; reverse because we listed it front-bottom -> back
    body = _profile_body(prof[::-1], s.width, s.paint, bevel=0.05)
    gh = [(s.cabin[3], s.belt), (s.cabin[0], s.belt), (s.cabin[1], s.roof), (s.cabin[2], s.roof)]
    cab = _greenhouse(gh, s.width - 0.12, s.width - 0.42, s.paint).moved(0, 0, -0.01)
    p = body + cab
    wr, ww = s.wheel_r, 0.24
    for sx in (-1, 1):
        for sy in (-1, 1):
            p += _wheel(wr, ww, sx * (s.width / 2 - ww / 2 - 0.02), sy * ax)
    if extras:
        hz = s.hood_front - 0.12
        for sx in (-1, 1):
            p += box(0.36, 0.06, 0.12, "headlight", bevel=0.02).moved(sx * (s.width / 2 - 0.3), L2 - 0.03, hz)
            p += box(0.32, 0.06, 0.12, "taillight", bevel=0.02).moved(sx * (s.width / 2 - 0.25), -L2 + 0.02, s.belt - 0.2)
            p += box(0.1, 0.18, 0.1, s.paint).moved(sx * (s.width / 2 + 0.02), s.cabin[0] - 0.05, s.belt)  # mirrors
        p += box(s.width - 0.9, 0.05, 0.16, "plastic_black").moved(0, L2 - 0.0, z0 + 0.12)  # grille / lower intake
        p += box(s.width - 0.3, 0.05, 0.12, "plastic_black").moved(0, -L2 + 0.0, z0 + 0.1)  # rear bumper insert
    return p


def sedan() -> Part:
    return _car(CarSpec(length=4.8, width=1.84, wheelbase=2.8, wheel_r=0.33, clearance=0.24, belt=0.98,
                        hood_front=0.8, roof=1.44, cabin=(0.95, 0.12, -0.92, -1.62)))


def suv() -> Part:
    p = _car(CarSpec(length=4.85, width=1.94, wheelbase=2.85, wheel_r=0.38, clearance=0.34, belt=1.12,
                     hood_front=1.0, roof=1.76, cabin=(1.05, 0.45, -2.12, -2.32), tail_drop=0.02))
    for sx in (-1, 1):
        p += box(0.05, 2.4, 0.06, "aluminum").moved(sx * 0.68, -0.6, 1.76)
    return p


def minivan() -> Part:
    p = _car(CarSpec(length=5.15, width=2.0, wheelbase=3.05, wheel_r=0.35, clearance=0.26, belt=1.02,
                     hood_front=0.92, roof=1.78, cabin=(1.45, 0.75, -2.38, -2.52), tail_drop=0.02))
    p += box(0.03, 1.2, 0.03, "plastic_black").moved(1.0, -0.6, 1.0)  # sliding door track hint
    return p


def pickup() -> Part:
    s = CarSpec(length=5.8, width=2.0, wheelbase=3.6, wheel_r=0.4, clearance=0.4, belt=1.22, hood_front=1.12,
                roof=1.92, cabin=(1.05, 0.55, -0.75, -0.95), tail_drop=0.0)
    p = _car(s)
    # open bed: dark liner inset on top of the rear body
    p += box(s.width - 0.24, 1.9, 0.02, "plastic_black").moved(0, -1.95, s.belt + 0.005)
    return p


def school_bus() -> Part:
    """Type C conventional school bus (~11 m, 72 passengers), National School Bus Glossy Yellow."""
    L, W, H = 11.0, 2.44, 3.15
    p = Part()
    body_front, body_rear = 3.6, -L / 2
    hood_len = 1.75
    # main box with rounded roof edges
    main = box(W, body_front - body_rear, H - 0.45, "paint_bus", top="paint_white", bevel=0.18, segments=2)
    p += main.moved(0, (body_front + body_rear) / 2, 0.45)
    # hood / engine
    hood = box(W - 0.35, hood_len, 0.95, "paint_bus", bevel=0.14, segments=2)
    p += hood.moved(0, body_front + hood_len / 2 - 0.05, 0.75)
    p += box(W - 0.75, 0.06, 0.6, "plastic_black").moved(0, body_front + hood_len - 0.03, 0.85)  # grille
    p += box(W - 0.3, 0.2, 0.25, "plastic_black", bevel=0.04).moved(0, body_front + hood_len + 0.02, 0.45)  # bumper
    # windshield + side window band
    p += box(W - 0.2, 0.05, 1.05, "car_glass").moved(0, body_front + 0.005, 1.85)
    p += box(W + 0.02, 7.6, 0.72, "car_glass").moved(0, body_front - 0.15 - 7.6 / 2, 1.95)
    p += box(W - 0.4, 0.05, 0.75, "car_glass").moved(0, body_rear - 0.005, 1.95)  # rear emergency door glass
    # black rub rails
    for z in (0.95, 1.35, 1.75):
        p += box(W + 0.04, body_front - body_rear - 0.4, 0.06, "plastic_black").moved(0, (body_front + body_rear) / 2, z)
    # roof sign and warning lights
    p += box(1.6, 0.12, 0.32, "paint_bus").moved(0, body_front - 0.05, H - 0.05)
    p += box(1.3, 0.13, 0.18, "plastic_black").moved(0, body_front - 0.04, H + 0.02)
    for sx in (-0.85, -0.55, 0.55, 0.85):
        col = "taillight" if abs(sx) > 0.7 else "amber"
        p += box(0.18, 0.1, 0.14, col, bevel=0.03).moved(sx, body_front + 0.02, H - 0.25)
        p += box(0.18, 0.1, 0.14, col, bevel=0.03).moved(sx, body_rear - 0.04, H - 0.25)
    p += box(0.05, 0.45, 0.45, "taillight").moved(-W / 2 - 0.05, body_front - 1.4, 1.45)  # stop arm
    for sx in (-1, 1):
        p += box(0.3, 0.07, 0.18, "headlight", bevel=0.02).moved(sx * 0.75, body_front + hood_len - 0.04, 1.1)
        p += box(0.08, 0.3, 0.3, "plastic_black").moved(sx * (W / 2 + 0.2), body_front + 0.2, 1.9)  # mirrors
    # wheels: front single, rear dual
    wr = 0.5
    for sx in (-1, 1):
        p += _wheel(wr, 0.3, sx * (W / 2 - 0.25), body_front + 0.9)
        p += _wheel(wr, 0.55, sx * (W / 2 - 0.35), -2.6)
    return p


def shuttle_van() -> Part:
    """High-roof passenger shuttle van (Transit/Sprinter class, ~6 m). White body tints via instanceColor."""
    s = CarSpec(length=6.0, width=2.06, wheelbase=3.75, wheel_r=0.36, clearance=0.32, belt=1.2, hood_front=1.05,
                roof=2.72, cabin=(2.1, 1.45, -2.9, -2.98), paint="paint_white", tail_drop=0.0, nose_slope=0.2)
    p = _car(s)
    # cabin of a high-roof van is mostly painted: cover the side glass with a painted roof band + window strip
    p += box(s.width - 0.36, 4.3, 0.55, "paint_white", top="paint_white", bevel=0.12).moved(0, -0.75, 2.2)
    p += box(s.width + 0.02, 0.9, 0.06, "accent_teal").moved(0, -0.4, 1.1)  # livery stripe (front)
    p += box(s.width + 0.02, 5.2, 0.12, "accent_teal").moved(0, -0.3, 0.95)  # livery stripe
    return p


PROPS = {
    # name: (builder, category, tintable, notes)
    "tree_oak": (coast_live_oak, "tree", False, "Coast live oak (Quercus agrifolia), native slopes and yards"),
    "tree_palm": (mexican_fan_palm, "tree", False, "Mexican fan palm (Washingtonia robusta), arterials and commercial"),
    "tree_eucalyptus": (eucalyptus, "tree", False, "Eucalyptus windbreak, slopes and edges"),
    "tree_jacaranda": (jacaranda, "tree", False, "Jacaranda in bloom, street and yard accent"),
    "tree_street": (street_tree, "tree", False, "Generic parkway street tree"),
    "shrub": (shrub, "shrub", False, "Irrigated landscape shrub"),
    "chaparral": (chaparral, "shrub", False, "Coastal sage scrub / chaparral clump"),
    "street_lamp": (street_lamp, "street", False, "Cobra-head street light, arm reaches forward (+Y Blender / -z glTF)"),
    "car_sedan": (sedan, "vehicle", True, "Mid-size sedan"),
    "car_suv": (suv, "vehicle", True, "Mid-size SUV"),
    "car_minivan": (minivan, "vehicle", True, "Minivan"),
    "car_pickup": (pickup, "vehicle", True, "Full-size pickup"),
    "school_bus": (school_bus, "vehicle", False, "Type C yellow school bus"),
    "shuttle_van": (shuttle_van, "vehicle", True, "High-roof shuttle van (white body tints)"),
}


def build(name: str) -> Part:
    return PROPS[name][0]()


__all__ = ["PROPS", "build", "merge"]
