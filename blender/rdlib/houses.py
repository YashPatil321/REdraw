"""Procedural SoCal tract houses textured with the material atlases (street-scene preview).

This is also a reference implementation of the facade UV convention and bay grammar in
materials_manifest.json: walls are split into 3 m x 3 m floor-bays, u = meters along the wall / 3,
v = meters above grade / 3, and every bay picks a facade_openings cell (or plain stucco).
Material keys are strings resolved by `materials_for(part)` in the Blender preview:
    wall:<variant>:<tint>     facade_walls cell, tinted
    open:<cell>:<k>:<tint>    facade_openings cell (span cell k), wall areas tinted via the mask
    roof:<cell> / ground:<cell>
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import atlas
from .mesh import Part


def quad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, key: str, uv: list[tuple[float, float]]) -> Part:
    return Part(np.array([p0, p1, p2, p3], float), [[0, 1, 2, 3]], [key], [uv])


def poly_face(pts: list[tuple[float, float, float]], key: str, size: float) -> Part:
    P = np.array(pts, float)
    return Part(P, [list(range(len(P)))], [key], [[(float(x) / size, float(y) / size) for x, y, _ in P]])


@dataclass
class HouseStyle:
    wall_tint: str
    trim_tint: str
    roof: str
    wall_variant: str = "stucco_sand"
    garage: str = "garage_2car"  # or garage_3car
    garage_side: int = -1  # -1 garage on the west (left from the street), +1 east
    floors: int = 2
    seed: int = 0


def wall_bays(a: np.ndarray, b: np.ndarray, z0: float, floors: int, cells: list[list[str | None]], tint: str,
              variant: str, trim: str) -> Part:
    """Wall a->b (outward normal on the right when walking a->b seen from above... CCW ring = outside on the right)
    split into floor-bays. cells[floor][bay] = facade_openings cell name ('name' or 'name#k' for span cell k) or None."""
    out = Part()
    e = b - a
    L = float(np.hypot(e[0], e[1]))
    t = e / L
    nb = int(L // atlas.FACADE_M)
    margin = (L - nb * atlas.FACADE_M) / 2
    edges = [0.0] + ([margin] if margin > 1e-3 else []) + [margin + atlas.FACADE_M * (i + 1) for i in range(nb)] + ([L] if margin > 1e-3 else [])
    edges = sorted(set(round(x, 6) for x in edges))
    for f in range(floors):
        za, zb = z0 + f * atlas.FACADE_M, z0 + (f + 1) * atlas.FACADE_M
        for i in range(len(edges) - 1):
            s0, s1 = edges[i], edges[i + 1]
            bay = int(round((s0 - margin) / atlas.FACADE_M)) if s1 - s0 > atlas.FACADE_M - 1e-3 else None
            name = cells[f][bay] if (bay is not None and f < len(cells) and bay < len(cells[f])) else None
            u0, u1 = (s0 - margin) / atlas.FACADE_M, (s1 - margin) / atlas.FACADE_M
            v0, v1 = (za - z0) / atlas.FACADE_M, (zb - z0) / atlas.FACADE_M
            if name:
                cname, _, k = name.partition("#")
                key = f"open:{cname}:{k or 0}:{tint}"
            else:
                key = f"wall:{variant}:{tint}"
            p0 = np.array([a[0] + t[0] * s0, a[1] + t[1] * s0, za])
            p1 = np.array([a[0] + t[0] * s1, a[1] + t[1] * s1, za])
            out += quad(p0, p1, p1 + [0, 0, zb - za], p0 + [0, 0, zb - za], key, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    return out


def hip(cx: float, cy: float, w: float, d: float, z: float, pitch_deg: float, roof_key: str, trim_key: str,
        overhang: float = 0.55) -> Part:
    """Hip roof over an axis-aligned w x d rectangle: tile faces (slope UVs), fascia, soffit."""
    W, D = w / 2 + overhang, d / 2 + overhang
    tp = math.tan(math.radians(pitch_deg))
    h = min(W, D) * tp
    if W >= D:
        r = W - D
        top = [(-r, 0, h), (r, 0, h)]
        V = [(-W, -D, 0), (W, -D, 0), (W, D, 0), (-W, D, 0), *top]
        F = [[0, 1, 5, 4], [1, 2, 5], [2, 3, 4, 5], [3, 0, 4]]
    else:
        r = D - W
        top = [(0, -r, h), (0, r, h)]
        V = [(-W, -D, 0), (W, -D, 0), (W, D, 0), (-W, D, 0), *top]
        F = [[0, 1, 4], [1, 2, 5, 4], [2, 3, 5], [3, 0, 4, 5]]
    P = np.array(V, float) + [cx, cy, z]
    p = Part()
    for f in F:
        pts = P[f]
        uv = atlas.face_uv(pts, atlas.MAT_TILE_ROOF)
        p += Part(pts.copy(), [list(range(len(f)))], [roof_key], [[tuple(x) for x in uv]])
    ring = P[:4]
    for i in range(4):  # fascia 0.25 m
        a, b = ring[i], ring[(i + 1) % 4]
        pts = np.array([a - [0, 0, 0.25], b - [0, 0, 0.25], b, a])
        uv = atlas.face_uv(pts, atlas.MAT_TRIM)
        p += Part(pts, [[0, 1, 2, 3]], [trim_key], [[tuple(x) for x in uv]])
    sof = ring[::-1] - [0, 0, 0.25]
    p += Part(sof.copy(), [[0, 1, 2, 3]], [trim_key], [[(float(x) / 3, float(y) / 3) for x, y, _ in sof]])
    return p


def house(x: float, y: float, st: HouseStyle) -> tuple[Part, dict]:
    """Two-storey tract house facing -Y (the street), garage wing projecting toward the street.
    Returns the Part and a layout dict (driveway, walk, beds, door position) in world meters."""
    rng = np.random.default_rng(st.seed)
    W, D = 15.0, 11.0  # main body (5 bays wide)
    gw = 9.6 if st.garage == "garage_3car" else 6.6
    gd = 7.0
    tint, trim = st.wall_tint, st.trim_tint
    wall_trim = f"wall:stucco_smooth:{trim}"
    p = Part()
    # main body ring (CCW seen from above), origin at the body's front-left corner
    x0, y0 = x - W / 2, y
    ring = [np.array(v, float) for v in ((x0, y0), (x0 + W, y0), (x0 + W, y0 + D), (x0, y0 + D))]
    gx0 = x0 if st.garage_side < 0 else x0 + W - gw
    nfront = int(W // 3)
    front0: list[str | None] = [None] * nfront
    front1: list[str | None] = []
    # garage occupies the front of the body's garage side on floor 0 (the wing projects in front of it)
    gb = int(math.ceil(gw / 3))
    if st.garage_side < 0:
        door_bay = gb
        win_bays = [b for b in range(gb + 1, nfront)]
    else:
        door_bay = nfront - gb - 1
        win_bays = [b for b in range(0, door_bay)]
    front0[door_bay] = "door_front"
    for b in win_bays:
        front0[b] = str(rng.choice(["window_picture", "window_slider", "window_arched"]))
    front1 = [str(rng.choice(["window_slider", "window_pair", "window_arched", "window_small"])) if rng.random() < 0.8 else None
              for _ in range(nfront)]
    front1[door_bay] = "window_arched"
    side_cells = [[None, "window_slider", None], ["window_small", None, "window_slider"]]
    back_cells = [["door_slider", "window_slider", None, "window_pair", "door_slider"], ["window_slider", None, "window_pair", "window_slider", None]]
    cells_by_edge = [[front0, front1], [side_cells[0], side_cells[1]], [back_cells[0], back_cells[1]], [side_cells[1], side_cells[0]]]
    for i in range(4):
        a, b = ring[i], ring[(i + 1) % 4]
        cells = cells_by_edge[i]
        if i == 0:
            # front: floor 0 bays behind the garage wing are hidden; keep them plain
            cells = [[c if not (st.garage_side < 0 and k < gb or st.garage_side > 0 and k >= nfront - gb) else None
                      for k, c in enumerate(cells[0])], cells[1]]
        p += wall_bays(a, b, 0.0, st.floors, cells, tint, st.wall_variant, trim)
    zt = st.floors * atlas.FACADE_M
    p += hip(x0 + W / 2, y0 + D / 2, W, D, zt, 20.0, f"roof:{st.roof}", wall_trim)
    # garage wing in front of the body
    gy0 = y0 - gd
    gring = [np.array(v, float) for v in ((gx0, gy0), (gx0 + gw, gy0), (gx0 + gw, y0), (gx0, y0))]
    nspan = 3 if st.garage == "garage_3car" else 2
    gcells = [[f"{st.garage}#{k}" for k in range(nspan)]]
    p += wall_bays(gring[0], gring[1], 0.0, 1, gcells, tint, st.wall_variant, trim)
    p += wall_bays(gring[1], gring[2], 0.0, 1, [[None, "window_small"]], tint, st.wall_variant, trim)
    p += wall_bays(gring[3], gring[0], 0.0, 1, [[None, None]], tint, st.wall_variant, trim)
    p += hip(gx0 + gw / 2, gy0 + gd / 2, gw, gd, 3.0, 20.0, f"roof:{st.roof}", wall_trim, overhang=0.45)
    # stone veneer wainscot on the garage front piers (0.9 m)
    for xs in (gx0 - 0.02, gx0 + gw - 0.0):
        pass
    a, b = gring[0] + [0, -0.03], gring[1] + [0, -0.03]
    e = (b - a) / np.linalg.norm(b - a)
    for s0, s1 in ((0.0, 0.42), (gw - 0.42, gw)):
        q0 = np.array([*(a + e * s0), 0.0])
        q1 = np.array([*(a + e * s1), 0.0])
        p += quad(q0, q1, q1 + [0, 0, 0.9], q0 + [0, 0, 0.9], "wall:stone_veneer:#FFFFFF",
                  [(s0 / 3, 0), (s1 / 3, 0), (s1 / 3, 0.3), (s0 / 3, 0.3)])
    door_x = x0 + (door_bay + 0.5) * 3 + (W - nfront * 3) / 2
    lay = {"driveway": (gx0 + 0.3, gx0 + gw - 0.3, gy0), "door": (door_x, y0), "garage_x": (gx0, gx0 + gw),
           "body": (x0, y0, W, D), "garage_front_y": gy0}
    return p, lay


def ground_rect(x0: float, y0: float, x1: float, y1: float, cell: str, size: float, z: float = 0.0) -> Part:
    return poly_face([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], f"ground:{cell}", size)


def curb(x0: float, x1: float, y: float, facing: int, z_road: float = 0.0) -> Part:
    """Curb & gutter strip along x at y (gutter on the road side). facing=+1: road at y < curb (curb rises toward +y)."""
    p = Part()
    w = 0.75
    prof = [(0.0, 0.0), (0.45, 0.02), (0.47, 0.17), (0.6, 0.17)]  # (across from the gutter lip, height)
    for i in range(len(prof) - 1):
        (a0, h0), (a1, h1) = prof[i], prof[i + 1]
        ya, yb = y + facing * a0, y + facing * a1
        va, vb = [0.0, 0.6, 0.8, 1.0][i], [0.6, 0.8, 1.0, 1.0][i]
        if i == 2:
            va, vb = 0.8, 1.0
        P = [(x0, ya, z_road + h0), (x1, ya, z_road + h0), (x1, yb, z_road + h1), (x0, yb, z_road + h1)]
        if facing < 0:
            P = [P[1], P[0], P[3], P[2]]
        uv = [(x0 / 3, va), (x1 / 3, va), (x1 / 3, vb), (x0 / 3, vb)] if facing > 0 else [(x1 / 3, va), (x0 / 3, va), (x0 / 3, vb), (x1 / 3, vb)]
        p += Part(np.array(P, float), [[0, 1, 2, 3]], ["ground:curb_gutter"], [uv])
    del w
    return p
