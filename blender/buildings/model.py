"""Building spec -> textured triangle soup (pure numpy + shapely; runs in .venv-blender and .venv).

A spec (pipeline/build_buildings_blender.py) carries the real footprint ring in scene coords,
ground elevations, and a roof model (lidar | tag | heuristic). `build_building(spec, lod, P)`
returns triangles in Blender axes (+X east, +Y north = -scene z, +Z up) with per-triangle
`_MAT`, `_VARIANT`, sRGB tint (COLOR_0) and per-corner UVs following
client/public/assets/materials/materials_manifest.json:

    walls / glass / trim / garage: u = m along the wall / 3, v = m above the floor / 3
    pitched roofs: u = m along the eave / 4, v = m up the slope / 4
    flat roofs: u = X / 4, v = Y / 4 (= -scene z / 4)

LOD0: regularized footprint, straight-skeleton hip / gable / shed / flat roofs from maximal
rectangles (geom.py), two roof levels (two-storey block + one-storey wing), eaves with fascia
and soffit, windows (trim surround, glass, sill) per floor, recessed garage door facing the street,
entry door with porch roof, storefront / school bands, parapets with caps, rooftop HVAC, PV arrays.
LOD1: walls to the roof line + roof planes (no overhang, no openings) on a coarser footprint.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient

from . import geom

MAT_WALL, MAT_TILE, MAT_FLAT, MAT_GLASS, MAT_TRIM, MAT_GARAGE = range(6)

# materials_manifest.json variants (index = _VARIANT)
WALL_VARIANTS = ["stucco_smooth", "stucco_sand", "stucco_lace", "stucco_catface", "stucco_weathered", "stucco_scored", "stone_veneer"]
TILE_VARIANTS = ["s_tile_terracotta", "s_tile_blend", "s_tile_brown", "s_tile_aged", "barrel_mission", "flat_tile_brown",
                 "flat_tile_grey", "flat_tile_charcoal", "flat_tile_sandstone", "solar_panel"]
FLAT_VARIANTS = ["flat_tpo", "flat_tpo_grime", "flat_gravel", "flat_modbit", "concrete_deck", "standing_seam"]
TRIM_VARIANTS = ["stucco_smooth", "stone_veneer"]
GARAGE_VARIANTS = ["garage_2car", "garage_3car"]

# Default parameters; the spec index carries the assumptions.yaml values (`params`) which win.
DEFAULT_PARAMS: dict[str, Any] = {
    "seed": 20261003,
    "overhang_m": {"house": 0.55, "apartments": 0.6, "other": 0.3, "commercial": 0.6, "school": 0.6},
    "fascia_m": 0.2,
    "story_m": {"house": 2.95, "apartments": 3.0, "commercial": 4.2, "school": 4.0, "other": 3.0},
    "slab_m": 0.15,
    "parapet_m": {"commercial": 1.0, "school": 0.6, "apartments": 0.5, "house": 0.4, "other": 0.3},
    "garage_door_w_m": 4.88,
    "garage_single_w_m": 2.44,
    "garage_door_h_m": 2.13,
    "garage_recess_m": 0.15,
    "entry_door_w_m": 0.95,
    "entry_door_h_m": 2.13,
    "two_level_share": 0.8,
    "solar_share": 0.18,
    "stone_wainscot_share": 0.3,
    "hvac_m2_per_unit": 220.0,
    "wall_palette": [([226, 212, 184], 0.2), ([212, 190, 154], 0.2), ([236, 230, 216], 0.1), ([222, 204, 172], 0.15),
                     ([196, 188, 172], 0.1), ([204, 178, 140], 0.1), ([186, 166, 136], 0.07), ([214, 196, 170], 0.08)],
    "trim_palette": [([242, 240, 233], 0.55), ([232, 222, 200], 0.2), ([92, 78, 66], 0.15), ([60, 58, 56], 0.1)],
    "tile_variants": {"s_tile_terracotta": 0.12, "s_tile_blend": 0.22, "s_tile_brown": 0.16, "s_tile_aged": 0.08,
                      "barrel_mission": 0.04, "flat_tile_brown": 0.14, "flat_tile_grey": 0.12, "flat_tile_charcoal": 0.08,
                      "flat_tile_sandstone": 0.04},
    "commercial_wall_palette": [([226, 214, 192], 0.35), ([206, 192, 168], 0.25), ([236, 232, 222], 0.2), ([188, 170, 146], 0.2)],
}

# fallback sRGB colors (COLOR_0 for clients without atlases; roofs / glass ignore the tint otherwise)
TILE_RGB = {"s_tile_terracotta": (176, 92, 60), "s_tile_blend": (158, 98, 72), "s_tile_brown": (122, 84, 64),
            "s_tile_aged": (140, 108, 88), "barrel_mission": (184, 104, 70), "flat_tile_brown": (112, 86, 70),
            "flat_tile_grey": (128, 126, 120), "flat_tile_charcoal": (78, 77, 76), "flat_tile_sandstone": (176, 150, 118),
            "solar_panel": (40, 46, 58)}
FLAT_RGB = {"flat_tpo": (214, 212, 204), "flat_tpo_grime": (196, 194, 186), "flat_gravel": (160, 152, 138),
            "flat_modbit": (120, 118, 112), "concrete_deck": (170, 164, 152), "standing_seam": (124, 132, 138)}
GLASS_RGB = (52, 64, 76)
DOOR_RGB = [(92, 62, 44), (70, 50, 38), (120, 84, 56), (48, 52, 58), (142, 54, 40)]
GARAGE_RGB = (236, 232, 222)
STONE_RGB = (126, 111, 92)  # sRGB of the stone_veneer cell mean (not tintable: COLOR_0 is only a fallback)


def _hash(*parts: Any) -> int:
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little")


def _pick(rng: np.random.Generator, items: list[tuple[Any, float]]) -> Any:
    w = np.array([s for _, s in items], float)
    return items[int(rng.choice(len(items), p=w / w.sum()))][0]


# ---------------------------------------------------------------------------
# triangle soup
# ---------------------------------------------------------------------------


@dataclass
class Soup:
    P: list[np.ndarray] = field(default_factory=list)  # (t, 3, 3)
    UV: list[np.ndarray] = field(default_factory=list)  # (t, 3, 2)
    A: list[np.ndarray] = field(default_factory=list)  # (t, 6): mat, var, r, g, b, bid

    def add(self, P: np.ndarray, UV: np.ndarray, mat: int, var: int, rgb: tuple[int, int, int], bid: int) -> None:
        P = np.asarray(P, float).reshape(-1, 3, 3)
        if not len(P):
            return
        UV = np.asarray(UV, float).reshape(-1, 3, 2)
        # drop degenerate triangles
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        ok = np.einsum("ij,ij->i", n, n) > 1e-10
        if not ok.all():
            P, UV = P[ok], UV[ok]
            if not len(P):
                return
        a = np.empty((len(P), 6))
        a[:, 0], a[:, 1], a[:, 2:5], a[:, 5] = mat, var, np.asarray(rgb, float) / 255.0, bid
        self.P.append(P)
        self.UV.append(UV)
        self.A.append(a)

    def extend(self, o: Soup) -> None:
        self.P += o.P
        self.UV += o.UV
        self.A += o.A

    @property
    def ntris(self) -> int:
        return int(sum(len(p) for p in self.P))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.P:
            return np.zeros((0, 3, 3)), np.zeros((0, 3, 2)), np.zeros((0, 6))
        return np.concatenate(self.P), np.concatenate(self.UV), np.concatenate(self.A)


def quad_tris(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    return np.array([[p0, p1, p2], [p0, p2, p3]], float)


def facade_uv(P: np.ndarray, a: np.ndarray, t: np.ndarray, floor: float) -> np.ndarray:
    """u = m along the wall from its start corner a (plan) / 3, v = m above the floor / 3."""
    P = np.asarray(P, float)
    u = ((P[..., 0] - a[0]) * t[0] + (P[..., 1] - a[1]) * t[1]) / 3.0
    v = (P[..., 2] - floor) / 3.0
    return np.stack([u, v], axis=-1)


# glass_curtain cell (facade_walls): vertical mullions at u = 0, 0.5, 1; vision lites for v in [0.27, 0.98]
GLASS_U = (0.02, 0.48)
GLASS_V = (0.27, 0.98)


def glass_uv(P: np.ndarray, a: np.ndarray, t: np.ndarray, z0: float, w: float, h: float, panes: int = 1,
             bay_m: float | None = None) -> np.ndarray:
    """Window glass UVs: each pane maps onto one vision lite of the glass_curtain cell (frame-free
    reflections, the center mullion = slider meeting rail). bay_m: storefront / ribbon glazing, one
    lite pair per bay_m meters along the wall."""
    P = np.asarray(P, float)
    s = (P[..., 0] - a[0]) * t[0] + (P[..., 1] - a[1]) * t[1]
    if bay_m:
        u = s / bay_m
    else:
        f = np.clip(s / max(w, 1e-6), 0, 1)
        u = GLASS_U[0] + f * ((GLASS_U[1] - GLASS_U[0]) if panes == 1 else (1.0 - 2 * GLASS_U[0]))
    v = GLASS_V[0] + np.clip((P[..., 2] - z0) / max(h, 1e-6), 0, 1) * (GLASS_V[1] - GLASS_V[0])
    return np.stack([u, v], axis=-1)


def roof_uv(P: np.ndarray, normal: np.ndarray, flat: bool) -> np.ndarray:
    P = np.asarray(P, float)
    if flat:
        return P[..., :2] / 4.0
    n = normal / np.linalg.norm(normal)
    e = np.array([-n[1], n[0], 0.0])
    le = np.linalg.norm(e)
    e = e / le if le > 1e-9 else np.array([1.0, 0.0, 0.0])
    s = np.cross(n, e)
    if s[2] < 0:
        s, e = -s, -e
    return np.stack([P @ e, P @ s], axis=-1) / 4.0


def box_tris(c: np.ndarray, ax: np.ndarray, ay: np.ndarray, z0: float, z1: float, top: bool = True) -> list[np.ndarray]:
    """Oriented box (plan center c, half-axes ax, ay) between z0 and z1: list of side quads (+ top)."""
    corners = [c - ax - ay, c + ax - ay, c + ax + ay, c - ax + ay]
    if ax[0] * ay[1] - ax[1] * ay[0] < 0:
        corners = corners[::-1]
    out = []
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        out.append(quad_tris(np.r_[p, z0], np.r_[q, z0], np.r_[q, z1], np.r_[p, z1]))
    if top:
        out.append(quad_tris(*[np.r_[k, z1] for k in corners]))
    return out


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------


@dataclass
class Style:
    wall_rgb: tuple[int, int, int]
    trim_rgb: tuple[int, int, int]
    wall_var: int
    roof_mat: int
    roof_var: int
    roof_rgb: tuple[int, int, int]
    door_rgb: tuple[int, int, int]
    soffit_rgb: tuple[int, int, int]
    stone: bool
    solar: bool
    three_car: bool


def make_style(spec: dict[str, Any], P: dict[str, Any], flat: bool, rng: np.random.Generator) -> Style:
    btype = spec["type"]
    pal = P["commercial_wall_palette"] if btype in ("commercial", "school") else P["wall_palette"]
    wall = tuple(spec["wall_rgb"]) if spec.get("wall_rgb") else tuple(_pick(rng, [(tuple(c), s) for c, s in pal]))
    trim = tuple(_pick(rng, [(tuple(c), s) for c, s in P["trim_palette"]]))
    if btype in ("commercial", "school"):
        trim = (236, 232, 222) if rng.random() < 0.6 else (82, 74, 66)
    wall_var = 5 if btype == "school" else int(_pick(rng, [(1, 0.55), (2, 0.2), (0, 0.15), (3, 0.06), (4, 0.04)]))
    tiles = list(P["tile_variants"].items())
    tile = str(_pick(rng, tiles))
    if flat:
        fv = int(_pick(rng, [(0, 0.35), (1, 0.3), (2, 0.15), (3, 0.12), (4, 0.08)])) if btype != "house" else 3
        roof_mat, roof_var, roof_rgb = MAT_FLAT, fv, FLAT_RGB[FLAT_VARIANTS[fv]]
    else:
        roof_mat, roof_var, roof_rgb = MAT_TILE, TILE_VARIANTS.index(tile), TILE_RGB[tile]
    if spec.get("roof_rgb") and not flat:
        roof_rgb = tuple(spec["roof_rgb"])
    door = DOOR_RGB[int(rng.integers(len(DOOR_RGB)))]
    soffit = trim if trim[0] > 150 else wall
    return Style(wall, trim, wall_var, roof_mat, roof_var, roof_rgb, door, soffit,
                 stone=btype == "house" and rng.random() < P["stone_wainscot_share"],
                 solar=btype == "house" and not flat and rng.random() < P["solar_share"],
                 three_car=rng.random() < 0.35)


# ---------------------------------------------------------------------------
# roof planning
# ---------------------------------------------------------------------------


@dataclass
class Level:
    pieces_ov: list[geom.Piece]
    pieces_core: list[geom.Piece]
    core: Polygon | MultiPolygon  # local, the part of the plan this level's walls enclose
    flat: bool
    z_wall: float
    clip: Polygon | MultiPolygon | None = None  # local area this level's roof must not cover


def _az_to_axis(az_deg: float | None, theta: float) -> int | None:
    """Compass azimuth of the ridge (deg from north, clockwise) -> local axis (0 = x, 1 = y)."""
    if az_deg is None or (isinstance(az_deg, float) and math.isnan(az_deg)):
        return None
    a = math.radians(az_deg)
    d = np.array([math.sin(a), math.cos(a)])  # plan (east, north)
    c, s = math.cos(-theta), math.sin(-theta)
    lx, ly = c * d[0] - s * d[1], s * d[0] + c * d[1]
    return 0 if abs(lx) >= abs(ly) else 1


def _cap(pieces: list[geom.Piece], zcap: float | None) -> list[geom.Piece]:
    if zcap is None:
        return pieces
    for p in pieces:
        if p.kind != "flat":
            p.planes = np.vstack([p.planes, [0.0, 0.0, zcap]])
    return pieces


def _hip_planes_poly(ring: np.ndarray, z_wall: float, tanp: float) -> np.ndarray:
    """Planes rising inward from every edge of a CCW ring (local)."""
    P = []
    n = len(ring)
    for i in range(n):
        p, q = ring[i], ring[(i + 1) % n]
        e = q - p
        L = float(np.hypot(*e))
        if L < 1e-6:
            continue
        nin = np.array([-e[1], e[0]]) / L
        P.append((tanp * nin[0], tanp * nin[1], z_wall - tanp * float(nin @ p)))
    return np.array(P, float)


def _convex_pieces(poly: Polygon, planes: np.ndarray, kind: str) -> list[geom.Piece]:
    out = []
    for t in geom.triangulate(poly):
        out.append(geom.Piece(np.asarray(t, float), planes.copy(), kind))
    return out


def lidar_end_is_hip(spec: dict[str, Any], fp: geom.Footprint, r: tuple[float, float, float, float], axis: int,
                     z_floor: float, z_wall: float) -> bool | None:
    """Lidar planes: does a roof plane fall toward the short end of rect r (axis = ridge axis)?"""
    planes = spec.get("planes")
    lc = spec.get("lidar_center")
    if not planes or lc is None:
        return None
    x0, y0, x1, y1 = r
    ends = ([((x0, (y0 + y1) / 2), (-1.0, 0.0)), ((x1, (y0 + y1) / 2), (1.0, 0.0))] if axis == 0 else
            [(((x0 + x1) / 2, y0), (0.0, -1.0)), (((x0 + x1) / 2, y1), (0.0, 1.0))])
    eave_rel = z_wall - z_floor
    votes = 0
    for (mx, my), (ox, oy) in ends:
        mp = fp.to_plan(np.array([[mx, my]]))[0]
        od = fp.to_plan(np.array([[mx + ox, my + oy]]))[0] - mp
        ex, ny = mp[0] - lc[0], mp[1] - lc[1]
        hit = False
        for pl in planes:
            try:
                nrm = pl["normal"]
                d = float(pl["d"])
                slope = float(pl.get("slope_deg", 0.0))
            except (KeyError, TypeError):
                continue
            if slope < 10.0 or abs(nrm[2]) < 1e-3:
                continue
            h = (d - nrm[0] * ex - nrm[1] * ny) / nrm[2]
            facing = (nrm[0] * od[0] + nrm[1] * od[1]) / max(math.hypot(nrm[0], nrm[1]), 1e-9)
            if facing > 0.8 and abs(h - eave_rel) < 0.9:
                hit = True
                break
        votes += hit
    return votes >= 1


def hgrid_values(spec: dict[str, Any], fp: geom.Footprint, region: Polygon) -> np.ndarray:
    """nDSM heights (m above ground) of the spec's lidar height grid inside a local region."""
    g = spec.get("hgrid")
    if not g:
        return np.zeros(0)
    h = np.asarray(g["h"], float).reshape(g["ny"], g["nx"]) / 10.0
    xs = g["x0"] + g["step"] * np.arange(g["nx"])
    ys = g["y0"] + g["step"] * np.arange(g["ny"])
    X, Y = np.meshgrid(xs, ys)
    loc = fp.to_local(np.column_stack([X.ravel(), Y.ravel()]))
    inside = shapely.contains_xy(region, loc[:, 0], loc[:, 1]) & (h.ravel() >= 0)
    return h.ravel()[inside]


@dataclass
class RoofPlan:
    levels: list[Level]
    roof_type: str
    eave_h: float  # above floor (main / highest level)
    ridge_h: float
    two_level: bool
    tanp: float
    ov: float


def plan_roof(spec: dict[str, Any], fp: geom.Footprint, lod: int, P: dict[str, Any], rng: np.random.Generator) -> RoofPlan:
    btype = spec["type"]
    floor = float(spec["floor_y"])
    E = float(spec["eave_h"])
    R = float(spec.get("ridge_h") or E)
    rtype = str(spec.get("roof_type") or "hip")
    if rtype in ("unknown", "none", ""):
        rtype = "hip" if btype in ("house", "apartments") else "flat"
    pitch = float(spec.get("pitch_deg") or 22.0)
    if rtype != "flat":
        pitch = float(np.clip(pitch, 14.0, 40.0))
    tanp = math.tan(math.radians(pitch))
    flat = rtype == "flat"
    ov = 0.0 if (flat or lod > 0) else float(P["overhang_m"].get(btype, 0.5))
    if lod > 0:
        ov = 0.0
    poly = fp.local_poly
    lidar = spec.get("roof_source") == "lidar"
    zcap = None
    if not flat:
        if lidar and R > E + 0.8:
            zcap = floor + R + 0.05
        else:
            zcap = floor + E + (min(3.0, max(1.2, R - E)) if btype == "house" else min(4.5, max(1.2, (R - E) if R > E else 4.5)))
    # ---------------- rect pieces ----------------
    rects: list[tuple[float, float, float, float]] = []
    if fp.rect:
        rects = geom.max_rectangles(poly, max_n=12 if lod == 0 else 4)
    ridge_axis = _az_to_axis(spec.get("ridge_az_deg"), fp.theta)

    def kinds_for(rs: list[tuple[float, float, float, float]], z_wall: float) -> list[tuple[str, int | None]]:
        out = []
        for k, r in enumerate(rs):
            w, d = r[2] - r[0], r[3] - r[1]
            long_ax = 0 if w >= d else 1
            if flat:
                out.append(("flat", None))
                continue
            if rtype == "shed":
                out.append(("shed", ridge_axis if k == 0 and ridge_axis is not None else long_ax))
                continue
            if rtype == "gable":
                ax = ridge_axis if (k == 0 and ridge_axis is not None) else long_ax
                out.append(("gable", ax))
                continue
            if rtype == "complex":
                hip_end = lidar_end_is_hip(spec, fp, r, long_ax, floor, z_wall)
                if hip_end is None:
                    hip_end = (_hash(spec["id"], k) % 3) != 0
                out.append(("hip" if hip_end else "gable", long_ax))
                continue
            out.append(("hip", None))
        return out

    def pieces(rs: list[tuple[float, float, float, float]], z_wall: float, o: float, kinds: list[tuple[str, int | None]],
               cap: float | None) -> list[geom.Piece]:
        ps = [geom.rect_piece(r, z_wall, tanp, kd, o, ax) for r, (kd, ax) in zip(rs, kinds, strict=True)]
        return _cap(ps, cap)

    levels: list[Level] = []
    two = False
    E_high = E
    if rects:
        covered = shapely.union_all([box(*r) for r in rects])
        rest = poly.difference(covered.buffer(1e-6))
        rest_polys = [g for g in getattr(rest, "geoms", [rest]) if g.geom_type == "Polygon" and g.area > 0.05]
        # ---- two-level massing ----
        high_rects: list[tuple[float, float, float, float]] = []
        E_low = E
        if not flat and btype == "house" and len(rects) >= 2 and lod == 0 or (not flat and btype == "house" and len(rects) >= 2 and lod > 0):
            high_rects, E_low, E_high = _split_levels(spec, fp, rects, E, R, tanp, P, rng)
            two = bool(high_rects)
        if two:
            high_core = shapely.union_all([box(*r) for r in high_rects])
            z_low, z_high = floor + E_low, floor + E_high
            cap_low = min(zcap, z_high - 0.35) if zcap is not None else z_high - 0.35
            k_all = kinds_for(rects, z_low)
            k_high = kinds_for(high_rects, z_high)
            cap_high = zcap if (zcap is not None and zcap > z_high + 0.8) else None
            if lidar and cap_high is None:
                cap_high = z_high + max(1.0, (R - E_high))
            hp_ov = pieces(high_rects, z_high, ov, k_high, cap_high)
            clip_low = high_core
            if ov > 0:
                hov = shapely.union_all([Polygon(p.region) for p in hp_ov])
                clip_low = shapely.union_all([high_core, hov.difference(poly)])
            levels.append(Level(pieces(rects, z_low, ov, k_all, cap_low), pieces(rects, z_low, 0.0, k_all, cap_low), poly, False, z_low,
                                clip=clip_low))
            levels.append(Level(hp_ov, pieces(high_rects, z_high, 0.0, k_high, cap_high), high_core, False, z_high))
        else:
            z = floor + E
            k_all = kinds_for(rects, z)
            ps_ov = pieces(rects, z, ov, k_all, zcap)
            ps_core = pieces(rects, z, 0.0, k_all, zcap)
            for g in rest_polys:  # area not covered by the rectangle cover: flat patches at the wall top
                fl = np.array([[0.0, 0.0, z]])
                ps_ov += _convex_pieces(g.buffer(ov, join_style=2) if ov else g, fl, "flat")
                ps_core += _convex_pieces(g, fl, "flat")
            levels.append(Level(ps_ov, ps_core, poly, flat, z))
    else:
        z = floor + E
        if flat:
            planes = np.array([[0.0, 0.0, z]])
        else:
            ring = fp.local
            hull = Polygon(ring).convex_hull
            if Polygon(ring).area / max(hull.area, 1e-9) > 0.93:
                hp = orient(hull, 1.0)
                planes = _hip_planes_poly(np.asarray(hp.exterior.coords)[:-1], z, tanp)
            else:
                mrr = orient(Polygon(ring).minimum_rotated_rectangle, 1.0)
                planes = _hip_planes_poly(np.asarray(mrr.exterior.coords)[:-1], z, tanp)
            if zcap is not None:
                planes = np.vstack([planes, [0.0, 0.0, zcap]])
        kind = "flat" if flat else "hip"
        big = poly.buffer(ov, join_style=2) if ov else poly
        levels.append(Level(_convex_pieces(big, planes, kind), _convex_pieces(poly, planes, kind), poly, flat, z))
    ridge = max((float(np.nanmax(geom.envelope_height(lv.pieces_core, _sample_pts(lv.core)))) for lv in levels), default=floor + E) - floor
    return RoofPlan(levels, rtype, E_high if two else E, max(ridge, E), two, tanp, ov)


def _sample_pts(g: Polygon | MultiPolygon) -> np.ndarray:
    pts = []
    for p in getattr(g, "geoms", [g]):
        pts.append(np.asarray(p.exterior.coords))
        pts.append(np.asarray(p.representative_point().coords))
        minx, miny, maxx, maxy = p.bounds
        xs = np.linspace(minx, maxx, 7)
        ys = np.linspace(miny, maxy, 7)
        X, Y = np.meshgrid(xs, ys)
        q = np.column_stack([X.ravel(), Y.ravel()])
        q = q[shapely.contains_xy(p, q[:, 0], q[:, 1])]
        pts.append(q)
    return np.vstack(pts)


def _split_levels(spec: dict[str, Any], fp: geom.Footprint, rects: list[tuple[float, float, float, float]], E: float, R: float,
                  tanp: float, P: dict[str, Any], rng: np.random.Generator) -> tuple[list[tuple[float, float, float, float]], float, float]:
    """Pick the two-storey block of a house: (high rects, low eave, high eave) or ([], E, E)."""
    fp_area = fp.local_poly.area
    story = float(P["story_m"]["house"])
    if spec.get("hgrid"):
        # lidar heights per rectangle: p90 of the nDSM inside the rectangle (shrunk 0.6 m)
        tops = []
        for r in rects:
            reg = box(r[0] + 0.6, r[1] + 0.6, r[2] - 0.6, r[3] - 0.6)
            v = hgrid_values(spec, fp, reg) if not reg.is_empty and reg.area > 1.0 else np.zeros(0)
            tops.append(float(np.percentile(v, 90)) if len(v) >= 4 else np.nan)
        tops = np.array(tops)
        if np.isfinite(tops).sum() >= 2:
            hi = np.nanmax(tops)
            lo = np.nanmin(tops)
            if hi - lo > 1.8 and lo < 5.6:
                thr = (hi + lo) / 2
                high = [r for r, t in zip(rects, tops, strict=True) if np.isfinite(t) and t >= thr]
                hcore = shapely.union_all([box(*r) for r in high])
                lowv = hgrid_values(spec, fp, fp.local_poly.difference(hcore.buffer(0.6)))
                highv = hgrid_values(spec, fp, hcore.buffer(-0.6))
                e_low = float(np.clip(np.percentile(lowv, 8) if len(lowv) >= 4 else 2.9, 2.4, 4.2))
                e_high = float(np.clip(np.percentile(highv, 8) if len(highv) >= 4 else e_low + story, e_low + 2.0, 12.0))
                return high, e_low, e_high
            return [], E, E
    if int(spec.get("levels") or 1) < 2 or E < 4.6:
        return [], E, E
    if rng.random() > float(P["two_level_share"]):
        return [], E, E
    # heuristic: the biggest rectangle (and rectangles mostly inside it) is the two-storey block;
    # what is left (garage wing, front projections) is one storey, if it is a real wing (>= 15 %)
    big = rects[0]
    bx = box(*big)
    high = [r for r in rects if box(*r).intersection(bx).area >= 0.7 * (r[2] - r[0]) * (r[3] - r[1])]
    hcore = shapely.union_all([box(*r) for r in high])
    low = fp.local_poly.difference(hcore)
    parts = [g for g in getattr(low, "geoms", [low]) if g.geom_type == "Polygon" and g.area > 6.0]
    low_area = sum(g.area for g in parts)
    if low_area < 0.15 * fp_area or hcore.area < 55.0:
        return [], E, E
    widest = max((min(g.minimum_rotated_rectangle.bounds[2] - g.minimum_rotated_rectangle.bounds[0],
                      g.minimum_rotated_rectangle.bounds[3] - g.minimum_rotated_rectangle.bounds[1]) for g in parts), default=0)
    if widest < 2.5:
        return [], E, E
    e_low = 2.75 + 0.3
    return high, e_low, max(E, e_low + story)


# ---------------------------------------------------------------------------
# geometry emitters
# ---------------------------------------------------------------------------


class Builder:
    def __init__(self, spec: dict[str, Any], lod: int, P: dict[str, Any]):
        self.spec = spec
        self.lod = lod
        self.P = P
        self.bid = int(spec["id"])
        self.rng = np.random.default_rng(_hash(P["seed"], self.bid))
        self.soup = Soup()
        ring = np.asarray(spec["ring"], float)
        ring = np.column_stack([ring[:, 0], -ring[:, 1]])  # scene (x, z) -> plan (x, y = north)
        holes = [np.column_stack([np.asarray(h, float)[:, 0], -np.asarray(h, float)[:, 1]]) for h in spec.get("holes") or []]
        pr = orient(Polygon(ring), 1.0)
        ring = np.asarray(pr.exterior.coords)[:-1]
        holes = [np.asarray(orient(Polygon(h), 1.0).exterior.coords)[:-1] for h in holes if len(h) >= 3]
        step = (0.9 if spec["type"] == "house" else 0.6) if lod == 0 else 1.6
        self.fp = geom.regularize(ring, holes, step_m=step, min_iou=0.86 if lod == 0 else 0.8)
        self.floor = float(spec["floor_y"]) + (float(P["slab_m"]) if spec["type"] == "house" else 0.0)
        self.spec_floor = float(spec["floor_y"])
        self.base = float(spec.get("ground_min_y", spec["floor_y"])) - 0.3
        self.btype = spec["type"]
        self.plan = plan_roof({**spec, "floor_y": self.spec_floor}, self.fp, lod, P, self.rng)
        self.style = make_style(spec, P, self.plan.levels[-1].flat, self.rng)
        self.front = self._front_dir()
        self.garages: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (door start, door end, outward normal) in plan
        self.entries: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # ---- helpers ----
    def to3(self, loc: np.ndarray, z: np.ndarray | float) -> np.ndarray:
        p = self.fp.to_plan(loc)
        z = np.broadcast_to(np.asarray(z, float), (len(p),))
        return np.column_stack([p, z])

    def _front_dir(self) -> np.ndarray | None:
        sts = self.spec.get("streets") or []
        c = np.array(self.fp.plan_poly.centroid.coords[0])
        if self.spec.get("driveway"):
            d = self.spec["driveway"]
            v = np.array([d[0], -d[1]]) - c
            if np.hypot(*v) > 1e-3:
                return v / np.hypot(*v)
        if sts:
            s = sts[0]
            v = np.array([s["x"], -s["z"]]) - c
            if np.hypot(*v) > 1e-3:
                return v / np.hypot(*v)
        return None

    def walls(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Outline edges in plan coords (a, b) with outward normal on the right (CCW exterior, CW holes)."""
        lp = self.fp.local_poly
        out = []
        for ring in [np.asarray(lp.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in lp.interiors]:
            P2 = self.fp.to_plan(ring)
            n = len(P2)
            for i in range(n):
                a, b = P2[i], P2[(i + 1) % n]
                if np.hypot(*(b - a)) > 0.05:
                    out.append((a, b))
        return out

    def core_profile(self, a: np.ndarray, b: np.ndarray, levels: list[Level] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Roof line (wall top) along plan segment a->b, probing just inside (left of a->b in plan)."""
        la, lb = self.fp.to_local(np.vstack([a, b]))
        d = lb - la
        inward = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-9)
        pcs = [p for lv in (levels or self.plan.levels) for p in lv.pieces_core]
        T, Z = geom.envelope_profile(pcs, la, lb, inward, probe=2e-3)
        Z = np.where(np.isnan(Z), np.nanmax(np.r_[Z, self.floor + 2.5]), Z)
        return T, Z

    # ---- walls ----
    def emit_wall(self, a: np.ndarray, b: np.ndarray, T: np.ndarray, Ztop: np.ndarray, zbot: np.ndarray | float,
                  cuts: list[tuple[float, float, float, float]] | None = None) -> None:
        """Wall strip a->b from zbot (scalar or per-T array) up to the profile; cuts = [(s0, s1, z0, z1)] holes."""
        e = b - a
        L = float(np.hypot(*e))
        t = e / L
        st = self.style
        S = T * L
        Zb = np.broadcast_to(np.asarray(zbot, float), S.shape) if np.ndim(zbot) else np.full(S.shape, float(zbot))
        brk = sorted(set(np.round(S, 6).tolist()) | {round(c, 6) for cu in (cuts or []) for c in cu[:2] if 0 < c < L})
        tris = []
        for k in range(len(brk) - 1):
            s0, s1 = brk[k], brk[k + 1]
            if s1 - s0 < 1e-4:
                continue
            ztop0, ztop1 = _interp_right(S, Ztop, s0), _interp_left(S, Ztop, s1)
            zb0, zb1 = _interp_right(S, Zb, s0), _interp_left(S, Zb, s1)
            p0, p1 = a + t * s0, a + t * s1
            cut = next((cu for cu in (cuts or []) if cu[0] - 1e-6 <= s0 and s1 <= cu[1] + 1e-6), None)
            spans = [(zb0, zb1, ztop0, ztop1)]
            if cut is not None:
                spans = [(zb0, zb1, cut[2], cut[2]), (cut[3], cut[3], ztop0, ztop1)]
            for za0, za1, zt0, zt1 in spans:
                if zt0 - za0 < 1e-3 and zt1 - za1 < 1e-3:
                    continue
                tris.append(quad_tris(np.r_[p0, za0], np.r_[p1, za1], np.r_[p1, zt1], np.r_[p0, zt0]))
        if tris:
            Pt = np.concatenate(tris)
            self.soup.add(Pt, facade_uv(Pt, a, t, self.floor), MAT_WALL, st.wall_var, st.wall_rgb, self.bid)

    # ---- roof ----
    def emit_roofs(self) -> None:
        st = self.style
        lvls = self.plan.levels
        fd = float(self.P["fascia_m"])
        for li, lv in enumerate(lvls):
            faces = geom.envelope_faces(lv.pieces_ov)
            groups: dict[tuple[int, int, int], list[Any]] = {}
            planes: dict[tuple[int, int, int], np.ndarray] = {}
            for f in faces:
                key = (int(round(f.plane[0] * 1000)), int(round(f.plane[1] * 1000)), int(round(f.plane[2] * 100)))
                groups.setdefault(key, []).append(f.poly)
                planes[key] = f.plane
            clip = lv.clip
            ov_union = safe_union([Polygon(p.region) for p in lv.pieces_ov]).simplify(1e-3) if lv.pieces_ov else None
            for key, polys in groups.items():
                g = safe_union(polys)
                if clip is not None:
                    g = safe_diff(g, clip)
                g = _clean(g, 0.01)
                if g is None:
                    continue
                pl = planes[key]
                flatf = abs(pl[0]) < 1e-6 and abs(pl[1]) < 1e-6
                tri2 = geom.triangulate(g)
                if not len(tri2):
                    continue
                Pw = self._lift(tri2, pl)
                nrm = self._plane_normal(pl)
                mat, var, rgb = st.roof_mat, st.roof_var, st.roof_rgb
                if flatf and st.roof_mat == MAT_TILE:
                    mat, var, rgb = MAT_FLAT, 3, FLAT_RGB["flat_modbit"]
                self.soup.add(Pw, roof_uv(Pw, nrm, flatf or mat == MAT_FLAT), mat, var, rgb, self.bid)
                if self.lod == 0 and st.solar and li == len(lvls) - 1 and not flatf:
                    self._solar(g, pl, nrm)
            # fascia along the outer roof edge
            if self.plan.ov > 0 and ov_union is not None and not lv.flat:
                self._fascia(lv, ov_union, clip, fd)

    def _plane_normal(self, pl: np.ndarray) -> np.ndarray:
        g = np.array([pl[0], pl[1]])
        c, s = math.cos(self.fp.theta), math.sin(self.fp.theta)
        gx, gy = c * g[0] - s * g[1], s * g[0] + c * g[1]
        n = np.array([-gx, -gy, 1.0])
        return n / np.linalg.norm(n)

    def _lift(self, tri2: np.ndarray, pl: np.ndarray) -> np.ndarray:
        flat = tri2.reshape(-1, 2)
        z = flat @ pl[:2] + pl[2]
        return self.to3(flat, z).reshape(-1, 3, 3)

    def _fascia(self, lv: Level, ov_union: Polygon | MultiPolygon, clip: Any, fd: float) -> None:
        st = self.style
        ov = self.plan.ov
        tris: list[np.ndarray] = []
        stris: list[np.ndarray] = []
        uvs: list[np.ndarray] = []
        for p in getattr(ov_union, "geoms", [ov_union]):
            p = orient(p, 1.0)  # exterior CCW, holes CW: the roof is always on the left
            for ring in [p.exterior] + list(p.interiors):
                c = np.asarray(ring.coords)
                for i in range(len(c) - 1):
                    la, lb = c[i], c[i + 1]
                    if np.hypot(*(lb - la)) < 0.02:
                        continue
                    seg = LineString([la, lb])
                    if clip is not None:
                        seg2 = seg.difference(clip.buffer(0.02))
                        segs = [s for s in getattr(seg2, "geoms", [seg2]) if s.geom_type == "LineString" and s.length > 0.02]
                    else:
                        segs = [seg]
                    for s in segs:
                        sa, sb = np.asarray(s.coords[0]), np.asarray(s.coords[-1])
                        if np.dot(sb - sa, lb - la) < 0:
                            sa, sb = sb, sa
                        d = sb - sa
                        inward = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-9)
                        T, Z = geom.envelope_profile(lv.pieces_ov, sa, sb, inward, probe=2e-3)
                        if np.isnan(Z).any():
                            continue
                        A3 = self.to3(np.vstack([sa, sb]), 0.0)[:, :2]
                        a, b = A3[0], A3[1]
                        L = float(np.hypot(*(b - a)))
                        t = (b - a) / max(L, 1e-9)
                        S = T * L
                        for k in range(len(S) - 1):
                            if S[k + 1] - S[k] < 1e-4:
                                continue
                            p0, p1 = a + t * S[k], a + t * S[k + 1]
                            z0, z1 = Z[k], Z[k + 1]
                            q = quad_tris(np.r_[p0, z0 - fd], np.r_[p1, z1 - fd], np.r_[p1, z1], np.r_[p0, z0])
                            tris.append(q)
                            uvs.append(facade_uv(q, a, t, self.floor))
                            # soffit: from the fascia bottom back to the wall line, parallel to the roof
                            lk0 = sa + (sb - sa) * T[k] + inward * ov
                            lk1 = sa + (sb - sa) * T[k + 1] + inward * ov
                            zi = geom.envelope_height(lv.pieces_ov, np.vstack([lk0, lk1]))
                            if np.isnan(zi).any():
                                zi = np.array([z0, z1])
                            n3 = np.r_[self.fp.to_plan(np.vstack([lk0, lk1]))]
                            sq = quad_tris(np.r_[p1, z1 - fd], np.r_[p0, z0 - fd], np.r_[n3[0], zi[0] - fd], np.r_[n3[1], zi[1] - fd])
                            stris.append(sq)
        if tris:
            self.soup.add(np.concatenate(tris), np.concatenate(uvs), MAT_TRIM, 0, st.trim_rgb, self.bid)
        if stris:
            Ps = np.concatenate(stris)
            self.soup.add(Ps, Ps[..., :2] / 3.0, MAT_TRIM, 0, st.soffit_rgb, self.bid)

    def _solar(self, g: Polygon | MultiPolygon, pl: np.ndarray, nrm: np.ndarray) -> None:
        """PV array on a roof face whose downslope faces south-ish (Blender -Y)."""
        if nrm[1] > -0.25:  # face must look south (normal toward -Y = scene +z)
            return
        # work in roof-plane 2D coords (u along eave, v up slope), in plan meters
        e = np.array([-nrm[1], nrm[0], 0.0])
        e /= np.linalg.norm(e)
        s = np.cross(nrm, e)
        if s[2] < 0:
            s, e = -s, -e
        poly_parts = [p for p in getattr(g, "geoms", [g]) if p.area > 12]
        if not poly_parts:
            return
        p = max(poly_parts, key=lambda q: q.area)
        ring = self.to3(np.asarray(p.exterior.coords)[:-1], np.asarray(p.exterior.coords)[:-1] @ pl[:2] + pl[2])
        U = np.column_stack([ring @ e, ring @ s])
        up = Polygon(U).buffer(-0.7, join_style=2)
        if up.is_empty or up.area < 6:
            return
        up = max(getattr(up, "geoms", [up]), key=lambda q: q.area)
        minx, miny, maxx, maxy = up.bounds
        # shrink the bbox until it fits (panels are rectangular arrays)
        for _ in range(12):
            rb = box(minx, miny, maxx, maxy)
            if up.contains(rb.buffer(-0.01)):
                break
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            minx, maxx = cx - (maxx - minx) * 0.45, cx + (maxx - minx) * 0.45
            miny, maxy = cy - (maxy - miny) * 0.47, cy + (maxy - miny) * 0.47
        else:
            return
        if (maxx - minx) < 2.0 or (maxy - miny) < 1.6:
            return
        # cap: real arrays are rows of ~1.0 x 1.7 m modules
        w = min(maxx - minx, 1.04 * int(min(maxx - minx, 9.0) / 1.04))
        h = min(maxy - miny, 1.7 * max(1, int((maxy - miny) / 1.7)))
        if w < 2.0:
            return
        cx = (minx + maxx) / 2
        u0, u1, v0, v1 = cx - w / 2, cx + w / 2, miny, miny + h
        # back to 3D: point = a*e + b*s + c*n ; the plane offset along n
        off = float(ring[0] @ nrm)
        lift = 0.12

        def P3(u: float, v: float, dl: float) -> np.ndarray:
            return u * e + v * s + (off + dl) * nrm

        c = [P3(u0, v0, lift), P3(u1, v0, lift), P3(u1, v1, lift), P3(u0, v1, lift)]
        q = quad_tris(*c)
        uv = np.stack([(q @ e - u0) / 1.04 / 4 * 4 / 4, (q @ s - v0) / 4.0], axis=-1)
        self.soup.add(q, np.stack([q @ e / 4.0, q @ s / 4.0], axis=-1), MAT_TILE, TILE_VARIANTS.index("solar_panel"), TILE_RGB["solar_panel"], self.bid)
        del uv
        # frame sides
        cb = [P3(u0, v0, 0.02), P3(u1, v0, 0.02), P3(u1, v1, 0.02), P3(u0, v1, 0.02)]
        sides = []
        for i in range(4):
            sides.append(quad_tris(cb[i], cb[(i + 1) % 4], c[(i + 1) % 4], c[i]))
        Ps = np.concatenate(sides)
        self.soup.add(Ps, Ps[..., :2] / 3.0, MAT_TRIM, 0, (70, 72, 76), self.bid)

    # ---- flat roofs: parapet ----
    def emit_parapets(self, lv: Level) -> None:
        st = self.style
        par = float(self.P["parapet_m"].get(self.btype, 0.5))
        thick = 0.25
        z0, z1 = lv.z_wall, lv.z_wall + par
        core = lv.core
        inner = core.buffer(-thick, join_style=2)
        tris = []
        uvs = []
        # inner faces (facing the roof)
        for p in getattr(inner, "geoms", [inner]):
            if p.is_empty or p.geom_type != "Polygon":
                continue
            p = orient(p, -1.0)  # exterior CW, holes CCW: the roof is on the right (quad normals face it)
            for ring in [p.exterior] + list(p.interiors):
                c = np.asarray(ring.coords)
                P2 = self.fp.to_plan(c)
                for i in range(len(P2) - 1):
                    a, b = P2[i], P2[i + 1]
                    L = np.hypot(*(b - a))
                    if L < 0.02:
                        continue
                    t = (b - a) / L
                    q = quad_tris(np.r_[a, z0], np.r_[b, z0], np.r_[b, z1], np.r_[a, z1])
                    tris.append(q)
                    uvs.append(facade_uv(q, a, t, self.floor))
        if tris:
            self.soup.add(np.concatenate(tris), np.concatenate(uvs), MAT_WALL, st.wall_var, st.wall_rgb, self.bid)
        # cap
        ring = core.difference(inner) if not inner.is_empty else core
        ring = _clean(ring, 0.001)
        if ring is not None:
            t2 = geom.triangulate(ring)
            Pc = self.to3(t2.reshape(-1, 2), z1).reshape(-1, 3, 3)
            self.soup.add(Pc, Pc[..., :2] / 3.0, MAT_TRIM, 0, st.trim_rgb, self.bid)

    # ---- openings ----
    def plan_openings(self, walls: list[tuple[np.ndarray, np.ndarray]], tops: list[tuple[np.ndarray, np.ndarray]]) -> dict[int, Any]:
        """Garage door and entry door positions: {wall index: [(kind, s0, s1, z0, z1)]}."""
        res: dict[int, list[tuple[str, float, float, float, float]]] = {}
        if self.btype not in ("house",) and not (self.btype == "other" and str(self.spec.get("tag") or "") in ("garage", "garages")):
            return res
        if self.front is None:
            return res
        P = self.P
        low_core = None
        if self.plan.two_level:
            low_core = self.fp.local_poly.difference(self.plan.levels[1].core)
        cands = []
        for i, (a, b) in enumerate(walls):
            e = b - a
            L = float(np.hypot(*e))
            n = np.array([e[1], -e[0]]) / L
            facing = float(n @ self.front)
            if facing < 0.55 or L < 5.4:
                continue
            T, Z = tops[i]
            if float(np.min(Z)) - self.floor < P["garage_door_h_m"] + 0.25:
                continue
            mid = (a + b) / 2
            # closeness to the street: project the centroid->mid along the front direction
            c = np.array(self.fp.plan_poly.centroid.coords[0])
            ahead = float((mid - c) @ self.front)
            lowb = 0.0
            if low_core is not None:
                lm = self.fp.to_local(mid[None])[0] - 0.3 * (self.fp.to_local((mid + n)[None])[0] - self.fp.to_local(mid[None])[0])
                if low_core.buffer(0.05).contains(shapely.Point(lm)):
                    lowb = 2.0
            cands.append((facing * 2 + ahead * 0.35 + lowb + min(L, 10) * 0.08, i, L))
        if not cands:
            return res
        cands.sort(reverse=True)
        _, gi, L = cands[0]
        a, b = walls[gi]
        gw = P["garage_door_w_m"]
        doors = [(0.0, gw)]
        if self.style.three_car and L >= gw + P["garage_single_w_m"] + 1.6:
            doors = [(0.0, gw), (gw + 0.6, gw + 0.6 + P["garage_single_w_m"])]
        span = doors[-1][1]
        side = 1 if (_hash(self.bid, "gside") % 2) else -1
        if L - span < 2.0:
            s0 = (L - span) / 2
        else:
            s0 = 0.55 if side < 0 else L - span - 0.55
        h = P["garage_door_h_m"]
        res[gi] = [("garage" if k == 0 else "garage1", s0 + d0, s0 + d1, self.floor, self.floor + h) for k, (d0, d1) in enumerate(doors)]
        # entry door: on the garage wall's free side if long enough, else the best other street-facing wall
        dw = P["entry_door_w_m"]
        free = (s0 + span + 0.9, L - 0.6) if side < 0 else (0.6, s0 - 0.9)
        placed = False
        if free[1] - free[0] >= dw + 1.0:
            sd = (free[0] + free[1]) / 2 if free[1] - free[0] < 4 else (free[0] + 1.2 if side < 0 else free[1] - 1.2 - dw)
            res[gi].append(("entry", sd, sd + dw, self.floor, self.floor + P["entry_door_h_m"]))
            placed = True
        if not placed:
            best = None
            gm = (a + b) / 2
            for i, (a2, b2) in enumerate(walls):
                if i == gi:
                    continue
                e = b2 - a2
                L2 = float(np.hypot(*e))
                n2 = np.array([e[1], -e[0]]) / L2
                fac = float(n2 @ self.front)
                if L2 < dw + 1.2 or fac < -0.1:
                    continue
                T, Z = tops[i]
                if float(np.min(Z)) - self.floor < P["entry_door_h_m"] + 0.3:
                    continue
                dist = float(np.hypot(*((a2 + b2) / 2 - gm)))
                sc = fac * 3 - dist * 0.25
                if best is None or sc > best[0]:
                    best = (sc, i, L2)
            if best is not None:
                _, i, L2 = best
                sd = (L2 - dw) / 2
                res.setdefault(i, []).append(("entry", sd, sd + dw, self.floor, self.floor + P["entry_door_h_m"]))
        return res

    def emit_garage(self, a: np.ndarray, b: np.ndarray, kind: str, s0: float, s1: float, z0: float, z1: float) -> None:
        st = self.style
        e = b - a
        L = float(np.hypot(*e))
        t = e / L
        n = np.array([t[1], -t[0]])
        d = float(self.P["garage_recess_m"])
        p0, p1 = a + t * s0, a + t * s1
        self.garages.append((p0, p1, n))
        q0, q1 = p0 - n * d, p1 - n * d
        # door panel at the back of the recess, UVs onto the cell's door rect
        door = quad_tris(np.r_[q0, z0], np.r_[q1, z0], np.r_[q1, z1], np.r_[q0, z1])
        w = s1 - s0
        if kind == "garage" and w > 3.0:
            var, ox = 0, 0.56
            uo = ox / 3.0
            scale = 4.88 / w
        else:
            var, ox = 1, 6.13
            uo = ox / 3.0
            scale = 2.44 / w
        loc_s = ((door[..., 0] - q0[0]) * t[0] + (door[..., 1] - q0[1]) * t[1]) * scale
        uv = np.stack([uo + loc_s / 3.0, (door[..., 2] - z0) * (2.13 / max(z1 - z0, 1e-3)) / 3.0], axis=-1)
        self.soup.add(door, uv, MAT_GARAGE, var, GARAGE_RGB, self.bid)
        # reveals + head (stucco) + threshold
        rev = [quad_tris(np.r_[p0, z0], np.r_[q0, z0], np.r_[q0, z1], np.r_[p0, z1]),
               quad_tris(np.r_[q0, z0], np.r_[p0, z0], np.r_[p1, z0], np.r_[q1, z0]),
               quad_tris(np.r_[q1, z0], np.r_[p1, z0], np.r_[p1, z1], np.r_[q1, z1]),
               quad_tris(np.r_[p0, z1], np.r_[q0, z1], np.r_[q1, z1], np.r_[p1, z1])]
        Pr = np.concatenate(rev)
        self.soup.add(Pr, facade_uv(Pr, a, t, self.floor), MAT_WALL, st.wall_var, st.wall_rgb, self.bid)
        # driveway apron lip: trim header above the door
        hd = quad_tris(np.r_[p0 - t * 0.12 + n * 0.04, z1], np.r_[p1 + t * 0.12 + n * 0.04, z1],
                       np.r_[p1 + t * 0.12 + n * 0.04, z1 + 0.22], np.r_[p0 - t * 0.12 + n * 0.04, z1 + 0.22])
        self.soup.add(hd, facade_uv(hd, a, t, self.floor), MAT_TRIM, 0, st.trim_rgb, self.bid)

    def emit_entry(self, a: np.ndarray, b: np.ndarray, s0: float, s1: float, z0: float, z1: float, top_min: float) -> None:
        st = self.style
        e = b - a
        L = float(np.hypot(*e))
        t = e / L
        n = np.array([t[1], -t[0]])
        d = 0.12
        p0, p1 = a + t * s0, a + t * s1
        self.entries.append((p0, p1, n))
        q0, q1 = p0 - n * d, p1 - n * d
        door = quad_tris(np.r_[q0, z0], np.r_[q1, z0], np.r_[q1, z1], np.r_[q0, z1])
        self.soup.add(door, facade_uv(door, a, t, self.floor), MAT_TRIM, 0, st.door_rgb, self.bid)
        rev = [quad_tris(np.r_[p0, z0], np.r_[q0, z0], np.r_[q0, z1], np.r_[p0, z1]),
               quad_tris(np.r_[q1, z0], np.r_[p1, z0], np.r_[p1, z1], np.r_[q1, z1]),
               quad_tris(np.r_[p0, z1], np.r_[q0, z1], np.r_[q1, z1], np.r_[p1, z1]),
               quad_tris(np.r_[q0, z0], np.r_[p0, z0], np.r_[p1, z0], np.r_[q1, z0])]
        Pr = np.concatenate(rev)
        self.soup.add(Pr, facade_uv(Pr, a, t, self.floor), MAT_TRIM, 0, st.trim_rgb, self.bid)
        # porch: concrete stoop + (when the wall is tall enough) a small hip roof on two posts
        pw0, pw1 = s0 - 0.9, s1 + 0.9
        pw0, pw1 = max(pw0, 0.05), min(pw1, L - 0.05)
        depth = 1.25
        A0, A1 = a + t * pw0, a + t * pw1
        B0, B1 = A0 + n * depth, A1 + n * depth
        zs = self.floor - 0.02
        stoop = [quad_tris(np.r_[A0, zs], np.r_[A1, zs], np.r_[B1, zs], np.r_[B0, zs])[:, ::-1]]
        Ps = np.concatenate(stoop)
        self.soup.add(Ps, facade_uv(Ps, a, t, self.floor), MAT_TRIM, 0, (196, 190, 178), self.bid)
        zr = self.floor + 2.75
        if top_min - zr > 0.9 and self.btype == "house":
            tanp = self.plan.tanp
            ov = 0.3
            C0, C1 = A0 - t * ov, A1 + t * ov
            D0, D1 = C0 + n * (depth + ov), C1 + n * (depth + ov)
            rise = (depth + ov) * tanp * 0.8
            ztop = zr + rise
            # shed-hip: back edge at the wall (high), three sloped faces
            K0, K1 = C0 + t * (depth + ov) * 0.8, C1 - t * (depth + ov) * 0.8
            if np.dot(K1 - K0, t) <= 0:
                K0 = K1 = (C0 + C1) / 2
            faces = [
                (np.r_[D0, zr], np.r_[D1, zr], np.r_[K1, ztop], np.r_[K0, ztop]),
                (np.r_[C0, zr], np.r_[D0, zr], np.r_[K0, ztop], None),
                (np.r_[D1, zr], np.r_[C1, zr], np.r_[K1, ztop], None),
            ]
            for f in faces:
                if f[3] is None:
                    Pt = np.array([f[:3]])
                else:
                    Pt = quad_tris(*f)
                nrm = np.cross(Pt[0, 1] - Pt[0, 0], Pt[0, 2] - Pt[0, 0])
                self.soup.add(Pt, roof_uv(Pt, nrm, False), st.roof_mat, st.roof_var, st.roof_rgb, self.bid)
            # fascia + soffit
            fd = 0.18
            fas = [quad_tris(np.r_[C0, zr - fd], np.r_[D0, zr - fd], np.r_[D0, zr], np.r_[C0, zr]),
                   quad_tris(np.r_[D0, zr - fd], np.r_[D1, zr - fd], np.r_[D1, zr], np.r_[D0, zr]),
                   quad_tris(np.r_[D1, zr - fd], np.r_[C1, zr - fd], np.r_[C1, zr], np.r_[D1, zr]),
                   quad_tris(np.r_[C0, zr - fd], np.r_[C1, zr - fd], np.r_[D1, zr - fd], np.r_[D0, zr - fd])]
            Pf = np.concatenate(fas)
            self.soup.add(Pf, facade_uv(Pf, a, t, self.floor), MAT_TRIM, 0, st.trim_rgb, self.bid)

    def emit_window(self, a: np.ndarray, t: np.ndarray, n: np.ndarray, s0: float, s1: float, z0: float, z1: float,
                    mullion: bool = True, sill: bool = True, frame: float = 0.09) -> None:
        st = self.style
        p0, p1 = a + t * s0, a + t * s1
        ofs_f, ofs_g = 0.03, 0.045
        fr = quad_tris(np.r_[p0 - t * frame + n * ofs_f, z0 - frame], np.r_[p1 + t * frame + n * ofs_f, z0 - frame],
                       np.r_[p1 + t * frame + n * ofs_f, z1 + frame], np.r_[p0 - t * frame + n * ofs_f, z1 + frame])
        vf = 0.05  # vinyl frame visible inside the surround
        gl = quad_tris(np.r_[p0 + t * vf + n * ofs_g, z0 + vf], np.r_[p1 - t * vf + n * ofs_g, z0 + vf],
                       np.r_[p1 - t * vf + n * ofs_g, z1 - vf], np.r_[p0 + t * vf + n * ofs_g, z1 - vf])
        self.soup.add(fr, facade_uv(fr, a, t, self.floor), MAT_TRIM, 0, st.trim_rgb, self.bid)
        self.soup.add(gl, glass_uv(gl, a + t * (s0 + vf), t, z0 + vf, (s1 - s0) - 2 * vf, (z1 - z0) - 2 * vf, panes=2 if s1 - s0 > 1.1 else 1),
                      MAT_GLASS, 0, GLASS_RGB, self.bid)
        parts = []
        if mullion and (s1 - s0) > 1.1:
            m = (s0 + s1) / 2
            q0, q1 = a + t * (m - 0.03), a + t * (m + 0.03)
            parts.append(quad_tris(np.r_[q0 + n * 0.055, z0], np.r_[q1 + n * 0.055, z0], np.r_[q1 + n * 0.055, z1], np.r_[q0 + n * 0.055, z1]))
        if sill:
            zs = z0 - frame
            e0, e1 = p0 - t * (frame + 0.06), p1 + t * (frame + 0.06)
            o = n * 0.1
            parts.append(quad_tris(np.r_[e0 + o, zs - 0.07], np.r_[e1 + o, zs - 0.07], np.r_[e1, zs + 0.01], np.r_[e0, zs + 0.01]))
        if parts:
            Pp = np.concatenate(parts)
            self.soup.add(Pp, facade_uv(Pp, a, t, self.floor), MAT_TRIM, 0, st.trim_rgb, self.bid)

    def windows_on_wall(self, wi: int, a: np.ndarray, b: np.ndarray, T: np.ndarray, Z: np.ndarray,
                        reserved: list[tuple[str, float, float, float, float]], zbase: np.ndarray | float | None = None,
                        step: bool = False) -> None:
        e = b - a
        L = float(np.hypot(*e))
        if L < 1.6:
            return
        t = e / L
        n = np.array([t[1], -t[0]])
        S = T * L
        rng = np.random.default_rng(_hash(self.P["seed"], self.bid, wi, step))
        btype = self.btype
        story = float(self.P["story_m"].get(btype, 3.0))
        facing = float(n @ self.front) if self.front is not None else 0.0
        has_garage = any(r[0].startswith("garage") for r in reserved)
        if btype in ("commercial",):
            self._storefront(wi, a, t, n, L, S, Z, facing, rng)
            return
        if btype == "school":
            self._ribbon(a, t, n, L, S, Z, story, rng)
            return
        if btype == "other" and L < 6:
            return
        nfl = max(1, int(self.spec.get("levels") or 1))
        if btype == "other":
            nfl = 1
        for k in range(nfl + 1):
            zf = self.floor + k * story
            if btype == "house":
                kinds = [("slider", 1.8, 1.35, 0.85), ("pair", 1.7, 1.5, 0.75), ("small", 0.8, 1.0, 1.25), ("tall", 1.0, 1.7, 0.55)]
                if k == 0 and facing > 0.5:
                    kinds = [("picture", 2.4, 1.6, 0.6), ("slider", 1.8, 1.4, 0.8), ("tall", 1.0, 2.0, 0.3)]
                elif k == 0 and facing < -0.5:
                    kinds = [("patio", 2.4, 2.05, 0.02), ("slider", 1.8, 1.2, 0.95), ("pair", 1.6, 1.5, 0.75)]
            elif btype == "apartments":
                kinds = [("slider", 1.8, 1.4, 0.8), ("pair", 1.5, 1.5, 0.75), ("patio", 2.1, 2.05, 0.05)]
            else:
                kinds = [("small", 1.2, 1.0, 1.2)]
            # this floor exists along the wall where the top leaves room for a window head + 0.3 m
            if k >= 1 and k >= nfl:
                break
            # choose window count and positions
            spacing = 3.8 if facing > 0.5 else (4.6 if facing < -0.5 else 8.0)
            if btype == "apartments":
                spacing = 3.2
            avail = L - 1.2
            if avail < 1.0 or L < (2.6 if abs(facing) > 0.5 else 4.0):
                continue
            nwin = int(max(0, round(avail / spacing + rng.uniform(-0.3, 0.3))))
            if L >= 3.2 and nwin == 0:
                nwin = 1 if rng.random() < (0.75 if facing > 0.5 else 0.35) else 0
            if k == 0 and has_garage:
                nwin = 0  # the garage wall: no ground-floor windows
            if nwin == 0:
                continue
            kind = kinds[int(rng.integers(len(kinds)))]
            for j in range(nwin):
                c = 0.6 + avail * (j + 0.5) / nwin
                kd = kind if rng.random() < 0.7 else kinds[int(rng.integers(len(kinds)))]
                _, w, h, sill = kd
                w = min(w, avail / nwin - 0.5)
                if w < 0.55:
                    continue
                s0, s1 = c - w / 2, c + w / 2
                z0, z1 = zf + sill, zf + sill + h
                # clearance from reserved openings
                if any(not (s1 + 0.35 < r[1] or s0 - 0.35 > r[2]) and not (z1 < r[3] - 0.1 or z0 > r[4] + 0.1) for r in reserved):
                    continue
                top = min(_interp_left(S, Z, s0), _interp_right(S, Z, s1), float(np.interp((s0 + s1) / 2, S, Z)))
                if z1 + 0.3 > top:
                    continue
                if zbase is not None:
                    zb = float(np.max(np.atleast_1d(zbase))) if np.ndim(zbase) else float(zbase)
                    if z0 - 0.12 < zb + 0.25:
                        continue
                self.emit_window(a, t, n, s0, s1, z0, z1, mullion=kd[0] in ("patio", "picture") and facing > 0.5,
                                 sill=facing > 0.3 and kd[0] != "patio")

    def _storefront(self, wi: int, a: np.ndarray, t: np.ndarray, n: np.ndarray, L: float, S: np.ndarray, Z: np.ndarray,
                    facing: float, rng: np.random.Generator) -> None:
        st = self.style
        top = float(np.min(Z))
        if L < 4.0:
            return
        front = facing > 0.3 or (self.front is None and L > 12)
        if front:
            z0, z1 = self.floor + 0.35, min(self.floor + 3.0, top - 1.2)
            if z1 - z0 < 1.6:
                return
            s0, s1 = 0.6, L - 0.6
            gl = quad_tris(np.r_[a + t * s0 + n * 0.03, z0], np.r_[a + t * s1 + n * 0.03, z0], np.r_[a + t * s1 + n * 0.03, z1], np.r_[a + t * s0 + n * 0.03, z1])
            nb_ = max(1, int(round((s1 - s0) / 3.0)))
            self.soup.add(gl, glass_uv(gl, a + t * s0, t, z0, s1 - s0, z1 - z0, bay_m=(s1 - s0) / nb_ / 1.0), MAT_GLASS, 0, GLASS_RGB, self.bid)
            parts = []
            nb = max(1, int(round((s1 - s0) / 3.0)))
            for j in range(nb + 1):
                m = s0 + (s1 - s0) * j / nb
                q0, q1 = a + t * (m - 0.06), a + t * (m + 0.06)
                parts.append(quad_tris(np.r_[q0 + n * 0.06, z0], np.r_[q1 + n * 0.06, z0], np.r_[q1 + n * 0.06, z1], np.r_[q0 + n * 0.06, z1]))
            # base kick plate + transom bar
            for za, zb in ((self.floor, z0), (z1, z1 + 0.12)):
                parts.append(quad_tris(np.r_[a + t * s0 + n * 0.06, za], np.r_[a + t * s1 + n * 0.06, za], np.r_[a + t * s1 + n * 0.06, zb], np.r_[a + t * s0 + n * 0.06, zb]))
            Pp = np.concatenate(parts)
            self.soup.add(Pp, facade_uv(Pp, a, t, self.floor), MAT_TRIM, 0, (58, 60, 64), self.bid)
            # sign band (proud box face) and a canopy over the glass
            zs0, zs1 = z1 + 0.45, min(z1 + 1.35, top - 0.15)
            if zs1 - zs0 > 0.4:
                o = n * 0.18
                sb = [quad_tris(np.r_[a + t * s0 + o, zs0], np.r_[a + t * s1 + o, zs0], np.r_[a + t * s1 + o, zs1], np.r_[a + t * s0 + o, zs1]),
                      quad_tris(np.r_[a + t * s0, zs0], np.r_[a + t * s1, zs0], np.r_[a + t * s1 + o, zs0], np.r_[a + t * s0 + o, zs0]),
                      quad_tris(np.r_[a + t * s0 + o, zs1], np.r_[a + t * s1 + o, zs1], np.r_[a + t * s1, zs1], np.r_[a + t * s0, zs1])]
                Ps = np.concatenate(sb)
                self.soup.add(Ps, facade_uv(Ps, a, t, self.floor), MAT_TRIM, 0, st.trim_rgb, self.bid)
            o = n * 1.2
            zc = z1 + 0.3
            can = [quad_tris(np.r_[a + t * s0, zc], np.r_[a + t * s1, zc], np.r_[a + t * s1 + o, zc], np.r_[a + t * s0 + o, zc]),
                   quad_tris(np.r_[a + t * s0, zc + 0.15], np.r_[a + t * s0 + o, zc + 0.15], np.r_[a + t * s1 + o, zc + 0.15], np.r_[a + t * s1, zc + 0.15])[:, ::-1][:, ::-1],
                   quad_tris(np.r_[a + t * s0 + o, zc], np.r_[a + t * s1 + o, zc], np.r_[a + t * s1 + o, zc + 0.15], np.r_[a + t * s0 + o, zc + 0.15])]
            Pc = np.concatenate(can)
            self.soup.add(Pc, facade_uv(Pc, a, t, self.floor), MAT_TRIM, 0, (70, 72, 76), self.bid)
        elif L > 8 and rng.random() < 0.5:
            # service door
            s0 = L * rng.uniform(0.2, 0.8)
            dd = quad_tris(np.r_[a + t * s0 + n * 0.03, self.floor], np.r_[a + t * (s0 + 1.0) + n * 0.03, self.floor],
                           np.r_[a + t * (s0 + 1.0) + n * 0.03, self.floor + 2.15], np.r_[a + t * s0 + n * 0.03, self.floor + 2.15])
            self.soup.add(dd, facade_uv(dd, a, t, self.floor), MAT_TRIM, 0, (150, 146, 138), self.bid)

    def _ribbon(self, a: np.ndarray, t: np.ndarray, n: np.ndarray, L: float, S: np.ndarray, Z: np.ndarray, story: float,
                rng: np.random.Generator) -> None:
        if L < 5:
            return
        top = float(np.min(Z))
        nfl = max(1, int(self.spec.get("levels") or 1))
        for k in range(nfl):
            z0, z1 = self.floor + k * story + 0.95, self.floor + k * story + 2.4
            if z1 + 0.4 > top:
                break
            nb = int((L - 1.0) // 3.0)
            for j in range(nb):
                if rng.random() < 0.2:
                    continue
                s0 = 0.5 + (L - 1.0 - nb * 3.0) / 2 + j * 3.0 + 0.2
                self.emit_window(a, t, n, s0, s0 + 2.6, z0, z1, mullion=True, sill=False, frame=0.06)

    def emit_hvac(self, lv: Level) -> None:
        area = lv.core.area
        nunits = int(area // float(self.P["hvac_m2_per_unit"]))
        if nunits == 0:
            return
        inner = lv.core.buffer(-3.0)
        if inner.is_empty:
            return
        rng = np.random.default_rng(_hash(self.P["seed"], self.bid, "hvac"))
        minx, miny, maxx, maxy = inner.bounds
        placed = 0
        tries = 0
        z0 = lv.z_wall
        boxes = []
        while placed < min(nunits, 14) and tries < nunits * 6:
            tries += 1
            cx, cy = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if not inner.contains(shapely.Point(cx, cy)):
                continue
            w, d, h = rng.uniform(1.2, 2.4), rng.uniform(1.0, 1.6), rng.uniform(0.9, 1.5)
            c = self.fp.to_plan(np.array([[cx, cy]]))[0]
            ax = self.fp.to_plan(np.array([[cx + w / 2, cy]]))[0] - c
            ay = self.fp.to_plan(np.array([[cx, cy + d / 2]]))[0] - c
            boxes += box_tris(c, ax, ay, z0, z0 + h)
            placed += 1
        if boxes:
            Pb = np.concatenate(boxes)
            uv = np.stack([(Pb[..., 0] + Pb[..., 1]) / 4.0, Pb[..., 2] / 4.0], axis=-1)
            self.soup.add(Pb, uv, MAT_FLAT, 5, (176, 180, 182), self.bid)

    def emit_stone(self, a: np.ndarray, b: np.ndarray, reserved: list[tuple[str, float, float, float, float]]) -> None:
        """Ledgestone wainscot (0.9 m) on the street walls of some houses (skips door openings)."""
        e = b - a
        L = float(np.hypot(*e))
        t = e / L
        n = np.array([t[1], -t[0]])
        cuts = sorted((r[1], r[2]) for r in reserved)
        segs, s = [], 0.0
        for c0, c1 in cuts:
            if c0 - s > 0.3:
                segs.append((s, c0))
            s = max(s, c1)
        if L - s > 0.3:
            segs.append((s, L))
        z0, z1 = self.base + 0.25, self.floor + 0.9
        tris = []
        for s0, s1 in segs:
            o = n * 0.04
            tris.append(quad_tris(np.r_[a + t * s0 + o, z0], np.r_[a + t * s1 + o, z0], np.r_[a + t * s1 + o, z1], np.r_[a + t * s0 + o, z1]))
            tris.append(quad_tris(np.r_[a + t * s0, z1], np.r_[a + t * s0 + o, z1], np.r_[a + t * s1 + o, z1], np.r_[a + t * s1, z1])[:, ::-1][:, ::-1])
        if tris:
            Pt = np.concatenate(tris)
            self.soup.add(Pt, facade_uv(Pt, a, t, self.floor), MAT_WALL, 6, STONE_RGB, self.bid)

    # ---- main ----
    def build(self) -> Soup:
        walls = self.walls()
        lvls = self.plan.levels
        tops = [self.core_profile(a, b) for a, b in walls]
        flat_top = lvls[-1].flat
        par = float(self.P["parapet_m"].get(self.btype, 0.5)) if flat_top else 0.0
        reserved = self.plan_openings(walls, tops) if self.lod == 0 else {}
        for i, (a, b) in enumerate(walls):
            T, Z = tops[i]
            Zw = Z + par if flat_top else Z
            cuts = [(r[1], r[2], r[3], r[4]) for r in reserved.get(i, [])]
            self.emit_wall(a, b, T, Zw, self.base, cuts)
            for r in reserved.get(i, []):
                if r[0].startswith("garage"):
                    self.emit_garage(a, b, r[0], r[1], r[2], r[3], r[4])
                elif r[0] == "entry":
                    self.emit_entry(a, b, r[1], r[2], r[3], r[4], float(np.min(Z)))
            if self.lod == 0:
                self.windows_on_wall(i, a, b, T, Z, reserved.get(i, []))
                if self.style.stone and self.front is not None:
                    e = b - a
                    nrm = np.array([e[1], -e[0]]) / np.hypot(*e)
                    if nrm @ self.front > 0.55:
                        self.emit_stone(a, b, reserved.get(i, []))
        # step walls between the one-storey wing and the two-storey block
        if self.plan.two_level:
            low, high = lvls[0], lvls[1]
            fb = self.fp.local_poly.boundary.buffer(0.03)
            for p in getattr(high.core, "geoms", [high.core]):
                c = np.asarray(orient(p, 1.0).exterior.coords)
                for i in range(len(c) - 1):
                    seg = LineString([c[i], c[i + 1]]).difference(fb)
                    for s in getattr(seg, "geoms", [seg]):
                        if s.is_empty or s.geom_type != "LineString" or s.length < 0.2:
                            continue
                        la, lb = np.asarray(s.coords[0]), np.asarray(s.coords[-1])
                        if np.dot(lb - la, c[i + 1] - c[i]) < 0:
                            la, lb = lb, la
                        a, b = self.fp.to_plan(np.vstack([la, lb]))
                        T, Zt = self.core_profile(a, b, [high])
                        # base: the low roof just outside the high block (probe to the right = outward)
                        Tb, Zb = self.core_profile(b, a, [low])
                        zb = np.interp(T, 1.0 - Tb[::-1], Zb[::-1]) - 0.05
                        self.emit_wall(a, b, T, Zt, zb)
                        if self.lod == 0:
                            self.windows_on_wall(1000 + i, a, b, T, Zt, [], zbase=zb, step=True)
        self.emit_roofs()
        if flat_top:
            self.emit_parapets(lvls[-1])
            if self.lod == 0 and self.btype in ("commercial", "school", "apartments", "other") and lvls[-1].core.area > 300:
                self.emit_hvac(lvls[-1])
        return self.soup


def _interp_left(S: np.ndarray, Z: np.ndarray, s: float) -> float:
    """Profile value approaching s from the left (jumps are repeated S entries)."""
    idx = np.searchsorted(S, s, side="left")
    if idx < len(S) and abs(S[idx] - s) < 1e-6:
        return float(Z[idx])
    return float(np.interp(s, S, Z))


def _interp_right(S: np.ndarray, Z: np.ndarray, s: float) -> float:
    idx = np.searchsorted(S, s, side="right") - 1
    if idx >= 0 and abs(S[idx] - s) < 1e-6:
        return float(Z[idx])
    return float(np.interp(s, S, Z))


def safe_union(gs: list[Any]) -> Any:
    try:
        return shapely.union_all(gs, grid_size=1e-5)
    except shapely.errors.GEOSException:
        return shapely.union_all([shapely.make_valid(g).buffer(0) for g in gs])


def safe_diff(a: Any, b: Any) -> Any:
    try:
        return shapely.difference(a, b, grid_size=1e-5)
    except shapely.errors.GEOSException:
        return shapely.make_valid(a).buffer(0).difference(shapely.make_valid(b).buffer(0))


def safe_inter(a: Any, b: Any) -> Any:
    try:
        return shapely.intersection(a, b, grid_size=1e-5)
    except shapely.errors.GEOSException:
        return shapely.make_valid(a).buffer(0).intersection(shapely.make_valid(b).buffer(0))


def _clean(g: Any, min_area: float) -> Polygon | MultiPolygon | None:
    if g is None or g.is_empty:
        return None
    parts = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area >= min_area]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


@dataclass
class Result:
    soup: Soup
    eave_h: float
    ridge_h: float
    roof_type: str
    two_level: bool
    rect: bool
    garages: list[list[float]] = field(default_factory=list)
    entries: list[list[float]] = field(default_factory=list)


def _doors_scene(doors: list[tuple[np.ndarray, np.ndarray, np.ndarray]], floor: float) -> list[list[float]]:
    """(start, end, normal) in plan coords -> [x, z, nx, nz, width, floor_y] in scene coords."""
    out = []
    for p0, p1, n in doors:
        m = (p0 + p1) / 2
        out.append([round(float(m[0]), 2), round(float(-m[1]), 2), round(float(n[0]), 3), round(float(-n[1]), 3),
                    round(float(np.hypot(*(p1 - p0))), 2), round(floor, 2)])
    return out


def build_building(spec: dict[str, Any], lod: int, P: dict[str, Any] | None = None) -> Result:
    params = {**DEFAULT_PARAMS, **(P or {})}
    b = Builder(spec, lod, params)
    soup = b.build()
    rt = b.plan.roof_type if not b.plan.levels[-1].flat else "flat"
    return Result(soup, round(b.plan.eave_h, 2), round(b.plan.ridge_h, 2), rt, b.plan.two_level, b.fp.rect,
                  _doors_scene(b.garages, b.floor), _doors_scene(b.entries, b.floor))


def box_fallback(spec: dict[str, Any], lod: int) -> Soup:
    """Last resort for a footprint the roof builder cannot handle: an extruded flat box."""
    soup = Soup()
    ring = np.asarray(spec["ring"], float)
    ring = np.column_stack([ring[:, 0], -ring[:, 1]])
    p = orient(Polygon(ring).buffer(0), 1.0)
    if p.geom_type != "Polygon":
        p = max(p.geoms, key=lambda g: g.area)
    ring = np.asarray(p.exterior.coords)[:-1]
    z0 = float(spec.get("ground_min_y", spec["floor_y"])) - 0.3
    z1 = float(spec["floor_y"]) + float(spec.get("eave_h") or 3.0)
    bid = int(spec["id"])
    floor = float(spec["floor_y"])
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        L = np.hypot(*(b - a))
        if L < 1e-3:
            continue
        q = quad_tris(np.r_[a, z0], np.r_[b, z0], np.r_[b, z1], np.r_[a, z1])
        soup.add(q, facade_uv(q, a, (b - a) / L, floor), MAT_WALL, 1, (220, 210, 190), bid)
    t2 = geom.triangulate(p)
    Pt = np.concatenate([t2.reshape(-1, 2), np.full((t2.size // 2, 1), z1)], axis=1).reshape(-1, 3, 3)
    soup.add(Pt, Pt[..., :2] / 4.0, MAT_FLAT, 0, FLAT_RGB["flat_tpo"], bid)
    return soup
