"""Hero campus models (Del Norte HS, Design39, 4S Commons) from real footprints + site layout.

Input: one entry of blender/build/hero_sites.json (see blender/extract_hero_sites.py and
blender/hero_layout.py): buildings (real OSM footprints with height / levels), the
planar ground partition (parking, drives, plazas, walks, fields, track, courts,
landscape), paint lines, and point features.

Output: Blender objects in LOCAL meters, origin = hero center, Blender +X east,
+Y north, +Z up (glTF: +x east, +y up, +z south). All faces use palette keys, so
the export carries per-face vertex colors (COLOR_0) on 3-4 class materials
(hero_matte / hero_glass / hero_metal / hero_glow), which the pipeline's hero
loader keeps when it bakes the model into the building tiles. Each vertex also
carries the material-atlas ids `_MAT` / `_VARIANT` and atlas UVs (rdlib/atlas.py),
so the client shades heroes with the same stucco / roof / ground atlases as houses.

Architecture vocabulary (SoCal Mediterranean-modern): stucco walls over a stone /
concrete base band, punched or banded windows with dark glass and trim, flat
roofs behind parapets with rooftop HVAC, standing-seam metal entry canopies and
covered walkways (schools), tile hip roofs / tile mansard bands, arcades and
fabric awnings on shop fronts (4S Commons), solar carport canopies over school
parking rows.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
from shapely.geometry import Point, Polygon

from .mesh import Part, box, cylinder, hip_roof

STYLE = {
    "del_norte_hs": {"walls": ["stucco_sand", "stucco_warm", "stucco_tan", "stucco_cream"], "accent": ["accent_red", "stucco_clay"],
                     "base": "stone_base", "trim": "trim_dark", "glass": "glass_dark", "roof": "roof_tpo",
                     "canopy": "roof_metal", "floor_h": 3.9, "win": "band"},
    "design39": {"walls": ["panel_white", "stucco_white", "stucco_gray"], "accent": ["accent_orange", "accent_teal"],
                 "base": "concrete_dark", "trim": "metal_dark", "glass": "glass_dark", "roof": "roof_tpo",
                 "canopy": "steel_white", "floor_h": 4.1, "win": "curtain"},
    "4s_commons": {"walls": ["stucco_cream", "stucco_warm", "stucco_sand", "stucco_olive", "stucco_clay"],
                   "accent": ["awning_green", "awning_red", "awning_cream", "awning_blue"],
                   "base": "stone_base", "trim": "trim_dark", "glass": "glass_dark", "roof": "roof_tpo",
                   "canopy": "roof_tile", "floor_h": 5.0, "win": "storefront"},
}

SURF = {
    "asphalt": "asphalt", "asphalt_light": "asphalt_light", "concrete": "concrete", "concrete_dark": "concrete_dark",
    "pavers": "pavers", "grass": "grass", "turf_field": "turf_field", "track_red": "track_red", "court_blue": "court_blue",
    "dirt_infield": "dirt_infield", "rubber_play": "rubber_play", "mulch": "mulch", "landscape": "landscape",
}


# ---------------------------------------------------------------------------
# fast accumulation
# ---------------------------------------------------------------------------


class Acc:
    """Collect many small Parts and merge once (Part.add re-stacks arrays on every call)."""

    def __init__(self) -> None:
        self.parts: list[Part] = []

    def __iadd__(self, p: Part) -> Acc:
        if len(p.F):
            self.parts.append(p)
        return self

    def raw(self, V: list[tuple[float, float, float]], F: list[list[int]], M: list[str]) -> None:
        if F:
            self.parts.append(Part(np.asarray(V, dtype=float), F, M))

    def merge(self) -> Part:
        if not self.parts:
            return Part()
        Vs, F, M = [], [], []
        off = 0
        for p in self.parts:
            Vs.append(p.V)
            F.extend([[i + off for i in f] for f in p.F])
            M.extend(p.M)
            off += len(p.V)
        return Part(np.vstack(Vs), F, M)


def _ccw(ring: list[tuple[float, float]]) -> bool:
    a = 0.0
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        a += x0 * y1 - x1 * y0
    return a > 0


def _clean(ring: list[list[float]] | list[tuple[float, float]], tol: float = 0.05) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for x, y in ring:
        if not out or math.hypot(x - out[-1][0], y - out[-1][1]) > tol:
            out.append((float(x), float(y)))
    if len(out) > 2 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= tol:
        out.pop()
    return out


def offset_ring(ring: list[tuple[float, float]], d: float, max_miter: float = 2.5) -> list[tuple[float, float]]:
    """Offset a CCW ring by d (positive = outward) with clamped miters; keeps the vertex count."""
    n = len(ring)
    P = np.array(ring, float)
    out = []
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        e1, e2 = b - a, c - b
        n1 = np.array([e1[1], -e1[0]]) / (np.linalg.norm(e1) + 1e-12)
        n2 = np.array([e2[1], -e2[0]]) / (np.linalg.norm(e2) + 1e-12)
        m = n1 + n2
        ml = np.linalg.norm(m)
        if ml < 1e-6:
            m = n1
            k = 1.0
        else:
            m = m / ml
            k = 1.0 / max(float(m @ n1), 1.0 / max_miter)
        out.append(tuple(b + m * d * k))
    return out


def fill_polygon(outer: list[tuple[float, float]], holes: list[list[tuple[float, float]]], z: float, key: str) -> Part:
    rings = [outer] + [h for h in holes if len(h) >= 3]
    vecs = [[Vector((x, y, 0.0)) for x, y in r] for r in rings]
    tris = tessellate_polygon(vecs)
    flat = [p for r in rings for p in r]
    V = np.array([(x, y, z) for x, y in flat], dtype=float)
    F = []
    for t in tris:
        a, b, c = t
        pa, pb, pc = flat[a], flat[b], flat[c]
        cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
        if abs(cross) < 1e-9:
            continue
        F.append([a, b, c] if cross > 0 else [a, c, b])
    return Part(V, F, [key] * len(F))


def wall_band(ring: list[tuple[float, float]], z0: float, z1: float, key: str, outward: bool = True) -> Part:
    """Vertical faces along a closed ring (CCW ring + outward=True -> faces point out)."""
    n = len(ring)
    V = [(x, y, z0) for x, y in ring] + [(x, y, z1) for x, y in ring]
    F = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    if not outward:
        F = [f[::-1] for f in F]
    return Part(np.array(V, dtype=float), F, [key] * n)


class EdgeQuads:
    """Accumulates facade quads on one wall edge a -> c (outward normal nrm)."""

    def __init__(self, a: np.ndarray, c: np.ndarray, nrm: np.ndarray):
        self.a, self.c, self.nrm = a, c, nrm
        self.V: list = []
        self.F: list[list[int]] = []
        self.M: list[str] = []

    def q(self, t0: float, t1: float, za: float, zb: float, off: float, key: str) -> None:
        v, f, k = quad_on_edge(self.a, self.c, self.nrm, t0, t1, za, zb, off, key)
        o = len(self.V)
        self.V.extend(v)
        self.F.append([o + j for j in f])
        self.M.append(k)

    def flush(self, acc: Acc) -> None:
        acc.raw(self.V, self.F, self.M)


def quad_on_edge(a: np.ndarray, b: np.ndarray, nrm: np.ndarray, t0: float, t1: float, z0: float, z1: float,
                 off: float, key: str) -> tuple[list, list[int], str]:
    p0 = a + (b - a) * t0 + nrm * off
    p1 = a + (b - a) * t1 + nrm * off
    return [(p0[0], p0[1], z0), (p1[0], p1[1], z0), (p1[0], p1[1], z1), (p0[0], p0[1], z1)], [0, 1, 2, 3], key


# ---------------------------------------------------------------------------
# campus pieces
# ---------------------------------------------------------------------------


def ground(site: dict[str, Any], acc: Acc) -> None:
    lay = site["layout"]
    for s in lay["surfaces"]:
        key = SURF.get(s["cls"], "concrete")
        outer = _clean(s["exterior"])
        holes = [_clean(h) for h in s["holes"]]
        if len(outer) < 3:
            continue
        if not _ccw(outer):
            outer = outer[::-1]
        holes = [h[::-1] if _ccw(h) else h for h in holes if len(h) >= 3]
        z = float(s["z"])
        acc += fill_polygon(outer, holes, z, key)
        side = "curb" if key in ("concrete", "pavers", "concrete_dark", "mulch", "grass", "landscape") else key
        acc += wall_band(outer, -0.3, z, side)
        for h in holes:
            acc += wall_band(h[::-1], -0.3, z, side, outward=False)
    for s in lay["paint"]:
        outer = _clean(s["exterior"], 0.01)
        if len(outer) < 3:
            continue
        if not _ccw(outer):
            outer = outer[::-1]
        holes = [_clean(h, 0.01) for h in s["holes"]]
        holes = [h[::-1] if _ccw(h) else h for h in holes if len(h) >= 3]
        acc += fill_polygon(outer, holes, float(s["z"]) + 0.004, s["cls"])


def building(b: dict[str, Any], front: list[int], hero: str, rng: np.random.Generator, acc: Acc) -> dict[str, int]:
    st = STYLE[hero]
    ext = _clean(b["exterior"])
    if len(ext) < 3:
        return {}
    if not _ccw(ext):
        ext = ext[::-1]
        n0 = len(ext)
        front = [(n0 - 2 - i) % n0 for i in front]  # edge i (v_i -> v_i+1) after reversal
    holes = [_clean(h) for h in b["holes"]]
    holes = [h[::-1] if _ccw(h) else h for h in holes if len(h) >= 3]
    H = float(b["height_m"])
    tag = b["building"]
    poly = Polygon(ext, holes)
    area = poly.area
    z0 = -0.8
    # open-sided structures: covered walkways, lunch shelters, canopies
    if tag in ("shelter", "roof", "carport") or (H < 3.0 and area < 400):
        hh = min(max(H, 3.2), 6.0)
        acc += fill_polygon(ext, holes, hh, st["canopy"] if hero != "4s_commons" else "roof_metal")
        acc += wall_band(ext, hh - 0.35, hh, "trim_white" if hero != "design39" else "steel_white")
        under = fill_polygon(ext, holes, hh - 0.35, "metal_dark")
        under.F = [f[::-1] for f in under.F]
        acc += under
        L = Polygon(ext).exterior
        n_post = max(4, int(L.length // 6.5))
        for k in range(n_post):
            p = L.interpolate(k * L.length / n_post)
            q = Polygon(ext).buffer(-0.5)
            pt = q.exterior.interpolate(q.exterior.project(p)) if not q.is_empty and hasattr(q, "exterior") else p
            acc += box(0.25, 0.25, hh - 0.35, "steel_white").moved(pt.x, pt.y, 0.0)
        return {"canopies": 1}
    walls = st["walls"]
    wall = walls[int(rng.integers(len(walls)))]
    is_store = hero == "4s_commons"
    fh = st["floor_h"]
    levels = b.get("levels")
    n_fl = int(levels) if levels else max(1, int(round((H - 1.2) / fh)))
    fh_eff = (H - 1.0) / n_fl
    para = 0.9 if not is_store else 1.2
    hip = False
    if is_store:
        mrr = poly.minimum_rotated_rectangle
        hip = area < 1700 and area / max(mrr.area, 1e-6) > 0.88 and rng.random() < 0.7
    top = H if hip else H + para
    # walls (exterior + courtyards)
    acc += wall_band(ext, z0, top, wall)
    for h in holes:
        acc += wall_band(h[::-1], z0, top, wall, outward=False)
    acc += wall_band(offset_ring(ext, 0.04), z0, 0.75, st["base"])  # base band
    # facade openings
    n = len(ext)
    P = np.array(ext)
    win = st["win"]
    for i in range(n):
        a, c = P[i], P[(i + 1) % n]
        e = c - a
        L = float(np.linalg.norm(e))
        if L < 3.5:
            continue
        nrm = np.array([e[1], -e[0]]) / L
        m0 = min(0.9, L * 0.15) / L
        if i in front and is_store:
            # storefront: tall glass with mullions, arcade or awning above
            eq = EdgeQuads(a, c, nrm)
            q = eq.q
            q(m0, 1 - m0, 0.35, 3.6, 0.03, st["glass"])
            q(m0, 1 - m0, 3.6, 3.85, 0.05, st["trim"])
            nm = max(1, int(L // 3.2))
            for k in range(1, nm):
                t = m0 + (1 - 2 * m0) * k / nm
                q(t - 0.06 / L, t + 0.06 / L, 0.35, 3.6, 0.05, "metal_dark")
            eq.flush(acc)
            if rng.random() < 0.5 and L > 12:  # arcade: projecting roof on columns
                d = 3.0
                corner = [a + nrm * 0.0 + e * m0 * 0.5, c - e * m0 * 0.5, c - e * m0 * 0.5 + nrm * d, a + e * m0 * 0.5 + nrm * d]
                ring = [(float(p[0]), float(p[1])) for p in corner]
                if not _ccw(ring):
                    ring = ring[::-1]
                acc += fill_polygon(ring, [], 4.6, "roof_tile")
                acc += wall_band(ring, 4.1, 4.6, wall)
                ncol = max(2, int(L // 4.5))
                for k in range(ncol + 1):
                    t = m0 * 0.5 + (1 - m0) * k / ncol
                    pc = a + e * t + nrm * (d - 0.35)
                    acc += box(0.45, 0.45, 4.1, wall, top=wall).moved(pc[0], pc[1], 0.0)
            else:  # fabric awning
                aw = st["accent"][int(rng.integers(len(st["accent"])))]
                p0, p1 = a + e * m0, c - e * m0
                V = [(p0[0] + nrm[0] * 0.05, p0[1] + nrm[1] * 0.05, 3.9), (p1[0] + nrm[0] * 0.05, p1[1] + nrm[1] * 0.05, 3.9),
                     (p1[0] + nrm[0] * 1.6, p1[1] + nrm[1] * 1.6, 3.25), (p0[0] + nrm[0] * 1.6, p0[1] + nrm[1] * 1.6, 3.25)]
                acc.raw(V, [[0, 1, 2, 3][::-1]], [aw])
            continue
        eq = EdgeQuads(a, c, nrm)
        q2 = eq.q
        if is_store:  # side / rear walls of shops: few high windows, service doors
            if L > 8 and rng.random() < 0.5:
                q2(0.2, 0.2 + 1.0 / L, 0.0, 2.3, 0.03, "metal_dark")
            eq.flush(acc)
            continue
        for f in range(n_fl):
            zb = 0.75 + f * fh_eff
            if win == "curtain" and (i % 3 != 1 or L > 14):
                q2(m0, 1 - m0, zb + 0.35, zb + fh_eff - 0.35, 0.03, st["glass"])
                nm = max(1, int(L // 1.6))
                for k in range(1, nm):
                    t = m0 + (1 - 2 * m0) * k / nm
                    q2(t - 0.04 / L, t + 0.04 / L, zb + 0.35, zb + fh_eff - 0.35, 0.05, st["trim"])
                q2(m0, 1 - m0, zb + fh_eff - 0.45, zb + fh_eff - 0.3, 0.06, st["trim"])
            else:
                # punched windows: 1.8 m wide every 3.2 m (classroom wings), sill + head trim
                nw = max(1, int((L - 1.5) // 3.2))
                span = (L - 1.5) / nw
                for k in range(nw):
                    tc = (0.75 + span * (k + 0.5)) / L
                    w2 = min(1.0, span * 0.33) / L
                    q2(tc - w2, tc + w2, zb + 0.95, zb + fh_eff - 0.55, 0.03, st["glass"])
                    q2(tc - w2 - 0.08 / L, tc + w2 + 0.08 / L, zb + 0.85, zb + 0.95, 0.06, st["trim"] if f else "trim_white")
                if f == n_fl - 1 and rng.random() < 0.6:  # accent band under the parapet
                    acc_key = st["accent"][int(rng.integers(len(st["accent"])))]
                    q2(0.0, 1.0, H - 0.35, H + 0.15, 0.025, acc_key)
        if i in front and not is_store and L > 8:  # entry canopy / covered walkway along the front
            d = 3.2
            p0, p1 = a + e * m0, c - e * m0
            ring = [(float(x), float(y)) for x, y in (p0, p1, p1 + nrm * d, p0 + nrm * d)]
            if not _ccw(ring):
                ring = ring[::-1]
            acc += fill_polygon(ring, [], 3.6, st["canopy"])
            acc += wall_band(ring, 3.35, 3.6, "trim_white")
            ncol = max(2, int(L // 6))
            for k in range(ncol + 1):
                pc = p0 + (p1 - p0) * k / ncol + nrm * (d - 0.3)
                acc += box(0.22, 0.22, 3.35, "steel_white").moved(pc[0], pc[1], 0.0)
        eq.flush(acc)
    # roof
    if hip:
        cx, cy = poly.minimum_rotated_rectangle.centroid.coords[0]
        xs, ys = poly.minimum_rotated_rectangle.exterior.coords.xy
        e0 = math.dist((xs[0], ys[0]), (xs[1], ys[1]))
        e1 = math.dist((xs[1], ys[1]), (xs[2], ys[2]))
        ang = math.degrees(math.atan2(ys[1] - ys[0], xs[1] - xs[0]))
        roof = hip_roof(e0, e1, min(e0, e1) * 0.28, "roof_tile", overhang=0.7, fascia="trim_white").rotated_z(ang)
        acc += roof.moved(cx, cy, H)
        acc += fill_polygon(ext, holes, H - 0.05, "roof_flat")
        return {"hip": 1}
    inner = offset_ring(ext, -0.3)
    roof_key = ["roof_tpo", "roof_gravel", "roof_flat", "roof_tpo"][int(rng.integers(4))]
    if Polygon(inner).is_valid and Polygon(inner).area > area * 0.5:
        acc += fill_polygon(inner, [h for h in holes], H, roof_key)
        acc += wall_band(inner, H, top, wall, outward=False)
        # parapet cap (annulus ext -> inner)
        nn = len(ext)
        Vc = [(x, y, top) for x, y in ext] + [(x, y, top) for x, y in inner]
        Fc = [[i, (i + 1) % nn, nn + (i + 1) % nn, nn + i] for i in range(nn)]
        acc.raw(Vc, Fc, ["trim_white" if not is_store else "stucco_cream"] * nn)
    else:
        acc += fill_polygon(ext, holes, top, roof_key)
    if is_store and not hip:  # tile mansard on the shop fronts
        for i in front:
            a, c = P[i], P[(i + 1) % n]
            e = c - a
            L = float(np.linalg.norm(e))
            nrm = np.array([e[1], -e[0]]) / L
            p0, p1 = a - e / L * 0.4, c + e / L * 0.4
            V = [(p0[0] + nrm[0] * 0.6, p0[1] + nrm[1] * 0.6, H - 0.5), (p1[0] + nrm[0] * 0.6, p1[1] + nrm[1] * 0.6, H - 0.5),
                 (p1[0] - nrm[0] * 0.6, p1[1] - nrm[1] * 0.6, top + 0.6), (p0[0] - nrm[0] * 0.6, p0[1] - nrm[1] * 0.6, top + 0.6)]
            acc.raw(V, [[0, 1, 2, 3]], ["roof_tile"])
            if L > 20 and rng.random() < 0.6:  # tower element with a hip cap at the entry
                tc = a + e * 0.5 - nrm * 2.2
                tw = 5.0
                acc += box(tw, tw, top + 2.4, wall, top=wall).rotated_z(math.degrees(math.atan2(e[1], e[0]))).moved(tc[0], tc[1], 0)
                acc += hip_roof(tw, tw, 1.8, "roof_tile", overhang=0.5).rotated_z(math.degrees(math.atan2(e[1], e[0]))).moved(
                    tc[0], tc[1], top + 2.4)
    # rooftop HVAC units
    shrink = Polygon(inner).buffer(-2.5) if Polygon(inner).is_valid else Polygon()
    n_units = int(min(14, area / 260.0))
    placed = 0
    tries = 0
    if not shrink.is_empty:
        minx, miny, maxx, maxy = shrink.bounds
        while placed < n_units and tries < 80:
            tries += 1
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if shrink.contains(Point(x, y)):
                w, d = rng.uniform(1.4, 2.6), rng.uniform(1.8, 3.6)
                acc += box(w, d, rng.uniform(0.9, 1.5), "hvac", top="hvac").rotated_z(rng.uniform(0, 90)).moved(x, y, H)
                placed += 1
    return {"floors": n_fl}


def carport(c: dict[str, Any], acc: Acc) -> None:
    """Solar carport canopy: PV deck tilted ~6 deg on a row of steel columns (T-frame)."""
    L, W, H = float(c["length"]), float(c["width"]), float(c["height"])
    tilt = math.radians(6)
    p = Part()
    deck = box(L, W, 0.12, "solar_frame", top="solar_panel", bottom=True)
    deck = deck.rotated_x(-math.degrees(tilt)).moved(0, 0, H + W / 2 * math.sin(tilt))
    p += deck
    p += box(L, 0.3, 0.35, "pole_gray").moved(0, 0, H - 0.3)  # main beam
    ncol = max(2, int(L // 9.0) + 1)
    for k in range(ncol):
        x = -L / 2 + 1.0 + (L - 2.0) * k / (ncol - 1)
        p += box(0.35, 0.35, H - 0.3, "pole_gray").moved(x, 0, 0)
        p += box(0.25, W * 0.8, 0.25, "pole_gray").rotated_x(-math.degrees(tilt)).moved(x, 0, H - 0.1)
    ang = float(c["angle_deg"])
    # tilt the low edge toward the stall end (tilt_toward_deg is the stall-row normal)
    q = p.rotated_z(ang)
    side = math.radians(float(c["tilt_toward_deg"]))
    loc_normal = np.array([-math.sin(math.radians(ang)), math.cos(math.radians(ang))])
    if loc_normal @ np.array([math.cos(side), math.sin(side)]) < 0:
        q = p.rotated_z(ang + 180)
    acc += q.moved(c["x"], c["y"], 0.0)


def site_object(o: dict[str, Any], acc: Acc) -> None:
    t = o["type"]
    ang = float(o.get("angle", 0.0))
    p = Part()
    if t == "goal":  # soccer goal 7.3 x 2.44 m
        for sx in (-3.66, 3.66):
            p += box(0.12, 0.12, 2.44, "steel_white").moved(sx, 0, 0)
        p += box(7.44, 0.12, 0.12, "steel_white").moved(0, 0, 2.38)
        p += box(7.3, 0.04, 2.0, "net_dark").moved(0, 1.6, 0.0).rotated_x(0)
        p = p.rotated_z(-90)
    elif t == "hoop":
        p += cylinder(0.08, 3.4, "pole_dark", n=6)
        p += box(1.83, 0.06, 1.07, "steel_white").moved(0, 1.0, 2.9)
        p += box(0.6, 1.0, 0.08, "pole_dark").moved(0, 0.5, 3.3)
        p = p.rotated_z(-90)
    elif t == "tennis_net":
        L = o["size"][0]
        p += box(L, 0.05, 0.95, "net_dark")
        for sx in (-L / 2, L / 2):
            p += cylinder(0.05, 1.07, "pole_dark", n=6).moved(sx, 0, 0)
    elif t == "bleachers":
        L, D = o["size"]
        for k in range(6):
            p += box(L, D / 6, 0.45 * (k + 1), "aluminum", top="aluminum").moved(0, -D / 2 + D / 12 + k * D / 6, 0)
        p += box(L, 0.08, 1.2, "aluminum").moved(0, D / 2, 2.7)
    elif t == "play_structure":
        w, d = o["size"]
        for sx in (-w / 3, 0, w / 3):
            p += box(1.6, 1.6, 1.4, "play_blue", top="play_yellow").moved(sx, 0, 0)
            p += cylinder(0.07, 3.0, "play_red", n=5).moved(sx + 0.7, 0.7, 0)
        p += box(w * 0.9, 0.5, 0.3, "play_yellow").moved(0, -0.9, 1.6)
        p += box(0.8, 2.6, 0.12, "play_red").rotated_x(-30).moved(w / 3, -1.9, 0.9)
        V = [(-w / 2, -d / 2, 3.4), (w / 2, -d / 2, 3.9), (w / 2, d / 2, 3.2), (-w / 2, d / 2, 3.7)]
        p += Part(np.array(V, dtype=float), [[0, 1, 2, 3]], ["shade_sail"])
    else:
        return
    acc += p.rotated_z(ang).moved(o["x"], o["y"], 0.12)


def build_campus(site: dict[str, Any], atlas_man: dict[str, Any] | None = None) -> tuple[list, dict[str, Any]]:
    """Campus objects; with atlas_man (materials_manifest.json) every vertex also carries `_MAT`,
    `_VARIANT` and atlas TEXCOORD_0 (rdlib/atlas.py KEY_MAT), COLOR_0 stays the tint / fallback."""
    from . import bl

    rng = np.random.default_rng(sum(map(ord, site["id"])))
    acc = Acc()
    ground(site, acc)
    gpart = acc.merge()
    bacc = Acc()
    stats: dict[str, Any] = {"buildings": 0}
    fronts = site["layout"]["fronts"]
    for b, fr in zip(site["buildings"], fronts, strict=True):
        building(b, fr, site["id"], rng, bacc)
        stats["buildings"] += 1
    for c in site["layout"]["canopies"]:
        carport(c, bacc)
    for o in site["layout"]["objects"]:
        site_object(o, bacc)
    bpart = bacc.merge()
    objs = []
    if len(gpart.F):
        objs.append(bl.part_to_object(gpart, f"{site['id']}_ground", color_jitter=0.0, seed=1, atlas_man=atlas_man))
    if len(bpart.F):
        objs.append(bl.part_to_object(bpart, f"{site['id']}_buildings", color_jitter=0.03, seed=2, ao_ground=0.25,
                                        atlas_man=atlas_man))
    stats["tris"] = bl.tri_count(objs)
    stats["_trees"] = {
        "hero": site["id"], "frame": "local meters, x east, y north (glTF z = -y), origin = hero center",
        "lat": site["lat"], "lon": site["lon"], "rotation_deg": 0.0,
        "trees": site["layout"]["trees"], "lamps": site["layout"]["lamps"], "parked_cars": site["layout"]["parked"],
        # where real (lidar) trees must not stand: modeled buildings, sports surfaces and pavement
        "keepout": [{"cls": "building", "exterior": b["exterior"], "holes": b.get("holes", [])} for b in site["buildings"]]
        + [{"cls": sf["cls"], "exterior": sf["exterior"], "holes": sf.get("holes", [])} for sf in site["layout"]["surfaces"]
           if sf.get("cls") in ("turf_field", "track_red", "court_blue", "dirt_infield", "asphalt", "asphalt_light", "rubber_play")],
    }
    return objs, stats


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------


ATLAS_OF_MAT = {0: "facade_walls", 1: "roofs", 2: "roofs", 3: "facade_walls", 4: "facade_walls", 6: "ground"}


def atlas_preview_materials(obj: Any, man: dict[str, Any], mat_dir: Path) -> int:
    """Re-material an exported hero object for Cycles previews from its `_MAT` / `_VARIANT`
    attributes (the same lookup the client does). Faces with _MAT 7 keep the vertex-color class
    material. Returns the number of atlas materials used."""
    from . import atlas, bl

    me = obj.data
    if "_MAT" not in me.attributes:
        return 0
    n = len(me.vertices)
    mat = np.zeros(n, dtype=np.float32)
    var = np.zeros(n, dtype=np.float32)
    me.attributes["_MAT"].data.foreach_get("value", mat)
    me.attributes["_VARIANT"].data.foreach_get("value", var)
    slot_of = {m.name: i for i, m in enumerate(me.materials)}
    idx = np.zeros(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("material_index", idx)
    first = np.array([p.vertices[0] for p in me.polygons], dtype=np.int64)
    used = 0
    for (m, v), faces in _group_faces(mat[first].astype(int), var[first].astype(int)).items():
        if m not in ATLAS_OF_MAT:
            continue
        an = ATLAS_OF_MAT[m]
        cname = man["atlases"]["ground"]["cells"][v]["name"] if m == atlas.MAT_GROUND else man["materials"][str(m)]["variants"][v]
        cell = atlas.cell(man, an, cname)
        name = f"heroatlas_{m}_{v}"
        bm = bl.atlas_material(name, man, an, cname, mat_dir, tint_attr="Col" if cell.get("tintable") else None,
                               ao_mix=0.5 if m == atlas.MAT_GROUND else 0.6)
        if name not in slot_of:
            slot_of[name] = len(me.materials)
            me.materials.append(bm)
        idx[faces] = slot_of[name]
        used += 1
    me.polygons.foreach_set("material_index", idx)
    return used


def _group_faces(m: np.ndarray, v: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    key = m * 256 + v
    return {(int(k) // 256, int(k) % 256): np.nonzero(key == k)[0] for k in np.unique(key)}


def render_preview(site: dict[str, Any], objs: list, path: Path, samples: int = 32) -> None:
    import bpy

    from . import atlas, bl, foliage, materials, props, vehicles

    try:
        man = atlas.load_manifest(atlas.MAT_DIR)
        for o in objs:
            atlas_preview_materials(o, man, atlas.MAT_DIR)
    except FileNotFoundError:
        pass

    tex_dir = Path(bl.BUILD_DIR) / "textures"
    R = float(site["footprint_radius_m"])
    # surroundings: neutral suburban ground disc slightly below the site
    g = bl.ground_plane(R * 6, "#8C8A6A", z=-0.35, roughness=1.0)
    del g
    meshes: dict[str, Any] = {}

    def mesh_for(key: str):  # noqa: ANN202
        if key in meshes:
            return meshes[key]
        if key in foliage.SPECIES:
            built = foliage.build(key, tex_dir / f"{key}.png")
            bl.TEXTURES["foliage_atlas"] = built.tex
            if "foliage" in bpy.data.materials:
                bpy.data.materials["foliage"].name = f"foliage_{len(meshes)}"
            o = bl.part_to_object(built.part, f"proto_{key}")
            bl.set_custom_normals(o, built.normals)
            bl.set_vertex_ao(o, built.ao)
        elif key == "street_lamp":
            o = bl.part_to_object(props.street_lamp(), "proto_lamp", smooth_angle=45)
        else:
            o = bl.part_to_object(vehicles.VEHICLES[key][0](), f"proto_{key}", smooth_angle=40)
        o.hide_render = True
        o.hide_viewport = True
        meshes[key] = o
        return o

    coll = bpy.context.scene.collection
    for t in site["layout"]["trees"]:
        proto = mesh_for(t["species"])
        ob = bpy.data.objects.new("t", proto.data)
        ob.location = (t["x"], t["y"], 0.1)
        ob.rotation_euler = (0, 0, math.radians(t["rot_deg"]))
        ob.scale = (t["scale"],) * 3
        coll.objects.link(ob)
    for lp in site["layout"]["lamps"]:
        proto = mesh_for("street_lamp")
        ob = bpy.data.objects.new("l", proto.data)
        ob.location = (lp["x"], lp["y"], 0.1)
        ob.rotation_euler = (0, 0, math.radians(lp["rot_deg"] - 90))
        coll.objects.link(ob)
    rng = np.random.default_rng(5)
    kinds = ["car_sedan", "car_suv", "car_crossover_ev", "car_minivan", "car_pickup"]
    shares = np.array([vehicles.VEHICLES[k][1]["share"] for k in kinds], float)
    shares /= shares.sum()
    pc = materials.PAINT_COLORS
    cw = np.array([c["share"] for c in pc], float)
    cw /= cw.sum()
    painted: dict[tuple[str, int], Any] = {}
    for car in site["layout"]["parked"]:
        k = kinds[rng.choice(len(kinds), p=shares)]
        ci = int(rng.choice(len(pc), p=cw))
        if (k, ci) not in painted:
            proto = mesh_for(k)
            me = proto.data.copy()
            for i, m in enumerate(me.materials):
                if m.name == "paint":
                    mm = m.copy()
                    mm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*materials.hex_to_linear(pc[ci]["hex"]), 1.0)
                    me.materials[i] = mm
            painted[(k, ci)] = me
        ob = bpy.data.objects.new("c", painted[(k, ci)])
        ob.location = (car["x"], car["y"], 0.1)
        ob.rotation_euler = (0, 0, math.radians(car["heading_deg"] - 90))
        coll.objects.link(ob)
    bl.setup_render(1280, 760, samples=samples)
    bl.setup_world(sun_elev_deg=38, sun_azimuth_deg=135)
    d = max(R, 170.0) * 1.25
    bl.add_camera((d * 0.55, -d * 1.05, d * 0.85), (0.0, R * 0.05, 0.0), lens=34)
    bl.render(path)
