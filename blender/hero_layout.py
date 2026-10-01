"""2D site layout for hero campuses (main .venv: shapely). Used by extract_hero_sites.py.

Turns the raw site data (footprints, Overture land use, service roads / parking
aisles, footways) into a planar partition of ground surfaces (no overlaps, so the
Blender side can extrude them without z-fighting), paint lines (parking stalls,
track lanes, court and field lines), and point features (campus trees, parking lot
lights, parked-car stalls, solar carport canopies, hoops, goals, bleachers).

All coordinates are local meters: x east, y north, origin at the hero center.
Stall / aisle sizes and lot occupancy come from data/config/assumptions.yaml (props.*);
track lane (1.22 m), court and field sizes are fixed sport standards.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import assumption  # noqa: E402  (data/config/assumptions.yaml, props.*)

STALL_W = float(assumption("props.parking_stall_width_m"))  # 9 ft
STALL_L = float(assumption("props.parking_stall_length_m"))  # 18 ft
AISLE_W = float(assumption("props.parking_aisle_width_m"))  # 24 ft two-way aisle
OCCUPANCY = {k: float(assumption(f"props.parked_occupancy.{k}")) for k in ("school", "commercial")}
SERVICE_HALF = 3.5  # m, service road half width
FOOTWAY_HALF = 1.3  # m
LANE_W = 1.22  # m, track lane
PUBLIC_HALF = {"motorway": 16, "trunk": 14, "primary": 13, "secondary": 11, "tertiary": 8.5, "residential": 5.5,
               "unclassified": 5.0, "living_street": 4.5}


def polys(g) -> list[Polygon]:  # noqa: ANN001
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [x for x in getattr(g, "geoms", []) if isinstance(x, Polygon)]


def lines(g) -> list[LineString]:  # noqa: ANN001
    if g is None or g.is_empty:
        return []
    if isinstance(g, LineString):
        return [g]
    if isinstance(g, MultiLineString):
        return list(g.geoms)
    return [x for x in getattr(g, "geoms", []) if isinstance(x, LineString)]


def pj(p: Polygon, nd: int = 2) -> dict[str, Any]:
    p = p.buffer(0)
    return {
        "exterior": [[round(x, nd), round(y, nd)] for x, y in list(p.exterior.coords)[:-1]],
        "holes": [[[round(x, nd), round(y, nd)] for x, y in list(r.coords)[:-1]] for r in p.interiors],
    }


def rect_line(a: tuple[float, float], b: tuple[float, float], w: float) -> Polygon:
    return LineString([a, b]).buffer(w / 2, cap_style=2)


def _mrr(p: Polygon) -> tuple[float, float, float, float, float]:
    """Minimum rotated rectangle: (cx, cy, length, width, angle_deg of the long side)."""
    r = p.minimum_rotated_rectangle
    xs, ys = r.exterior.coords.xy
    e0 = math.dist((xs[0], ys[0]), (xs[1], ys[1]))
    e1 = math.dist((xs[1], ys[1]), (xs[2], ys[2]))
    if e0 >= e1:
        ang = math.degrees(math.atan2(ys[1] - ys[0], xs[1] - xs[0]))
        L, W = e0, e1
    else:
        ang = math.degrees(math.atan2(ys[2] - ys[1], xs[2] - xs[1]))
        L, W = e1, e0
    c = r.centroid
    return c.x, c.y, L, W, ang


def _oriented_rect(cx: float, cy: float, L: float, W: float, ang: float) -> Polygon:
    return rotate(box(cx - L / 2, cy - W / 2, cx + L / 2, cy + W / 2), ang, origin=(cx, cy))


def _outline_lines(p: Polygon, w: float) -> list[Polygon]:
    out = [LineString(p.exterior.coords).buffer(w / 2, cap_style=2, join_style=2)]
    return out


class Layout:
    def __init__(self, site: dict[str, Any], rng_seed: int = 7):
        self.site = site
        self.rng = np.random.default_rng(rng_seed)
        self.R = float(site["footprint_radius_m"])
        self.kind = site["type"]
        self.circle = Point(0, 0).buffer(self.R, 128)
        sp = Polygon(site["site"]["exterior"], site["site"]["holes"]).buffer(0)
        self.clip = sp.intersection(self.circle).buffer(0)
        self.bld = [Polygon(b["exterior"], b["holes"]).buffer(0) for b in site["buildings"]]
        self.keep = [Polygon(k["exterior"], k["holes"]).buffer(0) for k in site["keep_out"]]
        self.B = unary_union(self.bld + self.keep)
        self.surfaces: list[dict[str, Any]] = []
        self.paint: list[dict[str, Any]] = []
        self.taken = Polygon()
        self.trees: list[dict[str, Any]] = []
        self.lamps: list[dict[str, Any]] = []
        self.parked: list[dict[str, Any]] = []
        self.canopies: list[dict[str, Any]] = []
        self.objects: list[dict[str, Any]] = []

    # -- helpers -----------------------------------------------------------------
    def add_surface(self, cls: str, geom, z: float, take: bool = True) -> Any:  # noqa: ANN001
        g = geom.intersection(self.clip).difference(self.taken).buffer(0)
        g = unary_union([p for p in polys(g) if p.area > 2.0])
        if g.is_empty:
            return g
        if take:
            self.taken = unary_union([self.taken, g]).buffer(0)
        for p in polys(g.simplify(0.05)):
            if p.area > 2.0:
                self.surfaces.append({"cls": cls, "z": z, **pj(p)})
        return g

    def add_paint(self, cls: str, geom, z: float, within=None) -> None:  # noqa: ANN001
        g = geom.intersection(within if within is not None else self.clip)
        for p in polys(g):
            if p.area > 0.02:
                self.paint.append({"cls": cls, "z": z, **pj(p, 3)})

    def free(self, x: float, y: float, r: float, allowed) -> bool:  # noqa: ANN001
        pt = Point(x, y)
        if not allowed.contains(pt) or self.B.distance(pt) < r:
            return False
        return all(math.hypot(t["x"] - x, t["y"] - y) > r * 1.4 for t in self.trees[-400:])

    # -- builders ------------------------------------------------------------------
    def build(self) -> dict[str, Any]:
        segs = self.site["segments"]
        lu = self.site["landuse"]
        # public streets cut out: the pipeline's road ribbons show through
        pub = unary_union([LineString(s["coords"]).buffer(PUBLIC_HALF[s["class"]], cap_style=2)
                           for s in segs if s["class"] in PUBLIC_HALF and len(s["coords"]) > 1])
        self.taken = unary_union([self.B.buffer(0.3), pub]).buffer(0)
        # 1) sports + play surfaces
        self.sports(lu)
        # 2) parking lots from parking aisles, service drives
        aisles = [LineString(s["coords"]) for s in segs if s["subclass"] == "parking_aisle" and len(s["coords"]) > 1]
        service = [LineString(s["coords"]) for s in segs
                   if s["class"] == "service" and s["subclass"] != "parking_aisle" and len(s["coords"]) > 1]
        self.parking(aisles, service)
        # 3) plazas, walks, landscaping
        ped = unary_union([Polygon(l["exterior"], l["holes"]).buffer(0) for l in lu if l["class"] == "pedestrian"])
        walks = unary_union([LineString(s["coords"]).buffer(FOOTWAY_HALF, cap_style=2)
                             for s in segs if s["class"] in ("footway", "pedestrian", "path") and len(s["coords"]) > 1])
        near_b = self.B.buffer(4.0 if self.kind == "commercial" else 7.0).difference(self.B)
        self.add_surface("pavers" if self.kind == "commercial" else "concrete", unary_union([ped, near_b]), 0.14)
        self.add_surface("concrete", walks, 0.14)
        grass = unary_union([Polygon(l["exterior"], l["holes"]).buffer(0) for l in lu if l["class"] in ("grass",)])
        beds = unary_union([Polygon(l["exterior"], l["holes"]).buffer(0) for l in lu
                            if l["class"] in ("flowerbed", "garden", "park")])
        g_grass = self.add_surface("grass", grass, 0.10)
        g_beds = self.add_surface("mulch", beds, 0.12)
        # campus hardscape quads within 30 m of buildings (schools), rest is landscape
        if self.kind == "school":
            self.add_surface("concrete_dark", self.B.buffer(26.0), 0.13)
        rest = self.add_surface("landscape", self.clip, 0.08)
        self.landscape_trees(rest, g_grass, g_beds, walks)
        return {
            "surfaces": self.surfaces, "paint": self.paint, "trees": self.trees, "lamps": self.lamps,
            "parked": self.parked, "canopies": self.canopies, "objects": self.objects,
            "fronts": self.storefronts(aisles + service),
        }

    def sports(self, lu: list[dict[str, Any]]) -> None:
        track = [Polygon(l["exterior"], l["holes"]).buffer(0) for l in lu if l["class"] == "track"]
        for t in track:
            g = self.add_surface("track_red", t, 0.12)
            if g.is_empty:
                continue
            # lane lines: offsets of the inner edge
            # grandstands on both long sides, outside the running lanes
            tcx, tcy, tL, tW, tang = _mrr(t)
            for sgn, size in ((1, [60.0, 9.0]), (-1, [36.0, 6.0])):
                off = tW / 2 + size[1] / 2 + 2.0
                bx = tcx - sgn * off * math.sin(math.radians(tang))
                by = tcy + sgn * off * math.cos(math.radians(tang))
                if self.clip.contains(Point(bx, by)) and self.B.distance(Point(bx, by)) > size[1]:
                    self.objects.append({"type": "bleachers", "x": bx, "y": by, "angle": tang + (0 if sgn > 0 else 180), "size": size})
            inner = [Polygon(r) for r in t.interiors]
            if inner:
                ring = inner[0]
                for k in range(1, 9):
                    ln = ring.buffer(LANE_W * k, join_style=1).exterior
                    self.add_paint("stripe_white", ln.buffer(0.05), 0.135, within=g)
                self.add_paint("stripe_white", LineString(ring.exterior.coords).buffer(0.08), 0.135, within=g)
        for l in sorted(lu, key=lambda q: -Polygon(q["exterior"]).area):
            cls = l["class"]
            if cls not in ("pitch", "playground", "recreation_ground"):
                continue
            p = Polygon(l["exterior"], l["holes"]).buffer(0)
            a = p.area
            cx, cy, L, W, ang = _mrr(p)
            name = l["name"].lower() if isinstance(l.get("name"), str) else ""
            if cls == "playground":
                self.add_surface("rubber_play", p, 0.13)
                self.objects.append({"type": "play_structure", "x": cx, "y": cy, "angle": ang,
                                     "size": [min(L * 0.6, 14), min(W * 0.6, 9)]})
            elif "diamond" in name or "baseball" in name or "softball" in name:
                self.diamond(p, softball="softball" in name)
            elif cls == "recreation_ground" and a > 2000:
                surf = "asphalt_light" if self.kind == "school" and a < 12000 and "field" not in name else "turf_field"
                self.add_surface(surf, p, 0.11)
            elif a < 140:
                g = self.add_surface("asphalt_light", p, 0.11)
                self.add_paint("stripe_yellow" if a < 40 else "stripe_white", unary_union(_outline_lines(p, 0.1)), 0.125)
                del g
            elif 200 < a < 300 and L < 26:  # tennis
                g = self.add_surface("court_blue", p, 0.12)
                rc = _oriented_rect(cx, cy, 23.77, 10.97, ang)
                self.add_paint("stripe_white", unary_union(_outline_lines(rc, 0.08) + [
                    rect_line(*self._axis(cx, cy, 0, 10.97, ang + 90), 0.08),
                    rect_line(*self._axis(cx, cy, 6.4 * 2, 0, ang), 0.06)]), 0.135, within=g.buffer(0.5))
                self.objects.append({"type": "tennis_net", "x": cx, "y": cy, "angle": ang + 90, "size": [12.8, 1.07]})
            elif 300 <= a < 900 and L < 36:  # basketball
                g = self.add_surface("asphalt_light", p, 0.12)
                rc = _oriented_rect(cx, cy, min(L - 1.5, 26), min(W - 1.2, 15), ang)
                self.add_paint("stripe_white", unary_union(_outline_lines(rc, 0.08) + [
                    rect_line(*self._axis(cx, cy, 0, min(W - 1.2, 15), ang + 90), 0.08),
                    Point(cx, cy).buffer(1.83).exterior.buffer(0.04)]), 0.135, within=g.buffer(0.5))
                for s in (-1, 1):
                    hx = cx + s * (min(L - 1.5, 26) / 2 - 1.2) * math.cos(math.radians(ang))
                    hy = cy + s * (min(L - 1.5, 26) / 2 - 1.2) * math.sin(math.radians(ang))
                    self.objects.append({"type": "hoop", "x": hx, "y": hy, "angle": ang + (180 if s > 0 else 0)})
            elif a >= 2500:  # soccer / football field
                g = self.add_surface("turf_field", p, 0.11)
                Lf, Wf = min(L - 6, 105), min(W - 6, 68)
                rc = _oriented_rect(cx, cy, Lf, Wf, ang)
                self.add_paint("stripe_white", unary_union(_outline_lines(rc, 0.12) + [
                    rect_line(*self._axis(cx, cy, 0, Wf, ang + 90), 0.12),
                    Point(cx, cy).buffer(9.15).exterior.buffer(0.06)]), 0.125, within=g)
                for s in (-1, 1):
                    gx = cx + s * (Lf / 2) * math.cos(math.radians(ang))
                    gy = cy + s * (Lf / 2) * math.sin(math.radians(ang))
                    self.objects.append({"type": "goal", "x": gx, "y": gy, "angle": ang + 90})
            else:
                self.add_surface("track_red" if W < 6 else "turf_field", p, 0.11)

    @staticmethod
    def _axis(cx: float, cy: float, L: float, W: float, ang: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Segment of length max(L, W) through (cx, cy) along angle ang."""
        d = max(L, W) / 2
        c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        return (cx - c * d, cy - s * d), (cx + c * d, cy + s * d)

    def diamond(self, p: Polygon, softball: bool) -> None:
        hull = p.convex_hull
        pts = np.array(hull.exterior.coords)[:-1]
        # home plate = hull vertex with the sharpest angle (the apex of the fan)
        best, home = 1e9, pts[0]
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            v1, v2 = a - b, c - b
            if np.linalg.norm(v1) < 3 or np.linalg.norm(v2) < 3:
                continue
            ang = math.acos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
            if ang < best:
                best, home = ang, b
        cen = np.array(p.centroid.coords[0])
        d = cen - home
        d /= np.linalg.norm(d) + 1e-9
        base = 18.3 if softball else 27.4  # 60 ft / 90 ft base paths
        mound = home + d * (base * 0.67)
        ang = math.degrees(math.atan2(d[1], d[0]))
        if not softball:  # grass infield inside the base paths
            infield = _oriented_rect(*(home + d * base / math.sqrt(2)), base, base, ang + 45)
            self.add_surface("turf_field", infield.buffer(-1.4), 0.125)
        dirt = unary_union([Point(*mound).buffer(base * 1.05), Point(*home).buffer(4.0)]).intersection(p)
        self.add_surface("dirt_infield", dirt, 0.12)
        self.add_surface("turf_field", p, 0.11)
        for s in (-1, 1):  # foul lines
            a2 = math.radians(ang + s * 45)
            self.add_paint("stripe_white", rect_line(tuple(home), (home[0] + math.cos(a2) * 120, home[1] + math.sin(a2) * 120), 0.1),
                           0.13, within=p)

    def parking(self, aisles: list[LineString], service: list[LineString]) -> None:
        half = AISLE_W / 2 + STALL_L
        lots = unary_union([a.buffer(half, cap_style=2, join_style=2) for a in aisles])
        lot_g = self.add_surface("asphalt", lots, 0.10)
        drives = unary_union([s.buffer(SERVICE_HALF, cap_style=1) for s in service])
        self.add_surface("asphalt", drives, 0.10)
        if lot_g.is_empty:
            return
        lot_core = lot_g.buffer(-0.4)
        rng = self.rng
        occupancy = OCCUPANCY["school" if self.kind == "school" else "commercial"]
        canopy_rows = []
        for a in aisles:
            L = a.length
            if L < 12:
                continue
            n = int(L // STALL_W)
            for side in (-1, 1):
                row = []
                for k in range(n + 1):
                    t = (k * STALL_W + (L - n * STALL_W) / 2) / L
                    p0 = np.array(a.interpolate(t, normalized=True).coords[0])
                    p1 = np.array(a.interpolate(min(1.0, t + 0.01), normalized=True).coords[0])
                    if t >= 0.99:
                        p1 = p0 + (p0 - np.array(a.interpolate(t - 0.01, normalized=True).coords[0]))
                    tan = (p1 - p0)
                    tan /= np.linalg.norm(tan) + 1e-9
                    nrm = np.array([-tan[1], tan[0]]) * side
                    s0 = p0 + nrm * (AISLE_W / 2)
                    s1 = p0 + nrm * half
                    c0 = p0 + nrm * (AISLE_W / 2 + 0.3)
                    c1 = p0 + nrm * (half - 0.6)
                    if not (lot_core.contains(Point(*c0)) and lot_core.contains(Point(*c1))):
                        row.append(None)
                        continue
                    self.add_paint("stripe_white", rect_line(tuple(s0), tuple(s1), 0.1), 0.115, within=lot_g)
                    row.append((p0, tan, nrm))
                # stall centers between consecutive stripes
                for k in range(len(row) - 1):
                    if row[k] is None or row[k + 1] is None:
                        continue
                    p0, tan, nrm = row[k]
                    c = p0 + tan * STALL_W / 2 + nrm * (AISLE_W / 2 + STALL_L / 2)
                    if rng.random() < occupancy:
                        heading = math.degrees(math.atan2(-nrm[1], -nrm[0]))  # nose in toward the stall end
                        if rng.random() < 0.25:
                            heading += 180  # some back in
                        self.parked.append({"x": round(float(c[0]), 2), "y": round(float(c[1]), 2), "heading_deg": round(heading, 1)})
                # stall row runs for canopies / trees / lights
                runs, cur = [], []
                for e in row:
                    if e is None:
                        if len(cur) > 4:
                            runs.append(cur)
                        cur = []
                    else:
                        cur.append(e)
                if len(cur) > 4:
                    runs.append(cur)
                for run in runs:
                    canopy_rows.append(run)
                    # tree islands at run ends and every ~6 stalls (schools: only at ends where no canopy)
                    end_pts = [run[0], run[-1]]
                    for (p0, tan, nrm), sgn in zip(end_pts, (-1, 1), strict=True):
                        q = p0 + tan * sgn * 1.6 + nrm * (AISLE_W / 2 + STALL_L * 0.6)
                        self._tree(q[0], q[1], "commercial_lot" if self.kind == "commercial" else "lot")
                    if self.kind == "commercial":
                        for j in range(6, len(run) - 3, 7):
                            p0, tan, nrm = run[j]
                            q = p0 + nrm * (half + 0.9)
                            self._tree(q[0], q[1], "commercial_lot")
                    for j in range(0, len(run), 11):
                        p0, tan, nrm = run[j]
                        if side > 0:
                            q = p0 + nrm * (half + 0.4)
                            self.lamps.append({"x": round(float(q[0]), 2), "y": round(float(q[1]), 2),
                                               "rot_deg": round(math.degrees(math.atan2(-nrm[1], -nrm[0])), 1)})
        if self.kind == "school":  # solar carport canopies over the longest stall rows (PV shade structures)
            canopy_rows.sort(key=len, reverse=True)
            for run in canopy_rows[: 8]:
                p0, tan, nrm = run[0]
                p1 = run[-1][0]
                mid = (p0 + p1) / 2 + nrm * (AISLE_W / 2 + STALL_L / 2)
                length = float(np.linalg.norm(p1 - p0))
                ang = math.degrees(math.atan2(tan[1], tan[0]))
                tilt_dir = math.degrees(math.atan2(nrm[1], nrm[0]))
                self.canopies.append({"x": round(float(mid[0]), 2), "y": round(float(mid[1]), 2), "angle_deg": round(ang, 2),
                                      "length": round(length, 2), "width": 7.2, "height": 4.3, "tilt_toward_deg": round(tilt_dir, 1)})

    def _tree(self, x: float, y: float, ctx: str) -> None:
        rng = self.rng
        if not self.clip.contains(Point(x, y)) or self.B.distance(Point(x, y)) < 2.5:
            return
        if any(math.hypot(t["x"] - x, t["y"] - y) < 3.5 for t in self.trees[-200:]):
            return
        if ctx == "commercial_lot":
            sp = rng.choice(["tree_palm_queen", "tree_street", "tree_palm_queen", "tree_jacaranda", "tree_palm_fan"],
                            p=[0.3, 0.3, 0.1, 0.15, 0.15])
        elif ctx == "lot":
            sp = rng.choice(["tree_street", "tree_jacaranda", "tree_oak"], p=[0.6, 0.25, 0.15])
        elif ctx == "slope":
            sp = rng.choice(["tree_oak", "tree_eucalyptus", "tree_oak"], p=[0.5, 0.3, 0.2])
        elif ctx == "bed":
            sp = rng.choice(["shrub", "grass_ornamental", "shrub"], p=[0.5, 0.3, 0.2])
        elif ctx == "entry":
            sp = rng.choice(["tree_palm_queen", "tree_palm_fan", "tree_jacaranda"], p=[0.45, 0.3, 0.25])
        else:
            sp = rng.choice(["tree_street", "tree_jacaranda", "tree_oak", "tree_palm_queen"], p=[0.45, 0.2, 0.25, 0.1])
        self.trees.append({"species": str(sp), "x": round(float(x), 2), "y": round(float(y), 2),
                           "rot_deg": round(float(rng.uniform(0, 360)), 1), "scale": round(float(rng.uniform(0.85, 1.15)), 3)})

    def landscape_trees(self, rest, grass, beds, walks) -> None:  # noqa: ANN001
        rng = self.rng
        # along walks (every ~11 m, alternating sides)
        for ln in lines(walks.boundary) if not walks.is_empty else []:
            L = ln.length
            for k in range(int(L // 9)):
                p = ln.interpolate(k * 9 + 4)
                if rest.distance(p) < 1.5 or grass.distance(p) < 1.0:
                    self._tree(p.x, p.y, "walk")
        # around buildings in landscape strips
        for b in self.bld:
            ring = b.buffer(5.5).exterior
            for k in range(int(ring.length // 14)):
                p = ring.interpolate(k * 14 + rng.uniform(0, 4))
                if rest.contains(p) or grass.contains(p):
                    self._tree(p.x, p.y, "entry" if rng.random() < 0.15 else "walk")
        # scattered in grass and landscape (slopes at the edges get oaks / eucalyptus)
        for area, dens, ctx in ((grass, 1 / 200.0, "walk"), (rest, 1 / 260.0, "slope"), (beds, 1 / 14.0, "bed")):
            if area.is_empty:
                continue
            n = int(area.area * dens)
            minx, miny, maxx, maxy = area.bounds
            tries = 0
            placed = 0
            while placed < n and tries < n * 12:
                tries += 1
                x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
                if not area.contains(Point(x, y)):
                    continue
                c = ctx
                if ctx == "slope" and math.hypot(x, y) < self.R * 0.55:
                    c = "walk"
                before = len(self.trees)
                self._tree(x, y, c)
                placed += len(self.trees) > before

    def storefronts(self, drives: list[LineString]) -> list[list[int]]:
        """Per modeled building: wall edge indices that face a parking aisle / drive (shop fronts, entries)."""
        dr = unary_union(drives) if drives else None
        out = []
        for b in self.site["buildings"]:
            ext = b["exterior"]
            p = Polygon(ext)
            ccw = p.exterior.is_ccw
            fr = []
            n = len(ext)
            for i in range(n):
                a, c = np.array(ext[i]), np.array(ext[(i + 1) % n])
                e = c - a
                L = float(np.linalg.norm(e))
                if L < 6:
                    continue
                nrm = np.array([e[1], -e[0]]) / L
                if not ccw:
                    nrm = -nrm
                mid = (a + c) / 2
                probe = Point(*(mid + nrm * 18))
                if dr is not None and dr.distance(probe) < 14 and not self.B.contains(probe):
                    fr.append(i)
            out.append(fr)
        return out


def build_layout(site: dict[str, Any]) -> dict[str, Any]:
    return Layout(site).build()


__all__ = ["build_layout", "translate"]
