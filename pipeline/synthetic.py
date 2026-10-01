"""Synthetic stand-in world (offline dev / CI only). NOT real geography.

Generates, with fixed seeds, the same kinds of in-memory inputs the real
fetchers produce (DEM grid, OSM-style drive graph, OSM-style building
footprints, block group distributions, a work destination model) and hands
them to the same build functions as the real mode. Every output of this mode
is labeled `synthetic: true`.

Layout is a rough sketch of 4S Ranch / Del Sur: I 15 on the east (north-south),
SR 56 on the south (east-west), arterials named after region.yaml labels at
GUESSED positions, curvy residential pods, a few commercial/apartment
clusters and school campuses at the schools.yaml coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box
from shapely.ops import nearest_points, polygonize, substring, unary_union

from pipeline.build_terrain import AlbedoFn, Terrain, grid_shape
from pipeline.common import Extent, log
from pipeline.config import assumption
from pipeline.geo import latlon_to_scene, scene_origin

SEED = 20260930  # fixed (rule 14.5); population uses assumptions population.seed

# --- layout parameters (procedural design knobs, not real-world facts) ---
POD_CELL_M = 750.0
POD_FACE_INSET_M = 42.0
POD_CELL_INSET_M = 22.0
POD_MIN_AREA_M2 = 70_000.0
STREET_SPACING_M = 80.0
CROSS_SPACING_M = 150.0
WARP_AMPLITUDE_M = 16.0
WARP_WAVELENGTH_M = 520.0
LOT_SPACING_M = 18.0
OVERSHOOT_M = 12.0
DANGLE_MAX_M = 30.0
CARRIAGEWAY_HALF_SEP_M = 14.0
RAMP_SIDE_OFFSET_M = 110.0
RAMP_ALONG_M = 380.0

STREET_NAMES = [
    "Calle", "Via", "Paseo", "Camino", "Avenida", "Corte", "Vista", "Rancho", "Sierra", "Mesa",
]
STREET_ROOTS = [
    "Albero", "Bella", "Cielo", "Del Mar", "Encina", "Fuerte", "Gaviota", "Halcon", "Isla", "Jardin",
    "Lomas", "Manzanita", "Nube", "Olivo", "Palomar", "Quinta", "Roble", "Salvia", "Tierra", "Valle",
    "Arroyo", "Brisa", "Colina", "Dorado", "Estrella", "Flora", "Granada", "Higuera", "Loma", "Monte",
]


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------


def chaikin(pts: np.ndarray, iters: int = 2) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    for _ in range(iters):
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2] = q
        mid[1::2] = r
        p = np.vstack([p[0], mid, p[-1]])
    return p


def densify_line(pts: np.ndarray, step: float) -> np.ndarray:
    ls = LineString(pts)
    n = max(2, int(math.ceil(ls.length / step)) + 1)
    return np.array([ls.interpolate(d).coords[0] for d in np.linspace(0, ls.length, n)])


def offset_polyline(pts: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Offset each vertex to the RIGHT of travel by offsets[i] (scene x,z with z south)."""
    p = np.asarray(pts, dtype=np.float64)
    t = np.zeros_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    # Right of travel in a map where x=east and z=south: heading north is (0,-1), right is east (1,0).
    right = np.column_stack([-t[:, 1], t[:, 0]])
    return p + right * np.asarray(offsets)[:, None]


def extend_line(pts: np.ndarray, start: bool, end: bool, d: float = OVERSHOOT_M) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).copy()
    if start:
        v = p[0] - p[1]
        p = np.vstack([p[0] + v / max(np.linalg.norm(v), 1e-9) * d, p])
    if end:
        v = p[-1] - p[-2]
        p = np.vstack([p, p[-1] + v / max(np.linalg.norm(v), 1e-9) * d])
    return p


def iter_lines(g: Any) -> Iterable[LineString]:
    if g is None or g.is_empty:
        return
    if isinstance(g, LineString):
        yield g
    elif isinstance(g, MultiLineString) or hasattr(g, "geoms"):
        for part in g.geoms:
            yield from iter_lines(part)


def iter_polys(g: Any) -> Iterable[Polygon]:
    if g is None or g.is_empty:
        return
    if isinstance(g, Polygon):
        yield g
    elif isinstance(g, MultiPolygon) or hasattr(g, "geoms"):
        for part in g.geoms:
            yield from iter_polys(part)


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------

BLACK_MOUNTAIN = (650.0, 2440.0)  # scene x,z of the peak (guessed from 32.983, -117.118)
CANYONS = [
    [(-4800, -1500), (-3300, -2100), (-2500, -3000), (-2300, -4600)],
    [(-4800, 900), (-3200, 600), (-2200, 1000), (-1600, 1050)],
    [(-4800, 4300), (-1500, 4150), (1500, 4250), (4800, 4050)],
    [(400, -4600), (700, -3600), (300, -2900)],
]


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    n = ndimage.gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect")
    return (n - n.mean()) / (n.std() + 1e-9)


def _dist_to_polyline(gx: np.ndarray, gz: np.ndarray, pts: list[tuple[float, float]]) -> np.ndarray:
    d = np.full(gx.shape, np.inf)
    p = np.asarray(pts, dtype=np.float64)
    for a, b in zip(p[:-1], p[1:], strict=True):
        ab = b - a
        t = np.clip(((gx - a[0]) * ab[0] + (gz - a[1]) * ab[1]) / (ab @ ab), 0, 1)
        dx = gx - (a[0] + t * ab[0])
        dz = gz - (a[1] + t * ab[1])
        d = np.minimum(d, np.hypot(dx, dz))
    return d


def synthetic_dem(extent: Extent, spacing: float = 10.0, seed: int = SEED) -> Terrain:
    rng = np.random.default_rng(seed)
    rows, cols = grid_shape(extent, spacing)
    xs = extent.min_x + np.arange(cols) * spacing
    zs = extent.min_z + np.arange(rows) * spacing
    gx, gz = np.meshgrid(xs, zs)
    e = 195.0 + 0.004 * gx  # gentle rise to the east
    e += 28.0 * _smooth_noise(rng, (rows, cols), 32)
    e += 9.0 * _smooth_noise(rng, (rows, cols), 7)
    e += 1.0 * _smooth_noise(rng, (rows, cols), 1.5)
    r2 = (gx - BLACK_MOUNTAIN[0]) ** 2 + (gz - BLACK_MOUNTAIN[1]) ** 2
    e += 285.0 * np.exp(-r2 / (2 * 650.0**2))
    e += 40.0 * np.exp(-((gx + 4000) ** 2 + (gz - 2600) ** 2) / (2 * 900.0**2))  # western hills
    e -= 95.0 / (1.0 + np.exp((gz + 3750.0) / 220.0))  # San Dieguito-like river valley to the north
    for c in CANYONS:
        d = _dist_to_polyline(gx, gz, c)
        e -= 48.0 * np.exp(-(d**2) / (2 * 95.0**2))
    return Terrain(e.astype(np.float32), extent, spacing)


def grade_pods(terrain: Terrain, polys: list[Polygon], sigma_px: float = 7.0) -> None:
    """Smooth ("grade") the DEM under developed areas in place, like real subdivision grading."""
    ext = terrain.extent
    rows, cols = terrain.elev.shape
    m = Image.new("L", (cols, rows), 0)
    dr = ImageDraw.Draw(m)
    for poly in polys:
        for p in iter_polys(poly):
            pts = [((x - ext.min_x) / terrain.spacing, (z - ext.min_z) / terrain.spacing) for x, z in np.asarray(p.exterior.coords)]
            if len(pts) >= 3:
                dr.polygon(pts, fill=255)
    mask = ndimage.gaussian_filter(np.asarray(m, dtype=np.float32) / 255.0, 3.0)
    smooth = ndimage.gaussian_filter(terrain.elev, sigma_px)
    terrain.elev[:] = (terrain.elev * (1 - mask) + smooth * mask).astype(np.float32)


def synthetic_albedo(terrain: Terrain, developed: list[Polygon], paved: list[Polygon], fields: list[Polygon], seed: int = SEED, res_m: float = 3.0) -> AlbedoFn:
    """Procedural albedo: chaparral/grass by slope and noise, developed pods, lots and fields."""
    ext = terrain.extent
    w = int(round(ext.width / res_m))
    h = int(round(ext.depth / res_m))
    rng = np.random.default_rng(seed + 1)
    xs = ext.min_x + (np.arange(w) + 0.5) * res_m
    zs = ext.min_z + (np.arange(h) + 0.5) * res_m
    gx, gz = np.meshgrid(xs, zs)
    elev = terrain.sample(gx, gz)
    gy, gxx = np.gradient(elev, res_m)
    slope = np.hypot(gy, gxx)
    n1 = _smooth_noise(rng, (h, w), 25)
    n2 = _smooth_noise(rng, (h, w), 4)
    n3 = rng.standard_normal((h, w))
    grass = np.array([176, 160, 112], dtype=np.float32)
    chap = np.array([96, 102, 70], dtype=np.float32)
    rock = np.array([150, 138, 120], dtype=np.float32)
    t_chap = np.clip(0.5 + 0.35 * n1 + 1.8 * slope, 0, 1)[..., None]
    img = grass * (1 - t_chap) + chap * t_chap
    t_rock = np.clip((slope - 0.35) * 2.0, 0, 1)[..., None]
    img = img * (1 - t_rock) + rock * t_rock
    valley = (1.0 / (1.0 + np.exp((gz + 3900.0) / 120.0)))[..., None]
    img = img * (1 - 0.6 * valley) + np.array([70, 100, 60], dtype=np.float32) * 0.6 * valley

    def mask_of(polys: list[Polygon]) -> np.ndarray:
        m = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(m)
        for poly in polys:
            for p in iter_polys(poly):
                pts = [((x - ext.min_x) / res_m, (z - ext.min_z) / res_m) for x, z in np.asarray(p.exterior.coords)]
                if len(pts) >= 3:
                    dr.polygon(pts, fill=255)
        return np.asarray(m, dtype=np.float32)[..., None] / 255.0

    dev = mask_of(developed)
    yard = np.array([118, 132, 86], dtype=np.float32)
    roofs = np.array([190, 170, 150], dtype=np.float32)
    t = np.clip(0.5 + 0.5 * n2, 0, 1)[..., None]
    dev_col = yard * t + roofs * (1 - t)
    img = img * (1 - dev) + dev_col * dev
    pav = mask_of(paved)
    img = img * (1 - pav) + np.array([128, 126, 122], dtype=np.float32) * pav
    fld = mask_of(fields)
    img = img * (1 - fld) + np.array([92, 140, 70], dtype=np.float32) * fld
    img += (6.0 * n2 + 5.0 * n3)[..., None]
    full = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    def fn(b: Extent, size: int) -> np.ndarray:
        x0 = (b.min_x - ext.min_x) / res_m
        x1 = (b.max_x - ext.min_x) / res_m
        z0 = (b.min_z - ext.min_z) / res_m
        z1 = (b.max_z - ext.min_z) / res_m
        # clamp: the image is round(extent / res_m) px, so the far edge can overshoot by < 1 px
        x0, x1 = max(0.0, x0), min(float(full.width), x1)
        z0, z1 = max(0.0, z0), min(float(full.height), z1)
        crop = full.resize((size, size), Image.BILINEAR, box=(x0, z0, x1, z1))
        return np.asarray(crop)

    return fn


# ---------------------------------------------------------------------------
# Road layout
# ---------------------------------------------------------------------------


@dataclass
class RoadLine:
    coords: np.ndarray  # (N,2) scene x,z
    attrs: dict[str, Any]
    layer: str = "surf"  # surf lines are noded with each other; fwy lines only at forced splits
    splits: list[tuple[float, tuple[float, float]]] = field(default_factory=list)
    protect_ends: bool = False  # never remove as a dangle (exit stubs, driveways)

    @property
    def ls(self) -> LineString:
        return LineString(self.coords)

    def force_split_near(self, pt: tuple[float, float]) -> tuple[float, float]:
        ls = self.ls
        d = ls.project(Point(pt))
        q = ls.interpolate(d).coords[0]
        self.splits.append((d, (float(q[0]), float(q[1]))))
        return float(q[0]), float(q[1])


def _attrs(highway: str, name: str = "", ref: str = "", lanes: str | None = None, maxspeed: str | None = None, oneway: bool = False) -> dict[str, Any]:
    a: dict[str, Any] = {"highway": highway, "name": name, "ref": ref, "oneway": oneway, "osmid": ""}
    if lanes is not None:
        a["lanes"] = lanes
    if maxspeed is not None:
        a["maxspeed"] = maxspeed
    return a


@dataclass
class Freeway:
    center: np.ndarray
    pos: RoadLine  # carriageway in centerline direction
    neg: RoadLine  # carriageway in reverse direction


def make_freeway(ctrl: list[tuple[float, float]], name: str, ref: str, lanes: str, pinch_start: bool, pinch_end: bool) -> Freeway:
    c = densify_line(chaikin(np.asarray(ctrl, dtype=np.float64), 3), 25.0)
    ls = LineString(c)
    s = np.array([ls.project(Point(p)) for p in c])
    L = ls.length
    taper = np.ones(len(c))
    if pinch_start:
        taper = np.minimum(taper, np.clip(s / 250.0, 0, 1))
    if pinch_end:
        taper = np.minimum(taper, np.clip((L - s) / 250.0, 0, 1))
    off = CARRIAGEWAY_HALF_SEP_M * taper
    pos = offset_polyline(c, off)
    neg = offset_polyline(c[::-1], off[::-1])
    a = _attrs("motorway", name, ref, lanes, "65 mph", oneway=True)
    return Freeway(c, RoadLine(pos, dict(a), "fwy"), RoadLine(neg, dict(a), "fwy"))


def make_ramp(a: tuple[float, float], b: tuple[float, float], bulge: np.ndarray, ref: str) -> RoadLine:
    pa_, pb = np.asarray(a), np.asarray(b)
    mid = (pa_ + pb) / 2 + bulge
    pts = chaikin(np.vstack([pa_, mid, pb]), 2)
    return RoadLine(pts, _attrs("motorway_link", "", ref, None, None, oneway=True), "fwy")


def diamond(fw: Freeway, art: RoadLine, ref: str) -> list[RoadLine]:
    """Diamond interchange where arterial `art` crosses freeway `fw` (grade separated)."""
    P = LineString(fw.center).intersection(art.ls)
    if P.is_empty:
        raise ValueError(f"arterial {art.attrs['name']} does not cross freeway {ref}")
    P = P if isinstance(P, Point) else list(P.geoms)[0]
    ramps = []
    for cw in (fw.pos, fw.neg):
        cls = cw.ls
        side = LineString(offset_polyline(cw.coords, np.full(len(cw.coords), RAMP_SIDE_OFFSET_M)))
        A = side.intersection(art.ls)
        if A.is_empty:
            raise ValueError(f"ramp side point missing for {art.attrs['name']} x {ref}")
        A = A if isinstance(A, Point) else list(A.geoms)[0]
        a_pt = art.force_split_near((A.x, A.y))
        sP = cls.project(P)
        up = cw.force_split_near(cls.interpolate(max(sP - RAMP_ALONG_M, 5.0)).coords[0])
        down = cw.force_split_near(cls.interpolate(min(sP + RAMP_ALONG_M, cls.length - 5.0)).coords[0])
        outward = np.asarray(a_pt) - np.asarray(P.coords[0])
        outward = outward / max(np.linalg.norm(outward), 1e-9) * 25.0
        ramps.append(make_ramp(up, a_pt, outward, ref))
        ramps.append(make_ramp(a_pt, down, outward, ref))
    return ramps


def freeway_junction(main: Freeway, j: tuple[float, float], ref: str) -> list[RoadLine]:
    """Connect a freeway terminus node j to both carriageways of `main` (directional ramps)."""
    out = []
    for cw in (main.pos, main.neg):
        cls = cw.ls
        s = cls.project(Point(j))
        down = cw.force_split_near(cls.interpolate(min(s + 500.0, cls.length - 5)).coords[0])
        up = cw.force_split_near(cls.interpolate(max(s - 500.0, 5)).coords[0])
        b1 = (np.asarray(down) - np.asarray(j)) * 0.0 + np.array([0.0, 0.0])
        out.append(make_ramp(j, down, b1, ref))
        out.append(make_ramp(up, j, b1, ref))
    return out


def arterial(ctrl: list[tuple[float, float]], name: str, highway: str, lanes: str, maxspeed: str, ref: str = "") -> RoadLine:
    c = densify_line(chaikin(np.asarray(ctrl, dtype=np.float64), 2), 30.0)
    return RoadLine(c, _attrs(highway, name, ref, lanes, maxspeed))


def _x_on(line: np.ndarray, z: float) -> float:
    ls = LineString(line)
    hit = ls.intersection(LineString([(-1e5, z), (1e5, z)]))
    pts = [hit] if isinstance(hit, Point) else list(getattr(hit, "geoms", []))
    return float(pts[0].x) if pts else float(line[np.argmin(np.abs(line[:, 1] - z)), 0])


@dataclass
class Layout:
    lines: list[RoadLine]
    pods: list[Polygon]
    campuses: dict[str, Polygon]
    parcels: list[tuple[str, Polygon, float]]  # (kind, polygon, rotation)
    residential_lines: list[LineString]
    entrance_nodes: list[tuple[float, float]]


def schools_scene(schools_cfg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in schools_cfg:
        x, z = latlon_to_scene(float(s["lat"]), float(s["lon"]))
        ents = [(e["id"], *latlon_to_scene(float(e["lat"]), float(e["lon"]))) for e in s.get("entrances", [])]
        g0, g1 = s["grades"]
        r = 170.0 if g0 >= 9 else (135.0 if g0 >= 6 else 105.0)
        out.append({"id": s["id"], "name": s["name"], "x": x, "z": z, "r": r, "grades": (g0, g1), "entrances": ents})
    return out


def build_layout(ext: Extent, schools_cfg: list[dict[str, Any]], target_houses: int, seed: int = SEED) -> Layout:
    rng = np.random.default_rng(seed)
    zmin, zmax, xmin = ext.min_z + 5, ext.max_z - 5, ext.min_x + 5

    # Freeways: I 15 (east, north-south) and SR 56 (south, east-west). Positions GUESSED.
    i15 = make_freeway([(2650, zmax), (2850, 2500), (3150, 600), (3500, -1500), (3800, -3300), (3950, zmin)], "Escondido Freeway", "I 15", "4", True, True)
    x15 = _x_on(i15.center, 3350.0)
    J = (x15 - 320.0, 3350.0)
    sr56 = make_freeway([(xmin, 4150), (-2500, 3950), (-500, 3750), (1300, 3640), J], "Ted Williams Parkway", "CA 56", "2", True, True)
    for cw in (sr56.pos, sr56.neg):  # make the terminus exactly J on both carriageways
        if np.hypot(*(cw.coords[-1] - np.asarray(J))) < np.hypot(*(cw.coords[0] - np.asarray(J))):
            cw.coords[-1] = J
        else:
            cw.coords[0] = J
    for fw in (i15, sr56):  # exact shared terminus nodes
        fw.pos.coords[0] = fw.neg.coords[-1] = fw.center[0]
        fw.pos.coords[-1] = fw.neg.coords[0] = fw.center[-1]
        fw.pos.protect_ends = fw.neg.protect_ends = True

    P, S, T = "primary", "secondary", "tertiary"
    arts = [
        arterial([(4500, -1440), (3500, -1430), (2700, -1480), (2000, -1600), (1300, -1800), (500, -2050), (-400, -2400), (-1360, -2760)], "Camino Del Norte", P, "6", "45 mph"),
        arterial([(-900, zmax - 15), (-1000, 3750), (-1150, 2200), (-1300, 600), (-1250, -1000), (-1350, -2750), (-1700, -3600), (-2100, -4300)], "Camino Del Sur", P, "6", "45 mph"),
        arterial([(1300, -1800), (1150, -900), (800, -100), (200, 400), (-1300, 600)], "4S Ranch Parkway", S, "4", "40 mph"),
        arterial([(2700, -1480), (2550, -500), (2150, 300), (1500, 700), (800, -100)], "Dove Canyon Road", S, "4", "40 mph"),
        arterial([(2600, 1650), (2850, 600), (3100, -400), (3200, -1435), (3300, -2500), (3450, -3550)], "Bernardo Center Drive", S, "4", "40 mph"),
        arterial([(1500, 700), (2100, 1300), (2600, 1650), (3300, 1750), (4400, 1800)], "Carmel Mountain Road", S, "4", "40 mph"),
        arterial([(3450, -3550), (4500, -3450)], "Rancho Bernardo Road", S, "4", "40 mph"),
        arterial([(-4650, 2550), (-3600, 2250), (-2600, 1800), (-1900, 1500), (-1300, 1300)], "Carmel Valley Road", S, "4", "40 mph"),
        arterial([(-2650, zmax - 15), (-2600, 3950), (-2650, 3000), (-2600, 1800), (-2450, 700), (-2250, -300)], "Black Mountain Road", S, "4", "40 mph"),
        arterial([(-3700, -600), (-2250, -300), (-1250, -600), (-300, -900), (800, -100)], "Paseo Del Sur", T, "2", "35 mph"),
        arterial([(-4650, -3900), (-3500, -4000), (-2100, -4300)], "San Dieguito Road", S, "4", "40 mph"),
    ]
    by_name = {a.attrs["name"]: a for a in arts}
    ramps: list[RoadLine] = []
    ramps += diamond(i15, by_name["Camino Del Norte"], "I 15")
    ramps += diamond(i15, by_name["Carmel Mountain Road"], "I 15")
    ramps += diamond(i15, by_name["Rancho Bernardo Road"], "I 15")
    ramps += diamond(sr56, by_name["Black Mountain Road"], "CA 56")
    ramps += diamond(sr56, by_name["Camino Del Sur"], "CA 56")
    ramps += freeway_junction(i15, J, "I 15")
    for a in arts:  # overshoot so arterial ends cross whatever they end on
        a.coords = extend_line(a.coords, True, True)

    # Campuses and commercial / apartment parcels (exclusions for pods and houses).
    sch = schools_scene(schools_cfg)
    campuses = {s["id"]: Point(s["x"], s["z"]).buffer(s["r"], 24) for s in sch}
    parcel_specs = [
        ("commercial", (1300, -650), 330, 230, 0.15),  # "4S Commons"-like center
        ("commercial", (-1000, -1700), 260, 200, 0.0),  # "Del Sur town center"
        ("commercial", (3000, -2000), 260, 380, 0.1),  # offices near I 15
        ("commercial", (2700, 250), 220, 200, 0.2),
        ("apartments", (2050, -750), 300, 260, 0.3),
        ("apartments", (-850, -1200), 260, 260, 0.0),
        ("apartments", (-2700, -1050), 300, 240, -0.2),
        ("apartments", (2000, 950), 260, 220, 0.1),
        ("apartments", (-150, -50), 260, 200, 0.4),
    ]
    parcels = []
    for kind, (cx, cz), w, d, rot in parcel_specs:
        poly = box(-w / 2, -d / 2, w / 2, d / 2)
        c, s_ = math.cos(rot), math.sin(rot)
        poly = Polygon([(cx + x * c - z * s_, cz + x * s_ + z * c) for x, z in poly.exterior.coords])
        parcels.append((kind, poly, rot))
    exclusions = unary_union([p.buffer(25) for p in campuses.values()] + [p[1].buffer(20) for p in parcels])

    # Superblock faces from arterials + freeway centerlines + extent edge.
    frame = box(ext.min_x, ext.min_z, ext.max_x, ext.max_z).exterior
    face_lines = [a.ls for a in arts] + [LineString(i15.center), LineString(sr56.center), frame]
    faces = list(polygonize(unary_union(face_lines)))
    open_space = unary_union(
        [
            Point(*BLACK_MOUNTAIN).buffer(1250),
            box(ext.min_x, ext.min_z, ext.max_x, -3750),  # river valley
            box(ext.min_x, 3900, ext.max_x, ext.max_z),  # SR 56 canyon
            box(-4800, 1800, -3200, 3700),  # western hills
        ]
        + [LineString(c).buffer(140) for c in CANYONS]
    )
    fw_buffer = unary_union([LineString(i15.center).buffer(260), LineString(sr56.center).buffer(200)])
    zones = unary_union(
        [
            Polygon([(-300, -2900), (3100, -2500), (3100, 1100), (-300, 1100)]),  # 4S Ranch
            Polygon([(-3900, -3700), (-300, -3300), (-300, -150), (-3900, -150)]),  # Del Sur
            Polygon([(-4300, 1000), (-1500, 1000), (-1500, 3850), (-4300, 3850)]),  # south-west
            Polygon([(1800, 1000), (2800, 1000), (2700, 3500), (1800, 3500)]),  # Carmel Mountain
            Polygon([(3600, -3200), (4700, -3200), (4700, 1400), (3500, 1400)]),  # east of I 15
        ]
    )
    # dangling arterial ends do not split polygonized faces, so cut pods by arterial buffers too
    art_buffer = unary_union([a.ls.buffer(POD_FACE_INSET_M) for a in arts])
    cand: list[Polygon] = []
    xs = np.arange(ext.min_x, ext.max_x, POD_CELL_M)
    zs = np.arange(ext.min_z, ext.max_z, POD_CELL_M)
    for f in faces:
        fin = f.buffer(-POD_FACE_INSET_M)
        if fin.is_empty:
            continue
        for x0 in xs:
            for z0 in zs:
                cell = box(x0, z0, x0 + POD_CELL_M, z0 + POD_CELL_M).buffer(-POD_CELL_INSET_M, join_style=2)
                pod = fin.intersection(cell).difference(exclusions).difference(open_space).difference(fw_buffer)
                pod = pod.intersection(zones).difference(art_buffer)
                for part in iter_polys(pod):
                    part = part.buffer(-8).buffer(8).simplify(2.0)
                    for pp in iter_polys(part):
                        if pp.area >= POD_MIN_AREA_M2:
                            cand.append(pp)
    # Kids live near schools: develop pods closest to schools first (plus jitter).
    sch_pts = np.array([[s["x"], s["z"]] for s in sch])

    def prio(p: Polygon) -> float:
        c = p.centroid
        return float(np.min(np.hypot(sch_pts[:, 0] - c.x, sch_pts[:, 1] - c.y))) + float(rng.uniform(0, 900))

    cand.sort(key=prio)

    surf: list[RoadLine] = list(arts)
    residential: list[LineString] = []
    pods: list[Polygon] = []
    est_houses = 0.0
    existing_tree_geoms: list[LineString] = [a.ls for a in arts]
    for pod in cand:
        if est_houses >= target_houses * 1.08:
            break
        new_lines = _pod_streets(pod, rng)
        if not new_lines:
            continue
        tree = STRtree(existing_tree_geoms)
        conns = _pod_connectors(pod, existing_tree_geoms, tree, rng)
        if not conns:
            continue
        name_pool = [f"{rng.choice(STREET_NAMES)} {rng.choice(STREET_ROOTS)}" for _ in range(len(new_lines) + len(conns))]
        for k, ls in enumerate(new_lines + conns):
            surf.append(RoadLine(np.asarray(ls.coords)[:, :2], _attrs("residential", name_pool[k], "", None, "25 mph")))
            residential.append(ls)
            existing_tree_geoms.append(ls)
            est_houses += ls.length * 2 / LOT_SPACING_M * 0.44
        pods.append(pod)
    log(f"synthetic: {len(pods)} residential pods (of {len(cand)} candidates), est. {int(est_houses):,} house lots")

    # Parcel access: a service aisle from each parcel center to the nearest street.
    tree = STRtree(existing_tree_geoms)
    entrance_nodes = []
    for kind, poly, _rot in parcels:
        c = poly.centroid
        q = nearest_points(c, existing_tree_geoms[int(tree.nearest(c))])[1]
        pts = extend_line(np.array([[c.x, c.y], [q.x, q.y]]), False, True)
        surf.append(RoadLine(pts, _attrs("service", f"{kind.title()} Access", "", None, "15 mph"), protect_ends=True))
        entrance_nodes.append((float(c.x), float(c.y)))

    # School driveways: entrance -> nearest street, entrance end is a dead end.
    for s in sch:
        for _eid, ex, ez in s["entrances"]:
            E = Point(ex, ez)
            k = int(tree.nearest(E))
            q = nearest_points(E, existing_tree_geoms[k])[1]
            pts = np.array([[ex, ez], [q.x, q.y]])
            if np.hypot(*(pts[1] - pts[0])) < 25.0:  # too close: push the entrance node into the campus
                v = np.array([s["x"] - q.x, s["z"] - q.y])
                v = v / max(np.linalg.norm(v), 1e-9)
                pts[0] = pts[1] + v * 60.0
            pts = extend_line(pts, False, True)
            rl = RoadLine(pts, _attrs("service", f"{s['name']} Drive", "", None, "15 mph"), protect_ends=True)
            surf.append(rl)
            entrance_nodes.append((float(pts[0][0]), float(pts[0][1])))

    lines = surf + [i15.pos, i15.neg, sr56.pos, sr56.neg] + ramps
    return Layout(lines=lines, pods=pods, campuses=campuses, parcels=parcels, residential_lines=residential, entrance_nodes=entrance_nodes)


def _pod_streets(pod: Polygon, rng: np.random.Generator) -> list[LineString]:
    """Curvy local streets inside a pod: a loop road plus warped parallel/cross streets."""
    ring = LineString(np.asarray(pod.exterior.coords))
    out = [ring]
    theta = float(rng.uniform(0, math.pi))
    c, s = math.cos(theta), math.sin(theta)
    cen = pod.centroid
    R = math.sqrt(pod.area) * 1.2 + 200
    clip = pod.buffer(OVERSHOOT_M / 2)
    phase = float(rng.uniform(0, 2 * math.pi))
    for spacing, along_u in ((STREET_SPACING_M, True), (CROSS_SPACING_M, False)):
        offs = np.arange(-R, R, spacing) + float(rng.uniform(0, spacing))
        for o in offs:
            t = np.arange(-R, R + 1, 20.0)
            w = WARP_AMPLITUDE_M * np.sin(2 * math.pi * t / WARP_WAVELENGTH_M + phase + o * 0.01)
            if along_u:
                lu, lv = t, o + w
            else:
                lu, lv = o + w, t
            x = cen.x + lu * c - lv * s
            z = cen.y + lu * s + lv * c
            ls = LineString(np.column_stack([x, z]))
            for part in iter_lines(ls.intersection(clip)):
                if part.length > 60.0:
                    out.append(part)
    return out


def _pod_connectors(pod: Polygon, existing: list[LineString], tree: STRtree, rng: np.random.Generator) -> list[LineString]:
    ring = LineString(np.asarray(pod.exterior.coords))
    n = max(8, int(ring.length / 40.0))
    samples = [ring.interpolate(d) for d in np.linspace(0, ring.length, n, endpoint=False)]
    best = []
    for k, p in enumerate(samples):
        j = int(tree.nearest(p))
        q = nearest_points(p, existing[j])[1]
        best.append((p.distance(q), k, p, q))
    best.sort(key=lambda t: t[0])
    if best[0][0] > 650.0:
        return []
    chosen = [best[0]]
    for cand in best[1:]:
        if cand[0] > min(best[0][0] + 200.0, 450.0):
            break
        if all(abs(cand[1] - c[1]) * (ring.length / n) > max(350.0, ring.length / 4) and abs(cand[1] - c[1]) * (ring.length / n) < ring.length - 350.0 for c in chosen):
            chosen.append(cand)
        if len(chosen) >= 3:
            break
    out = []
    for _, _, p, q in chosen:
        pts = np.array([[p.x, p.y], [q.x, q.y]])
        if np.hypot(*(pts[1] - pts[0])) < 1.0:
            continue
        out.append(LineString(extend_line(pts, True, True)))
    return out


# ---------------------------------------------------------------------------
# Noding -> OSM-style MultiDiGraph (EPSG:32611)
# ---------------------------------------------------------------------------


def node_lines(lines: list[RoadLine]) -> list[tuple[np.ndarray, dict[str, Any], bool]]:
    """Split surf lines at mutual crossings and every line at forced splits.

    Returns pieces (coords, attrs, protect) whose shared endpoints have identical coords.
    """
    surf_idx = [i for i, ln in enumerate(lines) if ln.layer == "surf"]
    geoms = [lines[i].ls for i in surf_idx]
    tree = STRtree(geoms)
    for a_pos, i in enumerate(surf_idx):
        gi = geoms[a_pos]
        for b_pos in tree.query(gi):
            b_pos = int(b_pos)
            if b_pos <= a_pos:
                continue
            j = surf_idx[b_pos]
            inter = gi.intersection(geoms[b_pos])
            if inter.is_empty:
                continue
            pts: list[Point] = []
            if isinstance(inter, Point):
                pts = [inter]
            else:
                for g in getattr(inter, "geoms", [inter]):
                    if isinstance(g, Point):
                        pts.append(g)
                    elif isinstance(g, LineString):
                        pts += [Point(g.coords[0]), Point(g.coords[-1])]
            for p in pts:
                xy = (float(p.x), float(p.y))
                lines[i].splits.append((gi.project(p), xy))
                lines[j].splits.append((geoms[b_pos].project(p), xy))
    pieces = []
    for ln in lines:
        ls = ln.ls
        L = ls.length
        # recompute distances from the exact split coords (lines may have been extended after splitting)
        proj = [(ls.project(Point(xy)), xy) for _, xy in ln.splits]
        cuts = sorted([(d, xy) for d, xy in proj if 0.5 < d < L - 0.5], key=lambda t: t[0])
        marks: list[tuple[float, tuple[float, float]]] = [(0.0, (float(ln.coords[0][0]), float(ln.coords[0][1])))]
        for d, xy in cuts:
            if d - marks[-1][0] > 0.75:
                marks.append((d, xy))
        end = (float(ln.coords[-1][0]), float(ln.coords[-1][1]))
        if L - marks[-1][0] > 0.75:
            marks.append((L, end))
        else:
            marks[-1] = (L, end) if len(marks) > 1 else marks[-1]
        for (d0, p0), (d1, p1) in zip(marks[:-1], marks[1:], strict=True):
            seg = substring(ls, d0, d1)
            c = np.asarray(seg.coords)[:, :2].copy()
            if len(c) < 2:
                c = np.array([p0, p1])
            c[0] = p0
            c[-1] = p1
            pieces.append((c, ln.attrs, ln.protect_ends))
    return pieces


def pieces_to_graph(pieces: list[tuple[np.ndarray, dict[str, Any], bool]], protected_pts: list[tuple[float, float]]) -> nx.MultiDiGraph:
    o = scene_origin()

    def key(p: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        return (int(round(p[0] * 100)), int(round(p[1] * 100)))

    # remove dangling overshoot pieces (iteratively)
    prot = {key(p) for p in protected_pts}
    alive = list(range(len(pieces)))
    for _ in range(3):
        deg: dict[tuple[int, int], int] = {}
        for i in alive:
            c = pieces[i][0]
            for k in (key(c[0]), key(c[-1])):
                deg[k] = deg.get(k, 0) + 1
        keep = []
        for i in alive:
            c, _, protect = pieces[i]
            L = float(np.sum(np.hypot(*np.diff(c, axis=0).T)))
            k0, k1 = key(c[0]), key(c[-1])
            dangling = (deg[k0] == 1 and k0 not in prot) or (deg[k1] == 1 and k1 not in prot)
            if dangling and L < DANGLE_MAX_M and not protect:
                continue
            if L < 0.5:
                continue
            keep.append(i)
        if len(keep) == len(alive):
            break
        alive = keep

    G = nx.MultiDiGraph(crs="EPSG:32611", simplified=True, synthetic=True)
    ids: dict[tuple[int, int], int] = {}

    def nid(p: np.ndarray) -> int:
        k = key(p)
        if k not in ids:
            ids[k] = len(ids) + 1
            G.add_node(ids[k], x=float(p[0] + o.easting), y=float(o.northing - p[1]))
        return ids[k]

    for i in alive:
        c, attrs, _ = pieces[i]
        u, v = nid(c[0]), nid(c[-1])
        if u == v:
            continue
        utm = np.column_stack([c[:, 0] + o.easting, o.northing - c[:, 1]])
        L = float(np.sum(np.hypot(*np.diff(utm, axis=0).T)))
        a = dict(attrs)
        G.add_edge(u, v, **a, length=L, geometry=LineString(utm))
        if not attrs.get("oneway", False):
            G.add_edge(v, u, **a, length=L, geometry=LineString(utm[::-1]), reversed=True)
    # signals: junctions where two different named arterials meet, and ramp terminals
    major = {"primary", "secondary", "tertiary"}
    for n in G.nodes:
        names = set()
        ramp = False
        for _, _, d in list(G.in_edges(n, data=True)) + list(G.out_edges(n, data=True)):
            if d["highway"] in major:
                names.add(d["name"])
            if d["highway"] == "motorway_link":
                ramp = True
        if len(names) >= 2 or (ramp and names):
            G.nodes[n]["highway"] = "traffic_signals"
    return G


# ---------------------------------------------------------------------------
# Buildings
# ---------------------------------------------------------------------------


def _rect(cx: float, cz: float, w: float, d: float, t: np.ndarray) -> Polygon:
    n = np.array([-t[1], t[0]])
    pts = [np.array([cx, cz]) + a * t * w / 2 + b * n * d / 2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return Polygon(pts)


class _Occupancy:
    def __init__(self, cell: float = 40.0):
        self.cell = cell
        self.grid: dict[tuple[int, int], list[Polygon]] = {}

    def _cells(self, p: Polygon) -> list[tuple[int, int]]:
        x0, z0, x1, z1 = p.bounds
        return [(i, j) for i in range(int(x0 // self.cell), int(x1 // self.cell) + 1) for j in range(int(z0 // self.cell), int(z1 // self.cell) + 1)]

    def free(self, p: Polygon) -> bool:
        for c in self._cells(p):
            for q in self.grid.get(c, []):
                if q.intersects(p):
                    return False
        return True

    def add(self, p: Polygon) -> None:
        for c in self._cells(p):
            self.grid.setdefault(c, []).append(p)


def make_buildings(layout: Layout, ext: Extent, schools_cfg: list[dict[str, Any]], target_hh: int, seed: int = SEED) -> gpd.GeoDataFrame:
    """OSM-style footprints (EPSG:32611) with building/levels/height/name/amenity tags."""
    rng = np.random.default_rng(seed + 2)
    lane_w = float(assumption("roads.lane_width_m"))
    road_polys = []
    for ln in layout.lines:
        lanes = int(ln.attrs.get("lanes") or 2)
        lanes = lanes * 2 if ln.attrs.get("oneway") else lanes
        road_polys.append(ln.ls.buffer(max(lanes, 2) * lane_w / 2 + 2.5, cap_style=2))
    road_tree = STRtree(road_polys)
    occ = _Occupancy()
    excl = unary_union([p.buffer(10) for p in layout.campuses.values()] + [p[1].buffer(5) for p in layout.parcels])
    inner = box(ext.min_x + 30, ext.min_z + 30, ext.max_x - 30, ext.max_z - 30)
    feats: list[dict[str, Any]] = []

    def ok(p: Polygon) -> bool:
        if not inner.contains(p) or p.intersects(excl):
            return False
        if len(road_tree.query(p, predicate="intersects")):
            return False
        return occ.free(p)

    # Schools
    for s in schools_scene(schools_cfg):
        c = np.array([s["x"], s["z"]])
        hs = s["grades"][0] >= 9
        ms = s["grades"][0] >= 6
        if hs:
            specs = [((0, 0), 110, 65, "2"), ((-75, 55), 80, 24, "2"), ((75, 55), 80, 24, None), ((0, -85), 60, 45, None), ((-70, -60), 40, 30, None)]
        elif ms:
            specs = [((0, 0), 90, 50, "2"), ((-55, 50), 60, 22, None), ((55, -50), 45, 35, None)]
        else:
            specs = [((0, 0), 70, 34, None), ((0, 45), 60, 20, None), ((-50, -35), 30, 24, None)]
        t = np.array([1.0, 0.0])
        for k, ((dx, dz), w, d, lv) in enumerate(specs):
            p = _rect(c[0] + dx, c[1] + dz, w, d, t)
            if len(road_tree.query(p, predicate="intersects")):
                continue
            occ.add(p)
            feats.append({"geometry": p, "building": "school", "amenity": "school" if k == 0 else None, "name": s["name"] if k == 0 else None, "building:levels": lv, "height": None, "school_id": s["id"]})

    # Commercial and apartment parcels
    for kind, poly, rot in layout.parcels:
        t = np.array([math.cos(rot), math.sin(rot)])
        n = np.array([-t[1], t[0]])
        cen = np.asarray(poly.centroid.coords[0])
        x0, _, x1, _ = Polygon([(np.dot(np.asarray(q) - cen, t), np.dot(np.asarray(q) - cen, n)) for q in poly.exterior.coords]).bounds
        L = x1 - x0
        _, z0, _, z1 = Polygon([(np.dot(np.asarray(q) - cen, t), np.dot(np.asarray(q) - cen, n)) for q in poly.exterior.coords]).bounds
        D = z1 - z0
        if kind == "apartments":
            bw, bd, gap = 52.0, 28.0, 22.0
        else:
            bw, bd, gap = float(rng.choice([45.0, 70.0, 90.0])), float(rng.choice([35.0, 50.0])), 30.0
        nu = max(1, int((L - gap) // (bw + gap)))
        nv = max(1, int((D - gap) // (bd + gap)))
        for iu in range(nu):
            for iv in range(nv):
                u = -L / 2 + gap + bw / 2 + iu * (bw + gap)
                v = -D / 2 + gap + bd / 2 + iv * (bd + gap)
                pc = cen + t * u + n * v
                p = _rect(pc[0], pc[1], bw, bd, t)
                if len(road_tree.query(p, predicate="intersects")) or not occ.free(p):
                    continue
                occ.add(p)
                if kind == "apartments":
                    feats.append({"geometry": p, "building": "apartments", "building:levels": "3", "height": None, "name": None, "amenity": None})
                else:
                    r = rng.random()
                    lv = "2" if r < 0.3 else None
                    h = "12" if 0.3 <= r < 0.45 else None
                    feats.append({"geometry": p, "building": "commercial" if r < 0.7 else "retail", "building:levels": lv, "height": h, "name": None, "amenity": None, "shop": "yes" if r >= 0.7 else None})

    # Houses along residential streets
    houses = []
    for ls in layout.residential_lines:
        L = ls.length
        if L < 30:
            continue
        for d in np.arange(12.0, L - 12.0, LOT_SPACING_M):
            d2 = float(d + rng.uniform(-1.5, 1.5))
            p0 = np.asarray(ls.interpolate(max(d2 - 2, 0)).coords[0])
            p1 = np.asarray(ls.interpolate(min(d2 + 2, L)).coords[0])
            t = p1 - p0
            if np.linalg.norm(t) < 1e-6:
                continue
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            pc = (p0 + p1) / 2
            for side in (-1.0, 1.0):
                w = float(rng.uniform(11.5, 14.5))
                dep = float(rng.uniform(13.0, 17.0))
                setback = 6.5 + dep / 2
                hc = pc + side * n * setback
                p = _rect(hc[0], hc[1], w, dep, t)
                if rng.random() < 0.22:  # L-shaped (garage wing) -> flat roof path
                    wing_c = hc + t * (w / 2 + 3.0) + side * n * (dep / 4)
                    p = p.union(_rect(wing_c[0], wing_c[1], 6.5, dep / 2, t)).simplify(0.05)
                    if not isinstance(p, Polygon):
                        continue
                if ok(p):
                    occ.add(p)
                    houses.append(p)
    from pipeline.build_population import household_capacity

    apt_cap = sum(household_capacity("apartments", f["geometry"].area, int(f["building:levels"])) for f in feats if f["building"] == "apartments")
    target_houses = max(0, target_hh - apt_cap)
    if len(houses) > target_houses:
        keep = np.sort(rng.choice(len(houses), size=target_houses, replace=False))
        houses = [houses[i] for i in keep]
    for p in houses:
        r = rng.random()
        feats.append({"geometry": p, "building": "house" if r < 0.6 else ("yes" if r < 0.9 else "residential"), "building:levels": "2" if r < 0.15 else None, "height": None, "name": None, "amenity": None})
    log(f"synthetic: {len(houses):,} houses, {sum(1 for f in feats if f['building'] == 'apartments')} apartment buildings, {len(feats):,} footprints total")

    o = scene_origin()
    geoms = []
    for f in feats:
        c = np.asarray(f["geometry"].exterior.coords)
        geoms.append(Polygon(np.column_stack([c[:, 0] + o.easting, o.northing - c[:, 1]])))
    gdf = gpd.GeoDataFrame([{k: v for k, v in f.items() if k != "geometry"} for f in feats], geometry=geoms, crs="EPSG:32611")
    for col in ("shop", "school_id", "amenity", "name", "height", "building:levels"):
        if col not in gdf.columns:
            gdf[col] = None
    return gdf


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@dataclass
class SyntheticWorld:
    terrain: Terrain
    albedo: AlbedoFn
    graph: nx.MultiDiGraph
    footprints: gpd.GeoDataFrame
    layout: Layout


def make_world(region_ext: Extent, terrain_ext: Extent, schools_cfg: list[dict[str, Any]]) -> SyntheticWorld:
    target_hh = int(assumption("population.target_households_fallback"))
    # Rough split for sizing the street network: apartments take ~ a sixth of households.
    target_houses = int(target_hh * 0.84)
    terrain = synthetic_dem(terrain_ext)
    layout = build_layout(region_ext, schools_cfg, target_houses)
    grade_pods(terrain, [p.buffer(60) for p in layout.pods] + list(layout.campuses.values()) + [p[1].buffer(30) for p in layout.parcels])
    pieces = node_lines(layout.lines)
    G = pieces_to_graph(pieces, layout.entrance_nodes)
    log(f"synthetic: road graph {G.number_of_nodes():,} nodes, {G.number_of_edges():,} directed edges")
    fp = make_buildings(layout, region_ext, schools_cfg, target_hh)
    paved = [p[1] for p in layout.parcels]
    fields = list(layout.campuses.values())
    albedo = synthetic_albedo(terrain, [p.buffer(25) for p in layout.pods], paved, fields)
    return SyntheticWorld(terrain=terrain, albedo=albedo, graph=G, footprints=fp, layout=layout)
