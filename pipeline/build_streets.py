"""Street-level ground meshes (HD build): road surfaces with lane markings, intersections,
crosswalks, medians, curbs + sidewalks, house driveways and swimming pools.

Inputs are the unclipped projected drive graph (render roads also exist in the terrain
buffer beyond the region bbox), optional real Overture service / driveway / parking-aisle
segments and swimming-pool polygons, building shapes (for driveways and garage doors) and
the LOD0 render surface (`TerrainBuild.render`) everything is draped on.

Outputs per tile (docs/data_contract.md "HD world build"):
    roads/roads_r{r}_c{c}.glb   primitives "asphalt" (_MAT 6 ground) and "markings" (_MAT 8)
    ground/ground_r{r}_c{c}.glb primitives "sidewalks", "driveways", "medians", "pools"

UVs follow client/public/assets/materials/materials_manifest.json (ground atlas cells and
ground_markings.png columns); COLOR_0 is the color-only fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, substring, unary_union

from pipeline.build_roads import FREEWAY_CLASSES, as_str, first, lanes_per_direction, parse_bool
from pipeline.build_terrain import Terrain
from pipeline.building_geom import fix_winding, ring_coords, triangulate
from pipeline.common import Extent, TileGrid, log
from pipeline.config import assumption
from pipeline.geo import scene_origin
from pipeline.glb import MeshData, write_glb

MAT_GROUND = 6
MAT_MARKING = 8  # pipeline extension: _VARIANT = ground_markings.png column

GROUND_CELLS = [  # fallback = blender/rdlib/matgen.py GROUND order (materials_manifest.json wins)
    ("asphalt_fresh", (4, 4)), ("asphalt_worn", (4, 4)), ("asphalt_parking", (4, 4)), ("concrete_sidewalk", (3, 3)),
    ("concrete_driveway", (6, 6)), ("curb_gutter", (3, 0.75)), ("pavers", (3, 3)), ("concrete_plaza", (6, 6)),
    ("grass_lawn", (2, 2)), ("grass_patchy", (2, 2)), ("chaparral", (4, 4)), ("coastal_sage", (4, 4)),
    ("decomposed_granite", (2, 2)), ("bare_dirt", (4, 4)), ("mulch", (2, 2)), ("pool_water", (2, 2)),
]
MARKING_COLS = [  # name, column width m, period m (fallback = matgen MARKINGS)
    ("white_solid_4in", 0.3, 8.0), ("white_dashed_4in", 0.3, 12.0), ("yellow_solid_4in", 0.3, 8.0), ("yellow_double_4in", 0.5, 8.0),
    ("yellow_dashed_4in", 0.3, 12.0), ("white_solid_8in", 0.4, 8.0), ("white_bar_12in", 0.5, 4.0), ("crosswalk_bar_24in", 1.0, 3.0),
]
MARK_RGB = {"white": (236, 236, 230), "yellow": (227, 178, 60)}

# lifts above the LOD0 render surface (m); the client adds polygonOffset on top
LIFT_SERVICE = 0.04
LIFT_ROAD = 0.06
LIFT_GUTTER = 0.07
LIFT_PATCH = 0.08
LIFT_MARK = 0.10
LIFT_XWALK = 0.11
LIFT_DRIVEWAY = 0.12

ARTERIAL = {"primary", "secondary", "tertiary", "primary_link", "secondary_link", "tertiary_link", "trunk"}
RESIDENTIAL = {"residential", "living_street", "unclassified"}
SIDEWALK_CLASSES = {"primary", "secondary", "tertiary", "residential", "living_street", "unclassified"}
RAMP_CLASSES = {"motorway_link", "trunk_link"}


class Materials:
    """Ground cell / marking column lookup (materials_manifest.json, else the fallback lists)."""

    def __init__(self, manifest: dict[str, Any] | None = None):
        self.cells = {n: (i, w) for i, (n, w) in enumerate(GROUND_CELLS)}
        self.marks = {n: (i, w, p) for i, (n, w, p) in enumerate(MARKING_COLS)}
        if manifest:
            try:
                for c in manifest["atlases"]["ground"]["cells"]:
                    self.cells[c["name"]] = (int(c["index"]), tuple(c.get("world_size_m", (4, 4))))
                for c in manifest["markings"]["columns"]:
                    self.marks[c["name"]] = (int(c["column"]), float(c["world_width_m"]), float(c["period_m"]))
            except (KeyError, TypeError, ValueError):
                pass

    def cell(self, name: str) -> tuple[int, tuple[float, float]]:
        return self.cells[name]

    def mark(self, name: str) -> tuple[int, float, float]:
        return self.marks[name]


# ---------------------------------------------------------------------------
# Road records
# ---------------------------------------------------------------------------


@dataclass
class Road:
    line: LineString  # scene (x, z)
    cls: str
    lanes_fwd: int  # lanes in the geometry direction
    lanes_bwd: int  # lanes against it (0 for one-way)
    name: str
    half_w: float
    e_r: float  # extra width right of travel (bike lane / parking / shoulder), geometry direction
    e_l: float  # extra width left of the left-most lane (one-way) or 0
    u: Any = None
    v: Any = None
    kind: str = "road"  # road | service
    bridge: bool = False

    @property
    def oneway(self) -> bool:
        return self.lanes_bwd == 0

    @property
    def sidewalk(self) -> bool:
        return self.kind == "road" and self.cls in SIDEWALK_CLASSES


def road_extras(cls: str, oneway: bool) -> tuple[float, float]:
    """(right, left) extra widths beside the travel lanes (assumptions props.curb_extra_m.*,
    streets.*). Matches build_props' curb offsets for two-way streets."""
    if cls in ("motorway", "trunk"):
        return float(assumption("streets.freeway_shoulder_m")), 1.2
    if cls in RAMP_CLASSES:
        return 1.2, 0.6
    key = "arterial" if cls in ARTERIAL else "residential"
    e = float(assumption(f"props.curb_extra_m.{key}"))
    return (e, float(assumption("streets.median_shoulder_m"))) if oneway else (e, e)


def make_road(line: LineString, cls: str, fwd: int, bwd: int, name: str, u: Any = None, v: Any = None, bridge: bool = False) -> Road:
    lw = float(assumption("roads.lane_width_m"))
    e_r, e_l = road_extras(cls, bwd == 0)
    if bwd == 0:
        half = (fwd * lw + e_r + e_l) / 2.0
    else:
        half = (fwd + bwd) * lw / 2.0 + e_r
        e_l = e_r
    return Road(line=line, cls=cls, lanes_fwd=fwd, lanes_bwd=bwd, name=name, half_w=half, e_r=e_r, e_l=e_l, u=u, v=v, bridge=bridge)


def utm_line_to_scene(coords: np.ndarray) -> LineString:
    o = scene_origin()
    c = np.asarray(coords, dtype=np.float64)[:, :2]
    return LineString(np.column_stack([c[:, 0] - o.easting, o.northing - c[:, 1]]))


def visual_roads(G: nx.MultiDiGraph, extent: Extent) -> list[Road]:
    """Render roads from the unclipped projected drive graph: one record per carriageway
    (two-way pairs merged), clipped to `extent`."""
    clip = box(extent.min_x - 50, extent.min_z - 50, extent.max_x + 50, extent.max_z + 50)
    seen: dict[tuple[Any, ...], int] = {}
    out: list[Road] = []
    for u, v, _k, d in G.edges(keys=True, data=True):
        hw = as_str(first(d.get("highway"))) or "unclassified"
        if hw == "service":
            continue
        oneway = parse_bool(d.get("oneway"))
        g = d.get("geometry")
        coords = np.asarray(g.coords)[:, :2] if g is not None else np.array([[G.nodes[u]["x"], G.nodes[u]["y"]], [G.nodes[v]["x"], G.nodes[v]["y"]]])
        pu = np.array([G.nodes[u]["x"], G.nodes[u]["y"]])
        if np.hypot(*(coords[0] - pu)) > np.hypot(*(coords[-1] - pu)):
            coords = coords[::-1]
        key = (min(u, v), max(u, v), as_str(d.get("osmid")), int(round(float(d.get("length", 0.0)))))
        lanes = lanes_per_direction(hw, d.get("lanes"), oneway, d.get("lanes:forward"), d.get("lanes:backward"), parse_bool(d.get("reversed")))
        if not oneway and key in seen:
            r = out[seen[key]]
            r.lanes_bwd = max(r.lanes_bwd, lanes)
            continue
        line = utm_line_to_scene(coords)
        if not line.intersects(clip):
            continue
        seen[key] = len(out)
        bwd = 0 if oneway else lanes
        out.append(make_road(line, hw, lanes, bwd, as_str(d.get("name")), u, v, as_str(d.get("bridge")) == "yes"))
    # rebuild half widths now that two-way lane totals are known
    for i, r in enumerate(out):
        out[i] = make_road(r.line, r.cls, r.lanes_fwd, r.lanes_bwd, r.name, r.u, r.v, r.bridge)
    return out


def overture_service_roads(raw: Path, extent: Extent) -> list[Road]:
    """Real service roads, parking aisles, alleys and driveways from the Overture
    transportation cache (data/raw/overture/segment.parquet); [] if absent."""
    p = raw / "overture" / "segment.parquet"
    if not p.exists():
        return []
    import pyarrow.parquet as pq

    from pipeline.geo import lonlat_to_utm

    t = pq.read_table(p, columns=["geometry", "subtype", "class", "subclass"]).to_pandas()
    t = t[(t["subtype"] == "road") & (t["class"] == "service")]
    widths = {
        "parking_aisle": float(assumption("props.parking_aisle_width_m")),
        "driveway": float(assumption("streets.driveway_segment_width_m")),
    }
    default_w = float(assumption("streets.service_road_width_m"))
    clip = box(extent.min_x, extent.min_z, extent.max_x, extent.max_z)
    out = []
    for g, sub in zip(shapely.from_wkb(t["geometry"].to_numpy()), t["subclass"], strict=True):
        if g is None or g.geom_type != "LineString":
            continue
        c = np.asarray(g.coords)
        utm = np.array([lonlat_to_utm(float(x), float(y)) for x, y in c])
        line = utm_line_to_scene(utm)
        if not line.intersects(clip):
            continue
        w = widths.get(str(sub), default_w)
        out.append(Road(line=line, cls="service", lanes_fwd=1, lanes_bwd=1, name="", half_w=w / 2.0, e_r=0.0, e_l=0.0, kind="service"))
    return out


def overture_pools(raw: Path, extent: Extent) -> list[Polygon]:
    """Swimming-pool polygons (Overture base/water class=swimming_pool, i.e. OSM
    leisure=swimming_pool) in scene coordinates; [] if the cache is absent."""
    p = raw / "overture" / "water.parquet"
    if not p.exists():
        return []
    import pyarrow.parquet as pq

    from pipeline.geo import lonlat_to_utm

    t = pq.read_table(p, columns=["geometry", "class"]).to_pandas()
    t = t[t["class"] == "swimming_pool"]
    o = scene_origin()
    out = []
    for g in shapely.from_wkb(t["geometry"].to_numpy()):
        if g is None:
            continue
        for poly in getattr(g, "geoms", [g]):
            if poly.geom_type != "Polygon" or poly.area <= 0:
                continue
            ext = np.array([lonlat_to_utm(float(x), float(y)) for x, y in np.asarray(poly.exterior.coords)])
            sp = Polygon(np.column_stack([ext[:, 0] - o.easting, o.northing - ext[:, 1]]))
            if sp.is_valid and sp.area > 4.0 and extent.contains(sp.centroid.x, sp.centroid.y):
                out.append(sp)
    return out


# ---------------------------------------------------------------------------
# Bridges
# ---------------------------------------------------------------------------


@dataclass
class Draper:
    """Samples the render surface; on bridge decks (bare-earth lidar has no bridges) the
    elevation is interpolated linearly between the deck ends instead."""

    terrain: Terrain
    spans: list[tuple[LineString, float, float]] = field(default_factory=list)  # deck line, y at start, y at end
    _tree: Any = None

    def __post_init__(self) -> None:
        self._tree = shapely.STRtree([s[0] for s in self.spans]) if self.spans else None

    def sample(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return self.terrain.sample(x, z)

    def line(self, pts: np.ndarray, tol: float = 3.0) -> np.ndarray:
        """Elevations along a polyline (n, 2), honoring bridge decks it runs along."""
        y = self.terrain.sample(pts[:, 0], pts[:, 1])
        if self._tree is None or len(pts) < 2:
            return y
        tan = np.gradient(pts, axis=0)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        geoms = shapely.points(pts)
        hits = self._tree.query(geoms, predicate="dwithin", distance=tol)
        for pi, si in zip(hits[0], hits[1], strict=True):
            ln, y0, y1 = self.spans[int(si)]
            d = ln.project(geoms[pi])
            a = np.asarray(ln.interpolate(max(d - 1.0, 0)).coords[0])
            b = np.asarray(ln.interpolate(min(d + 1.0, ln.length)).coords[0])
            st = (b - a) / max(np.hypot(*(b - a)), 1e-9)
            if abs(float(st @ tan[pi])) < 0.85:
                continue  # crossing under / over, not along the deck
            f = d / max(ln.length, 1e-9)
            y[pi] = max(y[pi], y0 + (y1 - y0) * f)
        return y


def bridge_spans(raw: Path, terrain: Terrain, extent: Extent) -> list[tuple[LineString, float, float]]:
    """Bridge decks from Overture road_flags is_bridge (with their `between` ranges)."""
    p = raw / "overture" / "segment.parquet"
    if not p.exists():
        return []
    import pyarrow.parquet as pq
    from shapely.ops import substring

    from pipeline.geo import lonlat_to_utm

    t = pq.read_table(p, columns=["geometry", "road_flags", "subtype"]).to_pandas()
    t = t[(t["subtype"] == "road") & t["road_flags"].notna()]
    out = []
    for g, flags in zip(shapely.from_wkb(t["geometry"].to_numpy()), t["road_flags"], strict=True):
        if g is None or g.geom_type != "LineString":
            continue
        for rf in flags:
            vals = list(rf.get("values") if rf.get("values") is not None else [])
            if "is_bridge" not in vals:
                continue
            btw = rf.get("between")
            lo, hi = (0.0, 1.0) if btw is None else (float(btw[0]), float(btw[1]))
            utm = np.array([lonlat_to_utm(float(x), float(y)) for x, y in np.asarray(g.coords)])
            line = utm_line_to_scene(utm)
            deck = substring(line, lo, hi, normalized=True)
            if deck.geom_type != "LineString" or deck.length < 8.0:
                continue
            if not extent.contains(deck.centroid.x, deck.centroid.y):
                continue
            # extend a little onto the abutments so the deck ends sit on the approach fill
            c = np.asarray(deck.coords)
            y0 = float(terrain.sample(c[0, 0], c[0, 1]))
            y1 = float(terrain.sample(c[-1, 0], c[-1, 1]))
            out.append((deck, y0, y1))
    return out


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------


@dataclass
class Prim:
    """Growable primitive (one material family)."""

    name: str
    pos: list[np.ndarray] = field(default_factory=list)
    uv: list[np.ndarray] = field(default_factory=list)
    col: list[np.ndarray] = field(default_factory=list)
    mat: list[np.ndarray] = field(default_factory=list)
    var: list[np.ndarray] = field(default_factory=list)
    nrm: list[np.ndarray] = field(default_factory=list)
    tri: list[np.ndarray] = field(default_factory=list)
    n: int = 0

    def add(self, pos: np.ndarray, uv: np.ndarray, tri: np.ndarray, mat: int, var: int, color: tuple[int, int, int], nrm: np.ndarray | None = None) -> None:
        if len(pos) == 0 or len(tri) == 0:
            return
        k = len(pos)
        pos = np.asarray(pos, dtype=np.float64)
        tri = np.asarray(tri, dtype=np.int64)
        if nrm is None:
            nrm = np.tile([0.0, 1.0, 0.0], (k, 1))
            tri = fix_winding(pos, tri, np.tile([0.0, 1.0, 0.0], (len(tri), 1)))
        self.pos.append(pos)
        self.uv.append(np.asarray(uv, dtype=np.float64))
        self.nrm.append(np.asarray(nrm, dtype=np.float64))
        self.tri.append(tri + self.n)
        self.col.append(np.tile(np.array([*color, 255], dtype=np.uint8), (k, 1)))
        self.mat.append(np.full(k, mat, np.uint8))
        self.var.append(np.full(k, var, np.uint8))
        self.n += k

    def mesh(self) -> MeshData | None:
        if not self.n:
            return None
        return MeshData(
            name=self.name,
            positions=np.concatenate(self.pos).astype(np.float32),
            normals=np.concatenate(self.nrm).astype(np.float32),
            uvs=np.concatenate(self.uv).astype(np.float32),
            colors=np.concatenate(self.col),
            indices=np.concatenate(self.tri).reshape(-1).astype(np.uint32),
            custom={"_MAT": np.concatenate(self.mat), "_VARIANT": np.concatenate(self.var)},
            roughness=0.95,
        )


def polys_of(g: Any) -> list[Polygon]:
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon) and not p.is_empty]


def drape_polygon(prim: Prim, poly: Polygon, draper: Draper, lift: float, var: int, color: tuple[int, int, int], uv_size: tuple[float, float], seg: float = 4.0, cell: float | None = None, y_fixed: float | None = None) -> None:
    """Triangulate a polygon (densified boundary; optionally split on a `cell` grid so big
    polygons follow the terrain) and drape it. Planar UVs u = x / size, v = -z / size."""
    pieces = [poly]
    if cell is not None:
        x0, z0, x1, z1 = poly.bounds
        if max(x1 - x0, z1 - z0) > cell * 1.5:
            xs = np.arange(math.floor(x0 / cell) * cell, x1 + cell, cell)
            zs = np.arange(math.floor(z0 / cell) * cell, z1 + cell, cell)
            boxes = shapely.box(*np.meshgrid(xs[:-1], zs[:-1]), *np.meshgrid(xs[1:], zs[1:]))
            parts = shapely.intersection(poly, boxes.ravel())
            pieces = [p for g in parts for p in polys_of(g)]
    for p in pieces:
        p = shapely.segmentize(p, seg)
        v, t = triangulate(p)
        if not len(t):
            continue
        y = np.full(len(v), y_fixed) if y_fixed is not None else draper.sample(v[:, 0], v[:, 1]) + lift
        pos = np.column_stack([v[:, 0], y, v[:, 1]])
        uv = np.column_stack([v[:, 0] / uv_size[0], -v[:, 1] / uv_size[1]])
        prim.add(pos, uv, t, MAT_GROUND, var, color)


def strip_mesh(pts: np.ndarray, y: np.ndarray, off_a: float, off_b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strip between lateral offsets off_a..off_b (meters, + = left of travel in scene x/z
    seen from above with z south) along a polyline. Returns pos (2n, 3), across (2n,), along (2n,)."""
    seg = np.diff(pts, axis=0)
    sl = np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-9)
    t = seg / sl[:, None]
    tan = np.zeros_like(pts)
    tan[0], tan[-1] = t[0], t[-1]
    if len(pts) > 2:
        tan[1:-1] = t[:-1] + t[1:]
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    left = np.column_stack([tan[:, 1], -tan[:, 0]])  # left of travel (x east, z south)
    a = pts + left * off_a
    b = pts + left * off_b
    n = len(pts)
    pos = np.empty((2 * n, 3))
    pos[0::2] = np.column_stack([a[:, 0], y, a[:, 1]])
    pos[1::2] = np.column_stack([b[:, 0], y, b[:, 1]])
    along = np.repeat(np.concatenate([[0.0], np.cumsum(sl)]), 2)
    across = np.tile([off_a, off_b], n)
    return pos, across, along


def strip_tris(n: int) -> np.ndarray:
    i = np.arange(n - 1)
    a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
    return np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])


def densify_line(line: LineString, step: float) -> np.ndarray:
    seg = shapely.segmentize(line, step)
    return np.asarray(seg.coords)[:, :2]


# ---------------------------------------------------------------------------
# Lane markings
# ---------------------------------------------------------------------------


def marking_lines(r: Road) -> list[tuple[float, str]]:
    """(offset from the centerline, + = left of travel, marking column) for one carriageway."""
    if r.kind != "road":
        return []
    lw = float(assumption("roads.lane_width_m"))
    out: list[tuple[float, str]] = []
    hw = r.half_w
    if r.oneway:
        n = r.lanes_fwd
        right = -hw + r.e_r  # right edge of travel lanes
        out.append((right, "white_solid_4in"))
        for k in range(1, n):
            out.append((right + k * lw, "white_dashed_4in"))
        out.append((right + n * lw, "yellow_solid_4in"))
        return out
    n_f, n_b = r.lanes_fwd, r.lanes_bwd
    if r.cls in RESIDENTIAL and n_f + n_b <= 2:
        return []  # local streets: no paint
    c = (n_f - n_b) * lw / 2.0  # center line (boundary between the directions)
    out.append((c, "yellow_double_4in"))
    for k in range(1, n_f):
        out.append((c - k * lw, "white_dashed_4in"))
    for k in range(1, n_b):
        out.append((c + k * lw, "white_dashed_4in"))
    if r.e_r >= 1.2 and r.cls in ARTERIAL:  # bike lane / shoulder line
        out.append((c - n_f * lw, "white_solid_4in"))
        out.append((c + n_b * lw, "white_solid_4in"))
    return out


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass
class StreetLayout:
    roads: list[Road]
    roads_poly: Any  # all paved road surface (filleted), scene x/z
    junction_patches: list[Polygon]
    crosswalks: list[tuple[np.ndarray, np.ndarray, float, float]]  # (center, across dir, width, depth)
    stop_bars: list[tuple[np.ndarray, np.ndarray, float]]  # (center, across dir, length)
    medians: list[Polygon]
    sidewalks: list[Polygon]
    driveways: list[tuple[Polygon, np.ndarray]]  # polygon, direction (house -> street)
    pools: list[Polygon]
    frontage: dict[int, Any]
    stats: dict[str, Any]


def _buffer_roads(roads: list[Road], pad: float = 0.0) -> Any:
    geoms = [r.line.buffer(r.half_w + pad, cap_style="round", join_style="round", quad_segs=4) for r in roads]
    return unary_union(geoms) if geoms else Polygon()


def junctions(roads: list[Road]) -> dict[Any, list[tuple[int, bool]]]:
    """Graph node -> [(road index, starts_here)] for nodes where >= 3 carriageway ends meet."""
    ends: dict[Any, list[tuple[int, bool]]] = {}
    for i, r in enumerate(roads):
        if r.kind != "road" or r.u is None:
            continue
        ends.setdefault(r.u, []).append((i, True))
        ends.setdefault(r.v, []).append((i, False))
    return {n: e for n, e in ends.items() if len(e) >= 3}


def layout_streets(
    roads: list[Road],
    extent: Extent,
    signal_xz: np.ndarray,
    shapes: dict[int, Any],
    pools: list[Polygon],
    hero_disks: list[Polygon],
) -> StreetLayout:
    from pipeline.build_buildings import Frontage

    curb_r = float(assumption("streets.curb_return_radius_m"))
    walk_w = float(assumption("streets.sidewalk_width_m"))
    xw_depth = float(assumption("streets.crosswalk_depth_m"))
    road_main = [r for r in roads if r.kind == "road"]
    service = [r for r in roads if r.kind == "service" and not any(r.line.intersects(h) for h in hero_disks)]
    roads = road_main + service
    main_poly = _buffer_roads(road_main).buffer(curb_r).buffer(-curb_r)  # curb returns at corners
    roads_poly = unary_union([main_poly, _buffer_roads(service)])
    # junction patches + crosswalks
    jn = junctions(road_main)
    sig_tree = shapely.STRtree(shapely.points(signal_xz)) if len(signal_xz) else None
    patches: list[Polygon] = []
    crosswalks: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    stop_bars: list[tuple[np.ndarray, np.ndarray, float]] = []
    for _node, inc in jn.items():
        i0, s0 = inc[0]
        r0 = road_main[i0]
        p = np.asarray(r0.line.coords[0] if s0 else r0.line.coords[-1])
        if all(road_main[i].cls in FREEWAY_CLASSES for i, _ in inc):
            continue  # freeway merges / gores: no patch
        reach = max(road_main[i].half_w for i, _ in inc) + 1.5
        parts = []
        approaches = []
        for i, starts in inc:
            r = road_main[i]
            L = r.line.length
            trim = min(reach, L * 0.45)
            sub = substring(r.line, 0, trim) if starts else substring(r.line, L - trim, L)
            if sub.geom_type != "LineString" or sub.length < 0.5:
                continue
            parts.append(sub.buffer(r.half_w + 0.05, cap_style="flat"))
            q = np.asarray(sub.coords[-1] if starts else sub.coords[0])
            q2 = np.asarray(r.line.interpolate(min(trim + 1.0, L) if starts else max(L - trim - 1.0, 0)).coords[0])
            d = (q2 - q) / max(np.hypot(*(q2 - q)), 1e-9)  # away from the junction
            approaches.append((r, q, d, starts))
        if not parts:
            continue
        core = main_poly.intersection(Point(*p).buffer(reach, quad_segs=8))
        patch = unary_union([*parts, core]).intersection(roads_poly)
        patches.extend(polys_of(patch))
        signalized = sig_tree is not None and len(sig_tree.query(Point(*p), predicate="dwithin", distance=12.0)) > 0
        if not signalized:
            continue
        for r, q, d, starts in approaches:
            if r.cls in FREEWAY_CLASSES:
                continue
            across = np.array([d[1], -d[0]])
            c = q + d * (xw_depth / 2.0 + 0.3)
            crosswalks.append((c, across, 2 * r.half_w, xw_depth))
            # stop bar across the lanes that ENTER the junction on this approach
            lw = float(assumption("roads.lane_width_m"))
            n_in = r.lanes_bwd if starts else r.lanes_fwd
            if r.oneway and starts:
                continue  # one-way leaving the junction
            n_in = r.lanes_fwd if r.oneway else n_in
            if n_in <= 0:
                continue
            # entering lanes are on the right of the inbound travel direction (-d)
            side = np.array([d[1], -d[0]])  # right of travel when driving toward the junction (-d)
            if r.oneway:
                off = 0.0
            else:
                off = n_in * lw / 2.0
            sc = q + d * (xw_depth + 1.5) + side * off
            stop_bars.append((sc, across, n_in * lw))
    # medians between paired one-way carriageways of the same named arterial
    medians: list[Polygon] = []
    by_name: dict[str, list[Road]] = {}
    for r in road_main:
        if r.oneway and r.cls in ARTERIAL and r.name:
            by_name.setdefault(r.name.split(",")[0].strip().lower(), []).append(r)
    for _nm, rs in by_name.items():
        if len(rs) < 2:
            continue
        u = _buffer_roads(rs)
        closed = u.buffer(9.0, join_style="round").buffer(-9.0, join_style="round")
        gap = closed.difference(roads_poly)
        for g in polys_of(gap):
            if g.area < 30.0:
                continue
            # keep long thin strips (medians), not junction leftovers
            mrr = g.minimum_rotated_rectangle
            c = np.asarray(mrr.exterior.coords)
            e1, e2 = np.hypot(*(c[1] - c[0])), np.hypot(*(c[2] - c[1]))
            if min(e1, e2) > 0 and max(e1, e2) / min(e1, e2) >= 3.0:
                medians.append(g.buffer(-0.02))
    medians_u = unary_union(medians) if medians else Polygon()
    # building footprints (walls) in scene space
    bpolys = {bid: sh.walls for bid, sh in shapes.items()}
    btree_ids = list(bpolys)
    btree = shapely.STRtree([bpolys[b] for b in btree_ids]) if btree_ids else None
    # driveways + frontage for houses
    frontage: dict[int, Any] = {}
    driveways: list[tuple[Polygon, np.ndarray]] = []
    local = [r for r in road_main if r.cls not in FREEWAY_CLASSES]
    ltree = shapely.STRtree([r.line for r in local]) if local else None
    dw_w = float(assumption("streets.house_driveway_width_m"))
    dw_max = float(assumption("streets.house_driveway_max_m"))
    door_w = float(assumption("streets.garage_door_width_m"))
    houses = [(bid, sh) for bid, sh in shapes.items() if sh.type == "house"]
    if ltree is not None and houses:
        cents = shapely.points([(sh.walls.centroid.x, sh.walls.centroid.y) for _, sh in houses])
        nearest = ltree.query_nearest(cents, return_distance=False, all_matches=False)
        near_road = dict(zip(nearest[0].tolist(), nearest[1].tolist(), strict=True))
        for k, (bid, sh) in enumerate(houses):
            j = near_road.get(k)
            if j is None:
                continue
            road = local[j]
            poly = sh.walls
            pa, pb = nearest_points(poly.exterior, road.line)
            p = np.array([pa.x, pa.y])
            q = np.array([pb.x, pb.y])
            if np.hypot(*(q - p)) > dw_max + road.half_w + 5.0:
                continue
            ring = ring_coords(np.asarray(poly.exterior.coords))
            seg_a = ring
            seg_b = np.roll(ring, -1, axis=0)
            dd = seg_b - seg_a
            ln = np.hypot(dd[:, 0], dd[:, 1])
            # nearest wall edge to p
            tt = np.clip(np.einsum("ij,ij->i", p - seg_a, dd) / np.maximum(ln**2, 1e-9), 0, 1)
            proj = seg_a + dd * tt[:, None]
            ei = int(np.argmin(np.hypot(*(proj - p).T)))
            if ln[ei] < 3.0:
                # corner hit: take the longest edge facing the road
                to_r = (q - p) / max(np.hypot(*(q - p)), 1e-9)
                nn = np.column_stack([dd[:, 1], -dd[:, 0]]) / np.maximum(ln[:, None], 1e-9)
                score = (nn @ to_r) * ln
                ei = int(np.argmax(score))
            a, d_ = seg_a[ei], dd[ei]
            L = ln[ei]
            tdir = d_ / max(L, 1e-9)
            n_out = np.array([tdir[1], -tdir[0]])  # outward for CCW rings
            to_r = (q - p) / max(np.hypot(*(q - p)), 1e-9)
            if n_out @ to_r < 0.2:
                frontage[bid] = Frontage(nx=float(to_r[0]), nz=float(to_r[1]))
                continue
            frontage[bid] = Frontage(nx=float(n_out[0]), nz=float(n_out[1]))
            if L < 3.2:
                continue
            w = min(door_w, L - 0.6)
            s_proj = float(np.clip((p - a) @ tdir, w / 2 + 0.3, L - w / 2 - 0.3))
            gc = a + tdir * s_proj
            # ray from the garage toward the street: distance to the paved road
            ray = LineString([gc + n_out * 0.05, gc + n_out * (dw_max + 2 * road.half_w + 2)])
            hit = ray.intersection(roads_poly)
            if hit.is_empty:
                continue
            dist = min(Point(*gc).distance(g) for g in getattr(hit, "geoms", [hit]))
            if dist > dw_max or dist < 1.0:
                frontage[bid].garage = (float(gc[0]), float(gc[1]))
                frontage[bid].garage_width = float(w)
                continue
            dwy = Polygon([gc - tdir * dw_w / 2, gc + tdir * dw_w / 2, gc + tdir * dw_w / 2 + n_out * (dist + 0.4), gc - tdir * dw_w / 2 + n_out * (dist + 0.4)])
            # must not cross another building
            if btree is not None:
                hits = [btree_ids[int(i)] for i in btree.query(dwy, predicate="intersects")]
                if any(h != bid and bpolys[h].intersection(dwy).area > 0.5 for h in hits):
                    continue
            dwy = dwy.difference(poly)
            if dwy.is_empty:
                continue
            frontage[bid].garage = (float(gc[0]), float(gc[1]))
            frontage[bid].garage_width = float(w)
            driveways.append((dwy, n_out))
    dw_union = unary_union([d for d, _ in driveways]) if driveways else Polygon()
    # sidewalks (curb-adjacent walk on residential / arterial streets)
    walk_roads = [r for r in road_main if r.sidewalk]
    side_src = _buffer_roads(walk_roads).buffer(curb_r).buffer(-curb_r)
    side = side_src.buffer(walk_w, join_style="round", quad_segs=4).difference(roads_poly)
    cut = [medians_u, dw_union]
    if btree_ids:
        cut.append(unary_union([bpolys[b].buffer(0.2) for b in btree_ids]))
    if hero_disks:
        cut.append(unary_union(hero_disks))
    side = side.difference(unary_union([c for c in cut if not c.is_empty]))
    clipbox = box(extent.min_x, extent.min_z, extent.max_x, extent.max_z)
    side = side.intersection(clipbox)
    sidewalks = [p for p in polys_of(side) if p.area > 1.0]
    # pools (not under buildings, not inside hero campuses)
    pool_out = []
    for pl in pools:
        if any(pl.intersects(h) for h in hero_disks):
            continue
        pool_out.append(pl)
    stats = {
        "render_roads": len(road_main),
        "service_roads": len(service),
        "junction_patches": len(patches),
        "crosswalks": len(crosswalks),
        "stop_bars": len(stop_bars),
        "medians": len(medians),
        "sidewalk_area_m2": round(float(sum(p.area for p in sidewalks)), 1),
        "driveways": len(driveways),
        "garage_doors": sum(1 for f in frontage.values() if f.garage),
        "pools": len(pool_out),
    }
    return StreetLayout(roads=roads, roads_poly=roads_poly, junction_patches=patches, crosswalks=crosswalks, stop_bars=stop_bars,
                        medians=medians, sidewalks=sidewalks, driveways=driveways, pools=pool_out, frontage=frontage, stats=stats)


# ---------------------------------------------------------------------------
# Tile meshes
# ---------------------------------------------------------------------------


def _road_ribbon(asphalt: Prim, marks: Prim, r: Road, draper: Draper, mats: Materials, step: float = 6.0) -> None:
    pts = densify_line(r.line, step)
    if len(pts) < 2:
        return
    lift = LIFT_SERVICE if r.kind == "service" else LIFT_ROAD
    yl = draper.line(pts)
    deck = yl > draper.sample(pts[:, 0], pts[:, 1]) + 0.25  # on a bridge deck: level across
    pos, across, along = strip_mesh(pts, yl + lift, -r.half_w, r.half_w)
    ys = draper.sample(pos[:, 0], pos[:, 2]) + lift  # elsewhere follow the cross slope
    pos[:, 1] = np.where(np.repeat(deck, 2), pos[:, 1], ys)
    y = yl + lift
    cell = "asphalt_parking" if r.kind == "service" else "asphalt_worn"
    vi, size = mats.cell(cell)
    uv = np.column_stack([(across + r.half_w) / size[0], along / size[1]])
    asphalt.add(pos, uv, strip_tris(len(pts)), MAT_GROUND, vi, (92, 92, 90) if r.kind == "road" else (104, 103, 100))
    for off, col in marking_lines(r):
        mi, mw, period = mats.mark(col)
        mpos, macross, malong = strip_mesh(pts, y, off - mw / 2, off + mw / 2)
        mys = draper.sample(mpos[:, 0], mpos[:, 2]) + LIFT_MARK
        mpos[:, 1] = np.where(np.repeat(deck, 2), mpos[:, 1] + (LIFT_MARK - LIFT_ROAD), mys)
        muv = np.column_stack([np.tile([0.0, 1.0], len(pts)), malong / period])
        marks.add(mpos, muv, strip_tris(len(pts)), MAT_MARKING, mi, MARK_RGB["yellow" if col.startswith("yellow") else "white"])


def _bar_quad(prim: Prim, center: np.ndarray, across: np.ndarray, length: float, depth: float, draper: Draper, lift: float, mark: tuple[int, float, float], color: tuple[int, int, int], u_along_bar: bool) -> None:
    """Rectangle centered at `center`: `length` along `across`, `depth` perpendicular."""
    a = across / max(np.linalg.norm(across), 1e-9)
    b = np.array([-a[1], a[0]])
    c = np.array([center - a * length / 2 - b * depth / 2, center + a * length / 2 - b * depth / 2, center + a * length / 2 + b * depth / 2, center - a * length / 2 + b * depth / 2])
    y = draper.sample(c[:, 0], c[:, 1]) + lift
    pos = np.column_stack([c[:, 0], y, c[:, 1]])
    if u_along_bar:  # u across the painted line (depth), v along it (length)
        uv = np.array([[0, 0], [0, length / mark[2]], [1, length / mark[2]], [1, 0]], dtype=np.float64)
    else:
        uv = np.array([[0, 0], [1, 0], [1, depth / mark[2]], [0, depth / mark[2]]], dtype=np.float64)
    prim.add(pos, uv, np.array([[0, 1, 2], [0, 2, 3]]), MAT_MARKING, mark[0], color)


def _runs(flags: np.ndarray) -> list[tuple[int, int, bool]]:
    """Cyclic runs of equal flags over ring segments: (first segment, n segments, flag)."""
    n = len(flags)
    if n == 0:
        return []
    if flags.all() or not flags.any():
        return [(0, n, bool(flags[0]))]
    start = int(np.argmax(flags != flags[-1]))  # first change after the wrap
    out = []
    i = 0
    while i < n:
        j = i
        f = flags[(start + i) % n]
        while j < n and flags[(start + j) % n] == f:
            j += 1
        out.append(((start + i) % n, j - i, bool(f)))
        i = j
    return out


def _curb_strips(prim: Prim, poly: Polygon, road_tile: Any, draper: Draper, mats: Materials, top_lift: float, base_lift: float) -> None:
    """Curb face + gutter pan (continuous strips) along polygon edges that touch the road
    surface, and a short skirt (down into the ground) along the other edges."""
    ci, csize = mats.cell("curb_gutter")
    si, _ = mats.cell("concrete_sidewalk")
    gutter = float(assumption("streets.gutter_width_m"))
    rings = [poly.exterior, *poly.interiors]
    for ring in rings:
        c = ring_coords(np.asarray(shapely.segmentize(ring, 6.0).coords))
        if len(c) < 3:
            continue
        nxt = np.roll(c, -1, axis=0)
        seg = nxt - c
        ln = np.hypot(seg[:, 0], seg[:, 1])
        keep = ln > 0.05
        c, seg, ln = c[keep], seg[keep], ln[keep]
        if len(c) < 3:
            continue
        out_n = np.column_stack([seg[:, 1], -seg[:, 0]]) / ln[:, None]  # away from the polygon
        mid = c + seg / 2
        probe = mid + out_n * 0.2
        on_road = shapely.contains_xy(road_tile, probe[:, 0], probe[:, 1])
        n = len(c)
        for s0, k, flag in _runs(on_road):
            si_ = (s0 + np.arange(k)) % n  # segments
            vi = (s0 + np.arange(k + 1)) % n  # vertices
            pts = c[vi]
            # vertex normals: average of the adjacent run segments
            vn = np.zeros((k + 1, 2))
            vn[:-1] += out_n[si_]
            vn[1:] += out_n[si_]
            vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
            g = draper.sample(pts[:, 0], pts[:, 1])
            top = g + top_lift
            bot = g + (base_lift if flag else -0.15)
            pos = np.empty((2 * (k + 1), 3))
            pos[0::2] = np.column_stack([pts[:, 0], bot, pts[:, 1]])
            pos[1::2] = np.column_stack([pts[:, 0], top, pts[:, 1]])
            along = np.concatenate([[0.0], np.cumsum(ln[si_])])
            uv = np.empty((2 * (k + 1), 2))
            uv[:, 0] = np.repeat(along / csize[0], 2)
            uv[:, 1] = np.tile([0.6, 0.8], k + 1)
            nrm = np.repeat(np.column_stack([vn[:, 0], np.zeros(k + 1), vn[:, 1]]), 2, axis=0)
            i = np.arange(k)
            a, b_, cc, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
            tri = np.concatenate([np.stack([a, cc, d], 1), np.stack([a, d, b_], 1)])
            tri = fix_winding(pos, tri, nrm[tri[:, 0]])
            prim.add(pos, uv, tri, MAT_GROUND, ci if flag else si, (196, 192, 184), nrm=nrm)
            if flag:
                gp = pts + vn * gutter
                gpos = np.empty((2 * (k + 1), 3))
                gpos[0::2] = np.column_stack([pts[:, 0], g + LIFT_GUTTER, pts[:, 1]])
                gpos[1::2] = np.column_stack([gp[:, 0], draper.sample(gp[:, 0], gp[:, 1]) + LIFT_GUTTER, gp[:, 1]])
                guv = np.empty_like(uv)
                guv[:, 0] = uv[:, 0]
                guv[:, 1] = np.tile([0.6, 0.0], k + 1)
                gt = np.concatenate([np.stack([a, cc, d], 1), np.stack([a, d, b_], 1)])
                prim.add(gpos, guv, gt, MAT_GROUND, ci, (170, 168, 162))


def write_street_tiles(layout: StreetLayout, draper: Draper, grid: TileGrid, assets: Path, mats: Materials) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """roads/roads_{tid}.glb + ground/ground_{tid}.glb for every tile."""
    from shapely.geometry.polygon import orient

    (assets / "roads").mkdir(parents=True, exist_ok=True)
    (assets / "ground").mkdir(parents=True, exist_ok=True)
    curb_h = float(assumption("streets.curb_height_m"))
    walk_lift = LIFT_ROAD + curb_h
    coping = float(assumption("streets.pool_coping_m"))
    out: dict[str, dict[str, Any]] = {}
    tris = {"roads": 0, "ground": 0}
    # assign line features to tiles by cutting them at tile borders
    tile_boxes = {grid.tile_id(r, c): grid.bounds(r, c) for r, c in grid.iter()}
    tb_geom = {t: box(b.min_x, b.min_z, b.max_x, b.max_z) for t, b in tile_boxes.items()}

    def tile_of_pt(x: float, z: float) -> str:
        r, c = grid.tile_of(np.array([x]), np.array([z]))
        return grid.tile_id(int(r[0]), int(c[0]))

    road_parts: dict[str, list[Road]] = {t: [] for t in tile_boxes}
    rtree = shapely.STRtree([r.line for r in layout.roads])
    for t, g in tb_geom.items():
        for ri in rtree.query(g, predicate="intersects"):
            r = layout.roads[int(ri)]
            part = r.line.intersection(g)
            for ln in [part] if part.geom_type == "LineString" else [p for p in getattr(part, "geoms", []) if p.geom_type == "LineString"]:
                if ln.length < 0.3:
                    continue
                road_parts[t].append(Road(**{**r.__dict__, "line": ln}))

    def by_tile_polys(polys: list[Polygon]) -> dict[str, list[Polygon]]:
        res: dict[str, list[Polygon]] = {t: [] for t in tile_boxes}
        for p in polys:
            x0, z0, x1, z1 = p.bounds
            cand = [t for t, b in tile_boxes.items() if not (x1 < b.min_x or x0 > b.max_x or z1 < b.min_z or z0 > b.max_z)]
            if len(cand) == 1:
                res[cand[0]].append(p)
                continue
            for t in cand:
                b = tile_boxes[t]
                for q in polys_of(shapely.clip_by_rect(p, b.min_x, b.min_z, b.max_x, b.max_z)):
                    res[t].append(q)
        return res

    patches_t = by_tile_polys(layout.junction_patches)
    medians_t = by_tile_polys(layout.medians)
    walks_t = by_tile_polys(layout.sidewalks)
    pools_t: dict[str, list[Polygon]] = {t: [] for t in tile_boxes}
    for p in layout.pools:
        pools_t[tile_of_pt(p.centroid.x, p.centroid.y)].append(p)
    dws_t: dict[str, list[tuple[Polygon, np.ndarray]]] = {t: [] for t in tile_boxes}
    for d, n in layout.driveways:
        dws_t[tile_of_pt(d.centroid.x, d.centroid.y)].append((d, n))
    xw_t: dict[str, list[Any]] = {t: [] for t in tile_boxes}
    for x in layout.crosswalks:
        xw_t[tile_of_pt(x[0][0], x[0][1])].append(x)
    sb_t: dict[str, list[Any]] = {t: [] for t in tile_boxes}
    for x in layout.stop_bars:
        sb_t[tile_of_pt(x[0][0], x[0][1])].append(x)
    pi_, psize = mats.cell("asphalt_worn")
    xmark = mats.mark("crosswalk_bar_24in")
    smark = mats.mark("white_bar_12in")
    for t in tile_boxes:
        asphalt = Prim("asphalt")
        marks = Prim("markings")
        for r in road_parts[t]:
            _road_ribbon(asphalt, marks, r, draper, mats)
        for p in patches_t[t]:
            drape_polygon(asphalt, p, draper, LIFT_PATCH, pi_, (90, 90, 88), psize, seg=4.0, cell=6.0)
        for c, across, width, depth in xw_t[t]:
            # continental crosswalk: 0.6 m bars parallel to traffic, 1.2 m on center
            n_bars = max(1, int(width / 1.2))
            a = across / max(np.linalg.norm(across), 1e-9)
            for k in range(n_bars):
                off = (k - (n_bars - 1) / 2.0) * 1.2
                _bar_quad(marks, c + a * off, np.array([-a[1], a[0]]), depth, 0.6, draper, LIFT_XWALK, xmark, MARK_RGB["white"], u_along_bar=True)
        for c, across, length in sb_t[t]:
            _bar_quad(marks, c, across, length, 0.3, draper, LIFT_XWALK, smark, MARK_RGB["white"], u_along_bar=False)
        side = Prim("sidewalks")
        si, ssize = mats.cell("concrete_sidewalk")
        bb = tile_boxes[t]
        road_tile = shapely.clip_by_rect(layout.roads_poly, bb.min_x - 30, bb.min_z - 30, bb.max_x + 30, bb.max_z + 30)
        shapely.prepare(road_tile)
        for p in walks_t[t]:
            p = orient(p.simplify(0.15), 1.0)
            if p.is_empty or p.geom_type != "Polygon":
                continue
            drape_polygon(side, p, draper, walk_lift, si, (200, 196, 188), ssize, seg=6.0)
            _curb_strips(side, p, road_tile, draper, mats, walk_lift, LIFT_ROAD)
        med = Prim("medians")
        mi, msize = mats.cell("coastal_sage")
        for p in medians_t[t]:
            p = orient(p, 1.0)
            drape_polygon(med, p, draper, walk_lift, mi, (118, 128, 92), msize, seg=4.0, cell=12.0)
            _curb_strips(med, p, road_tile, draper, mats, walk_lift, LIFT_ROAD)
        dwy = Prim("driveways")
        di, dsize = mats.cell("concrete_driveway")
        for p, n in dws_t[t]:
            for q in polys_of(p):
                q2 = shapely.segmentize(q, 3.0)
                v, tr = triangulate(q2)
                if not len(tr):
                    continue
                y = draper.sample(v[:, 0], v[:, 1]) + LIFT_DRIVEWAY
                pos = np.column_stack([v[:, 0], y, v[:, 1]])
                tdir = np.array([n[1], -n[0]])
                uv = np.column_stack([(v @ tdir) / dsize[0], (v @ n) / dsize[1]])  # u across, v along
                dwy.add(pos, uv, tr, MAT_GROUND, di, (198, 192, 182))
        pools = Prim("pools")
        wi, wsize = mats.cell("pool_water")
        ci, csize = mats.cell("concrete_plaza")
        for p in pools_t[t]:
            ring = np.asarray(p.exterior.coords)
            y0 = float(draper.sample(ring[:, 0], ring[:, 1]).max()) + 0.05
            drape_polygon(pools, p, draper, 0.0, wi, (70, 150, 190), wsize, y_fixed=y0)
            cop = p.buffer(coping, join_style="mitre").difference(p)
            for q in polys_of(cop):
                drape_polygon(pools, q, draper, 0.0, ci, (214, 208, 196), csize, y_fixed=y0 + 0.12)
        b = tile_boxes[t]
        rp = assets / "roads" / f"roads_{t}.glb"
        gp = assets / "ground" / f"ground_{t}.glb"
        rmeshes = [m for m in (asphalt.mesh(), marks.mesh()) if m is not None]
        gmeshes = [m for m in (side.mesh(), med.mesh(), dwy.mesh(), pools.mesh()) if m is not None]
        nr = write_glb(rp, rmeshes)
        ng = write_glb(gp, gmeshes)
        tris["roads"] += nr
        tris["ground"] += ng
        ys = [m.positions[:, 1] for m in rmeshes + gmeshes]
        out[t] = {
            "roads": f"roads/{rp.name}",
            "ground": f"ground/{gp.name}",
            "triangles": {"roads": nr, "ground": ng},
            "min_y": float(min(y.min() for y in ys)) if ys else None,
            "max_y": float(max(y.max() for y in ys)) if ys else None,
            "bounds": b.as_dict(),
        }
    log(f"streets: roads {tris['roads']:,} tris, ground {tris['ground']:,} tris in {len(out)} tiles; {layout.stats}")
    return out, tris
