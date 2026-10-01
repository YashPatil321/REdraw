"""Roads: OSMnx drive graph -> contract network tables, graphml, geojson, ribbons (spec 5.3).

`build_network` takes an in-memory projected (EPSG:32611) MultiDiGraph with
OSM-style edge tags, so the real OSM graph and the synthetic graph go through
exactly the same code.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from shapely.geometry import LineString

from pipeline.build_terrain import Terrain, jpeg_bytes
from pipeline.common import log, write_json
from pipeline.config import assumption, assumptions
from pipeline.geo import scene_origin, scene_to_latlon, utm_to_lonlat
from pipeline.glb import MeshData, write_glb

DRAPE_MAX_SEGMENT_M = 20.0
RIBBON_LIFT_M = 0.3  # contract: ribbons lifted 0.3 m above terrain
EXIT_SEARCH_RADIUS_M = 1500.0  # geometric search radius for exit turnaround pairing
TURNAROUND_MAX_GAP_M = 400.0  # max distance between opposite carriageway ends to join
LANE_MARK_REPEAT_M = 12.0

FREEWAY_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link"}
DIRECTION_WORDS = {"north", "south", "east", "west", "northbound", "southbound", "eastbound", "westbound"}

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "motorway": (205, 200, 190),
    "trunk": (200, 195, 185),
    "primary": (190, 186, 178),
    "secondary": (180, 177, 170),
    "tertiary": (170, 168, 162),
    "residential": (160, 158, 154),
    "service": (150, 148, 145),
}


# ---------------------------------------------------------------------------
# Tag parsing and class defaults
# ---------------------------------------------------------------------------


def first(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return v[0] if v else None
    return v


def as_str(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v if x is not None and str(x) != "")
    return str(v)


def parse_number(v: Any) -> float | None:
    """First number in an OSM tag value ('3', '2;3', ['2','3'] -> max)."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        vals = [parse_number(x) for x in v]
        vals2 = [x for x in vals if x is not None]
        return max(vals2) if vals2 else None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    nums = re.findall(r"[-+]?\d*\.?\d+", str(v))
    if not nums:
        return None
    return max(float(n) for n in nums)


def parse_maxspeed_kph(v: Any) -> float | None:
    """OSM maxspeed -> kph. Handles '45 mph', '72', lists (max)."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        vals = [parse_maxspeed_kph(x) for x in v]
        vals2 = [x for x in vals if x is not None]
        return max(vals2) if vals2 else None
    s = str(v).strip().lower()
    n = parse_number(s)
    if n is None or n <= 0:
        return None
    if "mph" in s:
        return n * 1.609344
    if "knots" in s:
        return n * 1.852
    return n


def parse_bool(v: Any) -> bool:
    v = first(v)
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"yes", "true", "1", "-1"}


def normalize_class(highway: Any) -> str:
    """OSM highway value -> a class present in assumptions roads.defaults_by_class."""
    hw = as_str(first(highway)) or "unclassified"
    known = assumptions()["roads"]["defaults_by_class"]
    if hw in known:
        return hw
    if hw.endswith("_link") and hw[:-5] in known:
        return hw[:-5]
    return "unclassified"


def class_default(highway: str, key: str) -> float:
    return float(assumption(f"roads.defaults_by_class.{normalize_class(highway)}.{key}"))


def lanes_per_direction(
    highway: str,
    lanes_tag: Any,
    oneway: bool,
    lanes_forward: Any = None,
    lanes_backward: Any = None,
    reversed_: bool = False,
) -> int:
    """Lanes in this edge's direction. OSM `lanes` is the total for two-way roads."""
    directional = parse_number(lanes_backward if reversed_ else lanes_forward)
    if directional is not None and directional >= 1:
        return int(directional)
    total = parse_number(lanes_tag)
    if total is None or total < 1:
        return max(1, int(class_default(highway, "lanes")))
    if oneway:
        return max(1, int(round(total)))
    return max(1, int(math.ceil(total / 2.0)))


def capacity_vph(highway: str, lanes: int) -> float:
    """lanes * capacity_vphpl(class) (spec 5.3.2)."""
    return float(lanes) * class_default(highway, "capacity_vphpl")


def free_flow_seconds(length_m: float, maxspeed_kph: float) -> float:
    return float(length_m) / (float(maxspeed_kph) / 3.6)


def normalize_ref(ref: str) -> list[str]:
    out = []
    for part in re.split(r"[;,]", ref or ""):
        p = part.strip().upper().replace("-", " ")
        p = re.sub(r"\s+", " ", p)
        p = re.sub(r"^(SR|CA|STATE ROUTE)\s+", "CA ", p)
        p = re.sub(r"^(I|INTERSTATE)\s+", "I ", p)
        if p:
            out.append(p)
    return out


def label_for(name: str, ref: str, labels: list[dict[str, Any]]) -> str:
    """Arterial label (region.yaml arterial_labels) by case-insensitive name prefix or ref."""
    lname = (name or "").lower()
    refs = set(normalize_ref(ref))
    # Name matches win over ref matches, so "Ted Williams Parkway" (ref CA 56) keeps its own
    # label even though "Ted Williams Freeway" (also CA 56) is listed first.
    for lab in labels:
        ln = str(lab.get("name", "")).lower()
        if ln and any(part.strip().startswith(ln) for part in lname.split(",")):
            return str(lab["name"])
    for lab in labels:
        lref = lab.get("ref")
        if lref and refs & set(normalize_ref(str(lref))):
            return str(lab["name"])
    return ""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def densify(coords: np.ndarray, max_seg: float = DRAPE_MAX_SEGMENT_M) -> np.ndarray:
    """Insert points so no segment is longer than max_seg. coords (N,2)."""
    coords = np.asarray(coords, dtype=np.float64)
    out = [coords[0]]
    for a, b in zip(coords[:-1], coords[1:], strict=True):
        d = float(np.hypot(*(b - a)))
        n = max(1, int(math.ceil(d / max_seg)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out)


def drape(coords_xz: np.ndarray, terrain: Terrain, lift: float = 0.0, max_seg: float = DRAPE_MAX_SEGMENT_M) -> np.ndarray:
    """Densify a scene polyline (N,2: x,z) and sample terrain -> (M,3) x,y,z."""
    d = densify(coords_xz, max_seg)
    y = terrain.sample(d[:, 0], d[:, 1]) + lift
    return np.column_stack([d[:, 0], y, d[:, 1]])


def utm_coords_to_scene(coords: np.ndarray) -> np.ndarray:
    o = scene_origin()
    c = np.asarray(coords, dtype=np.float64)
    return np.column_stack([c[:, 0] - o.easting, o.northing - c[:, 1]])


# ---------------------------------------------------------------------------
# Network build
# ---------------------------------------------------------------------------


@dataclass
class RoadNetwork:
    G: nx.MultiDiGraph  # projected EPSG:32611, only the retained strongly connected component
    nodes: pd.DataFrame
    edges: pd.DataFrame  # row order == edge_idx
    exits: list[dict[str, Any]]
    signal_source: str = "osm"  # "osm" (tagged / snapped), "inferred" (signal_inference.*) or "none"
    G_visual: nx.MultiDiGraph | None = None  # unclipped projected graph (render-only roads beyond the bbox)
    boundary_nodes: int = 0  # nodes created where roads cross the region bbox edge
    exits_dropped: list[dict[str, Any]] = field(default_factory=list)  # region.yaml exits not reachable


def road_key(data: dict[str, Any]) -> str:
    """Identity of the road an edge belongs to: name, else ref, else way id."""
    name = as_str(data.get("name")).strip().lower()
    if name:
        return "n:" + name.split(",")[0].strip()
    ref = normalize_ref(as_str(data.get("ref")))
    if ref:
        return "r:" + ref[0]
    return "w:" + as_str(first(data.get("osmid")))


def infer_signals(G: nx.MultiDiGraph) -> set[Any]:
    """Signalized junctions for sources without signal data (Overture), per assumptions
    signal_inference.*:

    1. two or more distinct roads (road_key) of `road_classes` meet at the node, or
    2. a freeway ramp (`ramp_classes`) meets an edge of `ramp_arterial_classes` (ramp terminal).

    `_link` classes other than ramps never count as a road (slip lanes).
    """
    road_classes = set(assumption("signal_inference.road_classes"))
    min_roads = int(assumption("signal_inference.min_distinct_roads"))
    ramp_classes = set(assumption("signal_inference.ramp_classes"))
    ramp_arterial = set(assumption("signal_inference.ramp_arterial_classes"))
    out: set[Any] = set()
    for n in G.nodes:
        roads: set[str] = set()
        has_ramp = False
        has_arterial = False
        for _, _, d in list(G.in_edges(n, data=True)) + list(G.out_edges(n, data=True)):
            hw = as_str(first(d.get("highway")))
            if hw in ramp_classes:
                has_ramp = True
            if hw in ramp_arterial:
                has_arterial = True
            if hw in road_classes:
                roads.add(road_key(d))
        if len(roads) >= min_roads or (has_ramp and has_arterial):
            out.add(n)
    return out


def exit_road_key(label: str) -> str:
    """Road name of an exit label: 'Camino del Sur south (to SR 56 west)' -> 'Camino del Sur'."""
    words = re.sub(r"\(.*?\)", " ", label).split()
    while words and words[-1].lower() in DIRECTION_WORDS:
        words = words[:-1]
    return " ".join(words)


def exit_matcher(exit_cfg: dict[str, Any], labels: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """(name prefix to match, refs to match) for an exit's road."""
    key = exit_road_key(str(exit_cfg.get("label", exit_cfg["id"])))
    key_refs = normalize_ref(key)
    for lab in labels:
        lrefs = normalize_ref(str(lab.get("ref", "") or ""))
        if str(lab.get("name", "")).lower() == key.lower() or (key_refs and set(key_refs) & set(lrefs)):
            return str(lab["name"]).lower(), lrefs
    return key.lower(), key_refs


def edge_matches(data: dict[str, Any], name_prefix: str, refs: list[str]) -> bool:
    name = as_str(data.get("name")).lower()
    if name_prefix and any(p.strip().startswith(name_prefix) for p in name.split(",")):
        return True
    erefs = set(normalize_ref(as_str(data.get("ref"))))
    return bool(refs and erefs & set(refs))


EXIT_SPLIT_SEARCH_M = 300.0  # matching road edges within this distance of an exit point are split there
EXIT_SPLIT_MIN_GAIN_M = 30.0  # split only when the nearest matching node is this much farther than the road
EXIT_NODE_ID_BASE = 9_000_000_000  # synthetic node ids for exit split points (never OSM ids)


def split_edges_at_exits(G: nx.MultiDiGraph, exits_xy: list[tuple[dict[str, Any], float, float]], labels: list[dict[str, Any]]) -> int:
    """Insert a node where each exit's road crosses the exit point (UTM graph, in place).

    Simplified freeways have no nodes between interchanges, so the nearest existing node of
    the exit road can be far from the bbox edge. The nearest carriageway (and, for a divided
    road, the nearest opposite-direction carriageway) is split at the point closest to the
    exit; the two split nodes are joined both ways by short virtual edges so the exit node can
    absorb outbound and emit inbound trips. Returns the number of nodes added.
    """
    from shapely.geometry import Point
    from shapely.ops import substring

    o = scene_origin()
    added = 0
    for ei, (ex, x, z) in enumerate(exits_xy):
        pt = Point(x + o.easting, o.northing - z)
        prefix, refs = exit_matcher(ex, labels)
        groups: dict[tuple[Any, Any], list[tuple[Any, Any, Any, dict[str, Any]]]] = {}
        node_d = math.inf
        for u, v, k, d in G.edges(keys=True, data=True):
            if not edge_matches(d, prefix, refs):
                continue
            for n in (u, v):
                node_d = min(node_d, math.hypot(G.nodes[n]["x"] - pt.x, G.nodes[n]["y"] - pt.y))
            groups.setdefault((min(u, v), max(u, v)), []).append((u, v, k, d))

        def geom_of(u: Any, v: Any, d: dict[str, Any]) -> LineString:
            g = d.get("geometry")
            if g is None:
                return LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
            c = list(g.coords)
            if math.hypot(c[0][0] - G.nodes[u]["x"], c[0][1] - G.nodes[u]["y"]) > math.hypot(c[-1][0] - G.nodes[u]["x"], c[-1][1] - G.nodes[u]["y"]):
                c = c[::-1]
            return LineString(c)

        scored = []
        for key, es in groups.items():
            u, v, _, d = es[0]
            g = geom_of(u, v, d)
            dist = g.distance(pt)
            if dist <= EXIT_SPLIT_SEARCH_M:
                scored.append((dist, key, es))
        scored.sort(key=lambda t: (t[0], str(t[1])))
        if not scored or node_d - scored[0][0] < EXIT_SPLIT_MIN_GAIN_M:
            continue
        chosen = [scored[0]]
        first_two_way = len(scored[0][2]) > 1
        if not first_two_way:
            # divided road: also split the nearest carriageway running the other way
            u0, v0, _, d0 = scored[0][2][0]
            g0 = geom_of(u0, v0, d0)
            t0 = g0.interpolate(g0.project(pt))
            h0 = g0.interpolate(min(g0.length, g0.project(pt) + 5.0))
            dir0 = (h0.x - t0.x, h0.y - t0.y)
            for cand in scored[1:]:
                u1, v1, _, d1 = cand[2][0]
                g1 = geom_of(u1, v1, d1)
                t1 = g1.interpolate(g1.project(pt))
                h1 = g1.interpolate(min(g1.length, g1.project(pt) + 5.0))
                if (h1.x - t1.x) * dir0[0] + (h1.y - t1.y) * dir0[1] < 0:
                    chosen.append(cand)
                    break
        new_nodes = []
        for j, (_, _, es) in enumerate(chosen):
            u, v, _, d = es[0]
            g = geom_of(u, v, d)
            at = g.project(pt)
            if at < 5.0 or at > g.length - 5.0:
                new_nodes.append(u if at < 5.0 else v)  # already (almost) at a node
                continue
            nid = EXIT_NODE_ID_BASE + 10 * ei + j
            sp = g.interpolate(at)
            G.add_node(nid, x=sp.x, y=sp.y, street_count=2, exit_split=str(ex["id"]))
            for uu, vv, kk, dd in es:
                gg = geom_of(uu, vv, dd)
                a = gg.project(sp)
                L = max(float(dd.get("length", gg.length)), 1.0)
                for s_, e_, part in ((uu, nid, substring(gg, 0, a)), (nid, vv, substring(gg, a, gg.length))):
                    nd = dict(dd)
                    nd["geometry"] = part
                    nd["length"] = max(L * part.length / max(gg.length, 1e-6), 0.5)
                    G.add_edge(s_, e_, **nd)
                G.remove_edge(uu, vv, kk)
            new_nodes.append(nid)
            added += 1
        if len(new_nodes) == 2 and new_nodes[0] != new_nodes[1]:
            a, b = new_nodes
            gap = math.hypot(G.nodes[a]["x"] - G.nodes[b]["x"], G.nodes[a]["y"] - G.nodes[b]["y"])
            tmpl = chosen[0][2][0][3]
            for s_, e_ in ((a, b), (b, a)):
                G.add_edge(
                    s_,
                    e_,
                    highway=as_str(first(tmpl.get("highway"))) or "motorway_link",
                    name="",
                    ref=as_str(tmpl.get("ref")),
                    lanes="1",
                    oneway=True,
                    osmid="virtual_exit_turnaround",
                    length=max(gap, 1.0),
                    geometry=LineString([(G.nodes[s_]["x"], G.nodes[s_]["y"]), (G.nodes[e_]["x"], G.nodes[e_]["y"])]),
                )
    return added


ISLAND_CONNECT_MAX_M = 300.0  # max gap bridged by a virtual connector to a disconnected road island


def connect_islands(G: nx.MultiDiGraph, max_gap: float = ISLAND_CONNECT_MAX_M) -> int:
    """Join road pieces outside the largest strongly connected component to it (UTM graph, in
    place) with short two-way virtual connectors (osmid 'virtual_connector').

    Overture's drive network leaves gated communities as islands (their private gate roads are
    not drivable for the public) and has a few one-way stubs; without connectors their homes
    would snap to a node kilometers away. Each remaining piece (largest first) is linked from its
    node closest to the main component, if within `max_gap`. Returns connectors added.
    """
    from scipy.spatial import cKDTree

    added = 0
    tried: set[frozenset[Any]] = set()
    while True:
        sccs = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
        if len(sccs) <= 1:
            return added
        main = sccs[0]
        pieces = [c for c in sccs[1:] if frozenset(c) not in tried]
        if not pieces:
            return added
        piece = pieces[0]
        tried.add(frozenset(piece))
        mids = list(main)
        tree = cKDTree(np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in mids]))
        pn = list(piece)
        d, k = tree.query(np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in pn]), k=1)
        i = int(np.argmin(d))
        if float(d[i]) > max_gap:
            continue
        a, b = pn[i], mids[int(k[i])]
        for s_, e_, rev in ((a, b, False), (b, a, True)):
            G.add_edge(
                s_,
                e_,
                highway="residential",
                name="",
                ref="",
                oneway=False,
                reversed=rev,
                osmid="virtual_connector",
                length=max(float(d[i]), 1.0),
                geometry=LineString([(G.nodes[s_]["x"], G.nodes[s_]["y"]), (G.nodes[e_]["x"], G.nodes[e_]["y"])]),
            )
        added += 1


def bridge_exit_islands(G: nx.MultiDiGraph, G_full: nx.MultiDiGraph, exits_xy: list[tuple[dict[str, Any], float, float]], labels: list[dict[str, Any]]) -> int:
    """Reconnect exit roads whose in-bbox piece only connects to the rest of the network outside
    the bbox (UTM graphs, G in place). For each such exit the shortest real road paths in the
    unclipped graph, main component -> piece and piece -> main, are added back. Returns edges added."""
    o = scene_origin()
    added = 0
    for ex, x, z in exits_xy:
        e, n = x + o.easting, o.northing - z
        prefix, refs = exit_matcher(ex, labels)
        cand = [nd for u, v, d in G.edges(data=True) if edge_matches(d, prefix, refs) for nd in (u, v)]
        if not cand:
            continue
        near = min(cand, key=lambda q: math.hypot(G.nodes[q]["x"] - e, G.nodes[q]["y"] - n))
        sccs = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
        main = sccs[0]
        if near in main:
            continue
        piece = next(c for c in sccs if near in c)
        piece_full = [q for q in piece if q in G_full]
        main_full = [q for q in main if q in G_full]
        if not piece_full or not main_full:
            continue

        def w(a: Any, b: Any, d: dict[Any, dict[str, Any]]) -> float:
            return min(float(dd.get("length", 1.0)) for dd in d.values())

        paths = []
        for src, dst, Gd in ((main_full, piece_full, G_full), (piece_full, main_full, G_full.reverse(copy=False))):
            try:
                _, path = nx.multi_source_dijkstra(Gd, set(src), weight=w, target=None)  # type: ignore[call-overload]
            except nx.NetworkXNoPath:
                continue
            best = min((q for q in dst if q in path), key=lambda q: len(path[q]), default=None)
            if best is None:
                continue
            pth = path[best]
            paths.append(pth if Gd is G_full else pth[::-1])
        for pth in paths:
            for a, b in zip(pth[:-1], pth[1:], strict=True):
                for q in (a, b):
                    if q not in G:
                        G.add_node(q, **G_full.nodes[q])
                if not G.has_edge(a, b):
                    for _, dd in G_full.get_edge_data(a, b).items():
                        G.add_edge(a, b, **dd)
                        added += 1
        n_e = sum(len(p) - 1 for p in paths)
        if n_e:
            log(f"exit {ex['id']}: in-bbox road piece reconnected through {n_e} real edge(s) just outside the bbox")
        else:
            log(f"exit {ex['id']}: its in-bbox road piece is not connected to the network anywhere in the raw data")
    return added


def add_exit_turnarounds(G: nx.MultiDiGraph, exits_xy: list[tuple[dict[str, Any], float, float]], labels: list[dict[str, Any]]) -> int:
    """Join dead-end carriageway ends near each exit (bbox truncation of divided roads).

    A divided freeway cut at the bbox edge leaves a sink (outbound carriageway)
    and a source (inbound carriageway). We join each sink to the nearest source
    on the same road with a short virtual edge so the exit node can both absorb
    and emit trips and the graph stays strongly connected. Works on a UTM graph.
    """
    o = scene_origin()
    added = 0
    for ex, x, z in exits_xy:
        e, n = x + o.easting, o.northing - z
        prefix, refs = exit_matcher(ex, labels)

        def on_road(node: Any, incoming: bool, prefix: str = prefix, refs: list[str] = refs) -> dict[str, Any] | None:
            it = G.in_edges(node, keys=True, data=True) if incoming else G.out_edges(node, keys=True, data=True)
            for _, _, _, d in it:
                if edge_matches(d, prefix, refs):
                    return d
            return None

        sinks, sources = [], []
        for node, nd in G.nodes(data=True):
            dist = math.hypot(nd["x"] - e, nd["y"] - n)
            if dist > EXIT_SEARCH_RADIUS_M:
                continue
            if G.out_degree(node) == 0 and on_road(node, True) is not None:
                sinks.append(node)
            elif G.in_degree(node) == 0 and on_road(node, False) is not None:
                sources.append(node)
        for s in sinks:
            if not sources:
                break
            sx, sy = G.nodes[s]["x"], G.nodes[s]["y"]
            best = min(sources, key=lambda t: math.hypot(G.nodes[t]["x"] - sx, G.nodes[t]["y"] - sy))
            gap = math.hypot(G.nodes[best]["x"] - sx, G.nodes[best]["y"] - sy)
            if gap > TURNAROUND_MAX_GAP_M:
                continue
            tmpl = on_road(s, True) or {}
            G.add_edge(
                s,
                best,
                highway=as_str(first(tmpl.get("highway"))) or "motorway_link",
                name="",
                ref=as_str(tmpl.get("ref")),
                lanes=str(lanes_per_direction(as_str(first(tmpl.get("highway"))), tmpl.get("lanes"), True)),
                oneway=True,
                osmid="virtual_turnaround",
                length=max(gap, 1.0),
                geometry=LineString([(sx, sy), (G.nodes[best]["x"], G.nodes[best]["y"])]),
            )
            added += 1
    return added


BOUNDARY_NODE_ID_BASE = 9_100_000_000  # synthetic node ids where roads cross the region bbox edge


def region_ring_utm(samples: int = 50) -> Any:
    """The WGS84 region bbox as a densified polygon in EPSG:32611 (edges bulge slightly in UTM)."""
    from shapely.geometry import Polygon

    from pipeline.config import region
    from pipeline.geo import lonlat_to_utm

    b = region()["bbox"]
    n = samples
    ring_ll = (
        [(b["west"] + (b["east"] - b["west"]) * i / n, b["south"]) for i in range(n)]
        + [(b["east"], b["south"] + (b["north"] - b["south"]) * i / n) for i in range(n)]
        + [(b["east"] - (b["east"] - b["west"]) * i / n, b["north"]) for i in range(n)]
        + [(b["west"], b["north"] - (b["north"] - b["south"]) * i / n) for i in range(n)]
    )
    return Polygon([lonlat_to_utm(lo, la) for lo, la in ring_ll])


def clip_graph_to_region(G: nx.MultiDiGraph, poly: Any | None = None) -> int:
    """Cut every edge that crosses the region bbox edge at the crossing (UTM graph, in place) and
    drop everything outside. The raw data may cover a larger area than region.yaml; the sim
    network is exactly the region. Crossing points become nodes (ids from BOUNDARY_NODE_ID_BASE,
    attribute `boundary=True`), shared by both directions of a two-way road, so exits snap to the
    true bbox-edge crossing. Returns the number of boundary nodes created."""
    import shapely
    from shapely.geometry import Point

    poly = poly if poly is not None else region_ring_utm()
    inside_poly = poly.buffer(0.5)
    shapely.prepare(inside_poly)
    node_in = {n: bool(inside_poly.contains(Point(d["x"], d["y"]))) for n, d in G.nodes(data=True)}
    created: dict[tuple[int, int], Any] = {}
    next_id = BOUNDARY_NODE_ID_BASE

    def node_at(x: float, y: float) -> Any:
        nonlocal next_id
        key = (int(round(x * 10)), int(round(y * 10)))
        if key not in created:
            G.add_node(next_id, x=float(x), y=float(y), street_count=1, boundary=True)
            node_in[next_id] = True
            created[key] = next_id
            next_id += 1
        return created[key]

    todo = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if node_in[u] and node_in[v]:
            continue
        g = d.get("geometry") or LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
        if not node_in[u] and not node_in[v] and not g.intersects(poly):
            continue
        todo.append((u, v, k, d, g))
    for u, v, k, d, g in todo:
        coords = list(g.coords)
        pu = (G.nodes[u]["x"], G.nodes[u]["y"])
        if math.hypot(coords[0][0] - pu[0], coords[0][1] - pu[1]) > math.hypot(coords[-1][0] - pu[0], coords[-1][1] - pu[1]):
            g = LineString(coords[::-1])
        inter = g.intersection(poly)
        parts = [p for p in ([inter] if inter.geom_type == "LineString" else list(getattr(inter, "geoms", []))) if p.geom_type == "LineString" and p.length > 0.5]
        L = max(float(d.get("length", g.length)), 1.0)
        G.remove_edge(u, v, k)
        for part in parts:
            c = list(part.coords)
            a = u if node_in[u] and math.hypot(c[0][0] - pu[0], c[0][1] - pu[1]) < 0.5 else node_at(*c[0])
            pv = (G.nodes[v]["x"], G.nodes[v]["y"])
            b = v if node_in[v] and math.hypot(c[-1][0] - pv[0], c[-1][1] - pv[1]) < 0.5 else node_at(*c[-1])
            if a == b:
                continue
            nd = dict(d)
            nd["geometry"] = part
            nd["length"] = max(L * part.length / max(g.length, 1e-6), 0.5)
            G.add_edge(a, b, **nd)
    G.remove_nodes_from([n for n, ok in node_in.items() if not ok and n in G])
    return len(created)


def build_network(
    G: nx.MultiDiGraph,
    terrain: Terrain,
    region_cfg: dict[str, Any],
    signal_points_utm: np.ndarray | None = None,
) -> RoadNetwork:
    """Projected OSM-style drive graph -> contract tables (strongly connected)."""
    if str(G.graph.get("crs", "")).upper().replace("EPSG:", "") != "32611":
        raise ValueError(f"build_network expects an EPSG:32611 graph, got crs={G.graph.get('crs')}")
    G_visual = G.copy()
    G = G.copy()
    labels = list(region_cfg.get("arterial_labels", []))
    o = scene_origin()
    n_boundary = clip_graph_to_region(G)
    log(f"roads: clipped to the region bbox ({n_boundary} bbox-edge crossings, {G.number_of_nodes():,} nodes inside)")

    # Exit turnarounds (before taking the strongly connected component).
    from pipeline.geo import latlon_to_scene

    exits_xy = []
    config_xz: dict[str, tuple[float, float]] = {}
    for ex in region_cfg.get("exits", []):
        x, z = latlon_to_scene(float(ex["lat"]), float(ex["lon"]))
        config_xz[str(ex["id"])] = (x, z)
        tx, tz = exit_target(G, ex, x, z, labels)
        if math.hypot(tx - x, tz - z) > 1.0:
            log(f"exit {ex['id']}: snap target {math.hypot(tx - x, tz - z):.0f} m from the region.yaml point (road / bbox edge crossing)")
        exits_xy.append((ex, tx, tz))
    n_split = split_edges_at_exits(G, exits_xy, labels)
    if n_split:
        log(f"roads: split {n_split} carriageway(s) at exit points")
    n_turn = add_exit_turnarounds(G, exits_xy, labels)

    n_conn = connect_islands(G)
    bridge_exit_islands(G, G_visual, exits_xy, labels)
    before = G.number_of_nodes()
    largest = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest).copy()
    log(f"roads: kept largest strongly connected component {G.number_of_nodes():,}/{before:,} nodes ({n_turn} exit turnarounds, {n_conn} island connectors added)")

    # Signals
    node_ids = np.array(list(G.nodes), dtype=np.int64)
    nx_ = np.array([G.nodes[n]["x"] for n in node_ids], dtype=np.float64)
    ny_ = np.array([G.nodes[n]["y"] for n in node_ids], dtype=np.float64)
    signal = np.array([as_str(first(G.nodes[n].get("highway"))) == "traffic_signals" for n in node_ids])
    if signal_points_utm is not None and len(signal_points_utm):
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(signal_points_utm)[:, :2])
        d, _ = tree.query(np.column_stack([nx_, ny_]), k=1)
        signal |= d <= float(assumption("pipeline.signal_snap_radius_m"))
    signal_source = "osm" if signal.any() else "none"
    if not signal.any() and bool(assumption("signal_inference.enabled_when_source_has_none")):
        inferred = infer_signals(G)
        signal = np.array([n in inferred for n in node_ids])
        signal_source = "inferred"
        log(f"roads: source has no traffic signals; inferred {int(signal.sum()):,} signalized junctions (assumptions signal_inference.*)")
    # Only junction nodes (degree > 2 undirected) carry a signal in the model.
    und = G.to_undirected(as_view=True)
    junction = np.array([und.degree(n) >= 3 for n in node_ids])
    signal &= junction

    # Edges in deterministic order
    edge_keys = sorted(G.edges(keys=True), key=lambda t: (int(t[0]), int(t[1]), int(t[2])))
    rows: list[dict[str, Any]] = []
    geoms: list[np.ndarray] = []
    for idx, (u, v, k) in enumerate(edge_keys):
        d = G.edges[u, v, k]
        hw_raw = as_str(first(d.get("highway"))) or "unclassified"
        oneway = parse_bool(d.get("oneway"))
        lanes = lanes_per_direction(
            hw_raw, d.get("lanes"), oneway, d.get("lanes:forward"), d.get("lanes:backward"), parse_bool(d.get("reversed"))
        )
        kph = parse_maxspeed_kph(d.get("maxspeed")) or class_default(hw_raw, "maxspeed_kph")
        geom = d.get("geometry")
        if geom is None:
            coords = np.array([[G.nodes[u]["x"], G.nodes[u]["y"]], [G.nodes[v]["x"], G.nodes[v]["y"]]])
        else:
            coords = np.asarray(geom.coords)[:, :2]
            # make sure geometry runs u -> v
            pu = np.array([G.nodes[u]["x"], G.nodes[u]["y"]])
            if np.hypot(*(coords[0] - pu)) > np.hypot(*(coords[-1] - pu)):
                coords = coords[::-1]
        length = float(np.sum(np.hypot(*np.diff(coords, axis=0).T))) if len(coords) > 1 else 0.0
        length = max(length, 1.0)
        sc = utm_coords_to_scene(coords)
        xyz = drape(sc, terrain).astype(np.float32)
        name = as_str(d.get("name"))
        ref = as_str(d.get("ref"))
        d["edge_idx"] = idx
        rows.append(
            {
                "edge_idx": idx,
                "u": int(u),
                "v": int(v),
                "osmid": as_str(d.get("osmid")),
                "name": name,
                "ref": ref,
                "highway": hw_raw,
                "lanes": lanes,
                "maxspeed_kph": kph,
                "oneway": oneway,
                "length_m": length,
                "capacity_vph": capacity_vph(hw_raw, lanes),
                "free_flow_s": free_flow_seconds(length, kph),
                "label": label_for(name, ref, labels),
            }
        )
        geoms.append(xyz.reshape(-1))

    edges = pd.DataFrame(rows)
    edges = edges.astype(
        {
            "edge_idx": "int32",
            "u": "int64",
            "v": "int64",
            "lanes": "int16",
            "maxspeed_kph": "float32",
            "oneway": "bool",
            "length_m": "float32",
            "capacity_vph": "float32",
            "free_flow_s": "float32",
        }
    )
    edges["geometry"] = geoms
    edges = edges[
        ["edge_idx", "u", "v", "osmid", "name", "ref", "highway", "lanes", "maxspeed_kph", "oneway", "length_m", "capacity_vph", "free_flow_s", "geometry", "label"]
    ]

    xs = nx_ - o.easting
    zs = o.northing - ny_
    ys = terrain.sample(xs, zs)
    lonlat = [utm_to_lonlat(e, n) for e, n in zip(nx_, ny_, strict=True)]
    nodes = pd.DataFrame(
        {
            "node_id": node_ids.astype(np.int64),
            "x": xs.astype(np.float64),
            "z": zs.astype(np.float64),
            "y": ys.astype(np.float32),
            "lat": np.array([ll[1] for ll in lonlat], dtype=np.float64),
            "lon": np.array([ll[0] for ll in lonlat], dtype=np.float64),
            "signalized": signal.astype(bool),
            "boundary_exit": [""] * len(node_ids),
        }
    )
    dropped: list[dict[str, Any]] = []
    exits = snap_exits(nodes, edges, exits_xy, labels, config_xz, dropped)
    if not exits:
        raise RuntimeError("no region.yaml exit could be snapped to the drive network; check region.yaml exits")
    exits += route_unreachable_exits(exits, dropped, config_xz)
    for ex in exits:
        if not ex.get("via"):
            nodes.loc[nodes["node_id"] == ex["node_id"], "boundary_exit"] = ex["id"]
    return RoadNetwork(G=G, nodes=nodes, edges=edges, exits=exits, signal_source=signal_source, G_visual=G_visual, boundary_nodes=n_boundary, exits_dropped=dropped)


EXIT_ON_ROAD_TOL_M = 50.0  # a region.yaml exit point this close to its road is used as is


def exit_target(G: nx.MultiDiGraph, ex: dict[str, Any], x: float, z: float, labels: list[dict[str, Any]]) -> tuple[float, float]:
    """Scene snap target for an exit: the nearest point of its road when the region.yaml point
    lies on the road (within EXIT_ON_ROAD_TOL_M); otherwise the point where the road crosses the
    WGS84 region bbox edge nearest the configured point (within EXIT_SEARCH_RADIUS_M); otherwise
    the configured point."""
    from shapely.geometry import Point
    from shapely.ops import nearest_points

    o = scene_origin()
    ring = region_ring_utm().exterior
    pt = Point(x + o.easting, o.northing - z)
    prefix, refs = exit_matcher(ex, labels)
    best, bd = None, EXIT_SEARCH_RADIUS_M
    on_road, od = None, EXIT_ON_ROAD_TOL_M
    for u, v, d in G.edges(data=True):
        if not edge_matches(d, prefix, refs):
            continue
        g = d.get("geometry") or LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
        gd = g.distance(pt)
        if gd <= od:
            on_road, od = nearest_points(g, pt)[0], gd
        if gd > bd:
            continue
        inter = g.intersection(ring)
        pts = [inter] if inter.geom_type == "Point" else list(getattr(inter, "geoms", []))
        for q in pts:
            if q.geom_type == "Point" and q.distance(pt) < bd:
                best, bd = q, q.distance(pt)
    target = on_road if on_road is not None else best
    if target is None:
        return x, z
    return target.x - o.easting, o.northing - target.y


def snap_exits(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    exits_xy: list[tuple[dict[str, Any], float, float]],
    labels: list[dict[str, Any]],
    config_xz: dict[str, tuple[float, float]] | None = None,
    dropped: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Snap each region.yaml exit to the nearest node of its road (spec 5.4.3).

    `exits_xy` holds the snap targets (road / bbox edge crossings); `snap_distance_m` is the
    distance from the chosen node to the region.yaml point (`config_xz`), so a wrong config
    coordinate stays visible."""
    out: list[dict[str, Any]] = []
    pos = nodes.set_index("node_id")[["x", "z"]]
    used: set[int] = set()
    dropped = dropped if dropped is not None else []
    for ex, x, z in exits_xy:
        prefix, refs = exit_matcher(ex, labels)
        mask = edges.apply(lambda r, p=prefix, rf=refs: edge_matches({"name": r["name"], "ref": r["ref"]}, p, rf), axis=1)
        cand = set(edges.loc[mask, "u"]).union(edges.loc[mask, "v"]) - used
        if not cand:
            log(f"WARNING exit {ex['id']}: no edge of road '{prefix}' {refs} is in the strongly connected network inside the bbox")
            dropped.append(
                {
                    "id": ex["id"],
                    "label": ex.get("label", ex["id"]),
                    "bearing_deg": float(ex.get("bearing_deg", 0)),
                    "verified": bool(ex.get("verified", False)),
                    "reason": f"road '{prefix}' is not connected to the drive network inside the region bbox",
                }
            )
            continue
        cp = pos.loc[sorted(cand)]
        d = np.hypot(cp["x"].to_numpy() - x, cp["z"].to_numpy() - z)
        i = int(np.argmin(d))
        nid = int(cp.index[i])
        used.add(nid)
        out.append(
            {
                "id": ex["id"],
                "label": ex.get("label", ex["id"]),
                "node_id": nid,
                "x": float(cp.iloc[i]["x"]),
                "z": float(cp.iloc[i]["z"]),
                "bearing_deg": float(ex.get("bearing_deg", 0)),
                "snap_distance_m": float(
                    np.hypot(float(cp.iloc[i]["x"]) - (config_xz or {}).get(str(ex["id"]), (x, z))[0], float(cp.iloc[i]["z"]) - (config_xz or {}).get(str(ex["id"]), (x, z))[1])
                ),
                "target_distance_m": float(d[i]),
                "verified": bool(ex.get("verified", False)),
            }
        )
    return out


def route_unreachable_exits(
    exits: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    config_xz: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    """Keep exits whose own road only clips a bbox corner (no junction with the network inside
    the bbox, e.g. Del Dios Highway at the NW corner) as `via` exits: trips leave the network at
    the nearest snapped exit (by region.yaml point) and continue outside the bbox. They keep
    their own id, label and bearing (so external jobs in that direction still exist) and share
    the via exit's node. Moves them out of `dropped` (in place); returns the new exit records."""
    out: list[dict[str, Any]] = []
    if not exits:
        return out
    keep: list[dict[str, Any]] = []
    for dr in dropped:
        xz = config_xz.get(str(dr["id"]))
        if xz is None:
            keep.append(dr)
            continue
        dist = [math.hypot(e["x"] - xz[0], e["z"] - xz[1]) for e in exits]
        k = int(np.argmin(dist))
        via = exits[k]
        log(f"exit {dr['id']}: its road is not connected inside the bbox; routed via exit {via['id']} ({dist[k]:.0f} m away)")
        out.append(
            {
                **{f: via[f] for f in ("node_id", "x", "z")},
                "id": dr["id"],
                "label": dr["label"],
                "bearing_deg": float(dr.get("bearing_deg", 0.0)),
                "snap_distance_m": float(dist[k]),
                "target_distance_m": float(dist[k]),
                "verified": bool(dr.get("verified", False)),
                "via": via["id"],
                "snap_method": "via_exit",
                "note": f"{dr['reason']}; trips use exit {via['id']} and continue outside the bbox",
            }
        )
    dropped[:] = keep
    return out


def local_node_mask(nodes: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    """True for nodes touched by at least one non-freeway edge (where homes/schools/jobs may snap)."""
    loc = edges[~edges["highway"].isin(FREEWAY_CLASSES)]
    ids = set(loc["u"]).union(loc["v"])
    return nodes["node_id"].isin(ids).to_numpy()


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def edges_table(edges: pd.DataFrame) -> pa.Table:
    schema = pa.schema(
        [
            ("edge_idx", pa.int32()),
            ("u", pa.int64()),
            ("v", pa.int64()),
            ("osmid", pa.string()),
            ("name", pa.string()),
            ("ref", pa.string()),
            ("highway", pa.string()),
            ("lanes", pa.int16()),
            ("maxspeed_kph", pa.float32()),
            ("oneway", pa.bool_()),
            ("length_m", pa.float32()),
            ("capacity_vph", pa.float32()),
            ("free_flow_s", pa.float32()),
            ("geometry", pa.list_(pa.float32())),
            ("label", pa.string()),
        ]
    )
    arrays = []
    for f in schema:
        col = edges[f.name]
        if f.name == "geometry":
            arrays.append(pa.array([np.asarray(g, dtype=np.float32) for g in col], type=f.type))
        else:
            arrays.append(pa.array(col.to_numpy(), type=f.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def write_network(net: RoadNetwork, processed: Path) -> None:
    import osmnx as ox

    processed.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(net.nodes, preserve_index=False), processed / "network_nodes.parquet")
    pq.write_table(edges_table(net.edges), processed / "network_edges.parquet")
    write_json(processed / "exits.json", {"exits": net.exits, "dropped": net.exits_dropped})

    # graphml (OSMnx MultiDiGraph, projected). Stringify list attrs for portability.
    G = net.G.copy()
    for _, _, d in G.edges(data=True):
        for key in list(d):
            if isinstance(d[key], (list, tuple)):
                d[key] = as_str(d[key])
    for _, d in G.nodes(data=True):
        for key in list(d):
            if isinstance(d[key], (list, tuple)):
                d[key] = as_str(d[key])
    ox.io.save_graphml(G, processed / "roads_drive.graphml")

    # roads.geojson (WGS84)
    feats = []
    for r, g in zip(net.edges.itertuples(index=False), net.edges["geometry"], strict=True):
        xyz = np.asarray(g, dtype=np.float64).reshape(-1, 3)
        coords = [list(reversed(scene_to_latlon(float(p[0]), float(p[2])))) for p in xyz[:: max(1, len(xyz) // 50)]]
        last = list(reversed(scene_to_latlon(float(xyz[-1, 0]), float(xyz[-1, 2]))))
        if coords[-1] != last:
            coords.append(last)
        props = {k: (v.item() if hasattr(v, "item") else v) for k, v in r._asdict().items() if k != "geometry"}
        feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[round(c[0], 7), round(c[1], 7)] for c in coords]}, "properties": props})
    (processed / "roads.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")


def lane_texture() -> bytes:
    """Asphalt with solid edge lines and a dashed center line (u across, v along, v repeats)."""
    w, h = 64, 128
    img = np.full((h, w, 3), 78, dtype=np.uint8)
    rng = np.random.default_rng(7)
    img = np.clip(img.astype(int) + rng.integers(-6, 7, size=(h, w, 1)), 0, 255).astype(np.uint8)
    img[:, 2:4] = (225, 225, 220)
    img[:, w - 4 : w - 2] = (225, 225, 220)
    img[: h // 2, w // 2 - 1 : w // 2 + 1] = (230, 200, 90)
    return jpeg_bytes(img, quality=90)


def ribbon_mesh(xyz_center: np.ndarray, width: float, terrain: Terrain, lift: float, color: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
    """Ribbon along a polyline (M,3). Returns pos, nrm, uv, col, idx (local indices)."""
    p = xyz_center[:, [0, 2]].astype(np.float64)
    if len(p) < 2:
        return tuple(np.zeros((0, k)) for k in (3, 3, 2, 4)) + (np.zeros(0, dtype=np.uint32),)
    seg = np.diff(p, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    seg_len[seg_len == 0] = 1e-6
    t = seg / seg_len[:, None]
    tan = np.zeros_like(p)
    tan[0] = t[0]
    tan[-1] = t[-1]
    if len(p) > 2:
        tan[1:-1] = t[:-1] + t[1:]
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    nrm2 = np.column_stack([-tan[:, 1], tan[:, 0]])  # left normal in (x,z)
    half = width / 2.0
    left = p + nrm2 * half
    right = p - nrm2 * half
    yl = terrain.sample(left[:, 0], left[:, 1])
    yr = terrain.sample(right[:, 0], right[:, 1])
    yc = terrain.sample(p[:, 0], p[:, 1])
    yl = np.maximum(yl, yc) + lift
    yr = np.maximum(yr, yc) + lift
    n = len(p)
    pos = np.empty((2 * n, 3))
    pos[0::2] = np.column_stack([left[:, 0], yl, left[:, 1]])
    pos[1::2] = np.column_stack([right[:, 0], yr, right[:, 1]])
    s = np.concatenate([[0.0], np.cumsum(seg_len)]) / LANE_MARK_REPEAT_M
    uv = np.empty((2 * n, 2))
    uv[0::2] = np.column_stack([np.zeros(n), s])
    uv[1::2] = np.column_stack([np.ones(n), s])
    nrm = np.tile([0.0, 1.0, 0.0], (2 * n, 1))
    col = np.tile([*color, 255], (2 * n, 1))
    i = np.arange(n - 1)
    a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
    # upward facing: winding (a, b, c) where a=left, b=right, c=next left
    tri = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    # choose winding so the normal points up
    v0 = pos[tri[:, 0]]
    v1 = pos[tri[:, 1]]
    v2 = pos[tri[:, 2]]
    ny = (v1[:, 2] - v0[:, 2]) * (v2[:, 0] - v0[:, 0]) - (v1[:, 0] - v0[:, 0]) * (v2[:, 2] - v0[:, 2])
    flip = ny < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return pos, nrm, uv, col, tri.reshape(-1).astype(np.uint32)


def build_road_ribbons(edges: pd.DataFrame, terrain: Terrain, out_path: Path) -> int:
    """One roads.glb with all ribbons; two-way pairs rendered once at combined width."""
    lane_w = float(assumption("roads.lane_width_m"))
    seen: dict[tuple[int, int, int], int] = {}
    pos_l, nrm_l, uv_l, col_l, idx_l = [], [], [], [], []
    base = 0
    lanes_by_dir = {(int(r.u), int(r.v), int(round(float(r.length_m)))): int(r.lanes) for r in edges.itertuples()}
    for r, g in zip(edges.itertuples(), edges["geometry"], strict=True):
        u, v = int(r.u), int(r.v)
        key = (min(u, v), max(u, v), int(round(float(r.length_m))))
        if not r.oneway and key in seen:
            continue
        seen[key] = 1
        lanes = int(r.lanes)
        if not r.oneway:
            lanes += lanes_by_dir.get((v, u, int(round(float(r.length_m)))), 0)
        width = max(lanes, 1) * lane_w
        cls = normalize_class(r.highway)
        color = CLASS_COLORS.get(cls, CLASS_COLORS["residential"])
        xyz = np.asarray(g, dtype=np.float64).reshape(-1, 3)
        pos, nrm, uv, col, idx = ribbon_mesh(xyz, width, terrain, RIBBON_LIFT_M, color)
        if len(idx) == 0:
            continue
        pos_l.append(pos)
        nrm_l.append(nrm)
        uv_l.append(uv)
        col_l.append(col)
        idx_l.append(idx + base)
        base += len(pos)
    mesh = MeshData(
        name="roads",
        positions=np.concatenate(pos_l).astype(np.float32),
        normals=np.concatenate(nrm_l).astype(np.float32),
        uvs=np.concatenate(uv_l).astype(np.float32),
        colors=np.concatenate(col_l).astype(np.uint8),
        indices=np.concatenate(idx_l).astype(np.uint32),
        texture_jpeg=lane_texture(),
        texture_repeat=True,
        roughness=0.95,
    )
    tris = write_glb(out_path, [mesh])
    log(f"roads: ribbons {tris:,} triangles")
    return tris


def lane_texture_preview(path: Path) -> None:  # pragma: no cover - debug helper
    Image.open(__import__("io").BytesIO(lane_texture())).save(path)


# ---------------------------------------------------------------------------
# Real mode loader
# ---------------------------------------------------------------------------


def load_osm_drive(raw_graphml: Path) -> tuple[nx.MultiDiGraph, np.ndarray]:
    """Unsimplified OSM drive graph (WGS84) -> (simplified projected graph, signal points UTM)."""
    import osmnx as ox

    G = ox.io.load_graphml(raw_graphml)
    sig = [(d["x"], d["y"]) for _, d in G.nodes(data=True) if as_str(first(d.get("highway"))) == "traffic_signals"]
    if not G.graph.get("simplified", False):
        n_add = explode_edge_geometry(G)
        if n_add:
            log(f"roads: {n_add:,} shape vertices of the raw edge geometry kept through simplification")
        G = ox.simplify_graph(G)
    Gp = ox.project_graph(G, to_crs="EPSG:32611")
    if sig:
        from pipeline.geo import lonlat_to_utm

        sig_utm = np.array([lonlat_to_utm(float(x), float(y)) for x, y in sig])
    else:
        sig_utm = np.zeros((0, 2))
    return Gp, sig_utm


SHAPE_NODE_ID_BASE = 8_000_000_000  # temporary ids of interior shape vertices (removed by simplify)


def explode_edge_geometry(G: nx.MultiDiGraph) -> int:
    """Turn the interior vertices of every edge `geometry` into degree-2 graph nodes (in place).

    Raw graphs built from Overture (pipeline/fetch_aws.py) have nodes only at connectors and keep
    the road shape in each edge's `geometry`; osmnx.simplify_graph rebuilds merged geometry from
    node coordinates only, which would turn every curve into straight chords. Both directions of
    a two-way way share the same shape nodes, so simplify removes them again and the merged edge
    follows the true shape. Edge `length` is split by the geodesic length of each piece."""
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    shape_ids: dict[tuple[Any, ...], int] = {}
    nxt = SHAPE_NODE_ID_BASE
    add_edges: list[tuple[Any, Any, dict[str, Any]]] = []
    remove: list[tuple[Any, Any, Any]] = []
    new_nodes: list[tuple[int, float, float]] = []
    for u, v, k, d in list(G.edges(keys=True, data=True)):
        g = d.get("geometry")
        if g is None:
            continue
        c = np.asarray(g.coords)[:, :2]
        if len(c) < 3:
            continue
        pu = np.array([G.nodes[u]["x"], G.nodes[u]["y"]])
        if np.hypot(*(c[0] - pu)) > np.hypot(*(c[-1] - pu)):
            c = c[::-1]
        inner = c[1:-1]
        keep = np.r_[True, np.any(np.abs(np.diff(inner, axis=0)) > 1e-9, axis=1)] if len(inner) else np.zeros(0, bool)
        inner = inner[keep]
        # drop interior vertices that coincide with the end nodes
        pv = np.array([G.nodes[v]["x"], G.nodes[v]["y"]])
        inner = inner[(np.abs(inner - pu).max(axis=1) > 1e-9) & (np.abs(inner - pv).max(axis=1) > 1e-9)]
        if not len(inner):
            continue
        way = (min(str(u), str(v)), max(str(u), str(v)), as_str(d.get("osmid")))
        ids = []
        for x, y in inner:
            key = (*way, round(float(x), 7), round(float(y), 7))
            nid = shape_ids.get(key)
            if nid is None:
                nid = nxt
                nxt += 1
                shape_ids[key] = nid
                new_nodes.append((nid, float(x), float(y)))
            ids.append(nid)
        chain = [u, *ids, v]
        pts = np.vstack([pu, inner, pv])
        seg = np.array([geod.inv(pts[i, 0], pts[i, 1], pts[i + 1, 0], pts[i + 1, 1])[2] for i in range(len(pts) - 1)])
        total = float(d.get("length") or seg.sum())
        frac = seg / max(seg.sum(), 1e-9)
        base = {kk: vv for kk, vv in d.items() if kk not in ("geometry", "length")}
        for i in range(len(chain) - 1):
            add_edges.append((chain[i], chain[i + 1], {**base, "length": total * float(frac[i])}))
        remove.append((u, v, k))
    for nid, x, y in new_nodes:
        G.add_node(nid, x=x, y=y, street_count=2)
    G.remove_edges_from(remove)
    for a, b, d in add_edges:
        G.add_edge(a, b, **d)
    return len(shape_ids)

