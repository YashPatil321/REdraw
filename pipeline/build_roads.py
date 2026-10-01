"""Roads: OSMnx drive graph -> contract network tables, graphml, geojson, ribbons (spec 5.3).

`build_network` takes an in-memory projected (EPSG:32611) MultiDiGraph with
OSM-style edge tags, so the real OSM graph and the synthetic graph go through
exactly the same code.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
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
    for lab in labels:
        ln = str(lab.get("name", "")).lower()
        if ln and any(part.strip().startswith(ln) for part in lname.split(",")):
            return str(lab["name"])
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


def exit_road_key(label: str) -> str:
    words = label.split()
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

        def on_road(node: Any, incoming: bool) -> dict[str, Any] | None:
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


def build_network(
    G: nx.MultiDiGraph,
    terrain: Terrain,
    region_cfg: dict[str, Any],
    signal_points_utm: np.ndarray | None = None,
) -> RoadNetwork:
    """Projected OSM-style drive graph -> contract tables (strongly connected)."""
    if str(G.graph.get("crs", "")).upper().replace("EPSG:", "") != "32611":
        raise ValueError(f"build_network expects an EPSG:32611 graph, got crs={G.graph.get('crs')}")
    G = G.copy()
    labels = list(region_cfg.get("arterial_labels", []))
    o = scene_origin()

    # Exit turnarounds (before taking the strongly connected component).
    from pipeline.geo import latlon_to_scene

    exits_xy = []
    for ex in region_cfg.get("exits", []):
        x, z = latlon_to_scene(float(ex["lat"]), float(ex["lon"]))
        exits_xy.append((ex, x, z))
    n_turn = add_exit_turnarounds(G, exits_xy, labels)

    before = G.number_of_nodes()
    largest = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest).copy()
    log(f"roads: kept largest strongly connected component {G.number_of_nodes():,}/{before:,} nodes ({n_turn} exit turnarounds added)")

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
    exits = snap_exits(nodes, edges, exits_xy, labels)
    for ex in exits:
        nodes.loc[nodes["node_id"] == ex["node_id"], "boundary_exit"] = ex["id"]
    return RoadNetwork(G=G, nodes=nodes, edges=edges, exits=exits)


def snap_exits(
    nodes: pd.DataFrame, edges: pd.DataFrame, exits_xy: list[tuple[dict[str, Any], float, float]], labels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Snap each region.yaml exit to the nearest node of its road (spec 5.4.3)."""
    out = []
    pos = nodes.set_index("node_id")[["x", "z"]]
    used: set[int] = set()
    for ex, x, z in exits_xy:
        prefix, refs = exit_matcher(ex, labels)
        mask = edges.apply(lambda r, p=prefix, rf=refs: edge_matches({"name": r["name"], "ref": r["ref"]}, p, rf), axis=1)
        cand = set(edges.loc[mask, "u"]).union(edges.loc[mask, "v"]) - used
        if not cand:
            log(f"WARNING exit {ex['id']}: no edges match road '{prefix}' {refs}; snapping to nearest node")
            cand = set(nodes["node_id"]) - used
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
                "snap_distance_m": float(d[i]),
                "verified": bool(ex.get("verified", False)),
            }
        )
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
    write_json(processed / "exits.json", {"exits": net.exits})

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
        G = ox.simplify_graph(G)
    Gp = ox.project_graph(G, to_crs="EPSG:32611")
    if sig:
        from pipeline.geo import lonlat_to_utm

        sig_utm = np.array([lonlat_to_utm(float(x), float(y)) for x, y in sig])
    else:
        sig_utm = np.zeros((0, 2))
    return Gp, sig_utm
