"""Plans and tools (spec 7).

A plan is a list of tool instances. Tools are data (``data/config/tools.yaml``)
and each MVP tool id has three functions here: validate/resolve (snap map
inputs), cost (from ``assumptions.costs``) and apply (modify a WorldState copy).

``check_plan`` never raises for player mistakes; it returns a ``PlanCheck``
with errors. ``apply_plan`` returns a modified clone of the world.

Custom tool (``custom``): the player's description goes to the custom tool
preview (``POST /tools/custom/preview``), which returns an LLM estimate the
player confirms. The confirmed ``params.estimate`` is re-validated here (the
client could have edited it) and translated into levers the sim understands::

    {"summary": "...",
     "levers": [{"type": "mode_utility_shift", "mode": "bike", "applies_to": "students",
                 "school": "del_norte_hs" | null, "utils": 0.4},
                {"type": "capacity_change", "target": "edge", "edge_idx": 12, "factor": 1.1},
                {"type": "capacity_change", "target": "entrance", "entrance": "del_norte_hs/main", "factor": 1.2}],
     "adoption_range": [0.05, 0.15], "cost_upfront_usd": 0, "cost_per_year_usd": 50000,
     "assumptions": ["..."]}

Mode utility shifts apply to the targeted people; ``adoption_range[1]`` caps the
share of them who switch into a boosted mode (like the carpool program cap).
Edge factors scale road capacity; entrance factors scale the drop-off service
rate. A plain description without an estimate is rejected with a clear error.
Custom tools are always flagged "LLM estimated". The older internal lever
format (``params.levers``: ``mode_utility_shifts`` / ``capacity_changes``) is
still accepted.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pipeline.config import load_yaml
from pipeline.geo import latlon_to_scene
from sim.assumptions import A, Af, clock_to_s
from sim.schools import service_rate
from sim.world import STUDENT_MODES, WORKER_MODES, Entrance, Shuttle, WorldState


class ToolInstance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    mission: str = "morning_crunch"
    title: str = ""
    pitch: str = ""
    author_id: str | None = None
    tools: list[ToolInstance] = Field(default_factory=list)
    created_at: str | None = None
    report: dict[str, Any] | None = None


class CostLine(BaseModel):
    tool: str
    upfront_usd: float = 0.0
    per_year_usd: float = 0.0
    note: str = ""


class PlanCheck(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    cost_upfront_usd: float = 0.0
    cost_per_year_usd: float = 0.0
    over_budget: bool = False
    budget_upfront_usd: float = 0.0
    budget_per_year_usd: float = 0.0
    constraint_violations: list[str] = Field(default_factory=list)
    resolved_tools: list[dict[str, Any]] = Field(default_factory=list)
    # additions beyond docs/sim_interface.md
    cost_lines: list[CostLine] = Field(default_factory=list)
    llm_estimated_tools: list[str] = Field(default_factory=list)


class ToolError(ValueError):
    pass


@lru_cache(maxsize=1)
def tool_defs() -> dict[str, dict[str, Any]]:
    return {t["id"]: t for t in load_yaml("tools.yaml")["tools"]}


@lru_cache(maxsize=8)
def mission_def(mission_id: str = "morning_crunch") -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9_]+", mission_id):
        raise ToolError(f"bad mission id {mission_id!r}")
    return load_yaml(f"missions/{mission_id}.yaml")


# ---------------------------------------------------------------------------
# snapping
# ---------------------------------------------------------------------------

@dataclass
class _Segments:
    x0: np.ndarray
    z0: np.ndarray
    x1: np.ndarray
    z1: np.ndarray
    edge: np.ndarray


_SEG_CACHE: dict[int, _Segments] = {}


def _segments(world: WorldState) -> _Segments:
    net = world.net
    key = id(net)
    if key not in _SEG_CACHE:
        offs = net.geom_offsets
        vals = net.geom_values
        n_pts = (offs[1:] - offs[:-1]) // 3
        x0s, z0s, x1s, z1s, es = [], [], [], [], []
        good = n_pts >= 2
        if good.any():
            pts = vals.reshape(-1, 3) if len(vals) % 3 == 0 else None
            if pts is not None:
                start = offs[:-1] // 3
                edge_of_pt = np.repeat(np.arange(net.n_edges), n_pts)
                idx = np.arange(len(pts))
                last_of_edge = np.zeros(len(pts), bool)
                ends = (start + n_pts - 1)[n_pts > 0]
                last_of_edge[ends] = True
                seg = idx[~last_of_edge & (idx < len(pts) - 1)]
                x0s.append(pts[seg, 0])
                z0s.append(pts[seg, 2])
                x1s.append(pts[seg + 1, 0])
                z1s.append(pts[seg + 1, 2])
                es.append(edge_of_pt[seg])
        missing = np.setdiff1d(np.arange(net.n_edges), np.concatenate(es) if es else np.zeros(0, int))
        if len(missing):
            x0s.append(net.x[net.eu[missing]])
            z0s.append(net.z[net.eu[missing]])
            x1s.append(net.x[net.ev[missing]])
            z1s.append(net.z[net.ev[missing]])
            es.append(missing)
        _SEG_CACHE[key] = _Segments(*(np.concatenate(a).astype(np.float64) for a in (x0s, z0s, x1s, z1s)), np.concatenate(es))
    return _SEG_CACHE[key]


def to_scene(lat: float, lon: float) -> tuple[float, float]:
    return latlon_to_scene(float(lat), float(lon))


def snap_edge(world: WorldState, x: float, z: float, exclude_freeway: bool = True) -> tuple[int, float]:
    """Nearest directed edge to scene point (x, z). Of a two-way pair, picks the one whose v is closer."""
    s = _segments(world)
    dx, dz = s.x1 - s.x0, s.z1 - s.z0
    L2 = np.maximum(dx * dx + dz * dz, 1e-9)
    t = np.clip(((x - s.x0) * dx + (z - s.z0) * dz) / L2, 0, 1)
    d = np.hypot(s.x0 + t * dx - x, s.z0 + t * dz - z)
    if exclude_freeway:
        net = world.net
        fwy = np.isin(net.highway, ["motorway", "motorway_link", "trunk", "trunk_link"])
        d = np.where(fwy[s.edge], np.inf, d)
    best = int(np.argmin(d))
    dist = float(d[best])
    e = int(s.edge[best])
    net = world.net
    tie = np.nonzero(np.abs(d - dist) < 0.5)[0]
    cands = np.unique(s.edge[tie])
    if len(cands) > 1:
        dv = np.hypot(net.x[net.ev[cands]] - x, net.z[net.ev[cands]] - z)
        e = int(cands[int(np.argmin(dv))])
    return e, dist


def snap_node(world: WorldState, x: float, z: float) -> tuple[int, float]:
    d, i = world.net.kdtree.query([x, z])
    return int(i), float(d)


def _parse_latlon(v: Any, what: str) -> tuple[float, float]:
    if isinstance(v, dict) and "lat" in v and "lon" in v:
        v = [v["lat"], v["lon"]]
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise ToolError(f"{what} must be [lat, lon]")
    try:
        lat, lon = float(v[0]), float(v[1])
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{what} must be numbers [lat, lon]") from exc
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ToolError(f"{what} is not a valid lat/lon")
    return lat, lon


def _path_edges(world: WorldState, nodes: list[int]) -> tuple[list[int], float]:
    """Edges along shortest-length paths through consecutive snapped nodes. Returns (edges, km)."""
    from sim.routing import trees

    net = world.net
    cost, best = net.graph.best_edges(net.length_m)
    fwd, _ = net.graph.matrices(cost)
    edges: list[int] = []
    km = 0.0
    for a, b in zip(nodes[:-1], nodes[1:], strict=True):
        if a == b:
            continue
        dist, pred = trees(fwd, np.array([a]))
        if not np.isfinite(dist[0, b]):
            km += float(np.hypot(net.x[a] - net.x[b], net.z[a] - net.z[b])) / 1000.0
            continue
        cur = b
        seg = []
        while cur != a:
            pv = int(pred[0, cur])
            pidx = net.graph.pair_index(np.array([pv]), np.array([cur]))[0]
            seg.append(int(best[pidx]))
            cur = pv
        seg.reverse()
        edges.extend(seg)
        km += float(dist[0, b]) / 1000.0
    return edges, km


def _point_polyline_dist(px: np.ndarray, pz: np.ndarray, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    d = np.full(len(px), np.inf)
    for i in range(len(xs) - 1):
        x0, z0, x1, z1 = xs[i], zs[i], xs[i + 1], zs[i + 1]
        dx, dz = x1 - x0, z1 - z0
        L2 = max(dx * dx + dz * dz, 1e-9)
        t = np.clip(((px - x0) * dx + (pz - z0) * dz) / L2, 0, 1)
        d = np.minimum(d, np.hypot(x0 + t * dx - px, z0 + t * dz - pz))
    if len(xs) == 1:
        d = np.hypot(px - xs[0], pz - zs[0])
    return d


# ---------------------------------------------------------------------------
# generic param validation from tools.yaml
# ---------------------------------------------------------------------------

def _validate_params(world: WorldState, tdef: dict[str, Any], params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {}
    warnings: list[str] = []
    specs = {p["id"]: p for p in tdef.get("params", [])}
    for k in params:
        if k not in specs and not (tdef["id"] == "custom" and k in ("levers", "estimate")):
            warnings.append(f"unknown parameter '{k}' ignored")
    for pid, spec in specs.items():
        v = params.get(pid)
        if v is None or (isinstance(v, str) and v == "" and spec["type"] not in ("text",)):
            if "default" in spec:
                v = spec["default"]
            elif spec.get("required"):
                raise ToolError(f"missing required parameter '{pid}'")
            else:
                out[pid] = None
                continue
        out[pid] = _coerce(world, pid, spec, v)
    if tdef["id"] == "custom":
        for k in ("levers", "estimate"):
            if k in params:
                out[k] = params[k]
    return out, warnings


def _coerce(world: WorldState, pid: str, spec: dict[str, Any], v: Any) -> Any:
    t = spec["type"]
    if t == "school":
        if spec.get("allow_all") and str(v).lower() in ("all", "*"):
            return "all"
        if world.school_index(str(v)) < 0:
            raise ToolError(f"unknown school '{v}'")
        return str(v)
    if t == "entrance":
        if world.entrance_index(str(v)) < 0:
            raise ToolError(f"unknown entrance '{v}' (use '<school_id>/<entrance_id>')")
        return str(v)
    if t == "time":
        try:
            s = clock_to_s(str(v))
        except (ValueError, TypeError) as exc:
            raise ToolError(f"'{pid}' must be a time HH:MM") from exc
        if "min" in spec and s < clock_to_s(spec["min"]):
            raise ToolError(f"'{pid}' {v} is earlier than {spec['min']}")
        if "max" in spec and s > clock_to_s(spec["max"]):
            raise ToolError(f"'{pid}' {v} is later than {spec['max']}")
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"
    if t in ("int", "float"):
        try:
            num = float(v)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"'{pid}' must be a number") from exc
        if not math.isfinite(num):
            raise ToolError(f"'{pid}' must be finite")
        if t == "int":
            if abs(num - round(num)) > 1e-9:
                raise ToolError(f"'{pid}' must be a whole number")
            num = int(round(num))
        if "min" in spec and num < spec["min"]:
            raise ToolError(f"'{pid}' must be >= {spec['min']}")
        if "max" in spec and num > spec["max"]:
            raise ToolError(f"'{pid}' must be <= {spec['max']}")
        return num
    if t == "select":
        opts = [o["value"] for o in spec.get("options", [])]
        if v not in opts:
            raise ToolError(f"'{pid}' must be one of {opts}")
        return v
    if t == "bool":
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.lower() in ("true", "false"):
            return v.lower() == "true"
        raise ToolError(f"'{pid}' must be true or false")
    if t == "text":
        s = str(v)
        if len(s) > int(spec.get("max_length", 10_000)):
            raise ToolError(f"'{pid}' is longer than {spec['max_length']} characters")
        return s
    if t == "point":
        return list(_parse_latlon(v, pid))
    if t in ("points", "polyline"):
        if not isinstance(v, (list, tuple)) or not v:
            raise ToolError(f"'{pid}' must be a list of [lat, lon]")
        pts = [list(_parse_latlon(p, f"{pid}[{i}]")) for i, p in enumerate(v)]
        if t == "points" and len(pts) > int(spec.get("max_count", 10_000)):
            raise ToolError(f"'{pid}' has more than {spec['max_count']} points")
        if t == "polyline" and len(pts) < 2:
            raise ToolError(f"'{pid}' needs at least 2 points")
        return pts
    if t == "edge":
        if isinstance(v, (list, tuple, dict)):
            return {"latlon": list(_parse_latlon(v, pid))}
        try:
            e = int(v)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"'{pid}' must be an edge index") from exc
        if not (0 <= e < world.net.n_edges):
            raise ToolError(f"edge {e} does not exist")
        return e
    if t == "node":
        if isinstance(v, (list, tuple, dict)):
            return {"latlon": list(_parse_latlon(v, pid))}
        try:
            nid = int(v)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"'{pid}' must be a node id") from exc
        if world.net.node_index(np.array([nid]))[0] < 0:
            raise ToolError(f"node {nid} does not exist")
        return nid
    raise ToolError(f"unsupported parameter type '{t}'")


# ---------------------------------------------------------------------------
# tools: resolve -> cost -> apply
# ---------------------------------------------------------------------------

def _snap_point_edge(world: WorldState, latlon: list[float], what: str) -> dict[str, Any]:
    x, z = to_scene(*latlon)
    e, d = snap_edge(world, x, z)
    if d > Af("sim_engine.snap_max_distance_m"):
        raise ToolError(f"{what} is {d:.0f} m from the nearest road")
    net = world.net
    return {"edge_idx": e, "node_id": int(net.node_ids[net.ev[e]]), "x": x, "z": z, "snap_m": round(d, 1)}


def _snap_points_nodes(world: WorldState, pts: list[list[float]], what: str) -> list[dict[str, Any]]:
    out = []
    for i, ll in enumerate(pts):
        x, z = to_scene(*ll)
        e, d = snap_edge(world, x, z)
        if d > Af("sim_engine.snap_max_distance_m"):
            raise ToolError(f"{what}[{i}] is {d:.0f} m from the nearest road")
        net = world.net
        cand = [int(net.eu[e]), int(net.ev[e])]
        dn = [math.hypot(net.x[c] - x, net.z[c] - z) for c in cand]
        n = cand[int(np.argmin(dn))]
        out.append({"node": n, "node_id": int(net.node_ids[n]), "edge_idx": e, "x": x, "z": z, "snap_m": round(d, 1)})
    return out


def _resolve(world: WorldState, tool: str, p: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    net = world.net
    r: dict[str, Any] = {}
    if tool == "bell_time":
        s = world.schools[world.school_index(p["school"])]
        r = {"school_idx": world.school_index(p["school"]), "old": s.bell_clock, "new": p["start"]}
    elif tool == "school_shuttle":
        stops = _snap_points_nodes(world, p["stops"], "stops")
        r = {"school_idx": world.school_index(p["school"]), "stops": stops}
    elif tool == "carpool_program":
        r = {"school_idx": world.school_index(p["school"]) if p.get("school") not in (None, "all") else -1}
    elif tool == "dropoff_redesign":
        r = {"entrance_idx": world.entrance_index(p["entrance"])}
        if p["extra_curb_spots"] == 0 and not p.get("faster_unload"):
            warnings.append("no change: 0 extra curb spots and no faster unload program")
    elif tool == "new_dropoff_entrance":
        r = _snap_point_edge(world, p["location"], "location")
        r["school_idx"] = world.school_index(p["school"])
        s = world.schools[r["school_idx"]]
        dist = math.hypot(r["x"] - s.x, r["z"] - s.z)
        r["dist_to_school_m"] = round(dist, 1)
        if dist > Af("sim_engine.snap_max_distance_m"):
            warnings.append(f"new entrance is {dist:.0f} m from {s.name}")
    elif tool == "signal_timing":
        node = p["node"]
        if isinstance(node, dict):
            x, z = to_scene(*node["latlon"])
            sig = np.nonzero(net.signalized)[0]
            if len(sig):
                dd = np.hypot(net.x[sig] - x, net.z[sig] - z)
                ni = int(sig[int(np.argmin(dd))])
                d = float(dd.min())
            else:
                ni, d = snap_node(world, x, z)
            if d > Af("sim_engine.snap_max_distance_m"):
                raise ToolError(f"no intersection within {Af('sim_engine.snap_max_distance_m'):.0f} m of the clicked point")
        else:
            ni = int(net.node_index(np.array([int(node)]))[0])
        if not net.signalized[ni]:
            warnings.append(f"node {int(net.node_ids[ni])} is not signalized; retiming only scales its stop/yield delay")
        r = {"node": ni, "node_id": int(net.node_ids[ni]), "x": float(net.x[ni]), "z": float(net.z[ni])}
    elif tool == "turn_lane":
        e = p["edge"]
        if isinstance(e, dict):
            e = _snap_point_edge(world, e["latlon"], "edge")["edge_idx"]
        through = list(A("roads.through_lane_classes"))
        if net.highway[e] in through:
            warnings.append(f"turn_lane on edge {e} is a through lane class ({net.highway[e]}); modeled as a turn pocket, not a through lane")
        r = {"edge_idx": int(e), "name": str(net.name[e]), "highway": str(net.highway[e])}
    elif tool in ("bike_route", "safe_walk_route"):
        pts = _snap_points_nodes(world, p["path"], "path")
        edges, km = _path_edges(world, [s["node"] for s in pts])
        xs = np.array([to_scene(*ll)[0] for ll in p["path"]])
        zs = np.array([to_scene(*ll)[1] for ll in p["path"]])
        r = {"edges": edges, "km": round(km, 3), "xs": xs.tolist(), "zs": zs.tolist()}
        if tool == "safe_walk_route":
            r["school_idx"] = world.school_index(p["school"]) if p.get("school") else -1
    elif tool == "teen_drive_policy":
        s_i = world.school_index(p["school"])
        if world.schools[s_i].grades[1] < 9:
            warnings.append(f"{world.schools[s_i].name} has no high school grades; the cap has no effect")
        r = {"school_idx": s_i}
    elif tool == "custom":
        if p.get("estimate") is not None:
            r = {"levers": _levers_from_estimate(world, p["estimate"])}
        else:
            r = {"levers": _validate_levers(world, p.get("levers"))}
    else:
        raise ToolError("no implementation for this tool")
    return r


class _Levers(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode_utility_shifts: dict[str, float] = Field(default_factory=dict)
    capacity_changes: list[dict[str, float]] = Field(default_factory=list)
    cost_upfront_usd: float = 0.0
    cost_per_year_usd: float = 0.0
    adoption_range: list[float] | None = None
    assumptions: list[str] = Field(default_factory=list)
    # translated from a confirmed preview estimate (see module docstring)
    targeted_shifts: list[dict[str, Any]] = Field(default_factory=list)
    entrance_factors: list[dict[str, float]] = Field(default_factory=list)
    summary: str = ""


class _EstShift(BaseModel):
    type: Literal["mode_utility_shift"]
    mode: str
    applies_to: Literal["students", "workers", "all"] = "all"
    school: str | None = None
    utils: float


class _EstCapacity(BaseModel):
    type: Literal["capacity_change"]
    target: Literal["edge", "entrance"]
    edge_idx: int | None = None
    entrance: str | None = None
    factor: float


class _Estimate(BaseModel):
    summary: str = Field(default="", max_length=400)
    levers: list[Annotated[_EstShift | _EstCapacity, Field(discriminator="type")]] = Field(min_length=1, max_length=8)
    adoption_range: tuple[float, float]
    cost_upfront_usd: float = Field(ge=0)
    cost_per_year_usd: float = Field(ge=0)
    assumptions: list[str] = Field(default_factory=list, max_length=10)


def _levers_from_estimate(world: WorldState, est: Any) -> dict[str, Any]:
    """Re-validate a confirmed custom tool estimate and translate it into sim levers."""
    try:
        e = _Estimate.model_validate(est)
    except ValidationError as exc:
        err = exc.errors()[0]
        raise ToolError(f"invalid estimate: {'.'.join(str(x) for x in err['loc'])}: {err['msg']}") from exc
    lo, hi = e.adoption_range
    if not (0.0 <= lo <= hi <= 1.0):
        raise ToolError("invalid estimate: adoption_range must be [low, high] with 0 <= low <= high <= 1")
    mx = Af("sim_engine.custom_max_utility_shift")
    cmin, cmax = Af("sim_engine.custom_min_capacity_factor"), Af("sim_engine.custom_max_capacity_factor")
    shifts: list[dict[str, Any]] = []
    caps: list[dict[str, float]] = []
    ents: list[dict[str, float]] = []
    for lv in e.levers:
        if isinstance(lv, _EstShift):
            ok_s = lv.mode in STUDENT_MODES and lv.applies_to in ("students", "all")
            ok_w = lv.mode in WORKER_MODES and lv.applies_to in ("workers", "all")
            if not (ok_s or ok_w):
                raise ToolError(f"invalid estimate: mode '{lv.mode}' is not a choice for {lv.applies_to}")
            if abs(lv.utils) > mx:
                raise ToolError(f"invalid estimate: utility shift for {lv.mode} exceeds {mx}")
            s_i = -1
            if lv.school:
                s_i = world.school_index(lv.school)
                if s_i < 0:
                    raise ToolError(f"invalid estimate: unknown school '{lv.school}'")
            shifts.append({"mode": lv.mode, "applies_to": lv.applies_to, "school_idx": s_i, "utils": float(lv.utils)})
        elif not (cmin <= lv.factor <= cmax):
            raise ToolError(f"invalid estimate: capacity factor {lv.factor} outside [{cmin}, {cmax}]")
        elif lv.target == "edge":
            if lv.edge_idx is None or not (0 <= lv.edge_idx < world.net.n_edges):
                raise ToolError(f"invalid estimate: edge {lv.edge_idx} does not exist")
            caps.append({"edge": float(lv.edge_idx), "factor": float(lv.factor)})
        else:
            ei = world.entrance_index(lv.entrance or "")
            if ei < 0:
                raise ToolError(f"invalid estimate: unknown entrance '{lv.entrance}'")
            ents.append({"entrance_idx": float(ei), "factor": float(lv.factor)})
    return _Levers(targeted_shifts=shifts, capacity_changes=caps, entrance_factors=ents,
                   cost_upfront_usd=e.cost_upfront_usd, cost_per_year_usd=e.cost_per_year_usd,
                   adoption_range=[lo, hi], assumptions=e.assumptions, summary=e.summary).model_dump()


def _validate_levers(world: WorldState, levers: Any) -> dict[str, Any]:
    if levers is None:
        raise ToolError("estimate this idea with the AI first (Preview), then confirm the estimate before running")
    try:
        lv = _Levers.model_validate(levers)
    except ValidationError as exc:
        raise ToolError(f"invalid levers: {exc.errors()[0]['msg']}") from exc
    if lv.targeted_shifts or lv.entrance_factors:
        raise ToolError("invalid levers: targeted levers come only from a confirmed estimate")
    mx = Af("sim_engine.custom_max_utility_shift")
    modes = set(STUDENT_MODES) | set(WORKER_MODES)
    for m, v in lv.mode_utility_shifts.items():
        if m not in modes:
            raise ToolError(f"invalid levers: unknown mode '{m}'")
        if abs(v) > mx:
            raise ToolError(f"invalid levers: utility shift for {m} exceeds {mx}")
    cmax = Af("sim_engine.custom_max_capacity_factor")
    for c in lv.capacity_changes:
        e, f = int(c.get("edge", -1)), float(c.get("factor", 1.0))
        if not (0 <= e < world.net.n_edges):
            raise ToolError(f"invalid levers: edge {e} does not exist")
        if not (0 < f <= cmax):
            raise ToolError(f"invalid levers: capacity factor {f} outside (0, {cmax}]")
    if lv.cost_upfront_usd < 0 or lv.cost_per_year_usd < 0:
        raise ToolError("invalid levers: costs must be >= 0")
    return lv.model_dump()


def _cost(world: WorldState, tool: str, p: dict[str, Any], r: dict[str, Any]) -> CostLine:
    c = CostLine(tool=tool)
    if tool == "bell_time":
        c.note = "No direct cost. Moving bell times can affect bus contracts (not modeled in v1)."
    elif tool == "school_shuttle":
        c.per_year_usd = p["buses"] * Af("costs.school_shuttle_per_bus_year")
        c.note = f"{p['buses']} buses x ${Af('costs.school_shuttle_per_bus_year'):,.0f}/yr"
    elif tool == "carpool_program":
        c.per_year_usd = Af(f"costs.carpool_program_per_year.{p['incentive']}")
        c.note = f"{p['incentive']} incentive program"
    elif tool == "dropoff_redesign":
        c.upfront_usd = p["extra_curb_spots"] * Af("costs.dropoff_curb_spot_upfront")
        if p.get("faster_unload"):
            c.per_year_usd = Af("costs.dropoff_faster_unload_per_year")
        c.note = f"{p['extra_curb_spots']} curb spots" + (" + faster unload program" if p.get("faster_unload") else "")
    elif tool == "new_dropoff_entrance":
        c.upfront_usd = Af("costs.new_dropoff_entrance_upfront")
        c.note = "new driveway, curb cut and loop"
    elif tool == "signal_timing":
        c.upfront_usd = Af("costs.signal_timing_upfront")
        c.note = "retiming study for one intersection"
    elif tool == "turn_lane":
        c.upfront_usd = Af("costs.turn_lane_upfront")
        c.note = "turn pocket on one approach"
    elif tool == "bike_route":
        mult = Af("sim_engine.protected_bikeway_cost_multiplier") if p["facility"] == "protected" else 1.0
        c.upfront_usd = r["km"] * Af("costs.bike_route_per_km") * mult
        c.note = f"{r['km']:.2f} km {p['facility']}"
    elif tool == "safe_walk_route":
        c.upfront_usd = r["km"] * Af("costs.safe_walk_route_per_km")
        if p.get("crossing_guards", True):
            guards = max(1, math.ceil(r["km"] * Af("costs.crossing_guards_per_km")))
            c.per_year_usd = guards * Af("costs.crossing_guard_per_year")
            c.note = f"{r['km']:.2f} km + {guards} crossing guards"
        else:
            c.note = f"{r['km']:.2f} km"
    elif tool == "teen_drive_policy":
        c.note = "No direct cost."
    elif tool == "custom":
        c.upfront_usd = float(r["levers"]["cost_upfront_usd"])
        c.per_year_usd = float(r["levers"]["cost_per_year_usd"])
        c.note = "LLM estimated"
    return c


def _apply(world: WorldState, tool: str, p: dict[str, Any], r: dict[str, Any]) -> None:
    net = world.net
    pers, hh = world.persons, world.households
    circ = Af("sim_engine.circuity_factor")
    if tool == "bell_time":
        s = world.schools[r["school_idx"]]
        s.bell_s = clock_to_s(p["start"])
        s.bell_clock = p["start"]
        s.bell_changed = True
    elif tool == "school_shuttle":
        j = len(world.shuttles)
        sh = Shuttle(school=r["school_idx"], stop_nodes=[s["node"] for s in r["stops"]],
                     stop_x=[float(net.x[s["node"]]) for s in r["stops"]], stop_z=[float(net.z[s["node"]]) for s in r["stops"]],
                     buses=int(p["buses"]), headway_min=float(p["headway_min"]))
        world.shuttles.append(sh)
        stu = np.nonzero(pers.school == r["school_idx"])[0]
        hx, hz = hh.x[pers.hh[stu]], hh.z[pers.hh[stu]]
        d = np.min(np.hypot(hx[:, None] - np.array(sh.stop_x)[None, :], hz[:, None] - np.array(sh.stop_z)[None, :]), axis=1) / 1000.0 * circ
        ok = d <= Af("modechoice.shuttle_max_walk_to_stop_km")
        cur = world.shuttle_walk_km[stu]
        better = ok & (~np.isfinite(cur) | (d < cur))
        world.shuttle_of_person[stu[better]] = j
        world.shuttle_walk_km[stu[better]] = d[better]
    elif tool == "carpool_program":
        lvl = p["incentive"]
        boost = Af(f"tool_effects.carpool_utility_boost.{lvl}")
        cap = Af(f"tool_effects.carpool_adoption_cap_share.{lvl}")
        if r["school_idx"] >= 0:
            mask = pers.school == r["school_idx"]
        else:
            mask = (pers.school >= 0) | pers.is_worker
        world.shift_student["carpool"][mask & (pers.school >= 0)] += boost
        if r["school_idx"] < 0:
            world.shift_worker["carpool"][mask & pers.is_worker] += boost
        world.carpool_caps.append((mask, cap))
    elif tool == "dropoff_redesign":
        e = world.entrances[r["entrance_idx"]]
        e.curb_spots += p["extra_curb_spots"]
        if p.get("faster_unload"):
            e.unload_s *= Af("tool_effects.faster_unload_factor")
    elif tool == "new_dropoff_entrance":
        s = world.schools[r["school_idx"]]
        base = world.entrances[s.entrances[0]] if s.entrances else None
        k = sum(1 for e in world.entrances if e.school == r["school_idx"] and e.added_by_plan) + 1
        e_idx = r["edge_idx"]
        node = int(net.ev[e_idx])
        from pipeline.geo import scene_to_latlon

        lat, lon = scene_to_latlon(float(net.x[node]), float(net.z[node]))
        world.entrances.append(Entrance(
            key=f"{s.id}/new_{k}", id=f"new_{k}", school=r["school_idx"], node=node, approach_edge=e_idx,
            curb_spots=float(p["curb_spots"]),
            unload_s=base.unload_s if base else Af("schools.defaults.unload_seconds"),
            x=float(net.x[node]), z=float(net.z[node]), lat=lat, lon=lon, verified=False, added_by_plan=True,
        ))
        # parents have no habit at a brand-new entrance: judge its line at the school's existing rate
        ne = world.entrances[-1]
        ne.ref_service_rate = base.ref_service_rate if base and base.ref_service_rate > 0 else service_rate(ne.curb_spots, ne.unload_s)
        s.entrances.append(len(world.entrances) - 1)
    elif tool == "signal_timing":
        approaches = np.nonzero((net.ev == r["node"]) & ~net.is_freeway)[0]
        ns = np.abs(net.end_dz[approaches]) >= np.abs(net.end_dx[approaches])
        prio = ns if p["priority"] == "north_south" else ~ns
        world.signal_factor[approaches[prio]] *= Af("tool_effects.signal_priority_delay_factor")
        world.signal_factor[approaches[~prio]] *= Af("tool_effects.signal_nonpriority_delay_factor")
    elif tool == "turn_lane":
        world.cap_factor[r["edge_idx"]] *= 1.0 + Af("tool_effects.turn_lane_capacity_gain")
    elif tool == "bike_route":
        xs, zs = np.array(r["xs"]), np.array(r["zs"])
        catch = Af("tool_effects.bike_route_catchment_km") * 1000.0
        boost = Af("tool_effects.bike_route_utility_boost")
        if p["facility"] == "protected":
            boost *= Af("sim_engine.protected_bikeway_boost_multiplier")
        near_home = _point_polyline_dist(hh.x[pers.hh], hh.z[pers.hh], xs, zs) <= catch
        sx = np.array([s.x for s in world.schools] + [np.nan])[pers.school]
        sz = np.array([s.z for s in world.schools] + [np.nan])[pers.school]
        dx = np.where(pers.school >= 0, sx, pers.work_x)
        dz = np.where(pers.school >= 0, sz, pers.work_z)
        near_dest = np.isfinite(dx) & (_point_polyline_dist(np.nan_to_num(dx), np.nan_to_num(dz), xs, zs) <= catch)
        m = near_home | near_dest
        world.shift_student["bike"][m & (pers.school >= 0)] += boost
        world.shift_worker["bike"][m & pers.is_worker] += boost
        world.plan_edges.setdefault("bike_route", []).extend(r["edges"])
    elif tool == "safe_walk_route":
        xs, zs = np.array(r["xs"]), np.array(r["zs"])
        catch = Af("tool_effects.safe_walk_catchment_km") * 1000.0
        boost = Af("tool_effects.safe_walk_utility_boost")
        if p.get("crossing_guards", True):
            boost += Af("tool_effects.crossing_guard_extra_boost")
        stu = pers.school >= 0 if r["school_idx"] < 0 else pers.school == r["school_idx"]
        near = _point_polyline_dist(hh.x[pers.hh], hh.z[pers.hh], xs, zs) <= catch
        world.shift_student["walk"][stu & near] += boost
        world.plan_edges.setdefault("safe_walk_route", []).extend(r["edges"])
    elif tool == "teen_drive_policy":
        s = world.schools[r["school_idx"]]
        s.permits_cap = int(p["permits_cap"]) if s.permits_cap is None else min(s.permits_cap, int(p["permits_cap"]))
    elif tool == "custom":
        lv = r["levers"]
        for m, v in lv["mode_utility_shifts"].items():
            if m in world.shift_student:
                world.shift_student[m] += v * (pers.school >= 0)
            if m in world.shift_worker:
                world.shift_worker[m] += v * pers.is_worker
        for c in lv["capacity_changes"]:
            world.cap_factor[int(c["edge"])] *= float(c["factor"])
        adopt_hi = (lv.get("adoption_range") or [0.0, 1.0])[1]
        is_stu = pers.school >= 0
        for sh in lv.get("targeted_shifts", []):
            m, v, s_i = sh["mode"], float(sh["utils"]), int(sh["school_idx"])
            if sh["applies_to"] in ("students", "all") and m in world.shift_student:
                mask = is_stu if s_i < 0 else pers.school == s_i
                world.shift_student[m][mask] += v
                if v > 0:
                    world.adoption_caps.append((mask.copy(), m, float(adopt_hi)))
            if sh["applies_to"] in ("workers", "all") and m in world.shift_worker:
                mask = np.asarray(pers.is_worker, dtype=bool)
                world.shift_worker[m][mask] += v
                if v > 0:
                    world.adoption_caps.append((mask.copy(), m, float(adopt_hi)))
        for ef in lv.get("entrance_factors", []):
            world.entrances[int(ef["entrance_idx"])].unload_s /= float(ef["factor"])
        world.llm_estimated.append("custom")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def check_plan(world: WorldState, plan: dict[str, Any] | Plan) -> PlanCheck:
    try:
        pl = plan if isinstance(plan, Plan) else Plan.model_validate(plan or {})
    except ValidationError as exc:
        err = exc.errors()[0]
        return PlanCheck(ok=False, errors=[f"plan: {'.'.join(str(x) for x in err['loc'])}: {err['msg']}"])
    errors: list[str] = []
    warnings: list[str] = []
    try:
        mission = mission_def(pl.mission)
    except (ToolError, FileNotFoundError):
        return PlanCheck(ok=False, errors=[f"plan: unknown mission '{pl.mission}'"])
    allowed = set(mission.get("tools", []))
    defs = tool_defs()
    resolved: list[dict[str, Any]] = []
    lines: list[CostLine] = []
    violations: list[str] = []
    llm: list[str] = []
    counts: dict[str, int] = {}
    seen_school: dict[tuple[str, str], int] = {}
    for i, ti in enumerate(pl.tools):
        counts[ti.tool] = counts.get(ti.tool, 0) + 1
        label = ti.tool if counts[ti.tool] == 1 else f"{ti.tool} #{counts[ti.tool]}"
        if ti.tool not in defs:
            errors.append(f"{label}: unknown tool '{ti.tool}'")
            continue
        if ti.tool not in allowed:
            errors.append(f"{label}: tool not available in mission '{pl.mission}'")
            continue
        tdef = defs[ti.tool]
        try:
            params, w = _validate_params(world, tdef, ti.params)
            warnings.extend(f"{label}: {x}" for x in w)
            tw: list[str] = []
            res = _resolve(world, ti.tool, params, tw)
            warnings.extend(f"{label}: {x}" for x in tw)
        except ToolError as exc:
            errors.append(f"{label}: {exc}")
            continue
        if ti.tool in ("bell_time", "teen_drive_policy"):
            key = (ti.tool, params["school"])
            if key in seen_school:
                errors.append(f"{label}: school '{params['school']}' set more than once")
                continue
            seen_school[key] = i
        if ti.tool == "turn_lane" and any(rt["tool"] == "turn_lane" and rt["resolved"]["edge_idx"] == res["edge_idx"] for rt in resolved):
            warnings.append(f"{label}: edge {res['edge_idx']} already has a turn lane in this plan")
        if ti.tool == "custom":
            llm.append(f"custom: {params.get('description', '')[:80]}")
            through = set(A("roads.through_lane_classes"))
            for c in res["levers"]["capacity_changes"]:
                e = int(c["edge"])
                if world.net.highway[e] in through and float(c["factor"]) >= (world.net.lanes[e] + 1) / world.net.lanes[e] - 1e-9:
                    if "no_new_through_lanes" in mission.get("constraints", []) and "no_new_through_lanes" not in violations:
                        violations.append("no_new_through_lanes")
        line = _cost(world, ti.tool, params, res)
        lines.append(line)
        resolved.append({"index": i, "tool": ti.tool, "params": params, "resolved": res})
    up = float(sum(c.upfront_usd for c in lines))
    yr = float(sum(c.per_year_usd for c in lines))
    bu = float(mission.get("budget_usd_upfront", 0))
    by = float(mission.get("budget_usd_per_year", 0))
    return PlanCheck(
        ok=not errors, errors=errors, warnings=warnings, cost_upfront_usd=up, cost_per_year_usd=yr,
        over_budget=(up > bu) or (yr > by), budget_upfront_usd=bu, budget_per_year_usd=by,
        constraint_violations=violations, resolved_tools=resolved, cost_lines=lines, llm_estimated_tools=llm,
    )


def apply_plan(world: WorldState, check: PlanCheck) -> WorldState:
    """Return a modified clone of ``world``. ``check`` must be ok (from check_plan on the same world)."""
    if not check.ok:
        raise ToolError("cannot apply a plan with errors: " + "; ".join(check.errors))
    new = world.clone()
    for rt in check.resolved_tools:
        _apply(new, rt["tool"], rt["params"], rt["resolved"])
    return new
