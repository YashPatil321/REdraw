"""WorldState: everything the sim needs, loaded once from data/processed/.

Reads only the files in docs/data_contract.md. Big immutable arrays (network,
persons, households) live on shared objects; ``WorldState.clone()`` copies only
the small mutable parts that plan tools modify (capacity factors, signal
factors, schools, mode shifts, shuttles), so applying a plan is cheap.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from pipeline.config import processed_dir
from sim.assumptions import A, Af, clock_to_s
from sim.routing import PairGraph, trees

log = logging.getLogger("sim.world")

FREEWAY_CLASSES = ("motorway", "trunk")
STUDENT_MODES = ("drive_dropoff", "carpool", "school_bus", "school_shuttle", "walk", "bike", "teen_drive")
WORKER_MODES = ("drive_alone", "carpool", "bike", "walk")
ALL_MODES = ("drive_alone", "drive_dropoff", "carpool", "school_shuttle", "school_bus", "walk", "bike", "teen_drive")
MODE_LABELS = {
    "drive_alone": "Drive alone",
    "drive_dropoff": "Drive with drop-off",
    "carpool": "Carpool",
    "school_shuttle": "School shuttle",
    "school_bus": "School bus",
    "walk": "Walk",
    "bike": "Bike",
    "teen_drive": "Teen drives self",
}
INPUT_FILES = (
    "region_meta.json",
    "network_nodes.parquet",
    "network_edges.parquet",
    "households.parquet",
    "persons.parquet",
    "schools_resolved.json",
    "exits.json",
)


class WorldDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class TimeGrid:
    bin_start_s: int
    bin_s: int
    n_bins: int
    report_start_s: int
    report_end_s: int

    @classmethod
    def from_assumptions(cls) -> TimeGrid:
        start = clock_to_s(A("time.sim_start"))
        end = clock_to_s(A("time.sim_end"))
        bin_s = int(Af("time.bin_minutes") * 60)
        return cls(
            bin_start_s=start,
            bin_s=bin_s,
            n_bins=(end - start) // bin_s,
            report_start_s=clock_to_s(A("time.report_start")),
            report_end_s=clock_to_s(A("time.report_end")),
        )

    @property
    def end_s(self) -> int:
        return self.bin_start_s + self.n_bins * self.bin_s

    def bin_of(self, t: np.ndarray) -> np.ndarray:
        b = np.floor((np.asarray(t, dtype=np.float64) - self.bin_start_s) / self.bin_s).astype(np.int64)
        return np.clip(b, 0, self.n_bins - 1)

    def report_bins(self) -> tuple[int, int]:
        b0 = (self.report_start_s - self.bin_start_s) // self.bin_s
        b1 = (self.report_end_s - self.bin_start_s) // self.bin_s
        return int(b0), int(b1)

    def as_dict(self) -> dict[str, int]:
        return {
            "bin_start_s": self.bin_start_s,
            "bin_s": self.bin_s,
            "n_bins": self.n_bins,
            "report_start_s": self.report_start_s,
            "report_end_s": self.report_end_s,
        }


class Network:
    """Immutable drive network plus derived routing structures and caches."""

    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, geom_values: np.ndarray, geom_offsets: np.ndarray) -> None:
        self.node_ids = nodes["node_id"].to_numpy(np.int64)
        self._sorter = np.argsort(self.node_ids)
        self._sorted_ids = self.node_ids[self._sorter]
        self.n_nodes = len(self.node_ids)
        self.x = nodes["x"].to_numpy(np.float64)
        self.z = nodes["z"].to_numpy(np.float64)
        self.y = nodes["y"].to_numpy(np.float64) if "y" in nodes else np.zeros(self.n_nodes)
        self.lat = nodes["lat"].to_numpy(np.float64) if "lat" in nodes else np.full(self.n_nodes, np.nan)
        self.lon = nodes["lon"].to_numpy(np.float64) if "lon" in nodes else np.full(self.n_nodes, np.nan)
        self.signalized = nodes["signalized"].fillna(False).to_numpy(bool) if "signalized" in nodes else np.zeros(self.n_nodes, bool)
        self.boundary_exit = nodes["boundary_exit"].fillna("").astype(str).to_numpy() if "boundary_exit" in nodes else np.full(self.n_nodes, "", dtype=object)

        self.n_edges = len(edges)
        eu = self.node_index(edges["u"].to_numpy(np.int64))
        ev = self.node_index(edges["v"].to_numpy(np.int64))
        if (eu < 0).any() or (ev < 0).any():
            raise WorldDataError("network_edges.parquet references node ids missing from network_nodes.parquet")
        self.eu = eu.astype(np.int32)
        self.ev = ev.astype(np.int32)
        self.length_m = edges["length_m"].to_numpy(np.float64)
        self.lanes = np.maximum(edges["lanes"].fillna(1).to_numpy(np.float64), 1.0)
        self.highway = edges["highway"].fillna("").astype(str).to_numpy()
        self.name = edges["name"].fillna("").astype(str).to_numpy() if "name" in edges else np.full(self.n_edges, "", dtype=object)
        self.label = edges["label"].fillna("").astype(str).to_numpy() if "label" in edges else np.full(self.n_edges, "", dtype=object)
        self.maxspeed_kph = edges["maxspeed_kph"].to_numpy(np.float64) if "maxspeed_kph" in edges else np.full(self.n_edges, np.nan)
        self.capacity_vph = edges["capacity_vph"].to_numpy(np.float64)
        ff = edges["free_flow_s"].to_numpy(np.float64) if "free_flow_s" in edges else self.length_m / (self.maxspeed_kph / 3.6)
        bad = ~np.isfinite(ff) | (ff <= 0)
        if bad.any():
            log.warning("%d edges have invalid free_flow_s; using length / 40 kph", int(bad.sum()))
            ff = np.where(bad, np.maximum(self.length_m, 1.0) / (40 / 3.6), ff)
        self.ff_s = np.maximum(ff, 0.1)
        badcap = ~np.isfinite(self.capacity_vph) | (self.capacity_vph <= 0)
        if badcap.any():
            log.warning("%d edges have invalid capacity_vph; using 600 vph per lane", int(badcap.sum()))
            self.capacity_vph = np.where(badcap, self.lanes * 600.0, self.capacity_vph)
        self.geom_values = geom_values
        self.geom_offsets = geom_offsets

        self.is_freeway = np.isin(self.highway, FREEWAY_CLASSES)
        nbrs = np.unique(np.concatenate([self.eu.astype(np.int64) * self.n_nodes + self.ev, self.ev.astype(np.int64) * self.n_nodes + self.eu]))
        deg = np.bincount((nbrs // self.n_nodes).astype(np.int64), minlength=self.n_nodes)
        self.intersection = deg >= 3
        self.graph = PairGraph(self.n_nodes, self.eu, self.ev)
        self._kdtree: cKDTree | None = None
        self._skims: dict[int, np.ndarray] = {}
        self._static_rows: dict[int, int] = {}
        self._static_pred = np.zeros((0, self.n_nodes), np.int32)
        self._zone_rep: np.ndarray | None = None
        self.mid_x, self.mid_z = self._edge_midpoints()
        self.end_dx, self.end_dz = self._edge_end_directions()

    # ---- lookups -------------------------------------------------------------
    def node_index(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self._sorted_ids, ids)
        pos = np.clip(pos, 0, max(self.n_nodes - 1, 0))
        found = self._sorted_ids[pos] == ids
        out = np.where(found, self._sorter[pos], -1)
        return out.astype(np.int64)

    @property
    def kdtree(self) -> cKDTree:
        if self._kdtree is None:
            self._kdtree = cKDTree(np.column_stack([self.x, self.z]))
        return self._kdtree

    def nearest_node(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        _, idx = self.kdtree.query(np.column_stack([np.atleast_1d(x), np.atleast_1d(z)]))
        return np.asarray(idx, dtype=np.int64)

    def edge_geometry(self, e: int) -> np.ndarray:
        a, b = self.geom_offsets[e], self.geom_offsets[e + 1]
        pts = self.geom_values[a:b]
        if len(pts) < 6:
            u, v = self.eu[e], self.ev[e]
            return np.array([self.x[u], self.y[u], self.z[u], self.x[v], self.y[v], self.z[v]], dtype=np.float32)
        return pts

    def _edge_midpoints(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.x[self.eu] + self.x[self.ev]) / 2.0, (self.z[self.eu] + self.z[self.ev]) / 2.0

    def _edge_end_directions(self) -> tuple[np.ndarray, np.ndarray]:
        """Direction of the last segment of each edge (for signal approach direction)."""
        dx = self.x[self.ev] - self.x[self.eu]
        dz = self.z[self.ev] - self.z[self.eu]
        offs = self.geom_offsets
        if offs is not None and len(offs) == self.n_edges + 1:
            n_pts = (offs[1:] - offs[:-1]) // 3
            has = n_pts >= 2
            if has.any():
                last = offs[1:][has] - 3
                prev = offs[1:][has] - 6
                gx = self.geom_values[last] - self.geom_values[prev]
                gz = self.geom_values[last + 2] - self.geom_values[prev + 2]
                okg = (np.abs(gx) + np.abs(gz)) > 1e-6
                idx = np.nonzero(has)[0][okg]
                dx[idx] = gx[okg]
                dz[idx] = gz[okg]
        return dx, dz

    # ---- derived routing data ------------------------------------------------
    def zero_flow_tt(self, signal_factor: np.ndarray | None = None) -> np.ndarray:
        """Edge traversal time at zero volume (free flow + fixed node control delay)."""
        nd = node_delay(self, np.zeros(self.n_edges), signal_factor)
        return self.ff_s + nd

    def skim_to(self, anchors: np.ndarray) -> np.ndarray:
        """Free-flow travel time (s) from every node to each anchor node. Shape (k, N)."""
        anchors = np.asarray(anchors, dtype=np.int64)
        missing = [int(a) for a in np.unique(anchors) if int(a) not in self._skims]
        if missing:
            cost, _ = self.graph.best_edges(self.zero_flow_tt())
            _, rev = self.graph.matrices(cost)
            dist, _ = trees(rev, np.array(missing))
            for a, row in zip(missing, dist, strict=True):
                self._skims[a] = row
        if len(anchors) == 0:
            return np.zeros((0, self.n_nodes), np.float32)
        return np.stack([self._skims[int(a)] for a in anchors])

    def zone_rep(self) -> np.ndarray:
        """Zone hub node for every node (grid cells, hub = intersection nearest the cell centroid)."""
        if self._zone_rep is None:
            g = int(A("sim_engine.routing_zone_grid"))
            x0, x1 = self.x.min(), self.x.max() + 1e-6
            z0, z1 = self.z.min(), self.z.max() + 1e-6
            cx = np.minimum(((self.x - x0) / (x1 - x0) * g).astype(int), g - 1)
            cz = np.minimum(((self.z - z0) / (z1 - z0) * g).astype(int), g - 1)
            cell = cz * g + cx
            rep = np.zeros(self.n_nodes, dtype=np.int64)
            pen = np.where(self.intersection, 0.0, 1e6)
            for c in np.unique(cell):
                members = np.nonzero(cell == c)[0]
                mx, mz = self.x[members].mean(), self.z[members].mean()
                d = (self.x[members] - mx) ** 2 + (self.z[members] - mz) ** 2 + pen[members]
                rep[members] = members[int(np.argmin(d))]
            self._zone_rep = rep
        return self._zone_rep

    def static_tree_rows(self, reps: np.ndarray) -> np.ndarray:
        """Forward free-flow shortest-path trees from hub nodes (cached). Returns row ids."""
        reps = np.asarray(reps, dtype=np.int64)
        missing = [int(r) for r in np.unique(reps) if int(r) not in self._static_rows]
        if missing:
            cost, _ = self.graph.best_edges(self.zero_flow_tt())
            fwd, _ = self.graph.matrices(cost)
            _, pred = trees(fwd, np.array(missing))
            start = len(self._static_pred)
            self._static_pred = np.concatenate([self._static_pred, pred]) if start else pred
            for i, r in enumerate(missing):
                self._static_rows[r] = start + i
        return np.array([self._static_rows[int(r)] for r in reps], dtype=np.int64)

    def static_best_edges(self) -> np.ndarray:
        if not hasattr(self, "_static_best"):
            _, best = self.graph.best_edges(self.zero_flow_tt())
            self._static_best = best[None, :]
        return self._static_best


def node_delay(net: Network, vc: np.ndarray, signal_factor: np.ndarray | None) -> np.ndarray:
    """Intersection delay (s) charged on each edge for its downstream node.

    Signalized node: fixed delay + vc-based term (vc^2 scaled). Unsignalized
    intersection (3+ neighbours): small fixed delay. Freeway edges get none.
    ``vc`` may be (E,) or (B, E). ``signal_factor`` scales per approach (signal_timing).
    """
    fixed = Af("assignment.signal_fixed_delay_s")
    vcd = Af("assignment.signal_vc_delay_s")
    unsig = Af("assignment.unsignalized_delay_s")
    vmax = Af("assignment.max_vc_for_bpr")
    sig = net.signalized[net.ev] & ~net.is_freeway
    inter = net.intersection[net.ev] & ~net.signalized[net.ev] & ~net.is_freeway
    vcc = np.minimum(vc, vmax)
    d = np.where(sig, fixed + vcd * vcc**2, np.where(inter, unsig, 0.0))
    if signal_factor is not None:
        d = d * signal_factor
    return d


@dataclass
class Entrance:
    key: str
    id: str
    school: int
    node: int
    approach_edge: int
    curb_spots: float
    unload_s: float
    x: float
    z: float
    lat: float
    lon: float
    verified: bool
    added_by_plan: bool = False


@dataclass
class School:
    id: str
    name: str
    grades: tuple[int, int]
    bell_s: int
    bell_clock: str
    x: float
    z: float
    lat: float
    lon: float
    verified: bool
    entrances: list[int]
    students: int
    source: str = ""
    permits_cap: int | None = None
    bell_changed: bool = False


@dataclass
class Exit:
    id: str
    label: str
    node: int
    x: float
    z: float
    bearing_deg: float


@dataclass
class Shuttle:
    school: int
    stop_nodes: list[int]
    stop_x: list[float]
    stop_z: list[float]
    buses: int
    headway_min: float


@dataclass
class Persons:
    pid: np.ndarray
    hh: np.ndarray  # household row index
    age: np.ndarray
    is_worker: np.ndarray
    wfh: np.ndarray
    work_node: np.ndarray  # node idx or -1
    work_external: np.ndarray
    work_exit: np.ndarray  # exit index or -1
    work_x: np.ndarray
    work_z: np.ndarray
    school: np.ndarray  # school idx or -1
    grade: np.ndarray
    has_license: np.ndarray
    mode_hint: np.ndarray

    def __len__(self) -> int:
        return len(self.pid)


@dataclass
class Households:
    hid: np.ndarray
    home_node: np.ndarray
    x: np.ndarray
    z: np.ndarray
    vehicles: np.ndarray
    size: np.ndarray
    n_kids: np.ndarray

    def __len__(self) -> int:
        return len(self.hid)


@dataclass
class WorldState:
    data_dir: Path
    meta: dict[str, Any]
    synthetic: bool
    time: TimeGrid
    net: Network
    persons: Persons
    households: Households
    schools: list[School]
    entrances: list[Entrance]
    exits: list[Exit]
    data_hash: str
    schools_raw: list[dict[str, Any]]
    # --- mutable plan state (copied by clone) ---
    cap_factor: np.ndarray = field(default_factory=lambda: np.zeros(0))
    signal_factor: np.ndarray = field(default_factory=lambda: np.zeros(0))
    shift_student: dict[str, np.ndarray] = field(default_factory=dict)
    shift_worker: dict[str, np.ndarray] = field(default_factory=dict)
    carpool_caps: list[tuple[np.ndarray, float]] = field(default_factory=list)
    shuttles: list[Shuttle] = field(default_factory=list)
    shuttle_of_person: np.ndarray = field(default_factory=lambda: np.zeros(0))
    shuttle_walk_km: np.ndarray = field(default_factory=lambda: np.zeros(0))
    events: list[Any] = field(default_factory=list)
    llm_estimated: list[str] = field(default_factory=list)
    plan_edges: dict[str, list[int]] = field(default_factory=dict)
    knobs: dict[str, float] = field(default_factory=dict)
    demand_overrides: list[Any] = field(default_factory=list)
    # calibrated ASC adjustments (sim.modechoice.calibrate_ascs); fixed from the baseline, shared by clones
    asc_student_adj: np.ndarray | None = None
    asc_worker_adj: np.ndarray | None = None
    asc_calibration: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        P, E = len(self.persons), self.net.n_edges
        if len(self.cap_factor) != E:
            self.cap_factor = np.ones(E)
        if len(self.signal_factor) != E:
            self.signal_factor = np.ones(E)
        for m in STUDENT_MODES:
            self.shift_student.setdefault(m, np.zeros(P, np.float32))
        for m in WORKER_MODES:
            self.shift_worker.setdefault(m, np.zeros(P, np.float32))
        if len(self.shuttle_of_person) != P:
            self.shuttle_of_person = np.full(P, -1, np.int32)
            self.shuttle_walk_km = np.full(P, np.nan, np.float32)
        if not self.knobs:
            self.knobs = default_knobs()

    # ---- helpers ---------------------------------------------------------------
    def clone(self) -> WorldState:
        new = copy.copy(self)
        new.cap_factor = self.cap_factor.copy()
        new.signal_factor = self.signal_factor.copy()
        new.shift_student = {k: v.copy() for k, v in self.shift_student.items()}
        new.shift_worker = {k: v.copy() for k, v in self.shift_worker.items()}
        new.carpool_caps = list(self.carpool_caps)
        new.shuttles = copy.deepcopy(self.shuttles)
        new.shuttle_of_person = self.shuttle_of_person.copy()
        new.shuttle_walk_km = self.shuttle_walk_km.copy()
        new.schools = copy.deepcopy(self.schools)
        new.entrances = copy.deepcopy(self.entrances)
        new.events = list(self.events)
        new.llm_estimated = list(self.llm_estimated)
        new.plan_edges = {k: list(v) for k, v in self.plan_edges.items()}
        new.knobs = dict(self.knobs)
        new.demand_overrides = list(self.demand_overrides)
        return new

    def school_index(self, school_id: str) -> int:
        for i, s in enumerate(self.schools):
            if s.id == school_id:
                return i
        return -1

    def entrance_index(self, key: str) -> int:
        for i, e in enumerate(self.entrances):
            if e.key == key:
                return i
        return -1

    def approach_u(self) -> np.ndarray:
        return np.array([self.net.eu[e.approach_edge] for e in self.entrances], dtype=np.int64)

    def entrance_nodes(self) -> np.ndarray:
        return np.array([e.node for e in self.entrances], dtype=np.int64)

    @property
    def hero_school(self) -> School | None:
        i = self.school_index("del_norte_hs")
        return self.schools[i] if i >= 0 else (self.schools[0] if self.schools else None)


def default_knobs() -> dict[str, float]:
    return {
        "demand_scale": Af("demand.demand_scale"),
        "departure_sd_min": Af("time_of_day.worker_departure_sd_min"),
        "capacity_factor": Af("assignment.demand_capacity_factor"),
    }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def data_hash(data_dir: Path) -> str:
    h = hashlib.sha1()
    for name in INPUT_FILES:
        p = data_dir / name
        if p.exists():
            st = p.stat()
            h.update(f"{name}:{st.st_size}:".encode())
            with open(p, "rb") as f:
                h.update(hashlib.sha1(f.read()).digest())
    return h.hexdigest()[:16]


def _require(data_dir: Path) -> None:
    missing = [n for n in INPUT_FILES if not (data_dir / n).exists()]
    if missing:
        raise WorldDataError(
            f"Missing processed world files in {data_dir}: {', '.join(missing)}. "
            "Run `.venv/bin/python pipeline/build_all.py` (real data) or "
            "`.venv/bin/python pipeline/build_all.py --synthetic` (offline dev fixture)."
        )


def _read_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    tbl = pq.read_table(path, columns=["geometry"])
    col = tbl.column("geometry").combine_chunks()
    try:
        offsets = np.asarray(col.offsets, dtype=np.int64)
        values = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.float32)
        offsets = offsets - offsets[0]
    except Exception:  # pragma: no cover - fallback for odd encodings
        lists = col.to_pylist()
        offsets = np.zeros(len(lists) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(x or []) for x in lists])
        values = np.concatenate([np.asarray(x or [], np.float32) for x in lists]) if lists else np.zeros(0, np.float32)
    return values, offsets


def load_world(data_dir: Path | str | None = None) -> WorldState:
    data_dir = Path(data_dir) if data_dir else processed_dir()
    _require(data_dir)
    meta = json.loads((data_dir / "region_meta.json").read_text())
    nodes = pd.read_parquet(data_dir / "network_nodes.parquet")
    edge_cols = [c for c in pq.read_schema(data_dir / "network_edges.parquet").names if c != "geometry"]
    edges = pd.read_parquet(data_dir / "network_edges.parquet", columns=edge_cols)
    if "edge_idx" in edges and not np.array_equal(edges["edge_idx"].to_numpy(), np.arange(len(edges))):
        log.warning("network_edges.parquet: edge_idx does not equal row order; using row order per contract")
    gv, go = _read_geometry(data_dir / "network_edges.parquet")
    net = Network(nodes, edges, gv, go)

    hh_df = pd.read_parquet(data_dir / "households.parquet")
    p_df = pd.read_parquet(data_dir / "persons.parquet")
    schools_raw = json.loads((data_dir / "schools_resolved.json").read_text()).get("schools", [])
    exits_raw = json.loads((data_dir / "exits.json").read_text()).get("exits", [])

    # households ------------------------------------------------------------
    home = net.node_index(hh_df["home_node"].to_numpy(np.int64))
    bad = home < 0
    if bad.any():
        log.warning("%d households have home_node not in network; snapping by x,z", int(bad.sum()))
        home[bad] = net.nearest_node(hh_df["x"].to_numpy()[bad], hh_df["z"].to_numpy()[bad])
    households = Households(
        hid=hh_df["household_id"].to_numpy(np.int64),
        home_node=home,
        x=hh_df["x"].to_numpy(np.float64),
        z=hh_df["z"].to_numpy(np.float64),
        vehicles=hh_df["vehicles"].fillna(1).to_numpy(np.int32) if "vehicles" in hh_df else np.ones(len(hh_df), np.int32),
        size=hh_df["size"].fillna(1).to_numpy(np.int32) if "size" in hh_df else np.ones(len(hh_df), np.int32),
        n_kids=hh_df["n_kids"].fillna(0).to_numpy(np.int32) if "n_kids" in hh_df else np.zeros(len(hh_df), np.int32),
    )

    # exits -----------------------------------------------------------------
    exits: list[Exit] = []
    for ex in exits_raw:
        ni = int(net.node_index(np.array([int(ex["node_id"])]))[0])
        if ni < 0:
            ni = int(net.nearest_node(np.array([ex.get("x", 0.0)]), np.array([ex.get("z", 0.0)]))[0])
            log.warning("exit %s node_id not in network; snapped", ex.get("id"))
        exits.append(Exit(id=str(ex["id"]), label=str(ex.get("label", ex["id"])), node=ni, x=float(net.x[ni]), z=float(net.z[ni]), bearing_deg=float(ex.get("bearing_deg", 0.0))))
    exit_ids = {e.id: i for i, e in enumerate(exits)}

    # schools ---------------------------------------------------------------
    schools: list[School] = []
    entrances: list[Entrance] = []
    for s in schools_raw:
        sidx = len(schools)
        bell_clock = str(s.get("bell_start") or A("schools.defaults.bell_start"))
        g = s.get("grades") or [0, 12]
        sx = float(s.get("x", np.nan))
        sz = float(s.get("z", np.nan))
        school = School(
            id=str(s["id"]), name=str(s.get("name", s["id"])), grades=(int(g[0]), int(g[-1])),
            bell_s=clock_to_s(bell_clock), bell_clock=bell_clock, x=sx, z=sz,
            lat=float(s.get("lat", np.nan)), lon=float(s.get("lon", np.nan)),
            verified=bool(s.get("verified", False)), entrances=[], students=int(s.get("students", 0) or 0),
            source=str(s.get("source", "")),
        )
        ents = s.get("entrances") or []
        if not ents:
            log.warning("school %s has no entrances; synthesizing one at the nearest node", school.id)
            ents = [{"id": "main_dropoff", "x": sx, "z": sz, "verified": False, "synthesized": True}]
        for ent in ents:
            ent_node = int(net.node_index(np.array([int(ent.get("node_id", -1))]))[0]) if ent.get("node_id") is not None else -1
            if ent_node < 0:
                ent_node = int(net.nearest_node(np.array([ent.get("x", sx)]), np.array([ent.get("z", sz)]))[0])
            ae = ent.get("approach_edge_idx")
            ae = int(ae) if ae is not None and 0 <= int(ae) < net.n_edges else -1
            if ae < 0 or int(net.ev[ae]) != ent_node:
                cand = np.nonzero(net.ev == ent_node)[0]
                if len(cand) == 0:
                    raise WorldDataError(f"school entrance {school.id}/{ent.get('id')} node has no incoming edge")
                cand = cand[~net.is_freeway[cand]] if (~net.is_freeway[cand]).any() else cand
                ae = int(cand[np.argmax(net.length_m[cand])])
                log.warning("entrance %s/%s: approach_edge_idx missing/invalid; using edge %d", school.id, ent.get("id"), ae)
            entrances.append(Entrance(
                key=f"{school.id}/{ent.get('id', 'main_dropoff')}", id=str(ent.get("id", "main_dropoff")), school=sidx,
                node=ent_node, approach_edge=ae,
                curb_spots=float(ent.get("curb_spots") or A("schools.defaults.curb_spots")),
                unload_s=float(ent.get("unload_seconds") or A("schools.defaults.unload_seconds")),
                x=float(ent.get("x", net.x[ent_node])), z=float(ent.get("z", net.z[ent_node])),
                lat=float(ent.get("lat", net.lat[ent_node])), lon=float(ent.get("lon", net.lon[ent_node])),
                verified=bool(ent.get("verified", False)),
            ))
            school.entrances.append(len(entrances) - 1)
        if not np.isfinite(school.x):
            school.x, school.z = entrances[school.entrances[0]].x, entrances[school.entrances[0]].z
        schools.append(school)
    school_ids = {s.id: i for i, s in enumerate(schools)}

    # persons ---------------------------------------------------------------
    hh_index = pd.Index(households.hid)
    hh_row = hh_index.get_indexer(p_df["household_id"].to_numpy(np.int64))
    if (hh_row < 0).any():
        log.warning("%d persons reference unknown households; dropped", int((hh_row < 0).sum()))
        p_df = p_df[hh_row >= 0].reset_index(drop=True)
        hh_row = hh_row[hh_row >= 0]
    P = len(p_df)
    is_worker = p_df["is_worker"].fillna(False).to_numpy(bool)
    wfh = p_df["works_from_home"].fillna(False).to_numpy(bool) if "works_from_home" in p_df else np.zeros(P, bool)
    wexit_str = p_df["work_exit"].fillna("").astype(str).to_numpy() if "work_exit" in p_df else np.full(P, "")
    work_exit = np.array([exit_ids.get(s, -1) for s in wexit_str], dtype=np.int32)
    unknown_exit = (wexit_str != "") & (work_exit < 0)
    if unknown_exit.any():
        log.warning("%d persons have unknown work_exit ids", int(unknown_exit.sum()))
    work_node = net.node_index(p_df["work_node"].fillna(-1).to_numpy(np.int64))
    work_x = p_df["work_x"].to_numpy(np.float64) if "work_x" in p_df else np.full(P, np.nan)
    work_z = p_df["work_z"].to_numpy(np.float64) if "work_z" in p_df else np.full(P, np.nan)
    ext = work_exit >= 0
    work_node[ext] = np.array([exits[i].node for i in work_exit[ext]], dtype=np.int64)
    fix = is_worker & (work_node < 0) & np.isfinite(work_x)
    if fix.any():
        work_node[fix] = net.nearest_node(work_x[fix], work_z[fix])
    sch_str = p_df["school_id"].fillna("").astype(str).to_numpy() if "school_id" in p_df else np.full(P, "")
    school_idx = np.array([school_ids.get(s, -1) for s in sch_str], dtype=np.int32)
    unknown_school = (sch_str != "") & (school_idx < 0)
    if unknown_school.any():
        log.warning("%d persons attend schools missing from schools_resolved.json; treated as non-students", int(unknown_school.sum()))
    persons = Persons(
        pid=p_df["person_id"].to_numpy(np.int64), hh=hh_row.astype(np.int64),
        age=p_df["age"].fillna(30).to_numpy(np.int32), is_worker=is_worker, wfh=wfh,
        work_node=work_node, work_external=ext, work_exit=work_exit,
        work_x=work_x, work_z=work_z, school=school_idx,
        grade=p_df["grade"].fillna(-1).to_numpy(np.int32) if "grade" in p_df else np.full(P, -1, np.int32),
        has_license=p_df["has_license"].fillna(False).to_numpy(bool) if "has_license" in p_df else np.zeros(P, bool),
        mode_hint=p_df["baseline_mode_hint"].fillna("").astype(str).to_numpy() if "baseline_mode_hint" in p_df else np.full(P, ""),
    )
    for i, s in enumerate(schools):
        if not s.students:
            s.students = int((school_idx == i).sum())

    world = WorldState(
        data_dir=data_dir, meta=meta, synthetic=bool(meta.get("synthetic", False)),
        time=TimeGrid.from_assumptions(), net=net, persons=persons, households=households,
        schools=schools, entrances=entrances, exits=exits, data_hash=data_hash(data_dir), schools_raw=schools_raw,
    )
    world.knobs.update(load_calibration_knobs(data_dir))
    from sim.modechoice import calibrate_ascs

    world.asc_calibration = calibrate_ascs(world)
    return world


def load_calibration_knobs(data_dir: Path) -> dict[str, float]:
    """Tuned knobs from calibration.json, applied only when the world is calibrated."""
    p = data_dir / "calibration.json"
    if not p.exists():
        return {}
    try:
        cal = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    if cal.get("status") in ("calibrated", "above_target") and isinstance(cal.get("params"), dict):
        return {k: float(v) for k, v in cal["params"].items() if k in ("demand_scale", "departure_sd_min", "capacity_factor")}
    return {}
