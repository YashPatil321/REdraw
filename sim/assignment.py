"""Traffic assignment (spec 6.4) with drop-off queues coupled inside the loop (6.5).

Built-in backend: time-dependent incremental assignment with BPR link times
and the method of successive averages (MSA).

Per MSA iteration k:
1. Shortest-path trees are rebuilt per routing period (15 min) from that
   period's average edge times. Legs are routed on trees rooted at hub nodes:
   destination-rooted for legs that end at an exit or a school approach,
   origin-rooted for legs that start at an exit or a school entrance, and for
   all other legs a destination-rooted tree to the destination's zone hub
   followed by the free-flow path from the hub to the exact destination.
   scipy's dijkstra runs from all roots of a period in one call.
2. Dynamic network loading: every vehicle walks its path edge by edge
   (vectorized across all vehicles) with the time-dependent edge time of the
   5-minute bin in which it enters each edge. Entries per (bin, edge) give
   flows. Drop-off chains are loaded leg by leg: arrivals at a school feed the
   entrance queue model (sim/schools.py), and the next leg leaves when the car
   leaves the curb.
3. Volumes (and the queue spillback effects) are averaged with MSA weights
   1/k, and edge times are recomputed:
       t = t0 * (1 + alpha * (v/c)^beta) + node delay + spill delay
   Node delay: signalized downstream node -> fixed + vc_delay * (v/c)^2;
   unsignalized intersection -> small fixed delay; freeway edges -> none;
   scaled per approach by signal_timing factors.
4. Stop after min iterations once total travel time changes < 1 percent, or
   at max iterations. School-trip departures are planned from the expected
   travel time, which is MSA-updated from experienced times (people learn).

``AssignmentBackend`` is the interface a SUMO/MATSim backend would implement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from sim.assumptions import A, Af
from sim.demand import Trips
from sim.routing import concat_paths, trees, walk_to_root
from sim.schools import QueueOutput, run_queue, spill_effects
from sim.world import WorldState, node_delay


def bpr(t0: np.ndarray, vc: np.ndarray, alpha: float | None = None, beta: float | None = None,
        vc_cap: float | None = None) -> np.ndarray:
    """BPR volume-delay: t = t0 * (1 + alpha * (v/c)^beta), v/c capped numerically."""
    a = Af("assignment.bpr_alpha") if alpha is None else alpha
    b = Af("assignment.bpr_beta") if beta is None else beta
    cap = Af("assignment.max_vc_for_bpr") if vc_cap is None else vc_cap
    return t0 * (1.0 + a * np.minimum(vc, cap) ** b)


@dataclass
class EntranceStats:
    key: str
    queue_end: np.ndarray
    queue_max: np.ndarray
    spill_m: np.ndarray
    avg_wait_s: np.ndarray
    arrivals: np.ndarray


@dataclass
class AssignmentResult:
    tt: np.ndarray  # (B, E) edge traversal time (s) incl. node delay
    vol_vph: np.ndarray  # (B, E)
    vc: np.ndarray  # (B, E)
    speed_kph: np.ndarray  # (B, E)
    depart: np.ndarray  # per trip
    arrive: np.ndarray  # per trip final arrival (NaN if failed)
    ok: np.ndarray
    wp_arrive: np.ndarray  # (n, S) arrival at waypoint (before queue)
    wp_leave: np.ndarray  # (n, S) leaves waypoint (after queue + unload / dwell)
    wp_wait: np.ndarray  # (n, S) queue wait (s)
    wp_ff: np.ndarray  # (n, S) zero-flow time of the leg ending at the waypoint
    wp_entr: np.ndarray  # (n, S) entrance index used (-1)
    entrances: list[EntranceStats]
    traj: dict[str, np.ndarray]
    vht_report_h: float
    iterations: int
    rel_change: list[float]
    timings: dict[str, float] = field(default_factory=dict)
    failed_trips: int = 0


class AssignmentBackend(Protocol):
    """Interface for traffic assignment engines (spec 6.4.5, 13.3)."""

    name: str

    def assign(self, world: WorldState, trips: Trips, traj_rng: np.random.Generator | None = None,
               max_traj: int | None = None) -> AssignmentResult: ...


class _Router:
    """Shortest-path trees for one MSA iteration (one cost set per routing period)."""

    def __init__(self, world: WorldState, tt: np.ndarray, period_bins: int) -> None:
        self.w = world
        self.net = world.net
        self.g = world.net.graph
        self.tt = tt
        self.period_bins = period_bins
        self.n_periods = int(np.ceil(tt.shape[0] / period_bins))
        self.N = self.net.n_nodes
        self._mats: dict[int, tuple[Any, Any]] = {}
        self.best = np.zeros((self.n_periods, self.g.n_pairs), dtype=np.int32)
        self._have_best = np.zeros(self.n_periods, dtype=bool)
        self.rows: dict[tuple[str, int, int], int] = {}
        self.pred = np.zeros((64, self.N), dtype=np.int32)
        self.n_rows = 0
        self.n_dijkstra = 0

    def _period(self, p: int) -> tuple[Any, Any]:
        if p not in self._mats:
            b0 = p * self.period_bins
            cost = self.tt[b0 : b0 + self.period_bins].mean(axis=0)
            pc, best = self.g.best_edges(cost)
            self.best[p] = best
            self._have_best[p] = True
            self._mats[p] = self.g.matrices(pc)
        return self._mats[p]

    def ensure(self, kind: str, roots: np.ndarray, periods: np.ndarray) -> np.ndarray:
        """Make sure trees exist for (kind, root, period); return row index per request."""
        keys = np.unique(np.stack([roots, periods], axis=1), axis=0) if len(roots) else np.zeros((0, 2), np.int64)
        for p in np.unique(keys[:, 1]) if len(keys) else []:
            need = [int(r) for r in keys[keys[:, 1] == p, 0] if (kind, int(r), int(p)) not in self.rows]
            if not need:
                continue
            fwd, rev = self._period(int(p))
            _, pred = trees(rev if kind == "D" else fwd, np.array(need))
            self.n_dijkstra += len(need)
            while self.n_rows + len(need) > len(self.pred):
                self.pred = np.concatenate([self.pred, np.zeros_like(self.pred)])
            self.pred[self.n_rows : self.n_rows + len(need)] = pred
            for i, r in enumerate(need):
                self.rows[(kind, r, int(p))] = self.n_rows + i
            self.n_rows += len(need)
        return np.array([self.rows[(kind, int(r), int(p))] for r, p in zip(roots, periods, strict=True)], dtype=np.int64)


class IncrementalMSABackend:
    name = "builtin_msa"

    def assign(self, world: WorldState, trips: Trips, traj_rng: np.random.Generator | None = None,
               max_traj: int | None = None) -> AssignmentResult:
        return _assign(world, trips, traj_rng, max_traj)


def _assign(world: WorldState, trips: Trips, traj_rng: np.random.Generator | None, max_traj: int | None) -> AssignmentResult:
    t_start = time.perf_counter()
    net, tg = world.net, world.time
    B, E = tg.n_bins, net.n_edges
    n = trips.n
    S = trips.wp_node.shape[1] if trips.wp_node.ndim == 2 else 0
    cap = net.capacity_vph * world.cap_factor * world.knobs["capacity_factor"]
    tt0 = net.zero_flow_tt(world.signal_factor)
    period_bins = max(1, int(round(Af("sim_engine.routing_period_min") * 60 / tg.bin_s)))
    min_it = int(A("assignment.msa_min_iterations"))
    max_it = int(A("assignment.msa_max_iterations"))
    conv = Af("assignment.msa_convergence")

    capf_ev = np.ones((B, E))
    extra_ev = np.zeros((B, E))
    for ev in world.events:
        ev.apply(world, capf_ev, extra_ev)

    def edge_times(vol: np.ndarray, capf: np.ndarray, extra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vc = vol / (cap[None, :] * capf)
        t = bpr(net.ff_s[None, :], vc) + node_delay(net, vc, world.signal_factor) + extra
        return t, vc

    TT, _ = edge_times(np.zeros((B, E)), capf_ev, extra_ev)
    expected = trips.exp_time.astype(np.float64).copy()
    V_avg = np.zeros((B, E))
    capf_avg = np.ones((B, E))
    extra_avg = np.zeros((B, E))
    n_ent = len(world.entrances)
    ent_wait = np.zeros(n_ent)
    sample = _traj_sample(trips, traj_rng, max_traj)
    history: list[float] = []
    prev_total = None
    timings = {"routing": 0.0, "loading": 0.0, "queues": 0.0, "update": 0.0}
    last = None
    k = 0
    n_dij = 0
    for k in range(1, max_it + 1):
        router = _Router(world, TT, period_bins)
        last = _load(world, trips, TT, tt0, expected, ent_wait, router, sample, timings)
        n_dij += router.n_dijkstra
        t0 = time.perf_counter()
        counts_vph = last["counts"].reshape(B, E) * (3600.0 / tg.bin_s)
        capf_k = capf_ev.copy()
        extra_k = extra_ev.copy()
        for qi, q in enumerate(last["queues"]):
            spill_effects(net, world.entrances[qi].approach_edge, q.spill_m, capf_k, extra_k)
        V_avg += (counts_vph - V_avg) / k
        capf_avg += (capf_k - capf_avg) / k
        extra_avg += (extra_k - extra_avg) / k
        TT, _ = edge_times(V_avg, capf_avg, extra_avg)
        exp_new = last["target_experienced"]
        upd = np.isfinite(exp_new)
        expected[upd] += (exp_new[upd] - expected[upd]) / (k + 1)
        for qi, q in enumerate(last["queues"]):
            tot = q.arrivals.sum()
            wmean = float(np.nansum(q.avg_wait_s * q.arrivals) / tot) if tot > 0 else 0.0
            ent_wait[qi] += (wmean - ent_wait[qi]) / k
        total = last["total_tt"]
        timings["update"] += time.perf_counter() - t0
        if prev_total is not None and prev_total > 0:
            rel = abs(total - prev_total) / prev_total
            history.append(float(rel))
            if k >= min_it and rel < conv:
                break
        prev_total = total
    assert last is not None
    vc = V_avg / (cap[None, :] * capf_avg)
    speed = net.length_m[None, :] / np.maximum(TT, 1e-3) * 3.6
    timings["total"] = time.perf_counter() - t_start
    timings["dijkstra_runs"] = float(n_dij)
    ents = [EntranceStats(world.entrances[i].key, q.queue_end, q.queue_max, q.spill_m, q.avg_wait_s, q.arrivals)
            for i, q in enumerate(last["queues"])]
    return AssignmentResult(
        tt=TT.astype(np.float32), vol_vph=V_avg.astype(np.float32), vc=vc.astype(np.float32), speed_kph=speed.astype(np.float32),
        depart=last["depart"], arrive=last["arrive"], ok=last["ok"], wp_arrive=last["wp_arrive"], wp_leave=last["wp_leave"],
        wp_wait=last["wp_wait"], wp_ff=last["wp_ff"], wp_entr=last["wp_entr"], entrances=ents, traj=last["traj"],
        vht_report_h=last["vht"] / 3600.0, iterations=k, rel_change=history, timings=timings,
        failed_trips=int((~last["ok"]).sum()),
    )


def _traj_sample(trips: Trips, rng: np.random.Generator | None, max_traj: int | None) -> np.ndarray:
    mask = np.zeros(trips.n, dtype=bool)
    m = int(A("report.playback_max_trajectories")) if max_traj is None else int(max_traj)
    if m <= 0 or trips.n == 0:
        return mask
    rng = rng or np.random.default_rng(0)
    shuttles = np.nonzero(trips.kind == 2)[0][:m]
    mask[shuttles] = True
    rest = np.nonzero(trips.kind != 2)[0]
    k = min(m - len(shuttles), len(rest))
    if k > 0:
        w = np.maximum(trips.weight[rest], 1e-9)
        pick = rng.choice(rest, size=k, replace=False, p=w / w.sum())
        mask[pick] = True
    return mask


def _load(world: WorldState, trips: Trips, TT: np.ndarray, tt0: np.ndarray, expected: np.ndarray,
          ent_wait: np.ndarray, router: _Router, sample: np.ndarray, timings: dict[str, float]) -> dict[str, Any]:
    """One dynamic network loading of all trips on fixed edge times TT."""
    net, tg = world.net, world.time
    B, E = TT.shape
    n = trips.n
    S = trips.wp_node.shape[1]
    TTf = TT.ravel()
    ents = world.entrances
    ent_au = world.approach_u()
    ent_node = world.entrance_nodes()
    ent_edge = np.array([e.approach_edge for e in ents], dtype=np.int64)
    ent_school = np.array([e.school for e in ents], dtype=np.int64)
    school_ents = [np.array(s.entrances, dtype=np.int64) for s in world.schools]
    au_skim = net.skim_to(ent_au) if len(ents) else np.zeros((0, net.n_nodes))

    # departures
    has_target = np.isfinite(trips.target_arr)
    depart = np.where(has_target, trips.target_arr - expected, trips.depart)
    depart = np.clip(depart, tg.bin_start_s - 3600.0, tg.end_s)

    # resolve waypoints (entrance choice for queue stops: min skim time + learned wait)
    wp_entr = np.full((n, S), -1, dtype=np.int64)
    wp_route = np.full((n, S), -1, dtype=np.int64)  # node the routed path must reach
    wp_after = np.full((n, S), -1, dtype=np.int64)  # node the vehicle is at after the waypoint
    wp_edge = np.full((n, S), -1, dtype=np.int64)  # appended edge (approach edge) or -1
    prev = trips.origin.copy()
    for j in range(S):
        has = trips.wp_node[:, j] >= 0
        isent = has & (trips.wp_school[:, j] >= 0)
        plain = has & ~isent
        wp_route[plain, j] = trips.wp_node[plain, j]
        wp_after[plain, j] = trips.wp_node[plain, j]
        rows = np.nonzero(isent)[0]
        if len(rows):
            sch = trips.wp_school[rows, j]
            choice = np.empty(len(rows), dtype=np.int64)
            for s_i in np.unique(sch):
                m = sch == s_i
                cand = school_ents[s_i]
                if len(cand) == 1:
                    choice[m] = cand[0]
                    continue
                r = rows[m]
                q = trips.wp_queue[r, j]
                cost = au_skim[cand][:, prev[r]].T + net.ff_s[ent_edge[cand]][None, :] + ent_wait[cand][None, :]
                pick = cand[np.argmin(cost, axis=1)]
                choice[m] = np.where(q, pick, cand[0])
            wp_entr[rows, j] = choice
            wp_route[rows, j] = ent_au[choice]
            wp_after[rows, j] = ent_node[choice]
            wp_edge[rows, j] = ent_edge[choice]
        prev = np.where(has, wp_after[:, j], prev)

    n_wp = (trips.wp_node >= 0).sum(axis=1)
    ok = np.ones(n, dtype=bool)
    wp_arrive = np.full((n, S), np.nan)
    wp_leave = np.full((n, S), np.nan)
    wp_wait = np.zeros((n, S))
    wp_ff = np.zeros((n, S))
    arrive = np.full(n, np.nan)
    counts = np.zeros(B * E)
    vht = 0.0
    traj_parts: list[tuple[np.ndarray, ...]] = []
    queue_arr: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in ents]
    cur_node = trips.origin.copy()
    cur_t = depart.copy()
    exits = np.array([e.node for e in world.exits], dtype=np.int64)
    d_roots = np.unique(np.concatenate([exits, ent_au]))
    o_roots = np.unique(np.concatenate([exits, ent_node]))
    rep = net.zone_rep()
    rs, re_ = float(tg.report_start_s), float(tg.report_end_s)

    for p in range(S + 1):
        if p < S:
            legs = np.nonzero(ok & (p < n_wp))[0]
            dest = wp_route[legs, p]
        else:
            legs = np.nonzero(ok & (trips.final >= 0))[0]
            dest = trips.final[legs]
        if len(legs) == 0:
            continue
        orig = cur_node[legs]
        t_leg = cur_t[legs]
        t0 = time.perf_counter()
        paths, lens, okl = _route(router, orig, dest, t_leg, d_roots, o_roots, rep, tg)
        if p < S:
            app = wp_edge[legs, p]
            has_app = app >= 0
            app_m = np.where(has_app, app, -1).astype(np.int32)[:, None]
            paths, lens = concat_paths([(paths, lens), (app_m, has_app.astype(np.int32))])
        timings["routing"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        t_end, ffsum, vht_p, cnt, tp = _propagate(paths, lens, t_leg, trips.weight[legs], TTf, tt0, E, tg, rs, re_,
                                                  sample[legs], legs, p)
        counts += cnt
        vht += vht_p
        if tp is not None:
            traj_parts.append(tp)
        timings["loading"] += time.perf_counter() - t0
        bad = ~okl
        ok[legs[bad]] = False
        legs, t_end, ffsum = legs[~bad], t_end[~bad], ffsum[~bad]
        if p == S:
            arrive[legs] = t_end
            continue
        wp_arrive[legs, p] = t_end
        wp_ff[legs, p] = ffsum
        leave = t_end + trips.wp_dwell[legs, p]
        queued = trips.wp_queue[legs, p] & (wp_entr[legs, p] >= 0)
        t0 = time.perf_counter()
        if queued.any():
            ql = legs[queued]
            qe = wp_entr[ql, p]
            for ei in np.unique(qe):
                m = qe == ei
                rows = ql[m]
                queue_arr[ei].append((rows, np.full(len(rows), p), t_end[queued][m]))
                allr = np.concatenate([a[0] for a in queue_arr[ei]])
                allp = np.concatenate([a[1] for a in queue_arr[ei]])
                allt = np.concatenate([a[2] for a in queue_arr[ei]])
                qo = run_queue(allt, trips.weight[allr], ents[ei].curb_spots, ents[ei].unload_s, tg)
                cur = (allp == p) & np.isin(allr, rows)
                rr = allr[cur]
                wp_wait[rr, p] = qo.wait_s[cur]
                wp_leave[rr, p] = qo.done_s[cur]
                vht += float(np.sum(trips.weight[rr] * np.clip(np.minimum(qo.done_s[cur], re_) - np.maximum(allt[cur], rs), 0, None)))
        nq = legs[~queued]
        wp_leave[nq, p] = leave[~queued]
        timings["queues"] += time.perf_counter() - t0
        done = legs[n_wp[legs] == p + 1]
        endhere = done[trips.final[done] < 0]
        arrive[endhere] = wp_leave[endhere, p]
        cur_node[legs] = wp_after[legs, p]
        cur_t[legs] = wp_leave[legs, p]

    # final queue stats per entrance (all arrivals of this loading)
    queues: list[QueueOutput] = []
    for ei, ent in enumerate(ents):
        if queue_arr[ei]:
            allr = np.concatenate([a[0] for a in queue_arr[ei]])
            allt = np.concatenate([a[2] for a in queue_arr[ei]])
            queues.append(run_queue(allt, trips.weight[allr], ent.curb_spots, ent.unload_s, tg))
        else:
            queues.append(run_queue(np.zeros(0), np.zeros(0), ent.curb_spots, ent.unload_s, tg))

    tw = trips.target_wp
    rows = np.arange(n)
    tw_c = np.clip(tw, 0, max(S - 1, 0))
    target_exp = np.full(n, np.nan)
    if S:
        reached = np.isfinite(trips.target_arr) & ok & (tw < n_wp)
        # experienced time until the car reaches the curb (includes the queue wait: parents learn it)
        target_exp[reached] = (wp_arrive[rows[reached], tw_c[reached]] + wp_wait[rows[reached], tw_c[reached]]) - depart[reached]
    tot_tt = float(np.nansum(trips.weight[ok] * (arrive[ok] - depart[ok])))
    return {
        "counts": counts, "depart": depart, "arrive": arrive, "ok": ok, "wp_arrive": wp_arrive, "wp_leave": wp_leave,
        "wp_wait": wp_wait, "wp_ff": wp_ff, "wp_entr": wp_entr, "queues": queues, "traj": _build_traj(traj_parts, trips),
        "vht": vht, "total_tt": tot_tt, "target_experienced": target_exp,
    }


def _route(router: _Router, orig: np.ndarray, dest: np.ndarray, t_leg: np.ndarray, d_roots: np.ndarray,
           o_roots: np.ndarray, rep: np.ndarray, tg: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    net = router.net
    n = len(orig)
    period = np.clip(((t_leg - tg.bin_start_s) // (router.period_bins * tg.bin_s)).astype(np.int64), 0, router.n_periods - 1)
    is_d = np.isin(dest, d_roots)
    is_o = ~is_d & np.isin(orig, o_roots)
    is_f = ~is_d & ~is_o
    same = orig == dest
    is_d &= ~same
    is_o &= ~same
    is_f &= ~same
    parts_rows: list[np.ndarray] = []
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    N = router.N
    max_steps = net.n_nodes

    def walk(rows_sel, kind, start, root, prd, forward, pred, best, best_row):
        return walk_to_root(pred.ravel(), N, rows_sel, start, root, net.graph, best, best_row, forward, max_steps)

    # destination-rooted
    idx = np.nonzero(is_d)[0]
    if len(idx):
        rows = router.ensure("D", dest[idx], period[idx])
        parts_rows.append(idx)
        parts.append(walk(rows, "D", orig[idx], dest[idx], period[idx], True, router.pred[: router.n_rows], router.best, period[idx]))
    idx = np.nonzero(is_o)[0]
    if len(idx):
        rows = router.ensure("O", orig[idx], period[idx])
        parts_rows.append(idx)
        parts.append(walk(rows, "O", dest[idx], orig[idx], period[idx], False, router.pred[: router.n_rows], router.best, period[idx]))
    idx = np.nonzero(is_f)[0]
    if len(idx):
        hub = rep[dest[idx]]
        rows = router.ensure("D", hub, period[idx])
        p1, l1, ok1 = walk(rows, "D", orig[idx], hub, period[idx], True, router.pred[: router.n_rows], router.best, period[idx])
        srows = net.static_tree_rows(hub)
        p2, l2, ok2 = walk_to_root(net._static_pred.ravel(), N, srows, dest[idx], hub, net.graph,
                                   net.static_best_edges(), np.zeros(len(idx), np.int64), False, max_steps)
        pc, lc = concat_paths([(p1, l1), (p2, l2)])
        parts_rows.append(idx)
        parts.append((pc, lc, ok1 & ok2))
    L = max([pp[0].shape[1] for pp in parts], default=0)
    paths = np.full((n, L), -1, dtype=np.int32)
    lens = np.zeros(n, dtype=np.int32)
    okv = np.ones(n, dtype=bool)
    for rows_i, (pp, ll, oo) in zip(parts_rows, parts, strict=True):
        if pp.shape[1]:
            paths[rows_i, : pp.shape[1]] = pp
        lens[rows_i] = ll
        okv[rows_i] = oo
    return paths, lens, okv


def _propagate(paths: np.ndarray, lens: np.ndarray, t_start: np.ndarray, w: np.ndarray, TTf: np.ndarray, tt0: np.ndarray,
               E: int, tg: Any, rs: float, re_: float, sample: np.ndarray, legs: np.ndarray, p: int):
    n = len(lens)
    order = np.argsort(-lens, kind="stable")
    P = paths[order]
    L = lens[order]
    t = t_start[order].astype(np.float64).copy()
    ww = w[order]
    ff = np.zeros(n)
    keys_list: list[np.ndarray] = []
    w_list: list[np.ndarray] = []
    vht = 0.0
    samp = sample[order]
    traj_cols: list[tuple[np.ndarray, ...]] = []
    cut = np.searchsorted(-L, -np.arange(1, P.shape[1] + 1), side="right") if P.shape[1] else np.zeros(0, int)
    for j in range(P.shape[1]):
        k = int(cut[j])  # legs with length > j
        if k == 0:
            break
        e = P[:k, j].astype(np.int64)
        tj = t[:k]
        b = np.floor((tj - tg.bin_start_s) / tg.bin_s).astype(np.int64)
        b_c = np.clip(b, 0, tg.n_bins - 1)
        dt = TTf[b_c * E + e]
        inwin = (b >= 0) & (b < tg.n_bins)
        keys_list.append((b_c * E + e)[inwin])
        w_list.append(ww[:k][inwin])
        t_exit = tj + dt
        vht += float(np.sum(ww[:k] * np.clip(np.minimum(t_exit, re_) - np.maximum(tj, rs), 0, None)))
        ff[:k] += tt0[e]
        sm = samp[:k]
        if sm.any():
            ii = np.nonzero(sm)[0]
            traj_cols.append((legs[order][ii], np.full(len(ii), p), np.full(len(ii), j), e[ii], tj[ii], t_exit[ii]))
        t[:k] = t_exit
    cnt = np.zeros(tg.n_bins * E)
    if keys_list:
        keys = np.concatenate(keys_list)
        cnt = np.bincount(keys, weights=np.concatenate(w_list), minlength=tg.n_bins * E)
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    tp = None
    if traj_cols:
        tp = tuple(np.concatenate([c[i] for c in traj_cols]) for i in range(6))
    return t[inv], ff[inv], vht, cnt, tp


def _build_traj(parts: list[tuple[np.ndarray, ...]], trips: Trips) -> dict[str, np.ndarray]:
    if not parts:
        return {"offsets": np.zeros(1, np.uint32), "kind": np.zeros(0, np.uint32), "edge": np.zeros(0, np.int32),
                "enter": np.zeros(0, np.float32), "exit": np.zeros(0, np.float32), "trip": np.zeros(0, np.int64)}
    trip, leg, step, edge, ent, ext = (np.concatenate([p[i] for p in parts]) for i in range(6))
    order = np.lexsort((step, leg, trip))
    trip, edge, ent, ext = trip[order], edge[order], ent[order], ext[order]
    uniq, start = np.unique(trip, return_index=True)
    offsets = np.r_[start, len(trip)].astype(np.uint32)
    return {"offsets": offsets, "kind": trips.kind[uniq].astype(np.uint32), "edge": edge.astype(np.int32),
            "enter": ent.astype(np.float32), "exit": ext.astype(np.float32), "trip": uniq}
