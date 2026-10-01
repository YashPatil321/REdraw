"""One simulation run (one seed) and its per-seed metrics.

``run_seed(world, seed, base_world=None)``: if ``base_world`` is given this is a
plan run and mode choice gets a habit term toward the mode the same person
chose in the baseline with the same seed (common random numbers).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.assignment import AssignmentBackend, AssignmentResult, IncrementalMSABackend
from sim.assumptions import Af
from sim.demand import ROLE_COMMUTE, ROLE_COMMUTE_CHAIN, ROLE_DROPOFF_HOME, Demand, build_demand
from sim.modechoice import S_IDX, choose_modes, make_draws
from sim.world import ALL_MODES, WorldState

DEFAULT_BACKEND: AssignmentBackend = IncrementalMSABackend()


@dataclass
class SeedResult:
    seed: int
    metrics: dict[str, float]
    per_school: dict[str, dict[str, Any]]
    mode_counts: dict[str, float]
    commute_min: np.ndarray  # per person (NaN if not commuting)
    dropoff_min: np.ndarray  # per person: drop-off leg time for the household's driver
    kid_min: np.ndarray  # per person: student's own trip time
    total_min: np.ndarray  # per person: all minutes attributed (NaN if no trip)
    mode: np.ndarray  # per person: index into ALL_MODES, -1 none
    edge_max_vc: np.ndarray  # (E,) max v/c over report bins
    edge_max_bin: np.ndarray  # (E,) bin index of the max
    dropoff_arr_hist: np.ndarray  # (n_bins,) drop-off vehicle arrivals at entrances per bin
    commute_onroad_hist: np.ndarray  # (n_bins,) commuter vehicles on the road per bin
    route_tt: np.ndarray | None = None  # (B, E) edge times, kept only when requested (calibration)
    playback: dict[str, Any] | None = None
    timings: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


def run_seed(world: WorldState, seed: int, base_world: WorldState | None = None, *, keep_playback: bool = False,
             keep_tt: bool = False, backend: AssignmentBackend | None = None) -> SeedResult:
    t0 = time.perf_counter()
    draws = make_draws(world, seed)
    habit = None
    if base_world is not None:
        habit = choose_modes(base_world, make_draws(base_world, seed))
    modes = choose_modes(world, draws, habit)
    demand = build_demand(world, draws, modes)
    t1 = time.perf_counter()
    res = (backend or DEFAULT_BACKEND).assign(world, demand.trips, traj_rng=draws.rng_traj,
                                              max_traj=None if keep_playback else 0)
    t2 = time.perf_counter()
    out = compute_seed_metrics(world, demand, res, seed)
    if keep_playback:
        out.playback = {
            "edge_vc": res.vc, "edge_speed_kph": res.speed_kph,
            "queue_len": np.stack([e.queue_end for e in res.entrances], axis=1) if res.entrances else np.zeros((world.time.n_bins, 0)),
            "entrance_ids": [e.key for e in res.entrances], "traj": res.traj,
        }
    if keep_tt:
        out.route_tt = res.tt
    out.timings = {"demand_s": t1 - t0, "assignment_s": t2 - t1, "metrics_s": time.perf_counter() - t2,
                   "total_s": time.perf_counter() - t0, **{f"asg_{k}": v for k, v in res.timings.items()}}
    out.info = {"iterations": res.iterations, "rel_change": res.rel_change, "failed_trips": res.failed_trips,
                "n_trips": int(demand.trips.n), "demand_factor": draws.demand_factor}
    return out


def compute_seed_metrics(world: WorldState, d: Demand, r: AssignmentResult, seed: int) -> SeedResult:
    tg = world.time
    trips = d.trips
    P = len(world.persons)
    walk_curb = Af("schools.walk_from_curb_min") * 60.0
    late_tol = Af("schools.late_tolerance_min") * 60.0
    n_wp = (trips.wp_node >= 0).sum(axis=1)
    dur = (r.arrive - r.depart) / 60.0

    # ---- workers' commute ----
    commute = np.full(P, np.nan)
    wt = d.worker_trip
    has = wt >= 0
    commute[has] = dur[wt[has]]
    wi, wfix = d.modes.worker_idx, d.modes.worker_fixed_min
    fx = np.isfinite(wfix)
    commute[wi[fx]] = wfix[fx]

    # ---- drop-off drivers ----
    dropoff = np.full(P, np.nan)
    total_extra = np.zeros(P)
    drv_rows = np.nonzero((trips.driver >= 0) & np.isin(trips.role, [ROLE_COMMUTE_CHAIN, ROLE_DROPOFF_HOME]) & r.ok)[0]
    if len(drv_rows):
        last_wp = n_wp[drv_rows] - 1
        dd = (r.wp_leave[drv_rows, last_wp] - r.depart[drv_rows]) / 60.0
        dropoff[trips.driver[drv_rows]] = dd
        home_rows = drv_rows[trips.role[drv_rows] == ROLE_DROPOFF_HOME]
        np.add.at(total_extra, trips.driver[home_rows], dur[home_rows])

    # ---- students ----
    kid = np.full(P, np.nan)
    si = d.modes.student_idx
    late = np.zeros(P, dtype=bool)
    kt = d.kid_trip[si]
    kw = d.kid_wp[si]
    veh = kt >= 0
    arr_class = np.full(len(si), np.nan)
    if veh.any():
        rows = kt[veh]
        cols = kw[veh]
        okk = r.ok[rows]
        leave = r.wp_leave[rows, cols]
        arrive_wp = r.wp_arrive[rows, cols]
        queued = trips.wp_queue[rows, cols]
        at_class = np.where(queued, leave, arrive_wp) + walk_curb
        is_shuttle = trips.kind[rows] == 2
        start = np.where(is_shuttle, np.nan, r.depart[rows])
        tmin = (at_class - start) / 60.0
        shuttle_est = d.modes.student_fixed_min[veh]
        tmin = np.where(is_shuttle, shuttle_est, tmin)
        arr_class[veh] = np.where(okk, at_class, np.nan)
        kid[si[veh]] = np.where(okk, tmin, np.nan)
    sfix = d.modes.student_fixed_min
    nonveh = ~veh & np.isfinite(sfix)
    kid[si[nonveh]] = sfix[nonveh]
    bell = np.array([s.bell_s for s in world.schools], dtype=np.float64)
    sch = world.persons.school[si]
    late_s = np.isfinite(arr_class) & (arr_class > bell[sch] + late_tol)
    late[si] = late_s

    total = np.where(np.isfinite(commute), commute, 0.0) + np.where(np.isfinite(kid), kid, 0.0) + total_extra
    any_trip = np.isfinite(commute) | np.isfinite(kid) | (total_extra > 0)
    total = np.where(any_trip, total, np.nan)

    # ---- per entrance / school ----
    per_school: dict[str, dict[str, Any]] = {}
    drop_rows, drop_cols = np.nonzero(trips.wp_queue & (r.wp_entr >= 0) & r.ok[:, None])
    depart_leg = np.where(drop_cols == 0, r.depart[drop_rows], r.wp_leave[drop_rows, np.maximum(drop_cols - 1, 0)])
    delay = (r.wp_arrive[drop_rows, drop_cols] - depart_leg - r.wp_ff[drop_rows, drop_cols]) + r.wp_wait[drop_rows, drop_cols]
    delay = np.maximum(delay, 0.0) / 60.0
    dw = trips.weight[drop_rows]
    d_school = trips.wp_school[drop_rows, drop_cols]
    for s_i, school in enumerate(world.schools):
        m = d_school == s_i
        ents = []
        for ei in school.entrances:
            st = r.entrances[ei]
            tot = st.arrivals.sum()
            ents.append({
                "key": st.key, "max_queue_cars": float(st.queue_max.max()), "max_spillback_m": float(st.spill_m.max()),
                "avg_wait_min": float(np.nansum(st.avg_wait_s * st.arrivals) / tot / 60.0) if tot > 0 else 0.0,
                "arrivals": float(tot),
            })
        in_school = sch == s_i
        per_school[school.id] = {
            "dropoff_delay_min": float(np.sum(delay[m] * dw[m]) / np.sum(dw[m])) if m.any() else 0.0,
            "max_spillback_m": max((e["max_spillback_m"] for e in ents), default=0.0),
            "max_queue_cars": max((e["max_queue_cars"] for e in ents), default=0.0),
            "avg_wait_min": float(np.mean([e["avg_wait_min"] for e in ents])) if ents else 0.0,
            "late_kids": float(late_s[in_school].sum()),
            "students": int(in_school.sum()),
            "dropoff_share": float((d.modes.student_mode[in_school] == S_IDX["drive_dropoff"]).mean()) if in_school.any() else 0.0,
            "entrances": ents,
        }

    # ---- histograms for peak overlap ----
    B = tg.n_bins
    drop_arr_hist = np.bincount(tg.bin_of(r.wp_arrive[drop_rows, drop_cols]), weights=dw, minlength=B).astype(np.float64)
    commuter = np.isin(trips.role, [ROLE_COMMUTE, ROLE_COMMUTE_CHAIN]) & r.ok
    cr = np.nonzero(commuter)[0]
    onroad = np.zeros(B + 1)
    if len(cr):
        b0 = tg.bin_of(r.depart[cr])
        b1 = tg.bin_of(r.arrive[cr])
        np.add.at(onroad, b0, trips.weight[cr])
        np.add.at(onroad, b1 + 1, -trips.weight[cr])
    onroad = np.cumsum(onroad)[:B]

    # ---- edges ----
    rb0, rb1 = tg.report_bins()
    vc_rep = r.vc[rb0:rb1]
    emax_bin = np.argmax(vc_rep, axis=0).astype(np.int16) + rb0
    emax = vc_rep.max(axis=0).astype(np.float32)

    # ---- metrics ----
    mode_counts = {m: 0.0 for m in ALL_MODES}
    pm = d.person_mode
    for i, m in enumerate(ALL_MODES):
        mode_counts[m] = float((pm == i).sum())
    workers_c = np.isfinite(commute)
    tot_w = float(np.sum(dw))
    metrics = {
        "avg_commute_min": float(np.nanmean(commute[workers_c])) if workers_c.any() else 0.0,
        "avg_dropoff_delay_min": float(np.sum(delay * dw) / tot_w) if tot_w > 0 else 0.0,
        "max_spillback_m": max((v["max_spillback_m"] for v in per_school.values()), default=0.0),
        "total_vht": float(r.vht_report_h),
        "late_kids": float(late_s.sum()),
    }
    return SeedResult(
        seed=seed, metrics=metrics, per_school=per_school, mode_counts=mode_counts,
        commute_min=commute.astype(np.float32), dropoff_min=dropoff.astype(np.float32), kid_min=kid.astype(np.float32),
        total_min=total.astype(np.float32), mode=pm.astype(np.int8), edge_max_vc=emax, edge_max_bin=emax_bin,
        dropoff_arr_hist=drop_arr_hist, commute_onroad_hist=onroad,
    )


def peak_window(commute_onroad_hist: np.ndarray, bin_s: int) -> tuple[int, int]:
    """[b0, b1) bins of the commute peak hour (max commuter vehicles on the road)."""
    w = max(1, int(round(Af("sim_engine.peak_hour_minutes") * 60 / bin_s)))
    h = np.asarray(commute_onroad_hist, dtype=np.float64)
    if len(h) <= w:
        return 0, len(h)
    cs = np.r_[0.0, np.cumsum(h)]
    sums = cs[w:] - cs[:-w]
    b0 = int(np.argmax(sums))
    return b0, b0 + w


def peak_overlap(dropoff_arr_hist: np.ndarray, window: tuple[int, int]) -> float:
    tot = float(np.sum(dropoff_arr_hist))
    if tot <= 0:
        return 0.0
    return float(np.sum(dropoff_arr_hist[window[0] : window[1]]) / tot)

