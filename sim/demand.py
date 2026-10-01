"""Trip generation (spec 6.2).

Builds the morning's vehicle trips from persons + chosen modes:

1. Workers: home -> work (or exit node). Habitual departure from the
   assumptions' normal distribution plus a per-seed day jitter.
2. Students: home -> school, arriving 5-20 minutes before the bell.
3. Parent drop-off chains (required): one vehicle per household visits every
   school its driven kids attend (earliest bell first), then continues to the
   driver's work (if a commuting driver, with probability
   ``dropoff_parent_continues_to_work_share``) or returns home. The driver's
   separate commute trip is removed.
4. High-school students may drive themselves (``teen_drive``).
5. Carpool kids share a car (weight 1 / kids_per_car), worker carpools share
   by occupancy, school bus / shuttle buses are network vehicles with bus PCE.
6. Inbound workers (exit -> in-region job) and freeway through traffic
   (exit -> exit) from assumptions.

Monte Carlo (spec 6.6): the seed's demand factor scales every vehicle weight
(+/- 10 percent), departures get a day jitter, and mode choice uses the seed's
Gumbel draws.

Extension hooks:
- ``world.demand_overrides``: callables ``fn(world, draws, modes, trips) -> trips``
  applied last (spec 13.4 real households: opt-in trip data replaces synthetic trips).
- ``baseline_mode_hint`` in persons.parquet forces a mode (see modechoice).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.assumptions import A, Af, clock_to_s
from sim.modechoice import S_IDX, W_IDX, Draws, ModeResult
from sim.world import WorldState

KIND_CAR, KIND_DROPOFF, KIND_SHUTTLE, KIND_BUS, KIND_CARPOOL = 0, 1, 2, 3, 4
KINDS = {"0": "car", "1": "dropoff_car", "2": "shuttle", "3": "school_bus", "4": "carpool"}

ROLE_NONE, ROLE_COMMUTE, ROLE_DROPOFF_HOME, ROLE_TEEN, ROLE_COMMUTE_CHAIN = 0, 1, 2, 3, 4


@dataclass
class Trips:
    kind: np.ndarray
    weight: np.ndarray
    origin: np.ndarray
    depart: np.ndarray  # planned departure (s); NaN when derived from target arrival
    target_arr: np.ndarray  # target arrival at the first waypoint (s) or NaN
    exp_time: np.ndarray  # initial expected time to the first waypoint (s)
    wp_node: np.ndarray  # (n, S) node idx, -1 pad (entrance waypoints are resolved each iteration)
    wp_school: np.ndarray  # (n, S) school idx for entrance waypoints, -1 otherwise
    wp_queue: np.ndarray  # (n, S) bool: queue at the drop-off curb
    wp_dwell: np.ndarray  # (n, S) seconds of dwell at non-queue waypoints
    final: np.ndarray  # final destination node idx or -1 (trip ends at last waypoint)
    driver: np.ndarray  # person idx or -1
    role: np.ndarray
    run_id: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))  # shuttle run id or -1
    target_wp: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))  # waypoint the target arrival refers to

    @property
    def n(self) -> int:
        return len(self.kind)

    @property
    def n_wp(self) -> np.ndarray:
        return (self.wp_node >= 0).sum(axis=1) if self.wp_node.size else np.zeros(self.n, np.int64)


@dataclass
class Demand:
    trips: Trips
    modes: ModeResult
    draws: Draws
    kid_trip: np.ndarray  # per person: trip idx carrying the student (-1 none)
    kid_wp: np.ndarray  # per person: waypoint index of the school in that trip
    worker_trip: np.ndarray  # per person: trip idx that is this worker's commute (-1)
    person_mode: np.ndarray  # per person: index into world.ALL_MODES, -1 if no trip
    student_target: np.ndarray  # per person: target arrival at school (s), NaN otherwise
    worker_depart: np.ndarray  # per person: commute departure (s), NaN otherwise


class _TripBuilder:
    def __init__(self, S: int) -> None:
        self.S = S
        self.cols: dict[str, list] = {k: [] for k in (
            "kind", "weight", "origin", "depart", "target_arr", "exp_time", "final", "driver", "role", "run_id", "target_wp")}
        self.wp_node: list[np.ndarray] = []
        self.wp_school: list[np.ndarray] = []
        self.wp_queue: list[np.ndarray] = []
        self.wp_dwell: list[np.ndarray] = []
        self.n = 0

    def add(self, n: int, *, kind, weight, origin, final, depart=np.nan, target_arr=np.nan, exp_time=0.0,
            driver=-1, role=ROLE_NONE, wp_node=None, wp_school=None, wp_queue=None, wp_dwell=None, run_id=-1, target_wp=0) -> np.ndarray:
        if n == 0:
            return np.zeros(0, np.int64)
        def full(v, dt):
            a = np.asarray(v, dtype=dt)
            return np.broadcast_to(a, (n,)).copy() if a.ndim == 0 else a.astype(dt)
        self.cols["kind"].append(full(kind, np.int8))
        self.cols["weight"].append(full(weight, np.float64))
        self.cols["origin"].append(full(origin, np.int64))
        self.cols["final"].append(full(final, np.int64))
        self.cols["depart"].append(full(depart, np.float64))
        self.cols["target_arr"].append(full(target_arr, np.float64))
        self.cols["exp_time"].append(full(exp_time, np.float64))
        self.cols["driver"].append(full(driver, np.int64))
        self.cols["role"].append(full(role, np.int8))
        self.cols["run_id"].append(full(run_id, np.int32))
        self.cols["target_wp"].append(full(target_wp, np.int32))
        S = self.S
        def mat(v, fill, dt):
            m = np.full((n, S), fill, dtype=dt)
            if v is not None:
                v = np.asarray(v, dtype=dt)
                if v.ndim == 1:
                    v = v[:, None]
                m[:, : v.shape[1]] = v
            return m
        self.wp_node.append(mat(wp_node, -1, np.int64))
        self.wp_school.append(mat(wp_school, -1, np.int32))
        self.wp_queue.append(mat(wp_queue, False, bool))
        self.wp_dwell.append(mat(wp_dwell, 0.0, np.float32))
        idx = np.arange(self.n, self.n + n)
        self.n += n
        return idx

    def build(self) -> Trips:
        def cat(k, dt):
            return np.concatenate(self.cols[k]) if self.cols[k] else np.zeros(0, dt)
        S = self.S
        def catm(lst, dt):
            return np.concatenate(lst) if lst else np.zeros((0, S), dt)
        return Trips(
            kind=cat("kind", np.int8), weight=cat("weight", np.float64), origin=cat("origin", np.int64),
            depart=cat("depart", np.float64), target_arr=cat("target_arr", np.float64), exp_time=cat("exp_time", np.float64),
            wp_node=catm(self.wp_node, np.int64), wp_school=catm(self.wp_school, np.int32),
            wp_queue=catm(self.wp_queue, bool), wp_dwell=catm(self.wp_dwell, np.float32),
            final=cat("final", np.int64), driver=cat("driver", np.int64), role=cat("role", np.int8), run_id=cat("run_id", np.int32),
            target_wp=cat("target_wp", np.int32),
        )


def _clip_departure(world: WorldState, t: np.ndarray) -> np.ndarray:
    tg = world.time
    return np.clip(t, tg.bin_start_s, tg.end_s - tg.bin_s)


def build_demand(world: WorldState, draws: Draws, modes: ModeResult) -> Demand:
    from sim.modechoice import school_car_time_s
    from sim.world import ALL_MODES

    p, hh, net = world.persons, world.households, world.net
    P = len(p)
    s = draws.demand_factor
    jit = Af("time_of_day.monte_carlo_departure_jitter_min") * 60.0
    dep_mean = clock_to_s(A("time_of_day.worker_departure_mean"))
    dep_sd = world.knobs["departure_sd_min"] * 60.0
    b_lo, b_hi = (float(v) for v in A("time_of_day.student_arrive_before_bell_min"))
    max_stops = int(A("sim_engine.max_school_stops_per_chain"))
    S = max(max_stops, 1)
    for sh in world.shuttles:
        S = max(S, len(sh.stop_nodes))
    tb = _TripBuilder(S)
    bell = np.array([sc.bell_s for sc in world.schools], dtype=np.float64)
    car_to_school = school_car_time_s(world)

    person_mode = np.full(P, -1, dtype=np.int16)
    mode_id = {m: i for i, m in enumerate(ALL_MODES)}
    kid_trip = np.full(P, -1, np.int64)
    kid_wp = np.zeros(P, np.int64)
    worker_trip = np.full(P, -1, np.int64)
    student_target = np.full(P, np.nan)
    worker_depart = np.full(P, np.nan)

    # ---- students: targets ----
    si, sm = modes.student_idx, modes.student_mode
    st_school = p.school[si]
    # day jitter moves the planned arrival but never past the earliest planned buffer, so
    # lateness comes from traffic and queues, not from the jitter itself
    target = np.minimum(bell[st_school] - (b_lo + draws.buffer_u[si] * (b_hi - b_lo)) * 60.0 + jit * draws.jitter_z[si],
                        bell[st_school] - b_lo * 60.0)
    student_target[si] = target
    smode_name = np.array(["drive_dropoff", "carpool", "school_bus", "school_shuttle", "walk", "bike", "teen_drive"])[sm]
    person_mode[si] = np.array([mode_id[m] for m in smode_name], dtype=np.int16) if len(si) else person_mode[si]
    st_home = hh.home_node[p.hh[si]]

    # ---- workers: departures ----
    wi, wm = modes.worker_idx, modes.worker_mode
    wdep = _clip_departure(world, dep_mean + dep_sd * draws.dep_habit_z[wi] + jit * draws.jitter_z[wi])
    worker_depart[wi] = wdep
    wname = np.array(["drive_alone", "carpool", "bike", "walk"])[wm]
    person_mode[wi] = np.array([mode_id[m] for m in wname], dtype=np.int16) if len(wi) else person_mode[wi]

    # ---- drop-off chains ----
    drop = sm == S_IDX["drive_dropoff"]
    d_si = si[drop]
    d_hh = p.hh[d_si]
    chain_driver_of_hh: dict[int, int] = {}
    if len(d_si):
        # commuting drivers available per household (drive_alone workers), first one wins
        drv_w = wi[wm == W_IDX["drive_alone"]]
        hh_of_drv = p.hh[drv_w]
        order = np.argsort(hh_of_drv, kind="stable")
        uniq_h, first = np.unique(hh_of_drv[order], return_index=True)
        drv_by_hh = dict(zip(uniq_h.tolist(), drv_w[order][first].tolist(), strict=True))
        adults = np.nonzero(p.age >= 18)[0]
        a_order = np.argsort(p.hh[adults], kind="stable")
        ua, fa = np.unique(p.hh[adults][a_order], return_index=True)
        adult_by_hh = dict(zip(ua.tolist(), adults[a_order][fa].tolist(), strict=True))
        cont_share = Af("demand.dropoff_parent_continues_to_work_share")
        order = np.lexsort((bell[p.school[d_si]], d_hh))
        d_si_o = d_si[order]
        h_o = p.hh[d_si_o]
        starts = np.flatnonzero(np.r_[True, h_o[1:] != h_o[:-1]])
        ends = np.r_[starts[1:], len(d_si_o)]
        tgt_by_person = dict(zip(si.tolist(), target.tolist(), strict=True))
        rows_origin, rows_final, rows_target, rows_exp, rows_driver, rows_role = [], [], [], [], [], []
        rows_wp, rows_kids = [], []
        for a, b in zip(starts.tolist(), ends.tolist(), strict=True):
            h = int(h_o[a])
            kids = d_si_o[a:b]
            ksch = p.school[kids]
            uniq_s = list(dict.fromkeys(ksch.tolist()))  # already sorted by bell
            chunks = [uniq_s[i : i + max_stops] for i in range(0, len(uniq_s), max_stops)]
            for ci, chunk in enumerate(chunks):
                home = int(hh.home_node[h])
                drv = drv_by_hh.get(h, -1) if ci == 0 else -1
                if drv >= 0 and draws.hh_continue_u[h] < cont_share:
                    final, role = int(p.work_node[drv]), ROLE_COMMUTE_CHAIN
                    chain_driver_of_hh[h] = drv
                else:
                    final, role = home, ROLE_DROPOFF_HOME
                    drv = adult_by_hh.get(h, -1) if ci == 0 else -1
                first_kids = kids[ksch == chunk[0]]
                rows_origin.append(home)
                rows_final.append(final)
                rows_target.append(min(tgt_by_person[int(k)] for k in first_kids))
                rows_exp.append(float(car_to_school[chunk[0], home]))
                rows_driver.append(drv)
                rows_role.append(role)
                rows_wp.append(chunk)
                rows_kids.append([(int(k), chunk.index(int(p.school[k]))) for k in kids if int(p.school[k]) in chunk])
        n = len(rows_origin)
        wps = np.full((n, S), -1, np.int32)
        for r, chunk in enumerate(rows_wp):
            wps[r, : len(chunk)] = chunk
        wp_node = np.where(wps >= 0, 0, -1)  # resolved to entrance nodes per iteration
        idx = tb.add(n, kind=KIND_DROPOFF, weight=s, origin=np.array(rows_origin), final=np.array(rows_final),
                     target_arr=np.array(rows_target), exp_time=np.array(rows_exp), driver=np.array(rows_driver),
                     role=np.array(rows_role), wp_node=wp_node, wp_school=wps, wp_queue=wps >= 0)
        for r, kl in enumerate(rows_kids):
            for k, w in kl:
                kid_trip[k] = idx[r]
                kid_wp[k] = w
        for r in range(n):
            if rows_role[r] == ROLE_COMMUTE_CHAIN:
                worker_trip[rows_driver[r]] = idx[r]
                person_mode[rows_driver[r]] = mode_id["drive_dropoff"]

    # ---- worker commute trips (not already chained) ----
    car_w = np.isin(wm, [W_IDX["drive_alone"], W_IDX["carpool"]])
    chained = np.isin(wi, np.array(list(chain_driver_of_hh.values()), dtype=np.int64))
    cw = car_w & ~chained
    occ = Af("modechoice.carpool_worker_occupancy")
    is_cp = wm[cw] == W_IDX["carpool"]
    idx = tb.add(int(cw.sum()), kind=np.where(is_cp, KIND_CARPOOL, KIND_CAR), weight=np.where(is_cp, s / occ, s),
                 origin=hh.home_node[p.hh[wi[cw]]], final=p.work_node[wi[cw]], depart=wdep[cw], driver=wi[cw], role=ROLE_COMMUTE)
    worker_trip[wi[cw]] = idx

    # ---- carpool kids ----
    cm = sm == S_IDX["carpool"]
    idx = tb.add(int(cm.sum()), kind=KIND_CARPOOL, weight=s / Af("modechoice.carpool_kids_per_car"),
                 origin=st_home[cm], final=st_home[cm], target_arr=target[cm],
                 exp_time=car_to_school[st_school[cm], st_home[cm]], wp_node=np.zeros(int(cm.sum())),
                 wp_school=st_school[cm], wp_queue=np.ones(int(cm.sum()), bool))
    kid_trip[si[cm]] = idx

    # ---- teen drivers (park, no curb queue) ----
    tm = sm == S_IDX["teen_drive"]
    idx = tb.add(int(tm.sum()), kind=KIND_CAR, weight=s, origin=st_home[tm], final=-1, target_arr=target[tm],
                 exp_time=car_to_school[st_school[tm], st_home[tm]], driver=si[tm], role=ROLE_TEEN,
                 wp_node=np.zeros(int(tm.sum())), wp_school=st_school[tm])
    kid_trip[si[tm]] = idx

    # ---- district school bus riders (fractional bus vehicles) ----
    bm = sm == S_IDX["school_bus"]
    bus_w = s * Af("sim_engine.bus_pce") / Af("modechoice.shuttle_capacity")
    idx = tb.add(int(bm.sum()), kind=KIND_BUS, weight=bus_w, origin=st_home[bm], final=-1, target_arr=target[bm],
                 exp_time=car_to_school[st_school[bm], st_home[bm]], wp_node=np.zeros(int(bm.sum())), wp_school=st_school[bm])
    kid_trip[si[bm]] = idx

    # ---- shuttle buses ----
    dwell = Af("sim_engine.shuttle_stop_dwell_s")
    bus_kph = Af("modechoice.speeds_kph.bus_avg")
    shm = sm == S_IDX["school_shuttle"]
    for j, (sh, plan) in enumerate(zip(world.shuttles, modes.shuttle_plans, strict=True)):
        order = plan.order
        nodes = [sh.stop_nodes[k] for k in order]
        n_runs = plan.n_runs
        wpn = np.full((n_runs, S), -1, np.int64)
        wps = np.full((n_runs, S), -1, np.int32)
        wpd = np.zeros((n_runs, S), np.float32)
        rest = nodes[1:]
        wpn[:, : len(rest)] = rest
        wpd[:, : len(rest)] = dwell
        wpn[:, len(rest)] = 0
        wps[:, len(rest)] = sh.school
        idx = tb.add(n_runs, kind=KIND_SHUTTLE, weight=Af("sim_engine.bus_pce"), origin=nodes[0], final=-1,
                     target_arr=plan.run_arrivals, exp_time=plan.route_km / bus_kph * 3600.0,
                     wp_node=wpn, wp_school=wps, wp_dwell=wpd, run_id=np.arange(n_runs), target_wp=len(rest))
        riders = si[shm & (world.shuttle_of_person[si] == j)]
        if len(riders):
            ro = riders[np.argsort(student_target[riders], kind="stable")]
            run_of = (np.arange(len(ro)) * n_runs) // max(len(ro), 1)
            kid_trip[ro] = idx[run_of]
            kid_wp[ro] = len(rest)

    # ---- inbound workers and through traffic ----
    rng = draws.rng_external
    if world.exits:
        ex_nodes = np.array([e.node for e in world.exits], dtype=np.int64)
        nocom = Af("demand.worker_not_commuting_today_share")
        n_in = int(round(Af("demand.inbound_workers") * (1 - Af("demand.work_from_home_share")) * (1 - nocom)))
        internal = p.work_node[p.is_worker & ~p.work_external & (p.work_node >= 0)]
        if len(internal) == 0:
            internal = np.unique(world.entrance_nodes())
        o = ex_nodes[rng.integers(0, len(ex_nodes), n_in)]
        d = internal[rng.integers(0, len(internal), n_in)]
        dep = _clip_departure(world, dep_mean + dep_sd * rng.standard_normal(n_in))
        tb.add(n_in, kind=KIND_CAR, weight=s, origin=o, final=d, depart=dep)
        hours = (world.time.end_s - world.time.bin_start_s) / 3600.0
        n_th = int(round(Af("demand.external_through_trips_per_hour") * hours))
        if len(ex_nodes) >= 2 and n_th > 0:
            bear = np.array([e.bearing_deg for e in world.exits])
            diff = np.abs(((bear[:, None] - bear[None, :]) + 180) % 360 - 180)
            pairs = np.argwhere(diff >= 90)
            if len(pairs) == 0:
                pairs = np.argwhere(~np.eye(len(ex_nodes), dtype=bool))
            pk = pairs[rng.integers(0, len(pairs), n_th)]
            dep = _clip_departure(world, dep_mean + dep_sd * rng.standard_normal(n_th))
            tb.add(n_th, kind=KIND_CAR, weight=s, origin=ex_nodes[pk[:, 0]], final=ex_nodes[pk[:, 1]], depart=dep)

    trips = tb.build()
    dem = Demand(trips=trips, modes=modes, draws=draws, kid_trip=kid_trip, kid_wp=kid_wp, worker_trip=worker_trip,
                 person_mode=person_mode, student_target=student_target, worker_depart=worker_depart)
    for fn in world.demand_overrides:
        dem = fn(world, draws, modes, dem)
    return dem
