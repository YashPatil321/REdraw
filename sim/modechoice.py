"""Mode choice (spec 6.3): multinomial logit with common random numbers.

Utility ``U_m = ASC_m + b_time * time_min + b_cost * cost_usd + b_habit * [m == baseline mode]
+ tool shifts + Gumbel draw``. The Gumbel draws come from the seed's random
stream and are identical for baseline and plan runs (common random numbers),
so a plan only flips people whose utilities actually changed. The habit term
applies in plan runs, toward the mode the same person chose in the baseline
with the same seed.

Walk and bike are only available under the distance limits by age in
assumptions.yaml. Tools change availability (shuttle), utilities (carpool,
bike/walk routes) and caps (carpool adoption cap, teen parking permits).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.assumptions import A, Af
from sim.world import STUDENT_MODES, WORKER_MODES, Shuttle, WorldState

S_IDX = {m: i for i, m in enumerate(STUDENT_MODES)}
W_IDX = {m: i for i, m in enumerate(WORKER_MODES)}


@dataclass
class Draws:
    """All random numbers for one seed. Habit draws are fixed per person across seeds."""

    seed: int
    demand_factor: float
    dep_habit_z: np.ndarray
    buffer_u: np.ndarray
    selfdrive_u: np.ndarray
    bus_u: np.ndarray
    hh_continue_u: np.ndarray
    jitter_z: np.ndarray
    nocommute_u: np.ndarray
    gumbel_student: np.ndarray
    gumbel_worker: np.ndarray
    rng_external: np.random.Generator
    rng_traj: np.random.Generator


def make_draws(world: WorldState, seed: int) -> Draws:
    P, H = len(world.persons), len(world.households)
    habit = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(A("population.seed")), 1])))
    root = np.random.SeedSequence([int(A("sim_engine.seed_base")), int(seed)])
    s_fac, s_jit, s_noc, s_gs, s_gw, s_ext, s_traj = (np.random.Generator(np.random.PCG64(c)) for c in root.spawn(7))
    var = Af("demand.monte_carlo_demand_variation")
    return Draws(
        seed=int(seed),
        demand_factor=float(world.knobs["demand_scale"] * (1.0 + s_fac.uniform(-var, var))),
        dep_habit_z=habit.standard_normal(P),
        buffer_u=habit.random(P),
        selfdrive_u=habit.random(P),
        bus_u=habit.random(P),
        hh_continue_u=habit.random(H),
        jitter_z=s_jit.standard_normal(P),
        nocommute_u=s_noc.random(P),
        gumbel_student=s_gs.gumbel(size=(P, len(STUDENT_MODES))).astype(np.float32),
        gumbel_worker=s_gw.gumbel(size=(P, len(WORKER_MODES))).astype(np.float32),
        rng_external=s_ext,
        rng_traj=s_traj,
    )


# ---------------------------------------------------------------------------
# shuttles
# ---------------------------------------------------------------------------

@dataclass
class ShuttlePlan:
    order: list[int]  # stop indices in route order (first stop first)
    ride_km: np.ndarray  # per stop (original index): distance riding to school
    route_km: float
    n_runs: int
    capacity: float
    run_arrivals: np.ndarray  # absolute seconds at school


def shuttle_plan(world: WorldState, sh: Shuttle) -> ShuttlePlan:
    school = world.schools[sh.school]
    circ = Af("sim_engine.circuity_factor")
    sx, sz = school.x, school.z
    xs, zs = np.array(sh.stop_x), np.array(sh.stop_z)
    n = len(xs)
    remaining = list(range(n))
    d_school = np.hypot(xs - sx, zs - sz)
    cur = int(np.argmax(d_school))
    order = [cur]
    remaining.remove(cur)
    while remaining:
        d = [np.hypot(xs[r] - xs[cur], zs[r] - zs[cur]) for r in remaining]
        cur = remaining[int(np.argmin(d))]
        order.append(cur)
        remaining.remove(cur)
    ride_km = np.zeros(n)
    acc = 0.0
    nxt_x, nxt_z = sx, sz
    for k in reversed(order):
        acc += np.hypot(xs[k] - nxt_x, zs[k] - nxt_z) / 1000.0 * circ
        ride_km[k] = acc
        nxt_x, nxt_z = xs[k], zs[k]
    route_km = float(acc)
    bus_kph = Af("modechoice.speeds_kph.bus_avg")
    window = Af("sim_engine.shuttle_service_window_min")
    cycle = 2.0 * route_km / bus_kph * 60.0 + n * Af("sim_engine.shuttle_stop_dwell_s") / 60.0
    runs_by_headway = max(1, int(window // max(sh.headway_min, 1)))
    runs_per_bus = max(1, int(window // max(cycle, 1e-6)))
    n_runs = max(1, min(runs_by_headway, int(sh.buses) * runs_per_bus))
    last = school.bell_s - Af("sim_engine.shuttle_arrive_before_bell_min") * 60.0
    arrivals = last - np.arange(n_runs)[::-1] * sh.headway_min * 60.0
    return ShuttlePlan(order=order, ride_km=ride_km, route_km=route_km, n_runs=n_runs,
                       capacity=float(n_runs * Af("modechoice.shuttle_capacity")), run_arrivals=arrivals)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _age_limits(age: np.ndarray, kind: str) -> np.ndarray:
    out = np.full(len(age), Af(f"modechoice.{kind}.adult"))
    out[age <= 18] = Af(f"modechoice.{kind}.14-18")
    out[age <= 13] = Af(f"modechoice.{kind}.11-13")
    out[age <= 10] = Af(f"modechoice.{kind}.5-10")
    return out


def school_car_time_s(world: WorldState) -> np.ndarray:
    """Free-flow car time (s) from every node to each school's best entrance. Shape (n_schools, N)."""
    au = world.approach_u()
    sk = world.net.skim_to(au) + world.net.ff_s[[e.approach_edge for e in world.entrances]][:, None]
    out = np.full((len(world.schools), world.net.n_nodes), np.inf, dtype=np.float64)
    for k, ent in enumerate(world.entrances):
        out[ent.school] = np.minimum(out[ent.school], sk[k])
    return out


def _choose(V: np.ndarray, avail: np.ndarray, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U = np.where(avail, V + G, -np.inf)
    return np.argmax(U, axis=1), U


@dataclass
class ModeResult:
    student_idx: np.ndarray  # person indices of students
    student_mode: np.ndarray  # index into STUDENT_MODES
    student_car_s: np.ndarray
    student_dist_km: np.ndarray
    student_fixed_min: np.ndarray  # travel time for non-car modes (NaN for car modes)
    worker_idx: np.ndarray  # commuting workers today
    worker_mode: np.ndarray  # index into WORKER_MODES
    worker_dist_km: np.ndarray
    worker_fixed_min: np.ndarray
    shuttle_plans: list[ShuttlePlan]


def choose_modes(world: WorldState, draws: Draws, habit: ModeResult | None = None) -> ModeResult:
    p = world.persons
    hh = world.households
    bt = Af("modechoice.beta_time_per_min")
    bc = Af("modechoice.beta_cost_per_usd")
    bh = Af("modechoice.beta_habit")
    cpk = Af("modechoice.vehicle_cost_per_km")
    circ = Af("sim_engine.circuity_factor")
    walk_c = Af("modechoice.speeds_kph.walk_child")
    walk_a = Af("modechoice.speeds_kph.walk_adult")
    bike_v = Af("modechoice.speeds_kph.bike")
    bus_v = Af("modechoice.speeds_kph.bus_avg")

    # ---------------- students ----------------
    si = np.nonzero(p.school >= 0)[0]
    sch = p.school[si]
    home = hh.home_node[p.hh[si]]
    sx = np.array([s.x for s in world.schools])[sch]
    sz = np.array([s.z for s in world.schools])[sch]
    dist = np.hypot(hh.x[p.hh[si]] - sx, hh.z[p.hh[si]] - sz) / 1000.0 * circ
    car_s = school_car_time_s(world)[sch, home] if len(si) else np.zeros(0)
    car_min = car_s / 60.0
    age = p.age[si]
    walk_speed = np.where(age <= 13, walk_c, walk_a)
    M = len(STUDENT_MODES)
    V = np.zeros((len(si), M))
    av = np.zeros((len(si), M), dtype=bool)
    tmin = np.full((len(si), M), np.nan)
    veh = hh.vehicles[p.hh[si]] >= 1
    # drive_dropoff
    k = S_IDX["drive_dropoff"]
    V[:, k] = Af("modechoice.asc_student.drive_dropoff") + bt * car_min + bc * cpk * dist
    av[:, k] = veh
    # carpool
    k = S_IDX["carpool"]
    V[:, k] = Af("modechoice.asc_student.carpool") + bt * car_min + bc * cpk * dist / Af("modechoice.carpool_kids_per_car")
    av[:, k] = True
    # school bus (existing district busing)
    k = S_IDX["school_bus"]
    tmin[:, k] = dist / bus_v * 60.0
    V[:, k] = Af("modechoice.asc_student.school_bus") + bt * tmin[:, k]
    av[:, k] = draws.bus_u[si] < Af("modechoice.school_bus_available_share")
    # shuttle (tool)
    k = S_IDX["school_shuttle"]
    plans = [shuttle_plan(world, sh) for sh in world.shuttles]
    shp = world.shuttle_of_person[si]
    has_sh = shp >= 0
    if has_sh.any():
        ride = np.zeros(len(si))
        wait = np.zeros(len(si))
        for j, sh in enumerate(world.shuttles):
            m = shp == j
            if not m.any():
                continue
            pidx = si[m]
            d = np.hypot(hh.x[p.hh[pidx]][:, None] - np.array(sh.stop_x)[None, :], hh.z[p.hh[pidx]][:, None] - np.array(sh.stop_z)[None, :])
            near = np.argmin(d, axis=1)
            ride[m] = plans[j].ride_km[near]
            wait[m] = sh.headway_min / 2.0
        wk = np.nan_to_num(world.shuttle_walk_km[si])
        tmin[:, k] = wk / walk_speed * 60.0 + wait + ride / bus_v * 60.0
        V[:, k] = Af("modechoice.asc_student.school_shuttle") + bt * np.nan_to_num(tmin[:, k])
    av[:, k] = has_sh
    # walk
    k = S_IDX["walk"]
    tmin[:, k] = dist / walk_speed * 60.0
    V[:, k] = Af("modechoice.asc_student.walk") + bt * tmin[:, k]
    av[:, k] = dist <= _age_limits(age, "max_walk_km_by_age")
    # bike
    k = S_IDX["bike"]
    tmin[:, k] = dist / bike_v * 60.0
    V[:, k] = Af("modechoice.asc_student.bike") + bt * tmin[:, k]
    av[:, k] = dist <= _age_limits(age, "max_bike_km_by_age")
    # teen drives self
    k = S_IDX["teen_drive"]
    V[:, k] = Af("modechoice.asc_student.self_drive") + bt * car_min + bc * cpk * dist
    av[:, k] = (
        (p.grade[si] >= 9) & (age >= int(A("demand.min_driving_age"))) & p.has_license[si] & veh
        & (draws.selfdrive_u[si] < Af("demand.hs_self_drive_share"))
    )
    for m, i in S_IDX.items():
        V[:, i] += world.shift_student[m][si]
    _apply_hints(p.mode_hint[si], STUDENT_MODES, V, av)
    if habit is not None:
        V[np.arange(len(si)), habit.student_mode] += bh * av[np.arange(len(si)), habit.student_mode]
    G = draws.gumbel_student[si]
    smode, U = _choose(V, av, G)

    # caps: shuttle capacity, teen permits, carpool adoption
    for j, plan in enumerate(plans):
        riders = np.nonzero((smode == S_IDX["school_shuttle"]) & (shp == j))[0]
        cap = int(plan.capacity / max(draws.demand_factor, 1e-6))
        smode = _enforce_cap(riders, cap, S_IDX["school_shuttle"], U, av, smode)
    for s_i, school in enumerate(world.schools):
        if school.permits_cap is None:
            continue
        drivers = np.nonzero((smode == S_IDX["teen_drive"]) & (sch == s_i))[0]
        smode = _enforce_cap(drivers, int(school.permits_cap), S_IDX["teen_drive"], U, av, smode)
    if habit is not None:
        smode = _carpool_caps(world, si, smode, habit.student_mode, U, S_IDX["carpool"],
                              {S_IDX["drive_dropoff"], S_IDX["teen_drive"]}, student=True)
    sfixed = np.where(np.isin(smode, [S_IDX["drive_dropoff"], S_IDX["carpool"], S_IDX["teen_drive"]]), np.nan,
                      tmin[np.arange(len(si)), smode])

    # ---------------- workers ----------------
    wmask = p.is_worker & ~p.wfh & (p.work_node >= 0) & (draws.nocommute_u >= Af("demand.worker_not_commuting_today_share"))
    wi = np.nonzero(wmask)[0]
    whome = hh.home_node[p.hh[wi]]
    hx, hz = hh.x[p.hh[wi]], hh.z[p.hh[wi]]
    ext = p.work_external[wi]
    wx = np.where(np.isfinite(p.work_x[wi]), p.work_x[wi], world.net.x[p.work_node[wi]])
    wz = np.where(np.isfinite(p.work_z[wi]), p.work_z[wi], world.net.z[p.work_node[wi]])
    wdist = np.hypot(hx - wx, hz - wz) / 1000.0 * circ
    car_min_w = wdist / Af("sim_engine.car_local_avg_speed_kph") * 60.0
    if ext.any() and world.exits:
        ex_nodes = np.array([e.node for e in world.exits])
        sk = world.net.skim_to(ex_nodes)
        car_min_w[ext] = sk[p.work_exit[wi][ext], whome[ext]] / 60.0
    MW = len(WORKER_MODES)
    VW = np.zeros((len(wi), MW))
    aw = np.zeros((len(wi), MW), dtype=bool)
    tw = np.full((len(wi), MW), np.nan)
    wveh = hh.vehicles[p.hh[wi]] >= 1
    k = W_IDX["drive_alone"]
    VW[:, k] = Af("modechoice.asc_worker.drive_alone") + bt * car_min_w + bc * cpk * wdist
    aw[:, k] = wveh & p.has_license[wi]
    k = W_IDX["carpool"]
    VW[:, k] = Af("modechoice.asc_worker.carpool") + bt * car_min_w + bc * cpk * wdist / Af("modechoice.carpool_worker_occupancy")
    aw[:, k] = True
    k = W_IDX["bike"]
    tw[:, k] = wdist / bike_v * 60.0
    VW[:, k] = Af("modechoice.asc_worker.bike") + bt * tw[:, k]
    aw[:, k] = ~ext & (wdist <= Af("modechoice.max_bike_km_by_age.adult"))
    k = W_IDX["walk"]
    tw[:, k] = wdist / walk_a * 60.0
    VW[:, k] = Af("modechoice.asc_worker.walk") + bt * tw[:, k]
    aw[:, k] = ~ext & (wdist <= Af("modechoice.max_walk_km_by_age.adult"))
    for m, i in W_IDX.items():
        VW[:, i] += world.shift_worker[m][wi]
    _apply_hints(p.mode_hint[wi], WORKER_MODES, VW, aw)
    habit_w = None
    if habit is not None:
        habit_w = _align_habit(habit.worker_idx, habit.worker_mode, wi)
        has = habit_w >= 0
        rows = np.nonzero(has)[0]
        VW[rows, habit_w[has]] += bh * aw[rows, habit_w[has]]
    wmode, UW = _choose(VW, aw, draws.gumbel_worker[wi])
    if habit_w is not None:
        wmode = _carpool_caps(world, wi, wmode, np.where(habit_w >= 0, habit_w, wmode), UW, W_IDX["carpool"],
                              {W_IDX["drive_alone"]}, student=False)
    wfixed = np.where(np.isin(wmode, [W_IDX["drive_alone"], W_IDX["carpool"]]), np.nan, tw[np.arange(len(wi)), wmode])
    return ModeResult(si, smode, car_s, dist, sfixed, wi, wmode, wdist, wfixed, plans)


def _apply_hints(hints: np.ndarray, modes: tuple[str, ...], V: np.ndarray, av: np.ndarray) -> None:
    """Real-household override path (spec 13.4): a non-empty baseline_mode_hint forces that mode."""
    if not len(hints) or not (hints != "").any():
        return
    for i, m in enumerate(modes):
        rows = np.nonzero(hints == m)[0]
        if len(rows):
            av[rows, i] = True
            V[rows, i] += 1e6


def _align_habit(prev_idx: np.ndarray, prev_mode: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(len(idx), -1, dtype=np.int64)
    if not len(prev_idx):
        return out
    pos = np.searchsorted(prev_idx, idx)
    pos = np.clip(pos, 0, len(prev_idx) - 1)
    hit = prev_idx[pos] == idx
    out[hit] = prev_mode[pos[hit]]
    return out


def _enforce_cap(members: np.ndarray, cap: int, mode: int, U: np.ndarray, av: np.ndarray, choice: np.ndarray) -> np.ndarray:
    """Keep at most ``cap`` of ``members`` on ``mode`` (highest margin kept); others re-choose."""
    if len(members) <= max(cap, 0):
        return choice
    others = U[members].copy()
    others[:, mode] = -np.inf
    margin = U[members, mode] - others.max(axis=1)
    drop = members[np.argsort(-margin)[max(cap, 0):]]
    U2 = U[drop].copy()
    U2[:, mode] = -np.inf
    choice = choice.copy()
    choice[drop] = np.argmax(U2, axis=1)
    return choice


def _carpool_caps(world: WorldState, idx: np.ndarray, choice: np.ndarray, base: np.ndarray, U: np.ndarray,
                  carpool: int, drive_modes: set[int], student: bool) -> np.ndarray:
    """Carpool program adoption cap: switchers to carpool <= cap_share * targeted baseline drivers."""
    choice = choice.copy()
    for mask_all, cap_share in world.carpool_caps:
        target = mask_all[idx]
        if not target.any():
            continue
        drivers = target & np.isin(base, list(drive_modes))
        switch = np.nonzero(target & (choice == carpool) & (base != carpool))[0]
        cap = int(np.floor(cap_share * drivers.sum()))
        if len(switch) <= cap:
            continue
        margin = U[switch, carpool] - U[switch, base[switch]]
        revert = switch[np.argsort(-margin)[cap:]]
        choice[revert] = base[revert]
    return choice
