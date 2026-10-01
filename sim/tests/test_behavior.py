"""Behavioural regression tests: bell-time direction (M5), common random numbers,
plausible magnitudes, mode-share calibration, balking heterogeneity, road closures."""

from __future__ import annotations

import numpy as np
import pytest

from sim.assumptions import A, Af
from sim.demand import KIND_DROPOFF, balk_threshold_s, build_demand
from sim.engine import peak_overlap, run_seed
from sim.hazards import RoadClosure
from sim.modechoice import LEVELS, S_IDX, choose_modes, make_draws
from sim.plan import apply_plan, check_plan
from sim.report import baseline_peak_window, winners_losers

SEEDS = (0, 1, 2)


def _bell_world(world, start: str = "09:00"):
    chk = check_plan(world, {"tools": [{"tool": "bell_time", "params": {"school": "del_norte_hs", "start": start}}]})
    assert chk.ok, chk.errors
    return apply_plan(world, chk)


@pytest.fixture(scope="module")
def bell_runs(world):
    """Baseline with Del Norte at 08:00 (its drop-offs inside the fixture's commute peak), plan at 09:00."""
    base_world = world.clone()
    dn = base_world.schools[base_world.school_index("del_norte_hs")]
    dn.bell_s, dn.bell_clock = 8 * 3600, "08:00"
    plan_world = _bell_world(base_world)
    base = [run_seed(base_world, s) for s in SEEDS]
    plan = [run_seed(plan_world, s, base_world=base_world) for s in SEEDS]
    return base_world, plan_world, base, plan


def test_later_bell_shifts_targets_and_dropoff_chains_later(world):
    plan_world = _bell_world(world)
    dn = world.school_index("del_norte_hs")
    shift = plan_world.schools[dn].bell_s - world.schools[dn].bell_s
    assert shift == 30 * 60
    draws = make_draws(world, 0)
    db = build_demand(world, draws, choose_modes(world, draws))
    dp = build_demand(plan_world, draws, choose_modes(plan_world, draws, habit=choose_modes(world, draws)))
    kids = np.nonzero(world.persons.school == dn)[0]
    # common random numbers: same targets except for the bell shift
    assert np.allclose(dp.student_target[kids] - db.student_target[kids], shift)
    others = np.nonzero((world.persons.school >= 0) & (world.persons.school != dn))[0]
    assert np.allclose(dp.student_target[others], db.student_target[others])

    def dn_targets(d):
        t = d.trips
        m = (t.wp_school == dn) & (t.kind[:, None] == KIND_DROPOFF)
        return t.wp_target[m]

    # every Del Norte stop of a drop-off chain targets a time 30 minutes later
    assert np.allclose(np.sort(dn_targets(dp)) - np.sort(dn_targets(db)), shift)


def test_m5_later_bell_lowers_overlap_with_fixed_commute_peak(bell_runs):
    world, _, base, plan = bell_runs
    win = baseline_peak_window(world, base)
    ov_b = np.median([peak_overlap(r.dropoff_arr_hist, win) for r in base])
    ov_p = np.median([peak_overlap(r.dropoff_arr_hist, win) for r in plan])
    assert ov_p < ov_b
    # the drop-off arrival histogram moves later
    tb = np.arange(world.time.n_bins)
    mean_b = np.mean([np.average(tb, weights=r.dropoff_arr_hist) for r in base])
    mean_p = np.mean([np.average(tb, weights=r.dropoff_arr_hist) for r in plan])
    assert mean_p > mean_b
    # Del Norte's own drop-off delay does not get worse when its arrivals leave the peak
    dn_b = np.median([r.per_school["del_norte_hs"]["dropoff_delay_min"] for r in base])
    dn_p = np.median([r.per_school["del_norte_hs"]["dropoff_delay_min"] for r in plan])
    assert dn_p <= dn_b + 0.5


def test_common_random_numbers_most_people_unaffected_by_bell_change(bell_runs):
    world, _, base, plan = bell_runs
    p = world.persons
    dn = world.school_index("del_norte_hs")
    dn_hh = np.zeros(len(world.households), dtype=bool)
    dn_hh[p.hh[p.school == dn]] = True
    unaffected = ~dn_hh[p.hh]
    # modes never change for a bell change (same draws, same options)
    for b, q in zip(base, plan, strict=True):
        assert np.array_equal(b.mode, q.mode)
    d = np.stack([np.nan_to_num(q.total_min.astype(float)) - np.nan_to_num(b.total_min.astype(float))
                  for b, q in zip(base, plan, strict=True)])
    med = np.median(d, axis=0)
    thr = Af("report.winner_loser_threshold_min")
    travellers = np.isfinite(base[0].total_min) & unaffected
    share_changed = np.mean(np.abs(med[travellers]) >= thr)
    assert share_changed < 0.05, share_changed
    winners, losers = winners_losers(base, plan)
    n_trav = int(np.sum(np.isfinite(base[0].total_min)))
    assert winners["median"] + losers["median"] < 0.35 * n_trav
    assert winners["p10"] <= winners["median"] <= winners["p90"]
    assert losers["p10"] <= losers["median"] <= losers["p90"]


def test_plausible_magnitudes(world):
    r = run_seed(world, 0)
    n_students = int((world.persons.school >= 0).sum())
    assert r.metrics["late_kids"] < 0.05 * n_students
    assert r.metrics["avg_dropoff_delay_min"] < 15
    for s in world.schools:
        ps = r.per_school[s.id]
        # the formal curb line is bounded by balking (no thousand-car lines)
        mu_per_min = max(world.entrances[e].curb_spots / world.entrances[e].unload_s for e in s.entrances) * 60
        max_wait = Af("sim_engine.dropoff_balk_wait_min") * (1 + Af("sim_behavior.balk_threshold_spread"))
        assert ps["max_queue_cars"] <= mu_per_min * max_wait * len(s.entrances) + 2


def test_mode_share_calibration_hits_targets(world):
    cal = world.asc_calibration
    for lv in LEVELS:
        for mode in ("drive_dropoff", "carpool"):
            row = cal["students"].get(f"{lv}.{mode}")
            if row is None or row["at_bound"]:
                continue
            assert row["modeled"] == pytest.approx(Af(f"sim_behavior.student_share_targets.{lv}.{mode}"), abs=0.01)
    assert cal["workers"]["carpool"]["modeled"] == pytest.approx(Af("sim_behavior.worker_share_targets.carpool"), abs=0.01)
    # siblings decide together: one draw per household
    draws = make_draws(world, 0)
    assert draws.gumbel_student.shape[0] == len(world.households)
    m = choose_modes(world, draws)
    assert set(np.unique(m.student_mode)) <= set(S_IDX.values())


def test_balk_thresholds_are_heterogeneous_and_bounded():
    u = np.linspace(0, 1, 1001)
    t = balk_threshold_s(u) / 60.0
    mode = Af("sim_engine.dropoff_balk_wait_min")
    half = mode * Af("sim_behavior.balk_threshold_spread")
    assert t.min() == pytest.approx(mode - half) and t.max() == pytest.approx(mode + half)
    assert np.all(np.diff(t) >= 0)
    assert np.median(t) == pytest.approx(mode, abs=0.05)


def test_road_closure_reports_vc_against_real_capacity_and_routes_around(world):
    base = run_seed(world, 0, keep_playback=True)
    busiest = int(np.argmax(base.edge_max_vc))
    w2 = world.clone()
    w2.events.append(RoadClosure(edges=[busiest], start_s=21600, end_s=36000))
    closed = run_seed(w2, 0, keep_playback=True)
    # v/c is against the road's own capacity: a closed road shows (almost) no traffic, not a huge ratio
    assert closed.edge_max_vc[busiest] < 0.05 * base.edge_max_vc[busiest]
    assert np.all(np.isfinite(closed.edge_max_vc)) and closed.edge_max_vc.max() < 10
    # routing avoids it: (almost) no sampled vehicle traverses the closed edge
    tr = closed.playback["traj"]
    assert np.sum(tr["edge"] == busiest) <= 0.01 * max(1, np.sum(base.playback["traj"]["edge"] == busiest))
    assert closed.info["failed_trips"] == 0
    assert int(A("sim_engine.seed_base")) > 0
