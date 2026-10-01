"""BPR, queue model, trip chaining, MSA convergence."""

from __future__ import annotations

import numpy as np
import pytest

from sim.assignment import IncrementalMSABackend, bpr
from sim.assumptions import Af, clock_to_s
from sim.demand import ROLE_COMMUTE, ROLE_COMMUTE_CHAIN, ROLE_DROPOFF_HOME, build_demand
from sim.engine import run_seed
from sim.modechoice import S_IDX, choose_modes, make_draws
from sim.schools import run_queue, service_rate, spill_effects
from sim.world import TimeGrid


def test_bpr_function():
    a, b = Af("assignment.bpr_alpha"), Af("assignment.bpr_beta")
    assert bpr(np.array([10.0]), np.array([0.0]))[0] == pytest.approx(10.0)
    assert bpr(np.array([10.0]), np.array([1.0]))[0] == pytest.approx(10.0 * (1 + a))
    assert bpr(np.array([10.0]), np.array([2.0]))[0] == pytest.approx(10.0 * (1 + a * 2.0**b))
    capped = bpr(np.array([10.0]), np.array([50.0]))[0]
    assert capped == pytest.approx(10.0 * (1 + a * Af("assignment.max_vc_for_bpr") ** b))
    # monotone in v/c
    vc = np.linspace(0, 3, 50)
    assert np.all(np.diff(bpr(np.full(50, 5.0), vc)) >= 0)


def _tg() -> TimeGrid:
    return TimeGrid(bin_start_s=0, bin_s=300, n_bins=12, report_start_s=0, report_end_s=3600)


def test_queue_service_rate_and_waits():
    assert service_rate(10, 45) == pytest.approx(10 / 45)
    # 1 spot, 10 s per car: three cars arriving together wait 0, 10, 20 s
    q = run_queue(np.array([0.0, 0.0, 0.0]), np.ones(3), curb_spots=1, unload_s=10, tg=_tg())
    assert np.allclose(np.sort(q.wait_s), [0, 10, 20])
    assert np.allclose(q.done_s - q.start_s, 10)
    # arrivals slower than service never wait
    q = run_queue(np.arange(0, 600, 60.0), np.ones(10), curb_spots=2, unload_s=40, tg=_tg())
    assert np.allclose(q.wait_s, 0)
    assert q.queue_max.max() == pytest.approx(0)


def test_queue_length_and_spillback():
    car = Af("schools.car_length_m")
    # 100 cars arrive in the first 100 s, rate 0.1 car/s -> queue builds to ~90 cars
    arr = np.linspace(0, 100, 100)
    q = run_queue(arr, np.ones(100), curb_spots=1, unload_s=10, tg=_tg())
    assert q.queue_max[0] == pytest.approx(q.queue_max.max())
    assert 85 <= q.queue_max[0] <= 100
    assert np.allclose(q.spill_m, q.queue_max * car)
    assert q.arrivals.sum() == pytest.approx(100)
    # last car starts service after ~990 s
    assert q.start_s.max() == pytest.approx(990, abs=1)
    # queue drains: zero at the end of the window
    assert q.queue_end[-1] == pytest.approx(0)
    # fractional (carpool) weights: 10 half-cars = 5 cars of work
    q2 = run_queue(np.zeros(10), np.full(10, 0.5), curb_spots=1, unload_s=10, tg=_tg())
    assert q2.start_s.max() == pytest.approx(45)


def test_spill_effects_reduce_capacity_and_add_upstream_delay(world):
    net = world.net
    ent = world.entrances[0]
    e = ent.approach_edge
    B, E = world.time.n_bins, net.n_edges
    capf, extra = np.ones((B, E)), np.zeros((B, E))
    spill = np.zeros(B)
    spill[10] = net.length_m[e] + 10 * Af("schools.car_length_m")
    spill[11] = net.length_m[e] * 0.5
    spill_effects(net, e, spill, capf, extra)
    assert capf[10, e] == pytest.approx(Af("schools.queue_capacity_reduction_per_spill"))
    assert capf[11, e] == 1.0
    up = np.nonzero((net.ev == net.eu[e]) & (net.eu != net.ev[e]))[0]
    assert len(up) > 0
    assert np.allclose(extra[10, up], 10 * Af("schools.upstream_spill_delay_s_per_car"))
    assert extra[11].sum() == 0


def test_trip_chaining(world):
    draws = make_draws(world, 0)
    modes = choose_modes(world, draws)
    d = build_demand(world, draws, modes)
    t = d.trips
    p, hh = world.persons, world.households
    bell = np.array([s.bell_s for s in world.schools])
    chains = np.nonzero(np.isin(t.role, [ROLE_COMMUTE_CHAIN, ROLE_DROPOFF_HOME]))[0]
    assert len(chains) > 0
    dropkids = modes.student_idx[modes.student_mode == S_IDX["drive_dropoff"]]
    for k in dropkids:
        ti = d.kid_trip[k]
        assert ti >= 0 and t.kind[ti] == 1
        assert t.origin[ti] == hh.home_node[p.hh[k]]  # chain starts at home
        assert t.wp_school[ti, d.kid_wp[k]] == p.school[k]  # kid's school is a stop
        assert t.wp_queue[ti, d.kid_wp[k]]
    n_commute_chain = 0
    for ti in chains:
        schools = t.wp_school[ti][t.wp_school[ti] >= 0]
        assert np.all(np.diff(bell[schools]) >= 0)  # earliest bell first
        home = t.origin[ti]
        if t.role[ti] == ROLE_COMMUTE_CHAIN:
            n_commute_chain += 1
            drv = t.driver[ti]
            assert p.is_worker[drv] and t.final[ti] == p.work_node[drv]  # home -> school -> work
            # the driver has no separate commute trip
            assert not np.any((t.driver == drv) & (t.role == ROLE_COMMUTE))
            assert d.worker_trip[drv] == ti
        else:
            assert t.final[ti] == home  # home -> school -> home
    assert n_commute_chain > 0
    # one drop-off vehicle per household: siblings share the chain
    hh_of_kid = p.hh[dropkids]
    for h in np.unique(hh_of_kid):
        assert len(np.unique(d.kid_trip[dropkids[hh_of_kid == h]])) == 1 or len(np.unique(p.school[dropkids[hh_of_kid == h]])) > 3


def test_departure_targets_before_bell(world):
    draws = make_draws(world, 1)
    modes = choose_modes(world, draws)
    d = build_demand(world, draws, modes)
    si = modes.student_idx
    bell = np.array([s.bell_s for s in world.schools])[world.persons.school[si]]
    lead = (bell - d.student_target[si]) / 60.0
    assert lead.min() >= 5 - 1e-9


def test_msa_converges_on_tiny_network(world):
    draws = make_draws(world, 0)
    d = build_demand(world, draws, choose_modes(world, draws))
    res = IncrementalMSABackend().assign(world, d.trips, max_traj=50)
    from sim.assumptions import A

    assert int(A("assignment.msa_min_iterations")) <= res.iterations <= int(A("assignment.msa_max_iterations"))
    if res.iterations < int(A("assignment.msa_max_iterations")):
        assert res.rel_change[-1] < Af("assignment.msa_convergence")
    assert res.failed_trips == 0
    assert np.all(np.isfinite(res.tt)) and np.all(res.tt > 0)
    assert np.all(res.speed_kph <= world.net.maxspeed_kph.max() + 1e-3)
    # every arrival after its departure
    ok = res.ok
    assert np.all(res.arrive[ok] >= res.depart[ok] - 1e-6)
    # trajectories are time ordered with consistent offsets
    tr = res.traj
    assert tr["offsets"][-1] == len(tr["edge"])
    assert np.all(tr["exit"] >= tr["enter"])


def test_run_seed_outputs(world):
    r = run_seed(world, 0, keep_playback=True)
    assert set(r.metrics) == {"avg_commute_min", "avg_dropoff_delay_min", "max_spillback_m", "total_vht", "late_kids"}
    assert all(np.isfinite(v) for v in r.metrics.values())
    assert set(r.per_school) == {s.id for s in world.schools}
    assert r.playback["edge_vc"].shape == (world.time.n_bins, world.net.n_edges)
    assert len(r.total_min) == len(world.persons)
    assert r.timings["total_s"] < 20


def test_determinism(world):
    a = run_seed(world, 4)
    b = run_seed(world, 4)
    assert a.metrics == b.metrics
    assert clock_to_s("08:30") == 30600
