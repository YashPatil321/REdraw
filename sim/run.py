"""Command line runner.

    python -m sim.run baseline [--seed 0] [--data-dir DIR]
    python -m sim.run plan path/to/plan.json [--seeds 20] [--workers 4] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sim.assumptions import s_to_clock
from sim.engine import run_seed
from sim.world import ALL_MODES, load_world


def cmd_baseline(a: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    world = load_world(a.data_dir)
    t_load = time.perf_counter() - t0
    if world.synthetic:
        print("*** SYNTHETIC DEV DATA: not real geography ***")
    print(f"world: {world.net.n_nodes} nodes, {world.net.n_edges} edges, {len(world.households)} households, "
          f"{len(world.persons)} persons  (load {t_load:.2f} s)")
    r = run_seed(world, a.seed, keep_playback=True)
    t = r.timings
    print(f"seed {a.seed}: total {t['total_s']:.2f} s  (demand {t['demand_s']:.2f}, assignment {t['assignment_s']:.2f}: "
          f"routing {t['asg_routing']:.2f}, loading {t['asg_loading']:.2f}, queues {t['asg_queues']:.2f}, "
          f"dijkstra runs {int(t['asg_dijkstra_runs'])})")
    print(f"MSA iterations {r.info['iterations']}, rel. change {[round(x, 4) for x in r.info['rel_change']]}, "
          f"trips {r.info['n_trips']}, failed {r.info['failed_trips']}, demand factor {r.info['demand_factor']:.3f}")
    print("\nkey metrics:")
    for k, v in r.metrics.items():
        print(f"  {k:<24} {v:10.2f}")
    print("\nper school drop-off queues:")
    print(f"  {'school':<22} {'bell':>5} {'students':>8} {'drop%':>6} {'maxQ cars':>9} {'spill m':>8} {'wait min':>8} {'delay min':>9} {'late':>5}")
    for s in world.schools:
        ps = r.per_school[s.id]
        print(f"  {s.id:<22} {s.bell_clock:>5} {ps['students']:>8} {100 * ps['dropoff_share']:>5.0f}% {ps['max_queue_cars']:>9.1f} "
              f"{ps['max_spillback_m']:>8.0f} {ps['avg_wait_min']:>8.1f} {ps['dropoff_delay_min']:>9.1f} {ps['late_kids']:>5.0f}")
    tot = sum(r.mode_counts.values())
    print("\nmode share (persons travelling): " + ", ".join(f"{m} {100 * r.mode_counts[m] / tot:.1f}%" for m in ALL_MODES if tot))
    from sim.engine import peak_overlap, peak_window

    w = peak_window(r.commute_onroad_hist, world.time.bin_s)
    tg = world.time
    print(f"commute peak hour {s_to_clock(tg.bin_start_s + w[0] * tg.bin_s)}-{s_to_clock(tg.bin_start_s + w[1] * tg.bin_s)}, "
          f"drop-off arrivals inside it: {100 * peak_overlap(r.dropoff_arr_hist, w):.1f}%")
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    from sim.service import SimService

    plan = json.loads(Path(a.plan).read_text())
    t0 = time.perf_counter()
    svc = SimService.load(a.data_dir)
    print(f"loaded world in {time.perf_counter() - t0:.2f} s")
    chk = svc.check_plan(plan)
    print(f"check: ok={chk.ok} cost upfront ${chk.cost_upfront_usd:,.0f} per year ${chk.cost_per_year_usd:,.0f} "
          f"over_budget={chk.over_budget} violations={chk.constraint_violations}")
    for e in chk.errors:
        print(f"  ERROR {e}")
    for wmsg in chk.warnings:
        print(f"  warning {wmsg}")
    if not chk.ok:
        return 2
    t1 = time.perf_counter()

    def prog(f: float, msg: str) -> None:
        print(f"  [{100 * f:5.1f}%] {msg}", flush=True)

    res = svc.run(plan, seeds=a.seeds, workers=a.workers, progress=prog)
    print(f"\nrun wall time: {time.perf_counter() - t1:.1f} s for {a.seeds} seeds x 2 with {a.workers} workers")
    rep = res.report or {}
    print(f"\n{'metric':<26} {'baseline':>10} {'plan':>10} {'delta':>10}   (median, delta p10..p90)")
    for m in rep.get("metrics", []):
        print(f"{m['id']:<26} {m['baseline']['median']:>10.2f} {m['plan']['median']:>10.2f} {m['delta']['median']:>10.2f}   "
              f"[{m['delta']['p10']:.2f} .. {m['delta']['p90']:.2f}]")
    print("\nper school (drop-off delay min, baseline -> plan):")
    for s in rep.get("per_school", []):
        print(f"  {s['school_id']:<22} {s['dropoff_delay_min']['baseline']['median']:7.2f} -> {s['dropoff_delay_min']['plan']['median']:7.2f}"
              f"   spill {s['max_spillback_m']['baseline']['median']:6.0f} -> {s['max_spillback_m']['plan']['median']:6.0f} m")
    po = rep.get("peak_overlap", {})
    print(f"\npeak overlap: {po.get('baseline')} -> {po.get('plan')}")
    print(f"side effects: {len(rep.get('side_effects', []))} edges; winners {rep.get('winners')}, losers {rep.get('losers')}")
    if a.out:
        Path(a.out).write_text(json.dumps({"report": rep, "baseline_summary": res.baseline_summary}, indent=1, default=str))
        print(f"report written to {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.run")
    ap.add_argument("--data-dir", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline", help="run one baseline seed and print timing and queues")
    b.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("plan", help="run baseline + plan report")
    p.add_argument("plan")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return cmd_baseline(a) if a.cmd == "baseline" else cmd_plan(a)


if __name__ == "__main__":
    sys.exit(main())
