"""Calibration against typical travel times (spec 6.7).

``python -m sim.calibrate [--seeds N] [--workers W] [--no-search]``

Simulates the baseline, measures time-dependent travel times on the routes in
``data/config/calibration_targets.yaml`` (departing at ``depart``), and, when
observed times exist, tunes demand scale, departure spread and the capacity
factor with a coarse coordinate search over each knob's assumptions range.
Prints a table and writes ``<data_dir>/calibration.json``. With no observed
values (all null) the status is "uncalibrated" and simulated times are still
printed. Tuned knobs are applied by the sim only when status is calibrated or
above_target (see sim.world.load_calibration_knobs).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import load_yaml
from pipeline.geo import latlon_to_scene
from sim.assumptions import A, Af, clock_to_s, leaf
from sim.engine import run_seed
from sim.routing import trees
from sim.world import WorldState, default_knobs, load_world

KNOB_RANGES = {
    "demand_scale": "demand.demand_scale",
    "departure_sd_min": "time_of_day.worker_departure_sd_min",
    "capacity_factor": "assignment.demand_capacity_factor",
}
GRID_POINTS = 5


def load_targets() -> list[dict[str, Any]]:
    return list(load_yaml("calibration_targets.yaml").get("targets", []))


def route_time_s(world: WorldState, tt: np.ndarray, a: int, b: int, depart_s: float) -> float:
    """Route a->b on the edge times of the departure bin, then drive it with time-dependent times."""
    net, tg = world.net, world.time
    b0 = int(tg.bin_of(np.array([depart_s]))[0])
    cost, best = net.graph.best_edges(tt[b0].astype(np.float64))
    fwd, _ = net.graph.matrices(cost)
    dist, pred = trees(fwd, np.array([a]))
    if not np.isfinite(dist[0, b]):
        return float("nan")
    path = []
    cur = b
    while cur != a:
        pv = int(pred[0, cur])
        path.append(int(best[net.graph.pair_index(np.array([pv]), np.array([cur]))[0]]))
        cur = pv
    t = float(depart_s)
    for e in reversed(path):
        t += float(tt[int(tg.bin_of(np.array([t]))[0]), e])
    return t - depart_s


def _target_nodes(world: WorldState, targets: list[dict[str, Any]]) -> list[tuple[int, int]]:
    out = []
    for t in targets:
        fx, fz = latlon_to_scene(float(t["from"]["lat"]), float(t["from"]["lon"]))
        tx, tz = latlon_to_scene(float(t["to"]["lat"]), float(t["to"]["lon"]))
        a = int(world.net.nearest_node(fx, fz)[0])
        b = int(world.net.nearest_node(tx, tz)[0])
        out.append((a, b))
    return out


_CW: dict[str, Any] = {}


def _init(world: WorldState, nodes: list[tuple[int, int]], depart: float) -> None:
    _CW.update(world=world, nodes=nodes, depart=depart)


def _eval(knobs: dict[str, float], seed: int) -> list[float]:
    w = _CW["world"].clone()
    w.knobs = dict(knobs)
    r = run_seed(w, seed, keep_tt=True)
    return [route_time_s(w, r.route_tt, a, b, _CW["depart"]) / 60.0 for a, b in _CW["nodes"]]


def _simulate(pool: ProcessPoolExecutor | None, configs: list[dict[str, float]], seeds: list[int]) -> list[np.ndarray]:
    jobs = [(c, s) for c in configs for s in seeds]
    if pool is None:
        res = [_eval(c, s) for c, s in jobs]
    else:
        res = list(pool.map(_eval, [c for c, _ in jobs], [s for _, s in jobs]))
    out = []
    for i in range(len(configs)):
        arr = np.array(res[i * len(seeds) : (i + 1) * len(seeds)], dtype=np.float64)
        out.append(np.nanmedian(arr, axis=0))
    return out


def _error(sim: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(obs) & np.isfinite(sim) & (obs > 0)
    if not m.any():
        return float("nan")
    return float(np.median(np.abs(sim[m] - obs[m]) / obs[m]))


def calibrate(data_dir: str | Path | None = None, seeds: int = 2, workers: int = 4, search: bool = True) -> dict[str, Any]:
    from sim.report import _mp_context

    world = load_world(data_dir)
    world.knobs = default_knobs()
    cfg = load_yaml("calibration_targets.yaml")
    targets = load_targets()
    depart = float(clock_to_s(cfg.get("depart", "07:45")))
    nodes = _target_nodes(world, targets)
    obs = np.array([np.nan if t.get("observed_minutes") is None else float(t["observed_minutes"]) for t in targets])
    n_obs = int(np.isfinite(obs).sum())
    seed_list = list(range(max(1, seeds)))
    _init(world, nodes, depart)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=_mp_context(), initializer=_init,
                               initargs=(world, nodes, depart)) if workers > 1 else None
    try:
        best = dict(world.knobs)
        sim = _simulate(pool, [best], seed_list)[0]
        err = _error(sim, obs)
        if search and n_obs > 0:
            for knob, path in KNOB_RANGES.items():
                lo, hi = (float(x) for x in leaf(path)["range"])
                cands = [dict(best, **{knob: float(v)}) for v in np.linspace(lo, hi, GRID_POINTS)]
                sims = _simulate(pool, cands, seed_list)
                errs = [_error(s, obs) for s in sims]
                i = int(np.nanargmin(errs))
                if np.isfinite(errs[i]) and (not np.isfinite(err) or errs[i] < err):
                    best, sim, err = cands[i], sims[i], errs[i]
    finally:
        if pool is not None:
            pool.shutdown()
    min_t = int(A("calibration.min_targets"))
    target_err = Af("calibration.target_median_error")
    if n_obs >= min_t and np.isfinite(err):
        status = "calibrated" if err <= target_err else "above_target"
    else:
        status = "uncalibrated"
    table = []
    for t, s, o in zip(targets, sim, obs, strict=True):
        table.append({"id": t["id"], "label": t.get("label", t["id"]),
                      "observed_min": None if not np.isfinite(o) else float(o),
                      "simulated_min": None if not np.isfinite(s) else round(float(s), 2),
                      "error_pct": None if not (np.isfinite(o) and np.isfinite(s) and o > 0) else round(float(abs(s - o) / o * 100), 1)})
    result = {
        "status": status, "median_error_pct": None if not np.isfinite(err) else round(err * 100, 1),
        "targets_with_data": n_obs, "targets_total": len(targets), "table": table,
        "params": best, "depart": cfg.get("depart", "07:45"), "seeds": len(seed_list),
        "synthetic": world.synthetic, "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target_median_error_pct": target_err * 100,
    }
    (world.data_dir / "calibration.json").write_text(json.dumps(result, indent=1))
    return result


def print_table(res: dict[str, Any]) -> None:
    print(f"{'route':<18} {'observed':>9} {'simulated':>10} {'error':>7}  label")
    for r in res["table"]:
        o = "-" if r["observed_min"] is None else f"{r['observed_min']:.1f}"
        s = "-" if r["simulated_min"] is None else f"{r['simulated_min']:.1f}"
        e = "-" if r["error_pct"] is None else f"{r['error_pct']:.0f}%"
        print(f"{r['id']:<18} {o:>9} {s:>10} {e:>7}  {r['label']}")
    med = res["median_error_pct"]
    print(f"status: {res['status']}  median error: {'n/a' if med is None else f'{med:.1f}%'} "
          f"(target < {res['target_median_error_pct']:.0f}%)  targets with data: {res['targets_with_data']}/{res['targets_total']}")
    print(f"params: {json.dumps(res['params'])}")
    if res.get("synthetic"):
        print("NOTE: synthetic world; simulated times are not comparable to real roads.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate the baseline against calibration_targets.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-search", action="store_true")
    a = ap.parse_args(argv)
    res = calibrate(a.data_dir, seeds=a.seeds, workers=a.workers, search=not a.no_search)
    print_table(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
