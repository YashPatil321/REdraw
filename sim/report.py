"""Report card and Monte Carlo ranges (spec 6.6, docs/api.md "Report").

Baseline and plan run with the same seeds (common random numbers) in a
process pool. Every metric gets median and p10/p90 across seeds; deltas are
computed per seed (plan - baseline, same seed) and then summarized.

Baseline results for a seed list are cached in memory and on disk at
``<data_dir>/cache/baseline_<hash>.pkl``; the hash covers the processed data
files, assumptions.yaml, the calibration knobs and the sim version.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

import numpy as np

from sim.assumptions import A, Af, assumptions_hash, unverified_used
from sim.engine import SeedResult, peak_overlap, peak_window, run_seed
from sim.world import ALL_MODES, MODE_LABELS, WorldState

log = logging.getLogger("sim.report")
SIM_VERSION = "sim-1.0"

METRIC_DEFS: list[dict[str, str]] = [
    {"id": "avg_commute_min", "label": "Average commute time, all workers", "unit": "min", "better": "lower"},
    {"id": "avg_dropoff_delay_min", "label": "Average drop-off delay (queue + congestion)", "unit": "min", "better": "lower"},
    {"id": "max_spillback_m", "label": "Max drop-off queue spillback", "unit": "m", "better": "lower"},
    {"id": "total_vht", "label": "Total vehicle hours traveled (06:30-09:30)", "unit": "h", "better": "lower"},
    {"id": "late_kids", "label": "Kids arriving late (after bell)", "unit": "kids", "better": "lower"},
    {"id": "cost_upfront_usd", "label": "Plan cost, upfront", "unit": "USD", "better": "lower"},
    {"id": "cost_per_year_usd", "label": "Plan cost, per year", "unit": "USD/yr", "better": "lower"},
    {"id": "resident_approval_pct", "label": "Resident approval", "unit": "%", "better": "higher"},
    {"id": "winners", "label": "Residents better off by 3+ min", "unit": "people", "better": "higher"},
    {"id": "losers", "label": "Residents worse off by 3+ min", "unit": "people", "better": "lower"},
]
TRAFFIC_METRICS = ("avg_commute_min", "avg_dropoff_delay_min", "max_spillback_m", "total_vht", "late_kids")
_DEF = {m["id"]: m for m in METRIC_DEFS}

ProgressFn = Callable[[float, str], None]

# ---------------------------------------------------------------------------
# process pool
# ---------------------------------------------------------------------------

_WORKER: dict[str, WorldState | None] = {"base": None, "plan": None}


def _init_worker(base: WorldState, plan: WorldState | None) -> None:
    _WORKER["base"] = base
    _WORKER["plan"] = plan


def _task(which: str, seed: int, keep_playback: bool) -> tuple[str, int, SeedResult]:
    base = _WORKER["base"]
    assert base is not None
    if which == "base":
        return which, seed, run_seed(base, seed, keep_playback=keep_playback)
    plan = _WORKER["plan"]
    assert plan is not None
    return which, seed, run_seed(plan, seed, base_world=base, keep_playback=keep_playback)


def _mp_context() -> mp.context.BaseContext:
    method = os.environ.get("REDRAW_MP_START") or ("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    return mp.get_context(method)


def run_tasks(base: WorldState, plan: WorldState | None, tasks: list[tuple[str, int]], workers: int,
              progress: ProgressFn | None = None, progress_offset: int = 0, progress_total: int | None = None) -> dict[tuple[str, int], SeedResult]:
    total = progress_total or len(tasks)
    out: dict[tuple[str, int], SeedResult] = {}
    done = progress_offset
    if workers <= 1 or len(tasks) <= 1:
        _init_worker(base, plan)
        for which, seed in tasks:
            _, s, r = _task(which, seed, seed == 0)
            out[(which, s)] = r
            done += 1
            if progress:
                progress(done / total, f"{'baseline' if which == 'base' else 'plan'} seed {s + 1}")
        return out
    with ProcessPoolExecutor(max_workers=workers, mp_context=_mp_context(), initializer=_init_worker, initargs=(base, plan)) as ex:
        futs = [ex.submit(_task, which, seed, seed == 0) for which, seed in tasks]
        for f in as_completed(futs):
            which, s, r = f.result()
            out[(which, s)] = r
            done += 1
            if progress:
                progress(done / total, f"{'baseline' if which == 'base' else 'plan'} seed {s + 1} done ({done}/{total})")
    return out


# ---------------------------------------------------------------------------
# baseline cache
# ---------------------------------------------------------------------------

_MEM_CACHE: dict[str, list[SeedResult]] = {}


def baseline_key(world: WorldState, seeds: list[int]) -> str:
    blob = json.dumps({"v": SIM_VERSION, "data": world.data_hash, "a": assumptions_hash(), "k": world.knobs,
                       "seeds": list(seeds)}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def get_baseline(world: WorldState, seeds: list[int], workers: int, progress: ProgressFn | None = None,
                 progress_total: int | None = None) -> list[SeedResult]:
    key = baseline_key(world, seeds)
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]
    path = world.data_dir / "cache" / f"baseline_{key}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                res = pickle.load(f)
            _MEM_CACHE[key] = res
            return res
        except Exception as exc:  # corrupt cache: recompute
            log.warning("ignoring unreadable baseline cache %s: %s", path, exc)
    out = run_tasks(world, None, [("base", s) for s in seeds], workers, progress, 0, progress_total)
    res = [out[("base", s)] for s in seeds]
    _MEM_CACHE[key] = res
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    except OSError as exc:
        log.warning("could not write baseline cache %s: %s", path, exc)
    return res


# ---------------------------------------------------------------------------
# aggregation helpers
# ---------------------------------------------------------------------------

def stats(values: list[float] | np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"median": 0.0, "p10": 0.0, "p90": 0.0}
    lo, hi = Af("report.percentile_low"), Af("report.percentile_high")
    return {"median": round(float(np.median(v)), 3), "p10": round(float(np.percentile(v, lo)), 3),
            "p90": round(float(np.percentile(v, hi)), 3)}


def _const(x: float) -> dict[str, float]:
    return {"median": float(x), "p10": float(x), "p90": float(x)}


def _mode_pct(r: SeedResult) -> dict[str, float]:
    tot = sum(r.mode_counts.values())
    return {m: (100.0 * r.mode_counts.get(m, 0.0) / tot if tot else 0.0) for m in ALL_MODES}


def _winners_losers(base: SeedResult, plan: SeedResult) -> tuple[int, int]:
    thr = Af("report.winner_loser_threshold_min")
    b, p = base.total_min.astype(np.float64), plan.total_min.astype(np.float64)
    either = np.isfinite(b) | np.isfinite(p)
    d = np.nan_to_num(p) - np.nan_to_num(b)
    return int(np.sum(either & (d <= -thr))), int(np.sum(either & (d >= thr)))


def _school_block(world: WorldState, results: list[SeedResult]) -> list[dict[str, Any]]:
    out = []
    for s in world.schools:
        rows = [r.per_school.get(s.id) for r in results if s.id in r.per_school]
        if not rows:
            continue
        ents = []
        keys = [e["key"] for e in rows[0]["entrances"]]
        for k in keys:
            vals = [next((e for e in row["entrances"] if e["key"] == k), None) for row in rows]
            vals = [v for v in vals if v]
            ents.append({"key": k, "max_queue_cars": stats([v["max_queue_cars"] for v in vals])["median"],
                         "max_spillback_m": stats([v["max_spillback_m"] for v in vals])["median"],
                         "avg_wait_min": stats([v["avg_wait_min"] for v in vals])["median"]})
        out.append({
            "school_id": s.id, "name": s.name, "bell_start": s.bell_clock,
            "dropoff_delay_min": stats([r["dropoff_delay_min"] for r in rows]),
            "max_spillback_m": stats([r["max_spillback_m"] for r in rows]),
            "late_kids": stats([r["late_kids"] for r in rows]),
            "max_queue_cars": stats([r["max_queue_cars"] for r in rows]),
            "avg_wait_min": stats([r["avg_wait_min"] for r in rows]),
            "entrances": ents,
        })
    return out


def baseline_summary(world: WorldState, base: list[SeedResult], calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = []
    for mid in TRAFFIC_METRICS:
        metrics.append({**_DEF[mid], "value": stats([r.metrics[mid] for r in base])})
    for mid in ("cost_upfront_usd", "cost_per_year_usd"):
        metrics.append({**_DEF[mid], "value": _const(0.0)})
    pct = [_mode_pct(r) for r in base]
    windows = [peak_window(r.commute_onroad_hist, world.time.bin_s) for r in base]
    overlap = [peak_overlap(r.dropoff_arr_hist, w) for r, w in zip(base, windows, strict=True)]
    w0 = windows[0]
    return {
        "metrics": metrics,
        "per_school": _school_block(world, base),
        "mode_share": [{"mode": m, "label": MODE_LABELS[m], "pct": round(float(np.median([p[m] for p in pct])), 2)} for m in ALL_MODES],
        "peak_overlap": {"value": round(float(np.median(overlap)), 4),
                         "peak_start_s": world.time.bin_start_s + w0[0] * world.time.bin_s,
                         "peak_end_s": world.time.bin_start_s + w0[1] * world.time.bin_s,
                         "note": "share of school drop-off arrivals in the commute peak hour"},
        "seeds": len(base), "synthetic": world.synthetic,
        "calibration": calibration or {},
    }


def build_report(base_world: WorldState, plan_world: WorldState, base: list[SeedResult], plan: list[SeedResult],
                 check: Any, plan_id: str, unverified: list[str], calibration: dict[str, Any]) -> dict[str, Any]:
    n = len(base)
    metrics = []
    for mid in TRAFFIC_METRICS:
        b = np.array([r.metrics[mid] for r in base])
        p = np.array([r.metrics[mid] for r in plan])
        metrics.append({**_DEF[mid], "baseline": stats(b), "plan": stats(p), "delta": stats(p - b)})
    cu, cy = float(check.cost_upfront_usd), float(check.cost_per_year_usd)
    metrics.append({**_DEF["cost_upfront_usd"], "baseline": _const(0), "plan": _const(cu), "delta": _const(cu)})
    metrics.append({**_DEF["cost_per_year_usd"], "baseline": _const(0), "plan": _const(cy), "delta": _const(cy)})
    wl = [_winners_losers(b, p) for b, p in zip(base, plan, strict=True)]
    winners = stats([w for w, _ in wl])
    losers = stats([lo for _, lo in wl])
    metrics.append({**_DEF["winners"], "baseline": _const(0), "plan": winners, "delta": winners})
    metrics.append({**_DEF["losers"], "baseline": _const(0), "plan": losers, "delta": losers})

    bs = {s["school_id"]: s for s in _school_block(base_world, base)}
    ps = {s["school_id"]: s for s in _school_block(plan_world, plan)}
    per_school = []
    for sid, b in bs.items():
        p = ps.get(sid, b)
        per_school.append({
            "school_id": sid, "name": b["name"],
            "bell_start": {"baseline": b["bell_start"], "plan": p["bell_start"]},
            **{k: {"baseline": b[k], "plan": p[k],
                   "delta": stats([pr.per_school[sid][k] - br.per_school[sid][k] for br, pr in zip(base, plan, strict=True)])}
               for k in ("dropoff_delay_min", "max_spillback_m", "late_kids", "max_queue_cars", "avg_wait_min")},
            "entrances": {"baseline": b["entrances"], "plan": p["entrances"]},
        })

    bpct = [_mode_pct(r) for r in base]
    ppct = [_mode_pct(r) for r in plan]
    mode_share = []
    for m in ALL_MODES:
        bm = float(np.median([x[m] for x in bpct]))
        pm = float(np.median([x[m] for x in ppct]))
        dm = float(np.median([y[m] - x[m] for x, y in zip(bpct, ppct, strict=True)]))
        mode_share.append({"mode": m, "label": MODE_LABELS[m], "baseline_pct": round(bm, 2), "plan_pct": round(pm, 2), "delta_pp": round(dm, 2)})

    # side effects: edges whose (median across seeds) max v/c rises above the threshold because of the plan
    thr = Af("report.side_effect_vc_threshold")
    bvc = np.median(np.stack([r.edge_max_vc for r in base]), axis=0)
    pvc = np.median(np.stack([r.edge_max_vc for r in plan]), axis=0)
    cand = np.nonzero((pvc > thr) & (bvc <= thr))[0]
    cand = cand[np.argsort(-(pvc[cand] - bvc[cand]))][: int(A("sim_engine.side_effects_max"))]
    pbins = np.stack([r.edge_max_bin for r in plan])
    net = plan_world.net
    side = []
    for e in cand:
        vals, cnt = np.unique(pbins[:, e], return_counts=True)
        b = int(vals[np.argmax(cnt)])
        side.append({"edge_idx": int(e), "name": str(net.name[e] or net.label[e] or net.highway[e]),
                     "baseline_vc": round(float(bvc[e]), 3), "plan_vc": round(float(pvc[e]), 3),
                     "bin_s": int(plan_world.time.bin_start_s + b * plan_world.time.bin_s),
                     "x": round(float(net.mid_x[e]), 1), "z": round(float(net.mid_z[e]), 1)})

    windows = [peak_window(r.commute_onroad_hist, base_world.time.bin_s) for r in base]
    ov_b = [peak_overlap(r.dropoff_arr_hist, w) for r, w in zip(base, windows, strict=True)]
    ov_p = [peak_overlap(r.dropoff_arr_hist, w) for r, w in zip(plan, windows, strict=True)]
    tg = base_world.time
    return {
        "plan_id": plan_id, "seeds": n, "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "synthetic": base_world.synthetic,
        "cost": {"upfront_usd": cu, "per_year_usd": cy, "over_budget": bool(check.over_budget),
                 "budget_upfront_usd": float(check.budget_upfront_usd), "budget_per_year_usd": float(check.budget_per_year_usd),
                 "lines": [c.model_dump() for c in check.cost_lines]},
        "constraint_violations": list(check.constraint_violations),
        "metrics": metrics,
        "per_school": per_school,
        "mode_share": mode_share,
        "winners": winners,
        "losers": losers,
        "side_effects": side,
        "peak_overlap": {"baseline": round(float(np.median(ov_b)), 4), "plan": round(float(np.median(ov_p)), 4),
                         "delta": stats(np.array(ov_p) - np.array(ov_b)),
                         "peak_start_s": tg.bin_start_s + windows[0][0] * tg.bin_s,
                         "peak_end_s": tg.bin_start_s + windows[0][1] * tg.bin_s,
                         "note": "share of school drop-off arrivals in the commute peak hour"},
        "unverified_inputs": unverified,
        "llm_estimated_tools": list(check.llm_estimated_tools),
        "calibration": {"status": calibration.get("status", "uncalibrated"), "median_error_pct": calibration.get("median_error_pct")},
    }


def person_deltas(world: WorldState, base: list[SeedResult], plan: list[SeedResult]) -> dict[str, Any]:
    def med(attr: str) -> np.ndarray:
        d = np.stack([getattr(p, attr).astype(np.float64) - getattr(b, attr).astype(np.float64) for b, p in zip(base, plan, strict=True)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(d, axis=0)

    net = world.net
    hn = world.households.home_node[world.persons.hh]

    def node_max(r: SeedResult) -> np.ndarray:
        m = np.zeros(net.n_nodes)
        np.maximum.at(m, net.eu, r.edge_max_vc)
        np.maximum.at(m, net.ev, r.edge_max_vc)
        return m

    hv = np.median(np.stack([node_max(p) - node_max(b) for b, p in zip(base, plan, strict=True)]), axis=0)
    names = np.array(list(ALL_MODES) + ["none"])

    def clean(a: np.ndarray) -> list[float]:
        return [float(round(x, 3)) if np.isfinite(x) else float("nan") for x in a]

    return {
        "person_ids": world.persons.pid.astype(int).tolist(),
        "commute_delta_min": clean(med("commute_min")),
        "dropoff_delta_min": clean(med("dropoff_min")),
        "kid_trip_delta_min": clean(med("kid_min")),
        "mode_baseline": names[base[0].mode.astype(int)].tolist(),
        "mode_plan": names[plan[0].mode.astype(int)].tolist(),
        "home_edges_vc_delta": clean(hv[hn]),
    }


def run_report(base_world: WorldState, plan_world: WorldState | None, check: Any, *, plan_id: str, seeds: int, workers: int,
               calibration: dict[str, Any], unverified: list[str], progress: ProgressFn | None = None) -> dict[str, Any]:
    """Run baseline (cached) and plan; return dict with report, baseline_summary, results."""
    t0 = time.perf_counter()
    seed_list = list(range(int(seeds)))
    key = baseline_key(base_world, seed_list)
    cached = key in _MEM_CACHE or (base_world.data_dir / "cache" / f"baseline_{key}.pkl").exists()
    total = (0 if cached else len(seed_list)) + (len(seed_list) if plan_world is not None else 0)
    total = max(total, 1)
    if plan_world is not None and not cached:
        # run baseline and plan seeds in one pool for better load balance
        out = run_tasks(base_world, plan_world, [(w, s) for s in seed_list for w in ("base", "plan")], workers, progress, 0, total)
        base = [out[("base", s)] for s in seed_list]
        plan = [out[("plan", s)] for s in seed_list]
        _store_baseline(base_world, seed_list, base)
    else:
        base = get_baseline(base_world, seed_list, workers, progress, total)
        plan = []
        if plan_world is not None:
            out = run_tasks(base_world, plan_world, [("plan", s) for s in seed_list], workers, progress, 0, total)
            plan = [out[("plan", s)] for s in seed_list]
    unv = list(unverified) + [u for u in unverified_used() if u not in unverified]
    summary = baseline_summary(base_world, base, calibration)
    result: dict[str, Any] = {"baseline": base, "plan": plan, "baseline_summary": summary, "report": None, "person_deltas": {}}
    if plan_world is not None:
        result["report"] = build_report(base_world, plan_world, base, plan, check, plan_id, unv, calibration)
        result["person_deltas"] = person_deltas(base_world, base, plan)
    result["wall_s"] = time.perf_counter() - t0
    if progress:
        progress(1.0, "done")
    return result


def _store_baseline(world: WorldState, seeds: list[int], res: list[SeedResult]) -> None:
    key = baseline_key(world, seeds)
    _MEM_CACHE[key] = res
    path = world.data_dir / "cache" / f"baseline_{key}.pkl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    except OSError as exc:
        log.warning("could not write baseline cache %s: %s", path, exc)
