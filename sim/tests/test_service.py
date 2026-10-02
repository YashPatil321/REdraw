"""SimService interface, report shape, CRN, caching, and the M5 directional bell-time test."""

from __future__ import annotations

import numpy as np

from sim.hazards import RoadClosure
from sim.playback import decode_playback
from sim.report import TRAFFIC_METRICS

REPORT_METRIC_IDS = {"avg_commute_min", "avg_dropoff_delay_min", "max_spillback_m", "total_vht", "late_kids",
                     "cost_upfront_usd", "cost_per_year_usd", "winners", "losers"}


def test_static_interface(svc):
    s = svc.world_summary()
    assert s["synthetic"] is True and s["calibration"]["status"] == "uncalibrated"
    assert s["time"]["n_bins"] == 48 and s["hero"]["school_id"] == "del_norte_hs"
    tools = svc.tools()
    assert [t["id"] for t in tools][0] == "bell_time" and tools[-1]["id"] == "custom" and tools[-1]["enabled_in_mvp"] is True
    assert svc.mission()["budget_usd_upfront"] == 2000000
    sch = svc.schools()
    assert sch[0]["entrances"][0]["key"] == "del_norte_hs/main_dropoff"
    net = svc.network_json()
    assert net["n_edges"] == len(net["edges"]) == svc.world.net.n_edges
    e = net["edges"][0]
    assert {"i", "u", "v", "name", "label", "highway", "lanes", "len", "pts"} <= set(e) and len(e["pts"]) % 3 == 0
    assert {"id", "x", "y", "z", "signal"} <= set(net["nodes"][0])
    unv = svc.unverified_inputs()
    assert any("Del Norte High School bell time 08:30" == u for u in unv)
    assert any(u.startswith("SYNTHETIC") for u in unv)


def test_baseline_only_run(svc):
    res = svc.run(None, seeds=2, workers=1)
    assert res.report is None and res.playback_plan is None
    d = decode_playback(res.playback_baseline)
    assert d["header"]["plan_id"] == "baseline"
    ids = [m["id"] for m in res.baseline_summary["metrics"]]
    assert set(TRAFFIC_METRICS) <= set(ids)
    m0 = res.baseline_summary["metrics"][0]
    assert set(m0["value"]) == {"median", "p10", "p90"}
    assert res.baseline_summary["per_school"][0]["entrances"][0]["key"] == "del_norte_hs/main_dropoff"
    # cached on disk
    assert list((svc.world.data_dir / "cache").glob("baseline_*.pkl"))


def test_empty_plan_has_zero_deltas_common_random_numbers(svc):
    res = svc.run({"id": "empty", "tools": []}, seeds=3, workers=1)
    rep = res.report
    for m in rep["metrics"]:
        assert m["delta"]["median"] == 0 and m["delta"]["p10"] == 0 and m["delta"]["p90"] == 0, m["id"]
    assert rep["winners"]["median"] == 0 and rep["losers"]["median"] == 0
    assert rep["side_effects"] == []
    pd_ = res.person_deltas
    assert len(pd_["person_ids"]) == len(svc.world.persons)
    assert np.nanmax(np.abs(np.array(pd_["commute_delta_min"], dtype=float))) == 0


def test_report_shape_and_m5_bell_time_reduces_peak_overlap(svc):
    """M5: moving Del Norte's bell later reduces overlap of drop-off arrivals with the commute peak."""
    plan = {"id": "later-bell", "title": "Del Norte at 9", "tools": [{"tool": "bell_time", "params": {"school": "del_norte_hs", "start": "09:15"}}]}
    msgs = []
    res = svc.run(plan, seeds=4, workers=2, progress=lambda f, m: msgs.append((f, m)))
    rep = res.report
    assert msgs and msgs[-1][0] == 1.0
    assert {m["id"] for m in rep["metrics"]} == REPORT_METRIC_IDS
    for m in rep["metrics"]:
        assert set(m) >= {"id", "label", "unit", "better", "baseline", "plan", "delta"}
        assert set(m["plan"]) == {"median", "p10", "p90"}
    assert rep["seeds"] == 4 and rep["synthetic"] is True
    assert {"upfront_usd", "per_year_usd", "over_budget", "budget_upfront_usd", "budget_per_year_usd", "lines"} <= set(rep["cost"])
    ps = rep["per_school"][0]
    assert {"dropoff_delay_min", "max_spillback_m", "late_kids"} <= set(ps)
    assert set(ps["dropoff_delay_min"]) >= {"baseline", "plan"}
    assert {"mode", "baseline_pct", "plan_pct", "delta_pp"} <= set(rep["mode_share"][0])
    assert {"baseline", "plan", "note"} <= set(rep["peak_overlap"])
    assert rep["peak_overlap"]["plan"] < rep["peak_overlap"]["baseline"]
    assert rep["unverified_inputs"] and rep["calibration"]["status"] == "uncalibrated"
    assert decode_playback(res.playback_plan)["header"]["plan_id"] == "later-bell"


def test_invalid_plan_raises(svc):
    import pytest

    from sim.service import PlanInvalidError

    with pytest.raises(PlanInvalidError):
        svc.run({"tools": [{"tool": "bell_time", "params": {"school": "nope", "start": "09:00"}}]}, seeds=1, workers=1)


def test_road_closure_event_hook(world):
    from sim.engine import run_seed

    base = run_seed(world, 0)
    busiest = int(np.argmax(base.edge_max_vc))
    w2 = world.clone()
    w2.events.append(RoadClosure(edges=[busiest], start_s=21600, end_s=36000))
    closed = run_seed(w2, 0)
    assert closed.edge_max_vc[busiest] < 0.05 * base.edge_max_vc[busiest]
