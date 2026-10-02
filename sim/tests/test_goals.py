"""Mission goals scorecard (sim/goals.py)."""

from sim.goals import evaluate_goals, goal_texts
from sim.plan import mission_def


def _report(dn_base: float, dn_plan: float, other_delta: float, approval: float | None) -> dict:
    def cell(b: float, p: float) -> dict:
        return {"baseline": {"median": b}, "plan": {"median": p}, "delta": {"median": p - b}}

    metrics = [{"id": "avg_commute_min", "baseline": {"median": 14}, "plan": {"median": 14}}]
    if approval is not None:
        metrics.append({"id": "resident_approval_pct", "baseline": {"median": None}, "plan": {"median": approval}})
    return {
        "metrics": metrics,
        "per_school": [
            {"school_id": "del_norte_hs", "name": "Del Norte High", "dropoff_delay_min": cell(dn_base, dn_plan)},
            {"school_id": "design39", "name": "Design39", "dropoff_delay_min": cell(3.0, 3.0 + other_delta)},
        ],
    }


def test_morning_crunch_goals_are_scored():
    m = mission_def("morning_crunch")
    assert goal_texts(m) == ["reduce average drop off delay at Del Norte by 30 percent", "no school gets worse",
                             "resident approval above 55 percent"]
    good = evaluate_goals(m, _report(8.0, 5.0, 0.0, 61.0))
    assert [g["status"] for g in good] == ["met", "met", "met"]
    assert "-38%" in good[0]["detail"]
    bad = evaluate_goals(m, _report(8.0, 7.0, 0.6, 40.0))
    assert [g["status"] for g in bad] == ["missed", "missed", "missed"]
    assert "Design39 +0.6 min" in bad[1]["detail"]


def test_goals_unknown_without_data():
    m = mission_def("morning_crunch")
    assert {g["status"] for g in evaluate_goals(m, None)} == {"unknown"}
    pending = evaluate_goals(m, _report(8.0, 5.0, 0.0, None))
    assert pending[2]["status"] == "unknown"
    # sim noise below the tolerance does not count as worse
    assert evaluate_goals(m, _report(8.0, 5.0, 0.05, 60.0))[1]["status"] == "met"
