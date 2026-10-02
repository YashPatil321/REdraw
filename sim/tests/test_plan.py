"""Every MVP tool validates, applies and changes something; validation errors."""

from __future__ import annotations

import numpy as np
import pytest

from sim.engine import run_seed
from sim.plan import apply_plan, check_plan, tool_defs


def _ll(world, node: int) -> list[float]:
    return [float(world.net.lat[node]), float(world.net.lon[node])]


def _near_school_nodes(world, school_id: str, k: int = 3) -> list[int]:
    s = world.schools[world.school_index(school_id)]
    d = np.hypot(world.net.x - s.x, world.net.z - s.z)
    return [int(i) for i in np.argsort(d)[1 : 1 + k]]


def tool_examples(world) -> dict[str, dict]:
    n = _near_school_nodes(world, "del_norte_hs", 6)
    sig = int(np.nonzero(world.net.signalized)[0][0])
    edge = int(np.nonzero(world.net.highway == "residential")[0][5])
    return {
        "bell_time": {"school": "del_norte_hs", "start": "09:00"},
        "school_shuttle": {"school": "del_norte_hs", "stops": [_ll(world, n[3]), _ll(world, n[5])], "buses": 2, "headway_min": 15},
        "carpool_program": {"school": "all", "incentive": "high"},
        "dropoff_redesign": {"entrance": "del_norte_hs/main_dropoff", "extra_curb_spots": 4, "faster_unload": True},
        "new_dropoff_entrance": {"school": "del_norte_hs", "location": _ll(world, n[1]), "curb_spots": 6},
        "signal_timing": {"node": int(world.net.node_ids[sig]), "priority": "east_west"},
        "turn_lane": {"edge": edge},
        "bike_route": {"path": [_ll(world, n[0]), _ll(world, n[4])], "facility": "protected"},
        "safe_walk_route": {"path": [_ll(world, n[0]), _ll(world, n[2])], "school": "del_norte_hs", "crossing_guards": True},
        "teen_drive_policy": {"school": "del_norte_hs", "permits_cap": 0},
        "custom": {"description": "Walking school bus with parent volunteers", "estimate": custom_estimate()},
    }


def custom_estimate(**over) -> dict:
    """A confirmed custom tool preview estimate (the format /tools/custom/preview returns)."""
    est = {
        "summary": "Parent-led walking groups to Del Norte, plus a faster drop-off line.",
        "levers": [
            {"type": "mode_utility_shift", "mode": "walk", "applies_to": "students", "school": "del_norte_hs", "utils": 0.6},
            {"type": "capacity_change", "target": "entrance", "entrance": "del_norte_hs/main_dropoff", "factor": 1.2},
        ],
        "adoption_range": [0.05, 0.15],
        "cost_upfront_usd": 0,
        "cost_per_year_usd": 20000,
        "assumptions": ["Enough parents volunteer to lead groups each morning."],
    }
    est.update(over)
    return est


def _changed(tool: str, before, after) -> bool:
    if tool == "bell_time":
        return before.schools[0].bell_s != after.schools[0].bell_s
    if tool == "school_shuttle":
        return len(after.shuttles) == 1 and (after.shuttle_of_person >= 0).any()
    if tool == "carpool_program":
        return after.shift_student["carpool"].sum() > 0 and after.shift_worker["carpool"].sum() > 0 and after.carpool_caps
    if tool == "dropoff_redesign":
        e0, e1 = before.entrances[0], after.entrances[0]
        return e1.curb_spots > e0.curb_spots and e1.unload_s < e0.unload_s
    if tool == "new_dropoff_entrance":
        return len(after.entrances) == len(before.entrances) + 1 and len(after.schools[0].entrances) == 2
    if tool == "signal_timing":
        return not np.allclose(after.signal_factor, before.signal_factor)
    if tool == "turn_lane":
        return after.cap_factor.max() > 1.0
    if tool == "bike_route":
        return after.shift_student["bike"].sum() + after.shift_worker["bike"].sum() > 0
    if tool == "custom":
        return (after.shift_student["walk"].sum() > 0 and after.entrances[0].unload_s < before.entrances[0].unload_s
                and after.adoption_caps and after.llm_estimated == ["custom"])
    if tool == "safe_walk_route":
        return after.shift_student["walk"].sum() > 0
    if tool == "teen_drive_policy":
        return after.schools[0].permits_cap == 0
    raise AssertionError(tool)


def test_every_mvp_tool_has_an_example(world):
    mvp = {t for t, d in tool_defs().items() if d.get("enabled_in_mvp", True)}
    assert mvp == set(tool_examples(world))


@pytest.mark.parametrize("tool", sorted(t for t, d in tool_defs().items() if d.get("enabled_in_mvp", True)))
def test_tool_applies_and_changes_something(world, tool):
    assert world.schools[0].id == "del_norte_hs"
    params = tool_examples(world)[tool]
    chk = check_plan(world, {"mission": "morning_crunch", "tools": [{"tool": tool, "params": params}]})
    assert chk.ok, chk.errors
    assert chk.cost_upfront_usd >= 0 and chk.cost_per_year_usd >= 0
    new = apply_plan(world, chk)
    assert _changed(tool, world, new)
    # base world untouched
    assert world.cap_factor.max() == 1.0 and not world.shuttles and len(world.entrances) == len(world.schools)


def test_all_tools_together_run(world):
    tools = [{"tool": t, "params": p} for t, p in tool_examples(world).items()]
    chk = check_plan(world, {"tools": tools})
    assert chk.ok, chk.errors
    assert chk.cost_upfront_usd > 0 and chk.cost_per_year_usd > 0
    new = apply_plan(world, chk)
    r = run_seed(new, 0, base_world=world, keep_playback=True)
    assert (new.shuttle_of_person >= 0).sum() > 0
    assert 2 in set(r.playback["traj"]["kind"].tolist())  # shuttle buses drive on the network
    assert "del_norte_hs/new_1" in r.playback["entrance_ids"]
    dn = new.school_index("del_norte_hs")
    teen = (r.mode == 7) & (new.persons.school == dn)
    assert teen.sum() == 0  # permits cap 0


def test_validation_errors(world):
    def errs(tools):
        c = check_plan(world, {"tools": tools})
        assert not c.ok
        return " | ".join(c.errors)

    assert "unknown tool" in errs([{"tool": "teleporter", "params": {}}])
    assert "bell_time: unknown school 'xyz'" in errs([{"tool": "bell_time", "params": {"school": "xyz", "start": "09:00"}}])
    assert "later than" in errs([{"tool": "bell_time", "params": {"school": "del_norte_hs", "start": "11:00"}}])
    assert "HH:MM" in errs([{"tool": "bell_time", "params": {"school": "del_norte_hs", "start": "noon"}}])
    assert "more than once" in errs([{"tool": "bell_time", "params": {"school": "del_norte_hs", "start": "09:00"}},
                                     {"tool": "bell_time", "params": {"school": "del_norte_hs", "start": "08:45"}}])
    assert "must be <=" in errs([{"tool": "school_shuttle", "params": {"school": "del_norte_hs", "stops": [[33.0, -117.1]], "buses": 99}}])
    assert "missing required" in errs([{"tool": "turn_lane", "params": {}}])
    assert "does not exist" in errs([{"tool": "turn_lane", "params": {"edge": 10**7}}])
    assert "must be one of" in errs([{"tool": "carpool_program", "params": {"incentive": "huge"}}])
    assert "unknown entrance" in errs([{"tool": "dropoff_redesign", "params": {"entrance": "nope/x", "extra_curb_spots": 2}}])
    assert "from the nearest road" in errs([{"tool": "new_dropoff_entrance", "params": {"school": "del_norte_hs", "location": [34.5, -116.0]}}])
    assert "at least 2 points" in errs([{"tool": "bike_route", "params": {"path": [[33.0, -117.1]]}}])
    assert "estimate this idea with the AI first" in errs([{"tool": "custom", "params": {"description": "more trees"}}])
    bad = check_plan(world, {"tools": "not a list"})
    assert not bad.ok and bad.errors[0].startswith("plan:")


def test_custom_levers_and_constraints(world):
    e = int(np.nonzero(world.net.highway == "primary")[0][0])
    levers = {"mode_utility_shifts": {"carpool": 0.5}, "capacity_changes": [{"edge": e, "factor": 1.5}],
              "cost_upfront_usd": 100000, "cost_per_year_usd": 0, "assumptions": ["people like it"]}
    c = check_plan(world, {"tools": [{"tool": "custom", "params": {"description": "add a lane", "levers": levers}}]})
    assert c.ok, c.errors
    assert c.llm_estimated_tools and "no_new_through_lanes" in c.constraint_violations
    new = apply_plan(world, c)
    assert new.llm_estimated == ["custom"] and new.cap_factor[e] == pytest.approx(1.5)
    c2 = check_plan(world, {"tools": [{"tool": "custom", "params": {"description": "x", "levers": {"mode_utility_shifts": {"walk": 9}}}}]})
    assert not c2.ok


def test_custom_estimate_is_revalidated(world):
    def errs(est):
        return check_plan(world, {"tools": [{"tool": "custom", "params": {"description": "x", "estimate": est}}]}).errors

    ok = check_plan(world, {"tools": [{"tool": "custom", "params": {"description": "x", "estimate": custom_estimate()}}]})
    assert ok.ok and ok.cost_per_year_usd == 20000 and ok.llm_estimated_tools
    shift = custom_estimate()["levers"][0]
    assert "exceeds" in errs(custom_estimate(levers=[shift | {"utils": 5}]))[0]
    assert "not a choice for workers" in errs(custom_estimate(levers=[shift | {"mode": "school_bus", "applies_to": "workers"}]))[0]
    assert "unknown school" in errs(custom_estimate(levers=[shift | {"school": "nope"}]))[0]
    cap = {"type": "capacity_change", "target": "entrance", "entrance": "nope/x", "factor": 1.1}
    assert "unknown entrance" in errs(custom_estimate(levers=[cap]))[0]
    assert "outside" in errs(custom_estimate(levers=[cap | {"factor": 9}]))[0]
    assert "adoption_range" in errs(custom_estimate(adoption_range=[0.5, 0.2]))[0]
    # targeted levers cannot be smuggled in through the raw lever format
    raw = {"targeted_shifts": [{"mode": "walk", "applies_to": "all", "school_idx": -1, "utils": 2}]}
    assert "only from a confirmed estimate" in check_plan(world, {"tools": [{"tool": "custom", "params": {"description": "x", "levers": raw}}]}).errors[0]


def test_custom_adoption_cap_limits_switching(svc):
    """A huge walk boost with a 2% adoption cap moves at most ~2% of targeted students to walking."""
    est = custom_estimate(levers=[{"type": "mode_utility_shift", "mode": "walk", "applies_to": "students", "utils": 2.0}],
                          adoption_range=[0.0, 0.02])
    def run(e):
        plan = {"mission": "morning_crunch", "tools": [{"tool": "custom", "params": {"description": "x", "estimate": e}}]}
        return svc.run(plan, seeds=1, workers=1).report

    capped = run(est)
    free = run(est | {"adoption_range": [0.0, 1.0]})

    def walk_pp(rep):
        return next(m["delta_pp"] for m in rep["mode_share"] if m["mode"] == "walk")

    assert walk_pp(capped) < walk_pp(free)
    assert walk_pp(capped) <= 2.5


def test_budget(world):
    many = [{"tool": "turn_lane", "params": {"edge": int(i)}} for i in range(6)]
    c = check_plan(world, {"tools": many})
    assert c.ok and c.over_budget
    assert c.cost_upfront_usd > c.budget_upfront_usd
    assert any("already has a turn lane" in w for w in check_plan(world, {"tools": many[:1] * 2}).warnings)
