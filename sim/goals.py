"""Mission goals scorecard (spec 11: goals are hints, not pass or fail).

A mission lists ``goals_suggested``. Each entry is either plain text (shown,
never evaluated) or ``{text, check}`` where ``check`` says how to read the
goal off a finished report card:

    {kind: school_change_pct, school: del_norte_hs, field: dropoff_delay_min, target_pct: -30}
        the plan median for that school's field changes by target_pct or better
    {kind: no_school_worse, field: dropoff_delay_min, tolerance: 0.1}
        no school's median delta is worse than +tolerance (in the field's unit)
    {kind: metric_min, metric: resident_approval_pct, min: 55}
        the plan median of a report metric is at least ``min``

Every result carries the numbers it was judged on, so the player can see how
close a plan came. Status is ``met``, ``missed`` or ``unknown`` (the report
lacks the data, for example approval before residents have reacted).
"""

from __future__ import annotations

from typing import Any


def goal_texts(mission: dict[str, Any]) -> list[str]:
    """Goal sentences for display, whatever form the mission uses."""
    out = []
    for g in mission.get("goals_suggested") or []:
        out.append(g if isinstance(g, str) else str(g.get("text", "")))
    return [t for t in out if t]


def _metric_plan(report: dict[str, Any], metric_id: str) -> tuple[float | None, float | None]:
    for m in report.get("metrics") or []:
        if m.get("id") == metric_id:
            b, p = m.get("baseline") or {}, m.get("plan") or {}
            bv = b.get("median") if isinstance(b, dict) else None
            pv = p.get("median") if isinstance(p, dict) else None
            return (float(bv) if isinstance(bv, (int, float)) else None,
                    float(pv) if isinstance(pv, (int, float)) else None)
    return None, None


def _school_row(report: dict[str, Any], school: str) -> dict[str, Any] | None:
    return next((r for r in report.get("per_school") or [] if r.get("school_id") == school), None)


def _median(block: Any, key: str) -> float | None:
    v = (block or {}).get(key) if isinstance(block, dict) else None
    v = v.get("median") if isinstance(v, dict) else v
    return float(v) if isinstance(v, (int, float)) else None


def _evaluate(check: dict[str, Any], report: dict[str, Any]) -> tuple[str, str]:
    kind = check.get("kind")
    if kind == "school_change_pct":
        row = _school_row(report, str(check.get("school")))
        field = str(check.get("field"))
        cell = (row or {}).get(field) or {}
        b, p = _median(cell, "baseline"), _median(cell, "plan")
        if b is None or p is None:
            return "unknown", "no result for this school"
        if b <= 0:
            return ("met" if p <= b else "missed"), f"{b:.1f} → {p:.1f} min"
        change = 100.0 * (p - b) / b
        target = float(check.get("target_pct", 0))
        ok = change <= target if target < 0 else change >= target
        return ("met" if ok else "missed"), f"{b:.1f} → {p:.1f} min ({change:+.0f}%, goal {target:+.0f}%)"
    if kind == "no_school_worse":
        field = str(check.get("field"))
        tol = float(check.get("tolerance", 0.0))
        rows = report.get("per_school") or []
        if not rows:
            return "unknown", "no per-school results"
        worse = []
        for r in rows:
            d = _median((r.get(field) or {}), "delta")
            if d is not None and d > tol:
                worse.append(f"{r.get('name', r.get('school_id'))} {d:+.1f} min")
        if worse:
            return "missed", "worse: " + ", ".join(worse)
        return "met", f"all {len(rows)} schools the same or better"
    if kind == "metric_min":
        _, p = _metric_plan(report, str(check.get("metric")))
        lo = float(check.get("min", 0))
        if p is None:
            return "unknown", "not computed yet"
        return ("met" if p >= lo else "missed"), f"{p:.0f} (goal ≥ {lo:g})"
    return "unknown", f"unknown goal check '{kind}'"


def evaluate_goals(mission: dict[str, Any], report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One row per suggested goal: ``{text, status, detail}``."""
    out = []
    for g in mission.get("goals_suggested") or []:
        if isinstance(g, str):
            out.append({"text": g, "status": "unknown", "detail": "not evaluated"})
            continue
        text = str(g.get("text", ""))
        if not report or not isinstance(g.get("check"), dict):
            out.append({"text": text, "status": "unknown", "detail": "no report yet" if not report else "not evaluated"})
            continue
        status, detail = _evaluate(g["check"], report)
        out.append({"text": text, "status": status, "detail": detail})
    return out
