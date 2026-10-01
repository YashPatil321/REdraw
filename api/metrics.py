"""Metric definitions served to the client (spec 13A.6: UI text comes from the API)."""

from __future__ import annotations

from typing import Any

from pipeline.config import assumption


def default_metrics() -> list[dict[str, Any]]:
    thr = assumption("report.winner_loser_threshold_min")
    thr_s = f"{thr:g}"
    return [
        {"id": "avg_commute_min", "label": "Average commute time, all workers", "unit": "min", "better": "lower"},
        {"id": "avg_dropoff_delay_min", "label": "Average school drop-off delay", "unit": "min", "better": "lower"},
        {"id": "max_spillback_m", "label": "Worst drop-off queue spillback", "unit": "m", "better": "lower"},
        {"id": "total_vht", "label": "Total vehicle hours traveled", "unit": "veh h", "better": "lower"},
        {"id": "late_kids", "label": "Students arriving late", "unit": "students", "better": "lower"},
        {"id": "cost_upfront_usd", "label": "Upfront cost", "unit": "USD", "better": "lower"},
        {"id": "cost_per_year_usd", "label": "Cost per year", "unit": "USD/yr", "better": "lower"},
        {"id": "resident_approval_pct", "label": "Resident approval", "unit": "%", "better": "higher"},
        {"id": "winners", "label": f"People saving {thr_s}+ minutes", "unit": "people", "better": "higher"},
        {"id": "losers", "label": f"People losing {thr_s}+ minutes", "unit": "people", "better": "lower"},
    ]


def metric_defs(world_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Sim-provided metric list if any, merged over the API defaults (approval always present)."""
    defaults = default_metrics()
    from_sim = (world_summary or {}).get("metrics") or []
    if not from_sim:
        return defaults
    by_id = {m["id"]: dict(m) for m in defaults}
    order = [m["id"] for m in from_sim]
    for m in from_sim:
        by_id[m["id"]] = {**by_id.get(m["id"], {}), **m}
    for m in defaults:
        if m["id"] not in order:
            order.append(m["id"])
    return [by_id[i] for i in order]


def metric_value(report: dict[str, Any] | None, metric_id: str) -> float | None:
    """Plan median of a metric in a report card (also reads cost / winners blocks)."""
    if not report:
        return None
    for m in report.get("metrics") or []:
        if m.get("id") == metric_id:
            plan = m.get("plan") or {}
            v = plan.get("median") if isinstance(plan, dict) else plan
            return float(v) if isinstance(v, (int, float)) else None
    cost = report.get("cost") or {}
    if metric_id == "cost_upfront_usd" and "upfront_usd" in cost:
        return float(cost["upfront_usd"])
    if metric_id == "cost_per_year_usd" and "per_year_usd" in cost:
        return float(cost["per_year_usd"])
    if metric_id in ("winners", "losers"):
        v = (report.get(metric_id) or {}).get("median")
        return float(v) if isinstance(v, (int, float)) else None
    return None


def headline(report: dict[str, Any] | None) -> dict[str, float | None]:
    if not report:
        return {}
    out: dict[str, float | None] = {}
    for m in report.get("metrics") or []:
        if m.get("id"):
            out[m["id"]] = metric_value(report, m["id"])
    for mid in ("cost_upfront_usd", "cost_per_year_usd", "winners", "losers"):
        if mid not in out:
            v = metric_value(report, mid)
            if v is not None:
                out[mid] = v
    return out
