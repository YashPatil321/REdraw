"""Resident reactions to a plan (spec 8.2).

Approval is deterministic: a logistic of a score built from each persona's
personal deltas (from `RunResult.person_deltas`) weighted by their values, with
every weight read from `assumptions.residents.*`. The LLM never decides
approval; it only writes a one to two sentence reaction given the numbers, and
any reply that cites a number it was not given is dropped (text = None).
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from pipeline.config import assumption
from residents.llm import ChatLLM, extract_json
from residents.persona import Persona
from residents.textcheck import collect_numbers, safe_text

log = logging.getLogger(__name__)

# Which tools speak to which values (game rules, not real-world numbers).
SAFETY_TOOLS = frozenset({"safe_walk_route", "bike_route"})
COMMUNITY_TOOLS = frozenset({"school_shuttle", "safe_walk_route", "carpool_program"})
ENVIRONMENT_TOOLS = frozenset({"bike_route", "safe_walk_route", "school_shuttle", "carpool_program"})
REACTION_BATCH = 10
BOOTSTRAP_SAMPLES = 400
BOOTSTRAP_SEED_OFFSET = 7


def _opt_assumption(path: str, fallback: float) -> float:
    """Assumptions that may not exist yet in assumptions.yaml (logged once)."""
    try:
        return float(assumption(path))
    except KeyError:
        _warn_missing(path, fallback)
        return fallback


@lru_cache(maxsize=None)
def _warn_missing(path: str, fallback: float) -> None:
    log.warning("assumptions.yaml has no '%s'; using %s (add it with a source)", path, fallback)


@dataclass(frozen=True)
class ApprovalParams:
    base: float
    per_min_commute_saved: float
    per_min_dropoff_saved: float
    per_100usd_cost: float
    street_change_penalty: float
    change_averse_penalty: float
    safety_tool_bonus: float
    households_sharing_cost: float
    street_change_vc_delta: float
    multipliers: dict[str, float]

    @classmethod
    def load(cls) -> ApprovalParams:
        r = "residents."
        mult_keys = ["time", "safety", "cost", "environment", "community", "property"]
        return cls(
            base=float(assumption(r + "approval_base")),
            per_min_commute_saved=float(assumption(r + "approval_per_min_commute_saved")),
            per_min_dropoff_saved=float(assumption(r + "approval_per_min_dropoff_saved")),
            per_100usd_cost=float(assumption(r + "approval_per_100usd_cost_exposure")),
            street_change_penalty=float(assumption(r + "approval_street_change_penalty")),
            change_averse_penalty=float(assumption(r + "approval_change_averse_penalty")),
            safety_tool_bonus=float(assumption(r + "safety_tool_bonus")),
            households_sharing_cost=float(assumption(r + "households_sharing_cost")),
            street_change_vc_delta=_opt_assumption(r + "street_change_vc_delta", 0.1),
            multipliers={k: float(assumption(f"{r}value_multipliers.{k}")) for k in mult_keys},
        )


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _r(v: float | None, nd: int = 1) -> float | None:
    return None if v is None else round(v, nd)


class DeltaIndex:
    """Fast lookup of person_deltas columns by person_id."""

    def __init__(self, person_deltas: dict[str, Any] | None) -> None:
        pd_ = person_deltas or {}
        ids = list(pd_.get("person_ids") or [])
        self.index = {int(p): i for i, p in enumerate(ids)}
        self.cols = {k: list(v) for k, v in pd_.items() if k != "person_ids" and v is not None}

    def get(self, person_id: int, col: str) -> Any:
        i = self.index.get(int(person_id))
        vals = self.cols.get(col)
        if i is None or vals is None or i >= len(vals):
            return None
        return vals[i]


def persona_deltas(p: Persona, idx: DeltaIndex, cost_per_year_usd: float,
                   params: ApprovalParams) -> dict[str, Any]:
    commute = _f(idx.get(p.person_id, "commute_delta_min"))
    kid_trip = _f(idx.get(p.person_id, "kid_trip_delta_min"))
    dropoff = _f(idx.get(p.person_id, "dropoff_delta_min"))
    if dropoff is None:  # household's driver delay applies to the whole household
        vals = [_f(idx.get(m, "dropoff_delta_min")) for m in p.household_member_ids]
        finite = [v for v in vals if v is not None]
        dropoff = float(np.mean(finite)) if finite else None
    vc = _f(idx.get(p.person_id, "home_edges_vc_delta"))
    if vc is None:
        vals = [_f(idx.get(m, "home_edges_vc_delta")) for m in p.household_member_ids]
        finite = [v for v in vals if v is not None]
        vc = max(finite) if finite else None
    cost = cost_per_year_usd / params.households_sharing_cost if params.households_sharing_cost else 0.0
    return {
        "commute_min": _r(commute),
        "dropoff_min": _r(dropoff),
        "kid_trip_min": _r(kid_trip),
        "cost_usd_year": round(cost, 1),
        "street_change": bool(vc is not None and vc >= params.street_change_vc_delta),
        "home_vc_delta": _r(vc, 2),
        "mode_baseline": idx.get(p.person_id, "mode_baseline"),
        "mode_plan": idx.get(p.person_id, "mode_plan"),
    }


def approval_score(values: list[str], deltas: dict[str, Any], tool_ids: list[str],
                   params: ApprovalParams) -> float:
    m = params.multipliers
    vs = set(values)

    def mult(v: str) -> float:
        return m.get(v, 1.0) if v in vs else 1.0

    commute = deltas.get("commute_min")
    if commute is None:
        commute = deltas.get("kid_trip_min")
    commute = commute or 0.0
    dropoff = deltas.get("dropoff_min") or 0.0
    score = params.base
    score += params.per_min_commute_saved * (-commute) * mult("time")
    score += params.per_min_dropoff_saved * (-dropoff) * mult("time")
    score += params.per_100usd_cost * (float(deltas.get("cost_usd_year") or 0.0) / 100.0) * mult("cost")
    if deltas.get("street_change"):
        score += params.street_change_penalty * mult("property")
    if "change_averse" in vs and tool_ids:
        score += params.change_averse_penalty
    tools = list(tool_ids)
    if "safety" in vs:
        score += params.safety_tool_bonus * mult("safety") * sum(t in SAFETY_TOOLS for t in tools)
    if "community" in vs:
        score += params.safety_tool_bonus * mult("community") * sum(t in COMMUNITY_TOOLS for t in tools)
    if "environment" in vs:
        score += params.safety_tool_bonus * mult("environment") * sum(t in ENVIRONMENT_TOOLS for t in tools)
    return score


def logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def compute_reactions(personas: list[Persona], person_deltas: dict[str, Any] | None,
                      plan: dict[str, Any], cost_per_year_usd: float,
                      params: ApprovalParams | None = None) -> list[dict[str, Any]]:
    """Deterministic approval for every persona. Same inputs -> same outputs."""
    params = params or ApprovalParams.load()
    idx = DeltaIndex(person_deltas)
    tool_ids = [str(t.get("tool")) for t in plan.get("tools") or []]
    out = []
    for p in personas:
        d = persona_deltas(p, idx, cost_per_year_usd, params)
        approval = logistic(approval_score(p.values, d, tool_ids, params))
        out.append({"persona_id": p.persona_id, "approval": round(approval, 3),
                    "approves": approval >= 0.5, "deltas": d, "text": None})
    return out


def approval_summary(reactions: list[dict[str, Any]], seed: int | None = None) -> dict[str, float | None]:
    """Approval percent plus a bootstrap p10/p90 over personas (fixed seed)."""
    if not reactions:
        return {"median": None, "p10": None, "p90": None}
    a = np.array([1.0 if r["approves"] else 0.0 for r in reactions])
    seed = int(assumption("residents.persona_seed")) + BOOTSTRAP_SEED_OFFSET if seed is None else seed
    rng = np.random.default_rng(seed)
    boots = rng.choice(a, size=(BOOTSTRAP_SAMPLES, len(a)), replace=True).mean(axis=1) * 100
    lo = float(assumption("report.percentile_low"))
    hi = float(assumption("report.percentile_high"))
    return {"median": round(float(a.mean() * 100), 1),
            "p10": round(float(np.percentile(boots, lo)), 1),
            "p90": round(float(np.percentile(boots, hi)), 1)}


def approval_metric(summary: dict[str, float | None], n_personas: int) -> dict[str, Any]:
    empty = {"median": None, "p10": None, "p90": None}
    return {
        "id": "resident_approval_pct", "label": "Resident approval", "unit": "%", "better": "higher",
        "baseline": dict(empty), "plan": summary, "delta": dict(empty),
        "note": f"Share of {n_personas} AI residents who approve (deterministic from their personal "
                "time and cost changes; range is a bootstrap over residents). No baseline value.",
    }


def insert_approval_metric(report: dict[str, Any], metric: dict[str, Any]) -> dict[str, Any]:
    metrics = [m for m in report.get("metrics") or []
               if m.get("id") not in ("resident_approval_pct", "resident_approval")]
    metrics.append(metric)
    report["metrics"] = metrics
    return report


# ----------------------------------------------------------------- LLM text
_REACTION_SYSTEM = (
    "You voice fictional residents reacting to a proposed neighborhood traffic plan in a city "
    "planning game. For each resident write ONE or TWO short first-person sentences in plain, "
    "natural words, in character, consistent with whether they approve. Use only the numbers "
    "given in that resident's facts (you may round them); never invent any other number, "
    "street address, surname or fact. Reply with JSON only: "
    '{"reactions": [{"id": <id>, "text": "..."}]}'
)


def describe_deltas(d: dict[str, Any]) -> dict[str, str]:
    """Plain-language deltas (absolute values + direction) for prompts."""
    out: dict[str, str] = {}
    for key, label in (("commute_min", "own commute"), ("dropoff_min", "school drop-off"),
                       ("kid_trip_min", "own trip to school")):
        v = d.get(key)
        if v is None:
            continue
        if abs(v) < 0.5:
            out[label] = "about the same"
        else:
            out[label] = f"{abs(v):.1f} minutes {'slower' if v > 0 else 'faster'}"
    cost = d.get("cost_usd_year")
    if cost:
        out["household share of yearly plan cost"] = f"about ${cost:.0f} per year"
    if d.get("street_change"):
        out["own street"] = "noticeably more traffic"
    if d.get("mode_baseline") and d.get("mode_plan") and d["mode_baseline"] != d["mode_plan"]:
        out["travel mode"] = f"switches from {d['mode_baseline']} to {d['mode_plan']}"
    return out


def reaction_facts(p: Persona, reaction: dict[str, Any], school_names: dict[str, str]) -> dict[str, Any]:
    f = p.facts(school_names)
    f["changes_under_plan"] = describe_deltas(reaction.get("deltas") or {})
    f["approves"] = bool(reaction["approves"])
    return f


def plan_brief(plan: dict[str, Any], tool_names: dict[str, str] | None = None) -> dict[str, Any]:
    names = tool_names or {}
    return {"title": plan.get("title", ""), "pitch": plan.get("pitch", ""),
            "tools": [names.get(str(t.get("tool")), str(t.get("tool"))) for t in plan.get("tools") or []]}


def generate_texts(personas: dict[int, Persona], reactions: list[dict[str, Any]], plan: dict[str, Any],
                   llm: ChatLLM, school_names: dict[str, str], tool_names: dict[str, str] | None = None,
                   max_workers: int = 4) -> dict[int, str]:
    """Batched LLM reactions. Returns persona_id -> text for valid replies only."""
    if not llm.available:
        return {}
    brief = plan_brief(plan, tool_names)
    todo = [r for r in reactions if r["persona_id"] in personas]
    batches = [todo[i:i + REACTION_BATCH] for i in range(0, len(todo), REACTION_BATCH)]

    def run(batch: list[dict[str, Any]]) -> dict[int, str]:
        if not llm.available:
            return {}
        facts = {r["persona_id"]: reaction_facts(personas[r["persona_id"]], r, school_names) for r in batch}
        user = json.dumps({"plan": brief, "residents": [{"id": k, **v} for k, v in facts.items()]})
        reply = llm.chat([{"role": "system", "content": _REACTION_SYSTEM},
                          {"role": "user", "content": user}],
                         model="fast", max_tokens=90 * len(batch), temperature=0.7, json_mode=True)
        data = extract_json(reply)
        items = data.get("reactions", []) if isinstance(data, dict) else []
        res: dict[int, str] = {}
        for it in items:
            try:
                pid = int(it["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if pid not in facts:
                continue
            nums, clocks = collect_numbers({"facts": facts[pid], "plan": brief})
            text = safe_text(it.get("text"), nums, clocks, max_chars=400)
            if text:
                res[pid] = text
        return res

    out: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(run, batches):
            out.update(res)
    return out
