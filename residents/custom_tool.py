"""Custom tool preview (spec 7.3).

The smart LLM converts a player's free-text idea into JSON levers the sim
already understands (mode utility shifts, capacity changes) plus cost, an
adoption range and plain-word assumptions. The output is validated with
Pydantic; anything invalid is rejected (the plan must not run). The result is
always labeled "LLM estimated".
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from pipeline.config import assumption, assumptions
from residents.llm import ChatLLM, extract_json

# Validation bounds that keep LLM levers inside what the model can represent
# (safety rails on the LLM, not real-world estimates). The sim re-checks the same
# bounds (sim.plan._levers_from_estimate) when the confirmed estimate runs.
MAX_UTILITY_SHIFT = float(assumption("sim_engine.custom_max_utility_shift"))
CAPACITY_FACTOR_RANGE = (float(assumption("sim_engine.custom_min_capacity_factor")),
                         float(assumption("sim_engine.custom_max_capacity_factor")))
MAX_COST_USD = 50_000_000


class CustomToolError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def known_modes() -> list[str]:
    mc = assumptions().get("modechoice", {})
    modes = set(mc.get("asc_student", {})) | set(mc.get("asc_worker", {}))
    return sorted(modes)


class ModeUtilityShift(BaseModel):
    type: Literal["mode_utility_shift"]
    mode: str
    applies_to: Literal["students", "workers", "all"] = "all"
    school: str | None = None
    utils: float = Field(ge=-MAX_UTILITY_SHIFT, le=MAX_UTILITY_SHIFT)

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in known_modes():
            raise ValueError(f"unknown mode '{v}' (known: {', '.join(known_modes())})")
        return v


class CapacityChange(BaseModel):
    type: Literal["capacity_change"]
    target: Literal["edge", "entrance"]
    edge_idx: int | None = Field(default=None, ge=0)
    entrance: str | None = None
    factor: float = Field(ge=CAPACITY_FACTOR_RANGE[0], le=CAPACITY_FACTOR_RANGE[1])

    @model_validator(mode="after")
    def _target(self) -> CapacityChange:
        if self.target == "edge" and self.edge_idx is None:
            raise ValueError("capacity_change on an edge needs edge_idx")
        if self.target == "entrance" and not self.entrance:
            raise ValueError("capacity_change on an entrance needs entrance '<school>/<entrance>'")
        return self


Lever = Annotated[ModeUtilityShift | CapacityChange, Field(discriminator="type")]


class CustomToolEstimate(BaseModel):
    summary: str = Field(min_length=1, max_length=400)
    levers: list[Lever] = Field(min_length=1, max_length=8)
    adoption_range: tuple[float, float]
    cost_upfront_usd: float = Field(ge=0, le=MAX_COST_USD)
    cost_per_year_usd: float = Field(ge=0, le=MAX_COST_USD)
    assumptions: list[str] = Field(min_length=1, max_length=10)

    @field_validator("adoption_range")
    @classmethod
    def _adoption(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError("adoption_range must be [low, high] shares with 0 <= low <= high <= 1")
        return v


def _system_prompt(schools: list[str], entrances: list[str]) -> str:
    return (
        "You convert a player's idea for reducing school drop-off and commute traffic into model "
        "levers for a traffic simulation. Reply with JSON only, matching exactly:\n"
        '{"summary": str, "levers": [lever, ...], "adoption_range": [low, high], '
        '"cost_upfront_usd": number, "cost_per_year_usd": number, "assumptions": [str, ...]}\n'
        "A lever is one of:\n"
        f'  {{"type": "mode_utility_shift", "mode": one of {known_modes()}, '
        f'"applies_to": "students"|"workers"|"all", "school": one of {schools} or null, '
        f'"utils": number between -{MAX_UTILITY_SHIFT} and {MAX_UTILITY_SHIFT}}}\n'
        f'  {{"type": "capacity_change", "target": "entrance", "entrance": one of {entrances}, '
        f'"factor": number between {CAPACITY_FACTOR_RANGE[0]} and {CAPACITY_FACTOR_RANGE[1]}}}\n'
        f'  {{"type": "capacity_change", "target": "edge", "edge_idx": int, "factor": number}}\n'
        "adoption_range is the share of affected people who change behavior (0 to 1). "
        "assumptions are short plain-English statements of everything you guessed. "
        "Be conservative and honest; if the idea cannot be represented, use the closest small lever "
        "and say so in assumptions."
    )


def preview(description: str, llm: ChatLLM, schools: list[dict[str, Any]]) -> dict[str, Any]:
    if not llm.available:
        raise CustomToolError(503, "The AI endpoint is unavailable, so custom ideas cannot be estimated right now.")
    school_ids = [s["id"] for s in schools]
    entrances = [f"{s['id']}/{e['id']}" for s in schools for e in s.get("entrances", [])]
    reply = llm.chat([{"role": "system", "content": _system_prompt(school_ids, entrances)},
                      {"role": "user", "content": description}],
                     model="smart", max_tokens=700, temperature=0.2, json_mode=True)
    if reply is None:
        raise CustomToolError(503, "The AI endpoint did not answer; try again later.")
    data = extract_json(reply)
    if not isinstance(data, dict):
        raise CustomToolError(422, "The AI reply was not valid JSON; the idea was not converted.")
    try:
        est = CustomToolEstimate.model_validate(data)
    except ValidationError as e:
        msgs = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:5])
        raise CustomToolError(422, f"The AI estimate failed validation ({msgs}); it cannot be run.") from e
    for lv in est.levers:
        if isinstance(lv, ModeUtilityShift) and lv.school and lv.school not in school_ids:
            raise CustomToolError(422, f"The AI estimate referenced unknown school '{lv.school}'.")
        if isinstance(lv, CapacityChange) and lv.entrance and lv.entrance not in entrances:
            raise CustomToolError(422, f"The AI estimate referenced unknown entrance '{lv.entrance}'.")
    out = json.loads(est.model_dump_json())
    return {
        "ok": True,
        "label": "LLM estimated",
        "description": description,
        "estimate": out,
        # What the client puts in plan.tools after the player confirms.
        "tool": {"tool": "custom", "params": {"description": description, "estimate": out}},
    }
