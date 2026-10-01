"""FakeSimService implementing docs/sim_interface.md, for API tests without the sim or real data."""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from pipeline.config import load_yaml
from residents.tests.fixtures import fake_person_deltas


class PlanCheck(BaseModel):
    ok: bool
    errors: list[str]
    warnings: list[str]
    cost_upfront_usd: float
    cost_per_year_usd: float
    over_budget: bool
    budget_upfront_usd: float
    budget_per_year_usd: float
    constraint_violations: list[str]
    resolved_tools: list[dict]


class RunResult(BaseModel):
    report: dict | None
    baseline_summary: dict
    playback_plan: bytes | None
    playback_baseline: bytes
    person_deltas: dict


def encode_rdpb(plan_id: str, n_edges: int = 4) -> bytes:
    header = {"version": 1, "plan_id": plan_id, "seed": 0, "synthetic": True, "bin_start_s": 21600,
              "bin_s": 300, "n_bins": 1, "report_start_s": 23400, "report_end_s": 34200,
              "n_edges": n_edges, "entrance_ids": [], "n_trajectories": 0, "n_points": 0, "kinds": {},
              "sections": [{"name": "edge_vc", "dtype": "float32", "offset": 0, "count": n_edges}]}
    hj = json.dumps(header).encode()
    pad = (-(12 + len(hj))) % 4
    return b"RDPB" + struct.pack("<II", 1, len(hj)) + hj + b"\0" * pad + struct.pack(f"<{n_edges}f", *([0.5] * n_edges))


class FakeSimService:
    def __init__(self, data_dir: Path, delay_s: float = 0.0) -> None:
        self.data_dir = data_dir
        self.delay_s = delay_s
        self.persons = pd.read_parquet(data_dir / "persons.parquet")
        self.schools_data = json.loads((data_dir / "schools_resolved.json").read_text())["schools"]
        self.runs: list[Any] = []

    @classmethod
    def load(cls) -> FakeSimService:  # pragma: no cover - tests construct directly
        raise NotImplementedError

    def world_summary(self) -> dict:
        return {"counts": {"households": 900}, "synthetic": True,
                "calibration": {"status": "uncalibrated", "median_error_pct": None,
                                "targets_with_data": 0, "targets_total": 6, "table": []}}

    def tools(self) -> list[dict]:
        return [t for t in load_yaml("tools.yaml")["tools"] if t["id"] != "custom"]

    def mission(self, mission_id: str = "morning_crunch") -> dict:
        return load_yaml(f"missions/{mission_id}.yaml")

    def schools(self) -> list[dict]:
        return self.schools_data

    def network_json(self) -> dict:
        return {"n_edges": 1, "edges": [{"i": 0, "u": 1, "v": 2, "name": "Test Lane", "label": "",
                                         "highway": "secondary", "lanes": 2, "len": 100.0,
                                         "pts": [0.0, 300.0, 0.0, 100.0, 300.0, 0.0]}],
                "nodes": [{"id": 1, "x": 0.0, "y": 300.0, "z": 0.0, "signal": False},
                          {"id": 2, "x": 100.0, "y": 300.0, "z": 0.0, "signal": True}]}

    def unverified_inputs(self) -> list[str]:
        return ["Test High bell time 08:00 (fake)"]

    def check_plan(self, plan: dict) -> PlanCheck:
        errors = []
        known = {t["id"] for t in self.tools()}
        school_ids = {s["id"] for s in self.schools_data}
        per_year = 0.0
        for t in plan.get("tools", []):
            if t["tool"] not in known:
                errors.append(f"{t['tool']}: unknown tool")
            elif t["tool"] == "bell_time" and t["params"].get("school") not in school_ids:
                errors.append(f"bell_time: unknown school '{t['params'].get('school')}'")
            elif t["tool"] == "school_shuttle":
                per_year += 95000.0 * int(t["params"].get("buses", 1))
        return PlanCheck(ok=not errors, errors=errors, warnings=[], cost_upfront_usd=0.0,
                         cost_per_year_usd=per_year, over_budget=per_year > 400000,
                         budget_upfront_usd=2e6, budget_per_year_usd=4e5, constraint_violations=[],
                         resolved_tools=plan.get("tools", []))

    def _summary(self) -> dict:
        return {"metrics": [{"id": "avg_commute_min", "label": "Average commute time", "unit": "min",
                             "better": "lower", "value": {"median": 21.0, "p10": 20.0, "p90": 22.0}}],
                "per_school": [], "mode_share": [],
                "per_entrance": {"del_norte_hs/main_dropoff": {"max_queue_cars": 12.0, "max_spillback_m": 40.0,
                                                               "avg_wait_min": 3.0}}}

    def run(self, plan: dict | None, seeds: int = 20, workers: int = 4,
            progress: Callable[[float, str], None] | None = None) -> RunResult:
        self.runs.append(plan)
        for i in range(seeds):
            if self.delay_s:
                time.sleep(self.delay_s / max(1, seeds))
            if progress:
                progress((i + 1) / seeds, f"seed {i + 1}/{seeds}")
        if plan is None:
            return RunResult(report=None, baseline_summary=self._summary(), playback_plan=None,
                             playback_baseline=encode_rdpb("baseline"), person_deltas={})
        n_bell = sum(1 for t in plan["tools"] if t["tool"] == "bell_time")
        commute = 21.0 - 0.5 * n_bell - 0.01 * len(plan.get("title", ""))
        check = self.check_plan(plan)
        report = {
            "plan_id": plan["id"], "seeds": seeds, "synthetic": True,
            "cost": {"upfront_usd": check.cost_upfront_usd, "per_year_usd": check.cost_per_year_usd,
                     "over_budget": check.over_budget, "budget_upfront_usd": 2e6, "budget_per_year_usd": 4e5,
                     "lines": []},
            "constraint_violations": [],
            "metrics": [{"id": "avg_commute_min", "label": "Average commute time", "unit": "min",
                         "better": "lower", "baseline": {"median": 21.0, "p10": 20.0, "p90": 22.0},
                         "plan": {"median": commute, "p10": commute - 1, "p90": float("nan")},
                         "delta": {"median": commute - 21.0, "p10": None, "p90": None}}],
            "winners": {"median": 10, "p10": 8, "p90": 12}, "losers": {"median": 2, "p10": 1, "p90": 3},
        }
        return RunResult(report=report, baseline_summary=self._summary(),
                         playback_plan=encode_rdpb(plan["id"]), playback_baseline=encode_rdpb("baseline"),
                         person_deltas=fake_person_deltas(self.persons, plan["tools"]))
