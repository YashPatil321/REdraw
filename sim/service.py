"""SimService: the only interface api/ and residents/ use (docs/sim_interface.md)."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pipeline.config import assumptions, region
from sim.goals import evaluate_goals
from sim.plan import PlanCheck, apply_plan, check_plan, mission_def, tool_defs
from sim.playback import DEFAULT_KINDS, playback_from_seed
from sim.report import METRIC_DEFS, baseline_summary, get_baseline, run_report
from sim.world import WorldState, load_world

log = logging.getLogger("sim.service")

__all__ = ["SimService", "RunResult", "PlanCheck", "PlanInvalidError"]

# assumption sections the sim reads; their verified:false leaves are listed as unverified inputs
_SIM_SECTIONS = ("assignment", "time_of_day", "demand", "modechoice", "schools", "sim_engine", "roads.through_lane_classes")
_TOOL_SECTIONS = ("costs", "tool_effects")


class PlanInvalidError(ValueError):
    pass


class RunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report: dict[str, Any] | None
    baseline_summary: dict[str, Any]
    playback_plan: bytes | None
    playback_baseline: bytes
    person_deltas: dict[str, Any]
    # additions beyond docs/sim_interface.md
    plan_check: PlanCheck | None = None
    timings: dict[str, float] = Field(default_factory=dict)


def _leaves(node: Any, prefix: str) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(node, dict) and "value" in node and "verified" in node:
        return [(prefix, node)]
    out: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.extend(_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


class SimService:
    def __init__(self, world: WorldState) -> None:
        self.world = world
        self._network_json: dict[str, Any] | None = None
        self._lock = threading.Lock()

    # ---- construction ----------------------------------------------------------
    @classmethod
    def load(cls, data_dir: str | Path | None = None) -> SimService:
        """Load WorldState from REDRAW_DATA_DIR (or ``data_dir``) once."""
        return cls(load_world(data_dir))

    # ---- static info -------------------------------------------------------------
    def calibration(self) -> dict[str, Any]:
        p = self.world.data_dir / "calibration.json"
        if p.exists():
            try:
                cal = json.loads(p.read_text())
                cal.setdefault("status", "uncalibrated")
                return cal
            except json.JSONDecodeError:
                log.warning("unreadable calibration.json")
        from sim.calibrate import load_targets

        targets = load_targets()
        return {"status": "uncalibrated", "median_error_pct": None,
                "targets_with_data": sum(1 for t in targets if t.get("observed_minutes") is not None),
                "targets_total": len(targets), "table": []}

    def world_summary(self) -> dict[str, Any]:
        w = self.world
        hero = w.hero_school
        meta = w.meta
        return {
            "name": meta.get("name"), "display_name": meta.get("display_name"), "synthetic": w.synthetic,
            "region": {"name": meta.get("name"), "display_name": meta.get("display_name"), "bbox": meta.get("bbox"),
                       "origin": meta.get("origin"), "timezone": region().get("timezone", "America/Los_Angeles"),
                       "extent_scene": meta.get("extent_scene")},
            "counts": {**meta.get("counts", {}), "sim_road_edges": w.net.n_edges, "sim_road_nodes": w.net.n_nodes,
                       "sim_persons": len(w.persons), "sim_households": len(w.households),
                       "students_by_school": {s.id: int((w.persons.school == i).sum()) for i, s in enumerate(w.schools)}},
            "calibration": self.calibration(),
            "time": w.time.as_dict(),
            "metrics": self.metric_defs(),
            "hero": {"school_id": hero.id, "x": hero.x, "z": hero.z} if hero else None,
            "data_hash": w.data_hash,
        }

    def metric_defs(self) -> list[dict[str, str]]:
        return [dict(m) for m in METRIC_DEFS]

    def tools(self, mission_id: str = "morning_crunch") -> list[dict[str, Any]]:
        """tools.yaml entries for the mission, in mission order (custom included with enabled_in_mvp: false)."""
        defs = tool_defs()
        out = []
        for tid in mission_def(mission_id).get("tools", []):
            if tid in defs:
                t = json.loads(json.dumps(defs[tid]))
                t.setdefault("enabled_in_mvp", True)
                out.append(t)
        return out

    def mission(self, mission_id: str = "morning_crunch") -> dict[str, Any]:
        return json.loads(json.dumps(mission_def(mission_id)))

    def evaluate_goals(self, mission_id: str, report: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Score a finished report against the mission's suggested goals (hints, not pass/fail)."""
        return evaluate_goals(mission_def(mission_id), report)

    def schools(self) -> list[dict[str, Any]]:
        net = self.world.net
        out = []
        for raw in self.world.schools_raw:
            s = json.loads(json.dumps(raw))
            for e in s.get("entrances", []):
                e["key"] = f"{s['id']}/{e.get('id')}"
                nid = e.get("node_id")
                if nid is not None:
                    ni = int(net.node_index([int(nid)])[0])
                    if ni >= 0:
                        e.setdefault("y", float(net.y[ni]))
            if "x" in s and "z" in s:
                ni = int(net.nearest_node(s["x"], s["z"])[0])
                s.setdefault("y", float(net.y[ni]))
            out.append(s)
        return out

    def network_json(self) -> dict[str, Any]:
        if self._network_json is None:
            net = self.world.net
            edges = []
            for e in range(net.n_edges):
                pts = net.edge_geometry(e)
                edges.append({"i": e, "u": int(net.node_ids[net.eu[e]]), "v": int(net.node_ids[net.ev[e]]),
                              "name": str(net.name[e]), "label": str(net.label[e]), "highway": str(net.highway[e]),
                              "lanes": int(net.lanes[e]), "len": round(float(net.length_m[e]), 1),
                              "pts": [round(float(v), 1) for v in pts]})
            nodes = [{"id": int(net.node_ids[i]), "x": round(float(net.x[i]), 1), "y": round(float(net.y[i]), 1),
                      "z": round(float(net.z[i]), 1), "signal": bool(net.signalized[i])} for i in range(net.n_nodes)]
            self._network_json = {"n_edges": net.n_edges, "edges": edges, "nodes": nodes}
        return self._network_json

    def unverified_inputs(self, plan_tools: list[str] | None = None) -> list[str]:
        """Human readable list of verified:false inputs used by the sim."""
        w = self.world
        out: list[str] = []
        if w.synthetic:
            out.append("SYNTHETIC DEV DATA: the world is procedurally generated, not real geography or population")
        for s in w.schools:
            if not s.verified:
                out.append(f"{s.name} bell time {s.bell_clock}")
            for ei in s.entrances:
                e = w.entrances[ei]
                if not e.verified and not e.added_by_plan:
                    out.append(f"{s.name} {e.id}: {e.curb_spots:g} curb spots, {e.unload_s:g} s unload per car")
        reg = region()
        if not reg.get("bbox_verified", True):
            out.append("Region bounding box (region.yaml)")
        for ex in reg.get("exits", []):
            if not ex.get("verified", True):
                out.append(f"Boundary exit {ex.get('label', ex.get('id'))} location")
        sections = list(_SIM_SECTIONS) + (list(_TOOL_SECTIONS) if plan_tools else [])
        a = assumptions()
        for sec in sections:
            node: Any = a
            for part in sec.split("."):
                node = node.get(part, {}) if isinstance(node, dict) else {}
            for key, lf in _leaves(node, sec):
                if not lf.get("verified", False):
                    out.append(f"Assumption {key} = {lf.get('value')} {lf.get('unit', '')} ({lf.get('source', '')})".replace("  ", " "))
        return out

    # ---- plans -------------------------------------------------------------------
    def check_plan(self, plan: dict[str, Any]) -> PlanCheck:
        return check_plan(self.world, plan)

    def baseline_summary(self, seeds: int = 20, workers: int = 4,
                         progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
        base = get_baseline(self.world, list(range(seeds)), workers, progress)
        return baseline_summary(self.world, base, self.calibration())

    def run(self, plan: dict[str, Any] | None, seeds: int = 20, workers: int = 4,
            progress: Callable[[float, str], None] | None = None) -> RunResult:
        w = self.world
        check = None
        plan_world = None
        if plan is not None:
            check = self.check_plan(plan)
            if not check.ok:
                raise PlanInvalidError("plan has errors: " + "; ".join(check.errors))
            plan_world = apply_plan(w, check)
        with self._lock:
            out = run_report(
                w, plan_world, check, plan_id=str((plan or {}).get("id") or "plan"), seeds=seeds, workers=workers,
                calibration=self.calibration(), unverified=self.unverified_inputs([t["tool"] for t in check.resolved_tools] if check else None),
                progress=progress,
            )
        tdict = w.time.as_dict()
        pb_base = playback_from_seed(out["baseline"][0].playback, plan_id="baseline", seed=0, synthetic=w.synthetic,
                                     time=tdict, kinds=DEFAULT_KINDS)
        pb_plan = None
        if plan_world is not None:
            pb_plan = playback_from_seed(out["plan"][0].playback, plan_id=str(plan.get("id") or "plan"), seed=0,
                                         synthetic=w.synthetic, time=tdict, kinds=DEFAULT_KINDS)
        return RunResult(report=out["report"], baseline_summary=out["baseline_summary"], playback_plan=pb_plan,
                         playback_baseline=pb_base, person_deltas=out["person_deltas"], plan_check=check,
                         timings={"wall_s": out["wall_s"]})
