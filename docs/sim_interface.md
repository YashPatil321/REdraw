# Sim engine interface (sim -> api / residents)

The API and residents packages call the sim ONLY through `sim/service.py`.
Everything here is plain Python + Pydantic so it can run in a process pool.

```python
from sim.service import SimService, RunResult, PlanCheck

svc = SimService.load()                 # loads WorldState from REDRAW_DATA_DIR once (~seconds)

svc.world_summary() -> dict             # counts, synthetic flag, calibration status (see below)
svc.tools() -> list[dict]               # tools.yaml entries enabled for the mission (data driven UI)
svc.mission(mission_id="morning_crunch") -> dict   # mission yaml as dict
svc.evaluate_goals(mission_id, report) -> list[dict]  # [{text, status: met|missed|unknown, detail}] (sim/goals.py)
svc.schools() -> list[dict]             # schools_resolved.json entries
svc.network_json() -> dict              # see GET /world/network in docs/api.md
svc.unverified_inputs() -> list[str]    # human readable list of verified:false inputs used

svc.check_plan(plan: dict) -> PlanCheck
#   Validates tool params against tools.yaml (Pydantic), resolves map inputs
#   (snaps points to edges/nodes), computes cost, budget and constraint status.
#   Never raises for player mistakes; returns errors instead.

svc.run(plan: dict | None, seeds: int = 20, workers: int = 4,
        progress: Callable[[float, str], None] | None = None) -> RunResult
#   plan=None runs the baseline only. Otherwise runs baseline AND plan with the
#   same seeds (common random numbers) and builds the report card.
#   Must finish 20 seeds x 2 in < 3 min with 4 workers; 1 seed < 20 s.
#   The baseline result for a given seed list is cached in memory and on disk
#   (data/processed/cache/baseline_<hash>.pkl) keyed by data+assumptions hash.
```

```python
class PlanCheck(BaseModel):
    ok: bool                       # False if any error (plan cannot run)
    errors: list[str]              # e.g. "bell_time: unknown school 'xyz'"
    warnings: list[str]            # e.g. "turn_lane on edge 12 is a through lane class"
    cost_upfront_usd: float
    cost_per_year_usd: float
    over_budget: bool
    budget_upfront_usd: float
    budget_per_year_usd: float
    constraint_violations: list[str]   # e.g. ["no_new_through_lanes"]
    resolved_tools: list[dict]     # tools with snapped map inputs (edge_idx/node_id added)

class RunResult(BaseModel):
    report: dict | None            # report card JSON (docs/api.md "Report"), None for baseline-only
    baseline_summary: dict         # same metric block shape for baseline alone
    playback_plan: bytes | None    # RDPB v1 for seed index 0 of the plan
    playback_baseline: bytes       # RDPB v1 for seed index 0 of the baseline
    person_deltas: dict            # for residents: see below
```

`person_deltas` (median across seeds, minutes; positive = plan is slower/worse):

```python
{
  "person_ids": list[int],
  "commute_delta_min": list[float],   # worker's own commute (NaN if not commuting)
  "dropoff_delta_min": list[float],   # drop-off leg delay for the household's driver (NaN if none)
  "kid_trip_delta_min": list[float],  # the student's own school trip (NaN if not a student)
  "mode_baseline": list[str],
  "mode_plan": list[str],
  "home_edges_vc_delta": list[float], # change in max v/c on the edges nearest home (street change)
}
```

Residents use `person_deltas` to compute approval deterministically
(`residents/reactions.py`); the API inserts the approval metric into the
report (`metrics[id="resident_approval_pct"]`) and the winners/losers counts
come from the sim: a person counts as a winner (loser) when the median across seeds of their total-minute change is <= -3 (>= +3); p10/p90 count people who win (lose) on at least 90% / 10% of seeds.

## Calibration status (in `world_summary()["calibration"]`)

```json
{"status": "uncalibrated" | "calibrated" | "above_target",
 "median_error_pct": null, "targets_with_data": 0, "targets_total": 6,
 "table": [{"id": "dsur_to_i15", "label": "...", "observed_min": null, "simulated_min": 8.4, "error_pct": null}]}
```
Written by `python -m sim.calibrate` to `data/processed/calibration.json`.
