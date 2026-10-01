# API contract (v1)

FastAPI app `api.main:app`, default `http://localhost:8000`. The Vite dev server
proxies `/api/*` to the API (stripping `/api`), so the client uses base `/api`.
All JSON. Anonymous player id: cookie `redraw_player` (UUID), set by any
response if missing. Errors: `{"detail": "..."}` with 4xx.

## World

### GET /world/meta
```json
{
  "region": {"name": "4s_ranch_del_sur", "display_name": "...", "bbox": {...},
             "origin": {"lat": 33.005, "lon": -117.125}, "timezone": "America/Los_Angeles",
             "extent_scene": {"min_x": ..., "max_x": ..., "min_z": ..., "max_z": ...}},
  "synthetic": true,
  "assets": {"base_url": "/assets/", "manifest": "/assets/manifest.json"},
  "counts": {...},
  "calibration": {"status": "uncalibrated", "median_error_pct": null, ...},
  "mission": {"id": "morning_crunch", "title": "...", "brief": "...", "budget_usd_upfront": 2000000,
              "budget_usd_per_year": 400000, "constraints": [...], "goals_suggested": [...]},
  "metrics": [{"id": "avg_commute_min", "label": "Average commute time", "unit": "min", "better": "lower"}],
  "time": {"bin_start_s": 21600, "bin_s": 300, "n_bins": 48, "report_start_s": 23400, "report_end_s": 34200},
  "hero": {"school_id": "del_norte_hs", "x": 2243.6, "z": -1826.8},
  "unverified_inputs": ["Del Norte High School bell time 08:30", "..."]
}
```
Asset URLs are relative to the API base (client prefixes `/api`).

### GET /world/network
Edge polylines for playback and map snapping.
```json
{"n_edges": 8123,
 "edges": [{"i": 0, "u": 1, "v": 2, "name": "Camino Del Sur", "label": "Camino Del Sur",
            "highway": "primary", "lanes": 3, "len": 412.5, "pts": [x0,y0,z0,x1,y1,z1]}],
 "nodes": [{"id": 1, "x": 0.0, "y": 300.0, "z": 0.0, "signal": true}]}
```

### GET /world/buildings/{id}
`buildings.geojson` properties for that id plus `"households": n` and
`"block": "<block label>"`. 404 if unknown.

### GET /world/schools
```json
[{"id": "del_norte_hs", "name": "...", "grades": [9,12], "bell_start": "08:30", "x": .., "z": .., "y": ..,
  "verified": false, "students": 2500,
  "entrances": [{"id": "main_dropoff", "key": "del_norte_hs/main_dropoff", "x": .., "z": .., "y": ..,
                 "curb_spots": 10, "unload_seconds": 45, "verified": false,
                 "baseline": {"max_queue_cars": 31.0, "max_spillback_m": 217.0, "avg_wait_min": 6.2}}]}]
```

## Baseline

### GET /baseline
`{"summary": <MetricBlock>, "playback_url": "/baseline/playback", "calibration": {...}}`

### GET /baseline/playback
RDPB v1 binary (docs/playback_format.md).

## Tools

### GET /tools
`{"mission": "morning_crunch", "tools": [<tools.yaml entries, custom included with enabled_in_mvp:false>]}`

## Plans

Plan JSON (spec 7.2): `{id, mission, title, pitch, author_id, tools: [{tool, params}], created_at, report, status, check, votes}`
- `status`: `draft | queued | running | done | failed`
- `check`: PlanCheck (docs/sim_interface.md) computed on create/update

### POST /plans
Body: `{mission, title, pitch, tools}` -> 201 plan (with `check`). Invalid tool
params still save (status draft) but `check.ok=false`. Pydantic validation of
shape (tools must be a list of `{tool: str, params: object}`) -> 422.

### POST /plans/check
Body: same as POST /plans. Returns PlanCheck without saving (live budget bar).

### GET /plans/{id}
Plan with `report` when done.

### POST /plans/{id}/run
-> `{"job_id": "..."}`. 400 if `check.ok` is false.

### GET /jobs/{id}
`{"id", "plan_id", "status": "queued|running|done|failed", "progress": 0.0-1.0, "message": "seed 7/20", "error": null}`

### GET /plans/{id}/playback
RDPB v1 binary of the plan (seed 0). 404 until done.

### GET /plans/{id}/residents
`{"approval_pct": 57.2, "reactions": [{"persona_id": 12, "first_name": "Maya", "age": 41, "block": "Del Sur, near Camino Del Sur",
  "x": .., "z": .., "school_ids": ["del_norte_hs"], "values": ["time","safety","cost"],
  "approval": 0.71, "approves": true, "deltas": {"commute_min": -3.1, "dropoff_min": -6.0, "cost_usd_year": 12.0, "street_change": false},
  "text": "..." | null}]}`
`text` is null when the LLM endpoint is down (spec 8.5).

### GET /plans?mission=&sort=
`sort` = `new` (default) | `votes` | any metric id (ascending if `better=lower`).
Returns `{"plans": [{id, title, pitch, created_at, votes, status, headline: {metric_id: plan_median}}]}`.

### POST /plans/{id}/vote
Body `{"value": 1 | -1 | 0}` -> `{"votes": n}`. (M7, implemented server side.)

## Report (plan.report)

```json
{
  "plan_id": "...", "seeds": 20, "generated_at": "iso", "synthetic": true,
  "cost": {"upfront_usd": 0, "per_year_usd": 0, "over_budget": false,
           "budget_upfront_usd": 2000000, "budget_per_year_usd": 400000, "lines": [{"tool": "school_shuttle", "upfront_usd": 0, "per_year_usd": 190000, "note": "..."}]},
  "constraint_violations": [],
  "metrics": [
    {"id": "avg_commute_min", "label": "Average commute time, all workers", "unit": "min", "better": "lower",
     "baseline": {"median": 21.3, "p10": 20.8, "p90": 22.0},
     "plan": {"median": 20.1, "p10": 19.6, "p90": 20.9},
     "delta": {"median": -1.2, "p10": -1.6, "p90": -0.8}}
  ],
  "per_school": [{"school_id": "del_norte_hs", "name": "...",
     "dropoff_delay_min": {"baseline": {...}, "plan": {...}},
     "max_spillback_m": {"baseline": {...}, "plan": {...}},
     "late_kids": {"baseline": {...}, "plan": {...}}}],
  "mode_share": [{"mode": "drive_dropoff", "baseline_pct": 61.0, "plan_pct": 55.2, "delta_pp": -5.8}],
  "winners": {"median": 2100, "p10": 1900, "p90": 2300},
  "losers": {"median": 300, "p10": 250, "p90": 380},
  "side_effects": [{"edge_idx": 812, "name": "Camino Del Norte", "baseline_vc": 0.82, "plan_vc": 0.97,
                    "bin_s": 27000, "x": 1200.0, "z": -300.0}],
  "peak_overlap": {"baseline": 0.42, "plan": 0.31, "note": "share of school drop-off arrivals in the commute peak hour"},
  "unverified_inputs": ["..."],
  "llm_estimated_tools": [],
  "calibration": {"status": "uncalibrated", "median_error_pct": null}
}
```
Metric ids (fixed): `avg_commute_min`, `avg_dropoff_delay_min`, `max_spillback_m`,
`total_vht`, `late_kids`, `cost_upfront_usd`, `cost_per_year_usd`,
`resident_approval_pct` (added by API from residents), `winners`, `losers`.
A `MetricBlock` (baseline summary) is `{"metrics": [{id,label,unit,better,value:{median,p10,p90}}], "per_school": [...], "mode_share": [...]}`.

## Additions and clarifications (implemented)

- `resident_approval_pct` metric: `baseline` and `delta` are `{median: null, p10: null, p90: null}` (the status quo has no approval number) and the metric carries a `note`. Clients must render null blocks as "n/a".
- Plans also return `job_id`, `is_mine`, `my_vote`. `GET /plans` returns `{plans, total}`, accepts `limit` and `offset`, and returns 400 for an unknown `sort`.
- `PUT /plans/{id}`: author only, body like POST, resets status to draft.
- `GET /plans/{id}/residents`: 404 until the run is done. Response adds `text_status` (`complete|pending|partial|unavailable`) and `llm_available`. Reaction `deltas` add `kid_trip_min`, `home_vc_delta`, `mode_baseline`, `mode_plan`.
- `GET /residents/{id}`, `GET /residents/{id}/chat`, `GET /health`.
- `POST /residents/{id}/chat` body `{message, plan_id?}` -> `{persona_id, reply|null, messages, llm_available, error}`.
- `POST /plans/{id}/townhall` body `{persona_id?, message?, regenerate?}` -> `{plan_id, speakers: [{persona_id, first_name, side, comment, ...}], followup: {persona_id, message, text}|null, llm_available}`.
- `POST /tools/custom/preview` body `{description}` -> `{ok, label: "LLM estimated", description, estimate, tool}`; 503 when the LLM is down, 422 when its output fails validation.
- 503 while the world/sim is not loaded or the baseline is warming up.
