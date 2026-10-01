# Redraw: Build Spec for Claude Code

Version 1.0 | 2026-09-30 | Owner: Yash

Redraw is an open source, browser based 3D copy of 4S Ranch and Del Sur (San Diego, CA) where players fix real neighborhood problems their own way. A simulation tests each plan honestly, AI residents react, and the best plans get packaged for real decision makers. This spec covers the full 3D world, the simulation, AI residents, the plan system, and the backend. VR is out of scope.

Read this whole file before writing code. Build milestone by milestone (Section 12). Do not skip acceptance criteria.

---

## 1. Goals and non goals

### Goals (v1)
1. A walkable, clickable 3D model of 4S Ranch and Del Sur built only from open data, running in a normal browser at 30+ FPS on a mid range laptop.
2. A morning traffic simulation (6:30 to 9:30 AM) that reproduces real rush hour patterns, including school drop off queues at Del Norte High, Oak Valley Middle, and Design39 Campus.
3. Mission 1, "The Morning Crunch": players build a plan from open ended tools, run it, and get an honest report card with uncertainty ranges.
4. 500 AI residents who live in the sim, react to plans in their own words, and speak at a virtual town hall.
5. Plans are saved, shareable by link, and votable.

### Non goals (v1)
1. VR or headset support. Keep the renderer standard WebGL.
2. Fire, housing, and transit missions. Design the code so they plug in later (Section 13).
3. Real resident accounts and home claiming. Design the schema for it, do not build the UI.
4. Photoreal imagery like Google 3D Tiles. Use open data only.
5. Mobile layout polish. Desktop browser first.

---

## 2. Stack decision

The MVP client is a web app. After the MVP is validated, the main client moves to Unreal Engine 5 (Section 13A), while the web client stays as a lightweight viewer for shared plan links. Everything except the client is engine independent from day one. Reasons for starting on web:
1. Claude Code can read and write every file in a TypeScript project. Unreal stores most content in binary .uasset files and Blueprints that Claude Code cannot edit reliably, which would slow the build down a lot.
2. Players open a link instead of installing a large game, which matters for sharing plans and getting officials to look at them.
3. Unreal is source available, not open source. The project must be fully open source.
4. The map is about 10 km by 10 km. A custom three.js scene handles that easily and gives full control over traffic animation.

| Layer | Choice | License |
| --- | --- | --- |
| Client | TypeScript, Vite, three.js | MIT |
| UI panels | Plain TypeScript with Lit web components (or Preact if preferred) | BSD / MIT |
| Simulation and API | Python 3.11, FastAPI, NetworkX, OSMnx, NumPy | BSD / MIT |
| Geo processing | GeoPandas, Shapely, PyProj, Rasterio, trimesh | BSD / MIT |
| Asset pipeline | Python scripts, Blender with Blender MCP for hero assets | GPL (Blender) |
| Database | Postgres with PostGIS (run via Docker Compose; self hosted Supabase optional later) | PostgreSQL / GPL |
| AI residents | Any OpenAI compatible endpoint. Default: vLLM on the team GPU server serving an open weight model | Apache 2.0 (vLLM) |
| Tests | Vitest (client), pytest (Python) | MIT |
| Project license | Apache 2.0 for code, CC BY 4.0 for assets and results, ODbL for OSM derived data | |

Dev environment: Windows with WSL2 (Ubuntu) is the primary dev setup. All scripts must run in WSL2 and Linux. Do not assume macOS.

---

## 3. Repo layout

```
redraw/
  CLAUDE.md                 short pointer to this spec plus coding rules
  REDRAW_SPEC.md            this file
  LICENSE                   Apache 2.0
  ATTRIBUTION.md            every data source and its license
  docker_compose.yml        postgres + postgis
  .env.example              all config keys, no secrets
  data/
    raw/                    downloaded source data (gitignored)
    processed/              cleaned GeoJSON, graphs, populations (gitignored)
    config/
      region.yaml           bounding box, projection, region name
      schools.yaml          schools, entrances, bell times, curb capacity
      assumptions.yaml      every modeling assumption with source and range
      tools.yaml            tool definitions for missions
      missions/
        morning_crunch.yaml
  pipeline/                 Python: fetch and build all world data
    fetch_osm.py
    fetch_dem.py
    fetch_imagery.py
    fetch_census.py
    build_terrain.py
    build_buildings.py
    build_roads.py
    build_population.py
    build_all.py            runs everything in order
  sim/                      Python package: simulation engine
    world.py                loads processed data into one WorldState
    demand.py               trip generation
    modechoice.py
    assignment.py           traffic assignment and travel times
    schools.py              drop off queue model
    plan.py                 applies a plan's tools to WorldState
    report.py               report card and Monte Carlo ranges
    calibrate.py
    tests/
  residents/                Python package: AI residents
    persona.py
    reactions.py
    townhall.py
    llm.py                  OpenAI compatible client wrapper
    tests/
  api/                      FastAPI app
    main.py
    routes_world.py
    routes_plans.py
    routes_residents.py
    db.py
    migrations/
  client/                   Vite + three.js app
    index.html
    src/
      main.ts
      scene/                terrain, buildings, roads, sky, lighting
      traffic/              instanced cars, playback, congestion coloring
      ui/                   panels, tool palette, report card, town hall
      api.ts
      state.ts
    public/assets/          built glTF and textures (gitignored, built by pipeline)
```

---

## 4. Region and coordinates

`data/config/region.yaml`:

```yaml
name: 4s_ranch_del_sur
# Starting bbox. Verify against OSM place boundaries for "4S Ranch" and
# "Del Sur, San Diego" and expand so I 15 (east), SR 56 (south),
# Black Mountain Open Space (west), and the San Dieguito River (north) are included.
bbox:
  south: 32.965
  north: 33.045
  west: -117.175
  east: -117.075
projection: EPSG:32611        # UTM 11N, meters, for all processing
scene_origin: center_of_bbox  # client uses local meters relative to this point
```

Rules:
1. All pipeline processing happens in EPSG:32611 meters.
2. The client scene uses local coordinates: x east, z south (three.js convention), y up, in meters, relative to `scene_origin`.
3. One shared function converts lat/lon to scene coordinates. Put it in both Python (`pipeline/geo.py`) and TypeScript (`client/src/geo.ts`) and unit test that they match.

---

## 5. Data pipeline

Every fetch script caches to `data/raw/` and skips download if the file exists. `build_all.py` must run end to end on a fresh machine with only the `.env` filled in.

| Data | Source | Script | Output |
| --- | --- | --- | --- |
| Roads | OpenStreetMap via OSMnx (drive, walk, bike networks) | fetch_osm.py, build_roads.py | `roads_drive.graphml`, `roads.geojson` |
| Buildings | OSM building footprints via OSMnx. Optional upgrade: SanGIS building outlines if downloadable | fetch_osm.py, build_buildings.py | `buildings.geojson`, `buildings.glb` tiles |
| Terrain | USGS 3DEP DEM (use the py3dep library or The National Map API), 10 m or better | fetch_dem.py, build_terrain.py | `terrain.glb`, `heightmap.png` |
| Aerial imagery | USDA NAIP (public domain), via Microsoft Planetary Computer STAC | fetch_imagery.py | `terrain_albedo.jpg` |
| Population and commutes | Census ACS 5 year (block group) and LEHD LODES origin destination for California | fetch_census.py, build_population.py | `households.parquet`, `persons.parquet` |
| Schools | OSM amenity=school plus `schools.yaml` | build_population.py | school assignments |

### 5.1 Terrain
1. Clip DEM to bbox plus 500 m buffer, reproject to EPSG:32611.
2. Build a grid mesh at about 10 m spacing (target under 1.2M triangles total). Split into 4 by 4 tiles for culling.
3. Drape NAIP imagery as the albedo texture. Max texture size 4096 per tile.
4. Export as glTF binary with Draco compression.

### 5.2 Buildings
1. Use OSM footprints. Height priority: `height` tag, then `building:levels` times 3.2 m, then defaults from `assumptions.yaml` (house 7 m, commercial 8 m, school 9 m).
2. Base elevation = DEM sampled at footprint centroid.
3. Extrude with trimesh. Simple flat or hipped roofs for houses (hip if footprint is roughly rectangular).
4. Merge buildings into tiles (same 4 by 4 grid), one mesh per tile, with a per vertex `building_id` attribute so the client can pick buildings by raycast.
5. Write `buildings.geojson` with id, type, height, address if present, and parcel fields (empty for now).
6. Hero assets (Del Norte High, Design39, 4S Commons) may later be replaced with Blender MCP built models. Leave a `hero_overrides/` folder the pipeline checks first.

### 5.3 Roads
1. Drive graph from OSMnx, simplified, with lanes, maxspeed, and oneway. Fill missing values from `assumptions.yaml` by road class.
2. Compute per edge capacity (vehicles per hour per lane by class) and free flow time.
3. Road ribbons for rendering: buffer each edge by lane count times 3.6 m, drape on terrain, export as one glTF with lane markings as a texture.
4. Mark key arterials in a config list for labels: Camino Del Norte, Camino Del Sur, Dove Canyon Road, 4S Ranch Parkway, Bernardo Center Drive, Carmel Valley Road, Black Mountain Road, I 15, SR 56. Verify names against OSM.

### 5.4 Synthetic population
1. For each census block group inside the bbox, sample households to match ACS totals: household size, kids by age band, vehicles, income band.
2. Place each household at a residential building in that block group, weighted by building footprint area.
3. Assign workers to work locations using LODES origin destination flows. Workers whose job is outside the bbox get an exit node on I 15 north, I 15 south, SR 56 west, or Camino Del Norte east based on job direction.
4. Assign each school age kid to a school by attendance boundary if available, else nearest school of the right grade level in `schools.yaml`.
5. Target: roughly the real household count of the area (pipeline should print it; expect on the order of 10k to 20k households). Use a fixed random seed.

### 5.5 schools.yaml format

```yaml
schools:
  - id: del_norte_hs
    name: Del Norte High School
    grades: [9, 12]
    bell_start: "08:30"        # placeholder, verify with real schedule
    entrances:
      - id: main_dropoff
        lat: 0.0               # fill in from OSM or manual lookup
        lon: 0.0
        curb_spots: 10         # cars that can unload at once
        unload_seconds: 45     # average time per car at curb
  - id: oak_valley_ms
    name: Oak Valley Middle School
    grades: [6, 8]
    bell_start: "08:00"
  - id: design39
    name: Design39 Campus
    grades: [0, 8]
    bell_start: "08:15"
# Add every elementary school in the bbox found in OSM.
```

All bell times and curb numbers are placeholders. Mark them `verified: false` until confirmed, and show unverified inputs in the report card footer.

---

## 6. Simulation engine

Goal: fast enough that one plan run (one seed) finishes in under 20 seconds on a laptop, and a full report card (20 seeds) in under 3 minutes using multiprocessing.

### 6.1 Time
Simulate 06:00 to 10:00 in 5 minute bins. Report on 06:30 to 09:30.

### 6.2 Demand (`demand.py`)
Each person gets a list of trips for the morning:
1. Workers: home to work (or exit node). Departure time sampled from a distribution in `assumptions.yaml` (peak around 07:30).
2. Students: home to school. Departure so they arrive 5 to 20 minutes before bell.
3. Parent drop off: if a kid is driven, the trip chain is home to school to work (or back home). Chaining is required; it is the core of the drop off problem.
4. High school students may drive themselves (share from assumptions).

### 6.3 Mode choice (`modechoice.py`)
Modes: drive alone, drive with drop off, carpool, school shuttle, bike, walk, school bus.
1. Multinomial logit on travel time, cost, distance, and habit. Coefficients in `assumptions.yaml` with source notes.
2. Walk and bike only allowed under distance limits by age (assumptions).
3. Tools change mode availability or utilities (Section 7).

### 6.4 Traffic assignment (`assignment.py`)
1. Time dependent incremental assignment on the drive graph using the BPR volume delay function: `t = t0 * (1 + alpha * (v / c) ** beta)`, alpha and beta from assumptions (default 0.15 and 4).
2. Iterate with method of successive averages, 5 to 8 iterations, until total travel time changes less than 1 percent.
3. Intersection delay: add a fixed delay per signalized node plus a v/c based term. Keep simple and documented.
4. Output: per edge per bin volume, speed, and v/c; per agent trip list of (edge id, enter time, exit time) for animation.
5. Leave a clean interface (`AssignmentBackend`) so SUMO can replace this later.

### 6.5 School drop off queues (`schools.py`)
1. Each entrance is a queue: arrivals from assignment, service rate = curb_spots / unload_seconds.
2. Queue length in cars times 7 m gives spillback length. If spillback exceeds the approach edge length, reduce that edge's capacity and add delay to upstream edges. Run this inside the assignment loop.
3. Output per entrance per bin: queue length, max spillback, average wait.

### 6.6 Report card (`report.py`)
Run the plan and the baseline with 20 seeds each, varying demand by plus or minus 10 percent, departure times, and mode choice draws. For each metric report median and 10th to 90th percentile range.

Metrics:
| Metric | Unit |
| --- | --- |
| Average commute time, all workers | minutes |
| Average drop off delay per school | minutes |
| Max queue spillback per school | meters |
| Total vehicle hours traveled | hours |
| Kids arriving late (after bell) | count |
| Plan cost, upfront and per year | USD |
| Mode share change | percent points |
| Resident approval | percent, from residents module |
| Winners and losers | count of residents better or worse by 3+ minutes |

Also list "side effects": any edge where v/c rose above 0.9 because of the plan.

### 6.7 Calibration (`calibrate.py`)
1. Compare baseline simulated travel times on 6 to 10 key routes against real typical travel times entered manually in `data/config/calibration_targets.yaml` (from Google Maps typical traffic or field timing).
2. Tune demand scale, departure spread, and capacity factors to minimize error.
3. Print a calibration table. Target: median route error under 20 percent. Show calibration status in the client.

---

## 7. Plans and tools

A plan is a list of tool instances. Tools are data, defined in `tools.yaml`, and implemented as functions that modify `WorldState` before the sim runs. There is no solution key. The engine never compares plans to a correct answer.

### 7.1 MVP tools for Morning Crunch

| Tool id | What the player sets | How it changes the sim | Cost source |
| --- | --- | --- | --- |
| bell_time | school, new start time | shifts student and drop off departure windows | 0 upfront, note bus contract effects |
| school_shuttle | stops on map, school, buses, frequency | adds shuttle mode for kids near stops; buses are vehicles on the network | per bus per year in assumptions |
| carpool_program | school or neighborhood, incentive level | raises carpool utility; adoption range from assumptions | per year program cost |
| dropoff_redesign | entrance, extra curb spots or faster unload | changes queue service rate | upfront construction |
| new_dropoff_entrance | point on map linked to a road edge | adds a new queue and graph connector | upfront construction |
| signal_timing | intersection, priority direction | changes node delay by approach | small upfront |
| turn_lane | edge | raises capacity on that approach | upfront construction |
| bike_route | polyline on map | adds or upgrades bike edges; raises bike utility for short trips | per km |
| safe_walk_route | polyline, crossing guards yes or no | raises walk utility for kids under distance limit | per km plus guards per year |
| teen_drive_policy | school, parking permits cap | limits student driver share | 0 |
| custom | free text description | converted by LLM, see 7.3 | from LLM estimate, flagged |

### 7.2 Plan JSON

```json
{
  "id": "uuid",
  "mission": "morning_crunch",
  "title": "Stagger and shuttle",
  "pitch": "Move Del Norte to 9:00 and run two shuttles from Del Sur.",
  "author_id": "uuid or null",
  "tools": [
    {"tool": "bell_time", "params": {"school": "del_norte_hs", "start": "09:00"}},
    {"tool": "school_shuttle", "params": {"school": "del_norte_hs", "stops": [[33.0,-117.13]], "buses": 2, "headway_min": 15}}
  ],
  "created_at": "iso time",
  "report": null
}
```

Validate with Pydantic. Missions define a budget and constraints (example: "no new through lanes"); plans over budget can still run but are marked over budget.

### 7.3 Custom tools
1. Player writes a short description.
2. The LLM returns JSON only: which existing levers it maps to (mode utility shifts, capacity changes, cost), an adoption range, and a list of assumptions in plain words.
3. The result is shown to the player for confirmation before running, with assumptions visible.
4. Custom tools are always labeled "LLM estimated" in the report card.
5. If the LLM output fails validation, show an error and do not run.

---

## 8. AI residents

### 8.1 Personas (`persona.py`)
1. Sample 500 persons from the synthetic population, stratified so every school, income band, and age band is represented.
2. Each persona JSON: id, first name (generated, never real), age, household summary, home building id, job area, kids and schools, commute mode, values (3 from a fixed list such as safety, time, cost, environment, community, property, change averse), and a short backstory generated once by the LLM and cached.
3. Never use real names or link personas to real addresses in the UI. Home locations are shown at the block level.

### 8.2 Reactions (`reactions.py`)
1. After a plan run, compute each persona's personal deltas: commute minutes, kid drop off minutes, cost exposure, changes on their street.
2. Approval is computed deterministically from deltas weighted by values (formula in assumptions). The LLM does not decide approval.
3. The LLM writes a one to two sentence reaction in character, given the deltas and approval. It must not invent numbers; pass it the numbers to use.
4. Batch requests. Cache by (persona id, plan id).

### 8.3 Talk to a resident
Chat endpoint. System prompt includes persona, their deltas for the current plan, and a rule to stay in character and only cite numbers provided. Keep last 10 messages per player per persona.

### 8.4 Town hall (`townhall.py`)
1. Pick 8 speakers: the 4 most negatively affected and 4 most positively affected, with at least 3 different schools represented.
2. Each gives a 2 to 3 sentence public comment.
3. Player can write a response; the LLM generates a short follow up from that speaker.

### 8.5 LLM config
`.env` keys: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_FAST`, `LLM_MODEL_SMART`. Fast model for reactions, smart model for town hall and custom tools. If the endpoint is down, the game still works: reactions show approval scores without quotes.

---

## 9. API (FastAPI)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /world/meta | region info, asset URLs, calibration status |
| GET | /world/buildings/{id} | building details |
| GET | /world/schools | schools with entrances and baseline queue stats |
| GET | /baseline | cached baseline report and traffic playback |
| GET | /tools | tool definitions for the current mission |
| POST | /plans | create plan |
| GET | /plans/{id} | plan with report if done |
| POST | /plans/{id}/run | start run, returns job id |
| GET | /jobs/{id} | job status and progress percent |
| GET | /plans/{id}/playback | traffic playback data (see 9.1) |
| GET | /plans/{id}/residents | reactions list |
| POST | /plans/{id}/townhall | generate or continue town hall |
| POST | /residents/{id}/chat | talk to a resident |
| POST | /tools/custom/preview | convert custom tool text to levers |
| POST | /plans/{id}/vote | upvote or downvote |
| GET | /plans?mission=&sort= | list plans, sort by votes or metric |

Runs execute in a background worker (a simple process pool is fine for v1). Cache baseline at startup.

### 9.1 Playback format
To keep the client smooth, do not send every agent. Send:
1. Per edge per bin: v/c and speed (for road coloring). Binary Float32Array is fine.
2. A sample of up to 3,000 vehicle trajectories as arrays of (edge index, enter time, exit time). Client interpolates positions along edges.
3. Per school entrance per bin: queue length.

### 9.2 Database tables
`players` (id, display_name, created_at), `plans` (id, mission, title, pitch, author_id, tools jsonb, report jsonb, created_at), `votes` (plan_id, player_id, value), `resident_reactions` (plan_id, persona_id, approval, text), `chats` (player_id, persona_id, messages jsonb).
Reserve for later, create empty: `households_real` (claimed homes, opt in fields, all nullable) so home claiming can be added without schema rewrites.

Auth in v1: anonymous player id stored in a cookie. Real auth later.

---

## 10. Client

### 10.1 Scene
1. Load terrain and building tiles as glTF with Draco. Frustum culling per tile.
2. Sky with sun position from real time of day for the region (use a small solar position function). Morning haze via fog.
3. Roads drawn above terrain with polygon offset to avoid z fighting.
4. Cars: one InstancedMesh with simple low poly car geometry, up to 3,000 instances, colored by speed. School shuttles are a separate instanced mesh.
5. Picking: raycast against building tiles, read `building_id`, open the info panel.
6. Camera: orbit controls plus a "fly to" helper with smooth easing. Opening shot: start high above the region and fly down to Del Norte in about 4 seconds.

### 10.2 Views
1. **Explore**: free camera, click buildings and schools.
2. **Traffic**: time scrubber from 06:30 to 09:30 with play, pause, and 1x to 60x speed. Roads colored green to red by v/c. Queue bars rise at school entrances.
3. **Ghost traffic**: toggle that draws car trails as glowing lines (additive blending) for a dramatic rush hour view.
4. **Plan builder**: tool palette on the left. Tools that need map input (stops, routes, points) use click to place on terrain with snapping to roads. Plan summary and budget bar on the right.
5. **Report card**: baseline vs plan side by side. Each metric shows median and range as a bar with whiskers. Side effects listed with "show on map" buttons.
6. **Before and after split**: vertical slider; left half renders baseline playback, right half renders plan playback (two scissor passes, same camera).
7. **Residents**: feed of reactions with approval color. Click a resident to fly to their block and open chat.
8. **Town hall**: a simple meeting room panel (2D is fine) with speaker cards and the player response box.
9. **Browse plans**: list sorted by votes or a chosen metric; open any plan by link `/#/plan/{id}`.

### 10.3 Performance budget
Target 30+ FPS at 1080p on integrated graphics: under 1.5M triangles visible, under 300 draw calls, textures under 300 MB GPU memory. Add a stats overlay toggle for dev.

### 10.4 Style
Minimal UI with dark translucent panels so the map stays the focus. One accent color for the player's plan, a different one for baseline. All text readable at 1366 by 768.

---

## 11. Mission 1: The Morning Crunch

`data/config/missions/morning_crunch.yaml`:

```yaml
id: morning_crunch
title: The Morning Crunch
brief: >
  School drop off and commute traffic stack up every weekday morning in
  4S Ranch and Del Sur. Cut the delay without adding through lanes.
budget_usd_upfront: 2000000
budget_usd_per_year: 400000
constraints:
  - no_new_through_lanes
goals_suggested:            # shown as hints, not pass or fail
  - reduce average drop off delay at Del Norte by 30 percent
  - no school gets worse
  - resident approval above 55 percent
tools: [bell_time, school_shuttle, carpool_program, dropoff_redesign,
        new_dropoff_entrance, signal_timing, turn_lane, bike_route,
        safe_walk_route, teen_drive_policy, custom]
```

Goals are suggestions. Players choose what they optimize. Leaderboards can be sorted by any metric.

---

## 12. Milestones and acceptance criteria

Build in order. Commit at the end of each milestone. Do not start the next milestone until all criteria pass.

### M0: Repo and tooling
- [ ] Repo layout from Section 3 exists with LICENSE, ATTRIBUTION.md, .env.example, CLAUDE.md
- [ ] `docker compose up` starts Postgres with PostGIS
- [ ] `pytest` and `npm test` run (empty tests pass)
- [ ] geo conversion functions exist in Python and TypeScript with matching unit tests

### M1: World data
- [ ] `python pipeline/build_all.py` runs end to end from empty `data/` folders
- [ ] Outputs terrain, building, and road glTF tiles plus GeoJSON and graphml
- [ ] Prints counts: buildings, road edges, households, persons, students per school
- [ ] Every source and license listed in ATTRIBUTION.md

### M2: 3D world in the browser
- [ ] `npm run dev` shows terrain with imagery, buildings, and roads for the whole region
- [ ] Opening flyover lands on Del Norte High
- [ ] Clicking a building opens its info panel
- [ ] 30+ FPS on a mid range laptop with the stats overlay

### M3: Baseline simulation
- [ ] `python -m sim.run baseline` finishes one seed in under 20 seconds
- [ ] Drop off queues produced for every school entrance
- [ ] Calibration table prints; record the median error even if above target
- [ ] Unit tests for BPR function, queue model, and trip chaining

### M4: Traffic playback
- [ ] Client plays baseline from 06:30 to 09:30 with moving cars and colored roads
- [ ] Queue bars at schools grow and shrink over time
- [ ] Ghost traffic toggle works

### M5: Plans and tools
- [ ] All MVP tools except custom can be placed and configured in the plan builder
- [ ] Plans save to the database and load by link
- [ ] Running a plan shows progress and then the report card with ranges from 20 seeds
- [ ] Before and after split view works
- [ ] Test: a plan that moves Del Norte bell time later reduces overlap with commute peak in the sim output (directional check, not a fixed number)

### M6: AI residents
- [ ] 500 personas generated and cached
- [ ] Reactions appear after a plan run, with deterministic approval and LLM text
- [ ] Chat with any resident works and stays in character
- [ ] Town hall generates 8 speakers and handles a player response
- [ ] Game still runs with the LLM endpoint turned off

### M7: Custom tools, voting, polish
- [ ] Custom tool preview shows mapped levers and assumptions before running
- [ ] Voting and plan browsing with sort by votes or metric
- [ ] Unverified input warning in the report card footer
- [ ] README with setup steps for WSL2, screenshots, and how to contribute

---

## 13. Extensibility (design for these now, do not build)

1. **Missions as data.** Fire evacuation, housing, and transit missions should be new YAML files plus new tool implementations, with no changes to the core engine loop.
2. **Region packs.** Everything region specific lives in `data/config/` and pipeline outputs. Another city should work by changing `region.yaml` and rerunning the pipeline.
3. **Sim backends.** `AssignmentBackend` interface so SUMO or MATSim can replace the built in assignment.
4. **Real households.** `households_real` table and an optional override path in `demand.py` that uses opt in trip data when present.
5. **Fire model.** Leave a `sim/hazards/` folder and an event hook in the time loop for road closures.

---

## 13A. MVP scope and Unreal migration

### MVP cut
The MVP is M0 through M5, plus a light version of M6: resident reactions and approval only, with no chat or town hall. Custom tools and voting (M7) wait until after the move to Unreal. The MVP is done when 30 local playtesters can run plans and the baseline calibration is recorded.

### Portability rules (follow these during the MVP)
1. **Thin client.** The client only renders and sends input. All game logic, scoring, and plan validation live in the Python API. No rule that affects results may exist only in TypeScript.
2. **Engine neutral assets.** All meshes are glTF 2.0. Also export terrain as a 16 bit grayscale PNG heightmap with its scale and offset in `terrain_meta.json`, since Unreal Landscape imports heightmaps directly.
3. **One coordinate contract.** Scene space is local meters: x east, z south, y up (three.js). Unreal mapping: `UE.X = x * 100`, `UE.Y = z * 100`, `UE.Z = y * 100` (centimeters; X east, Y south, Z up is left handed, which matches Unreal). Put this mapping in `docs/coordinates.md` with test points.
4. **Versioned playback format.** Document the playback binary layout in `docs/playback_format.md` with a version number. Both clients must parse the same files.
5. **API is the contract.** Every client feature uses only the endpoints in Section 9. If a client needs new data, add an endpoint; do not read files directly.
6. **Data driven UI text.** Tool names, descriptions, mission briefs, and metric labels come from the API, not hardcoded in the client.

### Migration steps (after MVP)
1. Install Epic's Unreal Engine skills plugin for Claude Code from the official plugin marketplace and confirm the unreal MCP server connects.
2. Create a UE5 C++ project `RedrawUE` in its own folder in the repo. Prefer C++ over Blueprints so Claude Code can edit logic as text.
3. Import the terrain heightmap into a Landscape actor using `terrain_meta.json` for scale, and drape the NAIP texture as a landscape material.
4. Import building and road glTF tiles with Unreal's glTF importer. Keep `building_id` on each building so picking still works.
5. Optional photoreal layer: add Cesium for Unreal with Google 3D Tiles for private demos and video only. The public build uses open data.
6. Write an `ARedrawApiClient` actor in C++ using Unreal's HTTP module to call the same API.
7. Render traffic with Hierarchical Instanced Static Meshes (or Mass Entity later) reading the same playback files.
8. Rebuild UI panels in UMG, driven by API data.
9. Use Sequencer for cinematic flyovers and trailers.
10. Keep the web client running as the viewer for shared plan links, since officials and players need to open plans without installing anything.

### Migration acceptance criteria
- [ ] Unreal client loads the full region and matches web client building positions within 1 m at test points
- [ ] Baseline playback in Unreal matches the web client frame for frame on vehicle positions
- [ ] A plan created in the web client opens and plays in Unreal, and the reverse
- [ ] No change was needed to `sim/`, `residents/`, or `api/` other than added endpoints

---

## 14. Coding rules for Claude Code

1. Python: type hints everywhere, Pydantic for all external data, ruff for linting, pytest for tests.
2. TypeScript: strict mode, no `any` without a comment explaining why.
3. Every number that models the real world goes in `assumptions.yaml` with a `source` and `range` field. No magic numbers in code.
4. Never commit data in `data/raw/` or `data/processed/`, built assets, or secrets.
5. Fixed random seeds for anything that affects results, so runs are reproducible.
6. If a data source is unavailable, write a clear error with the URL tried and a manual download instruction, then stop. Do not silently fake data.
7. When you have to guess a real world fact (bell time, curb length, road name), mark it `verified: false` and list it in `OPEN_QUESTIONS.md`.
8. Keep each milestone working end to end before polishing anything.

---

## 15. Open questions for the team

- [ ] Real bell times and drop off layouts for Del Norte, Oak Valley, Design39, and nearby elementary schools
- [ ] Attendance boundary data from Poway Unified, if published
- [ ] Which open weight model runs best on the GPU server for 500 residents
- [ ] Real typical travel times on 6 to 10 routes for calibration (can be timed by hand on school days)
- [ ] Whether SanGIS building outlines and parcels can be downloaded in bulk for the bbox
