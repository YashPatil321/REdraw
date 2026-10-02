# Redraw

Redraw is an open source, browser based 3D model of 4S Ranch and Del Sur
(San Diego, CA). Players try to fix real neighborhood problems their own way, a
traffic simulation tests each plan, and AI residents react to it. The first
mission, **The Morning Crunch**, is about school drop off and commute traffic
on weekday mornings.

The full design is in [`REDRAW_SPEC.md`](REDRAW_SPEC.md). How the packages fit
together is in [`docs/`](docs/).

> **Status: MVP (milestones M0 to M5, plus resident reactions).** All bell
> times, curb layouts, and modeling assumptions are placeholders until someone
> verifies them. See [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md).

![Explore view](client/docs-screenshots/explore.png)

## How it works

```
data/config/*.yaml  ──►  pipeline/  ──►  data/processed/ + client/public/assets/
                                              │
                                   sim/ (demand, mode choice, assignment, queues, report)
                                              │
                         residents/ (personas, approval, reactions)
                                              │
                                   api/ (FastAPI + Postgres)  ──►  client/ (three.js)
```

- **pipeline/** downloads open data (OpenStreetMap, USGS 3DEP, USDA NAIP, Census
  ACS and LODES) and builds the road network, buildings, terrain, glTF tiles and
  a synthetic population.
- **sim/** runs the 06:00 to 10:00 morning: trip chains with school drop offs,
  mode choice, time dependent traffic assignment (BPR + MSA), and drop off
  queues with spillback. A report card runs 20 seeds of baseline and plan and
  reports medians with 10th to 90th percentile ranges.
- **residents/** samples 500 personas, scores their approval with a fixed
  formula from their personal travel time changes, and (when an LLM endpoint is
  configured) writes short in-character reactions.
- **api/** serves the world, tools, plans, runs, playback, and residents.
- **client/** renders the 3D world, plays back traffic, and hosts the plan
  builder and report card.

## Setup (Windows with WSL2, or any Linux)

Everything runs inside WSL2 (Ubuntu 22.04 or newer) or Linux. Do the steps in
the WSL terminal, not PowerShell, and keep the repo inside the Linux file
system (for example `~/redraw`, not `/mnt/c/...`) for speed.

### 1. Prerequisites

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv git build-essential
# Node 20 or newer (via nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc && nvm install 22
```

Install Docker Desktop on Windows and turn on "Use the WSL 2 based engine"
and WSL integration for your distro (or install Docker Engine on Linux).

### 2. Clone and configure

```bash
git clone https://github.com/YashPatil321/REdraw.git ~/redraw
cd ~/redraw
cp .env.example .env      # fill in CENSUS_API_KEY and the LLM_* keys if you have them
```

### 3. Database

```bash
docker compose up -d      # Postgres 16 + PostGIS on localhost:5432
```

No Docker? Set `DATABASE_URL=sqlite:///data/processed/redraw.db` in `.env`.

### 4. Python packages

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[pipeline,dev]"
```

### 5. Build the world

```bash
.venv/bin/python pipeline/build_all.py
```

This downloads open data into `data/raw/` (cached, so later runs skip the
download) and writes `data/processed/` and `client/public/assets/`. It prints
counts of buildings, road edges, households, persons, and students per school.
If a source is unreachable it stops and prints the URL it tried and how to
download the file by hand.

For offline development there is a synthetic stand-in world:

```bash
.venv/bin/python pipeline/build_all.py --synthetic
```

It is made up geography. The client shows a "SYNTHETIC DEV DATA" banner and
every report says so.

### 6. Simulate and calibrate

```bash
.venv/bin/python -m sim.run baseline     # one seed, prints timing and queues
.venv/bin/python -m sim.calibrate        # needs observed times in calibration_targets.yaml
```

### 7. Run the game

```bash
.venv/bin/uvicorn api.main:app --port 8000      # terminal 1
cd client && npm install && npm run dev          # terminal 2
```

Open http://localhost:5173. The client talks to the API through the Vite
proxy at `/api`.

### Tests

```bash
.venv/bin/pytest
.venv/bin/ruff check .
cd client && npm test && npm run typecheck
```

## Playing The Morning Crunch

1. **Explore**: fly around, click buildings and schools.
2. **Traffic**: scrub 06:30 to 09:30, play at 1x to 60x, watch roads turn red and
   drop off queues grow. Toggle ghost traffic for car trails.
3. **Plan builder**: pick tools on the left (bell times, shuttles, carpools,
   drop off redesigns, new entrances, signal retiming, turn lanes, bike and walk
   routes, teen driving policy). Map tools are placed by clicking on the map.
   The budget bar on the right updates as you go.
4. **Run**: the server runs 20 seeds of baseline and plan. The report card shows
   each metric as a median with a range, per school results, mode share,
   winners and losers, side effects, and resident approval.
5. **Share**: every saved plan has a link, `/#/plan/<id>`.

There is no answer key. Goals in the mission brief are hints.

## Contributing

- Read `REDRAW_SPEC.md` and `CLAUDE.md` first. The coding rules in spec section
  14 apply to every change.
- Any number that models the real world goes in `data/config/assumptions.yaml`
  with a `source`, a `range`, and `verified`.
- The easiest high value contribution is verifying facts: bell times, drop off
  layouts, typical travel times. Update the YAML, set `verified: true`, and tick
  the item in `OPEN_QUESTIONS.md`.
- Keep the client thin: scoring and validation live in Python.
- Run the tests and ruff before opening a pull request.

## License

Code: Apache 2.0 ([LICENSE](LICENSE)). Assets and simulation results: CC BY 4.0.
OpenStreetMap derived data: ODbL. Data sources are listed in
[ATTRIBUTION.md](ATTRIBUTION.md).

## Deploying the viewer (Vercel)

The hosted viewer is a static build: the 3D world, the baseline traffic playback and the example
plans with their report cards. Running new plans needs the Python API on a normal server.

```bash
.venv/bin/uvicorn api.main:app --port 8000      # with the plans you want to show already run
cd client && npm run build:viewer               # writes client/dist-viewer/ (~225 MB)
```

Copy `client/dist-viewer/` into `site/` on the deploy-only `vercel-viewer` branch (built output only,
never merged into the code branches) and push. The Vercel project `redraw` builds that branch with
`node write-runtime-config.mjs`, which writes `site/runtime-config.json` from the project's
`GOOGLE_MAPS_API_KEY` environment variable.

**Google Photorealistic 3D Tiles:** create a Google Maps Platform key with the Map Tiles API
enabled and restrict it by HTTP referrer to your domains. Then either set `GOOGLE_MAPS_API_KEY` in
the Vercel project and redeploy, set `VITE_GOOGLE_MAPS_API_KEY` in `client/.env.local` for local dev,
or open the app once with `?gkey=YOUR_KEY` (kept in that browser only). With a key, Google's
photogrammetry is the world surface and the simulation draws on top; without one, the app shows
the open-data world (USGS lidar terrain and roofs, Overture/OSM buildings and roads).
