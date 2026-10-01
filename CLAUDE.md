# Redraw: notes for Claude Code

Read `REDRAW_SPEC.md` first. Contracts between packages live in `docs/`:
- `docs/data_contract.md`: pipeline outputs (the only files sim/api read)
- `docs/sim_interface.md`: `sim.service.SimService`, the only way api/residents call the sim
- `docs/api.md`: HTTP endpoints, the only way the client gets data
- `docs/playback_format.md`: RDPB binary traffic playback (versioned)
- `docs/coordinates.md`: lat/lon <-> scene <-> Unreal

## Commands
- Python env: `python3 -m venv .venv && .venv/bin/pip install -e ".[pipeline,dev]"`
- World data: `.venv/bin/python pipeline/build_all.py` (real data) or `--synthetic` (offline dev fixture, labeled as fake everywhere)
- Sim: `.venv/bin/python -m sim.run baseline`, `.venv/bin/python -m sim.calibrate`
- API: `.venv/bin/uvicorn api.main:app --reload`
- Client: `cd client && npm install && npm run dev`
- Tests: `.venv/bin/pytest` and `cd client && npm test`; lint: `.venv/bin/ruff check .`

## Coding rules (spec 14)
1. Python: type hints, Pydantic for external data, ruff, pytest.
2. TypeScript: strict, no `any` without a comment.
3. Every real-world number lives in `data/config/assumptions.yaml` (value, unit, range, source, verified). Read with `pipeline.config.assumption("a.b.c")`.
4. Never commit `data/raw/`, `data/processed/`, built assets, or secrets.
5. Fixed seeds for anything affecting results.
6. Unavailable data source -> clear error with URL and manual download steps, then stop. Never silently fake data (the `--synthetic` mode is explicit and labeled).
7. Guessed real-world facts get `verified: false` and an entry in `OPEN_QUESTIONS.md`.
8. Thin client: all scoring, validation and game logic live in Python. UI text (tool names, metric labels, mission brief) comes from the API.
