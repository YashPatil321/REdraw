"""World, baseline and tool endpoints (docs/api.md: World, Baseline, Tools)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response

from api.db import clean_json
from api.deps import get_state
from api.metrics import metric_defs
from api.state import AppState, ServiceUnavailable, as_dict
from pipeline.config import assumption, load_yaml, region

router = APIRouter()

OCTET = "application/octet-stream"


def clock_s(hhmm: str) -> int:
    h, m = str(hhmm).split(":")[:2]
    return int(h) * 3600 + int(m) * 60


def time_block() -> dict[str, int]:
    start = clock_s(assumption("time.sim_start"))
    end = clock_s(assumption("time.sim_end"))
    bin_s = int(assumption("time.bin_minutes")) * 60
    return {"bin_start_s": start, "bin_s": bin_s, "n_bins": (end - start) // bin_s,
            "report_start_s": clock_s(assumption("time.report_start")),
            "report_end_s": clock_s(assumption("time.report_end"))}


def _mission_view(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "title", "brief", "budget_usd_upfront", "budget_usd_per_year", "constraints",
            "goals_suggested", "tools")
    return {k: m.get(k) for k in keys if k in m} | {k: v for k, v in m.items() if k not in keys}


@router.get("/world/meta")
def world_meta(state: AppState = Depends(get_state)) -> dict[str, Any]:
    sim = state.sim()

    def build() -> dict[str, Any]:
        summary = clean_json(sim.world_summary() or {})
        rm = state.region_meta()
        reg = region()
        region_block = {
            "name": rm.get("name", reg.get("name")),
            "display_name": rm.get("display_name", reg.get("display_name")),
            "bbox": rm.get("bbox", reg.get("bbox")),
            "origin": {k: v for k, v in (rm.get("origin") or {}).items() if k in ("lat", "lon")} or None,
            "timezone": reg.get("timezone"),
            "extent_scene": rm.get("extent_scene"),
        }
        region_block.update(summary.get("region") or {})
        schools = clean_json(sim.schools())
        hero = summary.get("hero")
        if not hero and schools:
            s0 = schools[0]
            hero = {"school_id": s0["id"], "x": s0.get("x"), "z": s0.get("z")}
        mission_id = state.settings.default_mission
        return {
            "region": region_block,
            "synthetic": bool(summary.get("synthetic", rm.get("synthetic", False))),
            "assets": {"base_url": "/assets/", "manifest": "/assets/manifest.json"},
            "counts": summary.get("counts") or rm.get("counts") or {},
            "calibration": summary.get("calibration") or {"status": "uncalibrated", "median_error_pct": None},
            "mission": _mission_view(clean_json(sim.mission(mission_id))),
            "metrics": metric_defs(summary),
            "time": summary.get("time") or time_block(),
            "hero": hero,
            "unverified_inputs": clean_json(sim.unverified_inputs()),
        }

    return state.cached("world_meta", build)


@router.get("/world/network")
def world_network(state: AppState = Depends(get_state)) -> Response:
    sim = state.sim()
    body = state.cached("network_json",
                        lambda: json.dumps(clean_json(sim.network_json()), separators=(",", ":")).encode())
    return Response(content=body, media_type="application/json")


@router.get("/world/buildings/{building_id}")
def world_building(building_id: int, state: AppState = Depends(get_state)) -> dict[str, Any]:
    props = state.buildings_index().get(building_id)
    if props is None:
        raise HTTPException(status_code=404, detail=f"building {building_id} not found")
    out = dict(props)
    out["households"] = state.households_by_building().get(building_id, 0)
    x, z = out.get("centroid_x"), out.get("centroid_z")
    out["block"] = state.block_labeler().label(float(x), float(z)) if x is not None and z is not None else None
    return clean_json(out)


def _entrance_baseline(state: AppState) -> dict[str, dict[str, Any]]:
    """Per-entrance baseline queue stats, keyed '<school>/<entrance>' (best effort)."""
    bs = state.baseline_summary or {}
    out: dict[str, dict[str, Any]] = {}
    pe = bs.get("per_entrance")
    if isinstance(pe, dict):
        out.update({str(k): v for k, v in pe.items() if isinstance(v, dict)})
    elif isinstance(pe, list):
        for e in pe:
            if isinstance(e, dict) and e.get("key"):
                out[str(e["key"])] = {k: v for k, v in e.items() if k != "key"}
    return out


@router.get("/world/schools")
def world_schools(state: AppState = Depends(get_state)) -> list[dict[str, Any]]:
    schools = clean_json(state.sim().schools())
    eb = _entrance_baseline(state)
    for s in schools:
        for e in s.get("entrances", []):
            key = e.get("key") or f"{s['id']}/{e['id']}"
            e["key"] = key
            if e.get("baseline") is None:
                e["baseline"] = eb.get(key)
    return schools


@router.get("/baseline")
def baseline(state: AppState = Depends(get_state)) -> dict[str, Any]:
    summary = state.require_baseline()
    try:
        calib = clean_json(state.sim().world_summary() or {}).get("calibration")
    except ServiceUnavailable:
        calib = None
    return {"summary": summary, "playback_url": "/baseline/playback",
            "calibration": calib or {"status": "uncalibrated", "median_error_pct": None}}


@router.get("/baseline/playback")
def baseline_playback(state: AppState = Depends(get_state)) -> Response:
    state.require_baseline()
    if not state.baseline_playback:
        raise HTTPException(status_code=404, detail="baseline playback not available")
    return Response(content=state.baseline_playback, media_type=OCTET)


@router.get("/tools")
def tools(state: AppState = Depends(get_state)) -> dict[str, Any]:
    sim = state.sim()
    mission_id = state.settings.default_mission
    tool_list = [as_dict(t) for t in clean_json(sim.tools())]
    ids = {t.get("id") for t in tool_list}
    for t in load_yaml("tools.yaml").get("tools", []):  # docs/api.md: custom listed, disabled in MVP
        if t.get("id") == "custom" and "custom" not in ids:
            tool_list.append(t | {"enabled_in_mvp": t.get("enabled_in_mvp", False)})
    return {"mission": mission_id, "tools": tool_list}
