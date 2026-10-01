"""Plan endpoints (docs/api.md: Plans)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response

from api.db import clean_json
from api.deps import JobsDep, PlayerDep, StateDep, require_plan
from api.metrics import headline, metric_defs, metric_value
from api.schemas import PlanIn, VoteIn
from api.state import AppState, as_dict

router = APIRouter()

PUBLIC_PLAN_KEYS = ("id", "mission", "title", "pitch", "author_id", "tools", "created_at", "report",
                    "status", "check", "votes", "job_id")


def _check(state: AppState, plan: dict[str, Any]) -> dict[str, Any]:
    return clean_json(as_dict(state.sim().check_plan(plan)))


def _view(plan: dict[str, Any], me: str | None = None, my_vote: int | None = None) -> dict[str, Any]:
    out = {k: plan.get(k) for k in PUBLIC_PLAN_KEYS}
    if me is not None:
        out["is_mine"] = plan.get("author_id") == me
    if my_vote is not None:
        out["my_vote"] = my_vote
    return out


@router.post("/plans", status_code=201)
def create_plan(body: PlanIn, state: StateDep, me: PlayerDep) -> dict[str, Any]:
    plan = body.plan_dict(author_id=me)
    check = _check(state, plan)
    state.store.ensure_player(me)
    saved = state.store.create_plan(mission=plan["mission"], title=plan["title"], pitch=plan["pitch"],
                                    tools=plan["tools"], author_id=me, check=check)
    return _view(saved, me, 0)


@router.post("/plans/check")
def check_plan(body: PlanIn, state: StateDep) -> dict[str, Any]:
    return _check(state, body.plan_dict(id=None, author_id=None))


@router.get("/plans")
def list_plans(state: StateDep, mission: str | None = None, sort: str = "new",
               limit: int = Query(default=50, ge=1, le=200),
               offset: int = Query(default=0, ge=0)) -> dict[str, Any]:
    plans = state.store.list_plans(mission or None)
    if sort == "votes":
        plans.sort(key=lambda p: -int(p["votes"] or 0))  # stable: newest first among ties
    elif sort != "new":
        better = _metric_direction(state, sort)
        if better is None:
            raise HTTPException(status_code=400, detail=f"unknown sort '{sort}' (use new, votes or a metric id)")
        with_v = [(metric_value(p.get("report"), sort), p) for p in plans]
        have = [(v, p) for v, p in with_v if v is not None]
        missing = [p for v, p in with_v if v is None]
        have.sort(key=lambda vp: vp[0] if better == "lower" else -vp[0])
        plans = [p for _, p in have] + missing
    page = plans[offset:offset + limit]
    return {"plans": [{
        "id": p["id"], "title": p["title"], "pitch": p["pitch"], "created_at": p["created_at"],
        "votes": p["votes"], "status": p["status"], "mission": p["mission"],
        "headline": headline(p.get("report")),
    } for p in page], "total": len(plans)}


def _metric_direction(state: AppState, metric_id: str) -> str | None:
    try:
        summary = state.sim().world_summary()
    except Exception:  # noqa: BLE001
        summary = None
    for m in metric_defs(summary):
        if m["id"] == metric_id:
            return str(m.get("better", "lower"))
    return None


@router.get("/plans/{plan_id}")
def get_plan(plan_id: str, state: StateDep, me: PlayerDep) -> dict[str, Any]:
    plan = require_plan(state, plan_id)
    return _view(plan, me, state.store.my_vote(plan_id, me))


@router.put("/plans/{plan_id}")
def update_plan(plan_id: str, body: PlanIn, state: StateDep,
                me: PlayerDep) -> dict[str, Any]:
    plan = require_plan(state, plan_id)
    if plan["author_id"] != me:
        raise HTTPException(status_code=403, detail="only the author can edit a plan; save a copy instead")
    if plan["status"] in ("queued", "running"):
        raise HTTPException(status_code=409, detail="plan is running; wait for it to finish")
    new = body.plan_dict(id=plan_id, author_id=me)
    check = _check(state, new)
    state.store.update_plan_content(plan_id, title=new["title"], pitch=new["pitch"], mission=new["mission"],
                                    tools=new["tools"], check=check)
    p = state.store.get_plan(plan_id)
    assert p is not None
    return _view(p, me)


@router.post("/plans/{plan_id}/run")
def run_plan(plan_id: str, state: StateDep,
             jobs: JobsDep) -> dict[str, Any]:
    plan = require_plan(state, plan_id)
    check = plan.get("check") or {}
    if not check.get("ok", False):
        errs = "; ".join(check.get("errors") or []) or "plan check failed"
        raise HTTPException(status_code=400, detail=f"plan cannot run: {errs}")
    job = jobs.submit(plan)
    return {"job_id": job["id"]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, state: StateDep) -> dict[str, Any]:
    job = state.store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    return {k: job[k] for k in ("id", "plan_id", "status", "progress", "message", "error")}


@router.get("/plans/{plan_id}/playback")
def plan_playback(plan_id: str, state: StateDep,
                  jobs: JobsDep) -> Response:
    plan = require_plan(state, plan_id)
    path = jobs.playback_path(plan_id)
    if plan["status"] != "done" or not path.exists():
        raise HTTPException(status_code=404, detail="playback not available until the plan run is done")
    return Response(content=path.read_bytes(), media_type="application/octet-stream")


@router.post("/plans/{plan_id}/vote")
def vote(plan_id: str, body: VoteIn, state: StateDep,
         me: PlayerDep) -> dict[str, Any]:
    require_plan(state, plan_id)
    state.store.ensure_player(me)
    return {"votes": state.store.vote(plan_id, me, body.value)}
