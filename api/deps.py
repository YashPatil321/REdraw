"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import HTTPException, Request

from api.jobs import JobRunner
from api.state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.redraw


def get_jobs(request: Request) -> JobRunner:
    return request.app.state.jobs


def player_id(request: Request) -> str:
    return request.state.player_id


def require_plan(state: AppState, plan_id: str) -> dict:
    plan = state.store.get_plan(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"plan '{plan_id}' not found")
    return plan
