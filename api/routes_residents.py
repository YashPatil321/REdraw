"""AI resident endpoints: reactions, town hall, chat, custom tool preview (spec 8, 7.3)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.db import clean_json, iso, utcnow
from api.deps import JobsDep, PlayerDep, StateDep, require_plan
from api.metrics import metric_value
from api.schemas import ChatIn, CustomPreviewIn, TownhallIn
from api.state import AppState, ServiceUnavailable
from residents.custom_tool import CustomToolError
from residents.persona import Persona

router = APIRouter()


def _personas_ready(state: AppState) -> dict[int, Persona]:
    try:
        return state.residents.persona_map()
    except FileNotFoundError as e:
        raise ServiceUnavailable(str(e)) from e


def _done_plan(state: AppState, plan_id: str) -> dict[str, Any]:
    plan = require_plan(state, plan_id)
    if plan["status"] != "done":
        raise HTTPException(status_code=404, detail="resident reactions are available after the plan run is done")
    return plan


@router.get("/plans/{plan_id}/residents")
def plan_residents(plan_id: str, state: StateDep,
                   jobs: JobsDep) -> dict[str, Any]:
    plan = _done_plan(state, plan_id)
    _personas_ready(state)
    rows = state.store.get_reactions(plan_id)
    views = [v for v in (state.residents.reaction_view(r) for r in rows) if v is not None]
    pct = metric_value(plan.get("report"), "resident_approval_pct")
    if pct is None and views:
        pct = round(100.0 * sum(v["approves"] for v in views) / len(views), 1)
    missing = any(v["text"] is None for v in views)
    if missing and state.llm.available:
        jobs.fill_texts_async(plan_id)
    if not missing:
        status = "complete"
    elif jobs.texts_pending(plan_id):
        status = "pending"
    else:
        status = "unavailable" if not state.llm.available else "partial"
    return {"approval_pct": pct, "reactions": views, "text_status": status,
            "llm_available": bool(state.llm.available)}


@router.post("/plans/{plan_id}/townhall")
def plan_townhall(plan_id: str, state: StateDep, body: TownhallIn | None = None) -> dict[str, Any]:
    body = body or TownhallIn()
    plan = _done_plan(state, plan_id)
    personas = _personas_ready(state)
    reactions = {r["persona_id"]: r for r in state.store.get_reactions(plan_id)}
    if not reactions:
        raise HTTPException(status_code=404, detail="no resident reactions for this plan")
    stored = state.store.get_townhall(plan_id)
    need_text = stored is not None and any(s.get("comment") is None for s in stored) and state.llm.available
    if stored is None or body.regenerate or need_text:
        speakers = state.residents.select_speakers(list(reactions.values()))
        comments = state.residents.comments(speakers, plan)
        old = {s["persona_id"]: s.get("comment") for s in (stored or [])} if not body.regenerate else {}
        stored = [{"persona_id": p.persona_id, "side": side,
                   "comment": comments.get(p.persona_id) or old.get(p.persona_id)}
                  for p, _r, side in speakers]
        state.store.save_townhall(plan_id, stored)

    views = []
    for s in stored:
        r = reactions.get(s["persona_id"])
        v = state.residents.reaction_view(r) if r else None
        if v:
            views.append(v | {"side": s["side"], "comment": s.get("comment")})

    followup = None
    if body.message:
        if body.persona_id is None:
            raise HTTPException(status_code=400, detail="persona_id is required to respond to a speaker")
        s = next((s for s in stored if s["persona_id"] == body.persona_id), None)
        if s is None or body.persona_id not in personas:
            raise HTTPException(status_code=400, detail="that resident is not a town hall speaker")
        text = state.residents.followup(personas[body.persona_id], reactions[body.persona_id],
                                        s.get("comment"), body.message, plan)
        followup = {"persona_id": body.persona_id, "message": body.message, "text": text}
    return {"plan_id": plan_id, "speakers": views, "followup": followup,
            "llm_available": bool(state.llm.available)}


@router.get("/residents/{persona_id}")
def get_resident(persona_id: int, state: StateDep) -> dict[str, Any]:
    p = _personas_ready(state).get(persona_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"resident {persona_id} not found")
    return p.public(state.residents.school_names)


@router.get("/residents/{persona_id}/chat")
def get_chat(persona_id: int, state: StateDep, me: PlayerDep) -> dict[str, Any]:
    if _personas_ready(state).get(persona_id) is None:
        raise HTTPException(status_code=404, detail=f"resident {persona_id} not found")
    return {"persona_id": persona_id, "messages": state.store.get_chat(me, persona_id)}


@router.post("/residents/{persona_id}/chat")
def chat(persona_id: int, body: ChatIn, state: StateDep,
         me: PlayerDep) -> dict[str, Any]:
    p = _personas_ready(state).get(persona_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"resident {persona_id} not found")
    plan = None
    reaction = None
    if body.plan_id:
        plan = require_plan(state, body.plan_id)
        reaction = next((r for r in state.store.get_reactions(body.plan_id) if r["persona_id"] == persona_id), None)
    history = state.store.get_chat(me, persona_id)
    reply = state.residents.chat(p, reaction, plan, history, body.message)
    if reply is not None:
        now = iso(utcnow())
        state.store.ensure_player(me)
        history = state.store.save_chat(me, persona_id, history + [
            {"role": "user", "content": body.message, "plan_id": body.plan_id, "at": now},
            {"role": "assistant", "content": reply, "plan_id": body.plan_id, "at": now},
        ])
    return {"persona_id": persona_id, "reply": reply, "messages": history,
            "llm_available": bool(state.llm.available),
            "error": None if reply is not None else "The resident could not answer (AI endpoint unavailable or reply rejected)."}


@router.post("/tools/custom/preview")
def custom_preview(body: CustomPreviewIn, state: StateDep) -> dict[str, Any]:
    schools = clean_json(state.sim().schools())
    try:
        return state.residents.custom_preview(body.description, schools)
    except CustomToolError as e:
        raise HTTPException(status_code=e.status, detail=e.detail) from e
