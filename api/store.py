"""Persistence for plans, votes, jobs, reactions, chats and town halls."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Engine, and_, delete, func, insert, select, update
from sqlalchemy.exc import IntegrityError

from api.db import (
    chats,
    clean_json,
    iso,
    jobs,
    plans,
    players,
    resident_reactions,
    townhalls,
    utcnow,
    votes,
)

CHAT_HISTORY_LIMIT = 10  # spec 8.3: keep last 10 messages per player per persona


def new_id() -> str:
    return str(uuid.uuid4())


class Store:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ---------------------------------------------------------------- players
    def ensure_player(self, player_id: str) -> None:
        with self.engine.begin() as conn:
            exists = conn.execute(select(players.c.id).where(players.c.id == player_id)).first()
            if exists:
                return
            try:
                with conn.begin_nested():
                    conn.execute(insert(players).values(id=player_id, created_at=utcnow()))
            except IntegrityError:
                pass  # created concurrently

    # ------------------------------------------------------------------ plans
    @staticmethod
    def _plan_row(row: Any, job_id: str | None = None) -> dict[str, Any]:
        m = row._mapping
        return {
            "id": m["id"],
            "mission": m["mission"],
            "title": m["title"],
            "pitch": m["pitch"],
            "author_id": m["author_id"],
            "tools": m["tools"],
            "created_at": iso(m["created_at"]),
            "updated_at": iso(m["updated_at"]),
            "report": m["report"],
            "status": m["status"],
            "check": m["plan_check"],
            "votes": m["votes"],
            "job_id": job_id,
        }

    def create_plan(self, *, mission: str, title: str, pitch: str, tools: list[dict[str, Any]],
                    author_id: str | None, check: dict[str, Any] | None) -> dict[str, Any]:
        pid = new_id()
        now = utcnow()
        with self.engine.begin() as conn:
            conn.execute(insert(plans).values(
                id=pid, mission=mission, title=title, pitch=pitch, author_id=author_id,
                tools=clean_json(tools), report=None, plan_check=clean_json(check),
                status="draft", votes=0, created_at=now, updated_at=now,
            ))
        plan = self.get_plan(pid)
        assert plan is not None
        return plan

    def update_plan_content(self, plan_id: str, *, title: str, pitch: str, mission: str,
                            tools: list[dict[str, Any]], check: dict[str, Any] | None) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(plans).where(plans.c.id == plan_id).values(
                title=title, pitch=pitch, mission=mission, tools=clean_json(tools),
                plan_check=clean_json(check), report=None, status="draft", updated_at=utcnow(),
            ))
            conn.execute(delete(resident_reactions).where(resident_reactions.c.plan_id == plan_id))
            conn.execute(delete(townhalls).where(townhalls.c.plan_id == plan_id))

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(plans).where(plans.c.id == plan_id)).first()
            if row is None:
                return None
            job = conn.execute(
                select(jobs.c.id).where(jobs.c.plan_id == plan_id)
                .order_by(jobs.c.created_at.desc()).limit(1)
            ).first()
        return self._plan_row(row, job[0] if job else None)

    def list_plans(self, mission: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        q = select(plans).order_by(plans.c.created_at.desc()).limit(limit)
        if mission:
            q = q.where(plans.c.mission == mission)
        with self.engine.connect() as conn:
            return [self._plan_row(r) for r in conn.execute(q)]

    def set_plan_status(self, plan_id: str, status: str, report: dict[str, Any] | None = None,
                        clear_report: bool = False) -> None:
        values: dict[str, Any] = {"status": status, "updated_at": utcnow()}
        if report is not None:
            values["report"] = clean_json(report)
        elif clear_report:
            values["report"] = None
        with self.engine.begin() as conn:
            conn.execute(update(plans).where(plans.c.id == plan_id).values(**values))

    # ------------------------------------------------------------------ votes
    def vote(self, plan_id: str, player_id: str, value: int) -> int:
        with self.engine.begin() as conn:
            conn.execute(delete(votes).where(and_(votes.c.plan_id == plan_id,
                                                  votes.c.player_id == player_id)))
            if value != 0:
                conn.execute(insert(votes).values(plan_id=plan_id, player_id=player_id,
                                                  value=value, created_at=utcnow()))
            total = conn.execute(
                select(func.coalesce(func.sum(votes.c.value), 0)).where(votes.c.plan_id == plan_id)
            ).scalar_one()
            conn.execute(update(plans).where(plans.c.id == plan_id).values(votes=int(total)))
        return int(total)

    def my_vote(self, plan_id: str, player_id: str) -> int:
        with self.engine.connect() as conn:
            v = conn.execute(select(votes.c.value).where(and_(
                votes.c.plan_id == plan_id, votes.c.player_id == player_id))).scalar()
        return int(v or 0)

    # ------------------------------------------------------------------- jobs
    @staticmethod
    def _job_row(row: Any) -> dict[str, Any]:
        m = row._mapping
        return {
            "id": m["id"], "plan_id": m["plan_id"], "status": m["status"],
            "progress": float(m["progress"]), "message": m["message"], "error": m["error"],
            "created_at": iso(m["created_at"]), "started_at": iso(m["started_at"]),
            "finished_at": iso(m["finished_at"]),
        }

    def create_job(self, plan_id: str) -> dict[str, Any]:
        jid = new_id()
        now = utcnow()
        with self.engine.begin() as conn:
            conn.execute(insert(jobs).values(id=jid, plan_id=plan_id, status="queued", progress=0.0,
                                             message="queued", created_at=now, updated_at=now))
        job = self.get_job(jid)
        assert job is not None
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(jobs).where(jobs.c.id == job_id)).first()
        return self._job_row(row) if row else None

    def active_job_for_plan(self, plan_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(jobs).where(and_(jobs.c.plan_id == plan_id,
                                        jobs.c.status.in_(["queued", "running"])))
                .order_by(jobs.c.created_at.desc()).limit(1)
            ).first()
        return self._job_row(row) if row else None

    def update_job(self, job_id: str, **values: Any) -> None:
        values["updated_at"] = utcnow()
        with self.engine.begin() as conn:
            conn.execute(update(jobs).where(jobs.c.id == job_id).values(**values))

    def fail_stale_jobs(self) -> int:
        """Jobs left queued/running by a previous server process can never finish."""
        with self.engine.begin() as conn:
            stale = [r[0] for r in conn.execute(
                select(jobs.c.plan_id).where(jobs.c.status.in_(["queued", "running"])))]
            res = conn.execute(update(jobs).where(jobs.c.status.in_(["queued", "running"])).values(
                status="failed", error="server restarted before the run finished; run it again",
                finished_at=utcnow(), updated_at=utcnow()))
            if stale:
                conn.execute(update(plans).where(and_(plans.c.id.in_(stale),
                                                      plans.c.status.in_(["queued", "running"])))
                             .values(status="failed", updated_at=utcnow()))
        return int(res.rowcount or 0)

    # -------------------------------------------------------------- reactions
    def save_reactions(self, plan_id: str, rows: list[dict[str, Any]]) -> None:
        now = utcnow()
        with self.engine.begin() as conn:
            conn.execute(delete(resident_reactions).where(resident_reactions.c.plan_id == plan_id))
            if rows:
                conn.execute(insert(resident_reactions), [{
                    "plan_id": plan_id, "persona_id": int(r["persona_id"]),
                    "approval": float(r["approval"]), "approves": bool(r["approves"]),
                    "deltas": clean_json(r.get("deltas")), "text": r.get("text"),
                    "created_at": now, "updated_at": now,
                } for r in rows])

    def set_reaction_texts(self, plan_id: str, texts: dict[int, str]) -> None:
        if not texts:
            return
        with self.engine.begin() as conn:
            for pid, text in texts.items():
                conn.execute(update(resident_reactions).where(and_(
                    resident_reactions.c.plan_id == plan_id,
                    resident_reactions.c.persona_id == int(pid))).values(text=text, updated_at=utcnow()))

    def get_reactions(self, plan_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(resident_reactions)
                                .where(resident_reactions.c.plan_id == plan_id)
                                .order_by(resident_reactions.c.persona_id))
            return [{
                "persona_id": r.persona_id, "approval": r.approval, "approves": r.approves,
                "deltas": r.deltas or {}, "text": r.text,
            } for r in rows]

    # ------------------------------------------------------------------ chats
    def get_chat(self, player_id: str, persona_id: int) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            row = conn.execute(select(chats.c.messages).where(and_(
                chats.c.player_id == player_id, chats.c.persona_id == persona_id))).first()
        return list(row[0]) if row else []

    def save_chat(self, player_id: str, persona_id: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept = messages[-CHAT_HISTORY_LIMIT:]
        with self.engine.begin() as conn:
            conn.execute(delete(chats).where(and_(chats.c.player_id == player_id,
                                                  chats.c.persona_id == persona_id)))
            conn.execute(insert(chats).values(player_id=player_id, persona_id=persona_id,
                                              messages=clean_json(kept), updated_at=utcnow()))
        return kept

    # -------------------------------------------------------------- townhalls
    def get_townhall(self, plan_id: str) -> list[dict[str, Any]] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(townhalls.c.speakers).where(townhalls.c.plan_id == plan_id)).first()
        return list(row[0]) if row else None

    def save_townhall(self, plan_id: str, speakers: list[dict[str, Any]]) -> None:
        now = utcnow()
        with self.engine.begin() as conn:
            conn.execute(delete(townhalls).where(townhalls.c.plan_id == plan_id))
            conn.execute(insert(townhalls).values(plan_id=plan_id, speakers=clean_json(speakers),
                                                  created_at=now, updated_at=now))
