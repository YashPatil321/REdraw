"""MongoDB (Atlas) persistence: the same interface as `api.store.Store`.

Selected when DATABASE_URL starts with `mongodb://` or `mongodb+srv://` (for
example a free MongoDB Atlas M0 cluster). The database name comes from the URL
path (`.../redraw?retryWrites=true`), defaulting to `redraw`.

Collections mirror the SQL tables of spec 9.2 (`api/db.py`):

    players             {_id: player_id, display_name, created_at}
    plans               {_id: plan_id, mission, title, pitch, author_id, tools, report,
                         check, status, votes, created_at, updated_at}
    votes               {_id: "<plan_id>:<player_id>", plan_id, player_id, value, created_at}
    jobs                {_id: job_id, plan_id, status, progress, message, error,
                         created_at, updated_at, started_at, finished_at}
    resident_reactions  {_id: "<plan_id>:<persona_id>", plan_id, persona_id, approval,
                         approves, deltas, text, created_at, updated_at}
    chats               {_id: "<player_id>:<persona_id>", player_id, persona_id, messages, updated_at}
    townhalls           {_id: plan_id, speakers, created_at, updated_at}
    households_real     reserved for home claiming (spec 13.4); created empty

Datetimes are stored as BSON dates (UTC). Report and tool JSON go through
`clean_json` first, like the SQL store.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from api.db import clean_json, iso, utcnow
from api.store import CHAT_HISTORY_LIMIT, new_id

DEFAULT_DB = "redraw"
COLLECTIONS = ("players", "plans", "votes", "jobs", "resident_reactions", "chats", "townhalls", "households_real")


def is_mongo_url(url: str) -> bool:
    return url.startswith(("mongodb://", "mongodb+srv://"))


def db_name_from_url(url: str) -> str:
    path = urlsplit(url).path.strip("/")
    return path.split("/")[0] if path else DEFAULT_DB


def _iso(v: Any) -> str | None:
    if isinstance(v, datetime) and v.tzinfo is None:
        v = v.replace(tzinfo=UTC)  # BSON dates come back naive (UTC)
    return iso(v)


class MongoStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    @classmethod
    def from_url(cls, url: str, client: MongoClient | None = None) -> MongoStore:
        client = client or MongoClient(url, serverSelectionTimeoutMS=10000, tz_aware=True)
        return cls(client[db_name_from_url(url)])

    def create_schema(self) -> None:
        """Create collections and indexes (idempotent). Raises if the server is unreachable."""
        have = set(self.db.list_collection_names())
        for name in COLLECTIONS:
            if name not in have:
                self.db.create_collection(name)
        self.db.plans.create_index([("mission", ASCENDING), ("created_at", DESCENDING)])
        self.db.plans.create_index([("mission", ASCENDING), ("votes", DESCENDING)])
        self.db.votes.create_index([("plan_id", ASCENDING)])
        self.db.jobs.create_index([("plan_id", ASCENDING), ("created_at", DESCENDING)])
        self.db.jobs.create_index([("status", ASCENDING)])
        self.db.resident_reactions.create_index([("plan_id", ASCENDING), ("persona_id", ASCENDING)])

    # ---------------------------------------------------------------- players
    def ensure_player(self, player_id: str) -> None:
        try:
            self.db.players.update_one({"_id": player_id},
                                       {"$setOnInsert": {"display_name": None, "created_at": utcnow()}},
                                       upsert=True)
        except DuplicateKeyError:
            pass  # created concurrently

    # ------------------------------------------------------------------ plans
    @staticmethod
    def _plan(doc: dict[str, Any], job_id: str | None = None) -> dict[str, Any]:
        return {
            "id": doc["_id"], "mission": doc["mission"], "title": doc["title"], "pitch": doc.get("pitch", ""),
            "author_id": doc.get("author_id"), "tools": doc.get("tools", []),
            "created_at": _iso(doc.get("created_at")), "updated_at": _iso(doc.get("updated_at")),
            "report": doc.get("report"), "status": doc.get("status", "draft"), "check": doc.get("check"),
            "votes": int(doc.get("votes", 0)), "job_id": job_id,
        }

    def create_plan(self, *, mission: str, title: str, pitch: str, tools: list[dict[str, Any]],
                    author_id: str | None, check: dict[str, Any] | None) -> dict[str, Any]:
        pid = new_id()
        now = utcnow()
        self.db.plans.insert_one({
            "_id": pid, "mission": mission, "title": title, "pitch": pitch, "author_id": author_id,
            "tools": clean_json(tools), "report": None, "check": clean_json(check),
            "status": "draft", "votes": 0, "created_at": now, "updated_at": now,
        })
        plan = self.get_plan(pid)
        assert plan is not None
        return plan

    def update_plan_content(self, plan_id: str, *, title: str, pitch: str, mission: str,
                            tools: list[dict[str, Any]], check: dict[str, Any] | None) -> None:
        self.db.plans.update_one({"_id": plan_id}, {"$set": {
            "title": title, "pitch": pitch, "mission": mission, "tools": clean_json(tools),
            "check": clean_json(check), "report": None, "status": "draft", "updated_at": utcnow(),
        }})
        self.db.resident_reactions.delete_many({"plan_id": plan_id})
        self.db.townhalls.delete_one({"_id": plan_id})

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        doc = self.db.plans.find_one({"_id": plan_id})
        if doc is None:
            return None
        job = self.db.jobs.find_one({"plan_id": plan_id}, {"_id": 1}, sort=[("created_at", DESCENDING)])
        return self._plan(doc, job["_id"] if job else None)

    def list_plans(self, mission: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        q = {"mission": mission} if mission else {}
        return [self._plan(d) for d in self.db.plans.find(q).sort("created_at", DESCENDING).limit(limit)]

    def set_plan_status(self, plan_id: str, status: str, report: dict[str, Any] | None = None,
                        clear_report: bool = False) -> None:
        values: dict[str, Any] = {"status": status, "updated_at": utcnow()}
        if report is not None:
            values["report"] = clean_json(report)
        elif clear_report:
            values["report"] = None
        self.db.plans.update_one({"_id": plan_id}, {"$set": values})

    # ------------------------------------------------------------------ votes
    def vote(self, plan_id: str, player_id: str, value: int) -> int:
        key = f"{plan_id}:{player_id}"
        if value == 0:
            self.db.votes.delete_one({"_id": key})
        else:
            self.db.votes.replace_one({"_id": key}, {"plan_id": plan_id, "player_id": player_id,
                                                     "value": int(value), "created_at": utcnow()}, upsert=True)
        agg = list(self.db.votes.aggregate([{"$match": {"plan_id": plan_id}},
                                            {"$group": {"_id": None, "total": {"$sum": "$value"}}}]))
        total = int(agg[0]["total"]) if agg else 0
        self.db.plans.update_one({"_id": plan_id}, {"$set": {"votes": total}})
        return total

    def my_vote(self, plan_id: str, player_id: str) -> int:
        doc = self.db.votes.find_one({"_id": f"{plan_id}:{player_id}"})
        return int(doc["value"]) if doc else 0

    # ------------------------------------------------------------------- jobs
    @staticmethod
    def _job(doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": doc["_id"], "plan_id": doc["plan_id"], "status": doc["status"],
            "progress": float(doc.get("progress", 0.0)), "message": doc.get("message", ""),
            "error": doc.get("error"), "created_at": _iso(doc.get("created_at")),
            "started_at": _iso(doc.get("started_at")), "finished_at": _iso(doc.get("finished_at")),
        }

    def create_job(self, plan_id: str) -> dict[str, Any]:
        jid = new_id()
        now = utcnow()
        self.db.jobs.insert_one({"_id": jid, "plan_id": plan_id, "status": "queued", "progress": 0.0,
                                 "message": "queued", "error": None, "started_at": None,
                                 "finished_at": None, "created_at": now, "updated_at": now})
        job = self.get_job(jid)
        assert job is not None
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        doc = self.db.jobs.find_one({"_id": job_id})
        return self._job(doc) if doc else None

    def active_job_for_plan(self, plan_id: str) -> dict[str, Any] | None:
        doc = self.db.jobs.find_one({"plan_id": plan_id, "status": {"$in": ["queued", "running"]}},
                                    sort=[("created_at", DESCENDING)])
        return self._job(doc) if doc else None

    def update_job(self, job_id: str, **values: Any) -> None:
        values["updated_at"] = utcnow()
        self.db.jobs.update_one({"_id": job_id}, {"$set": values})

    def fail_stale_jobs(self) -> int:
        """Jobs left queued/running by a previous server process can never finish."""
        active = {"status": {"$in": ["queued", "running"]}}
        stale = [d["plan_id"] for d in self.db.jobs.find(active, {"plan_id": 1})]
        now = utcnow()
        res = self.db.jobs.update_many(active, {"$set": {
            "status": "failed", "error": "server restarted before the run finished; run it again",
            "finished_at": now, "updated_at": now}})
        if stale:
            self.db.plans.update_many({"_id": {"$in": stale}, **active},
                                      {"$set": {"status": "failed", "updated_at": now}})
        return int(res.modified_count)

    # -------------------------------------------------------------- reactions
    def save_reactions(self, plan_id: str, rows: list[dict[str, Any]]) -> None:
        now = utcnow()
        self.db.resident_reactions.delete_many({"plan_id": plan_id})
        if rows:
            self.db.resident_reactions.insert_many([{
                "_id": f"{plan_id}:{int(r['persona_id'])}", "plan_id": plan_id,
                "persona_id": int(r["persona_id"]), "approval": float(r["approval"]),
                "approves": bool(r["approves"]), "deltas": clean_json(r.get("deltas")),
                "text": r.get("text"), "created_at": now, "updated_at": now,
            } for r in rows])

    def set_reaction_texts(self, plan_id: str, texts: dict[int, str]) -> None:
        for pid, text in texts.items():
            self.db.resident_reactions.update_one({"_id": f"{plan_id}:{int(pid)}"},
                                                  {"$set": {"text": text, "updated_at": utcnow()}})

    def get_reactions(self, plan_id: str) -> list[dict[str, Any]]:
        return [{
            "persona_id": d["persona_id"], "approval": d["approval"], "approves": d["approves"],
            "deltas": d.get("deltas") or {}, "text": d.get("text"),
        } for d in self.db.resident_reactions.find({"plan_id": plan_id}).sort("persona_id", ASCENDING)]

    # ------------------------------------------------------------------ chats
    def get_chat(self, player_id: str, persona_id: int) -> list[dict[str, Any]]:
        doc = self.db.chats.find_one({"_id": f"{player_id}:{int(persona_id)}"})
        return list(doc["messages"]) if doc else []

    def save_chat(self, player_id: str, persona_id: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept = messages[-CHAT_HISTORY_LIMIT:]
        self.db.chats.replace_one({"_id": f"{player_id}:{int(persona_id)}"}, {
            "player_id": player_id, "persona_id": int(persona_id), "messages": clean_json(kept),
            "updated_at": utcnow()}, upsert=True)
        return kept

    # -------------------------------------------------------------- townhalls
    def get_townhall(self, plan_id: str) -> list[dict[str, Any]] | None:
        doc = self.db.townhalls.find_one({"_id": plan_id})
        return list(doc["speakers"]) if doc else None

    def save_townhall(self, plan_id: str, speakers: list[dict[str, Any]]) -> None:
        now = utcnow()
        self.db.townhalls.replace_one({"_id": plan_id}, {"speakers": clean_json(speakers),
                                                         "created_at": now, "updated_at": now}, upsert=True)
