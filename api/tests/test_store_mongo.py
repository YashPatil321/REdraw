"""MongoStore behaves like the SQL Store (run against in-memory mongomock)."""

from __future__ import annotations

import mongomock
import pytest

from api.store_mongo import MongoStore, db_name_from_url, is_mongo_url


@pytest.fixture
def store() -> MongoStore:
    s = MongoStore.from_url("mongodb://localhost/redraw_unit", client=mongomock.MongoClient(tz_aware=True))
    s.create_schema()
    s.create_schema()  # idempotent
    return s


def test_url_helpers():
    assert is_mongo_url("mongodb+srv://u:p@cluster0.x.mongodb.net/redraw?retryWrites=true")
    assert not is_mongo_url("sqlite:///x.db")
    assert db_name_from_url("mongodb+srv://u:p@c.mongodb.net/game?retryWrites=true") == "game"
    assert db_name_from_url("mongodb+srv://u:p@c.mongodb.net/?retryWrites=true") == "redraw"


def test_plans_votes_jobs(store: MongoStore):
    store.ensure_player("p1")
    store.ensure_player("p1")
    plan = store.create_plan(mission="morning_crunch", title="T", pitch="", tools=[{"tool": "bell_time", "params": {}}],
                             author_id="p1", check={"ok": True})
    assert plan["status"] == "draft" and plan["votes"] == 0 and plan["created_at"].endswith("Z")
    assert store.vote(plan["id"], "p1", 1) == 1
    assert store.vote(plan["id"], "p2", -1) == 0
    assert store.vote(plan["id"], "p2", 1) == 2
    assert store.my_vote(plan["id"], "p2") == 1
    assert store.vote(plan["id"], "p2", 0) == 1 and store.my_vote(plan["id"], "p2") == 0
    job = store.create_job(plan["id"])
    assert store.active_job_for_plan(plan["id"])["id"] == job["id"]
    assert store.get_plan(plan["id"])["job_id"] == job["id"]
    store.set_plan_status(plan["id"], "running")
    assert store.fail_stale_jobs() == 1
    assert store.get_job(job["id"])["status"] == "failed" and store.get_plan(plan["id"])["status"] == "failed"
    store.set_plan_status(plan["id"], "done", report={"metrics": []})
    assert store.list_plans("morning_crunch")[0]["report"] == {"metrics": []}


def test_reactions_chats_townhall(store: MongoStore):
    plan = store.create_plan(mission="morning_crunch", title="T", pitch="", tools=[], author_id=None, check=None)
    pid = plan["id"]
    store.save_reactions(pid, [{"persona_id": 2, "approval": 0.7, "approves": True, "deltas": {"commute_min": -1}},
                               {"persona_id": 1, "approval": 0.2, "approves": False}])
    store.set_reaction_texts(pid, {2: "Nice."})
    rs = store.get_reactions(pid)
    assert [r["persona_id"] for r in rs] == [1, 2] and rs[1]["text"] == "Nice." and rs[0]["deltas"] == {}
    msgs = [{"role": "user", "content": str(i)} for i in range(14)]
    assert len(store.save_chat("p1", 2, msgs)) == 10 and store.get_chat("p1", 2)[0]["content"] == "4"
    assert store.get_townhall(pid) is None
    store.save_townhall(pid, [{"persona_id": 1, "side": "against"}])
    assert store.get_townhall(pid) == [{"persona_id": 1, "side": "against"}]
    store.update_plan_content(pid, title="T2", pitch="", mission="morning_crunch", tools=[], check=None)
    assert store.get_reactions(pid) == [] and store.get_townhall(pid) is None
