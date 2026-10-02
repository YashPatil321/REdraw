from __future__ import annotations

import json
import struct
import threading

from api.tests.conftest import wait_baseline, wait_job
from api.tests.fake_sim import FakeSimService
from residents.tests.fixtures import echo_reactions_llm

BELL = {"mission": "morning_crunch", "title": "Later bell", "pitch": "Move Del Norte to 9:00",
        "tools": [{"tool": "bell_time", "params": {"school": "del_norte_hs", "start": "09:00"}}]}


def _rdpb_ok(data: bytes) -> dict:
    assert data[:4] == b"RDPB"
    version, hlen = struct.unpack("<II", data[4:12])
    assert version == 1
    return json.loads(data[12:12 + hlen])


def test_cookie_set_once(client):
    r = client.get("/health")
    assert r.status_code == 200
    cookie = r.cookies.get("redraw_player")
    assert cookie and len(cookie) == 36
    r2 = client.get("/health")
    assert "redraw_player" not in r2.headers.get("set-cookie", "")


def test_cors_preflight(client):
    r = client.options("/plans", headers={"Origin": "http://localhost:5173",
                                          "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert r.headers.get("access-control-allow-credentials") == "true"


def test_world_endpoints(client):
    meta = client.get("/world/meta").json()
    assert meta["synthetic"] is True
    assert meta["assets"] == {"base_url": "/assets/", "manifest": "/assets/manifest.json"}
    assert meta["mission"]["id"] == "morning_crunch"
    assert meta["time"] == {"bin_start_s": 21600, "bin_s": 300, "n_bins": 48,
                            "report_start_s": 23400, "report_end_s": 34200}
    assert any(m["id"] == "resident_approval_pct" for m in meta["metrics"])
    assert meta["hero"]["school_id"] == "del_norte_hs"
    assert meta["region"]["origin"] == {"lat": 33.005, "lon": -117.125}
    assert meta["unverified_inputs"]
    net = client.get("/world/network").json()
    assert net["n_edges"] == 1 and net["edges"][0]["pts"]
    tools = client.get("/tools").json()
    assert tools["mission"] == "morning_crunch"
    custom = [t for t in tools["tools"] if t["id"] == "custom"]
    assert custom and custom[0]["enabled_in_mvp"] is True
    wait_baseline(client)
    schools = client.get("/world/schools").json()
    dn = next(s for s in schools if s["id"] == "del_norte_hs")
    assert dn["entrances"][0]["key"] == "del_norte_hs/main_dropoff"
    assert dn["entrances"][0]["baseline"]["max_queue_cars"] == 12.0


def test_buildings(client):
    b = client.get("/world/buildings/1001").json()
    assert b["id"] == 1001 and "households" in b and b["block"].startswith("near ")
    assert client.get("/world/buildings/999999").status_code == 404


def test_baseline_warming_then_ready(make_client, data_dir):
    gate = threading.Event()

    class SlowSim(FakeSimService):
        def run(self, plan, **kw):
            if plan is None:
                gate.wait(10)
            return super().run(plan, **kw)

    c, _ = make_client(sim=SlowSim(data_dir))
    r = c.get("/baseline")
    assert r.status_code == 503 and "warming up" in r.json()["detail"]
    assert c.get("/baseline/playback").status_code == 503
    gate.set()
    body = wait_baseline(c)
    assert body["playback_url"] == "/baseline/playback"
    assert body["summary"]["metrics"][0]["id"] == "avg_commute_min"
    assert body["calibration"]["status"] == "uncalibrated"
    _rdpb_ok(c.get("/baseline/playback").content)


def test_create_get_check_plan(client):
    r = client.post("/plans", json=BELL)
    assert r.status_code == 201
    plan = r.json()
    assert plan["status"] == "draft" and plan["check"]["ok"] is True and plan["votes"] == 0
    assert plan["author_id"] == client.cookies.get("redraw_player")
    got = client.get(f"/plans/{plan['id']}").json()
    assert got["title"] == "Later bell" and got["report"] is None
    assert client.get("/plans/nope").status_code == 404
    chk = client.post("/plans/check", json=BELL | {"tools": [{"tool": "bell_time", "params": {"school": "x"}}]})
    assert chk.status_code == 200 and chk.json()["ok"] is False


def test_invalid_shapes_422_and_invalid_params_saved(client):
    assert client.post("/plans", json=BELL | {"tools": [{"params": {}}]}).status_code == 422
    assert client.post("/plans", json=BELL | {"tools": "bell_time"}).status_code == 422
    bad = client.post("/plans", json=BELL | {"tools": [{"tool": "bell_time", "params": {"school": "zzz"}}]})
    assert bad.status_code == 201 and bad.json()["check"]["ok"] is False
    r = client.post(f"/plans/{bad.json()['id']}/run")
    assert r.status_code == 400 and "unknown school" in r.json()["detail"]


def test_run_job_lifecycle_report_playback_residents(client):
    plan = client.post("/plans", json=BELL).json()
    assert client.get(f"/plans/{plan['id']}/playback").status_code == 404
    job_id = client.post(f"/plans/{plan['id']}/run").json()["job_id"]
    job = wait_job(client, job_id)
    assert job["status"] == "done", job
    assert job["progress"] == 1.0 and job["error"] is None and job["plan_id"] == plan["id"]
    done = client.get(f"/plans/{plan['id']}").json()
    assert done["status"] == "done" and done["job_id"] == job_id
    rep = done["report"]
    ids = [m["id"] for m in rep["metrics"]]
    assert "resident_approval_pct" in ids
    appr = next(m for m in rep["metrics"] if m["id"] == "resident_approval_pct")
    assert 0 <= appr["plan"]["median"] <= 100
    assert rep["metrics"][0]["plan"]["p90"] is None  # NaN from the sim -> null
    hdr = _rdpb_ok(client.get(f"/plans/{plan['id']}/playback").content)
    assert hdr["plan_id"] == plan["id"]
    # an ephemeral disk lost the file: the playback is rebuilt from the plan (fixed seeds)
    pb = client.app.state.jobs.playback_path(plan["id"])
    pb.unlink()
    assert _rdpb_ok(client.get(f"/plans/{plan['id']}/playback").content)["plan_id"] == plan["id"]
    assert pb.exists()

    res = client.get(f"/plans/{plan['id']}/residents").json()
    assert res["approval_pct"] == appr["plan"]["median"]
    assert len(res["reactions"]) == 500
    r0 = res["reactions"][0]
    for k in ("persona_id", "first_name", "age", "block", "x", "z", "school_ids", "values",
              "approval", "approves", "deltas", "text"):
        assert k in r0
    assert set(r0["deltas"]) >= {"commute_min", "dropoff_min", "cost_usd_year", "street_change"}
    assert all(r["text"] is None for r in res["reactions"])  # LLM endpoint is down
    assert res["text_status"] in ("unavailable", "pending")


def test_approval_deterministic_across_runs(client):
    a = client.post("/plans", json=BELL).json()
    b = client.post("/plans", json=BELL).json()
    for p in (a, b):
        assert wait_job(client, client.post(f"/plans/{p['id']}/run").json()["job_id"])["status"] == "done"
    ra = client.get(f"/plans/{a['id']}/residents").json()
    rb = client.get(f"/plans/{b['id']}/residents").json()
    assert ra["approval_pct"] == rb["approval_pct"]
    assert [r["approval"] for r in ra["reactions"]] == [r["approval"] for r in rb["reactions"]]


def test_run_failure_marks_job_failed(make_client, data_dir):
    class Boom(FakeSimService):
        def run(self, plan, **kw):
            if plan is None:
                return super().run(plan, **kw)
            raise RuntimeError("kaboom")

    c, _ = make_client(sim=Boom(data_dir))
    plan = c.post("/plans", json=BELL).json()
    job = wait_job(c, c.post(f"/plans/{plan['id']}/run").json()["job_id"])
    assert job["status"] == "failed" and "kaboom" in job["error"]
    assert c.get(f"/plans/{plan['id']}").json()["status"] == "failed"


def test_list_sort_and_vote(client):
    p1 = client.post("/plans", json=BELL | {"title": "A"}).json()
    p2 = client.post("/plans", json=BELL | {"title": "A much longer title"}).json()
    p3 = client.post("/plans", json={"mission": "morning_crunch", "title": "Empty", "pitch": "", "tools": []}).json()
    for p in (p1, p2):
        wait_job(client, client.post(f"/plans/{p['id']}/run").json()["job_id"])
    new = client.get("/plans").json()["plans"]
    assert [p["id"] for p in new[:3]] == [p3["id"], p2["id"], p1["id"]]
    assert client.post(f"/plans/{p1['id']}/vote", json={"value": 1}).json() == {"votes": 1}
    assert client.post(f"/plans/{p1['id']}/vote", json={"value": 1}).json() == {"votes": 1}  # idempotent
    other = client.__class__(client.app)
    assert other.post(f"/plans/{p1['id']}/vote", json={"value": -1}).json() == {"votes": 0}
    assert other.post(f"/plans/{p2['id']}/vote", json={"value": 1}).json() == {"votes": 1}
    assert client.post(f"/plans/{p2['id']}/vote", json={"value": 1}).json() == {"votes": 2}
    assert client.post(f"/plans/{p2['id']}/vote", json={"value": 0}).json() == {"votes": 1}
    assert client.post(f"/plans/{p2['id']}/vote", json={"value": 5}).status_code == 422
    by_votes = client.get("/plans?sort=votes").json()["plans"]
    assert by_votes[0]["id"] == p2["id"]
    # avg_commute_min is better=lower; p2's longer title gives a lower fake commute
    by_metric = client.get("/plans?sort=avg_commute_min&mission=morning_crunch").json()["plans"]
    assert [p["id"] for p in by_metric[:2]] == [p2["id"], p1["id"]]
    assert by_metric[-1]["headline"] == {}
    assert "avg_commute_min" in by_metric[0]["headline"]
    assert "resident_approval_pct" in by_metric[0]["headline"]
    assert client.get("/plans?sort=bogus").status_code == 400
    assert client.get("/plans?mission=other").json()["plans"] == []


def test_update_plan_author_only(client):
    plan = client.post("/plans", json=BELL).json()
    r = client.put(f"/plans/{plan['id']}", json=BELL | {"title": "Renamed"})
    assert r.status_code == 200 and r.json()["title"] == "Renamed"
    other = client.__class__(client.app)
    assert other.put(f"/plans/{plan['id']}", json=BELL).status_code == 403


def test_residents_chat_townhall_llm_down(client):
    plan = client.post("/plans", json=BELL).json()
    assert client.get(f"/plans/{plan['id']}/residents").status_code == 404
    wait_job(client, client.post(f"/plans/{plan['id']}/run").json()["job_id"])
    r = client.post("/residents/3/chat", json={"message": "Hi there", "plan_id": plan["id"]}).json()
    assert r["reply"] is None and r["messages"] == [] and r["llm_available"] is False
    th = client.post(f"/plans/{plan['id']}/townhall", json={}).json()
    assert len(th["speakers"]) == 8 and all(s["comment"] is None for s in th["speakers"])
    assert len({s for sp in th["speakers"] for s in sp["school_ids"]}) >= 3
    pub = client.get("/residents/3").json()
    assert pub["persona_id"] == 3 and "home_building_id" not in pub
    assert client.get("/residents/99999").status_code == 404
    cp = client.post("/tools/custom/preview", json={"description": "walking school bus"})
    assert cp.status_code == 503


def test_residents_with_llm_up(make_client):
    llm = echo_reactions_llm()
    c, _ = make_client(llm=llm)
    plan = c.post("/plans", json=BELL).json()
    wait_job(c, c.post(f"/plans/{plan['id']}/run").json()["job_id"])
    import time

    for _ in range(200):
        res = c.get(f"/plans/{plan['id']}/residents").json()
        if res["text_status"] == "complete":
            break
        time.sleep(0.05)
    assert res["text_status"] == "complete"
    assert all(r["text"] for r in res["reactions"])
    chat = c.post("/residents/5/chat", json={"message": "What do you think?", "plan_id": plan["id"]}).json()
    assert chat["reply"] and len(chat["messages"]) == 2
    for i in range(6):
        c.post("/residents/5/chat", json={"message": f"again {i}"})
    hist = c.get("/residents/5/chat").json()["messages"]
    assert len(hist) == 10  # last 10 messages kept
    th = c.post(f"/plans/{plan['id']}/townhall", json={}).json()
    assert all(s["comment"] for s in th["speakers"])
    spk = th["speakers"][0]["persona_id"]
    fu = c.post(f"/plans/{plan['id']}/townhall", json={"persona_id": spk, "message": "We will add guards."}).json()
    assert fu["followup"]["persona_id"] == spk and fu["followup"]["text"]
