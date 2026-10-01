from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest

from pipeline.config import assumption
from residents import custom_tool, persona, reactions, townhall
from residents.blocks import BlockLabeler
from residents.llm import LLMClient, extract_json
from residents.names import FIRST_NAMES
from residents.service import ResidentsService
from residents.tests.fixtures import (
    FakeLLM,
    echo_reactions_llm,
    fake_person_deltas,
    write_population,
)
from residents.textcheck import collect_numbers, numbers_ok, safe_text


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return write_population(tmp_path_factory.mktemp("processed"))


@pytest.fixture(scope="module")
def personas(data_dir):
    persons, households = persona.load_population(data_dir)
    return persona.build_personas(persons, households, BlockLabeler.from_data_dir(data_dir),
                                  persona.school_names(data_dir), persona.exit_labels(data_dir))


def test_persona_count_and_stratification(data_dir, personas):
    n = assumption("residents.persona_count")
    persons, households = persona.load_population(data_dir)
    assert len(personas) == n
    assert len({p.person_id for p in personas}) == n
    schools = {s for p in personas for s in p.school_ids}
    assert schools == {"del_norte_hs", "oak_valley_ms", "design39", "stone_ranch_es"}
    assert {p.income_band for p in personas} == set(households["income_band"])
    eligible = persons[persons["age"] >= persona.PERSONA_MIN_AGE]["age"].map(persona.age_band)
    assert {p.age_band for p in personas} == set(eligible)
    assert all(p.age >= persona.PERSONA_MIN_AGE for p in personas)


def test_persona_privacy_rules(personas):
    for p in personas:
        assert p.first_name in FIRST_NAMES
        assert " " not in p.first_name  # first name only, never a surname
        assert len(p.values) == 3 and len(set(p.values)) == 3
        assert set(p.values) <= set(assumption("residents.values_list"))
        assert p.block.startswith("near ")
        pub = p.public()
        for banned in ("home_building_id", "person_id", "household_id", "address", "building_id"):
            assert banned not in pub
        assert pub["x"] % 150 == 0 and pub["z"] % 150 == 0  # block-level snapping


def test_personas_deterministic(data_dir, personas):
    persons, households = persona.load_population(data_dir)
    again = persona.build_personas(persons, households, BlockLabeler.from_data_dir(data_dir),
                                   persona.school_names(data_dir), persona.exit_labels(data_dir))
    assert [p.model_dump() for p in again] == [p.model_dump() for p in personas]


def test_persona_cache_and_backstory_upgrade(data_dir, tmp_path):
    cache = tmp_path / "personas.json"
    down = FakeLLM(available=False)
    ps = persona.load_or_build(data_dir, cache, down)
    assert cache.exists()
    assert all(p.backstory_source == "template" for p in ps)
    assert json.loads(cache.read_text())["synthetic"] is True
    llm = echo_reactions_llm()
    n = persona.upgrade_backstories(data_dir, cache, ps, llm)
    assert n == len(ps)
    again = persona.load_or_build(data_dir, cache, None)
    assert all(p.backstory_source == "llm" for p in again)
    assert again[0].backstory.endswith("loves the canyon trails.")


def _plan(tools=None):
    return {"id": "p1", "title": "Later bell", "pitch": "", "tools": tools if tools is not None else [
        {"tool": "bell_time", "params": {"school": "del_norte_hs", "start": "09:00"}}]}


def test_approval_deterministic_and_directional(data_dir, personas):
    persons, _ = persona.load_population(data_dir)
    plan = _plan()
    pdl = fake_person_deltas(persons, plan["tools"])
    a = reactions.compute_reactions(personas, pdl, plan, 120000.0)
    b = reactions.compute_reactions(personas, pdl, plan, 120000.0)
    assert a == b
    assert all(0.0 <= r["approval"] <= 1.0 for r in a)
    assert all(r["approves"] == (r["approval"] >= 0.5) for r in a)
    # households with a Del Norte student save drop-off time -> they approve more on average
    dn = [r["approval"] for r, p in zip(a, personas, strict=True) if "del_norte_hs" in p.school_ids]
    other = [r["approval"] for r, p in zip(a, personas, strict=True) if not p.school_ids]
    assert sum(dn) / len(dn) > sum(other) / len(other)
    s1 = reactions.approval_summary(a)
    assert s1 == reactions.approval_summary(a)
    assert s1["p10"] <= s1["median"] <= s1["p90"]


def test_approval_formula_weights():
    params = reactions.ApprovalParams.load()
    base = {"commute_min": 0.0, "dropoff_min": 0.0, "cost_usd_year": 0.0, "street_change": False}
    s0 = reactions.approval_score(["cost", "community", "property"], base, [], params)
    assert s0 == pytest.approx(params.base)
    faster = reactions.approval_score(["time", "cost", "property"], base | {"commute_min": -2.0}, [], params)
    assert faster == pytest.approx(params.base + 2 * params.per_min_commute_saved * params.multipliers["time"])
    street = reactions.approval_score(["property", "cost", "time"], base | {"street_change": True}, [], params)
    assert street < s0
    averse = reactions.approval_score(["change_averse", "cost", "time"], base, ["bell_time"], params)
    assert averse == pytest.approx(params.base + params.change_averse_penalty)


def test_missing_person_deltas_is_neutral(personas):
    rs = reactions.compute_reactions(personas[:5], None, _plan([]), 0.0)
    for r, p in zip(rs, personas[:5], strict=True):
        if "change_averse" not in p.values:
            assert r["approval"] == 0.5


def test_insert_metric_replaces_existing():
    rep = {"metrics": [{"id": "avg_commute_min"}, {"id": "resident_approval_pct", "plan": {}}]}
    m = reactions.approval_metric({"median": 50.0, "p10": 45.0, "p90": 55.0}, 500)
    reactions.insert_approval_metric(rep, m)
    ids = [x["id"] for x in rep["metrics"]]
    assert ids == ["avg_commute_min", "resident_approval_pct"]
    assert rep["metrics"][-1]["plan"]["median"] == 50.0


def test_texts_with_llm_down_and_up(data_dir, personas):
    persons, _ = persona.load_population(data_dir)
    plan = _plan()
    rs = reactions.compute_reactions(personas[:25], fake_person_deltas(persons, plan["tools"]), plan, 0.0)
    pmap = {p.persona_id: p for p in personas}
    assert reactions.generate_texts(pmap, rs, plan, FakeLLM(available=False), {}) == {}
    texts = reactions.generate_texts(pmap, rs, plan, echo_reactions_llm(), {})
    assert set(texts) == {r["persona_id"] for r in rs}


def test_llm_invented_numbers_rejected(data_dir, personas):
    persons, _ = persona.load_population(data_dir)
    plan = _plan()
    rs = reactions.compute_reactions(personas[:3], fake_person_deltas(persons, plan["tools"]), plan, 0.0)
    bad = FakeLLM(lambda m, model, j: json.dumps(
        {"reactions": [{"id": r["persona_id"], "text": "This saves me 97 minutes!"} for r in rs]}))
    assert reactions.generate_texts({p.persona_id: p for p in personas}, rs, plan, bad, {}) == {}


def test_textcheck():
    nums, clocks = collect_numbers({"a": 3.14, "b": "6.0 minutes faster", "t": "starts 09:00"})
    assert numbers_ok("about 3.1 or 3 minutes, 6 faster at 09:00", nums, clocks)
    assert not numbers_ok("12 minutes", nums, clocks)
    assert safe_text("I live at 123 Maple Street", {123.0}) is None
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_townhall_speaker_selection(data_dir, personas):
    persons, _ = persona.load_population(data_dir)
    plan = _plan()
    rs = reactions.compute_reactions(personas, fake_person_deltas(persons, plan["tools"]), plan, 0.0)
    pmap = {p.persona_id: p for p in personas}
    sp = townhall.select_speakers(pmap, rs)
    assert len(sp) == 8
    assert sum(1 for *_x, side in sp if side == "against") == 4
    assert len({s for p, _r, _s in sp for s in p.school_ids}) >= 3
    against = sorted(r["approval"] for _p, r, side in sp if side == "against")
    favor = sorted(r["approval"] for _p, r, side in sp if side == "for")
    assert max(against) <= min(favor)


def test_custom_tool_preview_validation():
    schools = [{"id": "del_norte_hs", "entrances": [{"id": "main_dropoff"}]}]
    good = {"summary": "Walking school bus", "levers": [
        {"type": "mode_utility_shift", "mode": "walk", "applies_to": "students", "school": "del_norte_hs", "utils": 0.5}],
        "adoption_range": [0.05, 0.15], "cost_upfront_usd": 0, "cost_per_year_usd": 20000,
        "assumptions": ["Parents volunteer to walk groups"]}
    res = custom_tool.preview("walking school bus", FakeLLM(lambda *a: json.dumps(good)), schools)
    assert res["label"] == "LLM estimated" and res["tool"]["tool"] == "custom"
    bad = good | {"levers": [{"type": "mode_utility_shift", "mode": "teleport", "utils": 9}]}
    with pytest.raises(custom_tool.CustomToolError) as e:
        custom_tool.preview("x", FakeLLM(lambda *a: json.dumps(bad)), schools)
    assert e.value.status == 422
    with pytest.raises(custom_tool.CustomToolError) as e:
        custom_tool.preview("x", FakeLLM(available=False), schools)
    assert e.value.status == 503


def test_llm_client_endpoint_down_is_fast_and_breaks_circuit():
    c = LLMClient("http://127.0.0.1:9/v1", model_fast="m", timeout_s=2, connect_timeout_s=0.5)
    assert c.available
    assert c.chat([{"role": "user", "content": "hi"}]) is None
    assert not c.available  # circuit open: further calls skip immediately
    assert c.chat([{"role": "user", "content": "hi"}]) is None


def test_llm_client_parses_openai_reply(monkeypatch):
    def fake_post(url, headers, json, timeout):  # noqa: A002
        assert url.endswith("/chat/completions") and json["model"] == "fast-model"
        return httpx.Response(200, json={"choices": [{"message": {"content": " hello "}}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    c = LLMClient("http://llm/v1", model_fast="fast-model")
    assert c.chat([{"role": "user", "content": "hi"}]) == "hello"


def test_service_compute_metric(data_dir, tmp_path):
    svc = ResidentsService(data_dir, tmp_path / "personas.json", FakeLLM(available=False))
    persons = pd.read_parquet(data_dir / "persons.parquet")
    plan = _plan()
    rs, metric = svc.compute(plan, fake_person_deltas(persons, plan["tools"]), 50000.0)
    assert len(rs) == assumption("residents.persona_count")
    assert metric["id"] == "resident_approval_pct" and metric["plan"]["median"] is not None
    v = svc.reaction_view(rs[0])
    assert v is not None and v["text"] is None and "home_building_id" not in v
