"""Tiny fake processed-data directory for residents/api tests (clearly fake, not real geography)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHOOLS = [
    ("del_norte_hs", "Test High", (14, 18), 9, 12),
    ("oak_valley_ms", "Test Middle", (11, 13), 6, 8),
    ("design39", "Test K8", (5, 13), 0, 8),
    ("stone_ranch_es", "Test Elementary", (5, 10), 0, 5),
]
INCOME = ["lt50k", "50_100k", "100_150k", "150_200k", "gt200k"]
STREETS = ["Camino Del Sur", "Camino Del Norte", "Dove Canyon Road", "Test Lane"]


def write_population(d: Path, n_households: int = 900, seed: int = 1) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    hh_rows, p_rows = [], []
    pid = 1
    for h in range(1, n_households + 1):
        size = int(rng.integers(1, 6))
        x, z = float(rng.uniform(-2000, 2000)), float(rng.uniform(-2000, 2000))
        members = []
        n_adults = 1 if size == 1 else 2
        for _ in range(n_adults):
            age = int(rng.integers(19, 85))
            worker = bool(age < 67 and rng.random() < 0.75)
            wfh = bool(worker and rng.random() < 0.2)
            members.append(dict(age=age, is_worker=worker, works_from_home=wfh, school_id="", grade=-1,
                                work_exit="i15_south" if worker and not wfh and rng.random() < 0.7 else ""))
        for _ in range(size - n_adults):
            age = int(rng.integers(5, 19))
            if age >= 14:
                school, grade = "del_norte_hs", min(12, age - 5)
            elif age >= 11:
                school, grade = ("oak_valley_ms" if rng.random() < 0.6 else "design39"), age - 5
            else:
                school, grade = ("stone_ranch_es" if rng.random() < 0.6 else "design39"), max(0, age - 5)
            members.append(dict(age=age, is_worker=False, works_from_home=False, school_id=school,
                                grade=grade, work_exit=""))
        n_kids = sum(1 for m in members if m["school_id"])
        hh_rows.append(dict(household_id=h, building_id=1000 + h, block_group="", x=x, z=z,
                            home_node=h, size=size, vehicles=int(rng.integers(0, 4)),
                            income_band=INCOME[int(rng.integers(0, len(INCOME)))], n_kids=n_kids))
        for m in members:
            p_rows.append(dict(person_id=pid, household_id=h, age=m["age"], is_worker=m["is_worker"],
                               works_from_home=m["works_from_home"], work_node=-1,
                               work_exit=m["work_exit"], work_x=np.nan, work_z=np.nan,
                               school_id=m["school_id"], grade=m["grade"],
                               has_license=m["age"] >= 16, baseline_mode_hint=""))
            pid += 1
    pd.DataFrame(hh_rows).to_parquet(d / "households.parquet")
    pd.DataFrame(p_rows).to_parquet(d / "persons.parquet")

    edges = []
    for i, name in enumerate(STREETS):
        zline = -1500.0 + i * 1000.0
        geom = [c for xx in np.linspace(-2500, 2500, 11) for c in (float(xx), 300.0, zline)]
        edges.append(dict(edge_idx=i, u=i, v=i + 1, osmid="", name=name, ref="", highway="secondary",
                          lanes=2, maxspeed_kph=64.0, oneway=False, length_m=5000.0, capacity_vph=1700.0,
                          free_flow_s=280.0, geometry=geom, label=""))
    pd.DataFrame(edges).to_parquet(d / "network_edges.parquet")

    (d / "schools_resolved.json").write_text(json.dumps({"schools": [
        {"id": sid, "name": name, "grades": [g0, g1], "bell_start": "08:00", "x": 100.0 * k, "z": 0.0,
         "y": 300.0, "verified": False, "students": 100,
         "entrances": [{"id": "main_dropoff", "x": 100.0 * k, "z": 10.0, "node_id": k, "approach_edge_idx": 0,
                        "curb_spots": 8, "unload_seconds": 40, "verified": False}]}
        for k, (sid, name, _a, g0, g1) in enumerate(SCHOOLS)]}))
    (d / "exits.json").write_text(json.dumps({"exits": [
        {"id": "i15_south", "label": "I 15 south", "node_id": 1, "x": 0.0, "z": 0.0, "bearing_deg": 180}]}))
    (d / "region_meta.json").write_text(json.dumps({
        "contract_version": 1, "name": "test_region", "display_name": "Test Region (fake)", "synthetic": True,
        "bbox": {"south": 32.965, "north": 33.045, "west": -117.175, "east": -117.075},
        "origin": {"lat": 33.005, "lon": -117.125, "easting": 0.0, "northing": 0.0},
        "extent_scene": {"min_x": -4700.0, "max_x": 4700.0, "min_z": -4450.0, "max_z": 4450.0},
        "counts": {"households": len(hh_rows), "persons": len(p_rows)}}))
    (d / "buildings.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": None, "properties": {
            "id": 1000 + h, "type": "house", "height_m": 7.0, "address": None, "name": None,
            "centroid_x": hh_rows[h - 1]["x"], "centroid_z": hh_rows[h - 1]["z"]}}
        for h in range(1, 6)]}))
    return d


def fake_person_deltas(persons: pd.DataFrame, tools: list[dict]) -> dict:
    """Deterministic fake deltas: bell_time helps drop-off at that school, hurts some commutes."""
    ids = persons["person_id"].astype(int).tolist()
    tool_schools = {t["params"].get("school") for t in tools if t.get("tool") == "bell_time"}
    hh_school = persons[persons["school_id"] != ""].groupby("household_id")["school_id"].agg(set)
    commute, dropoff, kid, mb, mp, vc = [], [], [], [], [], []
    for r in persons.itertuples():
        schools = hh_school.get(r.household_id, set())
        commute.append((2.0 if r.person_id % 5 == 0 else -1.0) if r.is_worker and not r.works_from_home else float("nan"))
        hit = bool(schools & tool_schools)
        dropoff.append(-6.0 if hit else (float("nan") if not schools else 0.0))
        kid.append(-4.0 if r.school_id in tool_schools else (0.0 if r.school_id else float("nan")))
        mb.append("drive_alone" if r.is_worker else ("drive_dropoff" if r.school_id else ""))
        mp.append(mb[-1])
        vc.append(0.2 if r.person_id % 7 == 0 else 0.0)
    return {"person_ids": ids, "commute_delta_min": commute, "dropoff_delta_min": dropoff,
            "kid_trip_delta_min": kid, "mode_baseline": mb, "mode_plan": mp, "home_edges_vc_delta": vc}


class FakeLLM:
    """Scriptable LLM: `reply_fn(messages, model, json_mode) -> str | None`."""

    def __init__(self, reply_fn=None, available: bool = True) -> None:
        self.reply_fn = reply_fn
        self.available = available
        self.calls: list[dict] = []

    def chat(self, messages, *, model="fast", max_tokens=200, temperature=0.7, json_mode=False):
        self.calls.append({"messages": messages, "model": model, "json_mode": json_mode})
        if not self.available:
            return None
        return self.reply_fn(messages, model, json_mode) if self.reply_fn else None


def echo_reactions_llm() -> FakeLLM:
    """Writes a number-free reaction for every id in a batch request (or a fixed chat reply)."""

    def fn(messages, model, json_mode):
        user = messages[-1]["content"]
        try:
            data = json.loads(user)
        except ValueError:
            return "I think it could help families on my street, but I want to see it work first."
        key = "residents" if "residents" in data else ("speakers" if "speakers" in data else None)
        if key is None:
            return json.dumps({"backstories": [{"id": p["id"], "text": f"{p['first_name']} loves the canyon trails."}
                                               for p in data]})
        out_key = "reactions" if key == "residents" else "comments"
        return json.dumps({out_key: [{"id": r["id"], "text": f"As someone who values {r['values'][0]}, I "
                                      f"{'like' if r['approves'] else 'dislike'} this plan."} for r in data[key]]})

    return FakeLLM(fn)
