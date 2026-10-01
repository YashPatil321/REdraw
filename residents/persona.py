"""AI resident personas (spec 8.1).

500 persons are sampled from the synthetic population (persons/households
parquet), stratified so every school, income band and age band is represented.
Each persona gets a generated first name from a fixed built-in list (never a
real person), a block-level location label (never an address), three values
from `assumptions.residents.values_list`, and a short backstory written once by
the LLM and cached in `data/processed/personas.json` (template text when the
LLM is down; upgraded later when it comes back).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from pipeline.config import assumption, load_yaml, region
from residents.blocks import BlockLabeler, snap_to_grid
from residents.llm import ChatLLM, extract_json
from residents.names import FIRST_NAMES
from residents.textcheck import collect_numbers, safe_text

log = logging.getLogger(__name__)

CACHE_VERSION = 1
# Residents with a voice in the town hall: teens (students, new drivers) and adults.
PERSONA_MIN_AGE = 14
# Display bands for stratification (demographic buckets, not modeling numbers).
AGE_BANDS: tuple[tuple[int, int, str], ...] = (
    (14, 17, "14-17"), (18, 34, "18-34"), (35, 49, "35-49"), (50, 64, "50-64"), (65, 200, "65+"),
)
BACKSTORY_BATCH = 10


def age_band(age: int) -> str:
    for lo, hi, label in AGE_BANDS:
        if lo <= age <= hi:
            return label
    return AGE_BANDS[-1][2] if age > AGE_BANDS[-1][0] else AGE_BANDS[0][2]


class Kid(BaseModel):
    age: int
    school_id: str
    grade: int


class Persona(BaseModel):
    persona_id: int
    person_id: int
    household_id: int
    household_member_ids: list[int]
    first_name: str
    age: int
    age_band: str
    income_band: str
    household_size: int
    vehicles: int
    n_kids: int
    home_building_id: int  # internal only; never returned by the API (spec 8.1.3)
    x: float  # block-level (snapped) scene coordinates
    z: float
    block: str
    job_area: str
    is_worker: bool
    works_from_home: bool
    own_school_id: str | None
    school_ids: list[str]
    kids: list[Kid] = Field(default_factory=list)
    commute_mode: str | None = None
    values: list[str]
    backstory: str = ""
    backstory_source: Literal["llm", "template"] = "template"

    def public(self, school_names: dict[str, str] | None = None) -> dict[str, Any]:
        """Fields safe to show in the UI (no ids that link to buildings or addresses)."""
        names = school_names or {}
        return {
            "persona_id": self.persona_id,
            "first_name": self.first_name,
            "age": self.age,
            "block": self.block,
            "x": self.x,
            "z": self.z,
            "school_ids": self.school_ids,
            "schools": [names.get(s, s) for s in self.school_ids],
            "values": self.values,
            "household": {"size": self.household_size, "kids": self.n_kids,
                          "vehicles": self.vehicles, "income_band": self.income_band},
            "job_area": self.job_area,
            "commute_mode": self.commute_mode,
            "backstory": self.backstory,
        }

    def facts(self, school_names: dict[str, str] | None = None) -> dict[str, Any]:
        """Facts given to the LLM (all numbers here may be cited)."""
        names = school_names or {}
        return {
            "first_name": self.first_name,
            "age": self.age,
            "lives": self.block,
            "household_size": self.household_size,
            "kids_at_home": self.n_kids,
            "kids_schools": sorted({names.get(k.school_id, k.school_id) for k in self.kids}),
            "own_school": names.get(self.own_school_id, self.own_school_id) if self.own_school_id else None,
            "work": self.job_area,
            "values": self.values,
        }


# ----------------------------------------------------------------- data access
def school_names(data_dir: Path) -> dict[str, str]:
    path = data_dir / "schools_resolved.json"
    try:
        if path.exists():
            data = json.loads(path.read_text())
            return {s["id"]: s.get("name", s["id"]) for s in data.get("schools", [])}
        return {s["id"]: s.get("name", s["id"]) for s in load_yaml("schools.yaml").get("schools", [])}
    except Exception:  # noqa: BLE001
        return {}


def exit_labels(data_dir: Path) -> dict[str, str]:
    path = data_dir / "exits.json"
    if not path.exists():
        return {e["id"]: e.get("label", e["id"]) for e in region().get("exits", [])}
    return {e["id"]: e.get("label", e["id"]) for e in json.loads(path.read_text()).get("exits", [])}


def population_fingerprint(data_dir: Path) -> str:
    h = hashlib.sha256()
    for name in ("persons.parquet", "households.parquet"):
        h.update((data_dir / name).read_bytes())
    h.update(json.dumps([assumption("residents.persona_count"), assumption("residents.persona_seed"),
                         assumption("residents.values_list"), PERSONA_MIN_AGE, CACHE_VERSION]).encode())
    return h.hexdigest()[:16]


def load_population(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    persons_path = data_dir / "persons.parquet"
    hh_path = data_dir / "households.parquet"
    if not persons_path.exists() or not hh_path.exists():
        raise FileNotFoundError(
            f"Residents need {persons_path} and {hh_path}. Build world data first: "
            "`.venv/bin/python pipeline/build_all.py` (or `--synthetic` for the offline dev fixture)."
        )
    return pd.read_parquet(persons_path), pd.read_parquet(hh_path)


# ------------------------------------------------------------------- sampling
def _eligible(persons: pd.DataFrame, households: pd.DataFrame) -> pd.DataFrame:
    p = persons.copy()
    p["school_id"] = p["school_id"].fillna("").astype(str)
    kids = p[p["school_id"] != ""]
    hh_schools = kids.groupby("household_id")["school_id"].agg(lambda s: sorted(set(s)))
    hh = households.set_index("household_id")
    el = p[p["age"] >= PERSONA_MIN_AGE].copy()
    el = el.join(hh[["building_id", "x", "z", "size", "vehicles", "income_band", "n_kids"]],
                 on="household_id", how="inner")
    el["hh_schools"] = el["household_id"].map(hh_schools)
    el["hh_schools"] = el["hh_schools"].apply(lambda v: v if isinstance(v, list) else [])

    def _schools(row: Any) -> list[str]:
        own = [row.school_id] if row.school_id else []
        return sorted(set(own) | set(row.hh_schools))

    el["school_ids"] = [_schools(r) for r in el.itertuples()]
    el["primary_school"] = [r.school_id if r.school_id else (r.school_ids[0] if r.school_ids else "none")
                            for r in el.itertuples()]
    el["age_band"] = el["age"].astype(int).map(age_band)
    el["income_band"] = el["income_band"].astype(str)
    return el.sort_values("person_id").reset_index(drop=True)


def stratified_sample(el: pd.DataFrame, n: int, seed: int) -> list[int]:
    """Row indices into `el`: coverage of every school/income/age band first, then proportional."""
    rng = np.random.default_rng(seed)
    if len(el) <= n:
        return list(range(len(el)))
    chosen: list[int] = []
    chosen_set: set[int] = set()

    def take(mask: np.ndarray) -> None:
        idx = np.flatnonzero(mask)
        idx = np.array([i for i in idx if i not in chosen_set])
        if len(idx):
            i = int(rng.choice(idx))
            chosen.append(i)
            chosen_set.add(i)

    all_schools = sorted({s for lst in el["school_ids"] for s in lst})
    for school in all_schools:
        covered = any(school in el.at[i, "school_ids"] for i in chosen)
        if not covered:
            take(el["school_ids"].apply(lambda lst, s=school: s in lst).to_numpy())
    for col in ("income_band", "age_band"):
        for val in sorted(el[col].unique()):
            if not any(el.at[i, col] == val for i in chosen):
                take((el[col] == val).to_numpy())

    remaining = n - len(chosen)
    pool = el.drop(index=list(chosen_set))
    groups = pool.groupby(["primary_school", "income_band", "age_band"], sort=True).groups
    sizes = {k: len(v) for k, v in groups.items()}
    total = sum(sizes.values())
    quotas = {k: remaining * s / total for k, s in sizes.items()}
    alloc = {k: int(math.floor(q)) for k, q in quotas.items()}
    short = remaining - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), str(k)))[:short]:
        alloc[k] += 1
    for k in sorted(groups, key=str):
        cnt = min(alloc[k], sizes[k])
        if cnt <= 0:
            continue
        members = np.sort(np.asarray(groups[k]))
        picks = rng.choice(members, size=cnt, replace=False)
        chosen.extend(int(i) for i in sorted(picks))
    return chosen[:n]


def build_personas(persons: pd.DataFrame, households: pd.DataFrame, labeler: BlockLabeler,
                   schools: dict[str, str], exits: dict[str, str], n: int | None = None,
                   seed: int | None = None) -> list[Persona]:
    n = int(assumption("residents.persona_count")) if n is None else n
    seed = int(assumption("residents.persona_seed")) if seed is None else seed
    values_list = list(assumption("residents.values_list"))
    el = _eligible(persons, households)
    rows = stratified_sample(el, n, seed)
    rng = np.random.default_rng(seed + 1)
    p_by_hh = persons.groupby("household_id")
    out: list[Persona] = []
    for pid, i in enumerate(rows):
        r = el.loc[i]
        members = p_by_hh.get_group(r["household_id"])
        kids = [Kid(age=int(m.age), school_id=str(m.school_id), grade=int(m.grade))
                for m in members.itertuples() if str(m.school_id or "") and int(m.person_id) != int(r["person_id"])]
        x, z = float(r["x"]), float(r["z"])
        sx, sz = snap_to_grid(x, z)
        own_school = str(r["school_id"]) or None
        out.append(Persona(
            persona_id=pid,
            person_id=int(r["person_id"]),
            household_id=int(r["household_id"]),
            household_member_ids=[int(v) for v in members["person_id"]],
            first_name=str(rng.choice(FIRST_NAMES)),
            age=int(r["age"]),
            age_band=str(r["age_band"]),
            income_band=str(r["income_band"]),
            household_size=int(r["size"]),
            vehicles=int(r["vehicles"]),
            n_kids=int(r["n_kids"]),
            home_building_id=int(r["building_id"]),
            x=sx, z=sz,
            block=labeler.label(x, z),
            job_area=_job_area(r, schools, exits),
            is_worker=bool(r["is_worker"]),
            works_from_home=bool(r["works_from_home"]),
            own_school_id=own_school,
            school_ids=list(r["school_ids"]),
            kids=kids,
            commute_mode="works from home" if bool(r["works_from_home"]) else None,
            values=[str(v) for v in rng.choice(values_list, size=3, replace=False)],
        ))
    for p in out:
        p.backstory = template_backstory(p, schools)
    return out


def _job_area(r: Any, schools: dict[str, str], exits: dict[str, str]) -> str:
    if bool(r["is_worker"]):
        if bool(r["works_from_home"]):
            return "works from home"
        ex = str(r.get("work_exit") or "")
        if ex:
            return f"commutes out of the area via {exits.get(ex, ex)}"
        return "works inside the neighborhood"
    if str(r["school_id"] or ""):
        return f"student at {schools.get(str(r['school_id']), str(r['school_id']))}"
    return "not working outside the home"


def template_backstory(p: Persona, schools: dict[str, str]) -> str:
    kid_schools = sorted({schools.get(k.school_id, k.school_id) for k in p.kids})
    kids = f", with kids at {', '.join(kid_schools)}" if kid_schools else ""
    return (f"{p.first_name} is {p.age}, lives {p.block}{kids}, and {p.job_area}. "
            f"Cares most about {p.values[0]}, {p.values[1]} and {p.values[2].replace('_', ' ')}.")


# ------------------------------------------------------------- LLM backstories
_BACKSTORY_SYSTEM = (
    "You write short, plain backstories for fictional residents of a suburban San Diego "
    "neighborhood in a city planning game. Rules: exactly two sentences each; use only the "
    "facts given; do not invent surnames, addresses, house numbers, employers or any number "
    "not in the facts; keep it warm and ordinary. Reply with JSON only: "
    '{"backstories": [{"id": <id>, "text": "..."}]}'
)


def llm_backstories(personas: list[Persona], llm: ChatLLM, schools: dict[str, str],
                    max_workers: int = 4) -> dict[int, str]:
    if not llm.available:
        return {}
    batches = [personas[i:i + BACKSTORY_BATCH] for i in range(0, len(personas), BACKSTORY_BATCH)]

    def run(batch: list[Persona]) -> dict[int, str]:
        if not llm.available:
            return {}
        facts = {p.persona_id: p.facts(schools) for p in batch}
        msg = json.dumps([{"id": pid, **f} for pid, f in facts.items()])
        reply = llm.chat([{"role": "system", "content": _BACKSTORY_SYSTEM},
                          {"role": "user", "content": msg}],
                         model="fast", max_tokens=120 * len(batch), temperature=0.8, json_mode=True)
        data = extract_json(reply)
        items = data.get("backstories", []) if isinstance(data, dict) else []
        res: dict[int, str] = {}
        for it in items:
            try:
                pid = int(it["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if pid not in facts:
                continue
            nums, clocks = collect_numbers(facts[pid])
            text = safe_text(it.get("text"), nums, clocks)
            if text:
                res[pid] = text
        return res

    out: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(run, batches):
            out.update(res)
    return out


# ---------------------------------------------------------------------- cache
def load_cached(path: Path, fingerprint: str) -> list[Persona] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("version") != CACHE_VERSION or data.get("fingerprint") != fingerprint:
            return None
        return [Persona.model_validate(p) for p in data["personas"]]
    except Exception as e:  # noqa: BLE001
        log.warning("ignoring unreadable persona cache %s: %s", path, e)
        return None


def save_cache(path: Path, fingerprint: str, personas: list[Persona], synthetic: bool | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "version": CACHE_VERSION, "fingerprint": fingerprint, "synthetic": synthetic,
        "note": "AI resident personas: generated first names, block-level locations only.",
        "personas": [p.model_dump() for p in personas],
    }))
    tmp.replace(path)


def load_or_build(data_dir: Path, cache_path: Path, llm: ChatLLM | None,
                  on_built: Callable[[list[Persona]], None] | None = None) -> list[Persona]:
    fp = population_fingerprint(data_dir)
    cached = load_cached(cache_path, fp)
    if cached is not None:
        return cached
    persons, households = load_population(data_dir)
    schools = school_names(data_dir)
    personas = build_personas(persons, households, BlockLabeler.from_data_dir(data_dir), schools,
                              exit_labels(data_dir))
    if llm is not None:
        apply_backstories(personas, llm_backstories(personas, llm, schools))
    save_cache(cache_path, fp, personas, _synthetic(data_dir))
    if on_built:
        on_built(personas)
    return personas


def apply_backstories(personas: list[Persona], stories: dict[int, str]) -> int:
    n = 0
    for p in personas:
        if p.persona_id in stories:
            p.backstory = stories[p.persona_id]
            p.backstory_source = "llm"
            n += 1
    return n


def upgrade_backstories(data_dir: Path, cache_path: Path, personas: list[Persona], llm: ChatLLM) -> int:
    """Replace template backstories with LLM ones (once the endpoint is reachable)."""
    todo = [p for p in personas if p.backstory_source == "template"]
    if not todo or not llm.available:
        return 0
    n = apply_backstories(personas, llm_backstories(todo, llm, school_names(data_dir)))
    if n:
        save_cache(cache_path, population_fingerprint(data_dir), personas, _synthetic(data_dir))
    return n


def _synthetic(data_dir: Path) -> bool | None:
    meta = data_dir / "region_meta.json"
    try:
        return json.loads(meta.read_text()).get("synthetic") if meta.exists() else None
    except (OSError, ValueError):
        return None
