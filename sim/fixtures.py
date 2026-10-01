"""Self-contained synthetic world in the data_contract formats (tests and benchmarks).

NOT real geography. ``build_fixture_world`` writes a grid network with
arterials, a freeway column, exits, schools with entrances and a synthetic
population, so sim tests never depend on the pipeline. The same generator at
``scale="large"`` gives a world of the target size (~45k persons, ~25k edges)
for performance work::

    .venv/bin/python -m sim.fixtures --out /tmp/redraw_fixture --scale large
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import load_yaml, region
from pipeline.geo import _from_utm, scene_origin
from sim.assumptions import A

SCALES = {
    "tiny": {"grid": 10, "spacing": 300.0, "households": 450},
    "small": {"grid": 16, "spacing": 250.0, "households": 1500},
    "large": {"grid": 80, "spacing": 120.0, "households": 14500},
}

# fixture school placement as (row, col) fractions of the grid
_SCHOOL_SPOTS = {
    "del_norte_hs": (0.35, 0.70),
    "oak_valley_ms": (0.45, 0.55),
    "design39": (0.25, 0.25),
    "stone_ranch_es": (0.60, 0.45),
    "del_sur_es": (0.70, 0.20),
}


def _latlon(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o = scene_origin()
    lon, lat = _from_utm().transform(np.asarray(x) + o.easting, o.northing - np.asarray(z))
    return np.asarray(lat), np.asarray(lon)


def _cls(name: str, key: str) -> float:
    return float(A(f"roads.defaults_by_class.{name}.{key}"))


def build_fixture_world(out_dir: Path | str, scale: str = "tiny", seed: int = 1, schools: list[str] | None = None) -> Path:
    cfg = SCALES[scale]
    G, sp_, n_hh = int(cfg["grid"]), float(cfg["spacing"]), int(cfg["households"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    art_every = max(G // 5, 3)

    # ---- nodes -----------------------------------------------------------------
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    nid = ii * G + jj + 1000
    x = (jj - (G - 1) / 2.0) * sp_
    z = (ii - (G - 1) / 2.0) * sp_
    lat, lon = _latlon(x, z)
    is_art_row = ii % art_every == 0
    is_art_col = (jj % art_every == 0)
    fwy_col = G - 1
    signal = is_art_row & is_art_col & (jj != fwy_col)

    # ---- edges -----------------------------------------------------------------
    us, vs, hw = [], [], []

    def add(a: int, b: int, cls: str) -> None:
        us.extend([a, b])
        vs.extend([b, a])
        hw.extend([cls, cls])

    for i in range(G):
        for j in range(G):
            a = i * G + j
            if j + 1 < G:
                cls = "primary" if i % art_every == 0 else "residential"
                add(a, a + 1, cls)
            if i + 1 < G:
                if j == fwy_col:
                    cls = "motorway"
                elif j % art_every == 0:
                    cls = "secondary"
                else:
                    cls = "residential"
                add(a, a + G, cls)
    us_a, vs_a = np.array(us), np.array(vs)
    hw_a = np.array(hw, dtype=object)
    length = np.hypot(x[us_a] - x[vs_a], z[us_a] - z[vs_a]).astype(np.float32)
    lanes = np.array([int(_cls(h, "lanes")) for h in hw_a], dtype=np.int16)
    speed = np.array([_cls(h, "maxspeed_kph") for h in hw_a], dtype=np.float32)
    capl = np.array([_cls(h, "capacity_vphpl") for h in hw_a], dtype=np.float32)
    geom = [np.array([x[a], 0.0, z[a], x[b], 0.0, z[b]], dtype=np.float32).tolist() for a, b in zip(us_a, vs_a, strict=True)]
    names = np.where(hw_a == "primary", "Fixture Parkway", np.where(hw_a == "secondary", "Fixture Road", np.where(hw_a == "motorway", "Fixture Freeway", "")))
    E = len(us_a)
    edges = pd.DataFrame({
        "edge_idx": np.arange(E, dtype=np.int32), "u": nid[us_a].astype(np.int64), "v": nid[vs_a].astype(np.int64),
        "osmid": "", "name": names, "ref": np.where(hw_a == "motorway", "I 15", ""), "highway": hw_a.astype(str),
        "lanes": lanes, "maxspeed_kph": speed, "oneway": False, "length_m": length,
        "capacity_vph": (lanes * capl).astype(np.float32), "free_flow_s": (length / (speed / 3.6)).astype(np.float32),
        "label": np.where(hw_a == "primary", "Fixture Parkway", ""),
    })
    tbl = pa.Table.from_pandas(edges, preserve_index=False)
    tbl = tbl.append_column("geometry", pa.array(geom, type=pa.list_(pa.float32())))
    pq.write_table(tbl, out / "network_edges.parquet")

    # exits: freeway north/south ends, west and east arterial ends
    mid_art = (G // 2 // art_every) * art_every
    exit_defs = [
        ("i15_north", "I 15 north", 0 * G + fwy_col, 0),
        ("i15_south", "I 15 south", (G - 1) * G + fwy_col, 180),
        ("sr56_west", "SR 56 west", mid_art * G + 0, 260),
        ("cdn_east", "Camino Del Norte east", art_every * G + fwy_col, 90),
    ]
    bexit = np.full(G * G, "", dtype=object)
    for eid, _, n, _ in exit_defs:
        bexit[n] = eid
    nodes = pd.DataFrame({
        "node_id": nid.astype(np.int64), "x": x, "z": z, "y": np.zeros(G * G, np.float32), "lat": lat, "lon": lon,
        "signalized": signal, "boundary_exit": bexit.astype(str),
    })
    nodes.to_parquet(out / "network_nodes.parquet", index=False)
    exits = {"exits": [{"id": e, "label": lab, "node_id": int(nid[n]), "x": float(x[n]), "z": float(z[n]), "bearing_deg": b} for e, lab, n, b in exit_defs]}
    (out / "exits.json").write_text(json.dumps(exits, indent=1))

    # ---- schools ---------------------------------------------------------------
    cfg_schools = {s["id"]: s for s in load_yaml("schools.yaml")["schools"]}
    use = schools or list(_SCHOOL_SPOTS)
    sch_out = []
    for sid in use:
        s = cfg_schools[sid]
        fr, fc = _SCHOOL_SPOTS[sid]
        r = int(round(fr * (G - 1)))
        c = int(round(fc * (G - 1)))
        if r % art_every == 0:
            r += 1
        if c % art_every == 0:
            c += 1
        c = min(c, G - 2)
        ent_node = r * G + c
        # approach edge: from the nearest arterial row node towards the entrance (along the row)
        prev = r * G + (c - 1)
        cand = np.nonzero((us_a == prev) & (vs_a == ent_node))[0]
        ae = int(cand[0])
        e0 = s["entrances"][0]
        sch_out.append({
            "id": sid, "name": s["name"], "grades": s["grades"], "bell_start": s["bell_start"],
            "lat": float(lat[ent_node]), "lon": float(lon[ent_node]), "x": float(x[ent_node]) + 40.0, "z": float(z[ent_node]) + 40.0,
            "verified": False, "source": "fixture", "building_ids": [],
            "entrances": [{"id": e0["id"], "lat": float(lat[ent_node]), "lon": float(lon[ent_node]), "x": float(x[ent_node]), "z": float(z[ent_node]),
                           "node_id": int(nid[ent_node]), "approach_edge_idx": ae, "curb_spots": e0["curb_spots"],
                           "unload_seconds": e0["unload_seconds"], "verified": False}],
            "students": 0,
        })

    # ---- population ------------------------------------------------------------
    res_nodes = np.nonzero(~(is_art_row | is_art_col) & (jj != fwy_col))[0]
    job_nodes = rng.choice(np.nonzero(is_art_row & (jj != fwy_col))[0], size=max(5, G // 2), replace=False)
    home = rng.choice(res_nodes, size=n_hh)
    hx = x[home] + rng.uniform(-20, 20, n_hh)
    hz = z[home] + rng.uniform(-20, 20, n_hh)
    size = np.clip(rng.poisson(2.1, n_hh) + 1, 1, 7)
    has_kids = rng.random(n_hh) < float(A("population.share_households_with_kids"))
    n_kids = np.where(has_kids, np.clip(rng.poisson(0.8, n_hh) + 1, 1, 4), 0)
    n_adults = np.maximum(size - n_kids, 1)
    size = n_adults + n_kids
    vehicles = np.clip(rng.poisson(2.0, n_hh), 0, 4)
    hh = pd.DataFrame({
        "household_id": np.arange(1, n_hh + 1, dtype=np.int64), "building_id": np.arange(1, n_hh + 1, dtype=np.int64),
        "block_group": "", "x": hx, "z": hz, "home_node": nid[home].astype(np.int64), "size": size.astype(np.int16),
        "vehicles": vehicles.astype(np.int16), "income_band": "100_150k", "n_kids": n_kids.astype(np.int16),
    })
    hh.to_parquet(out / "households.parquet", index=False)

    by_level = {"es": [], "ms": [], "hs": [], "k8": []}
    for s in sch_out:
        g0, g1 = s["grades"]
        if g0 >= 9:
            by_level["hs"].append(s["id"])
        elif g0 >= 6:
            by_level["ms"].append(s["id"])
        elif g1 >= 8:
            by_level["k8"].append(s["id"])
        else:
            by_level["es"].append(s["id"])
    ext_shares = {"i15_south": 0.42, "sr56_west": 0.33, "i15_north": 0.15, "cdn_east": 0.10}
    ext_ids, ext_p = list(ext_shares), np.array(list(ext_shares.values()))
    exit_node = {e: n for e, _, n, _ in exit_defs}
    rows = []
    pid = 1
    for h in range(n_hh):
        for a in range(int(n_adults[h])):
            age = int(rng.integers(28, 60)) if a < 2 else int(rng.integers(19, 80))
            worker = rng.random() < (0.78 if age < 65 else 0.2)
            wfh = worker and rng.random() < float(A("demand.work_from_home_share"))
            wnode, wexit, wx, wz = -1, "", np.nan, np.nan
            if worker:
                if rng.random() < float(A("demand.external_job_share")):
                    wexit = str(rng.choice(ext_ids, p=ext_p))
                    n = exit_node[wexit]
                else:
                    n = int(rng.choice(job_nodes))
                wnode, wx, wz = int(nid[n]), float(x[n]), float(z[n])
            rows.append((pid, h + 1, age, worker, wfh, wnode, wexit, wx, wz, "", -1, rng.random() < 0.93))
            pid += 1
        for _ in range(int(n_kids[h])):
            age = int(rng.integers(5, 19))
            grade = int(np.clip(age - 5, 0, 12))
            if grade >= 9:
                pool = by_level["hs"]
            elif grade >= 6:
                pool = by_level["ms"] + by_level["k8"]
            else:
                pool = by_level["es"] + by_level["k8"]
            if not pool:
                pool = by_level["k8"] or by_level["hs"] or [""]
            sid = str(rng.choice(pool))
            rows.append((pid, h + 1, age, False, False, -1, "", np.nan, np.nan, sid, grade, age >= 16 and rng.random() < 0.45))
            pid += 1
    persons = pd.DataFrame(rows, columns=["person_id", "household_id", "age", "is_worker", "works_from_home", "work_node", "work_exit",
                                          "work_x", "work_z", "school_id", "grade", "has_license"])
    persons["age"] = persons["age"].astype(np.int16)
    persons["grade"] = persons["grade"].astype(np.int16)
    persons["baseline_mode_hint"] = ""
    persons.to_parquet(out / "persons.parquet", index=False)
    for s in sch_out:
        s["students"] = int((persons["school_id"] == s["id"]).sum())
    (out / "schools_resolved.json").write_text(json.dumps({"schools": sch_out}, indent=1))

    reg = region()
    meta = {
        "contract_version": 1, "name": "fixture", "display_name": "Fixture world (synthetic, NOT real geography)",
        "synthetic": True, "generated_at": "2026-10-01T00:00:00Z", "projection": "EPSG:32611", "bbox": reg["bbox"],
        "origin": {"lat": scene_origin().lat, "lon": scene_origin().lon, "easting": scene_origin().easting, "northing": scene_origin().northing},
        "extent_scene": {"min_x": float(x.min()), "max_x": float(x.max()), "min_z": float(z.min()), "max_z": float(z.max())},
        "counts": {"buildings": n_hh, "road_edges": E, "road_nodes": G * G, "households": n_hh, "persons": len(persons),
                   "students_by_school": {s["id"]: s["students"] for s in sch_out}},
        "sources": [{"name": "sim.fixtures", "license": "Apache-2.0", "retrieved": "2026-10-01"}],
    }
    (out / "region_meta.json").write_text(json.dumps(meta, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", default="large", choices=list(SCALES))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    p = build_fixture_world(a.out, a.scale, a.seed)
    print(f"fixture world written to {p}")


if __name__ == "__main__":
    main()
