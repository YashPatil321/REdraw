"""Synthetic population + school resolution (spec 5.4, 5.5).

`synthesize_population` takes in-memory inputs (buildings, block group
distribution table, a work destination model, network nodes, exits, schools),
so real (ACS + LODES) and synthetic builds share it. Fixed seed (rule 14.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from pipeline.build_roads import FREEWAY_CLASSES, local_node_mask
from pipeline.common import log
from pipeline.config import assumption
from pipeline.geo import latlon_to_scene, scene_to_latlon

INCOME_BANDS = ["lt50k", "50_100k", "100_150k", "150_200k", "gt200k"]
KID_MIN_AGE, KID_MAX_AGE = 5, 17  # n_kids counts persons aged 5-18; students are 5..17 (grade = age - 5)


# ---------------------------------------------------------------------------
# Schools
# ---------------------------------------------------------------------------


def grades_from_name(name: str) -> list[int] | None:
    n = name.lower()
    if "high" in n:
        return [9, 12]
    if "middle" in n:
        return [6, 8]
    if "elementary" in n:
        return [0, 5]
    return None


def snap_entrance(x: float, z: float, nodes: pd.DataFrame, edges: pd.DataFrame, local: np.ndarray) -> tuple[int, int, float]:
    """Nearest local node with an incoming non-freeway edge; approach = longest such edge.

    Returns (node_id, approach_edge_idx, snap distance m).
    """
    loc_edges = edges[~edges["highway"].isin(FREEWAY_CLASSES)]
    has_in = nodes["node_id"].isin(set(loc_edges["v"])).to_numpy() & local
    cand = nodes[has_in]
    d = np.hypot(cand["x"].to_numpy() - x, cand["z"].to_numpy() - z)
    i = int(np.argmin(d))
    nid = int(cand["node_id"].iloc[i])
    inc = loc_edges[loc_edges["v"] == nid]
    approach = int(inc.sort_values(["length_m", "edge_idx"], ascending=[False, True])["edge_idx"].iloc[0])
    return nid, approach, float(d[i])


def resolve_schools(
    schools_cfg: list[dict[str, Any]],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    buildings: pd.DataFrame,
    osm_schools: gpd.GeoDataFrame | None = None,
) -> list[dict[str, Any]]:
    """schools.yaml merged with OSM amenity=school (spec 5.5); entrances snapped to the network."""
    local = local_node_mask(nodes, edges)
    defaults = {
        "bell_start": assumption("schools.defaults.bell_start"),
        "curb_spots": int(assumption("schools.defaults.curb_spots")),
        "unload_seconds": float(assumption("schools.defaults.unload_seconds")),
    }
    entries = [dict(s, source="schools.yaml") for s in schools_cfg]
    if osm_schools is not None and len(osm_schools):
        max_d = float(assumption("pipeline.school_snap_max_dist_m"))
        for row in osm_schools.to_crs("EPSG:4326").itertuples():
            name = str(getattr(row, "name", "") or "")
            c = row.geometry.representative_point()
            sx, sz = latlon_to_scene(c.y, c.x)
            near = False
            for s in entries:
                x2, z2 = latlon_to_scene(float(s["lat"]), float(s["lon"]))
                if math.hypot(sx - x2, sz - z2) <= max_d or (name and name.lower() == str(s["name"]).lower()):
                    near = True
                    break
            grades = grades_from_name(name) if name else None
            if near or grades is None:
                continue  # already listed, or a school we cannot place in a grade band (e.g. preschool)
            sid = "osm_" + "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
            entries.append(
                {
                    "id": sid,
                    "name": name,
                    "grades": grades,
                    "bell_start": defaults["bell_start"],
                    "lat": c.y,
                    "lon": c.x,
                    "verified": False,
                    "source": "osm",
                    "entrances": [{"id": "main_dropoff", "lat": c.y, "lon": c.x, "curb_spots": defaults["curb_spots"], "unload_seconds": defaults["unload_seconds"], "verified": False}],
                }
            )
    out = []
    for s in entries:
        x, z = latlon_to_scene(float(s["lat"]), float(s["lon"]))
        bids = sorted(int(b) for b in buildings.loc[buildings["school_id"] == s["id"], "id"]) if "school_id" in buildings else []
        ents = []
        for e in s.get("entrances") or [{"id": "main_dropoff", "lat": s["lat"], "lon": s["lon"]}]:
            ex, ez = latlon_to_scene(float(e["lat"]), float(e["lon"]))
            nid, appr, dist = snap_entrance(ex, ez, nodes, edges, local)
            ents.append(
                {
                    "id": e["id"],
                    "lat": float(e["lat"]),
                    "lon": float(e["lon"]),
                    "x": ex,
                    "z": ez,
                    "node_id": nid,
                    "approach_edge_idx": appr,
                    "snap_distance_m": dist,
                    "curb_spots": int(e.get("curb_spots", defaults["curb_spots"])),
                    "unload_seconds": float(e.get("unload_seconds", defaults["unload_seconds"])),
                    "verified": bool(e.get("verified", False)),
                }
            )
        out.append(
            {
                "id": s["id"],
                "name": s["name"],
                "grades": [int(g) for g in s["grades"]],
                "bell_start": str(s.get("bell_start", defaults["bell_start"])),
                "lat": float(s["lat"]),
                "lon": float(s["lon"]),
                "x": x,
                "z": z,
                "verified": bool(s.get("verified", False)),
                "source": s.get("source", "schools.yaml"),
                "building_ids": bids,
                "entrances": ents,
                "students": 0,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Distributions and work model
# ---------------------------------------------------------------------------


@dataclass
class BlockGroupDist:
    """Household distributions for one block group (ACS in real mode, assumptions in synthetic)."""

    geoid: str
    households: int
    size_shares: np.ndarray  # sizes 1..6
    income_shares: np.ndarray  # INCOME_BANDS
    vehicle_shares: np.ndarray  # 0..4
    share_with_kids: float
    kids_shares: np.ndarray  # 1..4 kids
    workers_per_household: float
    wfh_share: float


def fallback_dist(geoid: str, households: int) -> BlockGroupDist:
    ps = "population_synthesis"
    inc = assumption(f"{ps}.income_band_shares")
    return BlockGroupDist(
        geoid=geoid,
        households=int(households),
        size_shares=_norm(assumption(f"{ps}.household_size_shares")),
        income_shares=_norm([inc[b] for b in INCOME_BANDS]),
        vehicle_shares=_norm(assumption(f"{ps}.vehicles_shares")),
        share_with_kids=float(assumption("population.share_households_with_kids")),
        kids_shares=_norm(assumption(f"{ps}.kids_count_shares_given_kids")),
        workers_per_household=float(assumption("population.workers_per_household")),
        wfh_share=float(assumption("demand.work_from_home_share")),
    )


def _norm(v: Any) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    s = a.sum()
    return a / s if s > 0 else np.full(len(a), 1.0 / len(a))


@dataclass
class WorkModel:
    """Where workers from each home block group work.

    For origin `geoid` (or "" as the default): `internal_share` of jobs are at
    `internal_nodes` (weights `internal_weights`); the rest leave via exits with
    probabilities `exit_probs`.
    """

    internal_share: dict[str, float]
    internal_nodes: dict[str, np.ndarray]
    internal_weights: dict[str, np.ndarray]
    exit_probs: dict[str, dict[str, float]]
    source: str = "assumptions"
    notes: list[str] = field(default_factory=list)

    def for_origin(self, geoid: str) -> str:
        return geoid if geoid in self.internal_share else ""


def bearing_deg(x0: float, z0: float, x1: float, z1: float) -> float:
    """Compass bearing (0 = north, 90 = east) from scene point 0 to scene point 1."""
    return (math.degrees(math.atan2(x1 - x0, -(z1 - z0))) + 360.0) % 360.0


def exit_for_bearing(b: float, exits: list[dict[str, Any]]) -> str:
    def diff(e: dict[str, Any]) -> float:
        d = abs((b - float(e["bearing_deg"])) % 360.0)
        return min(d, 360.0 - d)

    return str(min(exits, key=diff)["id"])


# ---------------------------------------------------------------------------
# Population synthesis
# ---------------------------------------------------------------------------


def household_capacity(btype: str, area_m2: float, levels: Any) -> int:
    if btype == "house":
        return 1
    if btype == "apartments":
        lv = int(levels) if levels is not None and not (isinstance(levels, float) and math.isnan(levels)) else int(assumption("pipeline.apartment_default_levels"))
        cap = int(round(area_m2 * lv / float(assumption("pipeline.apartment_m2_per_household"))))
        hi = 3 * int(assumption("population.persons_per_unit_apartment_building"))
        return int(np.clip(cap, 4, hi))
    return 0


def allocate_households(rng: np.random.Generator, caps: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    """Assign n households to buildings (indices). Fill unit slots without replacement;
    if demand exceeds total capacity, extra households go by footprint area weight."""
    slots = np.repeat(np.arange(len(caps)), caps)
    if len(slots) == 0:
        return np.zeros(0, dtype=np.int64)
    if n <= len(slots):
        pick = rng.choice(len(slots), size=n, replace=False)
        return np.sort(slots[pick])
    extra = n - len(slots)
    p = weights / weights.sum()
    more = rng.choice(len(caps), size=extra, replace=True, p=p)
    return np.sort(np.concatenate([slots, more]))


def synthesize_population(
    buildings: pd.DataFrame,
    dists: list[BlockGroupDist],
    work: WorkModel,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    exits: list[dict[str, Any]],
    schools: list[dict[str, Any]],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    local = local_node_mask(nodes, edges)
    lnodes = nodes[local]
    tree = cKDTree(lnodes[["x", "z"]].to_numpy())
    node_ids = lnodes["node_id"].to_numpy()
    node_pos = nodes.set_index("node_id")[["x", "z"]]
    exit_node = {e["id"]: int(e["node_id"]) for e in exits}

    res = buildings[buildings["type"].isin(["house", "apartments"])].copy()
    res["cap"] = [household_capacity(t, a, lv) for t, a, lv in zip(res["type"], res["area_m2"], res["levels"], strict=True)]

    hh_rows: list[dict[str, Any]] = []
    for dist in dists:
        if dist.households <= 0:
            continue
        sub = res if dist.geoid == "" and len(dists) == 1 else res[res["block_group"] == dist.geoid]
        if len(sub) == 0:
            log(f"WARNING block group {dist.geoid}: {dist.households} households but no residential buildings in the bbox; skipped")
            continue
        bidx = allocate_households(rng, sub["cap"].to_numpy(), sub["area_m2"].to_numpy(), dist.households)
        n = len(bidx)
        size = rng.choice(np.arange(1, 7), size=n, p=dist.size_shares)
        p_size2 = float(dist.size_shares[1:].sum())
        p_kids = min(1.0, dist.share_with_kids / max(p_size2, 1e-6))
        has_kids = (size >= 2) & (rng.random(n) < p_kids)
        nk = rng.choice(np.arange(1, 5), size=n, p=dist.kids_shares)
        n_kids = np.where(has_kids, np.minimum(nk, size - 1), 0)
        veh = rng.choice(np.arange(0, 5), size=n, p=dist.vehicle_shares)
        inc = rng.choice(len(INCOME_BANDS), size=n, p=dist.income_shares)
        brow = sub.iloc[bidx]
        for k in range(n):
            hh_rows.append(
                {
                    "building_id": int(brow["id"].iloc[k]),
                    "block_group": dist.geoid,
                    "x": float(brow["centroid_x"].iloc[k]),
                    "z": float(brow["centroid_z"].iloc[k]),
                    "size": int(size[k]),
                    "vehicles": int(veh[k]),
                    "income_band": INCOME_BANDS[int(inc[k])],
                    "n_kids": int(n_kids[k]),
                    "aux_wph": dist.workers_per_household,
                    "aux_wfh": dist.wfh_share,
                }
            )
    hh = pd.DataFrame(hh_rows)
    hh.insert(0, "household_id", np.arange(1, len(hh) + 1, dtype=np.int64))
    _, ni = tree.query(hh[["x", "z"]].to_numpy(), k=1)
    hh["home_node"] = node_ids[ni].astype(np.int64)

    # Persons
    senior_share = float(assumption("population_synthesis.senior_share_of_adults"))
    lic_adult = float(assumption("population_synthesis.adult_license_rate"))
    lic_teen = float(assumption("population_synthesis.teen_license_rate"))
    min_drive = int(assumption("demand.min_driving_age"))
    persons: list[dict[str, Any]] = []
    for h in hh.itertuples(index=False):
        n_ad = h.size - h.n_kids
        for _ in range(n_ad):
            if h.n_kids > 0:
                age = int(np.clip(rng.normal(43, 6), 25, 64))
            elif rng.random() < senior_share:
                age = int(rng.integers(65, 90))
            else:
                age = int(rng.integers(19, 65))
            persons.append({"household_id": h.household_id, "age": age, "_bg": h.block_group, "_hx": h.x, "_hz": h.z, "_wph": h.aux_wph, "_wfh": h.aux_wfh})
        for _ in range(h.n_kids):
            persons.append({"household_id": h.household_id, "age": int(rng.integers(KID_MIN_AGE, KID_MAX_AGE + 1)), "_bg": h.block_group, "_hx": h.x, "_hz": h.z, "_wph": h.aux_wph, "_wfh": h.aux_wfh})
    pp = pd.DataFrame(persons)
    pp.insert(0, "person_id", np.arange(1, len(pp) + 1, dtype=np.int64))
    age = pp["age"].to_numpy()
    working_age = (age >= 19) & (age <= 64)
    # employment probability so expected workers per household matches the target (per block group)
    p_emp = np.zeros(len(pp))
    for bg, grp in pp.groupby("_bg"):
        idx = grp.index.to_numpy()
        n_hh = hh.loc[hh["block_group"] == bg].shape[0]
        wa = working_age[idx].sum()
        target = float(grp["_wph"].iloc[0]) * n_hh
        p_emp[idx] = min(0.97, target / max(wa, 1))
    is_worker = working_age & (rng.random(len(pp)) < p_emp)
    wfh = is_worker & (rng.random(len(pp)) < pp["_wfh"].to_numpy())
    lic = np.where(age >= 19, rng.random(len(pp)) < lic_adult, np.where(age >= min_drive, rng.random(len(pp)) < lic_teen, False))

    work_node = np.full(len(pp), -1, dtype=np.int64)
    work_exit = np.array([""] * len(pp), dtype=object)
    wx = np.full(len(pp), np.nan)
    wz = np.full(len(pp), np.nan)
    commuters = np.where(is_worker)[0]
    bgs = pp["_bg"].to_numpy()
    for i in commuters:
        key = work.for_origin(str(bgs[i]))
        if rng.random() < work.internal_share[key] and len(work.internal_nodes[key]):
            w = work.internal_weights[key]
            nid = int(work.internal_nodes[key][rng.choice(len(w), p=w)])
            work_node[i] = nid
        else:
            probs = work.exit_probs[key]
            ids = list(probs)
            eid = ids[int(rng.choice(len(ids), p=_norm([probs[k] for k in ids])))]
            work_node[i] = exit_node[eid]
            work_exit[i] = eid
        wx[i], wz[i] = node_pos.loc[work_node[i], ["x", "z"]]

    # Students -> nearest school of the right grade (spec 5.4.4; no attendance boundaries in v1)
    grade = np.where((age >= KID_MIN_AGE) & (age <= KID_MAX_AGE), age - KID_MIN_AGE, -1)
    school_id = np.array([""] * len(pp), dtype=object)
    sch_xy = np.array([[s["x"], s["z"]] for s in schools]) if schools else np.zeros((0, 2))
    hx, hz = pp["_hx"].to_numpy(), pp["_hz"].to_numpy()
    for g in range(0, 13):
        sel = np.where(grade == g)[0]
        ok = [k for k, s in enumerate(schools) if s["grades"][0] <= g <= s["grades"][1]]
        if len(sel) == 0 or not ok:
            continue
        d = np.hypot(hx[sel, None] - sch_xy[ok, 0][None, :], hz[sel, None] - sch_xy[ok, 1][None, :])
        pick = np.asarray(ok)[np.argmin(d, axis=1)]
        school_id[sel] = [schools[k]["id"] for k in pick]
    grade = np.where(school_id != "", grade, -1)

    persons_df = pd.DataFrame(
        {
            "person_id": pp["person_id"].astype(np.int64),
            "household_id": pp["household_id"].astype(np.int64),
            "age": pp["age"].astype(np.int16),
            "is_worker": is_worker.astype(bool),
            "works_from_home": wfh.astype(bool),
            "work_node": work_node,
            "work_exit": work_exit.astype(str),
            "work_x": wx,
            "work_z": wz,
            "school_id": school_id.astype(str),
            "grade": grade.astype(np.int16),
            "has_license": lic.astype(bool),
            "baseline_mode_hint": [""] * len(pp),
        }
    )
    hh_df = hh[["household_id", "building_id", "block_group", "x", "z", "home_node", "size", "vehicles", "income_band", "n_kids"]].copy()
    hh_df = hh_df.astype({"household_id": "int64", "building_id": "int64", "home_node": "int64", "size": "int16", "vehicles": "int16", "n_kids": "int16"})
    hh_df["block_group"] = hh_df["block_group"].astype(str)
    hh_df["income_band"] = hh_df["income_band"].astype(str)
    return hh_df, persons_df


def students_by_school(persons: pd.DataFrame, schools: list[dict[str, Any]]) -> dict[str, int]:
    counts = persons.loc[persons["school_id"] != "", "school_id"].value_counts().to_dict()
    return {s["id"]: int(counts.get(s["id"], 0)) for s in schools}


def write_population(hh: pd.DataFrame, persons: pd.DataFrame, processed: Path) -> None:
    pq.write_table(pa.Table.from_pandas(hh, preserve_index=False), processed / "households.parquet")
    pq.write_table(pa.Table.from_pandas(persons, preserve_index=False), processed / "persons.parquet")


# ---------------------------------------------------------------------------
# Real mode: ACS + LODES
# ---------------------------------------------------------------------------

ACS_SIZE_COLS = {
    1: ["B11016_010E"],
    2: ["B11016_003E", "B11016_011E"],
    3: ["B11016_004E", "B11016_012E"],
    4: ["B11016_005E", "B11016_013E"],
    5: ["B11016_006E", "B11016_014E"],
    6: ["B11016_007E", "B11016_008E", "B11016_015E", "B11016_016E"],
}
ACS_VEH_COLS = {
    0: ["B25044_003E", "B25044_010E"],
    1: ["B25044_004E", "B25044_011E"],
    2: ["B25044_005E", "B25044_012E"],
    3: ["B25044_006E", "B25044_013E"],
    4: ["B25044_007E", "B25044_008E", "B25044_014E", "B25044_015E"],
}
ACS_INC_COLS = {
    "lt50k": [f"B19001_{i:03d}E" for i in range(2, 11)],
    "50_100k": ["B19001_011E", "B19001_012E", "B19001_013E"],
    "100_150k": ["B19001_014E", "B19001_015E"],
    "150_200k": ["B19001_016E"],
    "gt200k": ["B19001_017E"],
}
ACS_OTHER_COLS = ["B11016_001E", "B11005_001E", "B11005_002E", "B08301_001E", "B08301_021E"]


def acs_variables() -> list[str]:
    cols: list[str] = []
    for d in (ACS_SIZE_COLS, ACS_VEH_COLS, ACS_INC_COLS):
        for v in d.values():
            cols += v
    return sorted(set(cols + ACS_OTHER_COLS))


def dists_from_acs(acs: pd.DataFrame, bg_inside_fraction: dict[str, float]) -> list[BlockGroupDist]:
    """ACS block group rows (GEOID + variables) -> distributions; suppressed cells fall back."""
    out = []
    for r in acs.itertuples(index=False):
        d = r._asdict()
        geoid = str(d["GEOID"])
        frac = bg_inside_fraction.get(geoid, 0.0)
        if frac <= 0:
            continue

        def s(cols: list[str], dd: dict[str, Any] = d) -> float:
            return float(sum(max(float(dd.get(c) or 0), 0.0) for c in cols))

        hh_total = s(["B11016_001E"])
        fb = fallback_dist(geoid, int(round(hh_total * frac)))
        size = np.array([s(ACS_SIZE_COLS[k]) for k in range(1, 7)])
        veh = np.array([s(ACS_VEH_COLS[k]) for k in range(0, 5)])
        inc = np.array([s(ACS_INC_COLS[b]) for b in INCOME_BANDS])
        kids_total = s(["B11005_001E"])
        workers = s(["B08301_001E"])
        out.append(
            BlockGroupDist(
                geoid=geoid,
                households=fb.households,
                size_shares=_norm(size) if size.sum() > 0 else fb.size_shares,
                income_shares=_norm(inc) if inc.sum() > 0 else fb.income_shares,
                vehicle_shares=_norm(veh) if veh.sum() > 0 else fb.vehicle_shares,
                share_with_kids=s(["B11005_002E"]) / kids_total if kids_total > 0 else fb.share_with_kids,
                kids_shares=fb.kids_shares,
                workers_per_household=workers / hh_total if hh_total > 0 else fb.workers_per_household,
                wfh_share=s(["B08301_021E"]) / workers if workers > 0 else fb.wfh_share,
            )
        )
    return out


def work_model_from_lodes(
    od: pd.DataFrame,
    xwalk: pd.DataFrame,
    home_bgs: set[str],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    exits: list[dict[str, Any]],
    bbox: dict[str, float],
) -> WorkModel:
    """LODES OD (h_geocode, w_geocode, S000) -> per home block group work destinations (spec 5.4.3)."""
    od = od.copy()
    od["h_bg"] = od["h_geocode"].astype(str).str.zfill(15).str[:12]
    od = od[od["h_bg"].isin(home_bgs)]
    xw = xwalk[["tabblk2020", "blklatdd", "blklondd"]].copy()
    xw["tabblk2020"] = xw["tabblk2020"].astype(str).str.zfill(15)
    od["w_geocode"] = od["w_geocode"].astype(str).str.zfill(15)
    od = od.merge(xw, left_on="w_geocode", right_on="tabblk2020", how="left").dropna(subset=["blklatdd"])
    inside = (od["blklatdd"].between(bbox["south"], bbox["north"])) & (od["blklondd"].between(bbox["west"], bbox["east"]))
    local = local_node_mask(nodes, edges)
    ln = nodes[local]
    tree = cKDTree(ln[["x", "z"]].to_numpy())
    wx, wz = zip(*[latlon_to_scene(a, b) for a, b in zip(od["blklatdd"], od["blklondd"], strict=True)], strict=True) if len(od) else ((), ())
    od["wx"], od["wz"] = np.asarray(wx), np.asarray(wz)
    model = WorkModel({}, {}, {}, {}, source="LODES")
    for bg, g in od.groupby("h_bg"):
        gi = g[inside.loc[g.index]]
        ge = g[~inside.loc[g.index]]
        total = float(g["S000"].sum())
        if total <= 0:
            continue
        model.internal_share[bg] = float(gi["S000"].sum()) / total
        if len(gi):
            _, k = tree.query(gi[["wx", "wz"]].to_numpy(), k=1)
            agg = pd.Series(gi["S000"].to_numpy(), index=ln["node_id"].to_numpy()[k]).groupby(level=0).sum()
            model.internal_nodes[bg] = agg.index.to_numpy(dtype=np.int64)
            model.internal_weights[bg] = _norm(agg.to_numpy())
        else:
            model.internal_nodes[bg] = np.zeros(0, dtype=np.int64)
            model.internal_weights[bg] = np.zeros(0)
        cx, cz = 0.0, 0.0
        probs: dict[str, float] = {e["id"]: 0.0 for e in exits}
        for r in ge.itertuples(index=False):
            probs[exit_for_bearing(bearing_deg(cx, cz, r.wx, r.wz), exits)] += float(r.S000)
        tot_e = sum(probs.values())
        model.exit_probs[bg] = {k: v / tot_e for k, v in probs.items()} if tot_e > 0 else dict(assumption("population_synthesis.external_exit_shares"))
    # default ("") = job-weighted average of all origins
    all_int = od[inside]
    model.internal_share[""] = float(all_int["S000"].sum()) / max(float(od["S000"].sum()), 1.0)
    if len(all_int):
        _, k = tree.query(all_int[["wx", "wz"]].to_numpy(), k=1)
        agg = pd.Series(all_int["S000"].to_numpy(), index=ln["node_id"].to_numpy()[k]).groupby(level=0).sum()
        model.internal_nodes[""] = agg.index.to_numpy(dtype=np.int64)
        model.internal_weights[""] = _norm(agg.to_numpy())
    else:
        model.internal_nodes[""] = np.zeros(0, dtype=np.int64)
        model.internal_weights[""] = np.zeros(0)
    model.exit_probs[""] = dict(assumption("population_synthesis.external_exit_shares"))
    return model


def latlon_of(x: float, z: float) -> tuple[float, float]:
    return scene_to_latlon(x, z)
