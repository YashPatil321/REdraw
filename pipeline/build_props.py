"""Instance placements for props (trees, shrubs, street lamps, parked cars) in the open-data view.

    .venv/bin/python pipeline/build_props.py            # needs data/processed/ (real or synthetic)

Inputs (docs/data_contract.md): data/processed/region_meta.json, buildings.geojson,
network_edges.parquet, network_nodes.parquet, terrain_meta.json + the 16-bit
heightmap (assets/terrain/heightmap.png); client/public/assets/props/props_manifest.json
(built by blender/build_all_assets.py); pipeline/hero_overrides/hero_overrides.json and
<hero>_trees.json (campus trees, lamps and parked cars laid out from real site data).

Outputs (format documented in blender/README.md):
    client/public/assets/props/placements.json   header: props table, record layout, spatial cells
    client/public/assets/props/placements.bin    little-endian float32 records (x, y, z, rot_y, scale, prop_index)

Rules (all densities in data/config/assumptions.yaml -> props.*):
- street trees along residential and arterial curbs, offset from the centerline by
  lanes * lane_width (+ parking / bike lane) + parkway, both sides, fixed spacing with jitter
- yard trees, shrubs and ornamental grass around houses (not inside footprints or roads)
- palms / trees around commercial and apartment perimeters
- oaks / eucalyptus and scrub on steep undeveloped slopes (canyons), away from buildings
- street lamps at intersections (two at signals) and mid-block along streets
- hero campuses: skip generic props inside each hero radius; add the hero's own trees,
  lot lights and parked cars from <hero>_trees.json
Fixed seed (props.seed): identical inputs give identical bytes.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import assets_dir, assumption, processed_dir  # noqa: E402

FORMAT_VERSION = 1
RECORD_FIELDS = ["x", "y", "z", "rot_y", "scale", "prop_index"]
RECORD_DTYPE = np.dtype("<f4")
STRIDE = 4 * len(RECORD_FIELDS)
CELL_M = 500.0
HERO_DIR = Path(__file__).resolve().parent / "hero_overrides"

ARTERIAL = {"primary", "secondary", "tertiary", "primary_link", "secondary_link", "tertiary_link", "trunk"}
RESIDENTIAL = {"residential", "living_street", "unclassified"}
NO_TREES = {"motorway", "motorway_link", "trunk_link"}


def log(msg: str) -> None:
    print(f"[build_props] {msg}", flush=True)


class Fail(SystemExit):
    pass


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


class Terrain:
    """Bilinear sampler for the 16-bit heightmap (docs/data_contract.md terrain_meta.json)."""

    def __init__(self, meta: dict[str, Any], png: Path):
        from PIL import Image

        im = Image.open(png)
        a = np.asarray(im)
        if a.dtype != np.uint16:
            a = np.asarray(im.convert("I")).astype(np.float64)
        self.h = a.astype(np.float64) * float(meta["elev_scale"]) + float(meta["elev_offset"])
        self.min_x, self.max_x = float(meta["min_x"]), float(meta["max_x"])
        self.min_z, self.max_z = float(meta["min_z"]), float(meta["max_z"])
        self.rows, self.cols = self.h.shape
        self.dx = (self.max_x - self.min_x) / (self.cols - 1)
        self.dz = (self.max_z - self.min_z) / (self.rows - 1)

    def sample(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        c = np.clip((np.asarray(x) - self.min_x) / self.dx, 0, self.cols - 1.001)
        r = np.clip((np.asarray(z) - self.min_z) / self.dz, 0, self.rows - 1.001)
        c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
        fc, fr = c - c0, r - r0
        h = self.h
        return (h[r0, c0] * (1 - fc) * (1 - fr) + h[r0, c0 + 1] * fc * (1 - fr)
                + h[r0 + 1, c0] * (1 - fc) * fr + h[r0 + 1, c0 + 1] * fc * fr)

    def grade(self, x: np.ndarray, z: np.ndarray, d: float = 10.0) -> np.ndarray:
        gx = (self.sample(x + d, z) - self.sample(x - d, z)) / (2 * d)
        gz = (self.sample(x, z + d) - self.sample(x, z - d)) / (2 * d)
        return np.hypot(gx, gz)


def _need(p: Path, hint: str) -> Path:
    if not p.exists():
        raise Fail(f"missing {p}\n{hint}")
    return p


def load_inputs(proc: Path, assets: Path) -> dict[str, Any]:
    import geopandas as gpd
    import pandas as pd

    hint_pipe = "Run the world-data pipeline first: .venv/bin/python pipeline/build_all.py (see docs/data_contract.md)."
    meta = json.loads(_need(proc / "region_meta.json", hint_pipe).read_text())
    tmeta = json.loads(_need(proc / "terrain_meta.json", hint_pipe).read_text())
    hm = assets / tmeta.get("heightmap", "terrain/heightmap.png")
    if not hm.exists():
        hm = proc / "heightmap.png"
    terrain = Terrain(tmeta, _need(hm, hint_pipe))
    b = gpd.read_file(_need(proc / "buildings.geojson", hint_pipe))
    from pipeline.geo import scene_origin

    o = scene_origin()
    b = b.to_crs("EPSG:32611")
    from shapely.affinity import affine_transform

    b["geometry"] = b.geometry.apply(lambda g: affine_transform(g, [1, 0, 0, -1, -o.easting, o.northing]))
    edges = pd.read_parquet(_need(proc / "network_edges.parquet", hint_pipe))
    nodes = pd.read_parquet(_need(proc / "network_nodes.parquet", hint_pipe))
    manifest_p = _need(assets / "props" / "props_manifest.json",
                       "Build the prop library first: .venv-blender/bin/python blender/build_all_assets.py")
    manifest = json.loads(manifest_p.read_text())
    return {"meta": meta, "terrain": terrain, "buildings": b, "edges": edges, "nodes": nodes, "manifest": manifest}


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------


class Placer:
    def __init__(self, inp: dict[str, Any], seed: int):
        from shapely import STRtree

        self.inp = inp
        self.rng = np.random.default_rng(seed)
        self.manifest = inp["manifest"]
        self.ids = [p["id"] for p in self.manifest]
        self.recs: list[tuple[float, float, float, float, float, str]] = []  # x, z, y(None), rot, scale, id
        self.hero_y: list[float | None] = []
        b = inp["buildings"]
        self.bgeoms = list(b.geometry.values)
        self.btree = STRtree(self.bgeoms)
        self.btype = list(b["type"].values)
        self.lane_w = float(assumption("roads.lane_width_m"))
        # road ribbons (half widths) for clearance tests
        self.road_lines, self.road_half = [], []
        self.edges = inp["edges"]
        from shapely.geometry import LineString

        seen = set()
        for r in self.edges.itertuples(index=False):
            key = (min(r.u, r.v), max(r.u, r.v), r.osmid)
            if key in seen:
                continue
            seen.add(key)
            g = np.asarray(r.geometry, dtype=float).reshape(-1, 3)
            if len(g) < 2:
                continue
            ln = LineString(g[:, [0, 2]])
            self.road_lines.append(ln)
            self.road_half.append(self._half_width(r))
        self.rtree = STRtree(self.road_lines)
        self.heroes = self._load_heroes()

    # -- helpers -----------------------------------------------------------------
    def _two_way(self, r: Any) -> bool:
        return not bool(r.oneway)

    def _half_width(self, r: Any) -> float:
        lanes = max(1, int(r.lanes))
        travel = lanes * self.lane_w if self._two_way(r) else lanes * self.lane_w / 2
        cls = "arterial" if r.highway in ARTERIAL else "residential"
        if r.highway in NO_TREES:
            return travel + 2.5
        return travel + float(assumption(f"props.curb_extra_m.{cls}"))

    def _load_heroes(self) -> list[dict[str, Any]]:
        from pipeline.geo import latlon_to_scene

        p = HERO_DIR / "hero_overrides.json"
        if not p.exists():
            return []
        out = []
        cfg = json.loads(p.read_text())
        for h in cfg if isinstance(cfg, list) else cfg.get("heroes", []):
            x, z = latlon_to_scene(float(h["lat"]), float(h["lon"]))
            tj = HERO_DIR / str(h.get("trees") or f"{h['id']}_trees.json")
            out.append({"id": h["id"], "x": x, "z": z, "r": float(h["footprint_radius_m"]),
                        "rot": float(h.get("rotation_deg", 0.0)), "trees": json.loads(tj.read_text()) if tj.exists() else None})
        return out

    def in_hero(self, x: float, z: float) -> bool:
        return any(math.hypot(x - h["x"], z - h["z"]) <= h["r"] + 5.0 for h in self.heroes)

    def clear(self, x: float, z: float, bld: float, road: float) -> bool:
        from shapely.geometry import Point

        pt = Point(x, z)
        for i in self.btree.query(pt.buffer(bld)):
            if self.bgeoms[i].distance(pt) < bld:
                return False
        for i in self.rtree.query(pt.buffer(road + 25.0)):
            if self.road_lines[i].distance(pt) < self.road_half[i] + road:
                return False
        return not self.in_hero(x, z)

    def pick(self, mix_key: str) -> str:
        mix = assumption(f"props.species_mix.{mix_key}")
        keys = [k for k in mix if k in self.ids]
        w = np.array([mix[k] for k in keys], float)
        return keys[int(self.rng.choice(len(keys), p=w / w.sum()))]

    def add(self, pid: str, x: float, z: float, rot: float | None = None, scale: float | None = None,
            y: float | None = None) -> None:
        if pid not in self.ids:
            return
        entry = self.manifest[self.ids.index(pid)]
        lo, hi = entry.get("suggested_scale_range", [1.0, 1.0])
        s = float(self.rng.uniform(lo, hi)) if scale is None else scale
        r = float(self.rng.uniform(-math.pi, math.pi)) if rot is None else rot
        self.recs.append((x, z, 0.0, r, s, pid))
        self.hero_y.append(y)

    # -- rules -------------------------------------------------------------------
    def street_trees(self) -> dict[str, int]:
        presence = float(assumption("props.street_tree_presence"))
        park = float(assumption("props.parkway_offset_m"))
        clear_b = float(assumption("props.tree_building_clearance_m"))
        clear_r = float(assumption("props.tree_road_clearance_m"))
        n0 = len(self.recs)
        seen = set()
        for r in self.edges.itertuples(index=False):
            if r.highway in NO_TREES or (r.highway not in ARTERIAL and r.highway not in RESIDENTIAL):
                continue
            key = (min(r.u, r.v), max(r.u, r.v), r.osmid)
            if key in seen:
                continue
            seen.add(key)
            cls = "arterial" if r.highway in ARTERIAL else "residential"
            spacing = float(assumption(f"props.street_tree_spacing_m.{cls}"))
            off = self._half_width(r) + park
            g = np.asarray(r.geometry, dtype=float).reshape(-1, 3)[:, [0, 2]]
            seg = np.diff(g, axis=0)
            L = np.hypot(seg[:, 0], seg[:, 1])
            total = float(L.sum())
            if total < spacing:
                continue
            cum = np.concatenate([[0], np.cumsum(L)])
            for side in (-1, 1):
                t = spacing * (0.5 + self.rng.uniform(0, 0.5))
                while t < total - 4.0:
                    if self.rng.random() < presence:
                        k = int(np.searchsorted(cum, t, side="right") - 1)
                        k = min(k, len(seg) - 1)
                        f = (t - cum[k]) / max(L[k], 1e-6)
                        p = g[k] + seg[k] * f
                        tan = seg[k] / max(L[k], 1e-6)
                        nrm = np.array([-tan[1], tan[0]]) * side
                        q = p + nrm * off + tan * self.rng.normal(0, 0.6)
                        if self.clear(q[0], q[1], clear_b, clear_r):
                            self.add(self.pick(f"{cls}_street"), float(q[0]), float(q[1]))
                    t += spacing * self.rng.uniform(0.85, 1.15)
        return {"street_trees": len(self.recs) - n0}

    def yards(self) -> dict[str, int]:
        from shapely.geometry import Point

        per_house = {"yard": float(assumption("props.yard_trees_per_house")),
                     "shrub": float(assumption("props.shrubs_per_house")),
                     "grass_ornamental": float(assumption("props.grass_clumps_per_house"))}
        spacing = float(assumption("props.commercial_palm_spacing_m"))
        clear_b = float(assumption("props.tree_building_clearance_m"))
        clear_r = float(assumption("props.tree_road_clearance_m"))
        counts = {"yard_trees": 0, "shrubs": 0, "grass": 0, "commercial_trees": 0}
        for g, typ in zip(self.bgeoms, self.btype, strict=True):
            if g is None or g.is_empty:
                continue
            c = g.centroid
            if self.in_hero(c.x, c.y):
                continue
            if typ == "house":
                for kind, rate in per_house.items():
                    n = int(rate) + (1 if self.rng.random() < rate - int(rate) else 0)
                    for _ in range(n):
                        for _try in range(6):
                            d = self.rng.uniform(clear_b + 0.5, clear_b + 6.0) if kind == "yard" else self.rng.uniform(0.9, 2.5)
                            ring = g.exterior
                            p = ring.interpolate(self.rng.uniform(0, ring.length))
                            v = np.array([p.x - c.x, p.y - c.y])
                            v /= np.linalg.norm(v) + 1e-9
                            q = (p.x + v[0] * d, p.y + v[1] * d)
                            bl = clear_b if kind == "yard" else 0.6
                            if self.clear(q[0], q[1], bl, clear_r if kind == "yard" else 0.8):
                                if kind == "yard":
                                    self.add(self.pick("yard"), q[0], q[1])
                                    counts["yard_trees"] += 1
                                else:
                                    self.add(kind, q[0], q[1])
                                    counts["shrubs" if kind == "shrub" else "grass"] += 1
                                break
            elif typ in ("commercial", "apartments"):
                ring = g.buffer(clear_b + 3.5).exterior
                n = int(ring.length // spacing)
                for k in range(n):
                    p = ring.interpolate((k + self.rng.uniform(0.2, 0.8)) * spacing)
                    if self.clear(p.x, p.y, clear_b + 2.0, clear_r):
                        self.add(self.pick("commercial"), p.x, p.y)
                        counts["commercial_trees"] += 1
        del Point
        return counts

    def slopes(self) -> dict[str, int]:
        terr: Terrain = self.inp["terrain"]
        meta = self.inp["meta"]
        ext = meta.get("extent_scene") or {"min_x": terr.min_x, "max_x": terr.max_x, "min_z": terr.min_z, "max_z": terr.max_z}
        grade_min = float(assumption("props.slope_min_grade"))
        tree_ha = float(assumption("props.slope_trees_per_ha"))
        shrub_ha = float(assumption("props.slope_shrubs_per_ha"))
        b_clear = float(assumption("props.open_space_building_clearance_m"))
        step = 20.0  # sample grid (m); each cell gets Poisson(density * area) props
        xs = np.arange(ext["min_x"] + step / 2, ext["max_x"], step)
        zs = np.arange(ext["min_z"] + step / 2, ext["max_z"], step)
        gx, gz = np.meshgrid(xs, zs)
        gx, gz = gx.ravel(), gz.ravel()
        gr = terr.grade(gx, gz)
        cand = np.where(gr >= grade_min)[0]
        # clumping: smooth value noise mask so oaks gather in groves
        nz = self.rng.random((len(zs) // 6 + 2, len(xs) // 6 + 2))
        counts = {"slope_trees": 0, "slope_shrubs": 0}
        area_ha = step * step / 10000.0
        for i in cand:
            x, z = gx[i], gz[i]
            m = nz[int((z - ext["min_z"]) // (step * 6)), int((x - ext["min_x"]) // (step * 6))]
            lam_t = tree_ha * area_ha * (0.3 + 1.4 * m) * min(1.0, gr[i] / (grade_min * 2))
            lam_s = shrub_ha * area_ha * (1.3 - m)
            for kind, lam in (("tree", lam_t), ("shrub", lam_s)):
                for _ in range(self.rng.poisson(lam)):
                    qx, qz = x + self.rng.uniform(-step / 2, step / 2), z + self.rng.uniform(-step / 2, step / 2)
                    if not self.clear(qx, qz, b_clear, 3.0):
                        continue
                    if kind == "tree":
                        self.add(self.pick("slope"), qx, qz)
                        counts["slope_trees"] += 1
                    else:
                        self.add("shrub", qx, qz, scale=float(self.rng.uniform(0.8, 1.5)))
                        counts["slope_shrubs"] += 1
        return counts

    def lamps(self) -> dict[str, int]:
        nodes = self.inp["nodes"].set_index("node_id")
        corner = float(assumption("props.lamp_corner_offset_m"))
        adj: dict[int, list[tuple[np.ndarray, float, str]]] = {}
        seen = set()
        n0 = len(self.recs)
        for r in self.edges.itertuples(index=False):
            if r.highway in NO_TREES:
                continue
            g = np.asarray(r.geometry, dtype=float).reshape(-1, 3)[:, [0, 2]]
            if len(g) < 2:
                continue
            hw = self._half_width(r)
            for node, a, b in ((r.u, g[0], g[1]), (r.v, g[-1], g[-2])):
                d = b - a
                n = np.linalg.norm(d)
                if n < 1e-6:
                    continue
                adj.setdefault(int(node), []).append((d / n, hw, r.highway))
            # mid-block lamps along one side (alternating by edge parity), skip duplicates of two-way pairs
            key = (min(r.u, r.v), max(r.u, r.v), r.osmid)
            if key in seen:
                continue
            seen.add(key)
            cls = "arterial" if r.highway in ARTERIAL else "residential"
            sp = float(assumption(f"props.lamp_midblock_spacing_m.{cls}"))
            seg = np.diff(g, axis=0)
            L = np.hypot(seg[:, 0], seg[:, 1])
            total = float(L.sum())
            cum = np.concatenate([[0], np.cumsum(L)])
            side = 1 if (r.edge_idx % 2) else -1
            t = sp
            while t < total - sp * 0.5:
                k = min(int(np.searchsorted(cum, t, side="right") - 1), len(seg) - 1)
                p = g[k] + seg[k] * ((t - cum[k]) / max(L[k], 1e-6))
                tan = seg[k] / max(L[k], 1e-6)
                nrm = np.array([-tan[1], tan[0]]) * side
                q = p + nrm * (hw + corner)
                if self.clear(q[0], q[1], 1.0, 0.3):
                    self.add("street_lamp", float(q[0]), float(q[1]), rot=_rot_toward(-nrm), scale=1.0)
                side = -side if cls == "arterial" else side
                t += sp
        for nid, arms in adj.items():
            if len(arms) < 3 or nid not in nodes.index:
                continue
            nx, nz = float(nodes.at[nid, "x"]), float(nodes.at[nid, "z"])
            angs = sorted(arms, key=lambda a: math.atan2(a[0][1], a[0][0]))
            # corners between consecutive arms; signalized: two opposite corners, else one
            k_n = 2 if bool(nodes.at[nid, "signalized"]) else 1
            picks = [0] if k_n == 1 else [0, len(angs) // 2]
            for j in picks:
                a1, a2 = angs[j], angs[(j + 1) % len(angs)]
                bis = a1[0] + a2[0]
                if np.linalg.norm(bis) < 1e-3:
                    bis = np.array([-a1[0][1], a1[0][0]])
                bis /= np.linalg.norm(bis)
                d = max(a1[1], a2[1]) + corner
                q = np.array([nx, nz]) + bis * d * 1.25
                if self.clear(q[0], q[1], 1.0, 0.2):
                    self.add("street_lamp", float(q[0]), float(q[1]), rot=_rot_toward(-bis), scale=1.0)
        return {"lamps": len(self.recs) - n0}

    def hero_props(self) -> dict[str, int]:
        terr: Terrain = self.inp["terrain"]
        counts = {"hero_trees": 0, "hero_lamps": 0, "hero_parked_cars": 0}
        vshare = assumption("props.vehicle_type_shares")
        vkeys = [k for k in vshare if k in self.ids]
        vw = np.array([vshare[k] for k in vkeys], float)
        for h in self.heroes:
            t = h["trees"]
            if not t:
                continue
            y0 = float(terr.sample(np.array([h["x"]]), np.array([h["z"]]))[0]) + 0.1
            th = math.radians(h["rot"])
            c, s = math.cos(th), math.sin(th)

            def to_scene(lx: float, ly: float) -> tuple[float, float]:
                # hero local (x east, y north) -> scene (x, z=-y), then rotation.y = rotation_deg about the center
                px, pz = lx, -ly
                return h["x"] + px * c + pz * s, h["z"] - px * s + pz * c

            for tr in t.get("trees", []):
                x, z = to_scene(tr["x"], tr["y"])
                self.add(tr["species"], x, z, rot=math.radians(tr["rot_deg"]), scale=float(tr["scale"]), y=y0)
                counts["hero_trees"] += 1
            for lp in t.get("lamps", []):
                x, z = to_scene(lp["x"], lp["y"])
                a = math.radians(lp["rot_deg"])
                self.add("street_lamp", x, z, rot=_rot_toward_local(a, th), scale=0.9, y=y0)
                counts["hero_lamps"] += 1
            for car in t.get("parked_cars", []):
                x, z = to_scene(car["x"], car["y"])
                vid = vkeys[int(self.rng.choice(len(vkeys), p=vw / vw.sum()))]
                self.add(vid, x, z, rot=_rot_toward_local(math.radians(car["heading_deg"]), th), scale=1.0, y=y0)
                counts["hero_parked_cars"] += 1
        return counts

    # -- output ------------------------------------------------------------------
    def finalize(self) -> np.ndarray:
        terr: Terrain = self.inp["terrain"]
        n = len(self.recs)
        arr = np.zeros((n, 6), dtype=np.float64)
        if n == 0:
            return arr.astype(RECORD_DTYPE)
        xs = np.array([r[0] for r in self.recs])
        zs = np.array([r[1] for r in self.recs])
        ys = terr.sample(xs, zs)
        for i, hy in enumerate(self.hero_y):
            if hy is not None:
                ys[i] = hy
        arr[:, 0] = xs
        arr[:, 1] = ys - 0.05  # sink trunks / tires a few cm into the ground
        arr[:, 2] = zs
        arr[:, 3] = [r[3] for r in self.recs]
        arr[:, 4] = [r[4] for r in self.recs]
        arr[:, 5] = [self.ids.index(r[5]) for r in self.recs]
        return arr


def _rot_toward(d: np.ndarray) -> float:
    """three.js rotation.y that turns a prop's forward (-z) toward scene direction d = (dx, dz)."""
    return float(math.atan2(-d[0], -d[1]))


def _rot_toward_local(a: float, hero_rot: float) -> float:
    """Direction angle a in hero-local (x east, y north) -> rotation.y in the scene."""
    dx, dz = math.cos(a), -math.sin(a)
    return float(math.atan2(-dx, -dz)) + hero_rot


def cells_index(arr: np.ndarray, ext: dict[str, float]) -> tuple[np.ndarray, dict[str, Any]]:
    """Sort records by 500 m cell (row-major from the north-west) and return the cell table."""
    cols = int(math.ceil((ext["max_x"] - ext["min_x"]) / CELL_M))
    rows = int(math.ceil((ext["max_z"] - ext["min_z"]) / CELL_M))
    ci = np.clip(((arr[:, 0] - ext["min_x"]) // CELL_M).astype(int), 0, cols - 1)
    ri = np.clip(((arr[:, 2] - ext["min_z"]) // CELL_M).astype(int), 0, rows - 1)
    cell = ri * cols + ci
    order = np.lexsort((arr[:, 5], cell))
    arr = arr[order]
    cell = cell[order]
    ranges = []
    if len(cell):
        starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
        ends = np.r_[starts[1:], len(cell)]
        ranges = [[int(cell[s]), int(s), int(e - s)] for s, e in zip(starts, ends, strict=True)]
    return arr, {"size_m": CELL_M, "min_x": ext["min_x"], "min_z": ext["min_z"], "cols": cols, "rows": rows,
                 "order": "row-major from (min_x, min_z) = north-west; cell = row * cols + col",
                 "ranges": ranges, "ranges_fields": ["cell", "first_record", "count"]}


def write_outputs(out_dir: Path, arr: np.ndarray, placer: Placer, stats: dict[str, int], meta: dict[str, Any]) -> Path:
    ext = meta.get("terrain_extent_scene") or meta.get("extent_scene")
    arr, cells = cells_index(arr, ext)
    out_dir.mkdir(parents=True, exist_ok=True)
    binp = out_dir / "placements.bin"
    binp.write_bytes(arr.astype(RECORD_DTYPE).tobytes())
    counts = np.bincount(arr[:, 5].astype(int), minlength=len(placer.ids)) if len(arr) else np.zeros(len(placer.ids), int)
    header = {
        "format": "redraw-placements",
        "version": FORMAT_VERSION,
        "synthetic": bool(meta.get("synthetic", False)),
        "generated_from": meta.get("generated_at"),
        "seed": int(assumption("props.seed")),
        "frame": "scene meters: x east, y up (terrain elevation), z south (docs/coordinates.md)",
        "bin": "placements.bin",
        "count": int(len(arr)),
        "record": {
            "fields": RECORD_FIELDS,
            "dtype": "float32",
            "endianness": "little",
            "stride_bytes": STRIDE,
            "notes": {
                "y": "ground elevation at the prop origin (trunk base / pole base / tire contact), already sunk 5 cm",
                "rot_y": "radians, three.js Object3D.rotation.y; the prop's forward is -z (lamp arms and car noses point -z at 0)",
                "scale": "uniform scale",
                "prop_index": "index into props[] below (stored as float32, integral)",
            },
        },
        "props": [{"index": i, "id": pid, "kind": placer.manifest[i]["kind"], "file": placer.manifest[i]["file"],
                   "count": int(counts[i])} for i, pid in enumerate(placer.ids)],
        "cells": cells,
        "stats": stats,
        "tint_note": "vehicle records (parked cars) carry no color: pick paint_colors from props_manifest.json by share, "
                     "seeded by the record index, for a stable look",
    }
    (out_dir / "placements.json").write_text(json.dumps(header, indent=1) + "\n")
    return binp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=None, help="data/processed directory (default from config)")
    ap.add_argument("--assets", type=Path, default=None, help="client/public/assets directory (default from config)")
    ap.add_argument("--skip", nargs="*", default=[], choices=["street", "yards", "slopes", "lamps", "heroes"])
    a = ap.parse_args(argv)
    proc = a.processed or processed_dir()
    assets = a.assets or assets_dir()
    inp = load_inputs(proc, assets)
    meta = inp["meta"]
    if meta.get("synthetic"):
        log("WARNING region_meta.json says synthetic: placements follow the SYNTHETIC dev world (labeled in the header)")
    placer = Placer(inp, int(assumption("props.seed")))
    stats: dict[str, int] = {}
    steps = [("street", placer.street_trees), ("yards", placer.yards), ("slopes", placer.slopes),
             ("lamps", placer.lamps), ("heroes", placer.hero_props)]
    for name, fn in steps:
        if name in a.skip:
            continue
        s = fn()
        stats.update(s)
        log(f"{name}: {s}")
    arr = placer.finalize()
    binp = write_outputs(assets / "props", arr, placer, stats, meta)
    log(f"wrote {len(arr)} records -> {binp} ({binp.stat().st_size / 1e6:.1f} MB) + placements.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Fail as e:
        print(f"[build_props] ERROR {e}", file=sys.stderr)
        raise SystemExit(2) from None
    del struct
