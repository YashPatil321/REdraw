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

Trees come from the real world when the lidar step has run: data/raw/lidar/trees.parquet
(pipeline/lidar_features.py; individual trees from the 2014 USGS 3DEP point cloud with
height, crown radius and a crude palm / broadleaf / conifer guess). Every lidar tree inside
the region bbox becomes an instance at its real position, scaled to its measured height and
crown, species mapped from the class + context (street / yard / commercial / open space,
props.lidar_trees.*) with seeded variety. The rules below then only add trees in GAPS the
lidar cannot know about (houses built after the survey, developed cells without any lidar
tree, outside the lidar coverage) plus everything below tree size (shrubs, hedges, agave,
grass), street lamps and parked cars. Without trees.parquet every tree is rule-based (logged
and recorded in the header as trees_source = "procedural").

Rules (all densities in data/config/assumptions.yaml -> props.*):
- street trees along residential and arterial curbs, offset from the centerline by
  lanes * lane_width (+ parking / bike lane) + parkway, both sides, fixed spacing with jitter
- yard trees, shrubs and ornamental grass around houses (not inside footprints or roads)
- palms / trees around commercial and apartment perimeters
- oaks / eucalyptus and scrub on steep undeveloped slopes (canyons), away from buildings
- street lamps at intersections (two at signals) and mid-block along streets
- hero campuses: skip generic props inside each hero radius; add the hero's own trees,
  lot lights and parked cars from <hero>_trees.json (with lidar: real trees outside the
  hero's keep-out polygons - buildings, fields, courts, track, parking - and layout trees
  only in lidar gaps)
- everything is clipped to the region bbox (data/config/region.yaml); the run refuses to
  start while data/processed/region_meta.json was built for a different bbox
Fixed seed (props.seed): identical inputs give identical bytes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import assets_dir, assumption, processed_dir, raw_dir, region  # noqa: E402

FORMAT_VERSION = 1
RECORD_FIELDS = ["x", "y", "z", "rot_y", "scale", "prop_index"]
RECORD_DTYPE = np.dtype("<f4")
STRIDE = 4 * len(RECORD_FIELDS)
CELL_M = 500.0
HERO_DIR = Path(__file__).resolve().parent / "hero_overrides"

ARTERIAL = {"primary", "secondary", "tertiary", "primary_link", "secondary_link", "tertiary_link", "trunk"}
RESIDENTIAL = {"residential", "living_street", "unclassified"}
NO_TREES = {"motorway", "motorway_link", "trunk_link"}
LIDAR_CLASSES = ("palm", "broadleaf", "conifer")
HERO_KEEPOUT_CLASSES = {"building", "turf_field", "track_red", "court_blue", "dirt_infield", "asphalt", "asphalt_light",
                        "rubber_play"}


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


def region_polygon(n: int = 24) -> Any:
    """The region bbox (region.yaml, lat/lon) as a scene-coordinate polygon (edges densified:
    a lat/lon box is slightly curved / rotated in UTM)."""
    from shapely.geometry import Polygon

    from pipeline.geo import latlon_to_scene

    bb = region()["bbox"]
    s_, n_, w, e = (float(bb[k]) for k in ("south", "north", "west", "east"))
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    ll = ([(s_, w + (e - w) * k) for k in t] + [(s_ + (n_ - s_) * k, e) for k in t]
          + [(n_, e - (e - w) * k) for k in t] + [(n_ - (n_ - s_) * k, w) for k in t])
    return Polygon([latlon_to_scene(lat, lon) for lat, lon in ll])


def check_region(meta: dict[str, Any], allow_mismatch: bool = False) -> None:
    """region_meta.json must describe the bbox in region.yaml (same scene origin as the lidar trees)."""
    want = region()["bbox"]
    got = meta.get("bbox") or {}
    bad = [k for k in ("south", "north", "west", "east") if k not in got or abs(float(got[k]) - float(want[k])) > 1e-6]
    if not bad:
        return
    msg = (f"data/processed/region_meta.json bbox {got} does not match data/config/region.yaml bbox {want}: the "
           "processed world (terrain, buildings, roads) is stale or being rebuilt. Rebuild it first: "
           ".venv/bin/python pipeline/build_all.py")
    if not allow_mismatch:
        raise Fail(msg)
    log(f"WARNING {msg} (continuing: --allow-region-mismatch)")


def load_lidar_trees(path: Path) -> tuple[Any, dict[str, int]]:
    """trees.parquet (docs/data_contract.md "Lidar features") -> DataFrame x, z, h, r, cls in the CURRENT
    scene frame (recomputed from easting / northing when present). Trees on ground regraded since the
    survey (|ground_y - ground_lidar_m| > lidar.ground_change_m) are stale and dropped."""
    import pandas as pd

    if not path.exists():
        return None, {}
    df = pd.read_parquet(path)
    need = {"height_m", "crown_radius_m", "species_guess"}
    if not need <= set(df.columns) or not ({"easting", "northing"} <= set(df.columns) or {"x", "z"} <= set(df.columns)):
        raise Fail(f"{path}: expected columns {sorted(need)} + easting/northing (or x/z), got {list(df.columns)}")
    stats = {"lidar_trees_in_file": int(len(df))}
    if {"easting", "northing"} <= set(df.columns):
        from pipeline.geo import scene_origin

        o = scene_origin()
        x = df["easting"].to_numpy(np.float64) - o.easting
        z = -(df["northing"].to_numpy(np.float64) - o.northing)
    else:
        x, z = df["x"].to_numpy(np.float64), df["z"].to_numpy(np.float64)
    keep = np.isfinite(x) & np.isfinite(z) & np.isfinite(df["height_m"].to_numpy(np.float64))
    if {"ground_y", "ground_lidar_m"} <= set(df.columns):
        dg = np.abs(df["ground_y"].to_numpy(np.float64) - df["ground_lidar_m"].to_numpy(np.float64))
        stale = np.nan_to_num(dg, nan=0.0) > float(assumption("lidar.ground_change_m"))
        stats["lidar_trees_stale_regraded"] = int((stale & keep).sum())
        keep &= ~stale
    cls = df["species_guess"].astype(str).str.lower().to_numpy()
    cls = np.where(np.isin(cls, LIDAR_CLASSES), cls, "broadleaf")
    r = np.nan_to_num(df["crown_radius_m"].to_numpy(np.float64), nan=2.0)
    out = pd.DataFrame({"x": x[keep], "z": z[keep], "h": df["height_m"].to_numpy(np.float64)[keep],
                        "r": np.clip(r[keep], 0.5, 15.0), "cls": cls[keep]})
    return out, stats


def lidar_coverage(raw: Path) -> Any:
    """Scene polygon of the lidar AOI (lidar.source.json aoi_utm), or None."""
    from shapely.geometry import box

    p = raw / "lidar" / "lidar.source.json"
    if not p.exists():
        return None
    aoi = json.loads(p.read_text()).get("aoi_utm")
    if not aoi:
        return None
    from pipeline.geo import scene_origin

    o = scene_origin()
    return box(aoi["minx"] - o.easting, -(aoi["maxy"] - o.northing), aoi["maxx"] - o.easting, -(aoi["miny"] - o.northing))


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
        self.region_poly = inp.get("region_poly")
        self.coverage = inp.get("lidar_coverage")
        self.lidar_xz: np.ndarray | None = None  # kept lidar trees (scene x, z) once lidar_trees() ran
        self.lidar_kd: Any = None
        self.gap_cells: set[tuple[int, int]] = set()
        self.gap_houses: set[int] = set()
        self.gap_m = float(assumption("props.lidar_trees.gap_cell_m"))

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
            hd = {"id": h["id"], "x": x, "z": z, "r": float(h["footprint_radius_m"]),
                  "rot": float(h.get("rotation_deg", 0.0)), "trees": json.loads(tj.read_text()) if tj.exists() else None}
            hd["keepout"] = self._hero_keepout(hd)
            out.append(hd)
        return out

    @staticmethod
    def _hero_local_to_scene(h: dict[str, Any], lx: np.ndarray, ly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hero local (x east, y north) -> scene (x, z = -y), rotated by rotation_deg about the center."""
        th = math.radians(h["rot"])
        c, s = math.cos(th), math.sin(th)
        px, pz = np.asarray(lx, float), -np.asarray(ly, float)
        return h["x"] + px * c + pz * s, h["z"] - px * s + pz * c

    def _hero_keepout(self, h: dict[str, Any]) -> Any:
        """Union of the hero's keep-out polygons (buildings, fields, courts, track, parking) in scene coords."""
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        t = h["trees"] or {}
        polys = []
        for k in t.get("keepout", []):
            if k.get("cls") not in HERO_KEEPOUT_CLASSES:
                continue
            ext = np.asarray(k["exterior"], float)
            if len(ext) < 3:
                continue
            ex, ez = self._hero_local_to_scene(h, ext[:, 0], ext[:, 1])
            holes = []
            for hr in k.get("holes", []):
                hh = np.asarray(hr, float)
                if len(hh) >= 3:
                    hx, hz = self._hero_local_to_scene(h, hh[:, 0], hh[:, 1])
                    holes.append(list(zip(hx, hz, strict=True)))
            g = Polygon(list(zip(ex, ez, strict=True)), holes).buffer(0)
            if not g.is_empty:
                polys.append(g.buffer(1.0) if k["cls"] == "building" else g)
        return unary_union(polys) if polys else None

    def in_hero(self, x: float, z: float) -> bool:
        return any(math.hypot(x - h["x"], z - h["z"]) <= h["r"] + 5.0 for h in self.heroes)

    def hero_at(self, x: float, z: float) -> dict[str, Any] | None:
        for h in self.heroes:
            if math.hypot(x - h["x"], z - h["z"]) <= h["r"] + 5.0:
                return h
        return None

    # -- lidar gaps ----------------------------------------------------------------
    def covered(self, x: float, z: float) -> bool:
        """Inside the lidar survey (trees there come from trees.parquet, not from rules)."""
        if self.lidar_xz is None:
            return False
        if self.coverage is None:
            return True
        from shapely import contains_xy

        return bool(contains_xy(self.coverage, x, z))

    def _cell(self, x: float, z: float) -> tuple[int, int]:
        return int(math.floor(x / self.gap_m)), int(math.floor(z / self.gap_m))

    def tree_gap(self, x: float, z: float, near_m: float = 6.0) -> bool:
        """May a rule-based tree go here? Always without lidar; with lidar only outside the survey or in
        a gap cell (developed after the survey / no lidar tree at all), and never next to a real tree."""
        if self.lidar_xz is None or not self.covered(x, z):
            return True
        if self._cell(x, z) not in self.gap_cells:
            return False
        return not self.lidar_kd.query_ball_point([x, z], near_m)

    def clear(self, x: float, z: float, bld: float, road: float, allow_hero: bool = False) -> bool:
        from shapely.geometry import Point

        pt = Point(x, z)
        for i in self.btree.query(pt.buffer(bld)):
            if self.bgeoms[i].distance(pt) < bld:
                return False
        for i in self.rtree.query(pt.buffer(road + 25.0)):
            if self.road_lines[i].distance(pt) < self.road_half[i] + road:
                return False
        return allow_hero or not self.in_hero(x, z)

    def young(self, pid: str) -> float:
        """Scale of a rule-based tree: mature range without lidar; young planting in a lidar gap."""
        e = self.manifest[self.ids.index(pid)] if pid in self.ids else {}
        lo, hi = e.get("suggested_scale_range", [1.0, 1.0])
        s = float(self.rng.uniform(lo, hi))
        if self.lidar_xz is not None:
            ylo, yhi = (float(v) for v in assumption("props.lidar_trees.young_scale"))
            s *= float(self.rng.uniform(ylo, yhi))
        return s

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

    # -- real trees ----------------------------------------------------------------
    def _species_weights(self, mix_key: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        mix = assumption(f"props.lidar_trees.mix.{mix_key}")
        keys = [k for k in mix if k in self.ids]
        if not keys:
            raise Fail(f"assumptions props.lidar_trees.mix.{mix_key}: none of {list(mix)} is in props_manifest.json")
        w = np.array([mix[k] for k in keys], float)
        asp = np.array([math.log(max(self._dims(k)[1], 0.3) / max(self._dims(k)[0], 0.5)) for k in keys])
        return keys, w / w.sum(), asp

    def _dims(self, pid: str) -> tuple[float, float]:
        """(height_m, crown_radius_m) of a tree prop model."""
        e = self.manifest[self.ids.index(pid)]
        return float(e.get("height_m", 8.0)), float(e.get("crown_radius_m") or e.get("footprint_radius_m", 3.0))

    def lidar_trees(self, df: Any) -> dict[str, int]:
        """Real trees: one instance per lidar tree, scaled to its measured height / crown."""
        import shapely
        from scipy.spatial import cKDTree

        stats = {"lidar_trees": 0, "lidar_outside_region": 0, "lidar_dropped_building": 0, "lidar_dropped_road": 0,
                 "lidar_moved_off_road": 0, "lidar_dropped_hero_keepout": 0}
        x, z = df["x"].to_numpy(), df["z"].to_numpy()
        h, r, cls = df["h"].to_numpy(), df["r"].to_numpy(), df["cls"].to_numpy()
        inside = shapely.contains_xy(self.region_poly, x, z) if self.region_poly is not None else np.ones(len(x), bool)
        stats["lidar_outside_region"] = int((~inside).sum())
        x, z, h, r, cls = x[inside], z[inside], h[inside], r[inside], cls[inside]
        n = len(x)
        pts = shapely.points(x, z)
        # nearest building: distance + type
        bd = np.full(n, np.inf)
        btyp = np.full(n, "", dtype=object)
        if self.bgeoms:
            res, dist = self.btree.query_nearest(pts, max_distance=250.0, return_distance=True, all_matches=False)
            bd[res[0]] = dist
            btyp[res[0]] = np.asarray(self.btype, dtype=object)[res[1]]
        # road edge distance (centerline distance - half width), nearest road
        rd = np.full(n, np.inf)
        rj = np.full(n, -1, dtype=np.int64)
        pairs = self.rtree.query(pts, predicate="dwithin", distance=60.0)
        if pairs.shape[1]:
            lines = np.asarray(self.road_lines, dtype=object)
            half = np.asarray(self.road_half, float)
            d = shapely.distance(pts[pairs[0]], lines[pairs[1]]) - half[pairs[1]]
            order = np.lexsort((d, pairs[0]))
            first = order[np.r_[True, pairs[0][order][1:] != pairs[0][order][:-1]]]
            rd[pairs[0][first]] = d[first]
            rj[pairs[0][first]] = pairs[1][first]
        b_clear = float(assumption("props.lidar_trees.building_clearance_m"))
        r_clear = float(assumption("props.lidar_trees.road_clearance_m"))
        open_b = float(assumption("props.open_space_building_clearance_m"))
        open_r = float(assumption("props.lidar_trees.open_space_road_m"))
        street_r = float(assumption("props.lidar_trees.street_road_m"))
        hw = float(assumption("props.lidar_trees.height_weight"))
        s_lo, s_hi = (float(v) for v in assumption("props.lidar_trees.scale_clamp"))
        euc_h = float(assumption("props.lidar_trees.eucalyptus_min_height_m"))
        euc_p = float(assumption("props.lidar_trees.eucalyptus_tall_share"))
        fan_h = float(assumption("props.lidar_trees.palm_fan_min_height_m"))
        sig = float(assumption("props.lidar_trees.aspect_sigma"))
        mixes = {k: self._species_weights(k) for k in
                 ("palm", "conifer", "broadleaf_street", "broadleaf_yard", "broadleaf_commercial", "broadleaf_open")}
        kept: list[tuple[float, float]] = []
        from shapely.geometry import Point

        for i in range(n):
            xi, zi = float(x[i]), float(z[i])
            hero = self.hero_at(xi, zi)
            if hero is not None and hero["keepout"] is not None:
                if hero["keepout"].contains(Point(xi, zi)):
                    stats["lidar_dropped_hero_keepout"] += 1
                    continue
            elif bd[i] < b_clear:
                stats["lidar_dropped_building"] += 1
                continue
            if rd[i] < r_clear and hero is None:
                ln = self.road_lines[int(rj[i])]
                p0 = ln.interpolate(ln.project(Point(xi, zi)))
                v = np.array([xi - p0.x, zi - p0.y])
                lv = float(np.linalg.norm(v))
                if lv < 0.3:
                    stats["lidar_dropped_road"] += 1
                    continue
                q = np.array([p0.x, p0.y]) + v / lv * (self.road_half[int(rj[i])] + r_clear)
                if not self.clear(float(q[0]), float(q[1]), b_clear, r_clear * 0.5, allow_hero=True):
                    stats["lidar_dropped_road"] += 1
                    continue
                xi, zi = float(q[0]), float(q[1])
                stats["lidar_moved_off_road"] += 1
            hi, ri = float(h[i]), float(r[i])
            c = str(cls[i])
            if c == "palm":
                keys, w, asp = mixes["palm"]
                f = 1.0 / (1.0 + math.exp(-(hi - fan_h) / 1.2))
                w = w * np.array([f if k == "tree_palm_fan" else (1.0 - f if k == "tree_palm_queen" else 1.0) for k in keys])
            elif c == "conifer":
                keys, w, asp = mixes["conifer"]
            else:
                if hero is not None or (btyp[i] in ("commercial", "apartments", "school", "other") and bd[i] < 30.0):
                    ctx = "broadleaf_commercial"
                elif bd[i] > open_b and rd[i] > open_r:
                    ctx = "broadleaf_open"
                elif rd[i] < street_r:
                    ctx = "broadleaf_street"
                else:
                    ctx = "broadleaf_yard"
                keys, w, asp = mixes[ctx]
                if hi >= euc_h and "tree_eucalyptus" in self.ids and self.rng.random() < euc_p:
                    keys, w, asp = ["tree_eucalyptus"], np.ones(1), np.zeros(1)
            if len(keys) > 1:
                la = math.log(max(ri, 0.3) / max(hi, 0.5))
                w = w * np.exp(-((la - asp) ** 2) / (2 * sig * sig))
            w = w / w.sum() if w.sum() > 0 else np.full(len(keys), 1.0 / len(keys))
            pid = keys[int(self.rng.choice(len(keys), p=w))]
            h0, r0 = self._dims(pid)
            sc = (hi / h0) ** hw * (max(ri, 0.5) / r0) ** (1.0 - hw)
            self.add(pid, xi, zi, scale=float(np.clip(sc, s_lo, s_hi)))
            kept.append((xi, zi))
            stats["lidar_trees"] += 1
        self.lidar_xz = np.asarray(kept, float).reshape(-1, 2)
        self.lidar_kd = cKDTree(self.lidar_xz) if len(self.lidar_xz) else cKDTree(np.zeros((1, 2)) + 1e9)
        self._find_gaps()
        stats["lidar_gap_cells"] = len(self.gap_cells)
        stats["lidar_gap_houses"] = len(self.gap_houses)
        return stats

    def _find_gaps(self) -> None:
        """Gap cells: developed cells (>= 3 houses) without any lidar tree, and cells of houses built after
        the survey (parcel_year_built > props.lidar_trees.survey_year): rule-based trees fill those."""
        survey = int(assumption("props.lidar_trees.survey_year"))
        tree_cells: set[tuple[int, int]] = {self._cell(float(a), float(b)) for a, b in self.lidar_xz}
        houses: dict[tuple[int, int], int] = {}
        years = self.inp["buildings"].get("parcel_year_built")
        yv = None
        if years is not None:
            import pandas as pd

            yv = pd.to_numeric(years, errors="coerce").to_numpy()
        for k, (g, typ) in enumerate(zip(self.bgeoms, self.btype, strict=True)):
            if typ != "house" or g is None or g.is_empty:
                continue
            c = g.centroid
            cell = self._cell(c.x, c.y)
            houses[cell] = houses.get(cell, 0) + 1
            if yv is not None and np.isfinite(yv[k]) and yv[k] > survey:
                self.gap_houses.add(k)
                self.gap_cells.add(cell)
        self.gap_cells |= {c for c, nh in houses.items() if nh >= 3 and c not in tree_cells}

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
                        if self.clear(q[0], q[1], clear_b, clear_r) and self.tree_gap(float(q[0]), float(q[1])):
                            pid = self.pick(f"{cls}_street")
                            self.add(pid, float(q[0]), float(q[1]), scale=self.young(pid))
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
        counts = {"yard_trees": 0, "shrubs": 0, "hedge_segments": 0, "grass": 0, "commercial_trees": 0}
        lidar = self.lidar_xz is not None
        for k_b, (g, typ) in enumerate(zip(self.bgeoms, self.btype, strict=True)):
            if g is None or g.is_empty:
                continue
            c = g.centroid
            if self.in_hero(c.x, c.y):
                continue
            if typ == "house":
                # with lidar, yard trees only for houses the survey could not see (built later / gap cells)
                gap_house = not lidar or k_b in self.gap_houses or self.tree_gap(c.x, c.y, near_m=0.0)
                for kind, rate in per_house.items():
                    if kind == "yard" and not gap_house:
                        continue
                    n = int(rate) + (1 if self.rng.random() < rate - int(rate) else 0)
                    for _ in range(n):
                        if kind == "shrub":
                            sp = self.pick("yard_shrub")
                            if sp == "hedge":
                                counts["hedge_segments"] += self.hedge_run(g, c)
                                continue
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
                                    if lidar and self.lidar_kd.query_ball_point([q[0], q[1]], 5.0):
                                        break
                                    pid = self.pick("yard")
                                    self.add(pid, q[0], q[1], scale=self.young(pid))
                                    counts["yard_trees"] += 1
                                else:
                                    self.add(sp if kind == "shrub" else kind, q[0], q[1])
                                    counts["shrubs" if kind == "shrub" else "grass"] += 1
                                break
            elif typ in ("commercial", "apartments"):
                ring = g.buffer(clear_b + 3.5).exterior
                n = int(ring.length // spacing)
                for k in range(n):
                    p = ring.interpolate((k + self.rng.uniform(0.2, 0.8)) * spacing)
                    if self.clear(p.x, p.y, clear_b + 2.0, clear_r) and self.tree_gap(p.x, p.y):
                        pid = self.pick("commercial")
                        self.add(pid, p.x, p.y, scale=self.young(pid))
                        counts["commercial_trees"] += 1
        del Point
        return counts

    def hedge_run(self, g: Any, c: Any) -> int:
        """A clipped hedge run (2 m prop segments end to end) parallel to a house wall, 1.2 m out."""
        lo, hi = (int(v) for v in assumption("props.hedge_run_segments"))
        ring = g.exterior
        seg_len = 2.0
        if "hedge" not in self.ids:
            return 0
        for _try in range(4):
            s0 = float(self.rng.uniform(0, ring.length))
            p = ring.interpolate(s0)
            p2 = ring.interpolate(min(ring.length, s0 + 0.5))
            t = np.array([p2.x - p.x, p2.y - p.y])
            if np.linalg.norm(t) < 1e-6:
                continue
            t /= np.linalg.norm(t)
            nrm = np.array([t[1], -t[0]])
            if nrm @ np.array([p.x - c.x, p.y - c.y]) < 0:
                nrm = -nrm
            n_seg = int(self.rng.integers(lo, hi + 1))
            start = np.array([p.x, p.y]) + nrm * 1.2
            pts = [start + t * seg_len * (k + 0.5) for k in range(n_seg)]
            if all(self.clear(float(q[0]), float(q[1]), 0.7, 0.6) for q in pts):
                rot = float(math.atan2(-t[1], t[0]))  # prop local +x along the wall
                for q in pts:
                    self.add("hedge", float(q[0]), float(q[1]), rot=rot, scale=1.0)
                return n_seg
        return 0

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
                        if self.covered(qx, qz):  # lidar knows the real canyon trees
                            continue
                        self.add(self.pick("slope"), qx, qz)
                        counts["slope_trees"] += 1
                    else:
                        self.add("shrub", qx, qz, scale=float(self.rng.uniform(0.8, 1.3)))
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

            def to_scene(lx: float, ly: float, h: dict[str, Any] = h, c: float = c, s: float = s) -> tuple[float, float]:
                # hero local (x east, y north) -> scene (x, z=-y), then rotation.y = rotation_deg about the center
                px, pz = lx, -ly
                return h["x"] + px * c + pz * s, h["z"] - px * s + pz * c

            for tr in t.get("trees", []):
                x, z = to_scene(tr["x"], tr["y"])
                # with lidar: a layout tree survives only where no real tree stands within 20 m
                # (post-survey planting, or trees too small for the 2014 survey)
                if self.lidar_xz is not None and self.covered(x, z) and self.lidar_kd.query_ball_point([x, z], 20.0):
                    counts["hero_layout_trees_replaced_by_lidar"] = counts.get("hero_layout_trees_replaced_by_lidar", 0) + 1
                    continue
                self.add(tr["species"], x, z, rot=math.radians(tr["rot_deg"]), scale=float(tr["scale"]), y=y0)
                counts["hero_trees"] += 1
            for lp in t.get("lamps", []):
                x, z = to_scene(lp["x"], lp["y"])
                a = math.radians(lp["rot_deg"])
                self.add("street_lamp", x, z, rot=_rot_toward_local(a, th), scale=1.0, y=y0)
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
        inside = (xs >= terr.min_x) & (xs <= terr.max_x) & (zs >= terr.min_z) & (zs <= terr.max_z)
        if (~inside).any():
            log(f"dropped {int((~inside).sum())} props outside the terrain extent")
        if self.region_poly is not None:
            from shapely import contains_xy

            reg = contains_xy(self.region_poly, xs, zs)
            if (inside & ~reg).any():
                log(f"dropped {int((inside & ~reg).sum())} props outside the region bbox")
            inside &= reg
        return arr[inside]


def _rot_toward(d: np.ndarray) -> float:
    """three.js rotation.y that turns a prop's forward (-z) toward scene direction d = (dx, dz)."""
    return float(math.atan2(-d[0], -d[1]))


def _rot_toward_local(a: float, hero_rot: float) -> float:
    """Direction angle a in hero-local (x east, y north) -> rotation.y in the scene."""
    dx, dz = math.cos(a), -math.sin(a)
    return float(math.atan2(-dx, -dz)) + hero_rot


def cells_index(arr: np.ndarray, ext: dict[str, float], n_props: int) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Sort records by (prop_index, 500 m cell) and build per-prop sections with per-cell ranges.

    Each prop's records are contiguous (a section: byte offset + count), and inside a section
    they are grouped by grid cell (row-major from the north-west corner), so a client can
    instance one prop type per draw call and stream / cull by cell.
    """
    cols = int(math.ceil((ext["max_x"] - ext["min_x"]) / CELL_M))
    rows = int(math.ceil((ext["max_z"] - ext["min_z"]) / CELL_M))
    ci = np.clip(((arr[:, 0] - ext["min_x"]) // CELL_M).astype(int), 0, cols - 1)
    ri = np.clip(((arr[:, 2] - ext["min_z"]) // CELL_M).astype(int), 0, rows - 1)
    cell = ri * cols + ci
    prop = arr[:, 5].astype(int)
    order = np.lexsort((cell, prop))
    arr, cell, prop = arr[order], cell[order], prop[order]
    sections = []
    for k in range(n_props):
        idx = np.flatnonzero(prop == k)
        first = int(idx[0]) if len(idx) else 0
        ranges = []
        if len(idx):
            c = cell[idx]
            starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
            ends = np.r_[starts[1:], len(c)]
            ranges = [[int(c[s0]), first + int(s0), int(e - s0)] for s0, e in zip(starts, ends, strict=True)]
        sections.append({"offset": first * STRIDE, "first_record": first, "count": int(len(idx)), "cells": ranges})
    grid = {"size_m": CELL_M, "min_x": ext["min_x"], "min_z": ext["min_z"], "cols": cols, "rows": rows,
            "order": "row-major from (min_x, min_z) = north-west corner; cell = row * cols + col",
            "ranges_fields": ["cell", "first_record", "count"]}
    return arr, grid, sections


def write_outputs(out_dir: Path, arr: np.ndarray, placer: Placer, stats: dict[str, Any], meta: dict[str, Any]) -> Path:
    ext = meta.get("terrain_extent_scene") or meta.get("extent_scene")
    arr, grid, sections = cells_index(arr, ext, len(placer.ids))
    out_dir.mkdir(parents=True, exist_ok=True)
    binp = out_dir / "placements.bin"
    binp.write_bytes(arr.astype(RECORD_DTYPE).tobytes())
    header = {
        "format": "redraw-placements",
        "version": FORMAT_VERSION,
        "synthetic": bool(meta.get("synthetic", False)),
        "generated_from": meta.get("generated_at"),
        "seed": int(assumption("props.seed")),
        "frame": "scene meters: x east, y up (terrain elevation), z south (docs/coordinates.md)",
        "bin": "placements.bin",
        "count": int(len(arr)),
        "fields": RECORD_FIELDS,
        "stride": STRIDE,
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
        "props": [{"index": i, "id": pid, "kind": placer.manifest[i]["kind"], "file": placer.manifest[i]["file"], **sections[i]}
                  for i, pid in enumerate(placer.ids)],
        "cells": grid,
        "stats": stats,
        "trees_source": stats.get("trees_source", "procedural"),
        "region_bbox": region()["bbox"],
        "scale_note": "trees from lidar carry scale = measured size / model size (props.lidar_trees.scale_clamp), "
                      "which can exceed the manifest suggested_scale_range; everything else stays inside it",
        "tint_note": "vehicle records (parked cars) carry no color: pick paint_colors from props_manifest.json by share, "
                     "seeded by the record index, for a stable look",
    }
    (out_dir / "placements.json").write_text(json.dumps(header, indent=1) + "\n")
    return binp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=None, help="data/processed directory (default from config)")
    ap.add_argument("--assets", type=Path, default=None, help="client/public/assets directory (default from config)")
    ap.add_argument("--skip", nargs="*", default=[], choices=["lidar", "street", "yards", "slopes", "lamps", "heroes"])
    ap.add_argument("--lidar-trees", type=Path, default=None, help="trees.parquet (default data/raw/lidar/trees.parquet)")
    ap.add_argument("--allow-region-mismatch", action="store_true",
                    help="dev only: run although region_meta.json was built for another bbox than region.yaml")
    a = ap.parse_args(argv)
    proc = a.processed or processed_dir()
    assets = a.assets or assets_dir()
    inp = load_inputs(proc, assets)
    meta = inp["meta"]
    check_region(meta, a.allow_region_mismatch)
    inp["region_poly"] = region_polygon()
    stats: dict[str, Any] = {}
    lidar_df = None
    trees_p = a.lidar_trees or raw_dir() / "lidar" / "trees.parquet"
    if meta.get("synthetic"):
        log("WARNING region_meta.json says synthetic: placements follow the SYNTHETIC dev world (labeled in the header); "
            "lidar trees are not used")
    elif "lidar" not in a.skip:
        lidar_df, lst = load_lidar_trees(trees_p)
        stats.update(lst)
        if lidar_df is None:
            log(f"NOTE {trees_p} not found: every tree is rule-based (trees_source = procedural). Produce it with "
                ".venv/bin/python pipeline/fetch_lidar.py && .venv/bin/python pipeline/lidar_features.py")
        else:
            inp["lidar_coverage"] = lidar_coverage(raw_dir())
            log(f"lidar trees: {len(lidar_df)} usable of {lst.get('lidar_trees_in_file')} in {trees_p}")
    placer = Placer(inp, int(assumption("props.seed")))
    steps = [("lidar", lambda: placer.lidar_trees(lidar_df)) if lidar_df is not None else ("lidar", dict),
             ("street", placer.street_trees), ("yards", placer.yards), ("slopes", placer.slopes),
             ("lamps", placer.lamps), ("heroes", placer.hero_props)]
    for name, fn in steps:
        if name in a.skip:
            continue
        s = fn()
        stats.update(s)
        log(f"{name}: {s}")
    arr = placer.finalize()
    stats["trees_source"] = "lidar+gaps" if lidar_df is not None else "procedural"
    stats["lidar_file"] = trees_p.name if lidar_df is not None else None
    binp = write_outputs(assets / "props", arr, placer, stats, meta)
    log(f"wrote {len(arr)} records -> {binp} ({binp.stat().st_size / 1e6:.1f} MB) + placements.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Fail as e:
        print(f"[build_props] ERROR {e}", file=sys.stderr)
        raise SystemExit(2) from None
