"""Adapter: real footprints + lidar roof models -> per-tile building specs for the Blender HD builder.

Runs in the MAIN venv (geopandas, rasterio). Writes `blender/build/buildings_hd/specs/r{r}_c{c}.json`
(scene coordinates) that `blender/buildings/build.py` (headless bpy, .venv-blender) turns into
`client/public/assets/buildings_hd/` glTF tiles. See blender/README.md "HD buildings".

Building set and ids are exactly the pipeline's (`pipeline.build_buildings.load_osm_buildings` +
`prepare_buildings` + `apply_heroes`, same tile grid), so `_BUILDING_ID` matches buildings.geojson.
Hero campuses (pipeline/hero_overrides) are skipped: every footprint whose centroid is inside a
hero radius is replaced by the hero model in the pipeline tiles. Lidar-only buildings
(`data/raw/lidar/missing_buildings.geojson`) are appended with ids above the pipeline's maximum
unless the processed buildings.geojson already contains them.

Roof model per building, best source first:
  lidar     data/raw/lidar/buildings_roofs.parquet (eave / ridge height above ground, roof type,
            ridge azimuth, pitch, planes)  -> roof_source "lidar"
  tag       OSM/Overture `height` (+ roof:shape)                                   -> "tag"
  heuristic building class + footprint shape                                        -> "heuristic"

    .venv/bin/python pipeline/build_buildings_blender.py            # specs only
    .venv-blender/bin/python blender/buildings/build.py              # geometry + glTF (reads the specs)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.ops
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline import build_buildings as bb  # noqa: E402
from pipeline.common import log, tile_grid  # noqa: E402
from pipeline.config import assumption, load_yaml, processed_dir, raw_dir, region  # noqa: E402
from pipeline.geo import scene_origin  # noqa: E402

SPEC_DIR = REPO / "blender" / "build" / "buildings_hd" / "specs"
LIDAR_DIR_NAME = "lidar"
# roads that houses face (Overture segment classes); service = private streets in gated tracts / alleys
FRONT_CLASSES = {"residential", "living_street", "unclassified", "tertiary", "secondary", "primary", "service"}
ARTERIAL = {"primary", "secondary", "trunk", "motorway"}
SERVICE_SKIP_SUBCLASS = {"parking_aisle"}
STREET_SEARCH_M = 60.0
TAG_RIDGE_OFFSET_M = float(assumption("buildings_hd.tag_ridge_offset_m"))


# ---------------------------------------------------------------------------
# terrain
# ---------------------------------------------------------------------------


class DemSampler:
    """Bilinear sampling of the 3DEP lidar DEM (EPSG:32611) at scene points."""

    def __init__(self, path: Path):
        import rasterio

        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `.venv/bin/python pipeline/fetch_aws.py` (USGS 3DEP DEM)")
        with rasterio.open(path) as ds:
            self.a = ds.read(1).astype(np.float32)
            nod = ds.nodata
            t = ds.transform
        if nod is not None:
            bad = self.a == nod
            if bad.any():
                self.a[bad] = np.nan
                fill = float(np.nanmedian(self.a))
                self.a[bad] = fill
        o = scene_origin()
        # pixel centers: col j -> easting t.c + (j + .5) t.a ; row i -> northing t.f + (i + .5) t.e
        self.e0 = t.c + 0.5 * t.a - o.easting  # scene x of column 0
        self.n0 = o.northing - (t.f + 0.5 * t.e)  # scene z of row 0
        self.dx = t.a
        self.dz = -t.e

    def sample(self, x: Any, z: Any) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        rows, cols = self.a.shape
        fj = np.clip((x - self.e0) / self.dx, 0, cols - 1.000001)
        fi = np.clip((z - self.n0) / self.dz, 0, rows - 1.000001)
        j0 = np.floor(fj).astype(np.int64)
        i0 = np.floor(fi).astype(np.int64)
        tj, ti = fj - j0, fi - i0
        a = self.a
        return (a[i0, j0] * (1 - ti) * (1 - tj) + a[i0, j0 + 1] * (1 - ti) * tj + a[i0 + 1, j0] * ti * (1 - tj) + a[i0 + 1, j0 + 1] * ti * tj).astype(np.float64)


# ---------------------------------------------------------------------------
# building set (same ids as the pipeline)
# ---------------------------------------------------------------------------


def pipeline_buildings(dem: DemSampler) -> tuple[gpd.GeoDataFrame, list[Any]]:
    """Footprints exactly as pipeline/build_all.py prepares them (EPSG:32611) + heroes."""
    from pipeline import fetch_osm

    raw = raw_dir()
    src = raw / "osm_buildings.geojson"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing: run `.venv/bin/python pipeline/fetch_aws.py` or pipeline/fetch_osm.py")
    grid = tile_grid()
    footprints = bb.load_osm_buildings(src, bb.load_landuse(raw))
    school_areas = None
    sch = raw / "osm_schools.geojson"
    if sch.exists():
        _, school_areas = fetch_osm.load_schools(sch, load_yaml("schools.yaml")["schools"])
    bdf = bb.prepare_buildings(footprints, dem, grid, school_areas=school_areas)  # type: ignore[arg-type]
    heroes = bb.load_heroes()
    if heroes:
        bdf = bb.apply_heroes(bdf, heroes, dem, grid)  # type: ignore[arg-type]
    return bdf, heroes


def check_processed_ids(bdf: gpd.GeoDataFrame) -> str:
    """Compare against data/processed/buildings.geojson when it was built for the same bbox."""
    proc = processed_dir()
    meta_p, geo_p = proc / "region_meta.json", proc / "buildings.geojson"
    if not (meta_p.exists() and geo_p.exists()):
        return "processed buildings.geojson missing (ids follow the pipeline code)"
    meta = json.loads(meta_p.read_text())
    bbox = region()["bbox"]
    if any(abs(float(meta["bbox"][k]) - float(bbox[k])) > 1e-9 for k in ("south", "north", "west", "east")):
        return "processed buildings.geojson is from another bbox (stale); ids follow the pipeline code for region.yaml"
    pg = gpd.read_file(geo_p)
    a = bdf.set_index("id")[["centroid_x", "centroid_z"]]
    b = pg.set_index("id")[["centroid_x", "centroid_z"]]
    common = a.index.intersection(b.index)
    d = np.hypot(a.loc[common, "centroid_x"] - b.loc[common, "centroid_x"], a.loc[common, "centroid_z"] - b.loc[common, "centroid_z"])
    bad = int((d > 0.5).sum())
    msg = f"processed buildings.geojson: {len(pg)} ids, {len(common)} shared, {bad} with moved centroids"
    if bad or len(common) != len(a):
        log(f"WARNING {msg}")
    return msg


def align_to_processed(bdf: gpd.GeoDataFrame, tol_m: float = 1.5) -> tuple[gpd.GeoDataFrame, dict[int, dict[str, Any]]]:
    """Make data/processed/buildings.geojson the single source of building ids.

    The local footprint prep can differ slightly from the pipeline run that wrote buildings.geojson
    (lidar-only buildings, hero replacement order), so building ids could drift. Rows are matched to
    processed buildings by centroid (within ``tol_m``) and take the processed id; processed buildings
    with no match (e.g. lidar-only houses) are appended from the processed geometry, and their lidar
    roof fields are returned so they get real roofs. No-op when processed data is missing or stale.
    """
    proc = processed_dir()
    meta_p, geo_p = proc / "region_meta.json", proc / "buildings.geojson"
    if not (meta_p.exists() and geo_p.exists()):
        return bdf, {}
    meta = json.loads(meta_p.read_text())
    bbox = region()["bbox"]
    if any(abs(float(meta["bbox"][k]) - float(bbox[k])) > 1e-9 for k in ("south", "north", "west", "east")):
        return bdf, {}
    from scipy.spatial import cKDTree

    pg = gpd.read_file(geo_p).to_crs("EPSG:32611")
    pg = pg[pg.get("height_source", pd.Series(index=pg.index, dtype=object)).astype(str) != "hero"]
    tree = cKDTree(np.c_[pg["centroid_x"].to_numpy(), pg["centroid_z"].to_numpy()])
    d, j = tree.query(np.c_[bdf["centroid_x"].to_numpy(), bdf["centroid_z"].to_numpy()], distance_upper_bound=tol_m)
    hero = (bdf["height_rule"].astype(str) == "hero").to_numpy() if "height_rule" in bdf.columns else np.zeros(len(bdf), bool)
    ok = np.isfinite(d) & ~hero
    pid = pg["id"].to_numpy()
    out = bdf.copy()
    ids = out["id"].to_numpy().copy()
    ids[ok] = pid[j[ok]]
    out["id"] = ids
    # one row per processed id (keep the closest), heroes keep their own ids
    out["_d"] = np.where(ok, d, 0.0)
    keep = hero | ok
    dropped = int((~keep).sum())
    out = out[keep].sort_values("_d").drop_duplicates("id").drop(columns="_d")
    have = set(int(i) for i in out["id"])
    add = pg[~pg["id"].isin(have)].copy()
    extra: dict[int, dict[str, Any]] = {}
    if len(add):
        rows = gpd.GeoDataFrame({c: [None] * len(add) for c in out.columns if c != "geometry"}, geometry=add.geometry.to_numpy(), crs="EPSG:32611")
        for c in ("id", "type", "height_m", "base_elev_m", "levels", "address", "name", "area_m2", "centroid_x", "centroid_z",
                  "school_id", "tile", "source", "source_id"):
            if c in add.columns and c in rows.columns:
                rows[c] = add[c].to_numpy()
        rows["height_rule"] = "processed"
        rows["levels_est"] = add["levels"].fillna(1).to_numpy() if "levels" in add.columns else 1
        out = gpd.GeoDataFrame(pd.concat([out, rows], ignore_index=True), geometry="geometry", crs="EPSG:32611")
        for _, r in add.iterrows():
            props = {"eave_height_m": r.get("eave_height_m"), "ridge_height_m": r.get("ridge_height_m"),
                     "roof_type": r.get("roof_type"), "roof_pitch_deg": r.get("roof_pitch_deg"),
                     "ridge_azimuth_deg": r.get("ridge_azimuth_deg"), "quality": r.get("lidar_quality"),
                     "lidar_status": r.get("lidar_status")}
            if lidar_model(props):
                extra[int(r["id"])] = props
    log(f"buildings_hd: aligned to processed ids: {int(ok.sum())} matched, {dropped} local-only dropped, "
        f"{len(add)} processed-only added ({len(extra)} with lidar roofs)")
    return out, extra


# ---------------------------------------------------------------------------
# lidar
# ---------------------------------------------------------------------------


def load_lidar(bdf: gpd.GeoDataFrame) -> tuple[dict[int, dict[str, Any]], gpd.GeoDataFrame | None, str]:
    """Per-building lidar roof models keyed by our building id, plus lidar-only footprints.

    data/raw/lidar/buildings_roofs.parquet (pipeline/lidar_features.py) has one row per
    osm_buildings.geojson footprint with its centroid (scene x/z, UTM easting/northing). Join: centroid
    within 1 m of ours (same source footprint), else the lidar centroid inside our footprint."""
    from scipy.spatial import cKDTree

    ldir = raw_dir() / LIDAR_DIR_NAME
    roofs_p = ldir / "buildings_roofs.parquet"
    out: dict[int, dict[str, Any]] = {}
    missing = None
    if not roofs_p.exists():
        return out, None, "lidar: buildings_roofs.parquet not available (tag / heuristic roofs)"
    df = pd.read_parquet(roofs_p)
    if "lidar_status" in df.columns:
        df = df[df["lidar_status"].astype(str) == "present"]
    if "quality" in df.columns:  # poor = a handful of points (RVs, sheds, noise): tags / heuristics instead
        df = df[df["quality"].astype(str).isin(["good", "fair"])]
    df = df.reset_index(drop=True)
    rows = df.to_dict("records")
    n_cent = n_sp = 0
    cx_col, cz_col = ("scene_x", "scene_z") if "scene_x" in df.columns else ("centroid_x", "centroid_z")
    if {cx_col, cz_col} <= set(df.columns) and len(df):
        tree = cKDTree(np.column_stack([df[cx_col].to_numpy(), df[cz_col].to_numpy()]))
        d, k = tree.query(np.column_stack([bdf["centroid_x"].to_numpy(), bdf["centroid_z"].to_numpy()]), distance_upper_bound=1.0)
        for bid, dd, kk in zip(bdf["id"].to_numpy(), d, k, strict=True):
            if np.isfinite(dd):
                out[int(bid)] = rows[int(kk)]
                n_cent += 1
    pts = _lidar_points(df)
    if pts is not None and len(df):
        rest = bdf[~bdf["id"].isin(list(out))]
        j = gpd.sjoin(gpd.GeoDataFrame({"_r": np.arange(len(df))}, geometry=pts, crs="EPSG:32611"), rest[["id", "geometry"]],
                      predicate="within", how="inner")
        for ri, bid in zip(j["_r"].to_numpy(), j["id"].to_numpy(), strict=True):
            if int(bid) not in out:
                out[int(bid)] = rows[int(ri)]
                n_sp += 1
    mp = ldir / "missing_buildings.geojson"
    if mp.exists():
        missing = gpd.read_file(mp).to_crs("EPSG:32611")
    return out, missing, f"lidar: {len(df)} present roof models; joined {n_cent} by centroid, {n_sp} spatially"


def _lidar_points(df: pd.DataFrame) -> gpd.GeoSeries | None:
    cols = set(df.columns)
    if "geometry" in cols:
        g = shapely.from_wkb(df["geometry"].to_numpy()) if isinstance(df["geometry"].iloc[0], (bytes, bytearray)) else df["geometry"]
        gs = gpd.GeoSeries(g, crs="EPSG:4326")
        if gs.total_bounds[0] > 1000:  # already projected
            gs = gpd.GeoSeries(g, crs="EPSG:32611")
        return gpd.GeoSeries(gs.to_crs("EPSG:32611").representative_point(), crs="EPSG:32611")
    if {"lon", "lat"} <= cols:
        return gpd.GeoSeries(gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326").to_crs("EPSG:32611")
    if {"centroid_x", "centroid_z"} <= cols:
        o = scene_origin()
        return gpd.GeoSeries(gpd.points_from_xy(df["centroid_x"] + o.easting, o.northing - df["centroid_z"]), crs="EPSG:32611")
    return None


def lidar_model(r: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one lidar row to the spec fields (None when unusable)."""

    def f(*names: str) -> float | None:
        for n in names:
            v = r.get(n)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    eave = f("eave_h", "eave_height_m", "eave_m")
    ridge = f("ridge_h", "ridge_height_m", "ridge_m", "max_h")
    if eave is None and ridge is None:
        return None
    rt = str(r.get("roof_type") or "").lower() or None
    if rt in ("unknown", "none", "nan"):
        rt = None
    quality = r.get("quality")
    if str(quality) not in ("good", "fair"):
        rt = None  # too few / noisy points: keep the heights, not the shape
    planes = r.get("planes") if r.get("planes") is not None else r.get("planes_json")
    if isinstance(planes, str):
        try:
            planes = json.loads(planes)
        except json.JSONDecodeError:
            planes = None
    elif planes is not None and not isinstance(planes, list):
        try:
            planes = list(planes)
        except TypeError:
            planes = None
    if planes:
        planes = [_jsonable(p) for p in planes]
    m = {
        "eave_h": eave,
        "ridge_h": ridge,
        "roof_type": rt,
        "ridge_az_deg": f("ridge_azimuth_deg", "ridge_az_deg", "ridge_azimuth"),
        "pitch_deg": f("roof_pitch_deg", "pitch_deg", "pitch"),
        "planes": planes or None,
        "quality": _jsonable(quality),
        "levels_lidar": f("levels", "n_levels"),
        "chimney": bool(r.get("chimney")) if r.get("chimney") is not None else None,
        "second_level": _jsonable(r.get("levels_split") or r.get("second_level")),
        "center": [float(r["easting"]), float(r["northing"])] if r.get("easting") is not None and r.get("northing") is not None else None,
    }
    return m


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "tolist"):
        return _jsonable(v.tolist())
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


class NdsmSampler:
    """Per-footprint height grids (m above ground, 1 m cells) from the lidar nDSM (lidar_features.py)."""

    def __init__(self, path: Path):
        import rasterio

        self.ds = rasterio.open(path)
        self.o = scene_origin()

    def grid(self, poly_utm: Polygon, step: float = 1.0) -> dict[str, Any] | None:
        from rasterio.windows import from_bounds

        minx, miny, maxx, maxy = poly_utm.bounds
        xs = np.arange(minx + step / 2, maxx, step)
        ys = np.arange(miny + step / 2, maxy, step)
        if len(xs) < 2 or len(ys) < 2:
            return None
        win = from_bounds(minx - 1, miny - 1, maxx + 1, maxy + 1, self.ds.transform).round_offsets().round_lengths()
        a = self.ds.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
        tr = self.ds.window_transform(win)
        X, Y = np.meshgrid(xs, ys)
        col = np.clip(((X - tr.c) / tr.a).astype(int), 0, a.shape[1] - 1)
        row = np.clip(((Y - tr.f) / tr.e).astype(int), 0, a.shape[0] - 1)
        h = a[row, col]
        inside = shapely.contains_xy(poly_utm.buffer(-0.3), X, Y)
        q = np.where(inside & np.isfinite(h), np.round(np.clip(h, 0, 99) * 10), -1).astype(int)
        if (q >= 0).sum() < 4:
            return None
        return {"x0": round(float(xs[0] - self.o.easting), 2), "y0": round(float(ys[0] - self.o.northing), 2), "step": step,
                "nx": len(xs), "ny": len(ys), "h": q.ravel().tolist()}


# ---------------------------------------------------------------------------
# streets
# ---------------------------------------------------------------------------


def load_streets() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """(streets, driveways) as scene-coordinate LineStrings with `cls`."""
    import pyarrow.parquet as pq

    p = raw_dir() / "overture" / "segment.parquet"
    o = scene_origin()
    if p.exists():
        t = pq.read_table(p, columns=["geometry", "class", "subclass", "subtype"]).to_pandas()
        t = t[t["subtype"] == "road"]
        g = gpd.GeoSeries(shapely.from_wkb(t["geometry"].to_numpy()), crs="EPSG:4326").to_crs("EPSG:32611")
        df = gpd.GeoDataFrame({"cls": t["class"].astype(str).to_numpy(), "sub": t["subclass"].fillna("").astype(str).to_numpy()}, geometry=g.to_numpy(), crs="EPSG:32611")
    else:
        rp = processed_dir() / "roads.geojson"
        if not rp.exists():
            raise FileNotFoundError(f"no road source: {p} or {rp} (run pipeline/fetch_aws.py / build_all.py)")
        r = gpd.read_file(rp).to_crs("EPSG:32611")
        df = gpd.GeoDataFrame({"cls": r["highway"].astype(str).str.replace("_link", ""), "sub": ""}, geometry=r.geometry, crs="EPSG:32611")
    df["geometry"] = shapely.transform(df.geometry.to_numpy(), lambda c: np.column_stack([c[:, 0] - o.easting, o.northing - c[:, 1]]))
    drive = df[df["sub"] == "driveway"].reset_index(drop=True)
    st = df[df["cls"].isin(FRONT_CLASSES) & ~df["sub"].isin(SERVICE_SKIP_SUBCLASS | {"driveway"})].reset_index(drop=True)
    return st, drive


def street_context(polys: list[Polygon], streets: gpd.GeoDataFrame, drives: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    """Per footprint (scene coords): nearest street points by class, and a driveway end if one touches it."""
    tree = shapely.STRtree(streets.geometry.to_numpy())
    dtree = shapely.STRtree(drives.geometry.to_numpy()) if len(drives) else None
    sg = streets.geometry.to_numpy()
    scls = streets["cls"].to_numpy()
    out = []
    for poly in polys:
        idx = tree.query(poly.buffer(STREET_SEARCH_M))
        cands = []
        for k in idx:
            d = float(sg[k].distance(poly))
            if d > STREET_SEARCH_M:
                continue
            p_on = shapely.ops.nearest_points(sg[k], poly)[0]
            cands.append((d, str(scls[k]), round(p_on.x, 2), round(p_on.y, 2)))
        cands.sort()
        # keep one point per distinct direction (corner lots): drop points within 25 deg of a closer one
        cx, cz = poly.centroid.x, poly.centroid.y
        kept: list[tuple[float, str, float, float]] = []
        for c in cands:
            a = math.atan2(c[3] - cz, c[2] - cx)
            if all(abs((a - math.atan2(k[3] - cz, k[2] - cx) + math.pi) % (2 * math.pi) - math.pi) > math.radians(25) for k in kept):
                kept.append(c)
            if len(kept) >= 3:
                break
        drive = None
        if dtree is not None:
            for k in dtree.query(poly.buffer(6.0)):
                ln = drives.geometry.iloc[int(k)]
                ends = [shapely.Point(ln.coords[0]), shapely.Point(ln.coords[-1])]
                e = min(ends, key=lambda q: q.distance(poly))
                if e.distance(poly) < 6.0:
                    far = ends[1] if e is ends[0] else ends[0]
                    drive = [round(e.x, 2), round(e.y, 2), round(far.x, 2), round(far.y, 2)]
                    break
        out.append({"streets": [{"d": round(c[0], 2), "cls": c[1], "x": c[2], "z": c[3]} for c in kept], "driveway": drive})
    return out


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------


def _rgb(v: Any) -> list[int] | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return [int(x) for x in v][:3]
    except TypeError:
        return None


def scene_poly(poly: Polygon) -> Polygon:
    p = bb.utm_poly_to_scene(poly)
    # scene (x, z) with z south: a CCW ring in (east, north) is CW in (x, z). Store CCW in (x, -z),
    # i.e. orient in UTM then map; Blender side works in (x, y=-z) where it is CCW again.
    return p


def heuristic_roof(btype: str, poly: Polygon, height_m: float, rule: str, roof_shape: str | None, levels: int | None) -> dict[str, Any]:
    """Roof model without lidar. Heights above the finished floor."""
    area = poly.area
    shape = roof_shape.lower() if isinstance(roof_shape, str) else ""
    rect = bb.rectangularity(poly)
    if btype in ("commercial", "school") or (btype == "other" and area > 400):
        rtype = "flat"
    elif btype == "other":
        rtype = "flat" if area < 60 or shape in ("flat", "skillion") else "gable"
    else:
        rtype = {"gabled": "gable", "gable": "gable", "hipped": "hip", "hip": "hip", "flat": "flat", "skillion": "shed"}.get(shape, "hip")
    lv = levels
    pitch = 22.6 if btype == "house" else 20.0
    if rtype == "flat":
        eave = height_m if rule in ("height", "levels") else {"commercial": 7.5, "school": 7.0, "apartments": 10.0, "other": 3.6}.get(btype, 6.0)
        return {"eave_h": round(max(eave, 2.6), 2), "ridge_h": round(max(eave, 2.6), 2), "roof_type": "flat", "pitch_deg": 0.0, "levels": lv, "rect": rect}
    # pitched: the height tag is the total height (OSM convention) -> eave = height - rise
    w = _min_width(poly)
    rise = 0.5 * w * math.tan(math.radians(pitch))
    rise = min(rise, 4.2)
    if btype == "house":
        if lv is None:
            lv = 2 if area >= 140 else 1
        eave_default = 2.75 + 2.95 * (lv - 1) + 0.3
    elif btype == "apartments":
        lv = lv or 3
        eave_default = 3.0 * lv + 0.3
    else:
        lv = lv or 1
        eave_default = 3.0
    eave = eave_default
    if rule in ("height", "levels") and height_m > 0:
        # Overture / OSM height tags here read ~TAG_RIDGE_OFFSET_M below the lidar ridge (lidar_validation.json):
        # treat a tag as a mid-roof height, ridge = tag + offset, eave = ridge - rise
        e2 = height_m + TAG_RIDGE_OFFSET_M - rise if rule == "height" else height_m
        # tags that put a house eave under 2.4 m are total heights of the wrong thing; keep the default
        if e2 >= 2.4:
            eave = e2
            if btype == "house" and rule == "height":
                lv = 1 if eave < 4.4 else (2 if eave < 7.5 else 3)
    return {"eave_h": round(eave, 2), "ridge_h": round(eave + rise, 2), "roof_type": rtype, "pitch_deg": pitch, "levels": lv, "rect": rect}


def _min_width(poly: Polygon) -> float:
    r = poly.minimum_rotated_rectangle
    c = np.asarray(r.exterior.coords)
    a = float(np.hypot(*(c[1] - c[0])))
    b = float(np.hypot(*(c[2] - c[1])))
    return min(a, b)


def model_params() -> dict[str, Any]:
    """assumptions.yaml values for blender/buildings/model.py (DEFAULT_PARAMS keys)."""
    a = "buildings_hd."
    keys = ["seed", "overhang_m", "fascia_m", "story_m", "slab_m", "parapet_m", "garage_single_w_m", "garage_recess_m",
            "entry_door_w_m", "entry_door_h_m", "two_level_share", "solar_share", "stone_wainscot_share", "hvac_m2_per_unit"]
    p: dict[str, Any] = {k: assumption(a + k) for k in keys}
    p["garage_door_w_m"] = float(assumption("streets.garage_door_width_m"))
    p["garage_door_h_m"] = float(assumption("streets.garage_door_height_m"))
    p["wall_palette"] = [[c["rgb"], c["share"]] for c in assumption(a + "wall_palette")]
    p["trim_palette"] = [[c["rgb"], c["share"]] for c in assumption(a + "trim_palette")]
    p["tile_variants"] = dict(assumption("building_style.tile_roof_variants"))
    return p


def build_specs(out_dir: Path = SPEC_DIR) -> dict[str, Any]:
    t0 = time.time()
    dem = DemSampler(raw_dir() / "dem_3dep_10m.tif")
    bdf, heroes = pipeline_buildings(dem)
    bdf, extra_lidar = align_to_processed(bdf)
    log(f"buildings_hd: {len(bdf):,} pipeline buildings ({time.time() - t0:.0f}s)")
    id_note = check_processed_ids(bdf)
    hero_ids = set(int(i) for i in bdf.loc[bdf["height_rule"] == "hero", "id"]) if "height_rule" in bdf.columns else set()
    lidar, missing, lidar_note = load_lidar(bdf)
    log(f"buildings_hd: {lidar_note}")
    for bid, props in extra_lidar.items():
        lidar.setdefault(bid, props)
    grid = tile_grid()
    rows = bdf[~bdf["id"].isin(hero_ids)].copy()
    rows["source_kind"] = "osm"
    if missing is not None and len(missing):
        miss = missing[missing.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].explode(index_parts=False)
        if "quality" in miss.columns:  # poor lidar-only objects are RVs, sheds, carports
            miss = miss[miss["quality"].astype(str).isin(["good", "fair"])]
        # drop lidar-only footprints overlapping ours (already modeled) or inside hero radii
        ov = gpd.sjoin(miss, rows[["geometry"]], predicate="intersects", how="left")
        miss = miss.loc[~miss.index.isin(ov.index[ov["index_right"].notna()])]
        o = scene_origin()
        c = miss.geometry.centroid
        cx, cz = c.x.to_numpy() - o.easting, o.northing - c.y.to_numpy()
        keep = grid.extent.contains(cx, cz)
        for h in heroes:
            keep &= np.hypot(cx - h.x, cz - h.z) > h.radius
        miss, cx, cz = miss[keep].copy(), cx[keep], cz[keep]
        start = int(bdf["id"].max()) + 1
        miss["id"] = np.arange(start, start + len(miss), dtype=np.int64)
        r, cc = grid.tile_of(cx, cz)
        miss["tile"] = [grid.tile_id(int(a), int(b)) for a, b in zip(r, cc, strict=True)]
        miss["centroid_x"], miss["centroid_z"] = cx, cz
        miss["area_m2"] = miss.geometry.area
        miss["type"] = [("house" if 60 <= a < 600 else "commercial" if a >= 600 else "other") for a in miss["area_m2"]]
        miss["height_m"], miss["height_rule"], miss["levels"], miss["name"] = 0.0, "lidar", None, None
        miss["source_kind"] = "lidar_missing"
        for col in ("roof_shape", "wall_rgb", "roof_rgb", "building_tag", "address"):
            miss[col] = None
        # roof models for missing buildings come from the same lidar table if present (spatial), else
        # from height columns on the geojson itself
        for _i, row in miss.iterrows():
            props = {k: row.get(k) for k in miss.columns if k != "geometry"}
            m = lidar_model(props)
            if m:
                lidar[int(row["id"])] = props
        rows = pd.concat([rows, miss[[c for c in rows.columns if c in miss.columns]]], ignore_index=True)
        rows = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32611")
        log(f"buildings_hd: +{len(miss)} lidar-only buildings (ids {start}..{start + len(miss) - 1})")

    scene_polys = [orient(scene_poly(p), 1.0) for p in rows.geometry]
    utm_polys = list(rows.geometry)
    ndsm_p = raw_dir() / LIDAR_DIR_NAME / "ndsm_0p5m.tif"
    ndsm = NdsmSampler(ndsm_p) if (ndsm_p.exists() and lidar) else None
    o_ = scene_origin()
    streets, drives = load_streets()
    ctx = street_context(scene_polys, streets, drives)
    log(f"buildings_hd: street context for {len(ctx):,} buildings ({time.time() - t0:.0f}s)")

    tiles: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for k, (rec, poly, sc, putm) in enumerate(zip(rows.to_dict("records"), scene_polys, ctx, utm_polys, strict=True)):
        bid = int(rec["id"])
        ring = np.asarray(poly.exterior.coords)[:-1]
        holes = [np.asarray(h.coords)[:-1] for h in poly.interiors]
        gz = dem.sample(ring[:, 0], ring[:, 1])
        cy = float(dem.sample(poly.centroid.x, poly.centroid.y))
        floor = float(np.median(np.append(gz, cy)))
        btype = str(rec.get("type") or "other")
        lv_tag = rec.get("levels")
        lv_tag = None if lv_tag is None or (isinstance(lv_tag, float) and math.isnan(lv_tag)) else int(lv_tag)
        lm = lidar_model(lidar[bid]) if bid in lidar else None
        heur = heuristic_roof(btype, poly, float(rec.get("height_m") or 0.0), str(rec.get("height_rule") or "default"), rec.get("roof_shape"), lv_tag)
        if lm is not None and (lm.get("eave_h") or lm.get("ridge_h")):
            eave = lm["eave_h"] if lm["eave_h"] is not None else (lm["ridge_h"] or 0) - 2.5
            ridge = lm["ridge_h"] if lm["ridge_h"] is not None else eave
            rtype = lm["roof_type"] or heur["roof_type"]
            if rtype in ("complex", "mixed"):
                rtype = "complex"
            source = "lidar"
            pu = putm if putm.geom_type == "Polygon" else max(putm.geoms, key=lambda g: g.area)
            cu = pu.centroid
            roof = {"eave_h": round(float(eave), 2), "ridge_h": round(float(max(ridge, eave)), 2), "roof_type": rtype,
                    "pitch_deg": lm["pitch_deg"] if lm["pitch_deg"] is not None else heur["pitch_deg"],
                    "ridge_az_deg": lm["ridge_az_deg"], "planes": lm["planes"], "quality": lm["quality"],
                    "chimney": lm["chimney"], "second_level": lm["second_level"],
                    "lidar_center": [round(cu.x - o_.easting, 3), round(cu.y - o_.northing, 3)]}
            if ndsm is not None:
                roof["hgrid"] = ndsm.grid(pu)
            lv = lv_tag or (int(lm["levels_lidar"]) if lm.get("levels_lidar") else None)
            if lv is None and btype in ("house", "apartments"):
                lv = 1 if eave < 4.4 and ridge < 6.6 else (2 if eave < 7.6 else 3)
        else:
            source = "tag" if rec.get("height_rule") in ("height", "levels") else "heuristic"
            roof = {k: heur[k] for k in ("eave_h", "ridge_h", "roof_type", "pitch_deg")}
            roof["ridge_az_deg"] = None
            lv = heur["levels"]
        if lv is None:
            lv = max(1, int(round(max(roof["eave_h"] - 0.3, 2.8) / 2.95)))
        counts[source] = counts.get(source, 0) + 1
        spec = {
            "id": bid,
            "type": btype,
            "tag": rec.get("building_tag"),
            "name": rec.get("name") if isinstance(rec.get("name"), str) else None,
            "ring": np.round(ring, 3).tolist(),
            "holes": [np.round(h, 3).tolist() for h in holes],
            "floor_y": round(floor, 3),
            "ground_min_y": round(float(gz.min()), 3),
            "ground_max_y": round(float(gz.max()), 3),
            "levels": int(lv),
            "roof_source": source,
            "origin": rec.get("source_kind", "osm"),
            "height_tag_m": float(rec["height_m"]) if rec.get("height_rule") == "height" else None,
            "wall_rgb": _rgb(rec.get("wall_rgb")),
            "roof_rgb": _rgb(rec.get("roof_rgb")),
            **roof,
            **sc,
        }
        tiles.setdefault(str(rec["tile"]), []).append(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("r*_c*.json"):
        old.unlink()
    for tid, specs in tiles.items():
        (out_dir / f"{tid}.json").write_text(json.dumps({"tile": tid, "buildings": specs}, separators=(",", ":")))
    b = grid.extent
    index = {
        "format": "redraw-buildings-hd-specs",
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bbox": region()["bbox"],
        "origin": {"easting": scene_origin().easting, "northing": scene_origin().northing},
        "grid": {"rows": grid.rows, "cols": grid.cols, "extent": {"min_x": b.min_x, "max_x": b.max_x, "min_z": b.min_z, "max_z": b.max_z}},
        "tiles": {t: len(v) for t, v in sorted(tiles.items())},
        "count": int(sum(len(v) for v in tiles.values())),
        "roof_source_counts": counts,
        "params": model_params(),
        "hero_ids": sorted(hero_ids),
        "heroes": [{"id": h.key, "x": h.x, "z": h.z, "radius": h.radius, "building_id": h.building_id} for h in heroes],
        "notes": {"ids": id_note, "lidar": lidar_note},
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    log(f"buildings_hd: wrote {index['count']:,} specs in {len(tiles)} tiles to {out_dir} ({time.time() - t0:.0f}s); sources {counts}")
    return index


# ---------------------------------------------------------------------------
# preview inputs (terrain, imagery, roads around a view center) for blender/buildings/preview.py
# ---------------------------------------------------------------------------

PREVIEW_DIR = REPO / "blender" / "build" / "buildings_hd" / "preview"
# view centers (scene x, z): a 4S Ranch tract block and a Del Sur tract block
PREVIEW_VIEWS = {"4s_ranch": (1375.0, -775.0), "del_sur": (-1625.0, 225.0)}


def preview_data(name: str, cx: float, cz: float, radius: float = 320.0, step: float = 1.0) -> Path:
    """DEM heights + NAIP colors on a `step` m grid and road centerlines around (cx, cz) -> npz + json."""
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject

    o = scene_origin()
    n = int(round(2 * radius / step)) + 1
    xs = cx - radius + step * np.arange(n)
    zs = cz - radius + step * np.arange(n)
    X, Z = np.meshgrid(xs, zs)  # row = z (south), col = x (east)
    dem = DemSampler(raw_dir() / "dem_3dep_10m.tif")
    H = dem.sample(X.ravel(), Z.ravel()).reshape(n, n).astype(np.float32)
    # NAIP resampled to the same grid (row 0 = north = min z)
    dst = np.zeros((3, n, n), np.uint8)
    naip_p = raw_dir() / "naip_mosaic.tif"
    if naip_p.exists():
        tr = Affine(step, 0, o.easting + xs[0] - step / 2, 0, -step, o.northing - zs[0] + step / 2)
        with rasterio.open(naip_p) as ds:
            for b in range(3):
                reproject(rasterio.band(ds, b + 1), dst[b], dst_transform=tr, dst_crs="EPSG:32611", resampling=Resampling.cubic)
    streets, drives = load_streets()
    area = shapely.box(cx - radius - 50, cz - radius - 50, cx + radius + 50, cz + radius + 50)
    roads = []
    for df, kind in ((streets, None), (drives, "driveway")):
        sub = df[df.intersects(area)]
        for g, cls, sb in zip(sub.geometry, sub["cls"], sub["sub"], strict=True):
            g = g.intersection(area)
            for ln in getattr(g, "geoms", [g]):
                if ln.geom_type != "LineString" or ln.length < 1:
                    continue
                roads.append({"cls": kind or str(cls), "sub": str(sb), "pts": np.round(np.asarray(ln.coords), 2).tolist()})
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREVIEW_DIR / f"{name}.npz", height=H, naip=dst.transpose(1, 2, 0), x0=xs[0], z0=zs[0], step=step)
    (PREVIEW_DIR / f"{name}.json").write_text(json.dumps({"name": name, "center": [cx, cz], "radius": radius, "roads": roads}))
    log(f"preview data {name}: {n}x{n} grid, {len(roads)} road lines")
    return PREVIEW_DIR / f"{name}.npz"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=SPEC_DIR)
    ap.add_argument("--preview-data", action="store_true", help="also write terrain / imagery / roads for the preview views")
    ap.add_argument("--preview-only", action="store_true", help="only the preview inputs")
    a = ap.parse_args(argv)
    if not a.preview_only:
        build_specs(a.out)
    if a.preview_data or a.preview_only:
        for nm, (x, z) in PREVIEW_VIEWS.items():
            preview_data(nm, x, z)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
