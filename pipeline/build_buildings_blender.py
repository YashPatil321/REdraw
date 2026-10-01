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
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline import build_buildings as bb  # noqa: E402
from pipeline.common import log, tile_grid  # noqa: E402
from pipeline.config import load_yaml, processed_dir, raw_dir, region  # noqa: E402
from pipeline.geo import scene_origin  # noqa: E402

SPEC_DIR = REPO / "blender" / "build" / "buildings_hd" / "specs"
LIDAR_DIR_NAME = "lidar"
# roads that houses face (Overture segment classes); service = private streets in gated tracts / alleys
FRONT_CLASSES = {"residential", "living_street", "unclassified", "tertiary", "secondary", "primary", "service"}
ARTERIAL = {"primary", "secondary", "trunk", "motorway"}
SERVICE_SKIP_SUBCLASS = {"parking_aisle"}
STREET_SEARCH_M = 60.0


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


# ---------------------------------------------------------------------------
# lidar
# ---------------------------------------------------------------------------


def load_lidar(bdf: gpd.GeoDataFrame) -> tuple[dict[int, dict[str, Any]], gpd.GeoDataFrame | None, str]:
    """Per-building lidar roof models keyed by our building id, plus missing footprints.

    Join: by `building_id` when the lidar table carries ids of this build, else by OSM
    sort key, else spatially (lidar point / footprint inside our footprint)."""
    ldir = raw_dir() / LIDAR_DIR_NAME
    roofs_p = ldir / "buildings_roofs.parquet"
    out: dict[int, dict[str, Any]] = {}
    missing = None
    if not roofs_p.exists():
        return out, None, "lidar: buildings_roofs.parquet not available (tag / heuristic roofs)"
    df = pd.read_parquet(roofs_p)
    cols = set(df.columns)
    rows = df.to_dict("records")
    key_note = ""
    if "building_id" in cols and "centroid_x" in cols:
        # ids of a build: trust them only if the centroids agree with ours
        ours = bdf.set_index("id")[["centroid_x", "centroid_z"]]
        ok = 0
        for r in rows:
            bid = int(r["building_id"])
            if bid in ours.index and math.hypot(ours.at[bid, "centroid_x"] - float(r["centroid_x"]), ours.at[bid, "centroid_z"] - float(r["centroid_z"])) < 2.0:
                ok += 1
        if ok >= 0.9 * len(rows):
            for r in rows:
                out[int(r["building_id"])] = r
            key_note = "building_id"
    if not out:
        # spatial join: a lidar point (centroid lon/lat or x/z) inside our footprint
        pts = _lidar_points(df)
        if pts is not None:
            j = gpd.sjoin(gpd.GeoDataFrame({"_r": np.arange(len(df))}, geometry=pts, crs="EPSG:32611"), bdf[["id", "geometry"]], predicate="within", how="inner")
            for ri, bid in zip(j["_r"].to_numpy(), j["id"].to_numpy(), strict=True):
                out.setdefault(int(bid), rows[int(ri)])
            key_note = "spatial (lidar centroid within footprint)"
    mp = ldir / "missing_buildings.geojson"
    if mp.exists():
        missing = gpd.read_file(mp).to_crs("EPSG:32611")
    return out, missing, f"lidar: {len(df)} roof models, {len(out)} joined by {key_note or 'nothing'}"


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
    quality = r.get("quality")
    planes = r.get("planes")
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
        "pitch_deg": f("pitch_deg", "pitch"),
        "planes": planes or None,
        "quality": _jsonable(quality),
        "levels_lidar": f("levels", "n_levels"),
        "chimney": bool(r.get("chimney")) if r.get("chimney") is not None else None,
        "second_level": _jsonable(r.get("levels_split") or r.get("second_level")),
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
        e2 = height_m - rise if rule == "height" else height_m
        # tags that put a house eave under 2.4 m are total heights of the wrong thing; keep the default
        if e2 >= 2.4:
            eave = e2
    return {"eave_h": round(eave, 2), "ridge_h": round(eave + rise, 2), "roof_type": rtype, "pitch_deg": pitch, "levels": lv, "rect": rect}


def _min_width(poly: Polygon) -> float:
    r = poly.minimum_rotated_rectangle
    c = np.asarray(r.exterior.coords)
    a = float(np.hypot(*(c[1] - c[0])))
    b = float(np.hypot(*(c[2] - c[1])))
    return min(a, b)


def build_specs(out_dir: Path = SPEC_DIR) -> dict[str, Any]:
    t0 = time.time()
    dem = DemSampler(raw_dir() / "dem_3dep_10m.tif")
    bdf, heroes = pipeline_buildings(dem)
    log(f"buildings_hd: {len(bdf):,} pipeline buildings ({time.time() - t0:.0f}s)")
    id_note = check_processed_ids(bdf)
    hero_ids = set(int(i) for i in bdf.loc[bdf["height_rule"] == "hero", "id"]) if "height_rule" in bdf.columns else set()
    lidar, missing, lidar_note = load_lidar(bdf)
    log(f"buildings_hd: {lidar_note}")
    grid = tile_grid()
    rows = bdf[~bdf["id"].isin(hero_ids)].copy()
    rows["source_kind"] = "osm"
    if missing is not None and len(missing):
        miss = missing[missing.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].explode(index_parts=False)
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
        for i, row in miss.iterrows():
            props = {k: row.get(k) for k in miss.columns if k != "geometry"}
            m = lidar_model(props)
            if m:
                lidar[int(row["id"])] = props
        rows = pd.concat([rows, miss[[c for c in rows.columns if c in miss.columns]]], ignore_index=True)
        rows = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32611")
        log(f"buildings_hd: +{len(miss)} lidar-only buildings (ids {start}..{start + len(miss) - 1})")

    scene_polys = [orient(scene_poly(p), 1.0) for p in rows.geometry]
    streets, drives = load_streets()
    ctx = street_context(scene_polys, streets, drives)
    log(f"buildings_hd: street context for {len(ctx):,} buildings ({time.time() - t0:.0f}s)")

    tiles: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for k, (rec, poly, sc) in enumerate(zip(rows.to_dict("records"), scene_polys, ctx, strict=True)):
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
            roof = {"eave_h": round(float(eave), 2), "ridge_h": round(float(max(ridge, eave)), 2), "roof_type": rtype,
                    "pitch_deg": lm["pitch_deg"] if lm["pitch_deg"] is not None else heur["pitch_deg"],
                    "ridge_az_deg": lm["ridge_az_deg"], "planes": lm["planes"], "quality": lm["quality"],
                    "chimney": lm["chimney"], "second_level": lm["second_level"]}
            lv = lv_tag or (int(lm["levels_lidar"]) if lm.get("levels_lidar") else None)
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
        "hero_ids": sorted(hero_ids),
        "heroes": [{"id": h.key, "x": h.x, "z": h.z, "radius": h.radius, "building_id": h.building_id} for h in heroes],
        "notes": {"ids": id_note, "lidar": lidar_note},
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    log(f"buildings_hd: wrote {index['count']:,} specs in {len(tiles)} tiles to {out_dir} ({time.time() - t0:.0f}s); sources {counts}")
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=SPEC_DIR)
    a = ap.parse_args(argv)
    build_specs(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
