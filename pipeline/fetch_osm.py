"""Fetch OpenStreetMap data via OSMnx / Overpass (spec 5): drive, walk and bike networks,
building footprints and amenity=school features. Cached in data/raw/ (skipped if present).

    python pipeline/fetch_osm.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd

from pipeline.common import DataSourceUnavailable, log
from pipeline.config import assumption, raw_dir, region
from pipeline.geo import latlon_to_scene

FILES = {
    "drive": "osm_drive_raw.graphml",
    "walk": "osm_walk.graphml",
    "bike": "osm_bike.graphml",
    "buildings": "osm_buildings.geojson",
    "schools": "osm_schools.geojson",
}
EXTRA_WAY_TAGS = ["lanes:forward", "lanes:backward", "turn:lanes", "sidewalk", "cycleway"]


def _bbox() -> tuple[float, float, float, float]:
    b = region()["bbox"]
    return (b["west"], b["south"], b["east"], b["north"])  # OSMnx 2.x order: left, bottom, right, top


def _how(name: str) -> str:
    w, s, e, n = _bbox()
    return (
        f"Run `python pipeline/fetch_osm.py` on a machine that can reach the Overpass API and copy "
        f"data/raw/{name} here. Alternatively export the bbox (S {s}, W {w}, N {n}, E {e}) from "
        "https://overpass-turbo.eu (buildings / amenity=school as GeoJSON) or a Geofabrik extract "
        "(https://download.geofabrik.de/north-america/us/california/socal.html) converted with OSMnx."
    )


def _overpass_url() -> str:
    import osmnx as ox

    return f"{ox.settings.overpass_url}/interpreter"


def _configure() -> None:
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(raw_dir() / "osmnx_cache")
    ox.settings.requests_timeout = 300
    for t in EXTRA_WAY_TAGS:
        if t not in ox.settings.useful_tags_way:
            ox.settings.useful_tags_way = [*ox.settings.useful_tags_way, t]
    if "highway" not in ox.settings.useful_tags_node:
        ox.settings.useful_tags_node = [*ox.settings.useful_tags_node, "highway"]


def fetch_graph(kind: str, raw: Path) -> Path:
    import osmnx as ox

    dest = raw / FILES[kind]
    if dest.exists():
        log(f"cached: {dest.name}")
        return dest
    _configure()
    log(f"OSM: downloading {kind} network for bbox {_bbox()}")
    try:
        # Drive graph stays unsimplified here so traffic_signals nodes survive; build_roads simplifies.
        G = ox.graph_from_bbox(_bbox(), network_type=kind, simplify=(kind != "drive"), retain_all=True, truncate_by_edge=True)
    except Exception as e:  # noqa: BLE001
        raise DataSourceUnavailable(f"OpenStreetMap {kind} network (Overpass)", _overpass_url(), dest, _how(FILES[kind]), e) from e
    ox.io.save_graphml(G, dest)
    log(f"OSM: {kind} graph {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges -> {dest.name}")
    return dest


def _to_geojson(gdf: gpd.GeoDataFrame, dest: Path) -> None:
    gdf = gdf.reset_index()
    for col in gdf.columns:
        if col == "geometry":
            continue
        if gdf[col].dtype == object:
            gdf[col] = gdf[col].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else (",".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)))
    gdf.to_file(dest, driver="GeoJSON")


def fetch_features(kind: str, tags: dict[str, Any], raw: Path) -> Path:
    import osmnx as ox

    dest = raw / FILES[kind]
    if dest.exists():
        log(f"cached: {dest.name}")
        return dest
    _configure()
    log(f"OSM: downloading {kind} features {tags}")
    try:
        gdf = ox.features_from_bbox(_bbox(), tags=tags)
    except Exception as e:  # noqa: BLE001
        raise DataSourceUnavailable(f"OpenStreetMap {kind} (Overpass)", _overpass_url(), dest, _how(FILES[kind]), e) from e
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon", "Point"])]
    _to_geojson(gdf, dest)
    log(f"OSM: {len(gdf):,} {kind} features -> {dest.name}")
    return dest


def fetch_all(raw: Path | None = None) -> dict[str, Path]:
    raw = raw or raw_dir()
    raw.mkdir(parents=True, exist_ok=True)
    return {
        "drive": fetch_graph("drive", raw),
        "walk": fetch_graph("walk", raw),
        "bike": fetch_graph("bike", raw),
        "buildings": fetch_features("buildings", {"building": True}, raw),
        "schools": fetch_features("schools", {"amenity": "school"}, raw),
    }


SCHOOL_NAME_RE = re.compile(r"school|academy|elementary|middle|high|montessori|preschool|campus|college", re.I)


def school_name_key(name: Any) -> str:
    """Normalized school name for matching ('Design39 Campus' == 'Design39Campus')."""
    import unicodedata

    n = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z0-9]", "", n)
    n = re.sub(r"^the", "", n)
    while n.endswith(("campus", "school", "schools")):
        n = re.sub(r"(campus|schools|school)$", "", n)
    return n


def match_campuses(gdf: gpd.GeoDataFrame, schools_cfg: list[dict[str, Any]]) -> list[str | None]:
    """schools.yaml id for each OSM school feature (EPSG:32611), or None.

    1. name match (normalized) always wins;
    2. otherwise a polygon gets the id of a schools.yaml point it contains, or the nearest one
       within pipeline.school_snap_max_dist_m, but only if the polygon is unnamed or its name
       is not another school's name (so a neighbouring campus is never captured);
    3. points (Overture places) match by name only.
    """
    from shapely.geometry import Point

    from pipeline.geo import lonlat_to_utm

    max_d = float(assumption("pipeline.school_snap_max_dist_m"))
    pts = [(str(s["id"]), Point(*lonlat_to_utm(float(s["lon"]), float(s["lat"]))), school_name_key(s["name"])) for s in schools_cfg]
    by_key = {k: i for i, _, k in pts}
    out: list[str | None] = []
    for name, g in zip(gdf["name"], gdf.geometry, strict=True):
        key = school_name_key(name)
        if key and key in by_key:
            out.append(by_key[key])
            continue
        if g.geom_type not in ("Polygon", "MultiPolygon"):
            out.append(None)
            continue
        # named after another school (not just "Maintenance" or similar): never captured
        named_other = bool(SCHOOL_NAME_RE.search(str(name or "")))
        inside = [i for i, pt, _ in pts if g.contains(pt)]
        if inside:
            out.append(None if named_other else inside[0])
            continue
        if named_other:
            out.append(None)
            continue
        best, bd = None, max_d
        for i, pt, _ in pts:
            d = g.distance(pt)
            if d <= bd:
                best, bd = i, d
        out.append(best)
    return out


def load_schools(path: Path, schools_cfg: list[dict[str, Any]]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """OSM amenity=school -> (all schools EPSG:32611 with a `school_id` column, campus polygons
    tagged with schools.yaml ids). Matching rules: `match_campuses`."""
    gdf = gpd.read_file(path).to_crs("EPSG:32611")
    if "name" not in gdf.columns:
        gdf["name"] = None
    gdf["school_id"] = match_campuses(gdf, schools_cfg)
    gdf["osm_added"] = False
    # OSM-only campuses (not in schools.yaml) become schools too (spec 5.5), but only campus
    # polygons inside the region bbox whose name gives a K-12 grade band (no preschools,
    # continuation or adult schools, no Overture place points).
    from pipeline.build_population import osm_school_grades, osm_school_id
    from pipeline.common import region_extent
    from pipeline.geo import scene_origin

    o = scene_origin()
    rext = region_extent()
    taken = {str(s["id"]) for s in schools_cfg}
    for i in gdf.index[gdf["school_id"].isna()]:
        g = gdf.geometry[i]
        name = gdf.at[i, "name"]
        if g.geom_type not in ("Polygon", "MultiPolygon") or not name or osm_school_grades(str(name)) is None:
            continue
        c = g.representative_point()
        if not rext.contains(c.x - o.easting, o.northing - c.y):
            continue
        sid = osm_school_id(str(name))
        if sid in taken and not (gdf["school_id"] == sid).any():
            continue
        gdf.at[i, "school_id"] = sid
        gdf.at[i, "osm_added"] = True
    areas = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]) & gdf["school_id"].notna()]
    areas = areas[["school_id", "name", "geometry"]].copy()
    log(f"schools: {int(gdf['osm_added'].sum())} OSM-only campuses added: {sorted(set(gdf.loc[gdf['osm_added'], 'school_id']))}")
    for s in schools_cfg:
        hit = areas[areas["school_id"] == s["id"]]
        if len(hit):
            log(f"schools: {s['id']} <- OSM campus {', '.join(repr(n) for n in hit['name'])} ({hit.geometry.area.sum():,.0f} m2)")
        else:
            log(f"WARNING schools: {s['id']} ({s['name']}) has no OSM campus polygon; buildings near its point are not tagged")
    return gdf, gpd.GeoDataFrame(areas, geometry="geometry", crs="EPSG:32611")


def school_scene_points(gdf: gpd.GeoDataFrame) -> pd.DataFrame:  # pragma: no cover - debug helper
    g = gdf.to_crs("EPSG:4326").geometry.representative_point()
    xz = [latlon_to_scene(p.y, p.x) for p in g]
    return pd.DataFrame({"name": gdf["name"], "x": [a for a, _ in xz], "z": [b for _, b in xz]})


if __name__ == "__main__":
    try:
        for k, v in fetch_all().items():
            print(k, v)
    except DataSourceUnavailable as err:
        print(err, file=sys.stderr)
        raise SystemExit(2) from None
