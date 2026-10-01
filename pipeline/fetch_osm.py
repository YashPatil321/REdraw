"""Fetch OpenStreetMap data via OSMnx / Overpass (spec 5): drive, walk and bike networks,
building footprints and amenity=school features. Cached in data/raw/ (skipped if present).

    python pipeline/fetch_osm.py
"""

from __future__ import annotations

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


def load_schools(path: Path, schools_cfg: list[dict[str, Any]]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """OSM amenity=school -> (all schools EPSG:32611, campus polygons tagged with schools.yaml ids).

    A campus polygon gets the id of the schools.yaml school whose point is inside it or within
    pipeline.school_snap_max_dist_m of it.
    """
    gdf = gpd.read_file(path).to_crs("EPSG:32611")
    if "name" not in gdf.columns:
        gdf["name"] = None
    max_d = float(assumption("pipeline.school_snap_max_dist_m"))
    pts = []
    for s in schools_cfg:
        from pipeline.geo import lonlat_to_utm

        pts.append((s["id"], *lonlat_to_utm(float(s["lon"]), float(s["lat"]))))
    areas = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    sid = []
    for g in areas.geometry:
        best, bd = None, max_d
        for i, e, n in pts:
            d = g.distance(gpd.points_from_xy([e], [n])[0])
            if d <= bd:
                best, bd = i, d
        sid.append(best)
    areas["school_id"] = sid
    areas = areas[areas["school_id"].notna()][["school_id", "name", "geometry"]]
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
