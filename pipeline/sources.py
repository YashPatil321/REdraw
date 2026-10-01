"""Data source attribution for region_meta.json `sources` (real mode).

The raw cache records where every file came from:
- `data/raw/aws_sources.json` (written by pipeline/fetch_aws.py): every URL, release, license;
- `data/raw/<file>.source.json` sidecars next to the DEM and imagery GeoTIFFs.

These are read instead of hardcoding OSM / py3dep / NAIP, so the attribution always matches
the files the build actually used. When neither exists (raw data fetched by the primary
fetchers fetch_osm / fetch_dem / fetch_imagery) the primary-source defaults are used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pipeline.common import today

AWS_SOURCES = "aws_sources.json"

# Primary fetchers (spec 5): used when the raw cache has no provenance files.
PRIMARY_SOURCES: dict[str, dict[str, Any]] = {
    "osm": {"name": "OpenStreetMap (roads, buildings, schools) via Overpass/OSMnx", "license": "ODbL 1.0", "url": "https://www.openstreetmap.org/copyright", "attribution": "(c) OpenStreetMap contributors"},
    "dem": {"name": "USGS 3D Elevation Program (3DEP) 1/3 arc-second DEM via py3dep", "license": "Public domain (USGS)", "url": "https://www.usgs.gov/3d-elevation-program", "attribution": "USGS 3D Elevation Program"},
    "imagery": {"name": "USDA NAIP imagery via Microsoft Planetary Computer", "license": "Public domain (USDA FSA)", "url": "https://planetarycomputer.microsoft.com/dataset/naip", "attribution": "USDA NAIP"},
}
CENSUS_SOURCES: list[dict[str, Any]] = [
    {"name": "US Census Bureau ACS 5-year estimates (block groups) and TIGER/Line", "license": "Public domain (US Census Bureau)", "url": "https://www.census.gov/data/developers.html", "kind": "population"},
    {"name": "US Census Bureau LEHD LODES 8 (OD, WAC, RAC)", "license": "Public domain (US Census Bureau)", "url": "https://lehd.ces.census.gov/data/", "kind": "population"},
]
FOOTPRINT_POPULATION_SOURCE: dict[str, Any] = {
    "name": "Footprint population estimate (Redraw pipeline): households from real residential building footprints, attributes from assumptions.yaml",
    "license": "Derived from the building footprint source above; method CC BY 4.0 (Redraw project)",
    "kind": "population",
    "note": (
        "Census ACS / LODES / TIGER were unreachable, so `build_all.py --population-source footprints` "
        "estimated households from footprints: 1 household per house, townhouse rows and apartments by "
        "area per unit (assumptions footprint_population.*, pipeline.apartment_*). Household size, kids, "
        "vehicles, income and workers are sampled from assumptions population.* / population_synthesis.*; "
        "jobs inside the region are commercial/school buildings weighted by area, the rest leave via the "
        "region.yaml exits by population_synthesis.external_exit_shares. Not census data."
    ),
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _merge_tiles(entries: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Collapse per-tile entries with the same name (e.g. 4 DEM tiles) into one with `files`."""
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        name = str(e.get("name", "unknown"))
        if name not in out:
            out[name] = {k: v for k, v in e.items() if k != "url"}
            out[name]["kind"] = kind
            out[name]["files"] = []
        if e.get("url"):
            out[name]["files"].append(e["url"])
    res = []
    for e in out.values():
        files = e.pop("files")
        if len(files) == 1:
            e["url"] = files[0]
        elif files:
            e["url"] = os.path.commonprefix(files).rsplit("/", 1)[0] + "/"
            e["files"] = files
        res.append(e)
    return res


def sidecar_sources(path: Path, kind: str) -> list[dict[str, Any]]:
    """Entries from `<raster>.source.json` (a single source object or {"sources": [...]})."""
    side = path.with_suffix(".source.json")
    data = _read_json(side) if side.exists() else None
    if not data:
        return []
    entries = data.get("sources") if isinstance(data, dict) and "sources" in data else [data]
    extra = {k: v for k, v in data.items() if k in ("resolution_m", "crs", "coverage")} if isinstance(data, dict) and "sources" in data else {}
    merged = _merge_tiles([dict(e) for e in entries if isinstance(e, dict)], kind)
    for e in merged:
        if extra:
            e["raster"] = {**extra, "file": path.name}
        if kind == "imagery" and "Sentinel" in str(e.get("name", "")):
            year = str(e.get("acquired") or e.get("retrieved") or today())[:4]
            e.setdefault("attribution", f"Contains modified Copernicus Sentinel data {year}")
        if kind == "dem" and "3DEP" in str(e.get("name", "")):
            e.setdefault("attribution", "USGS 3D Elevation Program (3DEP)")
    return merged


def aws_sources(raw: Path) -> dict[str, Any] | None:
    data = _read_json(raw / AWS_SOURCES)
    return data if isinstance(data, dict) else None


def _overture_attribution(e: dict[str, Any]) -> str:
    lic = str(e.get("license", ""))
    return "(c) OpenStreetMap contributors, Overture Maps Foundation" if "ODbL" in lic else "Overture Maps Foundation"


def raw_sources(raw: Path, dem_path: Path, imagery_path: Path) -> list[dict[str, Any]]:
    """Sources for the terrain, imagery and map layers actually present in the raw cache."""
    out: list[dict[str, Any]] = []
    aws = aws_sources(raw)
    dem = sidecar_sources(dem_path, "dem")
    img = sidecar_sources(imagery_path, "imagery")
    aws_entries = list(aws.get("sources", [])) if aws else []
    if not dem:
        dem = _merge_tiles([e for e in aws_entries if e.get("kind") == "dem"], "dem")
    if not img:
        img = _merge_tiles([e for e in aws_entries if e.get("kind") == "imagery"], "imagery")
    out += dem or [dict(PRIMARY_SOURCES["dem"], kind="dem", retrieved=today())]
    out += img or [dict(PRIMARY_SOURCES["imagery"], kind="imagery", retrieved=today())]
    overture = [dict(e) for e in aws_entries if e.get("kind") == "overture"]
    for e in overture:
        e.setdefault("attribution", _overture_attribution(e))
    if overture:
        out += overture
    else:
        out.append(dict(PRIMARY_SOURCES["osm"], kind="osm", retrieved=today()))
    return out


def population_sources(population_source: str) -> list[dict[str, Any]]:
    if population_source == "acs":
        return [dict(s, retrieved=today()) for s in CENSUS_SOURCES]
    return [dict(FOOTPRINT_POPULATION_SOURCE, retrieved=today())]


def not_available(raw: Path) -> list[str]:
    aws = aws_sources(raw)
    return list(aws.get("not_available", [])) if aws else []


def attribution_lines(sources: list[dict[str, Any]]) -> list[str]:
    """Short, de-duplicated attribution strings for a client footer."""
    seen: list[str] = []
    for s in sources:
        a = s.get("attribution")
        if a and a not in seen:
            seen.append(str(a))
    return seen
