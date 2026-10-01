"""Fetch the USGS 3DEP 10 m DEM (1/3 arc-second) for the terrain extent via py3dep (spec 5.1).

Cached as data/raw/dem_3dep_10m.tif (skipped if present).

    python pipeline/fetch_dem.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.common import DataSourceUnavailable, bbox_lonlat_of_extent, log, terrain_extent
from pipeline.config import raw_dir

DEM_FILE = "dem_3dep_10m.tif"
DEM_RESOLUTION_M = 10
SERVICE_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
HOW = (
    "Download the USGS 1/3 arc-second DEM tiles covering the bbox from The National Map "
    "(https://apps.nationalmap.gov/downloader/ -> Elevation Products (3DEP) -> 1/3 arc-second DEM), "
    "e.g. https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n34w118/USGS_13_n34w118.tif "
    "and .../n33w118/USGS_13_n33w118.tif, then merge them into one GeoTIFF, e.g. "
    "`gdalwarp USGS_13_n34w118.tif USGS_13_n33w118.tif data/raw/dem_3dep_10m.tif` (any CRS is fine)."
)


def fetch(raw: Path | None = None) -> Path:
    raw = raw or raw_dir()
    dest = raw / DEM_FILE
    if dest.exists():
        log(f"cached: {dest.name}")
        return dest
    raw.mkdir(parents=True, exist_ok=True)
    bbox = bbox_lonlat_of_extent(terrain_extent(), margin_deg=0.005)
    log(f"3DEP: requesting {DEM_RESOLUTION_M} m DEM for {bbox}")
    try:
        import py3dep

        dem = py3dep.get_dem(bbox, resolution=DEM_RESOLUTION_M, crs=4326)
        dem.rio.to_raster(dest)
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise DataSourceUnavailable("USGS 3DEP DEM (py3dep)", SERVICE_URL, dest, HOW, e) from e
    log(f"3DEP: wrote {dest.name}")
    return dest


if __name__ == "__main__":
    try:
        print(fetch())
    except DataSourceUnavailable as err:
        print(err, file=sys.stderr)
        raise SystemExit(2) from None
