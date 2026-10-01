"""Fetch USDA NAIP aerial imagery from the Microsoft Planetary Computer STAC API (spec 5).

Mosaics the most recent NAIP year covering the terrain extent into one RGB GeoTIFF in
EPSG:32611 (data/raw/naip_mosaic.tif, skipped if present). build_terrain warps it into each
tile's albedo texture.

    python pipeline/fetch_imagery.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pipeline.common import DataSourceUnavailable, bbox_lonlat_of_extent, load_env, log, terrain_extent
from pipeline.config import raw_dir
from pipeline.geo import scene_origin

NAIP_FILE = "naip_mosaic.tif"
MOSAIC_RES_M = 2.0  # 4 tiles x 2600 m / 1536 px textures need ~1.7 m; 2 m keeps the file < 100 MB
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
HOW = (
    "Download NAIP DOQQs (RGB GeoTIFF/JP2) covering the bbox from USGS EarthExplorer "
    "(https://earthexplorer.usgs.gov, dataset NAIP) or the Planetary Computer Explorer, mosaic them and "
    "save as an RGB GeoTIFF named data/raw/naip_mosaic.tif (any CRS; EPSG:32611 preferred), e.g. "
    "`gdalwarp -t_srs EPSG:32611 -tr 2 2 *.tif data/raw/naip_mosaic.tif`."
)


def latest_items(items: list[Any]) -> list[Any]:
    """Keep only items from the most recent NAIP year (pure helper, unit tested)."""
    if not items:
        return []

    def year(it: Any) -> int:
        dt = it.datetime or it.properties.get("datetime")
        return int(str(dt)[:4]) if dt else int(str(it.properties.get("start_datetime", "0"))[:4])

    best = max(year(i) for i in items)
    return [i for i in items if year(i) == best]


def fetch(raw: Path | None = None) -> Path:
    raw = raw or raw_dir()
    dest = raw / NAIP_FILE
    if dest.exists():
        log(f"cached: {dest.name}")
        return dest
    raw.mkdir(parents=True, exist_ok=True)
    load_env()
    ext = terrain_extent()
    bbox = bbox_lonlat_of_extent(ext)
    try:
        import planetary_computer
        import pystac_client
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, reproject

        if os.environ.get("PC_SDN_KEY"):
            planetary_computer.settings.set_subscription_key(os.environ["PC_SDN_KEY"])
        cat = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
        items = list(cat.search(collections=["naip"], bbox=list(bbox)).items())
        items = latest_items(items)
        if not items:
            raise RuntimeError("no NAIP items intersect the bbox")
        log(f"NAIP: {len(items)} items from {str(items[0].datetime)[:4]}")
        o = scene_origin()
        w = int(round(ext.width / MOSAIC_RES_M))
        h = int(round(ext.depth / MOSAIC_RES_M))
        transform = from_origin(o.easting + ext.min_x, o.northing - ext.min_z, MOSAIC_RES_M, MOSAIC_RES_M)
        mosaic = np.zeros((3, h, w), dtype=np.uint8)
        filled = np.zeros((h, w), dtype=bool)
        for it in items:
            href = it.assets["image"].href
            with rasterio.open(href) as src:
                tmp = np.zeros((3, h, w), dtype=np.uint8)
                for b in range(3):
                    reproject(
                        source=rasterio.band(src, b + 1),
                        destination=tmp[b],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs="EPSG:32611",
                        dst_nodata=0,
                        resampling=Resampling.average,
                    )
                new = (tmp.sum(axis=0) > 0) & ~filled
                mosaic[:, new] = tmp[:, new]
                filled |= new
        cover = filled.mean()
        if cover < 0.9:
            raise RuntimeError(f"NAIP mosaic covers only {cover:.0%} of the terrain extent")
        with rasterio.open(dest, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs="EPSG:32611", transform=transform, compress="deflate", tiled=True) as dst:
            dst.write(mosaic)
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise DataSourceUnavailable("USDA NAIP imagery (Planetary Computer STAC)", f"{STAC_URL}/collections/naip", dest, HOW, e) from e
    log(f"NAIP: wrote {dest.name}")
    return dest


if __name__ == "__main__":
    try:
        print(fetch())
    except DataSourceUnavailable as err:
        print(err, file=sys.stderr)
        raise SystemExit(2) from None
