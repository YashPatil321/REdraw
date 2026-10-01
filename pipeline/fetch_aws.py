"""Fetch the real-mode raw cache from AWS open-data mirrors (alternative to fetch_osm /
fetch_dem / fetch_imagery when Overpass, The National Map API and Planetary Computer are
unreachable).

Writes the SAME data/raw/ files the real build reads, for the region bbox plus the terrain
buffer (region.yaml terrain_buffer_m):

    dem_3dep_10m.tif         USGS 3DEP 1 m lidar DEM (prd-tnm S3, COG overview read at 2 m),
                             gaps filled from the 3DEP 1/3 arc-second tiles. Name kept for the
                             build; the actual resolution is in dem_3dep_10m.source.json.
    naip_mosaic.tif          RGB true colour from the clearest recent summer Sentinel-2 L2A scene
                             (sentinel-cogs S3, 10 m bands, natural colour stretch) + sidecar
                             naip_mosaic.source.json {"name", "license", "url"}. NAIP itself is
                             requester-pays on AWS and is not used.
    osm_drive_raw.graphml    OSMnx-compatible MultiDiGraph (WGS84, unsimplified) built from
    osm_walk.graphml         Overture transportation segments + connectors (OSM-derived, ODbL).
    osm_bike.graphml
    osm_buildings.geojson    Overture buildings mapped to OSM-style tags.
    osm_schools.geojson      Overture base land_use class=school polygons + Overture places
                             schools (points), tagged amenity=school.
    aws_sources.json         every URL, release, date and license used.

    .venv/bin/python -m pipeline.fetch_aws            # skip files that already exist
    .venv/bin/python -m pipeline.fetch_aws --force    # re-download everything
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pipeline.common import DataSourceUnavailable, log, now_iso, terrain_extent, today, write_json
from pipeline.config import raw_dir
from pipeline.geo import scene_origin

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

TNM_BUCKET_URL = "https://prd-tnm.s3.amazonaws.com"
DEM_1M_PROJECTS = ["CA_SanDiegoCo_D24", "San_Diego_CA_2014_LiDAR"]  # preference order (newest first)
DEM_13_TILES = ["n34w118", "n33w118"]  # 1/3 arc-second fallback / gap fill
DEM_RES_M = 2.0

S2_BUCKET_URL = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
S2_PREFIX = "sentinel-s2-l2a-cogs"
S2_MGRS = ["11SMS"]
S2_MONTHS = (6, 7, 8, 9)
S2_MAX_SCENE_CLOUD = 20.0
IMAGERY_RES_M = 2.5  # upsampled (Lanczos) from 10 m so the per-tile bilinear warp stays smooth

OVERTURE_BUCKET = "overturemaps-us-west-2"
OVERTURE_URL = f"https://{OVERTURE_BUCKET}.s3.amazonaws.com"
OVERTURE_DOCS = "https://docs.overturemaps.org/attribution/"
# Theme licenses per Overture's attribution page; the per-record `sources[].license` values
# actually present in the bbox are tallied into aws_sources.json as well.
OVERTURE_THEME_LICENSE = {
    "transportation": "ODbL 1.0 (OpenStreetMap-derived; attribution: (c) OpenStreetMap contributors, Overture Maps Foundation)",
    "buildings": "ODbL 1.0 (OpenStreetMap plus Esri Community Maps, Microsoft and Google open buildings; attribution: (c) OpenStreetMap contributors, Overture Maps Foundation)",
    "places": "CDLA-Permissive-2.0 (Meta, Microsoft and other sources; attribution: Overture Maps Foundation)",
    "base": "ODbL 1.0 (OpenStreetMap-derived land_use/water; attribution: (c) OpenStreetMap contributors, Overture Maps Foundation)",
}

FILES = {
    "dem": "dem_3dep_10m.tif",
    "imagery": "naip_mosaic.tif",
    "drive": "osm_drive_raw.graphml",
    "walk": "osm_walk.graphml",
    "bike": "osm_bike.graphml",
    "buildings": "osm_buildings.geojson",
    "schools": "osm_schools.geojson",
    "water": "overture_water.geojson",
    "sources": "aws_sources.json",
}

HOW = (
    "fetch_aws reads public AWS open-data buckets (prd-tnm, sentinel-cogs, overturemaps-us-west-2). "
    "If they are unreachable, use the primary fetchers on a machine with normal internet access: "
    "`python pipeline/fetch_osm.py`, `python pipeline/fetch_dem.py`, `python pipeline/fetch_imagery.py`, "
    "and copy data/raw/ here."
)

# ---------------------------------------------------------------------------
# Network setup (sandbox proxy + CA bundle)
# ---------------------------------------------------------------------------

CCR_CA_BUNDLE = Path("/root/.ccr/ca-bundle.crt")


def configure_network() -> None:
    """Point requests/GDAL/curl at the proxy CA bundle when present; tune GDAL /vsicurl/."""
    ca = os.environ.get("SSL_CERT_FILE") or (str(CCR_CA_BUNDLE) if CCR_CA_BUNDLE.exists() else None)
    if ca:
        for k in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            os.environ.setdefault(k, ca)
    for k, v in {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "VSI_CACHE": "TRUE",
    }.items():
        os.environ.setdefault(k, v)


def _session() -> Any:
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "redraw-pipeline/fetch_aws"
    return s


def s3_list(bucket_url: str, prefix: str, delimiter: str | None = "/") -> tuple[list[tuple[str, int]], list[str]]:
    """Anonymous S3 ListObjectsV2 over HTTPS -> ([(key, size)], [common prefixes])."""
    sess = _session()
    keys: list[tuple[str, int]] = []
    prefixes: list[str] = []
    token: str | None = None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        params: dict[str, str] = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        r = sess.get(bucket_url + "/", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for c in root.findall("s3:Contents", ns):
            keys.append((c.findtext("s3:Key", "", ns), int(c.findtext("s3:Size", "0", ns))))
        for p in root.findall("s3:CommonPrefixes", ns):
            prefixes.append(p.findtext("s3:Prefix", "", ns))
        if root.findtext("s3:IsTruncated", "false", ns) != "true":
            break
        token = root.findtext("s3:NextContinuationToken", None, ns)
    return keys, prefixes


# ---------------------------------------------------------------------------
# Extent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchExtent:
    """UTM (EPSG:32611) rectangle and the WGS84 box that covers it."""

    west_e: float
    south_n: float
    east_e: float
    north_n: float

    @property
    def lonlat(self) -> tuple[float, float, float, float]:
        from pipeline.geo import utm_to_lonlat

        pts = [utm_to_lonlat(e, n) for e in (self.west_e, self.east_e) for n in (self.south_n, self.north_n)]
        pts += [utm_to_lonlat((self.west_e + self.east_e) / 2, n) for n in (self.south_n, self.north_n)]
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        return min(lons), min(lats), max(lons), max(lats)


def fetch_extent(margin_m: float = 100.0) -> FetchExtent:
    """Terrain extent (region bbox + terrain buffer) in UTM, plus a small safety margin."""
    ext = terrain_extent()
    o = scene_origin()
    return FetchExtent(
        o.easting + ext.min_x - margin_m,
        o.northing - ext.max_z - margin_m,
        o.easting + ext.max_x + margin_m,
        o.northing - ext.min_z + margin_m,
    )


# ---------------------------------------------------------------------------
# DEM: 3DEP 1 m lidar (overview read at 2 m) + 1/3 arc-second gap fill
# ---------------------------------------------------------------------------


def dem_1m_tile_names(fx: FetchExtent) -> list[str]:
    """USGS 1 m DEM 10 km tile ids (x{E/10km}y{ceil(N/10km)}) covering the extent."""
    out = []
    for x in range(int(fx.west_e // 10000), int(fx.east_e // 10000) + 1):
        for y in range(int(math.ceil(fx.south_n / 10000)), int(math.ceil(fx.north_n / 10000)) + 1):
            out.append(f"x{x}y{y}")
    return out


def _read_into(dst: np.ndarray, dst_transform: Any, url: str, overview_factor: float) -> int:
    """Warp one remote raster into dst (EPSG:32611) where dst is NaN. Returns pixels filled."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import array_bounds
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import from_bounds

    h, w = dst.shape
    with rasterio.open(url) as src:
        b = transform_bounds("EPSG:32611", src.crs, *array_bounds(h, w, dst_transform), densify_pts=21)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if win.width <= 0 or win.height <= 0:
            return 0
        oh = max(1, int(round(win.height / overview_factor)))
        ow = max(1, int(round(win.width / overview_factor)))
        arr = src.read(1, window=win, out_shape=(oh, ow), resampling=Resampling.average, masked=True).astype(np.float32)
        arr = arr.filled(np.nan)
        arr[(arr < -500) | (arr > 9000)] = np.nan
        wt = src.window_transform(win) * rasterio.Affine.scale(win.width / ow, win.height / oh)
        tmp = np.full_like(dst, np.nan)
        reproject(arr, tmp, src_transform=wt, src_crs=src.crs, src_nodata=np.nan, dst_transform=dst_transform, dst_crs="EPSG:32611", dst_nodata=np.nan, resampling=Resampling.bilinear)
    fill = np.isnan(dst) & np.isfinite(tmp)
    dst[fill] = tmp[fill]
    return int(fill.sum())


def fetch_dem(raw: Path, fx: FetchExtent, force: bool, sources: list[dict[str, Any]]) -> Path:
    import rasterio
    from rasterio.transform import from_origin

    dest = raw / FILES["dem"]
    side = raw / "dem_3dep_10m.source.json"
    if dest.exists() and not force:
        log(f"cached: {dest.name}")
        if side.exists():
            sources.extend(json.loads(side.read_text())["sources"])
        return dest
    w = int(math.ceil((fx.east_e - fx.west_e) / DEM_RES_M))
    h = int(math.ceil((fx.north_n - fx.south_n) / DEM_RES_M))
    transform = from_origin(fx.west_e, fx.north_n, DEM_RES_M, DEM_RES_M)
    dem = np.full((h, w), np.nan, dtype=np.float32)
    used: list[dict[str, Any]] = []
    tiles = dem_1m_tile_names(fx)
    for proj in DEM_1M_PROJECTS:
        if not np.isnan(dem).any():
            break
        try:
            keys, _ = s3_list(TNM_BUCKET_URL, f"StagedProducts/Elevation/1m/Projects/{proj}/TIFF/", delimiter=None)
        except Exception as e:  # noqa: BLE001
            log(f"DEM: cannot list {proj}: {e}")
            continue
        for key, _size in sorted(keys):
            if not key.endswith(".tif") or not any(f"_{t}_" in key for t in tiles):
                continue
            url = f"{TNM_BUCKET_URL}/{key}"
            t0 = time.time()
            n = _read_into(dem, transform, "/vsicurl/" + url, overview_factor=DEM_RES_M)
            log(f"DEM: {key.rsplit('/', 1)[-1]} filled {n:,} px in {time.time() - t0:.1f}s")
            if n:
                used.append({"name": f"USGS 3DEP 1 m lidar DEM, project {proj}", "url": url, "resolution_m": 1.0, "read_at_m": DEM_RES_M})
    missing = float(np.isnan(dem).mean())
    if missing > 0:
        for t in DEM_13_TILES:
            url = f"{TNM_BUCKET_URL}/StagedProducts/Elevation/13/TIFF/current/{t}/USGS_13_{t}.tif"
            try:
                n = _read_into(dem, transform, "/vsicurl/" + url, overview_factor=1.0)
            except Exception as e:  # noqa: BLE001
                log(f"DEM: 1/3 arc-second {t} failed: {e}")
                continue
            log(f"DEM: 1/3 arc-second {t} filled {n:,} gap px")
            if n:
                used.append({"name": "USGS 3DEP 1/3 arc-second (~10 m) DEM (gap fill)", "url": url, "resolution_m": 10.0})
    cover = 1.0 - float(np.isnan(dem).mean())
    if cover < 0.98:
        raise DataSourceUnavailable("USGS 3DEP DEM (AWS prd-tnm)", f"{TNM_BUCKET_URL}/StagedProducts/Elevation/", dest, HOW, RuntimeError(f"DEM covers only {cover:.1%} of the extent"))
    nod = -9999.0
    out = np.where(np.isnan(dem), nod, dem).astype(np.float32)
    tmp = dest.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32", crs="EPSG:32611", transform=transform, nodata=nod, compress="deflate", predictor=3, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER") as f:
        f.write(out, 1)
    tmp.replace(dest)
    lic = "Public domain (USGS 3D Elevation Program)"
    for u in used:
        u.update(license=lic, retrieved=today())
    write_json(side, {"resolution_m": DEM_RES_M, "crs": "EPSG:32611", "coverage": cover, "sources": used})
    sources.extend(used)
    log(f"DEM: wrote {dest.name} {w}x{h} @ {DEM_RES_M} m, coverage {cover:.2%}, elev {np.nanmin(dem):.0f}..{np.nanmax(dem):.0f} m")
    return dest


# ---------------------------------------------------------------------------
# Imagery: Sentinel-2 L2A true colour
# ---------------------------------------------------------------------------

S2_BAD_SCL = (0, 1, 3, 8, 9, 10)  # nodata, saturated, cloud shadow, cloud med/high, cirrus


def natural_color(refl: np.ndarray, white: float = 0.30, black: float = 0.01, gamma: float = 1.0 / 2.0, saturation: float = 1.25, knee: float = 0.65) -> np.ndarray:
    """(3,H,W) surface reflectance -> (3,H,W) uint8 natural colour.

    Fixed black/white reflectance points shared by all bands (keeps the scene's colour balance;
    a percentile white point lets bright commercial roofs darken everything else), a soft
    highlight shoulder above `knee` so roofs do not clip hard, display gamma and a mild
    saturation boost.
    """
    x = np.nan_to_num(refl.astype(np.float32), nan=0.0)
    x = np.clip((x - black) / (white - black), 0.0, None)
    x = np.where(x > knee, knee + (1.0 - knee) * np.tanh((x - knee) / (1.0 - knee)), x)
    x = np.clip(x, 0.0, 1.0) ** gamma
    lum = x.mean(axis=0, keepdims=True)
    x = np.clip(lum + (x - lum) * saturation, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def scene_sort_key(prefix: str) -> tuple[str, int]:
    """(acquisition date, processing version) of an S2 scene prefix, for 'most recent' ties."""
    m = re.search(r"_(\d{8})_(\d+)_L2A", prefix)
    return (m.group(1), int(m.group(2))) if m else ("", 0)


def _s2_scenes(year: int) -> list[str]:
    out: list[str] = []
    for mgrs in S2_MGRS:
        zone, band, sq = mgrs[:2], mgrs[2], mgrs[3:]
        for m in S2_MONTHS:
            _, pre = s3_list(S2_BUCKET_URL, f"{S2_PREFIX}/{int(zone)}/{band}/{sq}/{year}/{m}/")
            out += pre
    return out


def _scene_item(prefix: str) -> dict[str, Any]:
    name = prefix.rstrip("/").rsplit("/", 1)[-1]
    r = _session().get(f"{S2_BUCKET_URL}/{prefix}{name}.json", timeout=60)
    r.raise_for_status()
    return r.json()


S2_ASSET_KEYS = {"B04": ("red", "B04"), "B03": ("green", "B03"), "B02": ("blue", "B02"), "SCL": ("scl", "SCL")}


def band_scale(item: dict[str, Any], band: str) -> tuple[float, float]:
    """(scale, offset) DN -> reflectance for an Earth Search S2 L2A item.

    Earth Search harmonises baseline >= 04.00 data (`earthsearch:boa_offset_applied: true`):
    the -1000 DN offset is already applied even though raster:bands still lists offset -0.1.
    """
    props = item.get("properties", {})
    assets = item.get("assets", {})
    a: dict[str, Any] = next((assets[k] for k in S2_ASSET_KEYS.get(band, (band,)) if k in assets), {})
    rb = (a.get("raster:bands") or [{}])[0]
    scale = float(rb.get("scale", 1e-4))
    if props.get("earthsearch:boa_offset_applied") is True:
        return scale, 0.0
    if "offset" in rb:
        return scale, float(rb["offset"])
    pb = str(props.get("s2:processing_baseline", "00.00"))
    return scale, (-0.1 if pb >= "04.00" else 0.0)


def _read_band(url: str, fx: FetchExtent, res: float, resampling: Any) -> np.ndarray:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    w = int(math.ceil((fx.east_e - fx.west_e) / res))
    h = int(math.ceil((fx.north_n - fx.south_n) / res))
    dst = np.zeros((h, w), dtype=np.float32)
    with rasterio.open("/vsicurl/" + url) as src:
        reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs, src_nodata=0, dst_transform=from_origin(fx.west_e, fx.north_n, res, res), dst_crs="EPSG:32611", dst_nodata=0, resampling=resampling)
    return dst


def fetch_imagery(raw: Path, fx: FetchExtent, force: bool, sources: list[dict[str, Any]], year: int | None = None) -> Path:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin

    dest = raw / FILES["imagery"]
    side = raw / "naip_mosaic.source.json"
    if dest.exists() and not force:
        log(f"cached: {dest.name}")
        if side.exists():
            sources.append(json.loads(side.read_text()))
        return dest
    years = [year] if year else [int(today()[:4]), int(today()[:4]) - 1]
    cands: list[tuple[float, str, dict[str, Any]]] = []
    for y in years:
        try:
            scenes = _s2_scenes(y)
        except Exception as e:  # noqa: BLE001
            raise DataSourceUnavailable("Sentinel-2 L2A COGs (AWS)", f"{S2_BUCKET_URL}/{S2_PREFIX}/", dest, HOW, e) from e
        with ThreadPoolExecutor(8) as ex:
            items = list(ex.map(lambda p: (p, _scene_item(p)), scenes))
        for p, it in items:
            cc = float(it.get("properties", {}).get("eo:cloud_cover", 100.0))
            if cc <= S2_MAX_SCENE_CLOUD:
                cands.append((cc, p, it))
        if len(cands) >= 3:
            break
    if not cands:
        raise DataSourceUnavailable("Sentinel-2 L2A COGs (AWS)", f"{S2_BUCKET_URL}/{S2_PREFIX}/", dest, HOW, RuntimeError("no summer scene under the cloud limit"))
    cands.sort(key=lambda t: (t[0], t[1]))
    # Local clear fraction from the scene classification layer (marine layer is local).
    best: tuple[float, str, dict[str, Any]] | None = None
    for cc, p, it in cands[:8]:
        scl = _read_band(f"{S2_BUCKET_URL}/{p}SCL.tif", fx, 20.0, Resampling.nearest)
        bad = float(np.isin(scl, S2_BAD_SCL).mean())
        log(f"S2: {p.rstrip('/').rsplit('/', 1)[-1]} scene cloud {cc:.1f}%, local bad {bad:.2%}")
        if best is None or bad < best[0] - 1e-4 or (abs(bad - best[0]) <= 1e-4 and scene_sort_key(p) > scene_sort_key(best[1])):
            best = (bad, p, it)
    assert best is not None
    bad, p, it = best
    name = p.rstrip("/").rsplit("/", 1)[-1]
    bands = []
    for b in ("B04", "B03", "B02"):
        dn = _read_band(f"{S2_BUCKET_URL}/{p}{b}.tif", fx, IMAGERY_RES_M, Resampling.lanczos)
        sc, off = band_scale(it, b)
        refl = np.where(dn > 0, dn * sc + off, np.nan)
        bands.append(refl)
    refl = np.stack(bands)
    rgb = natural_color(refl)
    h, w = rgb.shape[1:]
    tmp = dest.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs="EPSG:32611", transform=from_origin(fx.west_e, fx.north_n, IMAGERY_RES_M, IMAGERY_RES_M), compress="deflate", tiled=True, photometric="RGB") as f:
        f.write(rgb)
    tmp.replace(dest)
    dt = str(it.get("properties", {}).get("datetime", ""))[:10]
    meta = {
        "name": f"Copernicus Sentinel-2 L2A true colour, scene {name} ({dt}), 10 m bands B04/B03/B02 upsampled to {IMAGERY_RES_M} m",
        "license": f"Copernicus Sentinel data {dt[:4]}: free, full and open (Copernicus Sentinel Data Terms); attribution 'Contains modified Copernicus Sentinel data {dt[:4]}'",
        "url": f"{S2_BUCKET_URL}/{p}",
        "scene": name,
        "acquired": dt,
        "scene_cloud_cover_pct": float(it.get("properties", {}).get("eo:cloud_cover", float("nan"))),
        "local_cloud_or_shadow_fraction": bad,
        "resolution_m": 10.0,
        "retrieved": today(),
    }
    write_json(side, meta)
    sources.append(meta)
    log(f"S2: wrote {dest.name} {w}x{h} from {name}")
    return dest


# ---------------------------------------------------------------------------
# Overture GeoParquet (row groups pruned by bbox statistics)
# ---------------------------------------------------------------------------


def overture_s3() -> Any:
    import pyarrow.fs as pafs

    kw: dict[str, Any] = {"anonymous": True, "region": "us-west-2"}
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        kw["proxy_options"] = proxy
    ca = os.environ.get("SSL_CERT_FILE")
    if ca:
        kw["tls_ca_file_path"] = ca
    return pafs.S3FileSystem(**kw)


def latest_overture_release() -> str:
    _, pre = s3_list(OVERTURE_URL, "release/")
    rel = sorted(p.split("/")[1] for p in pre if p.count("/") >= 2 and re.match(r"release/\d{4}-\d{2}-\d{2}", p))
    if not rel:
        raise RuntimeError("no Overture releases listed")
    return rel[-1]


def row_groups_in_bbox(md: Any, bbox: tuple[float, float, float, float]) -> list[int]:
    """Row groups whose bbox.{xmin,xmax,ymin,ymax} statistics can intersect bbox (w, s, e, n)."""
    w, s, e, n = bbox
    out = []
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        st: dict[str, Any] = {}
        for j in range(rg.num_columns):
            c = rg.column(j)
            if c.path_in_schema in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax") and c.statistics is not None and c.statistics.has_min_max:
                st[c.path_in_schema] = c.statistics
        if len(st) < 4:
            out.append(i)  # no stats -> must read
            continue
        if st["bbox.xmin"].min <= e and st["bbox.xmax"].max >= w and st["bbox.ymin"].min <= n and st["bbox.ymax"].max >= s:
            out.append(i)
    return out


def overture_read(s3: Any, release: str, theme: str, typ: str, bbox: tuple[float, float, float, float], columns: list[str] | None = None, threads: int = 24) -> Any:
    """Rows of one Overture type whose bbox intersects bbox, as a pyarrow Table."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    base = f"{OVERTURE_BUCKET}/release/{release}/theme={theme}/type={typ}"
    files = sorted(f.path for f in s3.get_file_info(pafs.FileSelector(base)) if f.path.endswith(".parquet"))
    t0 = time.time()

    def scan(path: str) -> tuple[str, list[int]]:
        for attempt in range(4):
            try:
                return path, row_groups_in_bbox(pq.read_metadata(path, filesystem=s3), bbox)
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        return path, []

    with ThreadPoolExecutor(threads) as ex:
        hits = [(p, rgs) for p, rgs in ex.map(scan, files) if rgs]
    log(f"overture {typ}: {len(files)} files scanned in {time.time() - t0:.0f}s, {sum(len(r) for _, r in hits)} row groups in {len(hits)} files intersect")

    def read(arg: tuple[str, list[int]]) -> Any:
        path, rgs = arg
        t = pq.ParquetFile(path, filesystem=s3).read_row_groups(rgs, columns=columns)
        b = t.column("bbox")
        w, s, e, n = bbox
        m = pc.and_(
            pc.and_(pc.less_equal(pc.struct_field(b, "xmin"), e), pc.greater_equal(pc.struct_field(b, "xmax"), w)),
            pc.and_(pc.less_equal(pc.struct_field(b, "ymin"), n), pc.greater_equal(pc.struct_field(b, "ymax"), s)),
        )
        return t.filter(m)

    with ThreadPoolExecutor(min(8, max(1, len(hits)))) as ex:
        parts = list(ex.map(read, hits))
    if not parts:
        return None
    tab = pa.concat_tables(parts, promote_options="permissive")
    log(f"overture {typ}: {tab.num_rows:,} features in bbox ({time.time() - t0:.0f}s)")
    return tab


def overture_cached(raw: Path, s3: Any, release: str, theme: str, typ: str, bbox: tuple[float, float, float, float], columns: list[str] | None, force: bool) -> Any:
    """overture_read with a local parquet cache in data/raw/overture/ keyed by release + bbox."""
    import pyarrow.parquet as pq

    d = raw / "overture"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{typ}.parquet"
    key = json.dumps({"release": release, "bbox": [round(v, 6) for v in bbox], "columns": columns})
    if p.exists() and not force:
        md = pq.read_schema(p).metadata or {}
        if md.get(b"redraw_key", b"").decode() == key:
            log(f"cached: overture/{p.name}")
            return pq.read_table(p)
    tab = overture_read(s3, release, theme, typ, bbox, columns)
    if tab is None:
        return None
    tab = tab.replace_schema_metadata({**(tab.schema.metadata or {}), b"redraw_key": key.encode()})
    pq.write_table(tab, p, compression="zstd")
    return tab


def license_tally(tab: Any) -> dict[str, int]:
    """Count (dataset, license) pairs in an Overture `sources` column."""
    out: dict[str, int] = {}
    if tab is None or "sources" not in tab.column_names:
        return out
    for srcs in tab.column("sources").to_pylist():
        seen = set()
        for s in srcs or []:
            k = f"{s.get('dataset')} | {s.get('license')}"
            if k not in seen:
                out[k] = out.get(k, 0) + 1
                seen.add(k)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# Overture transportation -> OSMnx-compatible graphs (pure helpers are unit tested)
# ---------------------------------------------------------------------------

DRIVE_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street"}
WALK_EXCLUDE = {"motorway", "trunk"}
WALK_EXTRA = {"footway", "path", "pedestrian", "steps", "cycleway", "track", "living_street", "service", "bridleway"}
BIKE_EXCLUDE = {"motorway", "trunk", "steps", "footway", "pedestrian"}
MODE_SETS = {
    "drive": {"motor_vehicle", "car", "vehicle", "motorcycle"},
    "walk": {"foot"},
    "bike": {"bicycle", "vehicle"},
}
ROUTE_PREFIX = {"US:I": "I", "US:US": "US", "US:CA": "CA", "US:CA:SD": "CR"}


def highway_tag(cls: str | None, subclass: str | None, flags: Sequence[str] = ()) -> str:
    """Overture class/subclass -> OSM highway value (links -> *_link, sidewalks/crosswalks -> footway)."""
    c = (cls or "unclassified").strip()
    sub = (subclass or "").strip()
    if c == "unknown":
        c = "road"
    if sub == "link" or "is_link" in flags:
        if c in {"motorway", "trunk", "primary", "secondary", "tertiary"}:
            return f"{c}_link"
    if c == "footway" and sub in {"sidewalk", "crosswalk"}:
        return "footway"
    return c


def _applies(when: dict[str, Any] | None, modes: set[str]) -> bool:
    """Does an Overture access rule apply unconditionally to one of `modes`?"""
    if not when:
        return True
    if when.get("using") or when.get("recognized") or when.get("during") or when.get("vehicle"):
        return False  # conditional (customers, time windows, vehicle size): ignore like OSMnx
    m = when.get("mode")
    if m:
        return bool(set(m) & modes)
    return True


def _full_length(between: Any) -> bool:
    if not between:
        return True
    try:
        a, b = float(between[0]), float(between[1])
    except (TypeError, ValueError, IndexError):
        return True
    return a <= 0.01 and b >= 0.99


def is_private_rule(rule: dict[str, Any], modes: set[str]) -> bool:
    """Overture encodes OSM access=private as `allowed when recognized=[as_private]`."""
    when = rule.get("when") or {}
    if rule.get("access_type") != "allowed" or "as_private" not in (when.get("recognized") or []):
        return False
    m = when.get("mode")
    return not m or bool(set(m) & modes)


def directions_allowed(restrictions: Iterable[dict[str, Any]] | None, modes: set[str]) -> tuple[bool, bool]:
    """(forward allowed, backward allowed) for a mode set from Overture access_restrictions.

    Unconditional full-length denials apply first (heading-specific -> oneway); then allowances
    that name one of the modes explicitly re-open a heading (e.g. footway: denied, foot allowed).
    """
    fwd = bwd = True
    rules = [r for r in (restrictions or []) if _full_length(r.get("between"))]
    for r in rules:
        when = r.get("when") or {}
        if is_private_rule(r, modes):
            fwd = bwd = False  # access=private (OSMnx excludes these ways from every network type)
            continue
        if r.get("access_type") != "denied" or not _applies(when, modes):
            continue
        h = when.get("heading")
        if h in (None, "forward"):
            fwd = False
        if h in (None, "backward"):
            bwd = False
    for r in rules:
        when = r.get("when") or {}
        if r.get("access_type") not in ("allowed", "designated") or not when.get("mode") or not _applies(when, modes):
            continue
        h = when.get("heading")
        if h in (None, "forward"):
            fwd = True
        if h in (None, "backward"):
            bwd = True
    return fwd, bwd


def maxspeed_tag(limits: Iterable[dict[str, Any]] | None) -> str | None:
    """Overture speed_limits -> OSM maxspeed ('45 mph'); the highest unconditional full-length max."""
    best: tuple[float, str] | None = None
    for s in limits or []:
        ms = s.get("max_speed") or {}
        if ms.get("value") is None or not _full_length(s.get("between")):
            continue
        when = s.get("when") or {}
        if when.get("during") or when.get("vehicle") or when.get("using"):
            continue
        unit = (ms.get("unit") or "km/h").strip()
        v = float(ms["value"])
        kph = v * 1.609344 if unit == "mph" else v
        tag = f"{int(v)} mph" if unit == "mph" else f"{int(v)}"
        if best is None or kph > best[0]:
            best = (kph, tag)
    return None if best is None else best[1]


def ref_tag(routes: Iterable[dict[str, Any]] | None) -> str | None:
    """Overture routes -> OSM-style ref ('I 15', 'CA 56'); unique, ';'-joined, full-length only."""
    out: list[str] = []
    for r in routes or []:
        ref = (r.get("ref") or "").strip()
        if not ref or not _full_length(r.get("between")):
            continue
        if re.fullmatch(r"\d+[A-Z]?", ref):
            pre = ROUTE_PREFIX.get(r.get("network") or "")
            if pre is None:
                continue
            ref = f"{pre} {ref}"
        if ref not in out:
            out.append(ref)
    return ";".join(out) or None


def osm_way_id(sources: Iterable[dict[str, Any]] | None) -> int | None:
    for s in sources or []:
        m = re.match(r"^w(\d+)", str(s.get("record_id") or ""))
        if m and (s.get("dataset") or "").lower().startswith("openstreetmap"):
            return int(m.group(1))
    return None


@dataclass
class SegmentEdge:
    u: str
    v: str
    coords: list[tuple[float, float]]  # lon/lat, u -> v
    length_m: float
    attrs: dict[str, Any] = field(default_factory=dict)


def split_segment(coords_ll: Sequence[tuple[float, float]], connectors: Sequence[dict[str, Any]], to_utm: Callable[[float, float], tuple[float, float]]) -> list[tuple[str, str, list[tuple[float, float]], float]]:
    """Split a segment polyline at its connectors (linear reference `at` in 0..1).

    Returns [(from_connector, to_connector, lonlat coords, length_m)] in segment order.
    """
    from shapely.geometry import LineString
    from shapely.ops import substring

    cons = sorted(((float(c["at"]), str(c["connector_id"])) for c in connectors if c.get("connector_id") is not None), key=lambda t: t[0])
    if len(cons) < 2 or len(coords_ll) < 2:
        return []
    utm = [to_utm(x, y) for x, y in coords_ll]
    line = LineString(utm)
    L = line.length
    # map each utm vertex back to its lon/lat by interpolation along the line
    out = []
    for (a0, c0), (a1, c1) in zip(cons[:-1], cons[1:], strict=True):
        if c0 == c1 or a1 - a0 <= 0:
            continue
        sub = substring(line, a0 * L, a1 * L)
        if sub.geom_type != "LineString" or len(sub.coords) < 2:
            continue
        ll = _utm_coords_to_ll(list(sub.coords), utm, coords_ll)
        out.append((c0, c1, ll, float(sub.length)))
    return out


def _utm_coords_to_ll(sub: list[tuple[float, ...]], utm: list[tuple[float, float]], ll: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Map substring vertices back to lon/lat: original vertices exactly, cut points by local affine."""
    out = []
    for p in sub:
        x, y = p[0], p[1]
        best_i, best_d = 0, float("inf")
        for i, (ux, uy) in enumerate(utm):
            d = (ux - x) ** 2 + (uy - y) ** 2
            if d < best_d:
                best_i, best_d = i, d
        if best_d < 1e-6:
            out.append((float(ll[best_i][0]), float(ll[best_i][1])))
            continue
        # interpolate on the nearest original segment
        j = best_i + 1 if best_i + 1 < len(utm) else best_i - 1
        (x0, y0), (x1, y1) = utm[best_i], utm[j]
        seg2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
        t = 0.0 if seg2 == 0 else ((x - x0) * (x1 - x0) + (y - y0) * (y1 - y0)) / seg2
        lon = ll[best_i][0] + t * (ll[j][0] - ll[best_i][0])
        lat = ll[best_i][1] + t * (ll[j][1] - ll[best_i][1])
        out.append((float(lon), float(lat)))
    return out


def _mode_classes(kind: str, hw: str) -> bool:
    base = hw[:-5] if hw.endswith("_link") else hw
    if kind == "drive":
        return base in DRIVE_CLASSES
    if kind == "walk":
        return base not in WALK_EXCLUDE and (base in DRIVE_CLASSES or base in WALK_EXTRA or base == "road")
    return base not in BIKE_EXCLUDE and (base in DRIVE_CLASSES or base in {"cycleway", "path", "track", "service", "living_street", "road"})


def build_graphs(segments: Sequence[dict[str, Any]], connector_xy: dict[str, tuple[float, float]], kinds: Sequence[str] = ("drive", "walk", "bike")) -> dict[str, Any]:
    """Overture road segments (dicts with Overture columns + `coords` lon/lat list) -> OSMnx graphs.

    Nodes are connectors with deterministic integer ids (sorted connector GERS ids); each edge
    carries highway/name/ref/maxspeed/oneway/reversed/length/geometry plus osmid (OSM way id
    when Overture records it) and overture_id.
    """
    import networkx as nx
    from shapely.geometry import LineString

    from pipeline.geo import lonlat_to_utm

    pieces = []
    for seg in segments:
        if seg.get("subtype") != "road":
            continue
        flags = [f for rf in (seg.get("road_flags") or []) if _full_length(rf.get("between")) for f in (rf.get("values") or [])]
        if {"is_under_construction", "is_abandoned"} & set(flags):
            continue
        hw = highway_tag(seg.get("class"), seg.get("subclass"), flags)
        name = ((seg.get("names") or {}) or {}).get("primary")
        attrs = {
            "highway": hw,
            "name": name,
            "ref": ref_tag(seg.get("routes")),
            "maxspeed": maxspeed_tag(seg.get("speed_limits")),
            "bridge": "yes" if "is_bridge" in flags else None,
            "tunnel": "yes" if "is_tunnel" in flags else None,
            "overture_id": seg["id"],
        }
        way = osm_way_id(seg.get("sources"))
        for c0, c1, ll, length in split_segment(seg["coords"], seg.get("connectors") or [], lonlat_to_utm):
            pieces.append((seg, hw, attrs, way, c0, c1, ll, length))

    all_conn = sorted({p[4] for p in pieces} | {p[5] for p in pieces})
    nid = {c: i + 1 for i, c in enumerate(all_conn)}
    graphs: dict[str, Any] = {}
    for kind in kinds:
        G = nx.MultiDiGraph(crs="epsg:4326", simplified=False, created_with="redraw pipeline.fetch_aws (Overture Maps transportation)")
        for seg, hw, attrs, way, c0, c1, ll, length in pieces:
            if not _mode_classes(kind, hw):
                continue
            fwd, bwd = directions_allowed(seg.get("access_restrictions"), MODE_SETS[kind])
            if kind == "walk":
                fwd = bwd = fwd or bwd
            if not (fwd or bwd):
                continue
            u, v = nid[c0], nid[c1]
            for n, c, xy in ((u, c0, ll[0]), (v, c1, ll[-1])):
                if n not in G:
                    x, y = connector_xy.get(c, xy)
                    G.add_node(n, x=float(x), y=float(y), overture_id=c)
            oneway = fwd != bwd
            osmid = way if way is not None else _stable_way_id(seg["id"])
            base = {k: v_ for k, v_ in attrs.items() if v_ is not None}
            base.update(osmid=osmid, oneway=oneway, length=float(length))
            if fwd:
                G.add_edge(u, v, **base, reversed=False, geometry=LineString(ll))
            if bwd:
                G.add_edge(v, u, **base, reversed=not oneway, geometry=LineString(ll[::-1]))
        for n in G.nodes:
            G.nodes[n]["street_count"] = len(set(G.predecessors(n)) | set(G.successors(n)))
        graphs[kind] = G
    return graphs


def _stable_way_id(seg_id: str) -> int:
    import hashlib

    return 10**12 + int(hashlib.sha1(seg_id.encode()).hexdigest()[:9], 16)


# ---------------------------------------------------------------------------
# Overture buildings / places / land_use -> OSM-style GeoJSON
# ---------------------------------------------------------------------------

SUBTYPE_TO_BUILDING = {
    "residential": "residential",
    "commercial": "commercial",
    "education": "school",
    "outbuilding": "shed",
    "industrial": "industrial",
    "religious": "religious",
    "civic": "civic",
    "medical": "hospital",
    "service": "service",
    "transportation": "transportation",
    "agricultural": "farm_auxiliary",
    "military": "military",
}
SCHOOL_CATEGORIES = {"school", "elementary_school", "middle_school", "high_school", "public_school", "private_school", "charter_school", "primary_school", "secondary_school", "k_12_school"}


def building_tags(row: dict[str, Any]) -> dict[str, Any]:
    """Overture building row -> OSM-style tags (height in m, building:levels, building, ...)."""
    cls = row.get("class")
    sub = row.get("subtype")
    b = cls or SUBTYPE_TO_BUILDING.get(sub or "", "yes")
    names = row.get("names") or {}
    srcs = row.get("sources") or []
    tags: dict[str, Any] = {
        "element": "overture",
        "id": row["id"],
        "building": b,
        "height": None if row.get("height") is None else f"{float(row['height']):.2f}",
        "min_height": None if row.get("min_height") is None else f"{float(row['min_height']):.2f}",
        "building:levels": None if row.get("num_floors") is None else str(int(row["num_floors"])),
        "name": names.get("primary"),
        "roof:shape": row.get("roof_shape"),
        "roof:material": row.get("roof_material"),
        "roof:colour": row.get("roof_color"),
        "roof:height": None if row.get("roof_height") is None else f"{float(row['roof_height']):.2f}",
        "building:colour": row.get("facade_color"),
        "building:material": row.get("facade_material"),
        "overture_subtype": sub,
        "overture_class": cls,
        "source": srcs[0].get("dataset") if srcs else None,
    }
    way = osm_way_id(srcs)
    if way is not None:
        tags["osm_way_id"] = str(way)
    return tags


def is_school_place(row: dict[str, Any]) -> bool:
    tax = row.get("taxonomy") or {}
    cats = {tax.get("primary")} | set(tax.get("hierarchy") or []) | set(tax.get("alternates") or [])
    cats |= {row.get("basic_category")}
    c = row.get("categories") or {}
    if isinstance(c, dict):
        cats |= {c.get("primary")} | set(c.get("alternate") or [])
    cats.discard(None)
    if not cats & SCHOOL_CATEGORIES:
        return False
    # skip driving/music/dance/... schools and preschools whose primary category is not a K-12 school
    prim = tax.get("primary") or (c.get("primary") if isinstance(c, dict) else None) or row.get("basic_category")
    return prim in SCHOOL_CATEGORIES


SCHOOL_PLACE_MIN_CONFIDENCE = 0.75
SCHOOL_PLACE_NAME_EXCLUDE = re.compile(r"district|football|band|swim|music|kindermusik|garden|services|group|tutor|parents|sports|dance|driving|math", re.I)
SCHOOL_MATCH_M = 100.0


def merge_school_features(polys: Sequence[tuple[dict[str, Any], Any]], places: Sequence[tuple[dict[str, Any], Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Campus polygons (Overture land_use class=school, from OSM amenity=school areas) plus
    Overture places schools as points.

    A place inside / within 100 m of a campus polygon is a duplicate: it only names an unnamed
    polygon. Remaining places must have confidence >= 0.75 and a non-K-12-noise name.
    """
    from shapely.ops import transform

    from pipeline.geo import lonlat_to_utm

    def utm(g: Any) -> Any:
        return transform(lambda x, y, z=None: lonlat_to_utm(x, y), g)

    rows: list[dict[str, Any]] = []
    geoms: list[Any] = []
    putm = []
    for r, g in polys:
        way = osm_way_id(r.get("sources"))
        rows.append({"amenity": "school", "name": (r.get("names") or {}).get("primary"), "name_source": "overture_land_use", "overture_id": r["id"], "overture_type": "land_use", "category": "school", "confidence": None, "osm_way_id": None if way is None else str(way)})
        geoms.append(g)
        putm.append(utm(g))
    for r, g in sorted(places, key=lambda t: -(t[0].get("confidence") or 0.0)):
        name = (r.get("names") or {}).get("primary")
        conf = float(r.get("confidence") or 0.0)
        if conf < SCHOOL_PLACE_MIN_CONFIDENCE or not name or SCHOOL_PLACE_NAME_EXCLUDE.search(name):
            continue
        pu = utm(g)
        hit = min(range(len(putm)), key=lambda i: putm[i].distance(pu), default=None)
        if hit is not None and putm[hit].distance(pu) <= SCHOOL_MATCH_M:
            if not rows[hit]["name"]:
                rows[hit].update(name=name, name_source="overture_place", place_id=r["id"])
            continue
        if any(rr["name"] == name and gg.distance(g) < 0.003 for rr, gg in zip(rows, geoms, strict=True)):
            continue
        tax = r.get("taxonomy") or {}
        rows.append({"amenity": "school", "name": name, "name_source": "overture_place", "overture_id": r["id"], "overture_type": "place", "category": tax.get("primary") or r.get("basic_category"), "confidence": conf, "osm_way_id": None})
        geoms.append(g)
        putm.append(pu)
    return rows, geoms


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _wkb(col: Any) -> list[Any]:
    import shapely

    return list(shapely.from_wkb(np.asarray(col.to_pylist(), dtype=object)))


def _write_geojson(rows: list[dict[str, Any]], geoms: list[Any], dest: Path) -> None:
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    tmp = dest.with_suffix(".part.geojson")
    tmp.unlink(missing_ok=True)
    gdf.to_file(tmp, driver="GeoJSON")
    tmp.replace(dest)


def fetch_overture(raw: Path, fx: FetchExtent, force: bool, sources: list[dict[str, Any]], release: str | None) -> dict[str, Path]:
    import osmnx as ox
    from shapely.geometry import box as shapely_box

    targets = {k: raw / FILES[k] for k in ("drive", "walk", "bike", "buildings", "schools", "water")}
    if all(p.exists() for p in targets.values()) and not force:
        for p in targets.values():
            log(f"cached: {p.name}")
        old = raw / FILES["sources"]
        if old.exists():
            sources.extend(s for s in json.loads(old.read_text()).get("sources", []) if str(s.get("name", "")).startswith("Overture"))
        return targets
    try:
        rel = release or latest_overture_release()
        s3 = overture_s3()
    except Exception as e:  # noqa: BLE001
        raise DataSourceUnavailable("Overture Maps (AWS)", f"{OVERTURE_URL}/release/", raw, HOW, e) from e
    bbox = fx.lonlat
    log(f"overture release {rel}, bbox {tuple(round(v, 5) for v in bbox)}")

    def src(theme: str, typ: str, tab: Any, extra: dict[str, Any] | None = None) -> None:
        sources.append(
            {
                "name": f"Overture Maps {theme}/{typ}",
                "release": rel,
                "url": f"{OVERTURE_URL}/release/{rel}/theme={theme}/type={typ}/",
                "license": OVERTURE_THEME_LICENSE[theme],
                "license_url": OVERTURE_DOCS,
                "features_in_bbox": 0 if tab is None else int(tab.num_rows),
                "record_sources": license_tally(tab),
                "retrieved": today(),
                **(extra or {}),
            }
        )

    try:
        # ---- transportation
        seg_cols = ["id", "geometry", "bbox", "subtype", "class", "subclass", "names", "connectors", "road_flags", "access_restrictions", "speed_limits", "routes", "sources"]
        seg = overture_cached(raw, s3, rel, "transportation", "segment", bbox, seg_cols, force)
        con = overture_cached(raw, s3, rel, "transportation", "connector", bbox, ["id", "geometry", "bbox"], force)
        src("transportation", "segment", seg)
        src("transportation", "connector", con)
        cxy = {}
        for i, g in zip(con.column("id").to_pylist(), _wkb(con.column("geometry")), strict=True):
            cxy[i] = (g.x, g.y)
        segs = seg.drop_columns(["geometry", "bbox"]).to_pylist()
        for s_, g in zip(segs, _wkb(seg.column("geometry")), strict=True):
            s_["coords"] = [(c[0], c[1]) for c in g.coords] if g is not None and g.geom_type == "LineString" else []
        graphs = build_graphs(segs, cxy)
        for kind, G in graphs.items():
            ox.io.save_graphml(G, targets[kind])
            log(f"overture: {kind} graph {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges -> {targets[kind].name}")

        # ---- buildings
        b_cols = ["id", "geometry", "bbox", "names", "sources", "height", "min_height", "is_underground", "num_floors", "subtype", "class", "facade_color", "facade_material", "roof_material", "roof_shape", "roof_color", "roof_height", "has_parts"]
        bt = overture_cached(raw, s3, rel, "buildings", "building", bbox, b_cols, force)
        src("buildings", "building", bt)
        rows, geoms = [], []
        for r, g in zip(bt.drop_columns(["geometry", "bbox"]).to_pylist(), _wkb(bt.column("geometry")), strict=True):
            if r.get("is_underground") or g is None or g.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            rows.append(building_tags(r))
            geoms.append(g)
        _write_geojson(rows, geoms, targets["buildings"])
        log(f"overture: {len(rows):,} buildings -> {targets['buildings'].name}")

        # ---- schools: land_use polygons (OSM amenity=school areas) + places points
        lu = overture_cached(raw, s3, rel, "base", "land_use", bbox, ["id", "geometry", "bbox", "names", "subtype", "class", "sources"], force)
        pl = overture_cached(raw, s3, rel, "places", "place", bbox, ["id", "geometry", "bbox", "names", "taxonomy", "basic_category", "confidence", "addresses", "websites", "sources", "operating_status"], force)
        src("base", "land_use", lu)
        src("places", "place", pl)
        polys: list[tuple[dict[str, Any], Any]] = []
        if lu is not None:
            for r, g in zip(lu.drop_columns(["geometry", "bbox"]).to_pylist(), _wkb(lu.column("geometry")), strict=True):
                if r.get("class") == "school" and g is not None and g.geom_type in ("Polygon", "MultiPolygon"):
                    polys.append((r, g))
        places: list[tuple[dict[str, Any], Any]] = []
        if pl is not None:
            for r, g in zip(pl.drop_columns(["geometry", "bbox"]).to_pylist(), _wkb(pl.column("geometry")), strict=True):
                if g is not None and is_school_place(r) and r.get("operating_status") in (None, "open"):
                    places.append((r, g))
        srows, sgeoms = merge_school_features(polys, places)
        _write_geojson(srows, sgeoms, targets["schools"])
        log(f"overture: {len(srows):,} school features -> {targets['schools'].name}")

        # ---- water (previews / bbox checks only; not read by the build)
        wt = overture_cached(raw, s3, rel, "base", "water", bbox, ["id", "geometry", "bbox", "names", "subtype", "class", "sources"], force)
        src("base", "water", wt)
        wrows, wgeoms = [], []
        clip = shapely_box(*bbox)
        if wt is not None:
            for r, g in zip(wt.drop_columns(["geometry", "bbox"]).to_pylist(), _wkb(wt.column("geometry")), strict=True):
                g = None if g is None else g.intersection(clip)
                if g is None or g.is_empty:
                    continue
                wrows.append({"name": (r.get("names") or {}).get("primary"), "subtype": r.get("subtype"), "class": r.get("class"), "overture_id": r["id"]})
                wgeoms.append(g)
        _write_geojson(wrows, wgeoms, targets["water"])
    except DataSourceUnavailable:
        raise
    except Exception as e:  # noqa: BLE001
        raise DataSourceUnavailable("Overture Maps (AWS)", f"{OVERTURE_URL}/release/{rel}/", raw, HOW, e) from e
    return targets


def fetch_all(raw: Path | None = None, force: bool = False, release: str | None = None, only: Sequence[str] = ("dem", "imagery", "overture")) -> dict[str, Any]:
    configure_network()
    raw = raw or raw_dir()
    raw.mkdir(parents=True, exist_ok=True)
    fx = fetch_extent()
    log(f"fetch extent UTM E {fx.west_e:.0f}..{fx.east_e:.0f} N {fx.south_n:.0f}..{fx.north_n:.0f} (lon/lat {tuple(round(v, 5) for v in fx.lonlat)})")
    sources: list[dict[str, Any]] = []
    out: dict[str, Any] = {}
    if "dem" in only:
        out["dem"] = fetch_dem(raw, fx, force, sources)
    if "imagery" in only:
        out["imagery"] = fetch_imagery(raw, fx, force, sources)
    if "overture" in only:
        out.update(fetch_overture(raw, fx, force, sources, release))
    meta_path = raw / FILES["sources"]
    prev = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    names = {s.get("name") for s in sources}
    keep = [s for s in prev.get("sources", []) if s.get("name") not in names]
    write_json(
        meta_path,
        {
            "generated_at": now_iso(),
            "tool": "pipeline/fetch_aws.py",
            "extent_utm_32611": {"west": fx.west_e, "south": fx.south_n, "east": fx.east_e, "north": fx.north_n},
            "extent_lonlat": dict(zip(("west", "south", "east", "north"), fx.lonlat, strict=True)),
            "files": {k: str(Path(v).name) for k, v in out.items()},
            "not_available": [
                "US Census ACS 5-year and TIGER/Line (api.census.gov, www2.census.gov blocked; no AWS mirror used)",
                "LEHD LODES (lehd.ces.census.gov blocked)",
                "OSM traffic signals and lane counts (Overture transportation has neither)",
                "NAIP (AWS naip-* buckets are requester-pays)",
            ],
            "sources": keep + sources,
        },
    )
    out["sources"] = meta_path
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if the raw files exist")
    ap.add_argument("--release", help="Overture release (default: latest under release/)")
    ap.add_argument("--only", nargs="+", choices=["dem", "imagery", "overture"], default=["dem", "imagery", "overture"])
    args = ap.parse_args(argv)
    try:
        out = fetch_all(force=args.force, release=args.release, only=args.only)
    except DataSourceUnavailable as e:
        print(str(e), file=sys.stderr)
        return 2
    for k, v in out.items():
        p = Path(v)
        print(f"{k:<10} {p}  {p.stat().st_size / 1e6:.1f} MB" if p.exists() else f"{k:<10} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
