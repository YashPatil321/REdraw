"""Fetch USGS 3DEP lidar point clouds (Entwine EPT on AWS open data) for the region bbox.

Source: `usgs-lidar-public` bucket (public, no auth), project CA_SanDiegoQL2_2014
(USGS QL2, flown 2014, ~4.6 pts/m2 over our bbox when every EPT depth is read). It is the
only public EPT/LAZ point cloud that covers 4S Ranch / Del Sur:

- CA_SanDiego_2015_C17_1 (the denser 2015 county flight) only covers the east county; its
  octree has no nodes over our bbox.
- CA_SanDiegoCo_D24 (2024) LAZ tiles are only on rockyweb.usgs.gov, which the sandbox egress
  policy blocks (403). Its 1 m bare-earth DEM is what data/raw/dem_3dep_10m.tif holds.

EPT points are in EPSG:3857 (WGS 84 / Pseudo-Mercator, a straight reprojection by USGS of
NAD83(2011) UTM 11N); Z is NAVD88 orthometric meters (verified against the 3DEP DEM in
lidar_features). We reproject X/Y to EPSG:32611 with pyproj (no datum grid shift), which is
the convention every other raw file in the pipeline uses.

Steps (both resumable; files that already exist are skipped):

1. hierarchy: walk ept-hierarchy/*.json, keep every node (all depths: EPT spreads one
   point cloud across its depths, so full density needs all of them) whose cube intersects
   the bbox + `lidar.fetch_buffer_m`. Cached as data/raw/lidar/ept_nodes.json.
2. nodes: GET ept-data/<D-X-Y-Z>.laz in parallel into data/raw/lidar/ept-data/.
3. tiles: decode nodes, reproject to EPSG:32611, crop and re-tile into 1 km UTM tiles
   data/raw/lidar/tiles/utm_<E_km>_<N_km>.laz (LAS 1.2 PF1, 1 cm scale, CRS VLR, all
   original attributes we use: classification, return numbers, intensity, gps time).

    .venv/bin/python -m pipeline.fetch_lidar            # resume / skip cached
    .venv/bin/python -m pipeline.fetch_lidar --force    # re-tile (downloads stay cached)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pipeline.common import DataSourceUnavailable, log, now_iso, write_json
from pipeline.config import assumption, raw_dir, region
from pipeline.fetch_aws import configure_network
from pipeline.geo import PROJECTION, lonlat_to_utm

EPT_BUCKET_URL = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"
EPT_PROJECT = "CA_SanDiegoQL2_2014"
EPT_LICENSE = "Public domain (USGS 3D Elevation Program)"
TILE_M = 1000.0
LAS_SCALE = 0.01

HOW = (
    "fetch_lidar reads the public USGS EPT at "
    f"{EPT_BUCKET_URL}/{EPT_PROJECT}/ept.json (AWS open data, no auth). If it is unreachable, "
    "download the LAZ tiles for the bbox from https://apps.nationalmap.gov/downloader/ "
    "(Elevation Source Data / Lidar Point Cloud, project CA_SanDiego*), reproject them to "
    "EPSG:32611 with PDAL (filters.reprojection) and save them as "
    "data/raw/lidar/tiles/utm_<E_km>_<N_km>.laz (1 km tiles named by their SW corner in km)."
)


def lidar_dir() -> Path:
    return raw_dir() / "lidar"


# ---------------------------------------------------------------------------
# Geometry helpers (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    minx: float
    miny: float
    maxx: float
    maxy: float

    def intersects(self, o: Box) -> bool:
        return self.minx < o.maxx and self.maxx > o.minx and self.miny < o.maxy and self.maxy > o.miny

    def buffered(self, d: float) -> Box:
        return Box(self.minx - d, self.miny - d, self.maxx + d, self.maxy + d)


def utm_bbox(buffer_m: float = 0.0) -> Box:
    """Region bbox (region.yaml) as an EPSG:32611 box, buffered by `buffer_m`."""
    b = region()["bbox"]
    xs, ys = [], []
    for lon in (b["west"], b["east"]):
        for lat in (b["south"], b["north"]):
            e, n = lonlat_to_utm(lon, lat)
            xs.append(e)
            ys.append(n)
    return Box(min(xs), min(ys), max(xs), max(ys)).buffered(buffer_m)


def tile_keys_for(box: Box, tile_m: float = TILE_M) -> list[tuple[int, int]]:
    """SW corners (in units of tile_m) of the aligned tiles covering `box`."""
    c0, c1 = math.floor(box.minx / tile_m), math.ceil(box.maxx / tile_m)
    r0, r1 = math.floor(box.miny / tile_m), math.ceil(box.maxy / tile_m)
    return [(c, r) for c in range(c0, c1) for r in range(r0, r1)]


def tile_name(ckm: int, rkm: int) -> str:
    return f"utm_{ckm}_{rkm}.laz"


def utm_box_to_3857(box: Box, densify: int = 8) -> Box:
    """Envelope in EPSG:3857 of a UTM box (edges densified so the envelope is conservative)."""
    from pyproj import Transformer

    t = Transformer.from_crs(PROJECTION, "EPSG:3857", always_xy=True)
    s = np.linspace(0.0, 1.0, densify + 1)
    ex = np.concatenate([box.minx + s * (box.maxx - box.minx), np.full_like(s, box.maxx),
                         box.maxx - s * (box.maxx - box.minx), np.full_like(s, box.minx)])
    ey = np.concatenate([np.full_like(s, box.miny), box.miny + s * (box.maxy - box.miny),
                         np.full_like(s, box.maxy), box.maxy - s * (box.maxy - box.miny)])
    x, y = t.transform(ex, ey)
    return Box(float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y)))


def ept_node_box(key: str, cube: list[float]) -> Box:
    """XY bounds of EPT node 'D-X-Y-Z' inside root cube [minx,miny,minz,maxx,maxy,maxz]."""
    d, x, y, _ = (int(v) for v in key.split("-"))
    w = (cube[3] - cube[0]) / (2**d)
    return Box(cube[0] + x * w, cube[1] + y * w, cube[0] + (x + 1) * w, cube[1] + (y + 1) * w)


def ept_children(key: str) -> list[str]:
    d, x, y, z = (int(v) for v in key.split("-"))
    return [f"{d + 1}-{2 * x + a}-{2 * y + b}-{2 * z + c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def _session(pool: int = 32) -> Any:
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "redraw-pipeline/fetch_lidar"
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=3)
    s.mount("https://", adapter)
    return s


def _get(sess: Any, url: str, timeout: float = 120.0, tries: int = 4) -> bytes:
    last: BaseException | None = None
    for i in range(tries):
        try:
            r = sess.get(url, timeout=timeout)
            r.raise_for_status()
            return bytes(r.content)
        except Exception as e:  # noqa: BLE001 - retried, then surfaced
            last = e
            time.sleep(1.5 * (i + 1))
    raise DataSourceUnavailable("USGS lidar EPT", url, lidar_dir(), HOW, last)


def project_url() -> str:
    return f"{EPT_BUCKET_URL}/{EPT_PROJECT}"


def load_ept_meta(sess: Any) -> dict[str, Any]:
    path = lidar_dir() / "ept.json"
    if path.exists():
        return json.loads(path.read_text())
    meta = json.loads(_get(sess, f"{project_url()}/ept.json"))
    write_json(path, meta)
    return meta


def select_nodes(sess: Any, cube: list[float], aoi3857: Box) -> dict[str, int]:
    """Walk the EPT hierarchy; return {node key: point count} for nodes intersecting the AOI."""
    found: dict[str, int] = {}
    pending = ["0-0-0-0"]
    while pending:
        with ThreadPoolExecutor(16) as ex:
            pages = list(ex.map(lambda k: json.loads(_get(sess, f"{project_url()}/ept-hierarchy/{k}.json")), pending))
        pending = []
        for page in pages:
            for k, v in page.items():
                if not ept_node_box(k, cube).intersects(aoi3857):
                    continue
                if v == -1:
                    if k not in found:
                        pending.append(k)
                elif v > 0:
                    found[k] = int(v)
    return found


def download_nodes(sess: Any, keys: Iterable[str], workers: int = 32) -> tuple[int, int]:
    out = lidar_dir() / "ept-data"
    out.mkdir(parents=True, exist_ok=True)
    todo = [k for k in keys if not (out / f"{k}.laz").exists()]
    nbytes = 0

    def one(k: str) -> int:
        data = _get(sess, f"{project_url()}/ept-data/{k}.laz")
        tmp = out / f"{k}.laz.part"
        tmp.write_bytes(data)
        tmp.replace(out / f"{k}.laz")
        return len(data)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, k) for k in todo]
        for i, f in enumerate(as_completed(futs), 1):
            nbytes += f.result()
            if i % 500 == 0 or i == len(futs):
                dt = max(time.time() - t0, 1e-6)
                log(f"  lidar nodes {i}/{len(futs)}  {nbytes / 1e6:.0f} MB  {nbytes / 1e6 / dt:.1f} MB/s")
    return len(todo), nbytes


# ---------------------------------------------------------------------------
# Re-tiling (EPSG:3857 EPT nodes -> 1 km EPSG:32611 LAZ tiles)
# ---------------------------------------------------------------------------

KEEP_DIMS = ("intensity", "return_number", "number_of_returns", "classification", "gps_time",
             "scan_angle_rank", "point_source_id")


def read_node_utm(path: Path) -> dict[str, np.ndarray]:
    """Decode one EPT node LAZ and reproject X/Y to EPSG:32611 (Z unchanged, NAVD88 m)."""
    import laspy
    from pyproj import Transformer

    las = laspy.read(io.BytesIO(path.read_bytes()))
    t = Transformer.from_crs("EPSG:3857", PROJECTION, always_xy=True)
    x, y = t.transform(np.asarray(las.x), np.asarray(las.y))
    out = {"x": np.asarray(x), "y": np.asarray(y), "z": np.asarray(las.z, dtype=np.float64)}
    for d in KEEP_DIMS:
        out[d] = np.asarray(las[d])
    return out


def _write_tile(args: tuple[int, int, list[str], str, bool]) -> tuple[str, int]:
    import laspy
    from pyproj import CRS

    ckm, rkm, node_files, out_dir, force = args
    dest = Path(out_dir) / tile_name(ckm, rkm)
    if dest.exists() and not force:
        return dest.name, -1
    x0, y0 = ckm * TILE_M, rkm * TILE_M
    parts: dict[str, list[np.ndarray]] = {}
    for f in node_files:
        p = read_node_utm(Path(f))
        m = (p["x"] >= x0) & (p["x"] < x0 + TILE_M) & (p["y"] >= y0) & (p["y"] < y0 + TILE_M)
        if not m.any():
            continue
        for k, v in p.items():
            parts.setdefault(k, []).append(v[m])
    n = int(sum(len(a) for a in parts.get("x", [])))
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.scales = np.array([LAS_SCALE, LAS_SCALE, LAS_SCALE])
    header.offsets = np.array([x0, y0, 0.0])
    header.add_crs(CRS.from_epsg(32611))
    las = laspy.LasData(header)
    if n:
        cat = {k: np.concatenate(v) for k, v in parts.items()}
        las.x, las.y, las.z = cat["x"], cat["y"], cat["z"]
        for d in KEEP_DIMS:
            las[d] = cat[d]
    tmp = dest.with_suffix(".part.laz")
    las.write(str(tmp))
    tmp.replace(dest)
    return dest.name, n


def build_tiles(nodes: dict[str, int], cube: list[float], aoi_utm: Box, force: bool, workers: int = 4) -> dict[str, int]:
    node_dir = lidar_dir() / "ept-data"
    out_dir = lidar_dir() / "tiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ckm, rkm in tile_keys_for(aoi_utm):
        tb = Box(ckm * TILE_M, rkm * TILE_M, (ckm + 1) * TILE_M, (rkm + 1) * TILE_M)
        b3857 = utm_box_to_3857(tb).buffered(5.0)
        files = [str(node_dir / f"{k}.laz") for k in nodes if ept_node_box(k, cube).intersects(b3857)]
        jobs.append((ckm, rkm, files, str(out_dir), force))
    counts: dict[str, int] = {}
    t0 = time.time()
    import multiprocessing as mp

    # spawn, not fork: forking after the download thread pool has run can deadlock the workers
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        for i, (name, n) in enumerate(ex.map(_write_tile, jobs), 1):
            counts[name] = n
            if n >= 0:
                log(f"  tile {i}/{len(jobs)} {name}: {n:,} pts ({time.time() - t0:.0f}s)")
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild UTM tiles (node downloads stay cached)")
    ap.add_argument("--workers", type=int, default=32, help="parallel HTTP downloads")
    args = ap.parse_args(argv)

    configure_network()
    t0 = time.time()
    d = lidar_dir()
    d.mkdir(parents=True, exist_ok=True)
    sess = _session(args.workers)
    meta = load_ept_meta(sess)
    srs = meta.get("srs", {})
    if str(srs.get("horizontal")) != "3857":
        raise RuntimeError(f"unexpected EPT srs {srs.get('horizontal')}; this module assumes EPSG:3857")
    cube = [float(v) for v in meta["bounds"]]

    buf = float(assumption("lidar.fetch_buffer_m"))
    aoi_utm = utm_bbox(buf)
    aoi3857 = utm_box_to_3857(aoi_utm).buffered(10.0)
    nodes_path = d / "ept_nodes.json"
    if nodes_path.exists():
        nodes = {k: int(v) for k, v in json.loads(nodes_path.read_text())["nodes"].items()}
    else:
        log(f"lidar: walking EPT hierarchy of {EPT_PROJECT} ...")
        nodes = select_nodes(sess, cube, aoi3857)
        write_json(nodes_path, {"project": EPT_PROJECT, "aoi_3857": aoi3857.__dict__, "nodes": nodes})
    log(f"lidar: {len(nodes)} EPT nodes, {sum(nodes.values()):,} points intersect the bbox + {buf:.0f} m")
    t1 = time.time()
    n_new, nbytes = download_nodes(sess, nodes, workers=args.workers)
    log(f"lidar: downloaded {n_new} nodes ({nbytes / 1e6:.0f} MB) in {time.time() - t1:.0f}s")
    t2 = time.time()
    counts = build_tiles(nodes, cube, aoi_utm, force=args.force)
    log(f"lidar: tiles ready in {time.time() - t2:.0f}s")
    tile_points: dict[str, int] = {k: v for k, v in counts.items() if v >= 0}
    src_path = d / "lidar.source.json"
    if src_path.exists():  # keep the counts of tiles written by an earlier run
        old = json.loads(src_path.read_text()).get("tile_points")
        if isinstance(old, dict):
            tile_points = {**old, **tile_points}
    write_json(src_path, {
        "name": f"USGS 3DEP lidar point cloud, project {EPT_PROJECT} (Entwine EPT)",
        "url": f"{project_url()}/ept.json",
        "license": EPT_LICENSE,
        "retrieved": now_iso(),
        "ept_srs": "EPSG:3857 (horizontal), Z NAVD88 m",
        "tiles_crs": f"{PROJECTION} (horizontal), Z NAVD88 m",
        "aoi_utm": aoi_utm.__dict__,
        "nodes": len(nodes),
        "points": int(sum(nodes.values())),
        "tile_points": tile_points,
        "note": "2014 flight: buildings and trees newer than 2014 are absent from this point cloud.",
    })
    log(f"lidar: done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
