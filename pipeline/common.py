"""Shared pipeline helpers: region grid, tiles, errors, small IO utilities.

Everything here is mode independent (real and synthetic builds share it).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import assets_dir, processed_dir, raw_dir, region
from pipeline.geo import SceneOrigin, latlon_to_scene, scene_origin

CONTRACT_VERSION = 1


class DataSourceUnavailable(RuntimeError):
    """Raised when a real data source cannot be reached (spec rule 14.6).

    The message always names the URL that was tried and where a manually
    downloaded file must be placed in data/raw/.
    """

    def __init__(self, source: str, url: str, raw_path: Path | str, how: str, cause: BaseException | None = None):
        self.source = source
        self.url = url
        self.raw_path = str(raw_path)
        msg = (
            f"\n\n[redraw] DATA SOURCE UNAVAILABLE: {source}\n"
            f"  URL tried : {url}\n"
            f"  Error     : {type(cause).__name__ + ': ' + str(cause) if cause else 'n/a'}\n"
            f"  Manual fix: {how}\n"
            f"  Put file  : {raw_path}\n"
            "  Then re-run `python pipeline/build_all.py` (cached files are reused).\n"
            "  For offline development only, `python pipeline/build_all.py --synthetic` builds a\n"
            "  clearly labeled FAKE world instead.\n"
        )
        super().__init__(msg)


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def today() -> str:
    return datetime.now(UTC).date().isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def ensure_dirs() -> dict[str, Path]:
    d = {
        "raw": raw_dir(),
        "processed": processed_dir(),
        "assets": assets_dir(),
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    for sub in ("terrain", "buildings", "roads"):
        (d["assets"] / sub).mkdir(parents=True, exist_ok=True)
    return d


@dataclass(frozen=True)
class Extent:
    min_x: float
    max_x: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        return self.max_z - self.min_z

    def contains(self, x: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | bool:
        return (x >= self.min_x) & (x <= self.max_x) & (z >= self.min_z) & (z <= self.max_z)

    def buffered(self, d: float) -> Extent:
        return Extent(self.min_x - d, self.max_x + d, self.min_z - d, self.max_z + d)

    def as_dict(self) -> dict[str, float]:
        return {"min_x": self.min_x, "max_x": self.max_x, "min_z": self.min_z, "max_z": self.max_z}


def bbox_scene_extent(bbox: dict[str, float], origin: SceneOrigin | None = None, round_to: float = 50.0) -> Extent:
    """Axis aligned scene extent covering the WGS84 bbox (corners projected), rounded outward."""
    xs, zs = [], []
    for lat in (bbox["south"], bbox["north"]):
        for lon in (bbox["west"], bbox["east"]):
            x, z = latlon_to_scene(lat, lon, origin)
            xs.append(x)
            zs.append(z)
    # The bbox edges bulge slightly in UTM; sample edge midpoints too.
    for lat in (bbox["south"], bbox["north"]):
        x, z = latlon_to_scene(lat, (bbox["west"] + bbox["east"]) / 2, origin)
        zs.append(z)
    for lon in (bbox["west"], bbox["east"]):
        x, z = latlon_to_scene((bbox["south"] + bbox["north"]) / 2, lon, origin)
        xs.append(x)

    def rdn(v: float) -> float:
        return math.floor(v / round_to) * round_to

    def rup(v: float) -> float:
        return math.ceil(v / round_to) * round_to

    return Extent(rdn(min(xs)), rup(max(xs)), rdn(min(zs)), rup(max(zs)))


def region_extent() -> Extent:
    return bbox_scene_extent(region()["bbox"], scene_origin())


def terrain_extent() -> Extent:
    return region_extent().buffered(float(region().get("terrain_buffer_m", 500)))


@dataclass(frozen=True)
class TileGrid:
    extent: Extent
    rows: int
    cols: int

    def tile_id(self, r: int, c: int) -> str:
        return f"r{r}_c{c}"

    def bounds(self, r: int, c: int) -> Extent:
        w = self.extent.width / self.cols
        d = self.extent.depth / self.rows
        return Extent(
            self.extent.min_x + c * w,
            self.extent.min_x + (c + 1) * w,
            self.extent.min_z + r * d,
            self.extent.min_z + (r + 1) * d,
        )

    def tile_of(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row/col of scene points (clamped to the grid)."""
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        c = np.floor((x - self.extent.min_x) / (self.extent.width / self.cols)).astype(int)
        r = np.floor((z - self.extent.min_z) / (self.extent.depth / self.rows)).astype(int)
        return np.clip(r, 0, self.rows - 1), np.clip(c, 0, self.cols - 1)

    def iter(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]


def tile_grid() -> TileGrid:
    rows, cols = region().get("tiles", [4, 4])
    return TileGrid(terrain_extent(), int(rows), int(cols))


def log(msg: str) -> None:
    print(f"[redraw] {msg}", flush=True)
