"""The one coordinate contract (spec 4, 13A.3).

- Processing CRS: EPSG:32611 (UTM 11N, meters).
- Scene space: local meters relative to scene_origin (center of the region bbox):
    x = easting  - origin_easting      (east is +x)
    z = -(northing - origin_northing)  (south is +z, three.js convention)
    y = elevation in meters             (up is +y)
- Unreal: UE.X = x*100, UE.Y = z*100, UE.Z = y*100 (centimeters).

The TypeScript twin lives in client/src/geo.ts. Both are tested against
docs/geo_test_points.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pyproj import Transformer

from pipeline.config import region

PROJECTION = "EPSG:32611"


@dataclass(frozen=True)
class SceneOrigin:
    lat: float
    lon: float
    easting: float
    northing: float


@lru_cache(maxsize=4)
def _to_utm() -> Transformer:
    return Transformer.from_crs("EPSG:4326", PROJECTION, always_xy=True)


@lru_cache(maxsize=4)
def _from_utm() -> Transformer:
    return Transformer.from_crs(PROJECTION, "EPSG:4326", always_xy=True)


def lonlat_to_utm(lon: float, lat: float) -> tuple[float, float]:
    e, n = _to_utm().transform(lon, lat)
    return float(e), float(n)


def utm_to_lonlat(easting: float, northing: float) -> tuple[float, float]:
    lon, lat = _from_utm().transform(easting, northing)
    return float(lon), float(lat)


def origin_from_bbox(bbox: dict[str, float]) -> SceneOrigin:
    lat = (bbox["south"] + bbox["north"]) / 2.0
    lon = (bbox["west"] + bbox["east"]) / 2.0
    e, n = lonlat_to_utm(lon, lat)
    return SceneOrigin(lat=lat, lon=lon, easting=e, northing=n)


@lru_cache(maxsize=1)
def scene_origin() -> SceneOrigin:
    return origin_from_bbox(region()["bbox"])


def utm_to_scene(easting: float, northing: float, origin: SceneOrigin | None = None) -> tuple[float, float]:
    o = origin or scene_origin()
    return easting - o.easting, -(northing - o.northing)


def scene_to_utm(x: float, z: float, origin: SceneOrigin | None = None) -> tuple[float, float]:
    o = origin or scene_origin()
    return x + o.easting, o.northing - z


def latlon_to_scene(lat: float, lon: float, origin: SceneOrigin | None = None) -> tuple[float, float]:
    """(lat, lon) degrees -> scene (x, z) meters."""
    e, n = lonlat_to_utm(lon, lat)
    return utm_to_scene(e, n, origin)


def scene_to_latlon(x: float, z: float, origin: SceneOrigin | None = None) -> tuple[float, float]:
    """scene (x, z) meters -> (lat, lon) degrees."""
    e, n = scene_to_utm(x, z, origin)
    lon, lat = utm_to_lonlat(e, n)
    return lat, lon


def scene_to_unreal(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Scene meters -> Unreal centimeters (X east, Y south, Z up)."""
    return x * 100.0, z * 100.0, y * 100.0
