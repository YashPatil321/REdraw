"""Block-level location labels (spec 8.1.3: never link personas to addresses).

A label names the nearest named street ("near Camino Del Sur"), optionally
prefixed by a neighborhood name if `region.yaml` defines `neighborhoods`
(`[{name, lat, lon}]`, nearest center wins). Persona coordinates are snapped to
a coarse grid so a marker cannot be traced back to one house.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config import region

# Display privacy grid (not a modeling number): persona markers snap to this.
PRIVACY_GRID_M = 150.0


def snap_to_grid(x: float, z: float, grid_m: float = PRIVACY_GRID_M) -> tuple[float, float]:
    return (float(round(x / grid_m) * grid_m), float(round(z / grid_m) * grid_m))


class BlockLabeler:
    def __init__(self, points: np.ndarray, names: list[str],
                 neighborhoods: list[tuple[str, float, float]] | None = None) -> None:
        self.points = points  # (N, 2) x, z of named street vertices
        self.names = names
        self.neighborhoods = neighborhoods or []
        self._tree: Any = None
        if len(points):
            from scipy.spatial import cKDTree

            self._tree = cKDTree(points)

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> BlockLabeler:
        path = data_dir / "network_edges.parquet"
        pts: list[tuple[float, float]] = []
        names: list[str] = []
        if path.exists():
            df = pd.read_parquet(path, columns=["name", "label", "highway", "geometry"])
            for name, label, hw, geom in zip(df["name"], df["label"], df["highway"], df["geometry"],
                                             strict=True):
                street = (label or name or "").strip()
                if not street or str(hw).startswith("motorway"):
                    continue
                g = np.asarray(geom, dtype=np.float64).reshape(-1, 3)
                for x, _y, z in g:
                    pts.append((x, z))
                    names.append(street)
        return cls(np.asarray(pts, dtype=np.float64).reshape(-1, 2), names, _neighborhoods())

    def street_near(self, x: float, z: float) -> str | None:
        if self._tree is None:
            return None
        _, i = self._tree.query([x, z])
        return self.names[int(i)]

    def neighborhood(self, x: float, z: float) -> str | None:
        best, best_d = None, math.inf
        for name, nx, nz in self.neighborhoods:
            d = (nx - x) ** 2 + (nz - z) ** 2
            if d < best_d:
                best, best_d = name, d
        return best

    def label(self, x: float, z: float) -> str:
        street = self.street_near(x, z)
        hood = self.neighborhood(x, z)
        if street and hood:
            return f"{hood}, near {street}"
        if street:
            return f"near {street}"
        if hood:
            return hood
        return str(region().get("display_name", "the neighborhood"))


def _neighborhoods() -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for n in region().get("neighborhoods") or []:
        try:
            x, z = _latlon_to_scene(float(n["lat"]), float(n["lon"]))
            out.append((str(n["name"]), x, z))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _latlon_to_scene(lat: float, lon: float) -> tuple[float, float]:
    from pipeline.geo import latlon_to_scene  # lazy: pyproj is only needed for neighborhoods

    x, z = latlon_to_scene(lat, lon)
    return float(x), float(z)
