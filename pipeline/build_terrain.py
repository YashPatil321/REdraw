"""Terrain: DEM -> scene grid, 16-bit heightmap, 4x4 textured glTF tiles (spec 5.1).

Core functions take in-memory inputs (a `Terrain` grid and an albedo callback)
so the real (3DEP + NAIP) and synthetic builds share them.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline.common import Extent, TileGrid, log, write_json
from pipeline.geo import scene_origin
from pipeline.glb import MeshData, write_glb

# Rendering budgets (not real-world facts, so they live here, spec 5.1 / 10.3).
TERRAIN_TRIANGLE_BUDGET = 1_150_000
HEIGHTMAP_SIZE_PX = 1009  # Unreal Landscape friendly size (63 quads * 16 + 1)
DEM_SPACING_M = 10.0

AlbedoFn = Callable[[Extent, int], np.ndarray]  # (tile bounds, size px) -> (size, size, 3) uint8, row 0 = north


@dataclass
class Terrain:
    """Elevation samples on a regular scene grid.

    elev[i, j] is the elevation (m) at x = extent.min_x + j*spacing,
    z = extent.min_z + i*spacing (row 0 is north, +row is south).
    """

    elev: np.ndarray
    extent: Extent
    spacing: float

    def sample(self, x: np.ndarray | float, z: np.ndarray | float) -> np.ndarray:
        """Bilinear elevation at scene points (clamped at the grid edge)."""
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        rows, cols = self.elev.shape
        fj = np.clip((x - self.extent.min_x) / self.spacing, 0, cols - 1.000001)
        fi = np.clip((z - self.extent.min_z) / self.spacing, 0, rows - 1.000001)
        j0 = np.floor(fj).astype(np.int64)
        i0 = np.floor(fi).astype(np.int64)
        tj = fj - j0
        ti = fi - i0
        e = self.elev
        v = (
            e[i0, j0] * (1 - ti) * (1 - tj)
            + e[i0, j0 + 1] * (1 - ti) * tj
            + e[i0 + 1, j0] * ti * (1 - tj)
            + e[i0 + 1, j0 + 1] * ti * tj
        )
        return v

    def normals(self, x: np.ndarray, z: np.ndarray, h: float | None = None) -> np.ndarray:
        h = h or self.spacing
        dydx = (self.sample(x + h, z) - self.sample(x - h, z)) / (2 * h)
        dydz = (self.sample(x, z + h) - self.sample(x, z - h)) / (2 * h)
        n = np.stack([-dydx, np.ones_like(dydx), -dydz], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    @property
    def min_elev(self) -> float:
        return float(np.nanmin(self.elev))

    @property
    def max_elev(self) -> float:
        return float(np.nanmax(self.elev))


def grid_shape(extent: Extent, spacing: float) -> tuple[int, int]:
    return int(round(extent.depth / spacing)) + 1, int(round(extent.width / spacing)) + 1


def terrain_from_raster(path: Path, extent: Extent, spacing: float = DEM_SPACING_M) -> Terrain:
    """Reproject a DEM GeoTIFF (any CRS) onto the scene grid in EPSG:32611 (spec 5.1.1)."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    o = scene_origin()
    rows, cols = grid_shape(extent, spacing)
    west = o.easting + extent.min_x - spacing / 2
    north = o.northing - extent.min_z + spacing / 2
    dst_transform = from_origin(west, north, spacing, spacing)
    dst = np.full((rows, cols), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs="EPSG:32611",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    bad = ~np.isfinite(dst)
    if bad.mean() > 0.05:
        raise RuntimeError(f"DEM {path} covers only {100 * (1 - bad.mean()):.1f}% of the terrain extent; re-fetch with a larger bbox")
    if bad.any():
        from scipy import ndimage

        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        dst = dst[tuple(idx)]
    return Terrain(dst.astype(np.float32), extent, spacing)


def write_heightmap(terrain: Terrain, png_path: Path, size: int = HEIGHTMAP_SIZE_PX) -> dict[str, Any]:
    """16-bit grayscale heightmap over the terrain extent (spec 13A.2). Returns terrain_meta dict."""
    ext = terrain.extent
    xs = np.linspace(ext.min_x, ext.max_x, size)
    zs = np.linspace(ext.min_z, ext.max_z, size)
    gx, gz = np.meshgrid(xs, zs)
    e = terrain.sample(gx, gz)
    lo, hi = float(e.min()), float(e.max())
    scale = max(hi - lo, 1e-3) / 65535.0
    px = np.clip(np.round((e - lo) / scale), 0, 65535).astype(np.uint16)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(px, mode="I;16").save(png_path)
    return {
        "heightmap": "terrain/heightmap.png",
        "width_px": size,
        "height_px": size,
        "min_x": ext.min_x,
        "max_x": ext.max_x,
        "min_z": ext.min_z,
        "max_z": ext.max_z,
        "min_elev_m": lo,
        "max_elev_m": hi,
        "elev_scale": scale,
        "elev_offset": lo,
        "note": "elevation_m = elev_offset + pixel_value_uint16 * elev_scale; pixel (0,0) is at (min_x, min_z) i.e. north-west; +col is +x (east), +row is +z (south)",
    }


def mesh_step_for_budget(grid: TileGrid, budget: int = TERRAIN_TRIANGLE_BUDGET) -> float:
    area = grid.extent.width * grid.extent.depth
    step = math.sqrt(2.0 * area / budget)
    # never finer than the DEM
    return max(step, DEM_SPACING_M)


def terrain_tile_mesh(terrain: Terrain, b: Extent, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid mesh for one tile. Returns positions, normals, uvs, indices."""
    nx = max(1, int(math.ceil(b.width / step)))
    nz = max(1, int(math.ceil(b.depth / step)))
    xs = np.linspace(b.min_x, b.max_x, nx + 1)
    zs = np.linspace(b.min_z, b.max_z, nz + 1)
    gx, gz = np.meshgrid(xs, zs)
    gy = terrain.sample(gx, gz)
    pos = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)
    nrm = terrain.normals(gx.ravel(), gz.ravel()).astype(np.float32)
    uv = np.stack([(gx - b.min_x) / b.width, (gz - b.min_z) / b.depth], axis=-1).reshape(-1, 2).astype(np.float32)
    i = np.arange(nz)[:, None]
    j = np.arange(nx)[None, :]
    a = i * (nx + 1) + j
    bb = a + 1
    c = a + (nx + 1)
    d = c + 1
    # Counter-clockwise seen from +y (up): a(x0,z0) -> c(x0,z1) -> b(x1,z0)
    tri1 = np.stack([a, c, bb], axis=-1)
    tri2 = np.stack([bb, c, d], axis=-1)
    idx = np.concatenate([tri1.reshape(-1, 3), tri2.reshape(-1, 3)], axis=0).reshape(-1).astype(np.uint32)
    return pos, nrm, uv, idx


def jpeg_bytes(img: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8), mode="RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def build_terrain_tiles(
    terrain: Terrain,
    grid: TileGrid,
    albedo: AlbedoFn,
    out_dir: Path,
    texture_px: int = 1024,
    budget: int = TERRAIN_TRIANGLE_BUDGET,
    save_loose_albedo: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Write terrain/terrain_r{r}_c{c}.glb tiles. Returns (tile info list, total triangles)."""
    step = mesh_step_for_budget(grid, budget)
    out_dir.mkdir(parents=True, exist_ok=True)
    infos: list[dict[str, Any]] = []
    total = 0
    for r, c in grid.iter():
        b = grid.bounds(r, c)
        tid = grid.tile_id(r, c)
        pos, nrm, uv, idx = terrain_tile_mesh(terrain, b, step)
        img = albedo(b, texture_px)
        jpg = jpeg_bytes(img)
        if save_loose_albedo:
            (out_dir / f"albedo_{tid}.jpg").write_bytes(jpg)
        tris = write_glb(
            out_dir / f"terrain_{tid}.glb",
            [MeshData(name=f"terrain_{tid}", positions=pos, normals=nrm, uvs=uv, indices=idx, texture_jpeg=jpg)],
        )
        total += tris
        infos.append(
            {
                "id": tid,
                "row": r,
                "col": c,
                "bounds": {**b.as_dict(), "min_y": float(pos[:, 1].min()), "max_y": float(pos[:, 1].max())},
                "terrain": f"terrain/terrain_{tid}.glb",
            }
        )
    log(f"terrain: {len(infos)} tiles, step {step:.1f} m, {total:,} triangles")
    return infos, total


def raster_albedo(path: Path) -> AlbedoFn:
    """Albedo callback that warps an RGB GeoTIFF (e.g. the NAIP mosaic) into each tile."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    o = scene_origin()

    def fn(b: Extent, size: int) -> np.ndarray:
        west = o.easting + b.min_x
        east = o.easting + b.max_x
        north = o.northing - b.min_z
        south = o.northing - b.max_z
        dst_t = from_bounds(west, south, east, north, size, size)
        out = np.zeros((3, size, size), dtype=np.uint8)
        with rasterio.open(path) as src:
            for band in range(3):
                reproject(
                    source=rasterio.band(src, band + 1),
                    destination=out[band],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_t,
                    dst_crs="EPSG:32611",
                    resampling=Resampling.bilinear,
                )
        return np.transpose(out, (1, 2, 0))

    return fn


def write_terrain_outputs(
    terrain: Terrain,
    grid: TileGrid,
    albedo: AlbedoFn,
    processed: Path,
    assets: Path,
    texture_px: int,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Heightmap + meta (processed and assets copies) + tiles."""
    meta = write_heightmap(terrain, assets / "terrain" / "heightmap.png")
    write_json(processed / "terrain_meta.json", meta)
    write_json(assets / "terrain" / "terrain_meta.json", meta)
    infos, tris = build_terrain_tiles(terrain, grid, albedo, assets / "terrain", texture_px=texture_px)
    return infos, tris, meta


def run_real(processed: Path, assets: Path, grid: TileGrid, dem_path: Path, naip_path: Path) -> tuple[Terrain, list[dict[str, Any]], int]:
    from pipeline.build_buildings import flatten_terrain_for_heroes, load_heroes

    terrain = terrain_from_raster(dem_path, grid.extent)
    flatten_terrain_for_heroes(terrain, load_heroes())
    infos, tris, _ = write_terrain_outputs(terrain, grid, raster_albedo(naip_path), processed, assets, texture_px=1536)
    return terrain, infos, tris
