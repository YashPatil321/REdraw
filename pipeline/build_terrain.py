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
TERRAIN_TRIANGLE_BUDGET = 2_000_000  # LOD0 over all tiles (LOD1/LOD2 are far cheaper)
HEIGHTMAP_SIZE_PX = 1009  # Unreal Landscape friendly size (63 quads * 16 + 1)
DEM_SPACING_M = 10.0
REAL_DEM_SPACING_M = 2.0  # the 3DEP lidar DEM is 1-2 m; the scene grid keeps 2 m

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
        # A finer DEM (e.g. 2 m lidar) is averaged onto the grid instead of point-sampled.
        finer = src.crs is not None and src.crs.is_projected and abs(src.transform.a) < spacing / 1.5
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs="EPSG:32611",
            dst_nodata=np.nan,
            resampling=Resampling.average if finer else Resampling.bilinear,
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
    Image.fromarray(px.astype("<u2")).save(png_path)
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
    """Regular-grid step (m) that keeps a full-extent grid mesh under `budget` triangles."""
    area = grid.extent.width * grid.extent.depth
    step = math.sqrt(2.0 * area / budget)
    return max(step, DEM_SPACING_M)


def terrain_tile_mesh(terrain: Terrain, b: Extent, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Regular grid mesh for one tile (LOD1 / LOD2). Returns positions, normals, uvs, indices."""
    nx = max(1, int(math.ceil(b.width / step)))
    nz = max(1, int(math.ceil(b.depth / step)))
    xs = np.linspace(b.min_x, b.max_x, nx + 1)
    zs = np.linspace(b.min_z, b.max_z, nz + 1)
    gx, gz = np.meshgrid(xs, zs)
    gy = terrain.sample(gx, gz)
    pos = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)
    nrm = terrain.normals(gx.ravel(), gz.ravel(), h=max(terrain.spacing, step / 2)).astype(np.float32)
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


def _up_winding(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Make every triangle counter-clockwise seen from +y (normal up)."""
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    ny = (b[:, 2] - a[:, 2]) * (c[:, 0] - a[:, 0]) - (b[:, 0] - a[:, 0]) * (c[:, 2] - a[:, 2])
    tri = tri.copy()
    flip = ny < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri


def add_skirts(pos: np.ndarray, nrm: np.ndarray, uv: np.ndarray, tri: np.ndarray, depth: float) -> tuple[np.ndarray, ...]:
    """Hang a vertical skirt `depth` m below every boundary edge of a tile mesh, so tiles of
    different LODs (and RTIN T-junctions between neighbouring tiles) never show cracks."""
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = e[cnt[inv.reshape(-1)] == 1]  # directed as in their triangle (CCW from above)
    if not len(bnd):
        return pos, nrm, uv, tri
    n0 = len(pos)
    k = len(bnd)
    top_a, top_b = bnd[:, 0], bnd[:, 1]
    low = pos[np.concatenate([top_a, top_b])].copy()
    low[:, 1] -= depth
    pos2 = np.concatenate([pos, low])
    nrm2 = np.concatenate([nrm, nrm[np.concatenate([top_a, top_b])]])
    uv2 = np.concatenate([uv, uv[np.concatenate([top_a, top_b])]])
    la = n0 + np.arange(k)
    lb = n0 + k + np.arange(k)
    # boundary edge a->b is CCW from above, so the outside is to its right; the quad
    # (a, la, lb, b) seen from outside is counter-clockwise.
    skirt = np.concatenate([np.stack([top_a, la, lb], 1), np.stack([top_a, lb, top_b], 1)])
    return pos2, nrm2, uv2, np.concatenate([tri, skirt])


RTIN_GRID = 513  # samples per tile side for LOD0 (2^9 + 1): ~2.2 m x 1.5 m on 1.1 km x 0.8 km tiles
LOD_STEPS_M = {1: 10.0, 2: 25.0}  # regular-grid spacing of the coarser levels
SKIRT_DEPTH_M = {0: 3.0, 1: 6.0, 2: 12.0}
LOD_TEXTURE_DIV = {0: 1, 1: 2, 2: 4}  # embedded albedo size per LOD (px / div)


@dataclass
class TerrainBuild:
    """Result of build_terrain_tiles: tile infos (manifest), triangle counts and the LOD0
    surface resampled on the terrain grid (`render`), which every draped mesh follows."""

    infos: list[dict[str, Any]]
    triangles: dict[str, int]
    render: Terrain
    lod0_max_error_m: float
    albedo: dict[str, np.ndarray]  # tile id -> (px, px, 3) uint8, north up

    @property
    def total(self) -> int:
        return int(self.triangles.get("lod0", 0))


def _rtin_tile(terrain: Terrain, b: Extent) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = RTIN_GRID
    xs = np.linspace(b.min_x, b.max_x, n)
    zs = np.linspace(b.min_z, b.max_z, n)
    gx, gz = np.meshgrid(xs, zs)
    h = terrain.sample(gx, gz)
    from pipeline.rtin import rtin_errors

    return h, rtin_errors(h), np.stack([gx, gz], axis=-1)


def _lod0_threshold(errors: list[np.ndarray], budget: int) -> float:
    """Smallest global max-error (m) whose RTIN meshes fit `budget` triangles (bisection)."""
    from pipeline.rtin import rtin_count

    lo, hi = 0.005, 50.0
    if sum(rtin_count(e, lo) for e in errors) <= budget:
        return lo
    for _ in range(22):
        mid = math.sqrt(lo * hi)
        if sum(rtin_count(e, mid) for e in errors) > budget:
            lo = mid
        else:
            hi = mid
    return hi


def build_terrain_tiles(
    terrain: Terrain,
    grid: TileGrid,
    albedo: AlbedoFn,
    out_dir: Path,
    texture_px: int = 512,
    budget: int = TERRAIN_TRIANGLE_BUDGET,
    save_loose_albedo: bool = True,
) -> TerrainBuild:
    """Write terrain/terrain_r{r}_c{c}.glb (LOD0, adaptive RTIN from the full-resolution DEM,
    all tiles together <= `budget` triangles) plus terrain_r{r}_c{c}_lod1.glb / _lod2.glb
    (regular grids, LOD_STEPS_M), each with skirts and an embedded albedo (LOD0 at the full
    `texture_px`, coarser LODs smaller). Loose full-size albedo_r{r}_c{c}.jpg too."""
    from matplotlib.tri import LinearTriInterpolator, Triangulation

    from pipeline.rtin import rtin_mesh

    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = grid.iter()
    rt = [_rtin_tile(terrain, grid.bounds(r, c)) for r, c in tiles]
    thr = _lod0_threshold([e for _, e, _ in rt], int(budget * 0.97))  # ~3% for skirts
    render = Terrain(terrain.elev.copy(), terrain.extent, terrain.spacing)
    rows, cols = terrain.elev.shape
    tx = terrain.extent.min_x + np.arange(cols) * terrain.spacing
    tz = terrain.extent.min_z + np.arange(rows) * terrain.spacing
    infos: list[dict[str, Any]] = []
    tris = {"lod0": 0, "lod1": 0, "lod2": 0}
    albedo_imgs: dict[str, np.ndarray] = {}
    for (r, c), (h, err, gxz) in zip(tiles, rt, strict=True):
        b = grid.bounds(r, c)
        tid = grid.tile_id(r, c)
        img = albedo(b, texture_px)
        albedo_imgs[tid] = img
        if save_loose_albedo:
            (out_dir / f"albedo_{tid}.jpg").write_bytes(jpeg_bytes(img, quality=90))
        # LOD0: RTIN
        v, t = rtin_mesh(err, thr)
        x = gxz[v[:, 1], v[:, 0], 0]
        z = gxz[v[:, 1], v[:, 0], 1]
        y = h[v[:, 1], v[:, 0]]
        pos = np.column_stack([x, y, z])
        t = _up_winding(pos, t)
        # resample this LOD0 surface onto the terrain grid nodes inside the tile
        j0, j1 = np.searchsorted(tx, b.min_x), np.searchsorted(tx, b.max_x, side="right")
        i0, i1 = np.searchsorted(tz, b.min_z), np.searchsorted(tz, b.max_z, side="right")
        if j1 > j0 and i1 > i0:
            interp = LinearTriInterpolator(Triangulation(x, z, t), y)
            qx, qz = np.meshgrid(tx[j0:j1], tz[i0:i1])
            q = np.asarray(interp(qx.ravel(), qz.ravel()).filled(np.nan)).reshape(qx.shape)
            blk = render.elev[i0:i1, j0:j1]
            render.elev[i0:i1, j0:j1] = np.where(np.isfinite(q), q, blk).astype(np.float32)
        nrm = terrain.normals(x, z, h=max(terrain.spacing, 2.0))
        uv = np.column_stack([(x - b.min_x) / b.width, (z - b.min_z) / b.depth])
        lods = []
        meshes = {0: add_skirts(pos, nrm, uv, t, SKIRT_DEPTH_M[0])}
        for lod, step in LOD_STEPS_M.items():
            p2, n2, u2, i2 = terrain_tile_mesh(terrain, b, step)
            meshes[lod] = add_skirts(p2, n2, u2, i2.reshape(-1, 3), SKIRT_DEPTH_M[lod])
        ymin, ymax = float(pos[:, 1].min()), float(pos[:, 1].max())
        for lod, (p_, n_, u_, i_) in meshes.items():
            px = max(16, texture_px // LOD_TEXTURE_DIV[lod])
            small = img if px == texture_px else np.asarray(Image.fromarray(img).resize((px, px), Image.LANCZOS))
            name = f"terrain_{tid}.glb" if lod == 0 else f"terrain_{tid}_lod{lod}.glb"
            n_tri = write_glb(
                out_dir / name,
                [MeshData(name=f"terrain_{tid}_lod{lod}", positions=p_.astype(np.float32), normals=n_.astype(np.float32), uvs=u_.astype(np.float32), indices=i_.reshape(-1).astype(np.uint32), texture_jpeg=jpeg_bytes(small, quality=88))],
            )
            tris[f"lod{lod}"] += n_tri
            lods.append(
                {
                    "lod": lod,
                    "path": f"terrain/{name}",
                    "triangles": n_tri,
                    "spacing_m": round(min(b.width, b.depth) / (RTIN_GRID - 1), 3) if lod == 0 else LOD_STEPS_M[lod],
                    **({"max_error_m": round(thr, 4)} if lod == 0 else {}),
                    "texture_px": px,
                }
            )
        infos.append(
            {
                "id": tid,
                "row": r,
                "col": c,
                "bounds": {**b.as_dict(), "min_y": ymin, "max_y": ymax},
                "terrain": f"terrain/terrain_{tid}.glb",
                "terrain_lods": lods,
                "albedo": f"terrain/albedo_{tid}.jpg",
            }
        )
    log(f"terrain: {len(infos)} tiles; LOD0 RTIN max error {thr:.3f} m, {tris['lod0']:,} tris; LOD1 {tris['lod1']:,}; LOD2 {tris['lod2']:,}")
    return TerrainBuild(infos=infos, triangles=tris, render=render, lod0_max_error_m=thr, albedo=albedo_imgs)


def jpeg_bytes(img: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img.astype(np.uint8)[..., :3])).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img.astype(np.uint8))).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


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
    budget: int = TERRAIN_TRIANGLE_BUDGET,
) -> tuple[TerrainBuild, dict[str, Any]]:
    """Heightmap + meta (processed and assets copies) + LOD tiles."""
    meta = write_heightmap(terrain, assets / "terrain" / "heightmap.png")
    tb = build_terrain_tiles(terrain, grid, albedo, assets / "terrain", texture_px=texture_px, budget=budget)
    meta = dict(meta, texture_px=texture_px, dem_spacing_m=terrain.spacing, lod0_max_error_m=tb.lod0_max_error_m, lod_triangles=tb.triangles)
    write_json(processed / "terrain_meta.json", meta)
    write_json(assets / "terrain" / "terrain_meta.json", meta)
    return tb, meta


MAX_TEXTURE_PX = 4096  # spec 5.1.3


def native_texture_px(imagery_path: Path, grid: TileGrid) -> int:
    """Smallest power of two that keeps the imagery at (or above) its pixel size per tile."""
    import rasterio

    with rasterio.open(imagery_path) as src:
        res = float(abs(src.transform.a)) if src.crs is not None and src.crs.is_projected else 2.5
    tile_m = max(grid.extent.width / grid.cols, grid.extent.depth / grid.rows)
    need = tile_m / max(res, 0.1)
    return int(min(MAX_TEXTURE_PX, 2 ** math.ceil(math.log2(max(need, 16)))))


def terrain_sources(dem_path: Path, imagery_path: Path) -> list[dict[str, Any]]:
    """DEM + imagery provenance from the `.source.json` sidecars / data/raw/aws_sources.json."""
    from pipeline.sources import raw_sources

    return [s for s in raw_sources(dem_path.parent, dem_path, imagery_path) if s.get("kind") in ("dem", "imagery")]


def run_real(
    processed: Path, assets: Path, grid: TileGrid, dem_path: Path, naip_path: Path
) -> tuple[Terrain, TerrainBuild, list[dict[str, Any]], AlbedoFn, int]:
    """Real terrain: DEM -> grid (REAL_DEM_SPACING_M), imagery albedo at native resolution.
    Returns (terrain, terrain build, sources, albedo fn, texture px)."""
    from pipeline.build_buildings import flatten_terrain_for_heroes, load_heroes

    terrain = terrain_from_raster(dem_path, grid.extent, REAL_DEM_SPACING_M)
    flatten_terrain_for_heroes(terrain, load_heroes())
    px = native_texture_px(naip_path, grid)
    alb = raster_albedo(naip_path)
    tb, meta = write_terrain_outputs(terrain, grid, alb, processed, assets, texture_px=px)
    sources = terrain_sources(dem_path, naip_path)
    meta = dict(meta, sources=sources)
    write_json(processed / "terrain_meta.json", meta)
    write_json(assets / "terrain" / "terrain_meta.json", meta)
    for s in sources:
        log(f"terrain source ({s.get('kind')}): {s.get('name')}")
    return terrain, tb, sources, alb, px
