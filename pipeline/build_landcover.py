"""Per-tile land-cover splat masks for high-frequency terrain detail textures (HD build).

Two RGB PNGs per terrain tile, same footprint as the tile's albedo (north up, pixel (0, 0) at
(min_x, min_z)), 8-bit weights that sum to 255 over the six channels:

    terrain/splat_r{r}_c{c}_a.png   R lawn (irrigated grass, sports fields, golf)
                                    G chaparral / coastal sage scrub (dry native cover)
                                    B bare dirt / graded soil / sand
    terrain/splat_r{r}_c{c}_b.png   R paved (asphalt, concrete, rooftops)
                                    G water (pools, ponds, streams)
                                    B tree canopy (dense dark vegetation)

The weights start from a soft color classification of the imagery (Sentinel-2 / NAIP true
color) and are then overridden by vector data where it is authoritative: road surfaces,
sidewalks, driveways and building footprints (paved), Overture/OSM water and swimming pools
(water) and Overture land_use classes (golf / parks / pitches -> lawn, bunkers and
construction -> dirt, nature reserves -> no lawn). RGB (no alpha) so browsers never
premultiply the weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from pipeline.common import Extent, TileGrid, log

CHANNELS = ("lawn", "chaparral", "dirt", "paved", "water", "canopy")

LANDUSE_LAWN = {"fairway", "green", "tee", "golf_course", "driving_range", "pitch", "playground", "park", "grass", "garden", "recreation_ground", "dog_park", "schoolyard", "flowerbed", "cemetery", "track"}
LANDUSE_DIRT = {"bunker", "construction", "brownfield", "greenfield"}
LANDUSE_WILD = {"nature_reserve", "forest", "meadow", "rough"}


def _smooth(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def classify_imagery(img: np.ndarray) -> np.ndarray:
    """(h, w, 3) uint8 true color -> (h, w, 6) float soft memberships (not normalized)."""
    f = img.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    s = r + g + b + 1e-3
    v = s / 3.0
    exg = (2 * g - r - b) / s  # excess green index
    warm = (r - b) / s
    sat = (np.max(f, axis=-1) - np.min(f, axis=-1)) / (np.max(f, axis=-1) + 1e-3)
    canopy = _smooth(exg, 0.0, 0.08) * (1 - _smooth(v, 55, 85))
    lawn = _smooth(exg, 0.04, 0.12) * _smooth(v, 60, 90)
    chap = (1 - _smooth(exg, 0.06, 0.12)) * _smooth(exg, -0.06, 0.02) * (1 - _smooth(v, 115, 150)) * (1 - canopy)
    dirt = _smooth(warm, 0.04, 0.12) * _smooth(v, 110, 150) * (1 - _smooth(exg, 0.0, 0.05))
    paved = (1 - _smooth(sat, 0.10, 0.22)) * _smooth(v, 85, 115) * (1 - _smooth(exg, 0.01, 0.06))
    water = _smooth(b - r, 8, 25) * (1 - _smooth(v, 80, 120)) * (1 - _smooth(exg, 0.02, 0.08))
    out = np.stack([lawn, chap, dirt, paved, water, canopy], axis=-1)
    out[..., 1] += 0.05  # default to dry scrub where nothing is certain (SoCal hillsides)
    return out


def _raster(polys: list[Any], b: Extent, size: int) -> np.ndarray:
    """Binary mask (size, size) float of polygons (scene coords, holes honored)."""
    if not polys:
        return np.zeros((size, size), np.float32)
    m = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(m)
    sx = size / b.width
    sz = size / b.depth

    def px(c: np.ndarray) -> list[tuple[float, float]]:
        return [((x - b.min_x) * sx, (z - b.min_z) * sz) for x, z in c[:, :2]]

    for g in polys:
        for p in getattr(g, "geoms", [g]):
            if p.geom_type != "Polygon" or p.is_empty:
                continue
            dr.polygon(px(np.asarray(p.exterior.coords)), fill=255)
            for h in p.interiors:
                dr.polygon(px(np.asarray(h.coords)), fill=0)
    return np.asarray(m, dtype=np.float32) / 255.0


def write_splat_masks(
    grid: TileGrid,
    albedo: dict[str, np.ndarray],
    out_dir: Path,
    size: int,
    paved: Any,
    water: list[Any],
    landuse: list[tuple[Any, str]],
    lawn_extra: list[Any] | None = None,
) -> dict[str, list[str]]:
    """Write splat_{tid}_a.png / _b.png for every tile. `paved` is a (multi)polygon (scene),
    `water` a list of polygons, `landuse` (polygon, class) pairs. Returns tile id -> paths."""
    import shapely

    out: dict[str, list[str]] = {}
    lu_geoms = [g for g, _ in landuse]
    lu_tree = shapely.STRtree(lu_geoms) if lu_geoms else None
    w_tree = shapely.STRtree(water) if water else None
    lx_tree = shapely.STRtree(lawn_extra) if lawn_extra else None
    shares = np.zeros(6)
    for r, c in grid.iter():
        tid = grid.tile_id(r, c)
        b = grid.bounds(r, c)
        bx = shapely.box(b.min_x, b.min_z, b.max_x, b.max_z)
        img = np.asarray(Image.fromarray(albedo[tid]).resize((size, size), Image.BILINEAR))
        wts = classify_imagery(img)
        # land use
        if lu_tree is not None:
            idx = lu_tree.query(bx, predicate="intersects")
            groups: dict[str, list[Any]] = {"lawn": [], "dirt": [], "wild": []}
            for i in idx:
                cls = landuse[int(i)][1]
                if cls in LANDUSE_LAWN:
                    groups["lawn"].append(lu_geoms[int(i)])
                elif cls in LANDUSE_DIRT:
                    groups["dirt"].append(lu_geoms[int(i)])
                elif cls in LANDUSE_WILD:
                    groups["wild"].append(lu_geoms[int(i)])
            m = _raster(groups["lawn"], b, size)
            wts[..., 0] = wts[..., 0] * (1 - m) + m * np.maximum(wts[..., 0], 0.6 + 0.4 * (1 - wts[..., 5]))
            wts[..., 1] *= 1 - 0.8 * m
            wts[..., 2] *= 1 - 0.6 * m
            m = _raster(groups["dirt"], b, size)
            wts[..., 2] = np.maximum(wts[..., 2], m)
            wts[..., 0] *= 1 - 0.8 * m
            m = _raster(groups["wild"], b, size)
            wts[..., 1] += m * wts[..., 0] * 0.8
            wts[..., 0] *= 1 - 0.8 * m
        if lx_tree is not None:
            m = _raster([lawn_extra[int(i)] for i in lx_tree.query(bx, predicate="intersects")], b, size)  # type: ignore[index]
            wts[..., 0] = np.maximum(wts[..., 0], m * 0.7)
            wts[..., 1] = np.maximum(wts[..., 1], m * 0.3)
        # paved (roads, walks, driveways, roofs) and water are authoritative
        pv = shapely.clip_by_rect(paved, b.min_x - 5, b.min_z - 5, b.max_x + 5, b.max_z + 5) if paved is not None else None
        m = _raster([pv] if pv is not None and not pv.is_empty else [], b, size)
        m = ndimage.gaussian_filter(m, 0.6)
        wts = wts * (1 - m[..., None])
        wts[..., 3] += m * 1.0
        if w_tree is not None:
            m = _raster([water[int(i)] for i in w_tree.query(bx, predicate="intersects")], b, size)
            m = ndimage.gaussian_filter(m, 0.6)
            wts = wts * (1 - m[..., None])
            wts[..., 4] += m
        wts = ndimage.gaussian_filter(wts, (0.7, 0.7, 0))
        wts = np.maximum(wts, 0)
        tot = wts.sum(axis=-1, keepdims=True)
        wts = np.where(tot > 1e-6, wts / np.maximum(tot, 1e-6), np.array([0, 1, 0, 0, 0, 0], np.float32))
        q = np.floor(wts * 255.0).astype(np.int32)
        rem = 255 - q.sum(axis=-1)
        am = np.argmax(wts, axis=-1)
        np.put_along_axis(q, am[..., None], np.take_along_axis(q, am[..., None], -1) + rem[..., None], -1)
        q = np.clip(q, 0, 255).astype(np.uint8)
        shares += q.reshape(-1, 6).sum(axis=0)
        pa = out_dir / f"splat_{tid}_a.png"
        pb = out_dir / f"splat_{tid}_b.png"
        Image.fromarray(q[..., :3]).save(pa, optimize=True)
        Image.fromarray(q[..., 3:]).save(pb, optimize=True)
        out[tid] = [f"terrain/{pa.name}", f"terrain/{pb.name}"]
    shares /= max(shares.sum(), 1)
    log("landcover splat: " + ", ".join(f"{n} {s:.0%}" for n, s in zip(CHANNELS, shares, strict=True)) + f" ({size}px per tile)")
    return out


def load_landuse_scene(raw: Path) -> list[tuple[Any, str]]:
    """Overture base/land_use polygons (cache) in scene coordinates with their class."""
    p = raw / "overture" / "land_use.parquet"
    if not p.exists():
        return []
    import pyarrow.parquet as pq
    import shapely

    from pipeline.geo import scene_origin

    t = pq.read_table(p, columns=["geometry", "class"]).to_pandas()
    gs = shapely.from_wkb(t["geometry"].to_numpy())
    import geopandas as gpd

    gdf = gpd.GeoDataFrame({"class": t["class"].astype(str)}, geometry=gs, crs="EPSG:4326").to_crs("EPSG:32611")
    o = scene_origin()
    out = []
    for g, cls in zip(gdf.geometry, gdf["class"], strict=True):
        if g is None or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        out.append((shapely.affinity.affine_transform(g, [1, 0, 0, -1, -o.easting, o.northing]), cls))
    return out


def load_water_scene(raw: Path) -> list[Any]:
    """Overture base/water polygons (all classes incl. swimming pools) in scene coordinates."""
    p = raw / "overture" / "water.parquet"
    if not p.exists():
        return []
    import geopandas as gpd
    import pyarrow.parquet as pq
    import shapely

    from pipeline.geo import scene_origin

    t = pq.read_table(p, columns=["geometry", "class"]).to_pandas()
    gdf = gpd.GeoDataFrame({"class": t["class"].astype(str)}, geometry=shapely.from_wkb(t["geometry"].to_numpy()), crs="EPSG:4326").to_crs("EPSG:32611")
    o = scene_origin()
    out = []
    for g in gdf.geometry:
        if g is None:
            continue
        if g.geom_type in ("LineString", "MultiLineString"):
            g = g.buffer(1.5)  # streams / drains as thin water ribbons
        if g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        out.append(shapely.affinity.affine_transform(g, [1, 0, 0, -1, -o.easting, o.northing]))
    return out
