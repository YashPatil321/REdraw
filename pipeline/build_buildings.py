"""Buildings: footprints -> typed, heighted features + 4x4 glTF tiles (spec 5.2).

`prepare_buildings` and `build_building_tiles` take in-memory GeoDataFrames
(EPSG:32611) with OSM-style tag columns, so real OSM footprints and the
synthetic footprints share the code.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import mapbox_earcut as earcut
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient

from pipeline.build_terrain import Terrain
from pipeline.common import TileGrid, log
from pipeline.config import assumption
from pipeline.geo import scene_origin
from pipeline.glb import MeshData, load_glb_meshes, write_glb

HERO_DIR = Path(__file__).resolve().parent / "hero_overrides"
WALL_SINK_M = 0.5  # walls start this far below the lowest footprint corner so slopes never show gaps
BUILDING_TYPES = ("house", "apartments", "commercial", "school", "other")

WALL_COLORS: dict[str, tuple[int, int, int]] = {
    "house": (226, 212, 188),
    "apartments": (218, 204, 182),
    "commercial": (205, 200, 190),
    "school": (214, 194, 160),
    "other": (185, 182, 175),
}
ROOF_COLORS: dict[str, tuple[int, int, int]] = {
    "house": (168, 92, 64),
    "apartments": (160, 88, 62),
    "commercial": (150, 150, 148),
    "school": (140, 140, 135),
    "other": (130, 128, 124),
}

HOUSE_TAGS = {"house", "detached", "semidetached_house", "terrace", "bungalow", "villa", "cabin", "static_caravan"}
APARTMENT_TAGS = {"apartments", "dormitory", "residential_complex"}
COMMERCIAL_TAGS = {"commercial", "retail", "office", "supermarket", "industrial", "warehouse", "hotel", "kiosk", "mall", "hospital", "civic", "government", "public", "fire_station", "church", "religious"}
SCHOOL_TAGS = {"school", "kindergarten", "university", "college"}
OTHER_TAGS = {"garage", "garages", "shed", "carport", "roof", "hut", "greenhouse", "service", "utility", "construction", "parking", "transformer_tower", "water_tower"}


# ---------------------------------------------------------------------------
# Rules (unit tested)
# ---------------------------------------------------------------------------


def _clean(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s or None


def parse_height_m(v: Any) -> float | None:
    """OSM height tag -> meters ('12', '12 m', "30'", '30 ft', "30'6\"")."""
    s = _clean(v)
    if s is None:
        return None
    s = s.lower().replace(",", ".")
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*'\s*(?:(\d+(?:\.\d+)?)\s*\")?\s*$", s)
    if m:
        ft = float(m.group(1)) + (float(m.group(2)) / 12.0 if m.group(2) else 0.0)
        return ft * 0.3048
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None
    n = float(nums[0])
    if "ft" in s or "feet" in s:
        n *= 0.3048
    return n if n > 0 else None


def parse_levels(v: Any) -> int | None:
    s = _clean(v)
    if s is None:
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None
    n = int(round(float(nums[0])))
    return n if n > 0 else None


def building_type(
    building_tag: Any,
    area_m2: float,
    in_school: bool = False,
    amenity: Any = None,
    shop: Any = None,
    landuse: Any = None,
    estate_context: bool = False,
) -> str:
    """Map OSM tags to one of house/apartments/commercial/school/other.

    `landuse` is the OSM/Overture land_use class the footprint centroid lies in (real mode);
    it only reclassifies generic building=yes footprints.
    """
    tag = (_clean(building_tag) or "yes").lower()
    amen = (_clean(amenity) or "").lower()
    if tag in SCHOOL_TAGS or amen in {"school", "kindergarten"} or in_school:
        return "school"
    if tag in HOUSE_TAGS:
        return "house"
    if tag in APARTMENT_TAGS:
        return "apartments"
    if tag in COMMERCIAL_TAGS or _clean(shop) is not None:
        return "commercial"
    if tag in OTHER_TAGS:
        return "other"
    house_max = float(assumption("pipeline.house_max_area_m2"))
    apt_min = float(assumption("pipeline.apartment_min_area_m2"))
    if tag == "residential":
        if area_m2 >= apt_min:
            return "apartments"
        return "house"
    lu = (_clean(landuse) or "").lower()
    if lu and lu in set(assumption("footprint_population.commercial_landuse_classes")):
        return "commercial" if area_m2 >= float(assumption("footprint_population.house_min_area_m2")) else "other"
    if lu and lu in set(assumption("footprint_population.nonhome_landuse_classes")):
        return "other"
    # building=yes and unknown values: small footprints are houses, big ones commercial.
    if area_m2 < house_max:
        return "house" if area_m2 >= float(assumption("footprint_population.house_min_area_m2")) else "other"
    if estate_context and area_m2 <= float(assumption("footprint_population.estate_house_max_area_m2")):
        return "house"  # large home in a purely residential neighbourhood (estate_context_flags)
    return "commercial"


def building_height(btype: str, height_tag: Any = None, levels_tag: Any = None) -> tuple[float, int | None, str]:
    """Height priority (spec 5.2.1): height tag, then levels * level_height, then class default.

    Returns (height_m, levels, rule) where rule is 'height', 'levels' or 'default'.
    """
    levels = parse_levels(levels_tag)
    h = parse_height_m(height_tag)
    if h is not None:
        return h, levels, "height"
    if levels is not None:
        return levels * float(assumption("buildings.level_height_m")), levels, "levels"
    return float(assumption(f"buildings.default_height_m.{btype}")), None, "default"


def rectangularity(poly: Polygon) -> float:
    """Footprint area / minimum rotated rectangle area (1.0 = perfect rectangle)."""
    if poly.is_empty or poly.area <= 0:
        return 0.0
    rect = poly.minimum_rotated_rectangle
    return float(poly.area / rect.area) if rect.area > 0 else 0.0


ROOF_SHAPE_KIND = {
    "flat": "flat",
    "skillion": "flat",
    "hipped": "hipped",
    "half_hipped": "hipped",
    "side_hipped": "hipped",
    "pyramidal": "hipped",
    "gabled": "gabled",
    "side_gabled": "gabled",
    "saltbox": "gabled",
    "gambrel": "gabled",
    "mansard": "hipped",
}


def roof_kind(btype: str, poly: Polygon, roof_shape: Any = None) -> str:
    """'flat', 'hipped' or 'gabled'. An OSM roof:shape tag wins (gabled/hipped/flat and close
    relatives); otherwise houses get a hip roof when roughly rectangular (spec 5.2.3)."""
    tag = (_clean(roof_shape) or "").lower()
    if tag in ROOF_SHAPE_KIND:
        kind = ROOF_SHAPE_KIND[tag]
        if kind != "flat" and len(poly.interiors) > 0:
            return "flat"  # pitched roofs are built on the min rotated rectangle; courtyards stay flat
        return kind
    return "hipped" if use_hip_roof(btype, poly) else "flat"


def parse_colour(v: Any) -> tuple[int, int, int] | None:
    """OSM building:colour / roof:colour ('#778899', '#fff', 'white') -> RGB, None if unknown."""
    s = _clean(v)
    if s is None:
        return None
    from PIL import ImageColor

    try:
        rgb = ImageColor.getrgb(s if not re.fullmatch(r"[0-9a-fA-F]{6}", s) else "#" + s)
    except ValueError:
        return None
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def use_hip_roof(btype: str, poly: Polygon) -> bool:
    """Houses get a hip roof when the footprint is roughly rectangular (spec 5.2.3)."""
    if btype != "house":
        return False
    if len(poly.interiors) > 0:
        return False
    return rectangularity(poly) >= float(assumption("buildings.hip_roof_rectangularity"))


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


def _fix_winding(pos: np.ndarray, tri: np.ndarray, desired: np.ndarray) -> np.ndarray:
    v0, v1, v2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    flip = np.einsum("ij,ij->i", n, desired) < 0
    tri = tri.copy()
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri


def _walls(rings: list[np.ndarray], y0: float, y1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertical quads for each ring segment (flat shaded). rings are (N,2) x,z, closed or not."""
    P, N, T = [], [], []
    base = 0
    for ring in rings:
        r = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
        if len(r) < 3:
            continue
        p0 = r
        p1 = np.roll(r, -1, axis=0)
        d = p1 - p0
        ln = np.hypot(d[:, 0], d[:, 1])
        keep = ln > 1e-6
        p0, p1, d, ln = p0[keep], p1[keep], d[keep], ln[keep]
        k = len(p0)
        # 2D outward normal for a CCW ring in (x,z) math coords is (dz, -dx)
        n2 = np.column_stack([d[:, 1], -d[:, 0]]) / ln[:, None]
        quad = np.empty((k, 4, 3))
        quad[:, 0] = np.column_stack([p0[:, 0], np.full(k, y0), p0[:, 1]])
        quad[:, 1] = np.column_stack([p1[:, 0], np.full(k, y0), p1[:, 1]])
        quad[:, 2] = np.column_stack([p1[:, 0], np.full(k, y1), p1[:, 1]])
        quad[:, 3] = np.column_stack([p0[:, 0], np.full(k, y1), p0[:, 1]])
        nrm = np.repeat(np.column_stack([n2[:, 0], np.zeros(k), n2[:, 1]])[:, None, :], 4, axis=1)
        q = base + np.arange(k)[:, None] * 4
        tri = np.concatenate([np.hstack([q, q + 1, q + 2]), np.hstack([q, q + 2, q + 3])])
        P.append(quad.reshape(-1, 3))
        N.append(nrm.reshape(-1, 3))
        T.append(tri)
        base += 4 * k
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    pos = np.concatenate(P)
    nrm = np.concatenate(N)
    tri = np.concatenate(T)
    desired = (nrm[tri[:, 0]] + nrm[tri[:, 1]] + nrm[tri[:, 2]]) / 3.0
    return pos, nrm, _fix_winding(pos, tri, desired)


def _flat_cap(poly: Polygon, y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rings = [np.asarray(poly.exterior.coords)[:-1, :2]] + [np.asarray(r.coords)[:-1, :2] for r in poly.interiors]
    verts = np.concatenate(rings).astype(np.float64)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    idx = earcut.triangulate_float64(verts, ends).reshape(-1, 3).astype(np.int64)
    pos = np.column_stack([verts[:, 0], np.full(len(verts), y), verts[:, 1]])
    nrm = np.tile([0.0, 1.0, 0.0], (len(verts), 1))
    if len(idx) == 0:
        return pos, nrm, idx
    return pos, nrm, _fix_winding(pos, idx, np.tile([0.0, 1.0, 0.0], (len(idx), 1)))


def _hip_roof(rect: np.ndarray, eave: float, peak: float, gabled: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hip (or gabled) roof on a rectangle (4,2 x,z corners in order). The ridge runs along the
    long axis; a gabled roof has vertical triangular gable ends instead of hipped ends."""
    c = rect.astype(np.float64)
    L = np.hypot(*(c[1] - c[0]))
    W = np.hypot(*(c[2] - c[1]))
    if L < W:
        c = np.roll(c, -1, axis=0)
        L, W = W, L
    m = c.mean(axis=0)
    a = (c[1] - c[0]) / max(L, 1e-9)
    r = L / 2.0 if gabled else max(0.0, (L - W) / 2.0)
    R0 = m - a * r
    R1 = m + a * r

    def p3(p: np.ndarray, y: float) -> list[float]:
        return [p[0], y, p[1]]

    # Gabled: r = L/2 puts the ridge ends above the short-edge midpoints, so the two end
    # triangles are vertical gable walls; hipped: they slope.
    faces = [
        [p3(c[0], eave), p3(c[1], eave), p3(R1, peak), p3(R0, peak)],
        [p3(c[1], eave), p3(c[2], eave), p3(R1, peak)],
        [p3(c[2], eave), p3(c[3], eave), p3(R0, peak), p3(R1, peak)],
        [p3(c[3], eave), p3(c[0], eave), p3(R0, peak)],
    ]
    P, N, T = [], [], []
    base = 0
    for f in faces:
        fp = np.asarray(f)
        n = np.cross(fp[1] - fp[0], fp[2] - fp[0])
        if np.linalg.norm(n) < 1e-9 and len(fp) == 4:
            n = np.cross(fp[2] - fp[0], fp[3] - fp[0])
        n = n / max(np.linalg.norm(n), 1e-9)
        if abs(n[1]) < 1e-6:  # vertical gable end: face away from the rectangle center
            fc = fp.mean(axis=0)
            if (fc[0] - m[0]) * n[0] + (fc[2] - m[1]) * n[2] < 0:
                n = -n
        elif n[1] < 0:
            n = -n
        tris = [[0, 1, 2]] if len(fp) == 3 else [[0, 1, 2], [0, 2, 3]]
        t = np.asarray(tris) + base
        P.append(fp)
        N.append(np.tile(n, (len(fp), 1)))
        T.append(t)
        base += len(fp)
    pos = np.concatenate(P)
    nrm = np.concatenate(N)
    tri = np.concatenate(T)
    desired = nrm[tri[:, 0]]
    return pos, nrm, _fix_winding(pos, tri, desired)


def building_mesh(
    poly_xz: Polygon, base_y: float, height: float, hip: bool, roof_h: float, roof: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mesh for one footprint in scene (x,z) coordinates.

    `roof` ('flat', 'hipped', 'gabled') overrides the legacy `hip` flag when given.
    Returns pos (N,3), nrm (N,3), tri (M,3) and is_roof (N,) bool.
    """
    kind = roof or ("hipped" if hip else "flat")
    # Orient CCW in (x,z) math coordinates.
    poly_xz = orient(poly_xz, sign=1.0)
    top = base_y + height
    bottom = base_y - WALL_SINK_M
    if kind in ("hipped", "gabled"):
        rect_poly = orient(poly_xz.minimum_rotated_rectangle, sign=1.0)
        rect = np.asarray(rect_poly.exterior.coords)[:4, :2]
        eave = base_y + max(height - roof_h, 2.5)
        # walls stop at the eave; gable-end triangles above it are part of the roof faces
        wp, wn, wt = _walls([np.asarray(rect_poly.exterior.coords)[:, :2]], bottom, eave)
        rp, rn, rt = _hip_roof(rect, eave, max(top, eave + 0.5), gabled=kind == "gabled")
    else:
        rings = [np.asarray(poly_xz.exterior.coords)[:, :2]] + [np.asarray(r.coords)[:, :2] for r in poly_xz.interiors]  # holes are CW after orient(): normals face into the hole
        wp, wn, wt = _walls(rings, bottom, top)
        rp, rn, rt = _flat_cap(poly_xz, top)
    pos = np.concatenate([wp, rp])
    nrm = np.concatenate([wn, rn])
    tri = np.concatenate([wt, rt + len(wp)])
    # vertical gable-end triangles are wall-coloured
    is_roof = np.concatenate([np.zeros(len(wp), dtype=bool), rn[:, 1] > 1e-3])
    return pos, nrm, tri, is_roof


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def _polys(geom: Any) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


def utm_poly_to_scene(poly: Polygon) -> Polygon:
    o = scene_origin()

    def tr(coords: Any) -> list[tuple[float, float]]:
        c = np.asarray(coords)[:, :2]
        return list(zip(c[:, 0] - o.easting, o.northing - c[:, 1], strict=True))

    return Polygon(tr(poly.exterior.coords), [tr(r.coords) for r in poly.interiors])


def _col(gdf: pd.DataFrame, name: str) -> pd.Series:
    if name in gdf.columns:
        return gdf[name]
    return pd.Series([None] * len(gdf), index=gdf.index, dtype=object)


def prepare_buildings(
    footprints: gpd.GeoDataFrame,
    terrain: Terrain,
    grid: TileGrid,
    school_areas: gpd.GeoDataFrame | None = None,
    block_groups: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Footprints (EPSG:32611, OSM tag columns) -> contract building features (still EPSG:32611).

    Optional `school_id` column on footprints is kept. `school_areas` (EPSG:32611,
    columns geometry + school_id) marks buildings inside campuses as schools.
    `block_groups` (EPSG:32611, columns geometry + GEOID) fills block_group.
    """
    gdf = footprints.copy()
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geometry.geom_type == "Polygon"]
    gdf = gdf[gdf.geometry.area >= 10.0].copy()
    # keep only footprints whose centroid lies in the terrain extent
    o = scene_origin()
    cent = gdf.geometry.centroid
    cx = cent.x.to_numpy() - o.easting
    cz = o.northing - cent.y.to_numpy()
    inside = np.asarray(grid.extent.contains(cx, cz))
    gdf = gdf[inside].copy()
    cx, cz = cx[inside], cz[inside]
    gdf = gdf.reset_index(drop=True)

    if "school_id" not in gdf.columns:
        gdf["school_id"] = None
    if school_areas is not None and len(school_areas):
        pts = gpd.GeoDataFrame(geometry=gdf.geometry.centroid, crs=gdf.crs)
        j = gpd.sjoin(pts, school_areas[["school_id", "geometry"]].rename(columns={"school_id": "_sid"}), how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")]
        fill = j["_sid"].reindex(gdf.index)
        gdf["school_id"] = gdf["school_id"].where(gdf["school_id"].notna(), fill)

    if block_groups is not None and len(block_groups):
        pts = gpd.GeoDataFrame(geometry=gdf.geometry.centroid, crs=gdf.crs)
        j = gpd.sjoin(pts, block_groups[["GEOID", "geometry"]], how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")]
        gdf["block_group"] = j["GEOID"].reindex(gdf.index).fillna("").astype(str)
    elif "block_group" not in gdf.columns:
        gdf["block_group"] = ""

    areas = gdf.geometry.area.to_numpy()
    btag, amen, shop = _col(gdf, "building"), _col(gdf, "amenity"), _col(gdf, "shop")
    htag, ltag = _col(gdf, "height"), _col(gdf, "building:levels")
    lu = _col(gdf, "landuse_class")
    estate = gdf["estate_context"].fillna(False).astype(bool).to_numpy() if "estate_context" in gdf.columns else np.zeros(len(gdf), bool)
    types, heights, levels, rules = [], [], [], []
    for i in range(len(gdf)):
        in_school = _clean(gdf["school_id"].iloc[i]) is not None
        t = building_type(btag.iloc[i], float(areas[i]), in_school, amen.iloc[i], shop.iloc[i], lu.iloc[i], bool(estate[i]))
        h, lv, rule = building_height(t, htag.iloc[i], ltag.iloc[i])
        types.append(t)
        heights.append(h)
        levels.append(lv)
        rules.append(rule)
    gdf["type"] = types
    gdf["height_m"] = np.asarray(heights, dtype=np.float64)
    gdf["levels"] = levels
    gdf["height_rule"] = rules
    # Internal (not in buildings.geojson): raw tag, levels estimate, roof shape, colours.
    lh = float(assumption("buildings.level_height_m"))
    gdf["building_tag"] = [(_clean(v) or "yes").lower() for v in btag]
    gdf["levels_est"] = [
        lv if lv is not None else (max(1, int(round(h / lh))) if rule == "height" else None)
        for lv, h, rule in zip(levels, heights, rules, strict=True)
    ]
    gdf["roof_shape"] = [_clean(v) for v in _col(gdf, "roof:shape")]
    gdf["roof_height_m"] = [parse_height_m(v) for v in _col(gdf, "roof:height")]
    gdf["wall_rgb"] = [parse_colour(v) for v in _col(gdf, "building:colour")]
    gdf["roof_rgb"] = [parse_colour(v) for v in _col(gdf, "roof:colour")]
    gdf["area_m2"] = areas
    gdf["centroid_x"] = cx
    gdf["centroid_z"] = cz
    gdf["base_elev_m"] = terrain.sample(cx, cz).astype(np.float64)
    r, c = grid.tile_of(cx, cz)
    gdf["tile"] = [grid.tile_id(int(a), int(b)) for a, b in zip(r, c, strict=True)]

    street = _col(gdf, "addr:street")
    num = _col(gdf, "addr:housenumber")
    addr = []
    for n_, s_ in zip(num, street, strict=True):
        n2, s2 = _clean(n_), _clean(s_)
        addr.append(f"{n2} {s2}" if n2 and s2 else (s2 or None))
    gdf["address"] = addr
    gdf["name"] = [_clean(v) for v in _col(gdf, "name")]
    gdf["school_id"] = [_clean(v) for v in gdf["school_id"]]

    # Stable ids: by OSM id when present (real), else input order (synthetic).
    if "osm_sort_key" in gdf.columns:
        gdf = gdf.sort_values("osm_sort_key", kind="stable").reset_index(drop=True)
    gdf["id"] = np.arange(1, len(gdf) + 1, dtype=np.int64)
    gdf["parcel_apn"] = None
    gdf["parcel_land_use"] = None
    gdf["parcel_year_built"] = None
    keep = ["id", "type", "height_m", "base_elev_m", "levels", "address", "name", "area_m2", "centroid_x", "centroid_z", "block_group", "school_id", "tile", "parcel_apn", "parcel_land_use", "parcel_year_built", *INTERNAL_COLUMNS, "geometry"]
    return gpd.GeoDataFrame(gdf[keep], geometry="geometry", crs="EPSG:32611")


# prepare_buildings columns used inside the pipeline only (not part of buildings.geojson)
INTERNAL_COLUMNS = ["height_rule", "building_tag", "levels_est", "roof_shape", "roof_height_m", "wall_rgb", "roof_rgb"]


def write_buildings_geojson(bdf: gpd.GeoDataFrame, path: Path) -> None:
    out = bdf.drop(columns=[c for c in INTERNAL_COLUMNS if c in bdf.columns]).to_crs("EPSG:4326")
    feats = []
    for row in out.itertuples(index=False):
        d = row._asdict()
        geom = d.pop("geometry")
        props = {}
        for k, v in d.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            if isinstance(v, float) and math.isnan(v):
                v = None
            props[k] = v
        props["levels"] = None if props["levels"] is None else int(props["levels"])
        coords = [[[round(x, 7), round(y, 7)] for x, y in np.asarray(geom.exterior.coords)[:, :2]]]
        coords += [[[round(x, 7), round(y, 7)] for x, y in np.asarray(r.coords)[:, :2]] for r in geom.interiors]
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": coords}, "properties": props})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")


HERO_JSON = "hero_overrides.json"
HERO_BLEND_M = 40.0  # terrain blend ring outside a hero footprint when flattening


@dataclass
class Hero:
    """One hero model placed in the scene (spec 5.2.6, pipeline/hero_overrides/hero_overrides.json)."""

    key: str
    name: str
    school_id: str | None
    x: float
    z: float
    rotation_deg: float
    radius: float
    glb: Path
    replaces: list[int]
    kind: str  # buildings.geojson type for a new entry: school or commercial
    building_id: int = 0
    base_y: float = 0.0
    meshes: list[MeshData] = field(default_factory=list)


def load_heroes(hero_dir: Path = HERO_DIR) -> list[Hero]:
    """Parse hero_overrides.json (a list, or {"heroes": [...]}). Missing json -> []."""
    from pipeline.geo import latlon_to_scene

    path = hero_dir / HERO_JSON
    if not path.exists():
        return []
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"bad {path}: {e}") from e
    entries = cfg.get("heroes", []) if isinstance(cfg, dict) else cfg
    out = []
    for e in entries:
        key = str(e.get("id") or e.get("school_id") or Path(str(e.get("glb", "hero"))).stem)
        glb = hero_dir / str(e.get("glb", f"{key}.glb"))
        if not glb.exists():
            log(f"WARNING hero override {key}: {glb} missing; skipped")
            continue
        x, z = latlon_to_scene(float(e["lat"]), float(e["lon"]))
        rep = e.get("replaces")
        replaces = [int(i) for i in rep] if isinstance(rep, list) and all(isinstance(i, (int, float)) for i in rep) else []
        sid = e.get("school_id")
        out.append(
            Hero(
                key=key,
                name=str(e.get("name") or key.replace("_", " ").title()),
                school_id=str(sid) if sid else None,
                x=x,
                z=z,
                rotation_deg=float(e.get("rotation_deg", 0.0)),
                radius=float(e.get("footprint_radius_m", 100.0)),
                glb=glb,
                replaces=replaces,
                kind=str(e.get("type") or ("school" if sid else "commercial")),
            )
        )
    return out


def flatten_terrain_for_heroes(terrain: Terrain, heroes: list[Hero]) -> None:
    """Flatten the DEM to the center elevation inside each hero footprint (blend ring outside)."""
    if not heroes:
        return
    rows, cols = terrain.elev.shape
    xs = terrain.extent.min_x + np.arange(cols) * terrain.spacing
    zs = terrain.extent.min_z + np.arange(rows) * terrain.spacing
    gx, gz = np.meshgrid(xs, zs)
    for h in heroes:
        y0 = float(terrain.sample(h.x, h.z))
        d = np.hypot(gx - h.x, gz - h.z)
        w = np.clip(1.0 - (d - h.radius) / HERO_BLEND_M, 0.0, 1.0)
        terrain.elev[:] = (terrain.elev * (1 - w) + y0 * w).astype(np.float32)


def hero_transform(meshes: list[MeshData], x: float, y: float, z: float, rotation_deg: float) -> list[MeshData]:
    """Place local-meter meshes (origin = campus center, +x east, +y up, +z south) in the scene.

    rotation_deg is a right-handed rotation about +y (three.js `rotation.y` convention):
    seen from above, positive angles turn the model counter-clockwise (east toward north).
    """
    t = math.radians(rotation_deg)
    c, s = math.cos(t), math.sin(t)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    out = []
    for m in meshes:
        pos = np.asarray(m.positions, dtype=np.float64) @ R.T + np.array([x, y, z])
        nrm = None if m.normals is None else np.asarray(m.normals, dtype=np.float64) @ R.T
        out.append(dataclasses.replace(m, positions=pos.astype(np.float32), normals=None if nrm is None else nrm.astype(np.float32)))
    return out


def apply_heroes(bdf: gpd.GeoDataFrame, heroes: list[Hero], terrain: Terrain, grid: TileGrid) -> gpd.GeoDataFrame:
    """Drop auto footprints under each hero, give the hero a building id and geojson entry.

    School heroes reuse the id of that school's largest footprint (kept in the geojson with the
    hero outline); others get a new id. Hero meshes are loaded and placed (h.meshes).
    """
    if not heroes:
        return bdf
    o = scene_origin()
    bdf = bdf.copy()
    next_id = int(bdf["id"].max()) + 1 if len(bdf) else 1
    for h in heroes:
        h.base_y = float(terrain.sample(h.x, h.z))
        local = load_glb_meshes(h.glb)
        h.meshes = hero_transform(local, h.x, h.base_y, h.z, h.rotation_deg)
        allp = np.concatenate([m.positions for m in h.meshes]) if h.meshes else np.array([[h.x, h.base_y, h.z]])
        hull = MultiPoint([(float(p[0]), float(p[2])) for p in allp[:: max(1, len(allp) // 4000)]]).convex_hull
        if not isinstance(hull, Polygon) or hull.area < 1.0:
            hull = Point(h.x, h.z).buffer(h.radius, 24)
        hull_utm = Polygon([(px + o.easting, o.northing - pz) for px, pz in np.asarray(hull.exterior.coords)])
        near = np.hypot(bdf["centroid_x"] - h.x, bdf["centroid_z"] - h.z) <= h.radius
        drop = near | bdf["id"].isin(h.replaces)
        keep_id = None
        if h.school_id:
            sch = bdf[(bdf["school_id"] == h.school_id)].sort_values("area_m2", ascending=False)
            if len(sch):
                keep_id = int(sch["id"].iloc[0])
        r, c = grid.tile_of(np.array([h.x]), np.array([h.z]))
        row = {
            "type": h.kind if h.kind in BUILDING_TYPES else "other",
            "height_m": float(allp[:, 1].max() - h.base_y),
            "base_elev_m": h.base_y,
            "levels": None,
            "address": None,
            "name": h.name,
            "area_m2": float(hull.area),
            "centroid_x": h.x,
            "centroid_z": h.z,
            "block_group": "",
            "school_id": h.school_id,
            "tile": grid.tile_id(int(r[0]), int(c[0])),
            "parcel_apn": None,
            "parcel_land_use": None,
            "parcel_year_built": None,
            "height_rule": "hero",
            "geometry": hull_utm,
        }
        if keep_id is not None:
            idx = bdf.index[bdf["id"] == keep_id][0]
            bg = bdf.at[idx, "block_group"]
            for k, v in row.items():
                bdf.at[idx, k] = v
            bdf.at[idx, "block_group"] = bg
            drop &= bdf["id"] != keep_id
            h.building_id = keep_id
        else:
            h.building_id = next_id
            next_id += 1
            new = gpd.GeoDataFrame([{"id": h.building_id, **row}], geometry="geometry", crs=bdf.crs)
            bdf = gpd.GeoDataFrame(pd.concat([bdf, new], ignore_index=True), geometry="geometry", crs=bdf.crs)
            drop = np.concatenate([np.asarray(drop), [False]])
        bdf = bdf[~np.asarray(drop)].reset_index(drop=True)
        log(f"hero {h.key}: {int(np.asarray(drop).sum())} footprints replaced, building_id {h.building_id}, {sum(m.triangle_count for m in h.meshes):,} triangles")
    bdf["id"] = bdf["id"].astype(np.int64)
    return gpd.GeoDataFrame(bdf, geometry="geometry", crs="EPSG:32611")


# Rendering budget (spec 10.3), not a real-world fact.
BUILDING_TRIANGLE_BUDGET = 1_450_000
SIMPLIFY_STEPS_M = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def estimate_triangles(poly: Polygon, kind: str) -> int:
    """Triangle count building_mesh will produce for a footprint (walls + roof)."""
    if kind in ("hipped", "gabled"):
        return 8 + 6
    n_ext = len(poly.exterior.coords) - 1
    n_int = sum(len(r.coords) - 1 for r in poly.interiors)
    n = n_ext + n_int
    return 2 * n + (n - 2 + 2 * len(poly.interiors))


def simplify_tolerance_for_budget(polys: list[Polygon], kinds: list[str], budget: int) -> float:
    """Smallest SIMPLIFY_STEPS_M tolerance (m) that keeps flat-roof footprints under `budget`."""
    import shapely

    flat = [p for p, k in zip(polys, kinds, strict=True) if k == "flat"]
    pitched = sum(estimate_triangles(p, k) for p, k in zip(polys, kinds, strict=True) if k != "flat")
    arr = np.asarray(flat, dtype=object)
    for tol in SIMPLIFY_STEPS_M:
        simp = shapely.simplify(arr, tol, preserve_topology=True) if tol > 0 and len(arr) else arr
        total = pitched + sum(estimate_triangles(p, "flat") if isinstance(p, Polygon) and not p.is_empty else 0 for p in simp)
        if total <= budget:
            return tol
    return SIMPLIFY_STEPS_M[-1]


def build_building_tiles(
    bdf: gpd.GeoDataFrame,
    grid: TileGrid,
    out_dir: Path,
    heroes: list[Hero] | None = None,
    budget: int = BUILDING_TRIANGLE_BUDGET,
) -> tuple[dict[str, dict[str, Any]], int, list[dict[str, Any]]]:
    """Merge buildings into one mesh per tile with `_BUILDING_ID` + `COLOR_0` (spec 5.2.4).

    Hero models become extra primitives in their tile (own materials/colors, `_BUILDING_ID`).
    Returns (tile id -> {path, min_y, max_y, count}, triangles, hero list for the manifest).
    """
    roof_h_default = float(assumption("buildings.hip_roof_height_m"))
    heroes = heroes or []
    hero_ids = {h.building_id for h in heroes}
    work = bdf[~bdf["id"].isin(hero_ids)]
    polys = [utm_poly_to_scene(g) for g in work.geometry]
    shapes = work["roof_shape"] if "roof_shape" in work.columns else pd.Series([None] * len(work), index=work.index)
    kinds = [roof_kind(t, p, rs) for t, p, rs in zip(work["type"], polys, shapes, strict=True)]
    tol = simplify_tolerance_for_budget(polys, kinds, budget)
    if tol > 0:
        log(f"buildings: simplifying flat-roof footprints by {tol:.2f} m to stay under {budget:,} triangles")
    acc: dict[str, dict[str, list[np.ndarray]]] = {}
    for row, poly, kind in zip(work.itertuples(index=False), polys, kinds, strict=True):
        if poly.area < 1.0:
            continue
        if tol > 0 and kind == "flat":
            sp = poly.simplify(tol, preserve_topology=True)
            if isinstance(sp, Polygon) and sp.is_valid and sp.area > 0.5 * poly.area:
                poly = sp
        rh = getattr(row, "roof_height_m", None)
        roof_h = float(rh) if rh is not None and np.isfinite(rh) and 0.3 < float(rh) < float(row.height_m) else roof_h_default
        pos, nrm, tri, is_roof = building_mesh(poly, float(row.base_elev_m), float(row.height_m), kind == "hipped", roof_h, roof=kind)
        if len(tri) == 0:
            continue
        wall = getattr(row, "wall_rgb", None)
        roofc = getattr(row, "roof_rgb", None)
        wall_c = np.array(wall if isinstance(wall, tuple) else WALL_COLORS[row.type])
        roof_c = np.array(roofc if isinstance(roofc, tuple) else ROOF_COLORS[row.type])
        col = np.where(is_roof[:, None], roof_c, wall_c)
        a = acc.setdefault(row.tile, {"pos": [], "nrm": [], "tri": [], "col": [], "bid": [], "n": [0]})
        a["tri"].append(tri + a["n"][0])
        a["n"][0] += len(pos)
        a["pos"].append(pos)
        a["nrm"].append(nrm)
        a["col"].append(col)
        a["bid"].append(np.full(len(pos), float(row.id)))
    hero_by_tile: dict[str, list[MeshData]] = {}
    hero_out = []
    for h in heroes:
        r, c = grid.tile_of(np.array([h.x]), np.array([h.z]))
        tid = grid.tile_id(int(r[0]), int(c[0]))
        for k, m in enumerate(h.meshes):
            m2 = dataclasses.replace(m, name=f"hero_{h.key}_{k}", custom={"_BUILDING_ID": np.full(len(m.positions), float(h.building_id), dtype=np.float32)})
            hero_by_tile.setdefault(tid, []).append(m2)
        hero_out.append({"id": h.key, "name": h.name, "building_id": h.building_id, "tile": tid, "school_id": h.school_id, "triangles": sum(m.triangle_count for m in h.meshes)})
    tiles: dict[str, dict[str, Any]] = {}
    total = 0
    for r, c in grid.iter():
        tid = grid.tile_id(r, c)
        path = out_dir / f"buildings_{tid}.glb"
        a = acc.get(tid)
        meshes: list[MeshData] = []
        count = 0
        if a is not None:
            col = np.concatenate(a["col"]).astype(np.uint8)
            col = np.column_stack([col, np.full(len(col), 255, dtype=np.uint8)])
            meshes.append(
                MeshData(
                    name=f"buildings_{tid}",
                    positions=np.concatenate(a["pos"]).astype(np.float32),
                    normals=np.concatenate(a["nrm"]).astype(np.float32),
                    indices=np.concatenate(a["tri"]).reshape(-1).astype(np.uint32),
                    colors=col,
                    custom={"_BUILDING_ID": np.concatenate(a["bid"]).astype(np.float32)},
                    roughness=0.9,
                )
            )
            count = len(a["bid"])
        meshes += hero_by_tile.get(tid, [])
        tris = write_glb(path, meshes)
        total += tris
        ys = [m.positions[:, 1] for m in meshes if len(m.positions)]
        tiles[tid] = {
            "path": f"buildings/{path.name}",
            "min_y": float(min(y.min() for y in ys)) if ys else None,
            "max_y": float(max(y.max() for y in ys)) if ys else None,
            "count": int(count + len(hero_by_tile.get(tid, []))),
        }
    log(f"buildings: {len(bdf):,} footprints, {total:,} triangles in {sum(1 for t in tiles.values() if t['count'])} tiles, {len(heroes)} heroes")
    return tiles, total, hero_out


# ---------------------------------------------------------------------------
# Real mode loader
# ---------------------------------------------------------------------------


def load_landuse(raw: Path) -> gpd.GeoDataFrame | None:
    """OSM-derived land_use polygons (Overture base/land_use cache from fetch_aws), EPSG:32611
    with a `class` column; None if the cache is absent (e.g. Overpass-fetched raw data)."""
    path = raw / "overture" / "land_use.parquet"
    if not path.exists():
        return None
    import pyarrow.parquet as pq
    import shapely

    tab = pq.read_table(path, columns=["geometry", "class"]).to_pandas()
    geoms = shapely.from_wkb(tab["geometry"].to_numpy())
    gdf = gpd.GeoDataFrame({"class": tab["class"].astype(str)}, geometry=geoms, crs="EPSG:4326").to_crs("EPSG:32611")
    return gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)


def tag_landuse(gdf: gpd.GeoDataFrame, landuse: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    """Add `landuse_class`: the land_use class of the polygon containing each footprint centroid
    (smallest polygon wins when several overlap)."""
    gdf = gdf.copy()
    if landuse is None or not len(landuse):
        gdf["landuse_class"] = None
        return gdf
    lu = landuse.assign(_area=landuse.geometry.area).sort_values("_area")
    pts = gpd.GeoDataFrame(geometry=gdf.geometry.representative_point(), crs=gdf.crs)
    j = gpd.sjoin(pts, lu[["class", "_area", "geometry"]], how="left", predicate="within")
    j = j.sort_values("_area", na_position="last")
    j = j[~j.index.duplicated(keep="first")]
    gdf["landuse_class"] = j["class"].reindex(gdf.index)
    return gdf


def estate_context_flags(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """True where a footprint's neighbourhood is purely residential-sized (EPSG:32611 input):
    >= footprint_population.estate_context_min_neighbours neighbours within estate_context_radius_m
    sized house_min_area_m2..estate_house_max_area_m2, and that share >= estate_context_min_share."""
    from scipy.spatial import cKDTree

    if not len(gdf):
        return np.zeros(0, bool)
    fp = "footprint_population"
    lo, hi = float(assumption(f"{fp}.house_min_area_m2")), float(assumption(f"{fp}.estate_house_max_area_m2"))
    r = float(assumption(f"{fp}.estate_context_radius_m"))
    kmin, share = int(assumption(f"{fp}.estate_context_min_neighbours")), float(assumption(f"{fp}.estate_context_min_share"))
    c = gdf.geometry.centroid
    xy = np.column_stack([c.x.to_numpy(), c.y.to_numpy()])
    area = gdf.geometry.area.to_numpy()
    house_sized = (area >= lo) & (area <= hi)
    out = np.zeros(len(gdf), bool)
    for i, nb in enumerate(cKDTree(xy).query_ball_point(xy, r)):
        nb = [j for j in nb if j != i and area[j] >= lo]  # ignore sheds
        if not nb:
            continue
        n_house = int(house_sized[nb].sum())
        out[i] = n_house >= kmin and n_house / len(nb) >= share
    return out


def load_osm_buildings(raw_geojson: Path, landuse: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    """OSM building features (WGS84 GeoJSON from fetch_osm) -> EPSG:32611 with osm_sort_key
    (and `landuse_class` when land_use polygons are given)."""
    gdf = gpd.read_file(raw_geojson)
    gdf = gdf.to_crs("EPSG:32611")
    gdf = tag_landuse(gdf, landuse)
    gdf["estate_context"] = estate_context_flags(gdf)
    if "element" in gdf.columns and "id" in gdf.columns:
        gdf["osm_sort_key"] = gdf["element"].astype(str) + ":" + gdf["id"].astype(str).str.zfill(12)
    elif "osmid" in gdf.columns:
        gdf["osm_sort_key"] = gdf["osmid"].astype(str).str.zfill(12)
    return gdf
