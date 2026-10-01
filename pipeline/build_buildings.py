"""Buildings: footprints -> typed, heighted features + 4x4 glTF tiles (spec 5.2).

`prepare_buildings` and `build_building_tiles` take in-memory GeoDataFrames
(EPSG:32611) with OSM-style tag columns, so real OSM footprints and the
synthetic footprints share the code.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import mapbox_earcut as earcut
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

from pipeline.build_terrain import Terrain
from pipeline.common import TileGrid, log
from pipeline.config import assumption
from pipeline.geo import scene_origin
from pipeline.glb import MeshData, write_glb

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


def building_type(building_tag: Any, area_m2: float, in_school: bool = False, amenity: Any = None, shop: Any = None) -> str:
    """Map OSM tags to one of house/apartments/commercial/school/other."""
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
    # building=yes and unknown values: small footprints are houses, big ones commercial.
    if area_m2 < house_max:
        return "house" if area_m2 >= 40.0 else "other"
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


def _hip_roof(rect: np.ndarray, eave: float, peak: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hip roof on a rectangle (4,2 x,z corners in order)."""
    c = rect.astype(np.float64)
    L = np.hypot(*(c[1] - c[0]))
    W = np.hypot(*(c[2] - c[1]))
    if L < W:
        c = np.roll(c, -1, axis=0)
        L, W = W, L
    m = c.mean(axis=0)
    a = (c[1] - c[0]) / max(L, 1e-9)
    r = max(0.0, (L - W) / 2.0)
    R0 = m - a * r
    R1 = m + a * r

    def p3(p: np.ndarray, y: float) -> list[float]:
        return [p[0], y, p[1]]

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
        if n[1] < 0:
            n = -n
        n = n / max(np.linalg.norm(n), 1e-9)
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


def building_mesh(poly_xz: Polygon, base_y: float, height: float, hip: bool, roof_h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mesh for one footprint in scene (x,z) coordinates.

    Returns pos (N,3), nrm (N,3), tri (M,3) and is_roof (N,) bool.
    """
    # Orient CCW in (x,z) math coordinates.
    poly_xz = orient(poly_xz, sign=1.0)
    top = base_y + height
    bottom = base_y - WALL_SINK_M
    if hip:
        rect_poly = orient(poly_xz.minimum_rotated_rectangle, sign=1.0)
        rect = np.asarray(rect_poly.exterior.coords)[:4, :2]
        eave = base_y + max(height - roof_h, 2.5)
        wp, wn, wt = _walls([np.asarray(rect_poly.exterior.coords)[:, :2]], bottom, eave)
        rp, rn, rt = _hip_roof(rect, eave, max(top, eave + 0.5))
    else:
        rings = [np.asarray(poly_xz.exterior.coords)[:, :2]] + [np.asarray(r.coords)[:, :2] for r in poly_xz.interiors]  # holes are CW after orient(): normals face into the hole
        wp, wn, wt = _walls(rings, bottom, top)
        rp, rn, rt = _flat_cap(poly_xz, top)
    pos = np.concatenate([wp, rp])
    nrm = np.concatenate([wn, rn])
    tri = np.concatenate([wt, rt + len(wp)])
    is_roof = np.concatenate([np.zeros(len(wp), dtype=bool), np.ones(len(rp), dtype=bool)])
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
    types, heights, levels, rules = [], [], [], []
    for i in range(len(gdf)):
        in_school = _clean(gdf["school_id"].iloc[i]) is not None
        t = building_type(btag.iloc[i], float(areas[i]), in_school, amen.iloc[i], shop.iloc[i])
        h, lv, rule = building_height(t, htag.iloc[i], ltag.iloc[i])
        types.append(t)
        heights.append(h)
        levels.append(lv)
        rules.append(rule)
    gdf["type"] = types
    gdf["height_m"] = np.asarray(heights, dtype=np.float64)
    gdf["levels"] = levels
    gdf["height_rule"] = rules
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
    keep = ["id", "type", "height_m", "base_elev_m", "levels", "address", "name", "area_m2", "centroid_x", "centroid_z", "block_group", "school_id", "tile", "parcel_apn", "parcel_land_use", "parcel_year_built", "height_rule", "geometry"]
    return gpd.GeoDataFrame(gdf[keep], geometry="geometry", crs="EPSG:32611")


def write_buildings_geojson(bdf: gpd.GeoDataFrame, path: Path) -> None:
    out = bdf.drop(columns=["height_rule"]).to_crs("EPSG:4326")
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


def hero_overrides() -> list[dict[str, Any]]:
    """Hero assets (spec 5.2.6): hero_overrides/<name>.json -> {"glb": "<file>.glb", "replaces_building_ids": [..] | "school_id": ".."}."""
    out = []
    if not HERO_DIR.exists():
        return out
    for js in sorted(HERO_DIR.glob("*.json")):
        try:
            cfg = json.loads(js.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"bad hero override {js}: {e}") from e
        glb = HERO_DIR / str(cfg.get("glb", js.stem + ".glb"))
        if glb.exists():
            cfg["_glb_path"] = glb
            cfg["name"] = js.stem
            out.append(cfg)
    return out


def build_building_tiles(bdf: gpd.GeoDataFrame, grid: TileGrid, out_dir: Path) -> tuple[dict[str, dict[str, Any]], int, list[dict[str, Any]]]:
    """Merge buildings into one mesh per tile with `_BUILDING_ID` + `COLOR_0` (spec 5.2.4).

    Returns (tile id -> {path, min_y, max_y, count}, triangles, hero list).
    """
    roof_h = float(assumption("buildings.hip_roof_height_m"))
    heroes = hero_overrides()
    replaced: set[int] = set()
    hero_out = []
    for h in heroes:
        ids = set(int(i) for i in h.get("replaces_building_ids", []))
        if h.get("school_id"):
            ids |= set(bdf.loc[bdf["school_id"] == h["school_id"], "id"].astype(int))
        replaced |= ids
        dst = out_dir / f"hero_{h['name']}.glb"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(h["_glb_path"], dst)
        hero_out.append({"name": h["name"], "glb": f"buildings/{dst.name}", "replaces_building_ids": sorted(ids)})

    acc: dict[str, dict[str, list[np.ndarray]]] = {}
    for row in bdf.itertuples(index=False):
        if int(row.id) in replaced:
            continue
        poly = utm_poly_to_scene(row.geometry)
        if poly.area < 1.0:
            continue
        hip = use_hip_roof(row.type, poly)
        pos, nrm, tri, is_roof = building_mesh(poly, float(row.base_elev_m), float(row.height_m), hip, roof_h)
        if len(tri) == 0:
            continue
        col = np.where(is_roof[:, None], np.array(ROOF_COLORS[row.type]), np.array(WALL_COLORS[row.type]))
        a = acc.setdefault(row.tile, {"pos": [], "nrm": [], "tri": [], "col": [], "bid": [], "n": [0]})
        a["tri"].append(tri + a["n"][0])
        a["n"][0] += len(pos)
        a["pos"].append(pos)
        a["nrm"].append(nrm)
        a["col"].append(col)
        a["bid"].append(np.full(len(pos), float(row.id)))
    tiles: dict[str, dict[str, Any]] = {}
    total = 0
    for r, c in grid.iter():
        tid = grid.tile_id(r, c)
        path = out_dir / f"buildings_{tid}.glb"
        a = acc.get(tid)
        if a is None:
            write_glb(path, [])
            tiles[tid] = {"path": f"buildings/{path.name}", "min_y": None, "max_y": None, "count": 0}
            continue
        pos = np.concatenate(a["pos"]).astype(np.float32)
        col = np.concatenate(a["col"]).astype(np.uint8)
        col = np.column_stack([col, np.full(len(col), 255, dtype=np.uint8)])
        mesh = MeshData(
            name=f"buildings_{tid}",
            positions=pos,
            normals=np.concatenate(a["nrm"]).astype(np.float32),
            indices=np.concatenate(a["tri"]).reshape(-1).astype(np.uint32),
            colors=col,
            custom={"_BUILDING_ID": np.concatenate(a["bid"]).astype(np.float32)},
            roughness=0.9,
        )
        tris = write_glb(path, [mesh])
        total += tris
        tiles[tid] = {"path": f"buildings/{path.name}", "min_y": float(pos[:, 1].min()), "max_y": float(pos[:, 1].max()), "count": int(len(a["bid"]))}
    log(f"buildings: {len(bdf):,} footprints, {total:,} triangles in {sum(1 for t in tiles.values() if t['count'])} tiles")
    return tiles, total, hero_out


# ---------------------------------------------------------------------------
# Real mode loader
# ---------------------------------------------------------------------------


def load_osm_buildings(raw_geojson: Path) -> gpd.GeoDataFrame:
    """OSM building features (WGS84 GeoJSON from fetch_osm) -> EPSG:32611 with osm_sort_key."""
    gdf = gpd.read_file(raw_geojson)
    gdf = gdf.to_crs("EPSG:32611")
    if "element" in gdf.columns and "id" in gdf.columns:
        gdf["osm_sort_key"] = gdf["element"].astype(str) + ":" + gdf["id"].astype(str).str.zfill(12)
    elif "osmid" in gdf.columns:
        gdf["osm_sort_key"] = gdf["osmid"].astype(str).str.zfill(12)
    return gdf
