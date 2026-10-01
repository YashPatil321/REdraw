"""Export the real footprints the hero campus models are built from (runs in the MAIN .venv).

    .venv/bin/python blender/extract_hero_sites.py

Reads (all cached by the pipeline's fetch step, never downloaded here):
- data/raw/osm_buildings.geojson        building footprints (+ height / levels / name tags)
- data/raw/osm_schools.geojson          campus polygons (Del Norte High School, Design39Campus)
- data/raw/overture/land_use.parquet    pitches, track, grass, pedestrian plazas, retail site polygons
- data/raw/overture/segment.parquet     service roads / parking aisles / footways
- data/raw/overture/place.parquet       (only to confirm the 4S Commons location)

Writes blender/build/hero_sites.json: per hero, the site polygon and every footprint
with its centroid inside the replacement circle, in LOCAL meters (x east, y north,
origin = hero center) - Blender's frame; glTF export turns +y north into -z.

The replacement circle follows the pipeline rule (pipeline/build_buildings.py
apply_heroes): every building whose centroid is within footprint_radius_m of the
hero center is dropped and replaced by the hero model. The center and radius are
chosen here so the circle holds only buildings that belong to the site, with the
radius in the middle of a gap between site and non-site centroids.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, mapping
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hero_layout import build_layout  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
OUT = REPO / "blender" / "build" / "hero_sites.json"
UTM = "EPSG:32611"
LEVEL_H = 3.2  # assumptions.yaml buildings.level_height_m

HEROES = [
    {
        "id": "del_norte_hs",
        "school_id": "del_norte_hs",
        "name": "Del Norte High School",
        "type": "school",
        "glb": "del_norte_hs.glb",
        "seed": (33.01402, -117.12246),
        "site": ("osm_school", "Del Norte High School"),
    },
    {
        "id": "design39",
        "school_id": "design39",
        "name": "Design39Campus",
        "type": "school",
        "glb": "design39.glb",
        "seed": (33.01867, -117.12186),
        "site": ("osm_school", "Design39Campus"),
    },
    {
        # The task brief guessed 33.0195,-117.1260 (that is the Target / Del Sur Town Center).
        # Overture places put "4S Commons Town Center" (10511 4S Commons Dr) at 33.01950,-117.11293,
        # with the Overture land_use retail polygon of the same name around it.
        "id": "4s_commons",
        "school_id": None,
        "name": "4S Commons Town Center",
        "type": "commercial",
        "glb": "4s_commons.glb",
        "seed": (33.0195, -117.1129),
        "site": ("landuse", "4S Commons Town Center"),
    },
]


def _need(path: Path) -> Path:
    if not path.exists():
        sys.exit(
            f"missing {path}\n"
            "Run the real-data pipeline fetch first (python pipeline/build_all.py), which caches OSM and\n"
            "Overture extracts in data/raw/. See docs/data_contract.md."
        )
    return path


def _height(row: pd.Series, kind: str) -> tuple[float, int | None]:
    def num(v: object) -> float | None:
        try:
            f = float(str(v).split(";")[0].replace("m", "").strip())
            return f if math.isfinite(f) and f > 0 else None
        except (TypeError, ValueError):
            return None

    h = num(row.get("height"))
    lv = num(row.get("building:levels"))
    levels = int(round(lv)) if lv else None
    if h is not None and h >= 2.5:
        return h, levels
    if levels:
        return levels * LEVEL_H + 1.0, levels
    return {"school": 7.0, "commercial": 7.5}.get(kind, 6.0), None


def _local(geom, cx: float, cy: float):  # noqa: ANN001, ANN202
    from shapely.affinity import translate

    return translate(geom, -cx, -cy)


def _poly_json(p: Polygon, nd: int = 2) -> dict:
    return {
        "exterior": [[round(x, nd), round(y, nd)] for x, y in list(p.exterior.coords)[:-1]],
        "holes": [[[round(x, nd), round(y, nd)] for x, y in list(r.coords)[:-1]] for r in p.interiors],
    }


def _polys(g) -> list[Polygon]:  # noqa: ANN001
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [x for x in getattr(g, "geoms", []) if isinstance(x, Polygon)]


def choose_circle(site: Polygon, cents: np.ndarray, in_site: np.ndarray) -> tuple[float, float, float, int]:
    """Pick (cx, cy, R) inside the site maximizing site buildings in the circle with no outsiders.

    R sits in the middle of the widest gap between the farthest included site centroid and
    the nearest excluded centroid (>= 3 m each side when possible).
    """
    minx, miny, maxx, maxy = site.bounds
    best = None
    for cx in np.arange(minx, maxx, 10.0):
        for cy in np.arange(miny, maxy, 10.0):
            if not site.contains(Point(cx, cy)):
                continue
            d = np.hypot(cents[:, 0] - cx, cents[:, 1] - cy)
            out_d = d[~in_site]
            r_out = float(out_d.min()) if len(out_d) else 1e9
            ins = np.sort(d[in_site & (d < r_out)])
            if not len(ins):
                continue
            # largest included set whose last centroid leaves a gap to the next centroid
            nxt = np.append(ins[1:], r_out)
            gaps = nxt - ins
            ok = np.where(gaps >= 6.0)[0]
            if not len(ok):
                continue
            k = int(ok[-1])
            n_in = k + 1
            r = float((ins[k] + nxt[k]) / 2)
            score = (n_in, -r)
            if best is None or score > best[0]:
                best = (score, cx, cy, min(r, ins[k] + 30.0), n_in)
    if best is None:
        raise RuntimeError("no valid hero circle found")
    _, cx, cy, r, n_in = best
    return float(cx), float(cy), float(r), int(n_in)


def main() -> None:
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    to_ll = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    b = gpd.read_file(_need(RAW / "osm_buildings.geojson")).to_crs(UTM)
    b = b[b.geometry.notna() & b.geometry.is_valid & (b.geometry.area > 8)].copy()
    b["cx"] = b.geometry.centroid.x
    b["cy"] = b.geometry.centroid.y
    schools = gpd.read_file(_need(RAW / "osm_schools.geojson")).to_crs(UTM)
    lu = gpd.read_parquet(_need(RAW / "overture" / "land_use.parquet")).to_crs(UTM)
    lu["nm"] = lu["names"].apply(lambda n: n.get("primary") if isinstance(n, dict) else None)
    seg = gpd.read_parquet(_need(RAW / "overture" / "segment.parquet")).to_crs(UTM)
    out = []
    for h in HEROES:
        sx, sy = to_utm.transform(h["seed"][1], h["seed"][0])
        kind, nm = h["site"]
        if kind == "osm_school":
            cand = schools[schools["name"] == nm]
        else:
            cand = lu[lu["nm"] == nm]
        if cand.empty:
            raise RuntimeError(f"{h['id']}: site polygon '{nm}' not found")
        site = cand.geometry.iloc[int(np.argmin(cand.distance(Point(sx, sy)).to_numpy()))]
        site = max(_polys(site), key=lambda p: p.area)
        near = b[b.distance(Point(sx, sy)) < 900].copy()
        cents = near[["cx", "cy"]].to_numpy()
        in_site = np.array([site.buffer(4.0).contains(Point(x, y)) for x, y in cents])
        cx, cy, R, n_in = choose_circle(site, cents, in_site)
        d = np.hypot(near["cx"] - cx, near["cy"] - cy)
        inc = near[d <= R]
        lon, lat = to_ll.transform(cx, cy)
        circle = Point(cx, cy).buffer(R + 40.0, 64)
        bl = []
        for _, r in inc.iterrows():
            tag = str(r.get("building") or "yes")
            kindb = "school" if h["type"] == "school" else "commercial"
            ht, lv = _height(r, kindb)
            for p in _polys(_local(r.geometry, cx, cy)):
                bl.append({
                    "osm_id": str(r.get("id") or r.get("osm_way_id") or ""),
                    "building": tag,
                    "name": None if pd.isna(r.get("name")) else str(r.get("name")),
                    "height_m": round(ht, 2),
                    "height_tagged": not pd.isna(r.get("height")),
                    "levels": lv,
                    **_poly_json(p),
                })
        # footprints that touch the site but stay as normal pipeline buildings (keep paving off them)
        others = near[(d > R) & near.intersects(circle)]
        keep = [_poly_json(p) for g in others.geometry for p in _polys(_local(g, cx, cy))]
        lus = lu[lu.intersects(circle) & lu.intersects(site.buffer(30))]
        landuse = []
        for _, r in lus.iterrows():
            if r["nm"] == nm:
                continue
            g = r.geometry.intersection(circle)
            for p in _polys(_local(g, cx, cy)):
                if p.area < 1.0:
                    continue
                landuse.append({"subtype": r["subtype"], "class": r["class"], "name": r["nm"] if isinstance(r["nm"], str) else None, **_poly_json(p, 2)})
        segs = []
        ss = seg[seg.intersects(circle)]
        for _, r in ss.iterrows():
            g = r.geometry.intersection(circle)
            lines = [g] if isinstance(g, LineString) else [x for x in getattr(g, "geoms", []) if isinstance(x, LineString)]
            for ln in lines:
                if ln.length < 2:
                    continue
                segs.append({
                    "class": r["class"], "subclass": r["subclass"] if isinstance(r["subclass"], str) else None,
                    "coords": [[round(x - cx, 2), round(y - cy, 2)] for x, y in ln.coords],
                })
        site_l = _local(site, cx, cy)
        rec = {
            "id": h["id"], "school_id": h["school_id"], "name": h["name"], "type": h["type"], "glb": h["glb"],
            "lat": round(lat, 6), "lon": round(lon, 6), "utm": [round(cx, 2), round(cy, 2)],
            "footprint_radius_m": round(R, 1), "n_buildings": len(inc),
            "site": _poly_json(max(_polys(site_l), key=lambda p: p.area)),
            "site_area_m2": round(site.area),
            "buildings": bl, "keep_out": keep, "landuse": landuse, "segments": segs,
        }
        rec["layout"] = build_layout(rec)
        lay = rec["layout"]
        print(f"   layout: {len(lay['surfaces'])} surfaces, {len(lay['paint'])} paint, {len(lay['trees'])} trees, "
              f"{len(lay['parked'])} parked, {len(lay['canopies'])} canopies, {len(lay['lamps'])} lamps")
        out.append(rec)
        print(f"{h['id']}: center {lat:.6f},{lon:.6f} R={R:.1f} m  buildings={len(inc)} "
              f"landuse={len(landuse)} segments={len(segs)} keep_out={len(keep)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"crs_note": "local meters, x east, y north, origin = hero center (utm)",
                               "source": "OSM (ODbL) via data/raw/osm_*.geojson; Overture Maps land_use/segment",
                               "heroes": out}, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    _ = unary_union, mapping  # kept for interactive debugging


if __name__ == "__main__":
    main()
