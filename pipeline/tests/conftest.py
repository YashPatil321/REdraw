"""Shared fixtures: one synthetic world per test session, plus helpers that write mock
"raw" files in the real-source formats so the real-mode build path can be tested offline."""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely.geometry import LineString, Point, box

from pipeline.common import region_extent, terrain_extent
from pipeline.config import load_yaml
from pipeline.geo import latlon_to_scene, scene_origin, scene_to_latlon, utm_to_lonlat


@pytest.fixture(scope="session")
def world():
    from pipeline.synthetic import make_world

    return make_world(region_extent(), terrain_extent(), load_yaml("schools.yaml")["schools"])


def _env(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, Path]:
    d = {"raw": root / "raw", "processed": root / "processed", "assets": root / "assets"}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("REDRAW_RAW_DIR", str(d["raw"]))
    monkeypatch.setenv("REDRAW_DATA_DIR", str(d["processed"]))
    monkeypatch.setenv("REDRAW_ASSETS_DIR", str(d["assets"]))
    return d


@pytest.fixture(scope="session")
def synthetic_build(world, tmp_path_factory):
    """Run `build_all --synthetic` once into a temp dir (reusing the session world)."""
    import copy

    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("synthetic")
    d = _env(mp, root)
    import pipeline.synthetic as syn

    mp.setattr(syn, "make_world", lambda *a, **k: copy.deepcopy(world))
    from pipeline.build_all import run_synthetic

    meta = run_synthetic(skip_draco=True)
    yield d, meta
    mp.undo()


def write_mock_raw(world, raw: Path) -> None:
    """Write the synthetic world as if it came from OSM / 3DEP / NAIP / ACS / LODES."""
    import rasterio
    from rasterio.transform import from_origin

    o = scene_origin()
    t = world.terrain
    # DEM GeoTIFF (EPSG:32611, pixel centers on the terrain grid)
    tr = from_origin(o.easting + t.extent.min_x - t.spacing / 2, o.northing - t.extent.min_z + t.spacing / 2, t.spacing, t.spacing)
    with rasterio.open(raw / "dem_3dep_10m.tif", "w", driver="GTiff", width=t.elev.shape[1], height=t.elev.shape[0], count=1, dtype="float32", crs="EPSG:32611", transform=tr) as f:
        f.write(t.elev, 1)
    # NAIP mosaic (RGB, 8 m to keep it small)
    ext = t.extent
    res = 8.0
    img = world.albedo(ext, int(ext.width / res))
    img = np.asarray(Image.fromarray(img).resize((int(ext.width / res), int(ext.depth / res))))
    tr2 = from_origin(o.easting + ext.min_x, o.northing - ext.min_z, res, res)
    with rasterio.open(raw / "naip_mosaic.tif", "w", driver="GTiff", width=img.shape[1], height=img.shape[0], count=3, dtype="uint8", crs="EPSG:32611", transform=tr2) as f:
        f.write(np.transpose(img, (2, 0, 1)))
    # OSM drive graph in WGS84, marked unsimplified (build_roads simplifies it)
    import osmnx as ox

    G = nx.MultiDiGraph(crs="EPSG:4326", simplified=False)
    for n, d in world.graph.nodes(data=True):
        lon, lat = utm_to_lonlat(d["x"], d["y"])
        G.add_node(n, x=lon, y=lat, street_count=world.graph.degree(n), **({"highway": d["highway"]} if "highway" in d else {}))
    for k, (u, v, d) in enumerate(world.graph.edges(data=True)):
        g = d["geometry"]
        geom = LineString([utm_to_lonlat(x, y) for x, y in g.coords])
        a = {kk: vv for kk, vv in d.items() if kk not in ("geometry", "osmid")}
        G.add_edge(u, v, **a, osmid=1000 + k, geometry=geom)
    # Cut the I 15 north terminus (as a bbox truncation would) to exercise the exit turnaround logic.
    nx_, nz_ = 3950.0, region_extent().min_z
    term = min(G.nodes, key=lambda n: (latlon_to_scene(G.nodes[n]["y"], G.nodes[n]["x"])[0] - nx_) ** 2 + (latlon_to_scene(G.nodes[n]["y"], G.nodes[n]["x"])[1] - nz_) ** 2)
    G.remove_node(term)
    for p in ("drive", "walk", "bike"):
        ox.io.save_graphml(G, raw / f"osm_{p}{'_raw' if p == 'drive' else ''}.graphml")
    # buildings (drop the synthetic school_id so the campus join is exercised)
    fp = world.footprints.drop(columns=["school_id"]).to_crs("EPSG:4326").copy()
    fp["element"] = "way"
    fp["id"] = np.arange(len(fp)) + 1
    fp.to_file(raw / "osm_buildings.geojson", driver="GeoJSON")
    # schools: campuses (named) + one OSM-only elementary school + one preschool to ignore
    rows = []
    for sid, poly in world.layout.campuses.items():
        name = next(s["name"] for s in load_yaml("schools.yaml")["schools"] if s["id"] == sid)
        rows.append({"name": name, "amenity": "school", "geometry": poly})
    rows.append({"name": "Test Ridge Elementary School", "amenity": "school", "geometry": Point(-2600, 2600).buffer(80)})
    rows.append({"name": "Little Sprouts Preschool", "amenity": "school", "geometry": Point(-2000, 2000).buffer(40)})
    sg = gpd.GeoDataFrame([{k: v for k, v in r.items() if k != "geometry"} for r in rows], geometry=[syn_to_utm(r["geometry"]) for r in rows], crs="EPSG:32611")
    sg.to_crs("EPSG:4326").to_file(raw / "osm_schools.geojson", driver="GeoJSON")
    # TIGER block groups: 4 quadrants of the region
    rext = region_extent()
    quads = []
    geoids = []
    for i, (x0, x1) in enumerate(((rext.min_x - 300, 0.0), (0.0, rext.max_x + 300))):
        for j, (z0, z1) in enumerate(((rext.min_z - 300, 0.0), (0.0, rext.max_z + 300))):
            quads.append(syn_to_utm(box(x0, z0, x1, z1)))
            geoids.append(f"06073017{i}0{j}1")
    bg = gpd.GeoDataFrame({"GEOID": geoids}, geometry=quads, crs="EPSG:32611").to_crs("EPSG:4269")
    shp_dir = raw / "tiger_tmp"
    shp_dir.mkdir(exist_ok=True)
    bg.to_file(shp_dir / "tl_2022_06_bg.shp")
    with zipfile.ZipFile(raw / "tl_2022_06_bg.zip", "w") as z:
        for f in shp_dir.iterdir():
            z.write(f, f.name)
    # ACS
    from pipeline.build_population import acs_variables

    rng = np.random.default_rng(1)
    acs = pd.DataFrame({"GEOID": geoids, "state": "06", "county": "073", "tract": [g[5:11] for g in geoids], "block group": [g[11] for g in geoids]})
    for v in acs_variables():
        acs[v] = rng.integers(10, 400, size=len(geoids))
    acs["B11016_001E"] = 3500
    acs["B11005_001E"] = 3500
    acs["B11005_002E"] = 1700
    acs["B08301_001E"] = 5600
    acs["B08301_021E"] = 1000
    acs.loc[0, "B19001_017E"] = np.nan  # a suppressed cell
    acs.to_csv(raw / "acs5_2022_bg_06073.csv", index=False)
    # LODES: blocks = BG + 3 digits; work blocks inside (near commercial) and far away (outside)
    blocks, lat, lon = [], [], []
    for g in geoids:
        blocks.append(g + "001")
        x = -2000.0 if g[8] == "0" else 2000.0
        z = -2000.0 if g[10] == "0" else 2000.0
        la, lo = scene_to_latlon(x, z)
        lat.append(la)
        lon.append(lo)
    far = {"060730999991001": (32.72, -117.16), "060730999992001": (33.20, -117.10), "060730999993001": (32.90, -117.24)}
    for b, (la, lo) in far.items():
        blocks.append(b)
        lat.append(la)
        lon.append(lo)
    xw = pd.DataFrame({"tabblk2020": blocks, "blklatdd": lat, "blklondd": lon})
    with gzip.open(raw / "ca_xwalk.csv.gz", "wt") as f:
        xw.to_csv(f, index=False)
    od = []
    for g in geoids:
        for w in blocks:
            od.append({"w_geocode": w, "h_geocode": g + "001", "S000": int(rng.integers(1, 50))})
    with gzip.open(raw / "ca_od_main_JT00_2021.csv.gz", "wt") as f:
        pd.DataFrame(od).to_csv(f, index=False)
    for n in ("ca_wac_S000_JT00_2021.csv.gz", "ca_rac_S000_JT00_2021.csv.gz"):
        with gzip.open(raw / n, "wt") as f:
            f.write("placeholder\n")


def syn_to_utm(geom):
    from shapely.affinity import affine_transform

    o = scene_origin()
    # x_utm = x + E0 ; y_utm = N0 - z
    return affine_transform(geom, [1, 0, 0, -1, o.easting, o.northing])
