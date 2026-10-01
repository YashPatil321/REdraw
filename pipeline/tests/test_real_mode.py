"""Real-mode additions: footprint population fallback, inferred signals, exit splitting,
island connectors, roof shapes / colours, campus matching, and source attribution."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from pipeline.build_buildings import (
    building_mesh,
    building_type,
    estimate_triangles,
    parse_colour,
    roof_kind,
)
from pipeline.build_population import (
    footprint_units,
    household_capacity,
    osm_school_grades,
    osm_school_id,
)
from pipeline.build_roads import connect_islands, infer_signals, road_key, split_edges_at_exits
from pipeline.config import assumption, assumption_range, region
from pipeline.fetch_osm import match_campuses, school_name_key
from pipeline.sources import attribution_lines, population_sources, raw_sources

# ---------------------------------------------------------------- signals


def _g(edges: list[tuple[int, int, str, str]]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph(crs="EPSG:32611")
    for n in {e[0] for e in edges} | {e[1] for e in edges}:
        G.add_node(n, x=float(n) * 100.0, y=0.0)
    for u, v, hw, name in edges:
        G.add_edge(u, v, highway=hw, name=name, osmid=u * 1000 + v)
        G.add_edge(v, u, highway=hw, name=name, osmid=u * 1000 + v)
    return G


def test_infer_signals_distinct_roads_and_ramps() -> None:
    G = _g(
        [
            (1, 2, "primary", "Camino del Sur"),
            (2, 3, "primary", "Camino del Sur"),  # node 2: one road only -> no signal
            (3, 4, "primary", "Camino del Sur"),
            (3, 5, "tertiary", "Nighthawk Lane"),  # node 3: two distinct roads >= tertiary -> signal
            (4, 6, "residential", "Calle Uno"),  # node 4: residential cross street -> no signal
            (6, 7, "secondary", "Paseo del Sur"),
            (7, 8, "motorway_link", ""),  # node 7: ramp meets an arterial -> signal
        ]
    )
    sig = infer_signals(G)
    assert 3 in sig and 7 in sig
    assert 2 not in sig and 4 not in sig and 1 not in sig
    assert road_key({"name": "Camino del Sur", "ref": ""}) == road_key({"name": "camino del sur", "ref": "S6"})
    assert set(assumption("signal_inference.road_classes")) >= {"primary", "secondary", "tertiary"}


# ---------------------------------------------------------------- exits / islands


def test_split_edges_at_exit_and_connect_islands(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline.geo import scene_origin

    o = scene_origin()
    G = nx.MultiDiGraph(crs="EPSG:32611")
    # divided freeway: eastbound 1->2 (y=0), westbound 4->3 (y=30), nodes 2 km apart
    for n, (x, y) in {1: (0, 0), 2: (2000, 0), 3: (0, 30), 4: (2000, 30)}.items():
        G.add_node(n, x=o.easting + x, y=o.northing + y)
    for u, v in ((1, 2), (4, 3)):
        G.add_edge(u, v, highway="motorway", name="Ted Williams Freeway", ref="CA 56", oneway=True, osmid=u, length=2000.0)
    G.add_edge(2, 4, highway="motorway", name="", ref="", oneway=True, osmid=99, length=30.0)
    G.add_edge(3, 1, highway="motorway", name="", ref="", oneway=True, osmid=98, length=30.0)
    ex = {"id": "sr56_west", "label": "SR 56 west", "lat": 0.0, "lon": 0.0}
    labels = region()["arterial_labels"]
    added = split_edges_at_exits(G, [(ex, 1000.0, -10.0)], labels)  # scene z = -10 -> 10 m north of eastbound
    assert added == 2
    split = [n for n, d in G.nodes(data=True) if d.get("exit_split") == "sr56_west"]
    assert len(split) == 2
    assert all(abs(G.nodes[n]["x"] - (o.easting + 1000.0)) < 1.0 for n in split)
    a, b = split
    assert G.has_edge(a, b) and G.has_edge(b, a)  # turnaround between carriageways
    assert nx.is_strongly_connected(G)
    # an island 200 m away gets a two-way connector; one 2 km away does not
    G.add_node(10, x=o.easting + 1000, y=o.northing + 230)
    G.add_node(11, x=o.easting + 1050, y=o.northing + 260)
    G.add_edge(10, 11, highway="residential", osmid=10)
    G.add_edge(11, 10, highway="residential", osmid=10)
    G.add_node(20, x=o.easting + 1000, y=o.northing + 3000)
    G.add_node(21, x=o.easting + 1050, y=o.northing + 3000)
    G.add_edge(20, 21, highway="residential", osmid=20)
    G.add_edge(21, 20, highway="residential", osmid=20)
    assert connect_islands(G) == 1
    main = max(nx.strongly_connected_components(G), key=len)
    assert {10, 11} <= main and 20 not in main


# ---------------------------------------------------------------- buildings


def test_landuse_and_estate_rules() -> None:
    assert building_type("yes", 200, landuse="retail") == "commercial"
    assert building_type("yes", 200, landuse="golf_course") == "other"
    assert building_type("yes", 200, landuse="residential") == "house"
    assert building_type("yes", 50) == "other"  # below footprint_population.house_min_area_m2
    assert building_type("yes", 800) == "commercial"
    assert building_type("yes", 800, estate_context=True) == "house"
    assert building_type("yes", 5000, estate_context=True) == "commercial"
    assert building_type("house", 50) == "house"  # explicit tags win


def test_roof_shapes_and_colours() -> None:
    rect = box(0, 0, 10, 20)
    assert roof_kind("commercial", rect, "gabled") == "gabled"
    assert roof_kind("house", rect, "half_hipped") == "hipped"
    assert roof_kind("house", rect, "flat") == "flat"
    assert roof_kind("house", rect, None) == "hipped"  # rectangular house default
    assert roof_kind("commercial", rect, None) == "flat"
    roof_h = assumption("buildings.hip_roof_height_m")
    pos, nrm, tri, is_roof = building_mesh(rect, 10.0, 8.0, False, roof_h, roof="gabled")
    assert pos[:, 1].max() == pytest.approx(18.0)
    v0, v1, v2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    gn = np.cross(v1 - v0, v2 - v0)
    assert (np.einsum("ij,ij->i", gn, nrm[tri[:, 0]]) > 0).all()  # winding agrees with normals
    # gable ends are vertical, wall coloured and face outward
    gable = (~is_roof) & (pos[:, 1] > 10.0 + 8.0 - roof_h + 1e-3)
    assert gable.any() and np.allclose(nrm[gable, 1], 0.0, atol=1e-6)
    assert len(tri) == estimate_triangles(rect, "gabled")
    assert parse_colour("#778899") == (0x77, 0x88, 0x99)
    assert parse_colour("white") == (255, 255, 255)
    assert parse_colour("not a colour") is None and parse_colour(None) is None


# ---------------------------------------------------------------- population


def test_footprint_units() -> None:
    df = pd.DataFrame(
        {
            "type": ["house", "house", "apartments", "commercial", "apartments"],
            "area_m2": [200.0, 440.0, 1200.0, 3000.0, 900.0],
            "levels_est": [2, 2, 3, None, None],
            "building_tag": ["yes", "terrace", "apartments", "retail", "apartments"],
        }
    )
    u = footprint_units(df)
    town = assumption("footprint_population.townhouse_unit_footprint_m2")
    assert u[0] == assumption("footprint_population.households_per_house")
    assert u[1] == max(1, round(440.0 / town))
    assert u[2] == household_capacity("apartments", 1200.0, 3)
    assert u[3] == 0
    assert u[4] == household_capacity("apartments", 900.0, None)
    lo, hi = assumption_range("population.target_households_fallback")
    assert lo < hi


def test_osm_school_filters() -> None:
    assert osm_school_grades("Westview High School") == [9, 12]
    assert osm_school_grades("Abraxas Continuation High School") is None
    assert osm_school_grades("Discovery Isle Preschool") is None
    assert osm_school_id("Los Peñasquitos Elementary School") == "osm_los_penasquitos_elementary_school"


def test_campus_matching_by_name_and_containment() -> None:
    from pipeline.geo import lonlat_to_utm

    cfg = [
        {"id": "design39", "name": "Design39 Campus", "lat": 33.0187, "lon": -117.1219},
        {"id": "del_norte_hs", "name": "Del Norte High School", "lat": 33.0140, "lon": -117.1225},
    ]
    e1, n1 = lonlat_to_utm(-117.1219, 33.0187)
    e2, n2 = lonlat_to_utm(-117.1225, 33.0140)
    gdf = gpd.GeoDataFrame(
        {"name": ["Design39Campus", None, "Maranatha Christian Schools", "Mantinence"]},
        geometry=[Point(e1, n1).buffer(150), Point(e2, n2).buffer(250), Point(e2 + 300, n2).buffer(120), Point(e2 + 30, n2 + 20).buffer(5)],
        crs="EPSG:32611",
    )
    ids = match_campuses(gdf, cfg)
    assert ids == ["design39", "del_norte_hs", None, "del_norte_hs"]
    assert school_name_key("Design39 Campus") == school_name_key("Design39Campus") == "design39"


# ---------------------------------------------------------------- sources


def test_sources_from_aws_cache_and_sidecars(tmp_path: Path) -> None:
    dem, img = tmp_path / "dem_3dep_10m.tif", tmp_path / "naip_mosaic.tif"
    dem.write_bytes(b"x")
    img.write_bytes(b"x")
    tiles = [{"name": "USGS 3DEP 1 m lidar DEM, project P", "url": f"https://h/P/TIFF/t{i}.tif", "license": "Public domain", "retrieved": "2026-10-01"} for i in range(3)]
    (tmp_path / "dem_3dep_10m.source.json").write_text(json.dumps({"resolution_m": 2.0, "sources": tiles}))
    (tmp_path / "naip_mosaic.source.json").write_text(json.dumps({"name": "Copernicus Sentinel-2 L2A true colour", "license": "Copernicus", "url": "https://s2", "acquired": "2026-09-26"}))
    (tmp_path / "aws_sources.json").write_text(
        json.dumps(
            {
                "not_available": ["LEHD LODES"],
                "sources": [
                    {"name": "Overture Maps buildings/building", "license": "ODbL 1.0", "kind": "overture", "record_sources": {"Microsoft ML Buildings | ODbL-1.0": 5}},
                    {"name": "Overture Maps places/place", "license": "CDLA-Permissive-2.0", "kind": "overture"},
                    {"name": "dem tile", "kind": "dem", "url": "u"},
                ],
            }
        )
    )
    src = raw_sources(tmp_path, dem, img)
    kinds = [s["kind"] for s in src]
    assert kinds.count("dem") == 1 and kinds.count("imagery") == 1 and kinds.count("overture") == 2
    d = next(s for s in src if s["kind"] == "dem")
    assert len(d["files"]) == 3 and d["url"] == "https://h/P/TIFF/" and d["raster"]["resolution_m"] == 2.0
    lines = attribution_lines(src)
    assert "Contains modified Copernicus Sentinel data 2026" in lines
    assert "(c) OpenStreetMap contributors, Overture Maps Foundation" in lines and "Overture Maps Foundation" in lines
    assert not any("OpenStreetMap (roads" in s["name"] for s in src)  # no hardcoded OSM/Overpass entry
    assert population_sources("footprints")[0]["kind"] == "population"
    # no provenance files -> primary-source defaults
    bare = tmp_path / "bare"
    bare.mkdir()
    names = " ".join(s["name"] for s in raw_sources(bare, bare / "dem.tif", bare / "img.tif"))
    assert "OpenStreetMap" in names and "3DEP" in names and "NAIP" in names


# ---------------------------------------------------------------- end to end (offline mock raw)


def test_real_build_with_footprint_population(world, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`build_all --population-source footprints` on mock raw files: no census needed, labeled."""
    from pipeline.tests.conftest import _env, write_mock_raw

    d = _env(monkeypatch, tmp_path)
    write_mock_raw(world, d["raw"])
    for n in list(d["raw"].iterdir()):
        if n.name.startswith(("acs5_", "tl_", "ca_")):
            n.unlink()  # census files absent: the footprint fallback must not need them
    from pipeline.build_all import run_real

    meta = run_real(skip_draco=True, population_source="footprints")
    assert meta["synthetic"] is False and meta["population_source"] == "footprint_estimate"
    assert any(s.get("kind") == "population" and "Footprint" in s["name"] for s in meta["sources"])
    assert "ESTIMATED" in meta["population_note"]
    hh = pd.read_parquet(d["processed"] / "households.parquet")
    b = gpd.read_file(d["processed"] / "buildings.geojson")
    assert meta["counts"]["households"] == len(hh) > 0
    assert set(hh["building_id"]) <= set(b.loc[b["type"].isin(["house", "apartments"]), "id"])
    schools = json.loads((d["processed"] / "schools_resolved.json").read_text())["schools"]
    edges = pd.read_parquet(d["processed"] / "network_edges.parquet")
    for s in schools:
        for e in s["entrances"]:
            assert int(edges.loc[e["approach_edge_idx"], "v"]) == e["node_id"]
    manifest = json.loads((d["assets"] / "manifest.json").read_text())
    assert manifest["synthetic"] is False and manifest["draco"] is False
    with pytest.raises(ValueError):
        run_real(skip_draco=True, population_source="bogus")


def test_acs_mode_fails_loudly_without_census(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline import fetch_census
    from pipeline.common import DataSourceUnavailable

    def boom(raw: Path | None = None) -> dict[str, Path]:
        raise DataSourceUnavailable("US Census ACS", "https://api.census.gov/x", tmp_path / "acs.csv", "manual")

    monkeypatch.setattr(fetch_census, "fetch_all", boom)
    from pipeline import fetch_dem, fetch_imagery, fetch_osm

    monkeypatch.setattr(fetch_osm, "fetch_all", lambda raw=None: {})
    monkeypatch.setattr(fetch_dem, "fetch", lambda raw=None: tmp_path / "dem.tif")
    monkeypatch.setattr(fetch_imagery, "fetch", lambda raw=None: tmp_path / "img.tif")
    monkeypatch.setenv("REDRAW_DATA_DIR", str(tmp_path / "p"))
    monkeypatch.setenv("REDRAW_ASSETS_DIR", str(tmp_path / "a"))
    monkeypatch.setenv("REDRAW_RAW_DIR", str(tmp_path / "r"))
    from pipeline.build_all import run_real

    with pytest.raises(DataSourceUnavailable) as ei:
        run_real(skip_draco=True, population_source="acs")
    assert "--population-source footprints" in str(ei.value) and "api.census.gov" in str(ei.value)
    assert not (tmp_path / "p" / "region_meta.json").exists()


def test_line_helper_shapes() -> None:  # guards the edge geometry -> LineString conversion
    from pipeline.build_population import edge_lines

    e = pd.DataFrame({"geometry": [np.array([0, 0, 0, 10, 0, 0], dtype=np.float32), np.array([5, 0, 5], dtype=np.float32)]})
    ls = edge_lines(e)
    assert isinstance(ls[0], LineString) and ls[0].length == pytest.approx(10.0) and ls[1].length > 0
