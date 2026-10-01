"""Offline tests for pipeline/fetch_aws.py (pure helpers; no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import Point, box

from pipeline import fetch_aws as fa
from pipeline.geo import lonlat_to_utm

# A spot inside the region bbox (4S Ranch).
LON0, LAT0 = -117.12, 33.01


def test_highway_tag_links_and_footways() -> None:
    assert fa.highway_tag("motorway", "link") == "motorway_link"
    assert fa.highway_tag("primary", None, ["is_link"]) == "primary_link"
    assert fa.highway_tag("residential", "link") == "residential"  # no residential_link in OSM
    assert fa.highway_tag("footway", "sidewalk") == "footway"
    assert fa.highway_tag(None, None) == "unclassified"
    assert fa.highway_tag("unknown", None) == "road"


def test_directions_allowed_oneway_private_and_modes() -> None:
    drive = fa.MODE_SETS["drive"]
    walk = fa.MODE_SETS["walk"]
    assert fa.directions_allowed(None, drive) == (True, True)
    oneway = [{"access_type": "denied", "when": {"heading": "backward"}}]
    assert fa.directions_allowed(oneway, drive) == (True, False)
    # partial-length restrictions are ignored
    part = [{"access_type": "denied", "when": {"heading": "backward"}, "between": [0.2, 0.6]}]
    assert fa.directions_allowed(part, drive) == (True, True)
    private = [{"access_type": "allowed", "when": {"recognized": ["as_private"]}}]
    assert fa.directions_allowed(private, drive) == (False, False)
    footway = [{"access_type": "denied"}, {"access_type": "allowed", "when": {"mode": ["foot"]}}]
    assert fa.directions_allowed(footway, drive) == (False, False)
    assert fa.directions_allowed(footway, walk) == (True, True)
    no_bikes = [{"access_type": "denied", "when": {"mode": ["bicycle"]}}]
    assert fa.directions_allowed(no_bikes, drive) == (True, True)
    assert fa.directions_allowed(no_bikes, fa.MODE_SETS["bike"]) == (False, False)
    # conditional rules (time windows, customers) are ignored like OSMnx does
    cond = [{"access_type": "denied", "when": {"during": "Mo-Fr 07:00-09:00"}}]
    assert fa.directions_allowed(cond, drive) == (True, True)


def test_maxspeed_and_ref_tags() -> None:
    assert fa.maxspeed_tag([{"max_speed": {"value": 45, "unit": "mph"}}]) == "45 mph"
    assert (
        fa.maxspeed_tag(
            [
                {"max_speed": {"value": 25, "unit": "mph"}},
                {"max_speed": {"value": 40, "unit": "mph"}},
            ]
        )
        == "40 mph"
    )
    assert fa.maxspeed_tag([{"max_speed": {"value": 50, "unit": "km/h"}}]) == "50"
    assert (
        fa.maxspeed_tag([{"max_speed": {"value": 65, "unit": "mph"}, "between": [0.0, 0.3]}])
        is None
    )
    assert fa.maxspeed_tag(None) is None
    routes = [
        {"network": "US:I", "ref": "15"},
        {"network": "US:I", "ref": "15"},
        {"network": "US:CA", "ref": "56"},
        {"network": "US:CA:CR", "ref": "S6"},
        {"network": "US:I:Express", "ref": "15"},
    ]
    assert fa.ref_tag(routes) == "I 15;CA 56;S6"
    assert fa.ref_tag([]) is None


def test_osm_way_id() -> None:
    assert fa.osm_way_id([{"dataset": "OpenStreetMap", "record_id": "w528507466@5"}]) == 528507466
    assert fa.osm_way_id([{"dataset": "TomTom", "record_id": "w1@1"}]) is None
    assert fa.osm_way_id(None) is None


def _line(n: int = 5, step: float = 0.001) -> list[tuple[float, float]]:
    return [(LON0 + i * step, LAT0) for i in range(n)]


def test_split_segment_at_connectors() -> None:
    coords = _line(5)
    cons = [
        {"connector_id": "a", "at": 0.0},
        {"connector_id": "b", "at": 0.5},
        {"connector_id": "c", "at": 1.0},
    ]
    parts = fa.split_segment(coords, cons, lonlat_to_utm)
    assert [(p[0], p[1]) for p in parts] == [("a", "b"), ("b", "c")]
    total = sum(p[3] for p in parts)
    e0, _ = lonlat_to_utm(*coords[0])
    e1, _ = lonlat_to_utm(*coords[-1])
    assert total == pytest.approx(abs(e1 - e0), rel=0.01)
    assert parts[0][2][0] == pytest.approx(coords[0])
    assert parts[1][2][-1] == pytest.approx(coords[-1])
    assert parts[0][2][-1] == pytest.approx(parts[1][2][0], abs=1e-7)
    assert fa.split_segment(coords, cons[:1], lonlat_to_utm) == []


def _segments() -> tuple[list[dict], dict[str, tuple[float, float]]]:
    main = _line(3)  # A - B (two-way, primary)
    side = [(LON0 + 0.002, LAT0), (LON0 + 0.002, LAT0 + 0.002)]  # B - C (oneway backward)
    foot = [(LON0, LAT0), (LON0, LAT0 + 0.002)]  # A - D footway
    segs = [
        {
            "id": "s-main",
            "subtype": "road",
            "class": "primary",
            "subclass": None,
            "names": {"primary": "Camino del Sur"},
            "routes": [{"network": "US:CA:CR", "ref": "S5"}],
            "speed_limits": [{"max_speed": {"value": 45, "unit": "mph"}}],
            "connectors": [{"connector_id": "A", "at": 0.0}, {"connector_id": "B", "at": 1.0}],
            "sources": [{"dataset": "OpenStreetMap", "record_id": "w42@3"}],
            "coords": main,
        },
        {
            "id": "s-side",
            "subtype": "road",
            "class": "residential",
            "subclass": None,
            "names": None,
            "access_restrictions": [{"access_type": "denied", "when": {"heading": "forward"}}],
            "connectors": [{"connector_id": "B", "at": 0.0}, {"connector_id": "C", "at": 1.0}],
            "coords": side,
        },
        {
            "id": "s-foot",
            "subtype": "road",
            "class": "footway",
            "subclass": "sidewalk",
            "names": None,
            "connectors": [{"connector_id": "A", "at": 0.0}, {"connector_id": "D", "at": 1.0}],
            "coords": foot,
        },
        {
            "id": "s-rail",
            "subtype": "rail",
            "class": "standard_gauge",
            "connectors": [{"connector_id": "A", "at": 0}, {"connector_id": "D", "at": 1}],
            "coords": foot,
        },
    ]
    cxy = {"A": main[0], "B": main[-1], "C": side[-1], "D": foot[-1]}
    return segs, cxy


def test_build_graphs_drive_walk_bike() -> None:
    segs, cxy = _segments()
    g = fa.build_graphs(segs, cxy)
    drive, walk, bike = g["drive"], g["walk"], g["bike"]
    nid = {d["overture_id"]: n for n, d in drive.nodes(data=True)}
    assert set(nid) == {"A", "B", "C"}
    assert all(isinstance(n, int) for n in drive.nodes)
    # primary two-way -> 2 edges; side street denied forward -> only C -> B
    assert drive.number_of_edges() == 3
    assert drive.has_edge(nid["A"], nid["B"]) and drive.has_edge(nid["B"], nid["A"])
    assert drive.has_edge(nid["C"], nid["B"]) and not drive.has_edge(nid["B"], nid["C"])
    e = drive.edges[nid["A"], nid["B"], 0]
    assert e["highway"] == "primary" and e["name"] == "Camino del Sur" and e["ref"] == "S5"
    assert e["maxspeed"] == "45 mph" and e["oneway"] is False and e["osmid"] == 42
    side = drive.edges[nid["C"], nid["B"], 0]
    assert side["oneway"] is True and side["osmid"] >= 10**12
    assert side["geometry"].coords[0] == pytest.approx(cxy["C"])
    # walk: footway included, oneway ignored for pedestrians
    assert walk.number_of_edges() == 6
    # bike: footway excluded, oneway respected
    assert bike.number_of_edges() == 3


def test_drive_graph_roundtrips_through_build_roads(tmp_path: Path) -> None:
    import osmnx as ox

    from pipeline.build_roads import load_osm_drive

    segs, cxy = _segments()
    G = fa.build_graphs(segs, cxy, kinds=("drive",))["drive"]
    p = tmp_path / "osm_drive_raw.graphml"
    ox.io.save_graphml(G, p)
    Gp, sig = load_osm_drive(p)
    assert str(Gp.graph["crs"]).upper().endswith("32611")
    assert Gp.number_of_edges() >= 2
    assert sig.shape == (0, 2)
    lengths = [d["length"] for *_, d in Gp.edges(data=True)]
    assert all(v > 50 for v in lengths)


def test_building_tags_mapping() -> None:
    row = {
        "id": "b1",
        "class": None,
        "subtype": "education",
        "height": 9.456,
        "num_floors": 2,
        "names": {"primary": "Gym"},
        "roof_shape": "flat",
        "roof_material": "metal",
        "facade_color": "#ffffff",
        "roof_color": None,
        "sources": [{"dataset": "OpenStreetMap", "record_id": "w7@1"}],
    }
    t = fa.building_tags(row)
    assert t["building"] == "school" and t["height"] == "9.46" and t["building:levels"] == "2"
    assert t["name"] == "Gym" and t["roof:shape"] == "flat" and t["building:colour"] == "#ffffff"
    assert t["element"] == "overture" and t["id"] == "b1" and t["osm_way_id"] == "7"
    t2 = fa.building_tags({"id": "b2", "class": "house", "subtype": "residential", "sources": []})
    assert t2["building"] == "house" and t2["height"] is None and t2["source"] is None
    assert fa.building_tags({"id": "b3"})["building"] == "yes"


def test_school_place_filter_and_merge() -> None:
    assert fa.is_school_place({"taxonomy": {"primary": "elementary_school"}})
    assert not fa.is_school_place(
        {"taxonomy": {"primary": "music_school", "hierarchy": ["education", "school"]}}
    )
    assert fa.is_school_place({"basic_category": "high_school"})
    campus = box(LON0, LAT0, LON0 + 0.002, LAT0 + 0.002)
    named = box(LON0 + 0.01, LAT0, LON0 + 0.012, LAT0 + 0.002)
    polys = [
        ({"id": "lu1", "names": None, "sources": []}, campus),
        ({"id": "lu2", "names": {"primary": "Del Norte High School"}, "sources": []}, named),
    ]
    places = [
        (
            {
                "id": "p1",
                "names": {"primary": "Del Sur Elementary School"},
                "confidence": 0.92,
                "taxonomy": {"primary": "elementary_school"},
            },
            Point(LON0 + 0.001, LAT0 + 0.001),
        ),
        (
            {
                "id": "p2",
                "names": {"primary": "Del Norte High"},
                "confidence": 0.99,
                "taxonomy": {"primary": "high_school"},
            },
            Point(LON0 + 0.011, LAT0 + 0.001),
        ),
        (
            {
                "id": "p3",
                "names": {"primary": "Torah High Schools"},
                "confidence": 0.8,
                "taxonomy": {"primary": "high_school"},
            },
            Point(LON0 + 0.03, LAT0),
        ),
        (
            {
                "id": "p4",
                "names": {"primary": "Connect High School Group"},
                "confidence": 0.95,
                "taxonomy": {"primary": "high_school"},
            },
            Point(LON0 + 0.05, LAT0),
        ),
        (
            {
                "id": "p5",
                "names": {"primary": "Shaky School"},
                "confidence": 0.3,
                "taxonomy": {"primary": "school"},
            },
            Point(LON0 + 0.07, LAT0),
        ),
    ]
    rows, geoms = fa.merge_school_features(polys, places)
    names = [r["name"] for r in rows]
    assert names == ["Del Sur Elementary School", "Del Norte High School", "Torah High Schools"]
    assert rows[0]["name_source"] == "overture_place" and rows[0]["overture_id"] == "lu1"
    assert all(r["amenity"] == "school" for r in rows) and len(geoms) == 3


def test_row_groups_in_bbox(tmp_path: Path) -> None:
    xs = np.array([-117.5, -117.4, -117.12, -117.11, -116.0, -115.9])
    bbox_col = pa.StructArray.from_arrays(
        [
            pa.array(xs),
            pa.array(xs + 0.001),
            pa.array(np.full(6, 33.0)),
            pa.array(np.full(6, 33.001)),
        ],
        names=["xmin", "xmax", "ymin", "ymax"],
    )
    t = pa.table({"id": [str(i) for i in range(6)], "bbox": bbox_col})
    p = tmp_path / "t.parquet"
    pq.write_table(t, p, row_group_size=2)
    md = pq.read_metadata(p)
    assert fa.row_groups_in_bbox(md, (-117.2, 32.9, -117.0, 33.1)) == [1]
    assert fa.row_groups_in_bbox(md, (-118, 32, -115, 34)) == [0, 1, 2]
    assert fa.row_groups_in_bbox(md, (-110, 32, -109, 34)) == []


def test_imagery_helpers() -> None:
    refl = np.stack([np.linspace(-0.05, 0.8, 100).reshape(10, 10)] * 3)
    refl[0, 0, 0] = np.nan
    rgb = fa.natural_color(refl)
    assert rgb.dtype == np.uint8 and rgb.shape == (3, 10, 10)
    assert rgb[:, 0, 1].max() == 0 and rgb[:, -1, -1].min() >= 250
    assert np.all(np.diff(rgb[0].ravel()[1:].astype(int)) >= 0)  # monotonic tone curve
    item = {
        "properties": {"earthsearch:boa_offset_applied": True, "s2:processing_baseline": "05.12"},
        "assets": {"red": {"raster:bands": [{"scale": 0.0001, "offset": -0.1}]}},
    }
    assert fa.band_scale(item, "B04") == (0.0001, 0.0)
    item["properties"]["earthsearch:boa_offset_applied"] = False
    assert fa.band_scale(item, "B04") == (0.0001, -0.1)
    assert fa.band_scale(
        {"properties": {"s2:processing_baseline": "03.01"}, "assets": {}}, "B04"
    ) == (0.0001, 0.0)
    assert fa.scene_sort_key("x/S2A_11SMS_20260926_1_L2A/") > fa.scene_sort_key(
        "x/S2C_11SMS_20260901_0_L2A/"
    )


def test_dem_tile_names_cover_extent() -> None:
    fx = fa.FetchExtent(483024, 3646798, 493624, 3656898)
    assert fa.dem_1m_tile_names(fx) == ["x48y365", "x48y366", "x49y365", "x49y366"]
    w, s, e, n = fx.lonlat
    assert w < -117.17 < -117.08 < e and s < 32.97 < 33.04 < n
