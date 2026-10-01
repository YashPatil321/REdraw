"""Unit tests for pure pipeline rules (no network, no big builds)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from pipeline.build_buildings import (
    building_height,
    building_mesh,
    building_type,
    parse_height_m,
    rectangularity,
    use_hip_roof,
)
from pipeline.build_population import (
    allocate_households,
    bearing_deg,
    exit_for_bearing,
    grades_from_name,
    household_capacity,
)
from pipeline.build_roads import (
    capacity_vph,
    densify,
    drape,
    free_flow_seconds,
    label_for,
    lanes_per_direction,
    normalize_class,
    normalize_ref,
    parse_maxspeed_kph,
)
from pipeline.build_terrain import Terrain, terrain_tile_mesh
from pipeline.common import Extent, TileGrid, bbox_scene_extent
from pipeline.config import assumption, region
from pipeline.fetch_census import acs_rows_to_frame, acs_url, lodes_urls
from pipeline.fetch_imagery import latest_items

# ---------------------------------------------------------------- buildings


def test_height_priority_height_tag_wins() -> None:
    h, lv, rule = building_height("house", "12.5", "5")
    assert rule == "height" and h == pytest.approx(12.5) and lv == 5


def test_height_priority_levels_second() -> None:
    h, lv, rule = building_height("commercial", None, "3")
    assert rule == "levels"
    assert h == pytest.approx(3 * assumption("buildings.level_height_m"))
    assert lv == 3


@pytest.mark.parametrize("btype", ["house", "commercial", "school", "apartments", "other"])
def test_height_priority_default_by_type(btype: str) -> None:
    h, lv, rule = building_height(btype, None, None)
    assert rule == "default" and lv is None
    assert h == pytest.approx(assumption(f"buildings.default_height_m.{btype}"))


def test_height_tag_parsing() -> None:
    assert parse_height_m("10 m") == pytest.approx(10.0)
    assert parse_height_m("30'") == pytest.approx(30 * 0.3048)
    assert parse_height_m("30 ft") == pytest.approx(30 * 0.3048)
    assert parse_height_m("30'6\"") == pytest.approx(30.5 * 0.3048)
    assert parse_height_m("") is None and parse_height_m(None) is None and parse_height_m(float("nan")) is None
    # unparseable height falls through to levels
    assert building_height("house", "tall", "2")[2] == "levels"


def test_building_type_rules() -> None:
    assert building_type("house", 150) == "house"
    assert building_type("yes", 150) == "house"
    assert building_type("yes", 5000) == "commercial"
    assert building_type("residential", 2000) == "apartments"
    assert building_type("apartments", 300) == "apartments"
    assert building_type("yes", 900, in_school=True) == "school"
    assert building_type("yes", 900, amenity="school") == "school"
    assert building_type("garage", 40) == "other"
    assert building_type("yes", 300, shop="supermarket") == "commercial"


def test_hip_roof_decision() -> None:
    rect = box(0, 0, 12, 16)
    ell = Polygon([(0, 0), (20, 0), (20, 8), (8, 8), (8, 20), (0, 20)])
    assert rectangularity(rect) == pytest.approx(1.0)
    assert rectangularity(ell) < assumption("buildings.hip_roof_rectangularity")
    assert use_hip_roof("house", rect)
    assert not use_hip_roof("house", ell)
    assert not use_hip_roof("commercial", rect)  # only houses get hip roofs
    rot = Polygon([(0, 0), (10, 5), (5, 15), (-5, 10)])  # rotated rectangle still counts
    assert use_hip_roof("house", rot)


def test_building_mesh_heights_and_winding() -> None:
    poly = box(0, 0, 12, 16)
    roof_h = assumption("buildings.hip_roof_height_m")
    pos, nrm, tri, is_roof = building_mesh(poly, 100.0, 7.0, True, roof_h)
    assert pos[:, 1].max() == pytest.approx(107.0)
    assert is_roof.any() and (~is_roof).any()
    # every triangle's geometric normal agrees with the stored vertex normal
    v0, v1, v2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    gn = np.cross(v1 - v0, v2 - v0)
    assert (np.einsum("ij,ij->i", gn, nrm[tri[:, 0]]) > 0).all()
    pos2, _, _, roof2 = building_mesh(Polygon([(0, 0), (20, 0), (20, 8), (8, 8), (8, 20), (0, 20)]), 50.0, 8.0, False, roof_h)
    assert np.allclose(pos2[roof2, 1], 58.0)  # flat roof


# ---------------------------------------------------------------- roads


def test_capacity_and_free_flow() -> None:
    cap = capacity_vph("primary", 3)
    assert cap == pytest.approx(3 * assumption("roads.defaults_by_class.primary.capacity_vphpl"))
    assert capacity_vph("primary_link", 1) == pytest.approx(assumption("roads.defaults_by_class.primary_link.capacity_vphpl"))
    assert free_flow_seconds(1000.0, 72.0) == pytest.approx(50.0)
    assert normalize_class(["residential", "unclassified"]) == "residential"
    assert normalize_class("busway") == "unclassified"


def test_lanes_per_direction() -> None:
    assert lanes_per_direction("primary", "6", oneway=False) == 3
    assert lanes_per_direction("primary", "5", oneway=False) == 3
    assert lanes_per_direction("motorway", "4", oneway=True) == 4
    assert lanes_per_direction("residential", None, oneway=False) == assumption("roads.defaults_by_class.residential.lanes")
    assert lanes_per_direction("secondary", "4", False, lanes_forward="3", lanes_backward="1") == 3
    assert lanes_per_direction("secondary", "4", False, lanes_forward="3", lanes_backward="1", reversed_=True) == 1
    assert lanes_per_direction("secondary", ["2", "4"], False) == 2


def test_maxspeed_parsing() -> None:
    assert parse_maxspeed_kph("45 mph") == pytest.approx(72.42, abs=0.01)
    assert parse_maxspeed_kph("50") == pytest.approx(50.0)
    assert parse_maxspeed_kph(["25 mph", "35 mph"]) == pytest.approx(35 * 1.609344)
    assert parse_maxspeed_kph(None) is None and parse_maxspeed_kph("signals") is None


def test_labels_and_refs() -> None:
    labels = region()["arterial_labels"]
    assert normalize_ref("SR-56") == ["CA 56"] and normalize_ref("I-15;CA 56") == ["I 15", "CA 56"]
    # every configured label matches its own OSM name exactly (and case-insensitively)
    for lab in labels:
        assert label_for(lab["name"], "", labels) == lab["name"]
        assert label_for(lab["name"].upper(), "", labels) == lab["name"]
    # a ref alone maps to the first label carrying that ref; a name always wins over a ref
    for lab in labels:
        if lab.get("ref"):
            first_with_ref = next(x["name"] for x in labels if set(normalize_ref(str(x.get("ref", "")))) & set(normalize_ref(lab["ref"])))
            assert label_for("", lab["ref"], labels) == first_with_ref
            assert label_for(lab["name"], lab["ref"], labels) == lab["name"]
    by_ref = {n for n in (label_for("", "CA 56", labels), label_for("", "SR-56", labels)) if n}
    assert by_ref and by_ref <= {x["name"] for x in labels if "CA 56" in normalize_ref(str(x.get("ref", "")))}
    assert label_for("Calle Albero", "", labels) == ""


def test_edge_geometry_draping() -> None:
    ext = Extent(0.0, 100.0, 0.0, 100.0)
    xs = np.arange(0, 101, 10.0)
    elev = np.tile(xs * 0.5, (len(xs), 1)).astype(np.float32)  # y = 0.5 * x
    t = Terrain(elev, ext, 10.0)
    xyz = drape(np.array([[0.0, 50.0], [100.0, 50.0]]), t, lift=0.0, max_seg=20.0)
    assert len(xyz) == 6  # 100 m in <= 20 m steps
    assert np.allclose(xyz[:, 1], xyz[:, 0] * 0.5, atol=1e-4)
    assert np.allclose(xyz[[0, -1]][:, [0, 2]], [[0, 50], [100, 50]])
    d = densify(np.array([[0, 0], [0, 45.0]]), 20.0)
    assert len(d) == 4 and np.allclose(np.diff(d[:, 1]), 15.0)


# ---------------------------------------------------------------- terrain / grid


def test_terrain_sample_and_tile_mesh() -> None:
    ext = Extent(-100.0, 100.0, -50.0, 50.0)
    elev = np.full((11, 21), 200.0, dtype=np.float32)
    elev[5, 10] = 210.0
    t = Terrain(elev, ext, 10.0)
    assert t.sample(0.0, 0.0) == pytest.approx(210.0)
    assert t.sample(5.0, 0.0) == pytest.approx(205.0)
    pos, nrm, uv, idx = terrain_tile_mesh(t, ext, 25.0)
    assert idx.max() < len(pos) and len(idx) % 3 == 0
    tri = idx.reshape(-1, 3)
    v0, v1, v2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    assert (np.cross(v1 - v0, v2 - v0)[:, 1] > 0).all()  # all terrain triangles face up (+y)
    assert uv.min() >= 0 and uv.max() <= 1


def test_tile_grid_and_extent() -> None:
    from pipeline.geo import latlon_to_scene

    b = region()["bbox"]
    ext = bbox_scene_extent(b)
    for lat in (b["south"], b["north"]):
        for lon in (b["west"], b["east"]):
            x, z = latlon_to_scene(lat, lon)
            assert ext.min_x <= x <= ext.max_x and ext.min_z <= z <= ext.max_z
    assert ext.min_x < 0 < ext.max_x and ext.min_z < 0 < ext.max_z  # origin = bbox center
    g = TileGrid(Extent(0, 400, 0, 400), 4, 4)
    r, c = g.tile_of(np.array([10.0, 399.0, 500.0]), np.array([10.0, 150.0, -5.0]))
    assert list(r) == [0, 1, 0] and list(c) == [0, 3, 3]
    assert g.bounds(1, 2).min_x == 200 and g.bounds(1, 2).min_z == 100


# ---------------------------------------------------------------- population / census / imagery helpers


def test_bearings_and_exits() -> None:
    assert bearing_deg(0, 0, 0, -100) == pytest.approx(0.0)
    assert bearing_deg(0, 0, 100, 0) == pytest.approx(90.0)
    assert bearing_deg(0, 0, 0, 100) == pytest.approx(180.0)
    exits = region()["exits"]
    for e in exits:  # a job straight along an exit's bearing uses that exit
        assert exit_for_bearing(float(e["bearing_deg"]), exits) == e["id"]
        assert exit_for_bearing((float(e["bearing_deg"]) + 3.0) % 360.0, exits) == e["id"]


def test_household_allocation() -> None:
    rng = np.random.default_rng(0)
    caps = np.array([1, 1, 1, 10])
    w = np.array([100.0, 100.0, 100.0, 2000.0])
    a = allocate_households(rng, caps, w, 8)
    assert len(a) == 8 and np.bincount(a, minlength=4).max() <= 10
    b = allocate_households(rng, caps, w, 20)  # over capacity -> extras by area weight
    assert len(b) == 20 and (np.bincount(b, minlength=4) >= caps).all()
    assert household_capacity("house", 200.0, None) == 1
    assert household_capacity("commercial", 2000.0, None) == 0
    assert household_capacity("apartments", 1456.0, 3) == round(1456 * 3 / assumption("pipeline.apartment_m2_per_household"))


def test_grades_from_name() -> None:
    assert grades_from_name("Foo High School") == [9, 12]
    assert grades_from_name("Bar Middle School") == [6, 8]
    assert grades_from_name("Baz Elementary") == [0, 5]
    assert grades_from_name("Little Sprouts Preschool") is None


def test_acs_parsing_and_urls() -> None:
    rows = [["NAME", "B11016_001E", "state", "county", "tract", "block group"], ["x", "120", "06", "073", "017101", "2"], ["y", "-666666666", "06", "073", "017101", "3"]]
    df = acs_rows_to_frame(rows)
    assert list(df["GEOID"]) == ["060730171012", "060730171013"]
    assert df["B11016_001E"].iloc[0] == 120 and math.isnan(df["B11016_001E"].iloc[1])
    u = acs_url("2022", ["B11016_001E"], "KEY")
    assert "for=block%20group:*" in u and "in=state:06" in u and u.endswith("key=KEY")
    lu = lodes_urls("LODES8", "2021")
    assert lu["od_main"].endswith("LODES8/ca/od/ca_od_main_JT00_2021.csv.gz")
    assert lu["xwalk"].endswith("LODES8/ca/ca_xwalk.csv.gz")


def test_latest_naip_items() -> None:
    class It:
        def __init__(self, y: int):
            self.datetime = f"{y}-06-01T00:00:00Z"
            self.properties: dict[str, str] = {}

    items = [It(2020), It(2022), It(2022), It(2018)]
    assert len(latest_items(items)) == 2 and latest_items([]) == []


def test_unavailable_source_error_names_url_and_path(tmp_path) -> None:
    from pipeline.common import DataSourceUnavailable, download

    with pytest.raises(DataSourceUnavailable) as ei:
        download("http://127.0.0.1:9/nope.csv.gz", tmp_path / "nope.csv.gz", "Test source", "copy it by hand", timeout=2)
    msg = str(ei.value)
    assert "http://127.0.0.1:9/nope.csv.gz" in msg and "nope.csv.gz" in msg and "copy it by hand" in msg
    assert not (tmp_path / "nope.csv.gz").exists()
