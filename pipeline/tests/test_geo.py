import json
from pathlib import Path

import pytest

from pipeline.geo import latlon_to_scene, scene_origin, scene_to_latlon, scene_to_unreal

POINTS = json.loads((Path(__file__).resolve().parents[2] / "docs" / "geo_test_points.json").read_text())


def test_origin_matches_fixture() -> None:
    o = scene_origin()
    assert o.lat == pytest.approx(POINTS["origin"]["lat"])
    assert o.easting == pytest.approx(POINTS["origin"]["easting"], abs=1e-3)
    assert o.northing == pytest.approx(POINTS["origin"]["northing"], abs=1e-3)


@pytest.mark.parametrize("p", POINTS["points"])
def test_latlon_to_scene_matches_fixture(p: dict) -> None:
    x, z = latlon_to_scene(p["lat"], p["lon"])
    assert x == pytest.approx(p["x"], abs=POINTS["tolerance_m"])
    assert z == pytest.approx(p["z"], abs=POINTS["tolerance_m"])


@pytest.mark.parametrize("p", POINTS["points"])
def test_roundtrip(p: dict) -> None:
    lat, lon = scene_to_latlon(*latlon_to_scene(p["lat"], p["lon"]))
    assert lat == pytest.approx(p["lat"], abs=1e-9)
    assert lon == pytest.approx(p["lon"], abs=1e-9)


def test_north_is_negative_z_and_east_is_positive_x() -> None:
    o = scene_origin()
    _, z_north = latlon_to_scene(o.lat + 0.01, o.lon)
    x_east, _ = latlon_to_scene(o.lat, o.lon + 0.01)
    assert z_north < 0
    assert x_east > 0


def test_unreal_mapping() -> None:
    assert scene_to_unreal(1.0, 2.0, 3.0) == (100.0, 300.0, 200.0)
