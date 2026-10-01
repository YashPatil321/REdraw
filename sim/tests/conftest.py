"""Shared fixtures: a tiny self-contained synthetic world written in the data_contract formats."""

from __future__ import annotations

import pytest

from sim.fixtures import build_fixture_world
from sim.service import SimService
from sim.world import WorldState, load_world


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory):
    return build_fixture_world(tmp_path_factory.mktemp("redraw_world"), scale="tiny", seed=3)


@pytest.fixture(scope="session")
def world(fixture_dir) -> WorldState:
    return load_world(fixture_dir)


@pytest.fixture(scope="session")
def svc(fixture_dir) -> SimService:
    return SimService.load(fixture_dir)
