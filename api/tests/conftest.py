from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.db import make_engine, metadata
from api.main import create_app
from api.settings import Settings
from api.tests.fake_sim import FakeSimService
from residents.tests.fixtures import write_population

# Set REDRAW_TEST_DATABASE_URL=postgresql+psycopg://... to run the API tests on Postgres.
PG_URL = os.environ.get("REDRAW_TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    return write_population(tmp_path_factory.mktemp("processed"))


def make_settings(tmp: Path, data_dir: Path, **kw) -> Settings:
    if PG_URL and not str(kw.get("database_url", "")).startswith("mongodb"):
        eng = make_engine(PG_URL)
        with eng.begin() as conn:
            metadata.drop_all(conn)
            conn.execute(text("SELECT 1"))
        eng.dispose()
    base = dict(
        database_url=PG_URL or f"sqlite:///{tmp / 'test.db'}",
        redraw_data_dir=str(data_dir),
        redraw_assets_dir=str(tmp / "assets"),
        llm_base_url="http://127.0.0.1:9/v1",  # nothing listens here: LLM is "down"
        llm_model_fast="test",
        llm_timeout_s=1.0,
        report_seeds=3,
        sim_workers=1,
        redraw_warm_baseline=True,
    )
    base.update(kw)
    return Settings(_env_file=None, **base)


@pytest.fixture(params=["sql", "mongo"])
def backend(request, monkeypatch) -> str:
    """Every API test runs on the SQL store and on the MongoDB store (in-memory mongomock)."""
    if request.param == "mongo":
        import mongomock

        import api.store_mongo

        monkeypatch.setattr(api.store_mongo, "MongoClient", mongomock.MongoClient)
    return request.param


@pytest.fixture
def make_client(tmp_path, data_dir, backend):
    clients: list[TestClient] = []

    def factory(sim: FakeSimService | None = None, llm=None, **kw) -> tuple[TestClient, FakeSimService]:
        if backend == "mongo":
            kw.setdefault("database_url", "mongodb://localhost:27017/redraw_test")
        s = make_settings(tmp_path, data_dir, **kw)
        fake = sim or FakeSimService(data_dir)
        app = create_app(s, sim_factory=lambda: fake, llm=llm)
        c = TestClient(app)
        c.__enter__()
        clients.append(c)
        return c, fake

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> Iterator[TestClient]:
    c, _ = make_client()
    yield c


def wait_job(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def wait_baseline(client: TestClient, timeout: float = 30.0) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        r = client.get("/baseline")
        if r.status_code == 200:
            return r.json()
        time.sleep(0.05)
    raise AssertionError("baseline never became ready")
