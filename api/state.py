"""Process-wide API state: settings, DB store, the SimService, baseline cache, residents."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from api.db import clean_json
from api.settings import Settings
from api.store import Store
from residents.blocks import BlockLabeler
from residents.llm import ChatLLM
from residents.service import ResidentsService

log = logging.getLogger(__name__)

SimFactory = Callable[[], Any]
BASELINE_RETRY_S = 30.0


class ServiceUnavailable(Exception):
    """Mapped to HTTP 503 with {"detail": ...}."""


def default_sim_factory() -> Any:
    from sim.service import SimService  # imported lazily: the sim is optional for tests

    return SimService.load()


def as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(vars(obj))


def field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


class AppState:
    def __init__(self, settings: Settings, store: Store, sim_factory: SimFactory, llm: ChatLLM) -> None:
        self.settings = settings
        self.store = store
        self.sim_factory = sim_factory
        self.llm = llm
        self.residents = ResidentsService(settings.data_dir, settings.personas_path, llm)

        self._sim: Any = None
        self._sim_error: str | None = None
        self._sim_lock = threading.Lock()

        self.baseline_summary: dict[str, Any] | None = None
        self.baseline_playback: bytes | None = None
        self.baseline_error: str | None = None
        self.baseline_progress: tuple[float, str] = (0.0, "not started")
        self.baseline_done = threading.Event()
        self.warm_thread: threading.Thread | None = None
        self._baseline_failed_at = 0.0

        self._cache: dict[str, Any] = {}
        self._cache_lock = threading.RLock()

    # --------------------------------------------------------------- sim
    def sim(self) -> Any:
        if self._sim is not None:
            return self._sim
        with self._sim_lock:
            if self._sim is None:
                try:
                    t0 = time.monotonic()
                    self._sim = self.sim_factory()
                    self._sim_error = None
                    log.info("SimService loaded in %.1fs", time.monotonic() - t0)
                    self._on_sim_loaded()
                except Exception as e:  # noqa: BLE001
                    self._sim_error = f"{type(e).__name__}: {e}"
                    log.exception("SimService failed to load")
                    raise ServiceUnavailable(
                        f"Simulation world is not available ({self._sim_error}). Build world data with "
                        "`.venv/bin/python pipeline/build_all.py` (or --synthetic) and restart the API."
                    ) from e
        return self._sim

    def _on_sim_loaded(self) -> None:
        try:
            self.residents.tool_names = {t["id"]: t.get("name", t["id"]) for t in self._sim.tools()}
        except Exception:  # noqa: BLE001
            pass

    def cached(self, key: str, fn: Callable[[], Any]) -> Any:
        if key in self._cache:
            return self._cache[key]
        with self._cache_lock:
            if key not in self._cache:
                self._cache[key] = fn()
        return self._cache[key]

    # ---------------------------------------------------------- baseline
    def start_warmup(self) -> None:
        if self.warm_thread is not None and self.warm_thread.is_alive():
            return
        self.baseline_error = None
        self.baseline_done.clear()
        self._baseline_failed_at = 0.0
        self.warm_thread = threading.Thread(target=self.warm, name="redraw-warmup", daemon=True)
        self.warm_thread.start()

    def warm(self) -> None:
        try:
            sim = self.sim()
            self.baseline_progress = (0.0, "starting baseline")

            def cb(frac: float, msg: str) -> None:
                self.baseline_progress = (float(frac), str(msg))

            res = sim.run(None, seeds=self.settings.report_seeds, workers=self.settings.sim_workers,
                          progress=cb)
            self.baseline_summary = clean_json(as_dict(field(res, "baseline_summary")))
            pb = field(res, "playback_baseline")
            self.baseline_playback = bytes(pb) if pb else None
            if self.baseline_playback:
                atomic_write(self.settings.playback_dir / "baseline.rdpb", self.baseline_playback)
            self.baseline_progress = (1.0, "ready")
            log.info("baseline ready")
        except Exception as e:  # noqa: BLE001
            self.baseline_error = f"{type(e).__name__}: {e}"
            self._baseline_failed_at = time.monotonic()
            log.exception("baseline warm-up failed")
        finally:
            self.baseline_done.set()
        # Personas are needed by the first plan run; build/load them now too.
        try:
            self.residents.personas()
            self.residents.upgrade_backstories()
        except Exception as e:  # noqa: BLE001
            log.warning("personas not available yet: %s", e)

    def require_baseline(self) -> dict[str, Any]:
        if self.baseline_summary is not None:
            return self.baseline_summary
        if self.baseline_error:
            err = self.baseline_error
            if time.monotonic() - self._baseline_failed_at > BASELINE_RETRY_S:
                self.start_warmup()  # e.g. world data was built after the API started
            raise ServiceUnavailable(f"Baseline failed: {err}")
        if self._sim_error:
            raise ServiceUnavailable(f"Simulation world is not available ({self._sim_error})")
        if self.warm_thread is None:
            self.start_warmup()
        frac, msg = self.baseline_progress
        raise ServiceUnavailable(f"Baseline is warming up ({frac * 100:.0f}%: {msg}); try again shortly.")

    # ------------------------------------------------------------- world
    def region_meta(self) -> dict[str, Any]:
        def load() -> dict[str, Any]:
            p = self.settings.data_dir / "region_meta.json"
            return json.loads(p.read_text()) if p.exists() else {}
        return self.cached("region_meta", load)

    def block_labeler(self) -> BlockLabeler:
        return self.cached("labeler", lambda: BlockLabeler.from_data_dir(self.settings.data_dir))

    def buildings_index(self) -> dict[int, dict[str, Any]]:
        def load() -> dict[int, dict[str, Any]]:
            p = self.settings.data_dir / "buildings.geojson"
            if not p.exists():
                raise ServiceUnavailable(f"{p} not found; build world data first.")
            fc = json.loads(p.read_text())
            out: dict[int, dict[str, Any]] = {}
            for f in fc.get("features", []):
                props = f.get("properties") or {}
                if "id" in props:
                    out[int(props["id"])] = props
            return out
        return self.cached("buildings", load)

    def households_by_building(self) -> dict[int, int]:
        def load() -> dict[int, int]:
            import pandas as pd

            p = self.settings.data_dir / "households.parquet"
            if not p.exists():
                return {}
            s = pd.read_parquet(p, columns=["building_id"])["building_id"].value_counts()
            return {int(k): int(v) for k, v in s.items()}
        return self.cached("hh_by_building", load)
