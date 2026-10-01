"""API settings, read from the environment and `.env` (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from pipeline.config import REPO_ROOT

ENV_FILE = REPO_ROOT / ".env"


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), extra="ignore", env_file_encoding="utf-8")

    database_url: str = "sqlite:///data/processed/redraw.db"

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model_fast: str = ""
    llm_model_smart: str = ""
    llm_timeout_s: float = 30.0

    sim_workers: int = 4
    report_seeds: int = 20
    redraw_data_dir: str = "data/processed"
    redraw_assets_dir: str = "client/public/assets"

    default_mission: str = "morning_crunch"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # Compute the baseline in a background thread at startup (spec 9: cache baseline at startup).
    redraw_warm_baseline: bool = True
    # Plan runs executed concurrently (each run already uses SIM_WORKERS processes).
    redraw_job_concurrency: int = 1
    cookie_secure: bool = False

    @property
    def data_dir(self) -> Path:
        return _resolve(self.redraw_data_dir)

    @property
    def assets_dir(self) -> Path:
        return _resolve(self.redraw_assets_dir)

    @property
    def playback_dir(self) -> Path:
        return self.data_dir / "playback"

    @property
    def personas_path(self) -> Path:
        return self.data_dir / "personas.json"

    @property
    def sqlalchemy_url(self) -> str:
        """Database URL with relative SQLite paths resolved against the repo root."""
        url = self.database_url
        prefix = "sqlite:///"
        if url.startswith(prefix) and not url.startswith("sqlite:////"):
            rest = url[len(prefix):]
            if rest and rest != ":memory:" and not rest.startswith(":"):
                path = _resolve(rest)
                path.parent.mkdir(parents=True, exist_ok=True)
                return prefix + str(path)
        return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Export .env into os.environ too, so the sim (pipeline.config.processed_dir) sees
    # the same REDRAW_DATA_DIR as the API. Existing env vars win.
    load_dotenv(ENV_FILE, override=False)
    return Settings()
