"""Shared config loading for the pipeline, sim, residents and API.

All region specific inputs live in data/config/ (spec 13.2). Paths are resolved
relative to the repo root so scripts work from any working directory.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "data" / "config"


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def raw_dir() -> Path:
    return _env_path("REDRAW_RAW_DIR", REPO_ROOT / "data" / "raw")


def processed_dir() -> Path:
    return _env_path("REDRAW_DATA_DIR", REPO_ROOT / "data" / "processed")


def assets_dir() -> Path:
    return _env_path("REDRAW_ASSETS_DIR", REPO_ROOT / "client" / "public" / "assets")


def load_yaml(name: str) -> Any:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def region() -> dict[str, Any]:
    return load_yaml("region.yaml")


@lru_cache(maxsize=1)
def assumptions() -> dict[str, Any]:
    return load_yaml("assumptions.yaml")


def assumption(path: str) -> Any:
    """Return the `.value` of a dotted assumptions.yaml path, e.g. 'assignment.bpr_alpha'.

    Raises KeyError with the full path if missing, so missing assumptions fail loudly.
    """
    node: Any = assumptions()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"assumptions.yaml is missing '{path}' (failed at '{part}')")
        node = node[part]
    if isinstance(node, dict) and "value" in node:
        return node["value"]
    raise KeyError(f"assumptions.yaml '{path}' is not a leaf with a 'value' field")


def assumption_range(path: str) -> tuple[Any, Any]:
    """The `.range` [low, high] of a dotted assumptions.yaml leaf (KeyError if missing/null)."""
    node: Any = assumptions()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"assumptions.yaml is missing '{path}' (failed at '{part}')")
        node = node[part]
    rng = node.get("range") if isinstance(node, dict) else None
    if not isinstance(rng, list) or len(rng) != 2:
        raise KeyError(f"assumptions.yaml '{path}' has no [low, high] range")
    return rng[0], rng[1]
