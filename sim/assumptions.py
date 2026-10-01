"""Assumption access for the sim (spec rule 14.3: no magic numbers).

Every real-world number comes from ``data/config/assumptions.yaml`` through
``A("dotted.path")``. Each key read is recorded so the report card can list the
``verified: false`` assumptions the run actually depended on.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pipeline.config import assumption, assumptions

_USED: set[str] = set()


def A(path: str) -> Any:
    """Return the value of an assumptions.yaml leaf and record that it was used."""
    _USED.add(path)
    return assumption(path)


def Af(path: str) -> float:
    return float(A(path))


def leaf(path: str) -> dict[str, Any]:
    """Full leaf dict (value/unit/range/source/verified)."""
    node: Any = assumptions()
    for part in path.split("."):
        node = node[part]
    return dict(node)


def used_keys() -> list[str]:
    return sorted(_USED)


def unverified_used() -> list[str]:
    """Human readable lines for every verified:false assumption read so far."""
    out = []
    for key in used_keys():
        try:
            lf = leaf(key)
        except (KeyError, TypeError):
            continue
        if not lf.get("verified", False):
            out.append(f"Assumption {key} = {lf.get('value')} {lf.get('unit', '')} ({lf.get('source', 'no source')})".strip())
    return out


def clock_to_s(clock: str) -> int:
    """'07:30' -> seconds since local midnight."""
    hh, mm = str(clock).strip().split(":")[:2]
    h, m = int(hh), int(mm)
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"bad clock time {clock!r}")
    return h * 3600 + m * 60


def s_to_clock(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def assumptions_hash() -> str:
    blob = json.dumps(assumptions(), sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]
