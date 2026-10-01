"""Hazard / time-loop event hook (spec 13.5).

Events are objects with ``apply(world, capf, extra)`` that modify the per-bin
capacity factor ``capf`` (B, E) and extra delay ``extra`` (B, E) arrays in
place before every MSA iteration's edge times are computed. Add them to
``world.events`` (a plan tool or a future fire mission can do this). Routing
avoids closed edges because closures add a very large delay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class TimeEvent(Protocol):
    def apply(self, world: Any, capf: np.ndarray, extra: np.ndarray) -> None: ...


CLOSED_DELAY_S = 1.0e5  # numerical "infinite" delay for a closed edge (not a real-world number)


@dataclass
class RoadClosure:
    """Close edges between start_s and end_s (seconds since midnight)."""

    edges: list[int]
    start_s: float
    end_s: float

    def apply(self, world: Any, capf: np.ndarray, extra: np.ndarray) -> None:
        tg = world.time
        b0 = int(np.clip((self.start_s - tg.bin_start_s) // tg.bin_s, 0, tg.n_bins))
        b1 = int(np.clip(np.ceil((self.end_s - tg.bin_start_s) / tg.bin_s), 0, tg.n_bins))
        if b1 > b0 and self.edges:
            idx = np.asarray(self.edges, dtype=np.int64)
            extra[b0:b1, idx] += CLOSED_DELAY_S
            capf[b0:b1, idx] *= 1e-3
