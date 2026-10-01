"""Guards for LLM text: residents may only cite numbers they were given (spec 8.2.3)."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

_NUM = re.compile(r"\d[\d,]*(?:[.:]\d+)?")
_ADDRESS = re.compile(r"\b\d{2,6}\s+[A-Z][a-z]+\s+(?:St|Street|Rd|Road|Ave|Avenue|Dr|Drive|Ct|Court|"
                      r"Ln|Lane|Way|Pl|Place|Blvd|Boulevard|Pkwy|Parkway|Ter|Terrace)\b")


def collect_numbers(obj: Any, out: set[float] | None = None, clocks: set[str] | None = None
                    ) -> tuple[set[float], set[str]]:
    """All numbers (and HH:MM strings) that appear anywhere in a facts structure."""
    out = set() if out is None else out
    clocks = set() if clocks is None else clocks
    if isinstance(obj, bool) or obj is None:
        return out, clocks
    if isinstance(obj, (int, float)):
        if math.isfinite(float(obj)):
            out.add(abs(float(obj)))
    elif isinstance(obj, str):
        for tok in _NUM.findall(obj):
            if ":" in tok:
                clocks.add(tok)
            else:
                try:
                    out.add(abs(float(tok.replace(",", ""))))
                except ValueError:
                    pass
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, out, clocks)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            collect_numbers(v, out, clocks)
    return out, clocks


def numbers_ok(text: str, allowed: Iterable[float], clocks: Iterable[str] = ()) -> bool:
    """True if every number in `text` matches (within rounding) a number that was provided."""
    allowed_l = [abs(a) for a in allowed]
    clock_s = set(clocks)
    for tok in _NUM.findall(text):
        if ":" in tok:
            if tok not in clock_s and tok.lstrip("0") not in {c.lstrip("0") for c in clock_s}:
                return False
            continue
        try:
            n = float(tok.replace(",", "").rstrip("."))
        except ValueError:
            return False
        if not any(abs(a - n) <= 0.051 or round(a) == n or round(a, 1) == n for a in allowed_l):
            return False
    return True


def safe_text(text: str | None, allowed: Iterable[float], clocks: Iterable[str] = (),
              max_chars: int = 600) -> str | None:
    """Return cleaned text, or None if it cites unknown numbers or looks like an address."""
    if not text:
        return None
    t = " ".join(str(text).split()).strip().strip('"').strip()
    if not t or _ADDRESS.search(t):
        return None
    if not numbers_ok(t, allowed, clocks):
        return None
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0] + "..."
    return t
