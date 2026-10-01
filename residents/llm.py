"""OpenAI-compatible chat client (vLLM, llama.cpp server, OpenAI, ...) via httpx.

Robust to the endpoint being down (spec 8.5): every call returns None instead of
raising, connects with a short timeout, and a circuit breaker skips calls for a
cool-down period after a failure so 500 residents do not each wait for a timeout.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Literal, Protocol

import httpx

log = logging.getLogger(__name__)

ModelKind = Literal["fast", "smart"]


class ChatLLM(Protocol):
    """What residents code needs from an LLM (lets tests inject a fake)."""

    @property
    def available(self) -> bool: ...

    def chat(self, messages: list[dict[str, str]], *, model: ModelKind = "fast",
             max_tokens: int = 200, temperature: float = 0.7, json_mode: bool = False) -> str | None: ...


class LLMClient:
    def __init__(self, base_url: str, api_key: str = "", model_fast: str = "", model_smart: str = "",
                 timeout_s: float = 30.0, connect_timeout_s: float = 2.0,
                 breaker_cooldown_s: float = 60.0, breaker_threshold: int = 1) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_fast = model_fast
        self.model_smart = model_smart
        self.timeout = httpx.Timeout(timeout_s, connect=connect_timeout_s)
        self.breaker_cooldown_s = breaker_cooldown_s
        self.breaker_threshold = breaker_threshold
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0
        self._discovered_model: str | None = None
        self.last_error: str | None = None

    # ----------------------------------------------------------- breaker
    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    @property
    def available(self) -> bool:
        """True if configured and the circuit breaker is closed (may still fail)."""
        return self.configured and time.monotonic() >= self._open_until

    def _record_failure(self, err: str) -> None:
        with self._lock:
            was_closed = time.monotonic() >= self._open_until
            self._failures += 1
            self.last_error = err
            if self._failures >= self.breaker_threshold:
                self._open_until = time.monotonic() + self.breaker_cooldown_s
        if was_closed:
            log.warning("LLM endpoint failed (%s); residents continue without text for %.0fs",
                        err, self.breaker_cooldown_s)
        else:
            log.debug("LLM call failed (%s)", err)

    def _record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0
            self.last_error = None

    def reset(self) -> None:
        self._record_success()

    # ------------------------------------------------------------- calls
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _model_name(self, kind: ModelKind) -> str | None:
        name = self.model_smart if kind == "smart" else self.model_fast
        name = name or self.model_fast or self.model_smart
        if name:
            return name
        if self._discovered_model:
            return self._discovered_model
        # No model configured: ask the server (vLLM serves one model).
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout)
            r.raise_for_status()
            data = r.json().get("data") or []
            if data:
                self._discovered_model = str(data[0]["id"])
                return self._discovered_model
        except Exception as e:  # noqa: BLE001 - endpoint down is an expected state
            self._record_failure(f"model discovery: {e}")
        return None

    def chat(self, messages: list[dict[str, str]], *, model: ModelKind = "fast",
             max_tokens: int = 200, temperature: float = 0.7, json_mode: bool = False) -> str | None:
        if not self.available:
            return None
        name = self._model_name(model)
        if not name or not self.available:
            return None
        body: dict[str, Any] = {"model": name, "messages": messages, "max_tokens": max_tokens,
                                "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                           json=body, timeout=self.timeout)
            if r.status_code == 400 and json_mode:
                # Some servers reject response_format; retry once without it.
                body.pop("response_format", None)
                r = httpx.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                               json=body, timeout=self.timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            self._record_failure(str(e) or type(e).__name__)
            return None
        self._record_success()
        return (text or "").strip() or None


def extract_json(text: str | None) -> Any:
    """Parse the first JSON object/array in an LLM reply (tolerates code fences). None if absent."""
    if not text:
        return None
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = t.find(opener)
        end = t.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


class NullLLM:
    """LLM that is always down (used when no endpoint is configured)."""

    available = False

    def chat(self, messages: list[dict[str, str]], **_: Any) -> str | None:
        return None
