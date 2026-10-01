"""ResidentsService: the one object the API uses for everything resident related."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from residents import chat as chat_mod
from residents import custom_tool, persona, reactions, townhall
from residents.llm import ChatLLM
from residents.persona import Persona

log = logging.getLogger(__name__)


class ResidentsService:
    def __init__(self, data_dir: Path, cache_path: Path, llm: ChatLLM) -> None:
        self.data_dir = data_dir
        self.cache_path = cache_path
        self.llm = llm
        self._lock = threading.Lock()
        self._personas: list[Persona] | None = None
        self._by_id: dict[int, Persona] = {}
        self._school_names: dict[str, str] | None = None
        self.tool_names: dict[str, str] = {}

    # ------------------------------------------------------------ personas
    def personas(self) -> list[Persona]:
        if self._personas is None:
            with self._lock:
                if self._personas is None:
                    # Template backstories first (fast); LLM backstories come from
                    # upgrade_backstories() in the background and are cached.
                    ps = persona.load_or_build(self.data_dir, self.cache_path, None)
                    self._by_id = {p.persona_id: p for p in ps}
                    self._personas = ps
        return self._personas

    def persona_map(self) -> dict[int, Persona]:
        self.personas()
        return self._by_id

    def get(self, persona_id: int) -> Persona | None:
        return self.persona_map().get(persona_id)

    @property
    def school_names(self) -> dict[str, str]:
        if self._school_names is None:
            self._school_names = persona.school_names(self.data_dir)
        return self._school_names

    def upgrade_backstories(self) -> int:
        try:
            return persona.upgrade_backstories(self.data_dir, self.cache_path, self.personas(), self.llm)
        except Exception as e:  # noqa: BLE001
            log.warning("backstory upgrade failed: %s", e)
            return 0

    # ----------------------------------------------------------- reactions
    def compute(self, plan: dict[str, Any], person_deltas: dict[str, Any] | None,
                cost_per_year_usd: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Deterministic reactions + the `resident_approval_pct` report metric."""
        ps = self.personas()
        rs = reactions.compute_reactions(ps, person_deltas, plan, cost_per_year_usd)
        for r in rs:  # baseline mode is known after a run
            p = self._by_id.get(r["persona_id"])
            mb = (r.get("deltas") or {}).get("mode_baseline")
            if p is not None and mb and not p.commute_mode:
                p.commute_mode = str(mb)
        metric = reactions.approval_metric(reactions.approval_summary(rs), len(ps))
        return rs, metric

    def texts(self, plan: dict[str, Any], rs: list[dict[str, Any]]) -> dict[int, str]:
        return reactions.generate_texts(self.persona_map(), rs, plan, self.llm, self.school_names,
                                        self.tool_names)

    def reaction_view(self, r: dict[str, Any]) -> dict[str, Any] | None:
        p = self.get(int(r["persona_id"]))
        if p is None:
            return None
        pub = p.public(self.school_names)
        return {
            "persona_id": p.persona_id, "first_name": p.first_name, "age": p.age, "block": p.block,
            "x": p.x, "z": p.z, "school_ids": p.school_ids, "values": p.values,
            "approval": r["approval"], "approves": r["approves"], "deltas": r.get("deltas") or {},
            "text": r.get("text"), "job_area": pub["job_area"], "commute_mode": p.commute_mode,
        }

    # ------------------------------------------------------------ townhall
    def select_speakers(self, rs: list[dict[str, Any]]) -> list[tuple[Persona, dict[str, Any], str]]:
        return townhall.select_speakers(self.persona_map(), rs)

    def comments(self, speakers: list[tuple[Persona, dict[str, Any], str]], plan: dict[str, Any]) -> dict[int, str]:
        return townhall.generate_comments(speakers, plan, self.llm, self.school_names, self.tool_names)

    def followup(self, p: Persona, r: dict[str, Any], comment: str | None, message: str,
                 plan: dict[str, Any]) -> str | None:
        return townhall.generate_followup(p, r, comment, message, plan, self.llm, self.school_names,
                                          self.tool_names)

    # ---------------------------------------------------------------- chat
    def chat(self, p: Persona, r: dict[str, Any] | None, plan: dict[str, Any] | None,
             history: list[dict[str, Any]], message: str) -> str | None:
        return chat_mod.reply(p, r, plan, history, message, self.llm, self.school_names, self.tool_names)

    # --------------------------------------------------------- custom tool
    def custom_preview(self, description: str, schools: list[dict[str, Any]]) -> dict[str, Any]:
        return custom_tool.preview(description, self.llm, schools)
