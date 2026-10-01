"""Virtual town hall (spec 8.4).

Eight speakers: the four most negatively and four most positively affected
residents (by deterministic approval), adjusted so at least three different
schools are represented when the population has that many. Each gives a two to
three sentence public comment (smart model). A player can respond to one
speaker and gets a short in-character follow-up.
"""

from __future__ import annotations

import json
from typing import Any

from residents.llm import ChatLLM, extract_json
from residents.persona import Persona
from residents.reactions import plan_brief, reaction_facts
from residents.textcheck import collect_numbers, safe_text

N_PER_SIDE = 4
MIN_SCHOOLS = 3


def _schools(sel: list[tuple[Persona, dict[str, Any], str]]) -> set[str]:
    return {s for p, _, _ in sel for s in p.school_ids}


def select_speakers(personas: dict[int, Persona], reactions: list[dict[str, Any]],
                    n_per_side: int = N_PER_SIDE, min_schools: int = MIN_SCHOOLS
                    ) -> list[tuple[Persona, dict[str, Any], str]]:
    rs = sorted((r for r in reactions if r["persona_id"] in personas),
                key=lambda r: (r["approval"], r["persona_id"]))
    if not rs:
        return []
    k = min(n_per_side, len(rs) // 2) if len(rs) >= 2 else len(rs)
    neg = rs[:k]
    pos = list(reversed(rs[len(rs) - k:])) if k else []
    rest_neg = rs[k:len(rs) - k]          # ordered from most negative
    rest_pos = list(reversed(rest_neg))   # ordered from most positive
    sel = [(personas[r["persona_id"]], r, "against") for r in neg] + \
          [(personas[r["persona_id"]], r, "for") for r in pos]
    available = {s for p in personas.values() for s in p.school_ids}
    target = min(min_schools, len(available))
    used: set[int] = {r["persona_id"] for _, r, _ in sel}

    def counts() -> dict[str, int]:
        c: dict[str, int] = {}
        for p, _, _ in sel:
            for s in p.school_ids:
                c[s] = c.get(s, 0) + 1
        return c

    # Swap the least extreme speaker whose schools are all duplicated for the most extreme
    # unused resident (same side) who brings a new school.
    guard = 0
    while len(_schools(sel)) < target and guard < 2 * len(sel):
        guard += 1
        have = _schools(sel)
        swapped = False
        for side, pool in (("against", rest_neg), ("for", rest_pos)):
            cand = next((r for r in pool if r["persona_id"] not in used
                         and set(personas[r["persona_id"]].school_ids) - have), None)
            if cand is None:
                continue
            c = counts()
            idxs = [i for i, (p, _, sd) in enumerate(sel)
                    if sd == side and all(c.get(s, 0) > 1 for s in p.school_ids)]
            if not idxs:
                continue
            i = idxs[-1]  # least extreme on that side (lists are ordered by extremeness)
            used.add(cand["persona_id"])
            sel[i] = (personas[cand["persona_id"]], cand, side)
            swapped = True
            break
        if not swapped:
            break
    return sel


_COMMENT_SYSTEM = (
    "You voice fictional residents speaking at a public town hall about a proposed neighborhood "
    "traffic plan in a city planning game. Each speaker gives a 2 to 3 sentence public comment "
    "in first person, in character, consistent with whether they support it. Use only numbers "
    "given in that speaker's facts; never invent numbers, addresses or surnames. Reply with JSON "
    'only: {"comments": [{"id": <id>, "text": "..."}]}'
)


def generate_comments(speakers: list[tuple[Persona, dict[str, Any], str]], plan: dict[str, Any],
                      llm: ChatLLM, school_names: dict[str, str],
                      tool_names: dict[str, str] | None = None) -> dict[int, str]:
    if not llm.available or not speakers:
        return {}
    brief = plan_brief(plan, tool_names)
    facts = {p.persona_id: {**reaction_facts(p, r, school_names), "position": side}
             for p, r, side in speakers}
    user = json.dumps({"plan": brief, "speakers": [{"id": k, **v} for k, v in facts.items()]})
    reply = llm.chat([{"role": "system", "content": _COMMENT_SYSTEM},
                      {"role": "user", "content": user}],
                     model="smart", max_tokens=160 * len(speakers), temperature=0.8, json_mode=True)
    data = extract_json(reply)
    items = data.get("comments", []) if isinstance(data, dict) else []
    out: dict[int, str] = {}
    for it in items:
        try:
            pid = int(it["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if pid in facts:
            nums, clocks = collect_numbers({"f": facts[pid], "plan": brief})
            text = safe_text(it.get("text"), nums, clocks, max_chars=700)
            if text:
                out[pid] = text
    return out


def generate_followup(persona: Persona, reaction: dict[str, Any], comment: str | None,
                      player_message: str, plan: dict[str, Any], llm: ChatLLM,
                      school_names: dict[str, str], tool_names: dict[str, str] | None = None) -> str | None:
    if not llm.available:
        return None
    facts = reaction_facts(persona, reaction, school_names)
    brief = plan_brief(plan, tool_names)
    system = (
        f"You are {persona.first_name}, a fictional resident at a town hall in a city planning game. "
        f"Your facts: {json.dumps(facts)}. The plan: {json.dumps(brief)}. "
        f"Your earlier comment: {json.dumps(comment or '')}. Stay in character. Reply to the "
        "planner in 1 to 3 sentences. Use only numbers from your facts; never invent numbers, "
        "addresses or surnames. Do not change your position unless their reply addresses your concern."
    )
    reply = llm.chat([{"role": "system", "content": system},
                      {"role": "user", "content": player_message}],
                     model="smart", max_tokens=200, temperature=0.7)
    nums, clocks = collect_numbers({"f": facts, "plan": brief, "c": comment or "", "m": player_message})
    return safe_text(reply, nums, clocks, max_chars=700)
