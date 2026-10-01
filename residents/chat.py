"""Talk to a resident (spec 8.3)."""

from __future__ import annotations

import json
from typing import Any

from pipeline.config import region
from residents.llm import ChatLLM
from residents.persona import Persona
from residents.reactions import plan_brief, reaction_facts
from residents.textcheck import collect_numbers, safe_text


def system_prompt(persona: Persona, reaction: dict[str, Any] | None, plan: dict[str, Any] | None,
                  school_names: dict[str, str], tool_names: dict[str, str] | None = None) -> str:
    facts = reaction_facts(persona, reaction, school_names) if reaction else persona.facts(school_names)
    parts = [
        f"You are {persona.first_name}, a fictional resident of "
        f"{region().get('display_name', 'the neighborhood')} in a city planning game. Stay in character at all times and talk like a real neighbor: short, "
        "plain, 1 to 4 sentences.",
        f"About you: {persona.backstory}",
        f"Your facts: {json.dumps(facts)}",
    ]
    if plan:
        parts.append(f"The plan being discussed: {json.dumps(plan_brief(plan, tool_names))}")
        if reaction:
            parts.append(f"You {'support' if reaction.get('approves') else 'oppose'} this plan.")
    parts.append(
        "Rules: only cite numbers that appear in your facts above; if asked for a number you "
        "were not given, say you don't know. Never give a surname, street address or house "
        "number. Never claim to be a real person. If asked something unrelated to the "
        "neighborhood, steer back politely."
    )
    return "\n".join(parts)


def reply(persona: Persona, reaction: dict[str, Any] | None, plan: dict[str, Any] | None,
          history: list[dict[str, Any]], message: str, llm: ChatLLM, school_names: dict[str, str],
          tool_names: dict[str, str] | None = None) -> str | None:
    if not llm.available:
        return None
    sys = system_prompt(persona, reaction, plan, school_names, tool_names)
    msgs = [{"role": "system", "content": sys}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history
             if m.get("role") in ("user", "assistant") and m.get("content")]
    msgs.append({"role": "user", "content": message})
    text = llm.chat(msgs, model="fast", max_tokens=220, temperature=0.7)
    facts = reaction_facts(persona, reaction, school_names) if reaction else persona.facts(school_names)
    nums, clocks = collect_numbers({"f": facts, "p": plan_brief(plan, tool_names) if plan else {},
                                    "h": [m.get("content", "") for m in history if m.get("role") == "user"],
                                    "m": message})
    return safe_text(text, nums, clocks, max_chars=900)
