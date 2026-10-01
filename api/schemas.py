"""Request bodies (Pydantic validation -> 422 on bad shape)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolInstance(BaseModel):
    tool: str = Field(min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


class PlanIn(BaseModel):
    mission: str = Field(default="morning_crunch", min_length=1, max_length=64)
    title: str = Field(default="Untitled plan", max_length=200)
    pitch: str = Field(default="", max_length=4000)
    tools: list[ToolInstance] = Field(default_factory=list, max_length=40)

    def plan_dict(self, **extra: Any) -> dict[str, Any]:
        return {"mission": self.mission, "title": self.title.strip() or "Untitled plan",
                "pitch": self.pitch, "tools": [t.model_dump() for t in self.tools], **extra}


class VoteIn(BaseModel):
    value: Literal[1, -1, 0]


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    plan_id: str | None = None


class TownhallIn(BaseModel):
    persona_id: int | None = None
    message: str | None = Field(default=None, max_length=1000)
    regenerate: bool = False


class CustomPreviewIn(BaseModel):
    description: str = Field(min_length=3, max_length=600)
