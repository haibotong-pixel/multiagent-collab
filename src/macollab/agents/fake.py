from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from macollab.agents.base import AgentResponse, TokenUsage


@dataclass
class FakeAgent:
    id: str
    model: str = "fake"
    sdk: str = "fake"
    responder: Callable[[str], str] = field(default=lambda task: f"echo: {task}")

    async def run(self, task: str, *, system: str | None = None) -> AgentResponse:
        text = self.responder(task)
        inp = len(task.split())
        out = len(text.split())
        return AgentResponse(
            text=text,
            usage=TokenUsage(input=inp, output=out, total=inp + out),
            raw={"task": task, "system": system},
            meta={"sdk": self.sdk, "model": self.model},
        )
