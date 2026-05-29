from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def _add_opt(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    return (a or 0.0) + (b or 0.0)


@dataclass
class TokenUsage:
    input: int = 0
    output: int = 0
    total: int = 0
    cost_usd: float | None = None

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            total=self.total + other.total,
            cost_usd=_add_opt(self.cost_usd, other.cost_usd),
        )


@dataclass
class AgentResponse:
    text: str
    usage: TokenUsage
    structured: Any | None = None
    raw: Any = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    id: str
    sdk: str
    model: str

    async def run(self, task: str, *, system: str | None = None) -> AgentResponse:
        ...
