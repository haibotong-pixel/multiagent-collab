from __future__ import annotations

from macollab.agents.base import AgentResponse, TokenUsage


class PydanticAIAdapter:
    sdk = "pydantic"

    def __init__(self, id: str, model: str) -> None:
        self.id = id
        self.model = model

    def _make(self, system: str | None):
        # Imported lazily so the rest of the package works without pydantic-ai installed.
        from pydantic_ai import Agent as PydAgent

        if system:
            return PydAgent(self.model, instructions=system)
        return PydAgent(self.model)

    async def run(self, task: str, *, system: str | None = None) -> AgentResponse:
        agent = self._make(system)
        result = await agent.run(task)
        u = result.usage()
        usage = TokenUsage(
            input=getattr(u, "input_tokens", 0) or 0,
            output=getattr(u, "output_tokens", 0) or 0,
            total=getattr(u, "total_tokens", 0) or 0,
        )
        return AgentResponse(
            text=str(result.output),
            usage=usage,
            raw=result,
            meta={"sdk": self.sdk, "model": self.model},
        )
