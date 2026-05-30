from __future__ import annotations

from macollab.agents.base import Agent
from macollab.agents.claude import ClaudeAgentAdapter
from macollab.agents.fake import FakeAgent
from macollab.agents.pydantic_ai import PydanticAIAdapter


def make_agent(agent_id: str, sdk: str, model: str) -> Agent:
    if sdk == "fake":
        return FakeAgent(id=agent_id, model=model or "fake")
    if sdk == "pydantic":
        return PydanticAIAdapter(id=agent_id, model=model)
    if sdk == "claude":
        return ClaudeAgentAdapter(id=agent_id, model=model or "claude-sonnet-4-6")
    raise ValueError(f"unknown sdk: {sdk!r} (supported: fake, pydantic, claude)")
