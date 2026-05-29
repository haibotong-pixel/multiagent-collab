from __future__ import annotations

from typing import Any

from macollab.agents.base import AgentResponse, TokenUsage


def _extract_claude_response(messages: list[Any], *, model: str) -> AgentResponse:
    """Pure, duck-typed extraction of text + usage from a list of SDK messages.

    A ResultMessage is identified by having both ``result`` and ``total_cost_usd``;
    its ``usage`` is a dict (Anthropic Messages API shape). Otherwise we accumulate
    text from AssistantMessage-like objects whose ``content`` is a list of blocks
    with a ``text`` attribute.
    """
    result_text: str | None = None
    usage_dict: dict | None = None
    cost: float | None = None
    text_parts: list[str] = []

    for m in messages:
        if hasattr(m, "result") and hasattr(m, "total_cost_usd"):
            result_text = getattr(m, "result", None)
            usage_dict = getattr(m, "usage", None)
            cost = getattr(m, "total_cost_usd", None)
        elif isinstance(getattr(m, "content", None), list):
            for block in m.content:
                t = getattr(block, "text", None)
                if t:
                    text_parts.append(t)

    text = result_text if result_text else "".join(text_parts)
    u = usage_dict or {}
    inp = int(u.get("input_tokens", 0) or 0)
    out = int(u.get("output_tokens", 0) or 0)
    return AgentResponse(
        text=text or "",
        usage=TokenUsage(input=inp, output=out, total=inp + out, cost_usd=cost),
        raw=messages,
        meta={"sdk": "claude", "model": model},
    )


# Indirection points so tests can monkeypatch without importing the real SDK.
def _query(*, prompt: str, options):
    from claude_agent_sdk import query

    return query(prompt=prompt, options=options)


def _Options(**kwargs):
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(**kwargs)


class ClaudeAgentAdapter:
    sdk = "claude"

    def __init__(self, id: str, model: str = "claude-sonnet-4-6", *, max_turns: int = 1) -> None:
        self.id = id
        self.model = model
        self.max_turns = max_turns

    async def run(self, task: str, *, system: str | None = None) -> AgentResponse:
        options = _Options(
            model=self.model,
            max_turns=self.max_turns,
            system_prompt=system,
            permission_mode="bypassPermissions",
        )
        messages: list[Any] = []
        async for message in _query(prompt=task, options=options):
            messages.append(message)
        return _extract_claude_response(messages, model=self.model)
