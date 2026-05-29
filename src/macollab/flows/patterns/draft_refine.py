from __future__ import annotations

from pocketflow import AsyncFlow

from macollab.flows.nodes import AgentNode

DRAFT_SYSTEM = "Solve the problem. Show brief reasoning, then the final numeric answer."
REFINE_SYSTEM = (
    "You improve a draft answer. Check the reasoning, fix any error, and give a "
    "correct, concise final numeric answer."
)
_REFINE_TMPL = (
    "Problem:\n{task}\n\n"
    "A draft answer from another solver:\n{draft}\n\n"
    "Produce the improved, correct final answer."
)


def _refine_prompt(shared: dict) -> str:
    return _REFINE_TMPL.format(task=shared["task"], draft=shared["responses"]["draft"].text)


def build_draft_refine_flow() -> AsyncFlow:
    draft = AgentNode(
        role="drafter",
        prompt_builder=lambda s: s["task"],
        system=DRAFT_SYSTEM,
        output_key="draft",
    )
    refine = AgentNode(
        role="refiner",
        prompt_builder=_refine_prompt,
        system=REFINE_SYSTEM,
        output_key="refined",
    )
    draft >> refine
    return AsyncFlow(start=draft)
