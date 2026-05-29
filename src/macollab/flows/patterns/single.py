from __future__ import annotations

from pocketflow import AsyncFlow

from macollab.flows.nodes import AgentNode

SOLO_SYSTEM = "Solve the problem. Think briefly, then state the final numeric answer."


def build_single_flow() -> AsyncFlow:
    solo = AgentNode(
        role="solo",
        prompt_builder=lambda s: s["task"],
        system=SOLO_SYSTEM,
        output_key="solo",
    )
    return AsyncFlow(start=solo)
