from __future__ import annotations

from typing import Callable

from pocketflow import AsyncNode

from macollab.agents.base import Agent, AgentResponse


class AgentNode(AsyncNode):
    """A PocketFlow async node that runs one bound Agent for a given role.

    Reads the agent from ``shared["roles"][role]``, builds the prompt from the
    shared store via ``prompt_builder``, runs the agent, and writes the response
    to ``shared["responses"][output_key]``, appends usage to ``shared["usage_log"]``,
    and sets ``shared["final"]`` to the response text (last node wins).
    """

    def __init__(
        self,
        role: str,
        prompt_builder: Callable[[dict], str],
        *,
        system: str | None = None,
        output_key: str | None = None,
        max_retries: int = 2,
        wait: float = 1.0,
    ) -> None:
        super().__init__(max_retries=max_retries, wait=wait)
        self.role = role
        self.prompt_builder = prompt_builder
        self.system = system
        self.output_key = output_key or role

    async def prep_async(self, shared) -> tuple[Agent, str]:
        agent: Agent = shared["roles"][self.role]
        prompt = self.prompt_builder(shared)
        return agent, prompt

    async def exec_async(self, prep_res) -> AgentResponse:
        agent, prompt = prep_res
        return await agent.run(prompt, system=self.system)

    async def post_async(self, shared, prep_res, exec_res: AgentResponse) -> str:
        shared.setdefault("responses", {})[self.output_key] = exec_res
        shared.setdefault("usage_log", []).append(exec_res.usage)
        shared["final"] = exec_res.text
        return "default"
