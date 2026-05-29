from macollab.agents.fake import FakeAgent
from macollab.flows.nodes import AgentNode


async def test_agent_node_writes_response_usage_and_final():
    agent = FakeAgent(id="A", responder=lambda task: f"answer to: {task}")
    node = AgentNode(role="solo", prompt_builder=lambda s: s["task"], output_key="solo")
    shared = {"task": "2+2", "roles": {"solo": agent}, "responses": {}, "usage_log": [], "final": None}

    await node.run_async(shared)

    assert shared["responses"]["solo"].text == "answer to: 2+2"
    assert shared["final"] == "answer to: 2+2"
    assert len(shared["usage_log"]) == 1


async def test_agent_node_prompt_builder_can_read_prior_response():
    drafter = FakeAgent(id="A", responder=lambda task: "DRAFT")
    refiner = FakeAgent(id="B", responder=lambda task: f"refined<{task}>")
    shared = {
        "task": "Q",
        "roles": {"drafter": drafter, "refiner": refiner},
        "responses": {"draft": await drafter.run("Q")},  # pre-seed a draft
        "usage_log": [],
        "final": None,
    }
    node = AgentNode(
        role="refiner",
        prompt_builder=lambda s: f"{s['task']}|{s['responses']['draft'].text}",
        output_key="refined",
    )
    await node.run_async(shared)
    assert shared["responses"]["refined"].text == "refined<Q|DRAFT>"
