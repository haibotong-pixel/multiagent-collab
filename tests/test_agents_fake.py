from macollab.agents.base import Agent
from macollab.agents.fake import FakeAgent


async def test_fake_agent_default_echo():
    agent = FakeAgent(id="A", model="echo-bot")
    resp = await agent.run("hello world")
    assert resp.text == "echo: hello world"
    assert resp.usage.total == resp.usage.input + resp.usage.output
    assert resp.meta["sdk"] == "fake"


async def test_fake_agent_custom_responder_receives_prompt():
    agent = FakeAgent(id="B", model="fixed", responder=lambda task: f"saw[{task}]->42")
    resp = await agent.run("6*7?")
    assert resp.text == "saw[6*7?]->42"


def test_fake_agent_satisfies_protocol():
    assert isinstance(FakeAgent(id="A"), Agent)
