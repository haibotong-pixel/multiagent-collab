from macollab.agents.base import Agent
from macollab.agents.pydantic_ai import PydanticAIAdapter


class _FakeUsage:
    input_tokens = 7
    output_tokens = 3
    total_tokens = 10


class _FakeResult:
    def __init__(self, text):
        self._text = text

    @property
    def output(self):
        return self._text

    def usage(self):
        return _FakeUsage()


class _FakePydAgent:
    def __init__(self, text):
        self._text = text
        self.seen = None

    async def run(self, task, **kwargs):
        self.seen = task
        return _FakeResult(self._text)


async def test_pydantic_adapter_normalizes_output_and_usage(monkeypatch):
    adapter = PydanticAIAdapter(id="A", model="anthropic:claude-sonnet-4-6")
    fake = _FakePydAgent("the answer is 42")
    monkeypatch.setattr(adapter, "_make", lambda system: fake)

    resp = await adapter.run("6*7?", system="be concise")

    assert resp.text == "the answer is 42"
    assert (resp.usage.input, resp.usage.output, resp.usage.total) == (7, 3, 10)
    assert resp.meta["sdk"] == "pydantic"
    assert fake.seen == "6*7?"


def test_pydantic_adapter_satisfies_protocol():
    assert isinstance(PydanticAIAdapter(id="A", model="anthropic:claude-sonnet-4-6"), Agent)
