import macollab.agents.claude as claude_mod
from macollab.agents.base import Agent
from macollab.agents.claude import ClaudeAgentAdapter, _extract_claude_response


class _FakeResultMessage:
    def __init__(self, result, usage, cost):
        self.result = result
        self.usage = usage
        self.total_cost_usd = cost


class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


class _FakeAssistantMessage:
    def __init__(self, *texts):
        self.content = [_FakeTextBlock(t) for t in texts]


def test_extract_prefers_result_message_text_and_usage():
    messages = [
        _FakeAssistantMessage("partial..."),
        _FakeResultMessage("the answer is 42", {"input_tokens": 11, "output_tokens": 4}, 0.0007),
    ]
    resp = _extract_claude_response(messages, model="claude-sonnet-4-6")
    assert resp.text == "the answer is 42"
    assert (resp.usage.input, resp.usage.output, resp.usage.total) == (11, 4, 15)
    assert abs(resp.usage.cost_usd - 0.0007) < 1e-9


def test_extract_falls_back_to_assistant_text_when_no_result():
    messages = [_FakeAssistantMessage("hello ", "world")]
    resp = _extract_claude_response(messages, model="claude-sonnet-4-6")
    assert resp.text == "hello world"
    assert resp.usage.total == 0


async def test_run_collects_messages_from_query(monkeypatch):
    async def fake_query(*, prompt, options):
        assert prompt == "6*7?"
        yield _FakeAssistantMessage("thinking...")
        yield _FakeResultMessage("42", {"input_tokens": 5, "output_tokens": 1}, 0.0001)

    # ClaudeAgentOptions must be constructible; replace with a permissive stub.
    monkeypatch.setattr(claude_mod, "_query", fake_query)
    monkeypatch.setattr(claude_mod, "_Options", lambda **kw: dict(kw))

    adapter = ClaudeAgentAdapter(id="C", model="claude-sonnet-4-6")
    resp = await adapter.run("6*7?", system="be concise")
    assert resp.text == "42"
    assert resp.usage.total == 6
    assert resp.meta["sdk"] == "claude"


def test_claude_adapter_satisfies_protocol():
    assert isinstance(ClaudeAgentAdapter(id="C", model="claude-sonnet-4-6"), Agent)
