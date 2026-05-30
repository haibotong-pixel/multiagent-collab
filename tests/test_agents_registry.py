import pytest

from macollab.agents.claude import ClaudeAgentAdapter
from macollab.agents.fake import FakeAgent
from macollab.agents.pydantic_ai import PydanticAIAdapter
from macollab.agents.registry import make_agent


def test_make_fake():
    a = make_agent("A", "fake", "echo-bot")
    assert isinstance(a, FakeAgent)
    assert a.id == "A" and a.model == "echo-bot"


def test_make_pydantic():
    a = make_agent("A", "pydantic", "anthropic:claude-sonnet-4-6")
    assert isinstance(a, PydanticAIAdapter)
    assert a.model == "anthropic:claude-sonnet-4-6"


def test_make_claude():
    a = make_agent("C", "claude", "claude-sonnet-4-6")
    assert isinstance(a, ClaudeAgentAdapter)
    assert a.model == "claude-sonnet-4-6"


def test_unknown_sdk_raises():
    with pytest.raises(ValueError):
        make_agent("X", "nope", "m")
