import os

import pytest

from macollab.agents.pydantic_ai import PydanticAIAdapter

pytestmark = pytest.mark.smoke


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
async def test_pydantic_real_call_returns_number():
    adapter = PydanticAIAdapter(id="A", model="anthropic:claude-sonnet-4-6")
    resp = await adapter.run("What is 6 times 7? Reply with just the number.")
    assert "42" in resp.text
    assert resp.usage.total > 0
