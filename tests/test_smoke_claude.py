import os
import shutil

import pytest

from macollab.agents.claude import ClaudeAgentAdapter

pytestmark = pytest.mark.smoke


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") or shutil.which("node") is None,
    reason="needs ANTHROPIC_API_KEY and Node.js (Claude Code CLI)",
)
async def test_claude_real_call_returns_number():
    adapter = ClaudeAgentAdapter(id="C", model="claude-sonnet-4-6")
    resp = await adapter.run("What is 6 times 7? Reply with just the number.")
    assert "42" in resp.text
