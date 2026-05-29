from macollab.agents.base import TokenUsage, AgentResponse


def test_token_usage_add_sums_fields_and_costs():
    a = TokenUsage(input=10, output=5, total=15, cost_usd=0.01)
    b = TokenUsage(input=2, output=3, total=5, cost_usd=0.02)
    c = a + b
    assert (c.input, c.output, c.total) == (12, 8, 20)
    assert abs(c.cost_usd - 0.03) < 1e-9


def test_token_usage_add_handles_none_cost():
    a = TokenUsage(input=1, output=1, total=2)            # cost_usd None
    b = TokenUsage(input=1, output=1, total=2, cost_usd=0.05)
    assert (a + b).cost_usd == 0.05
    assert (a + a).cost_usd is None


def test_token_usage_sum_with_zero_start():
    items = [TokenUsage(1, 1, 2), TokenUsage(2, 2, 4)]
    total = sum(items, TokenUsage())
    assert (total.input, total.output, total.total) == (3, 3, 6)


def test_agent_response_defaults():
    r = AgentResponse(text="hi", usage=TokenUsage(1, 1, 2))
    assert r.text == "hi"
    assert r.structured is None
    assert r.meta == {}
