import pytest

from macollab.agents.fake import FakeAgent
from macollab.flows.patterns import PATTERNS, build_flow


def _shared(task, roles):
    return {"task": task, "ground_truth": None, "roles": roles,
            "responses": {}, "usage_log": [], "final": None}


async def test_single_flow_produces_final_from_solo():
    agent = FakeAgent(id="A", responder=lambda t: "the answer is 42")
    flow = PATTERNS["single"]()
    shared = _shared("6*7?", {"solo": agent})
    await flow.run_async(shared)
    assert shared["final"] == "the answer is 42"
    assert len(shared["usage_log"]) == 1


async def test_draft_refine_runs_both_roles_in_order():
    drafter = FakeAgent(id="A", responder=lambda t: "draft answer 40")
    # refiner echoes its prompt so we can assert it received the draft text
    refiner = FakeAgent(id="B", responder=lambda t: f"FINAL[{t}]")
    flow = PATTERNS["draft_refine"]()
    shared = _shared("6*7?", {"drafter": drafter, "refiner": refiner})
    await flow.run_async(shared)
    assert shared["responses"]["draft"].text == "draft answer 40"
    assert "draft answer 40" in shared["final"]          # refiner saw the draft
    assert shared["final"].startswith("FINAL[")
    assert len(shared["usage_log"]) == 2


def test_build_flow_unknown_pattern_raises():
    with pytest.raises(KeyError):
        build_flow("does_not_exist")
