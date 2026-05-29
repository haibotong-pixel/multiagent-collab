# Multi-Agent Collaboration Framework — v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v1 vertical slice of the framework: a uniform async `Agent` adapter (Pydantic AI + Claude Agent SDK + a no-network Fake), a PocketFlow orchestration layer with the **P4 Draft→Refine** pattern and the **B0 single-agent** baseline, a GSM8K-mini reasoning suite with a numeric scorer, and a config-driven runner that writes JSONL results and prints a metrics table (P4 vs B0).

**Architecture:** Each agent SDK is wrapped behind one async `Agent` protocol (`run(task, *, system) -> AgentResponse`). Collaboration patterns are PocketFlow `AsyncFlow`s of generic `AgentNode`s that read their bound agent from a shared store and write responses/usage back into it. A YAML config binds named agents (roster) to pattern roles; an async runner expands `runs × tasks × repetitions`, scores each result, and persists one JSONL row per run; a metrics module summarizes accuracy + cost and computes Δ vs the single-agent baseline.

**Tech Stack:** Python 3.11, `uv`, PocketFlow (orchestration), Pydantic AI + Claude Agent SDK (agents), pytest + pytest-asyncio (TDD), PyYAML (config). Spec: [`docs/superpowers/specs/2026-05-29-multiagent-collaboration-framework-design.md`](../specs/2026-05-29-multiagent-collaboration-framework-design.md).

---

## File Structure

All paths are relative to the repo root `/data/common/haibotong/multiagent-collab/`.

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, pytest config (asyncio auto mode) |
| `src/macollab/__init__.py` | Package marker |
| `src/macollab/agents/base.py` | `TokenUsage`, `AgentResponse`, `Agent` protocol |
| `src/macollab/agents/fake.py` | `FakeAgent` — no-network test/demo agent |
| `src/macollab/agents/pydantic_ai.py` | `PydanticAIAdapter` |
| `src/macollab/agents/claude.py` | `ClaudeAgentAdapter` + `_extract_claude_response` |
| `src/macollab/agents/registry.py` | `make_agent(agent_id, sdk, model)` factory |
| `src/macollab/flows/nodes.py` | `AgentNode` (generic PocketFlow AsyncNode) |
| `src/macollab/flows/patterns/single.py` | B0 single-agent flow |
| `src/macollab/flows/patterns/draft_refine.py` | P4 Draft→Refine flow |
| `src/macollab/flows/patterns/__init__.py` | `PATTERNS` registry |
| `src/macollab/tasks/base.py` | `Task`, `TaskSuite` protocol |
| `src/macollab/tasks/reasoning_gsm8k.py` | `Gsm8kMiniSuite` |
| `src/macollab/scoring/base.py` | `ScoreResult`, `Scorer` protocol |
| `src/macollab/scoring/numeric.py` | `NumericScorer`, `extract_final_number` |
| `src/macollab/experiment/config.py` | YAML → `ExperimentConfig` |
| `src/macollab/experiment/store.py` | `JsonlStore` (append/load/resume) |
| `src/macollab/experiment/runner.py` | `run_experiment` async orchestration |
| `src/macollab/experiment/metrics.py` | `summarize` + `format_table` |
| `src/macollab/cli.py` | `macollab run <config.yaml>` |
| `configs/demo_fake.yaml` | No-network wiring demo |
| `configs/p4_vs_baselines.yaml` | Real P4-vs-B0 demo (Claude + Pydantic AI) |
| `tests/...` | One test module per source module |

**Conventions:** every commit message uses Conventional Commits and **omits** any `Co-Authored-By` trailer. Run all Python/test commands via `uv run` so the 3.11 venv is used.

---

## Task 0: Project setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/macollab/__init__.py` and empty `__init__.py` in every subpackage
- Create: `tests/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "macollab"
version = "0.1.0"
description = "Research framework for two-agent collaboration patterns"
requires-python = ">=3.11"
dependencies = [
    "pocketflow>=0.0.3",
    "pydantic-ai-slim[anthropic]>=1.0",
    "claude-agent-sdk>=0.2.80",
    "pyyaml>=6.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/macollab"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "smoke: real network calls to a provider (deselect with -m 'not smoke')",
]
```

- [ ] **Step 2: Create the package skeleton**

```bash
cd /data/common/haibotong/multiagent-collab
mkdir -p src/macollab/agents src/macollab/flows/patterns src/macollab/tasks src/macollab/scoring src/macollab/experiment tests
touch src/macollab/__init__.py \
      src/macollab/agents/__init__.py \
      src/macollab/flows/__init__.py \
      src/macollab/flows/patterns/__init__.py \
      src/macollab/tasks/__init__.py \
      src/macollab/scoring/__init__.py \
      src/macollab/experiment/__init__.py \
      tests/__init__.py
```

- [ ] **Step 3: Create the 3.11 venv and install**

Run:
```bash
cd /data/common/haibotong/multiagent-collab
uv venv --python 3.11
uv sync
```
Expected: a `.venv/` is created with Python 3.11 and dependencies installed (no errors). `.venv/` is already gitignored.

- [ ] **Step 4: Verify pytest runs (no tests yet)**

Run: `uv run pytest -q`
Expected: `no tests ran` (exit code 5) — confirms pytest + the env work.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/macollab tests uv.lock
git commit -m "chore: project skeleton, deps, pytest config"
```

---

## Task 1: Agent base types (`TokenUsage`, `AgentResponse`, `Agent` protocol)

**Files:**
- Create: `src/macollab/agents/base.py`
- Test: `tests/test_agents_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agents_base.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agents_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.agents.base'`.

- [ ] **Step 3: Write the implementation**

```python
# src/macollab/agents/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def _add_opt(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    return (a or 0.0) + (b or 0.0)


@dataclass
class TokenUsage:
    input: int = 0
    output: int = 0
    total: int = 0
    cost_usd: float | None = None

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            total=self.total + other.total,
            cost_usd=_add_opt(self.cost_usd, other.cost_usd),
        )


@dataclass
class AgentResponse:
    text: str
    usage: TokenUsage
    structured: Any | None = None
    raw: Any = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    id: str
    sdk: str
    model: str

    async def run(self, task: str, *, system: str | None = None) -> AgentResponse:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agents_base.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/agents/base.py tests/test_agents_base.py
git commit -m "feat: agent base types (TokenUsage, AgentResponse, Agent protocol)"
```

---

## Task 2: FakeAgent (no-network agent for tests & demos)

**Files:**
- Create: `src/macollab/agents/fake.py`
- Test: `tests/test_agents_fake.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agents_fake.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agents_fake.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.agents.fake'`.

- [ ] **Step 3: Write the implementation**

```python
# src/macollab/agents/fake.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from macollab.agents.base import AgentResponse, TokenUsage


@dataclass
class FakeAgent:
    id: str
    model: str = "fake"
    sdk: str = "fake"
    responder: Callable[[str], str] = field(default=lambda task: f"echo: {task}")

    async def run(self, task: str, *, system: str | None = None) -> AgentResponse:
        text = self.responder(task)
        inp = len(task.split())
        out = len(text.split())
        return AgentResponse(
            text=text,
            usage=TokenUsage(input=inp, output=out, total=inp + out),
            raw={"task": task, "system": system},
            meta={"sdk": self.sdk, "model": self.model},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agents_fake.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/agents/fake.py tests/test_agents_fake.py
git commit -m "feat: FakeAgent for no-network tests and demos"
```

---

## Task 3: Scoring (numeric)

**Files:**
- Create: `src/macollab/scoring/base.py`
- Create: `src/macollab/scoring/numeric.py`
- Test: `tests/test_scoring_numeric.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scoring_numeric.py
from macollab.scoring.numeric import NumericScorer, extract_final_number


def test_extract_final_number_takes_last_number():
    assert extract_final_number("First 12, then the answer is 42") == 42.0
    assert extract_final_number("total = 1,234") == 1234.0
    assert extract_final_number("the answer is -3.5") == -3.5
    assert extract_final_number("no digits here") is None


def test_numeric_scorer_correct_and_incorrect():
    s = NumericScorer()
    good = s.score("After working it out, the answer is 42.", 42)
    bad = s.score("I think it's 41.", 42)
    assert good.correct is True and good.score == 1.0
    assert bad.correct is False and bad.score == 0.0


def test_numeric_scorer_no_number_is_incorrect():
    s = NumericScorer()
    assert s.score("I am not sure.", 42).correct is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scoring_numeric.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.scoring.numeric'`.

- [ ] **Step 3: Write the implementations**

```python
# src/macollab/scoring/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ScoreResult:
    correct: bool
    score: float


class Scorer(Protocol):
    def score(self, answer: str, ground_truth: Any) -> ScoreResult:
        ...
```

```python
# src/macollab/scoring/numeric.py
from __future__ import annotations

import re

from macollab.scoring.base import ScoreResult

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_number(text: str) -> float | None:
    cleaned = (text or "").replace(",", "")
    matches = _NUM_RE.findall(cleaned)
    if not matches:
        return None
    return float(matches[-1])


class NumericScorer:
    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = tol

    def score(self, answer: str, ground_truth) -> ScoreResult:
        got = extract_final_number(answer)
        correct = got is not None and abs(got - float(ground_truth)) <= self.tol
        return ScoreResult(correct=correct, score=1.0 if correct else 0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scoring_numeric.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/scoring tests/test_scoring_numeric.py
git commit -m "feat: numeric scorer with final-number extraction"
```

---

## Task 4: Task suite (GSM8K-mini)

**Files:**
- Create: `src/macollab/tasks/base.py`
- Create: `src/macollab/tasks/reasoning_gsm8k.py`
- Test: `tests/test_tasks_gsm8k.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tasks_gsm8k.py
from macollab.scoring.numeric import NumericScorer
from macollab.tasks.reasoning_gsm8k import Gsm8kMiniSuite


def test_suite_has_name_and_tasks():
    suite = Gsm8kMiniSuite()
    assert suite.name == "gsm8k_mini"
    tasks = suite.tasks()
    assert len(tasks) >= 8
    assert all(t.id and t.prompt and t.type == "reasoning" for t in tasks)


def test_ground_truths_are_self_consistent():
    # Sanity: every ground truth is numeric and a perfect answer scores correct.
    scorer = NumericScorer()
    for t in Gsm8kMiniSuite().tasks():
        assert scorer.score(f"The answer is {t.ground_truth}.", t.ground_truth).correct
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tasks_gsm8k.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.tasks.reasoning_gsm8k'`.

- [ ] **Step 3: Write the implementations**

```python
# src/macollab/tasks/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Task:
    id: str
    prompt: str
    ground_truth: Any
    type: str = "reasoning"


class TaskSuite(Protocol):
    name: str

    def tasks(self) -> list[Task]:
        ...
```

```python
# src/macollab/tasks/reasoning_gsm8k.py
from __future__ import annotations

from dataclasses import dataclass

from macollab.tasks.base import Task

_PROBLEMS: list[tuple[str, int]] = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as many "
     "clips in May. How many clips did she sell altogether in April and May?", 72),
    ("Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. "
     "How many dollars did she earn?", 10),
    ("Betty is saving for a $100 wallet. She has half of the money she needs. Her "
     "parents give her $15 and her grandparents twice as much as her parents. How "
     "much more money does Betty need to buy the wallet?", 5),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many "
     "bolts in total does it take?", 3),
    ("James writes a 3-page letter to 2 different friends twice a week. How many "
     "pages does he write a year?", 624),
    ("Mark has a garden with flowers. He planted 10 yellow, 80% more purple than "
     "yellow, and 25% as many green as the combined yellow and purple. How many "
     "flowers does Mark have in his garden?", 35),
    ("A store had 120 apples. They sold 45 in the morning and 30 in the afternoon. "
     "How many apples are left?", 45),
    ("Tom buys 4 books that cost $7 each and a pen that costs $3. How much does he "
     "spend in total?", 31),
    ("A train travels 60 miles per hour for 3 hours, then 40 miles per hour for 2 "
     "hours. How many miles does it travel in total?", 260),
    ("Sara has 5 boxes with 12 pencils each. She gives away 18 pencils. How many "
     "pencils does she have left?", 42),
]


@dataclass
class Gsm8kMiniSuite:
    name: str = "gsm8k_mini"

    def tasks(self) -> list[Task]:
        return [
            Task(id=f"gsm8k_mini_{i:02d}", prompt=p, ground_truth=ans, type="reasoning")
            for i, (p, ans) in enumerate(_PROBLEMS)
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tasks_gsm8k.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/tasks tests/test_tasks_gsm8k.py
git commit -m "feat: GSM8K-mini reasoning task suite"
```

---

## Task 5: AgentNode (generic PocketFlow async node)

**Files:**
- Create: `src/macollab/flows/nodes.py`
- Test: `tests/test_flows_nodes.py`

**Background (from API research):** PocketFlow's `AsyncNode` has the lifecycle `prep_async(shared) -> prep_res`, `exec_async(prep_res) -> exec_res`, `post_async(shared, prep_res, exec_res) -> action_str`. `await node.run_async(shared)` runs a single node and mutates `shared`. Constructor takes `max_retries` (total attempts) and `wait` (seconds between).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flows_nodes.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_flows_nodes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.flows.nodes'`.

- [ ] **Step 3: Write the implementation**

```python
# src/macollab/flows/nodes.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_flows_nodes.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/flows/nodes.py tests/test_flows_nodes.py
git commit -m "feat: generic AgentNode PocketFlow async node"
```

---

## Task 6: Patterns — single (B0) and draft_refine (P4) + registry

**Files:**
- Create: `src/macollab/flows/patterns/single.py`
- Create: `src/macollab/flows/patterns/draft_refine.py`
- Modify: `src/macollab/flows/patterns/__init__.py`
- Test: `tests/test_flows_patterns.py`

**Background:** PocketFlow transitions use `a >> b` (default edge). A flow is `AsyncFlow(start=node)` and is run with `await flow.run_async(shared)` (mutates `shared`; the return value is the terminal action string, not the output).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flows_patterns.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_flows_patterns.py -v`
Expected: FAIL — `ImportError: cannot import name 'PATTERNS'`.

- [ ] **Step 3: Write the implementations**

```python
# src/macollab/flows/patterns/single.py
from __future__ import annotations

from pocketflow import AsyncFlow

from macollab.flows.nodes import AgentNode

SOLO_SYSTEM = "Solve the problem. Think briefly, then state the final numeric answer."


def build_single_flow() -> AsyncFlow:
    solo = AgentNode(
        role="solo",
        prompt_builder=lambda s: s["task"],
        system=SOLO_SYSTEM,
        output_key="solo",
    )
    return AsyncFlow(start=solo)
```

```python
# src/macollab/flows/patterns/draft_refine.py
from __future__ import annotations

from pocketflow import AsyncFlow

from macollab.flows.nodes import AgentNode

DRAFT_SYSTEM = "Solve the problem. Show brief reasoning, then the final numeric answer."
REFINE_SYSTEM = (
    "You improve a draft answer. Check the reasoning, fix any error, and give a "
    "correct, concise final numeric answer."
)
_REFINE_TMPL = (
    "Problem:\n{task}\n\n"
    "A draft answer from another solver:\n{draft}\n\n"
    "Produce the improved, correct final answer."
)


def _refine_prompt(shared: dict) -> str:
    return _REFINE_TMPL.format(task=shared["task"], draft=shared["responses"]["draft"].text)


def build_draft_refine_flow() -> AsyncFlow:
    draft = AgentNode(
        role="drafter",
        prompt_builder=lambda s: s["task"],
        system=DRAFT_SYSTEM,
        output_key="draft",
    )
    refine = AgentNode(
        role="refiner",
        prompt_builder=_refine_prompt,
        system=REFINE_SYSTEM,
        output_key="refined",
    )
    draft >> refine
    return AsyncFlow(start=draft)
```

```python
# src/macollab/flows/patterns/__init__.py
from __future__ import annotations

from pocketflow import AsyncFlow

from macollab.flows.patterns.draft_refine import build_draft_refine_flow
from macollab.flows.patterns.single import build_single_flow

PATTERNS = {
    "single": build_single_flow,
    "draft_refine": build_draft_refine_flow,
}


def build_flow(pattern: str) -> AsyncFlow:
    return PATTERNS[pattern]()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_flows_patterns.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/flows/patterns tests/test_flows_patterns.py
git commit -m "feat: single (B0) and draft_refine (P4) patterns + registry"
```

---

## Task 7: PydanticAIAdapter

**Files:**
- Create: `src/macollab/agents/pydantic_ai.py`
- Test: `tests/test_agents_pydantic.py`

**Background (from API research):** Pydantic AI's `Agent("<provider>:<model>", instructions=...)`; `await agent.run(task)` returns a result with `.output` (text) and `.usage()` → `RunUsage` (`.input_tokens`/`.output_tokens`/`.total_tokens`). We isolate construction in `_make(system)` so tests can inject a fake without touching the network.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agents_pydantic.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agents_pydantic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.agents.pydantic_ai'`.

- [ ] **Step 3: Write the implementation**

```python
# src/macollab/agents/pydantic_ai.py
from __future__ import annotations

from macollab.agents.base import AgentResponse, TokenUsage


class PydanticAIAdapter:
    sdk = "pydantic"

    def __init__(self, id: str, model: str) -> None:
        self.id = id
        self.model = model

    def _make(self, system: str | None):
        # Imported lazily so the rest of the package works without pydantic-ai installed.
        from pydantic_ai import Agent as PydAgent

        if system:
            return PydAgent(self.model, instructions=system)
        return PydAgent(self.model)

    async def run(self, task: str, *, system: str | None = None) -> AgentResponse:
        agent = self._make(system)
        result = await agent.run(task)
        u = result.usage()
        usage = TokenUsage(
            input=getattr(u, "input_tokens", 0) or 0,
            output=getattr(u, "output_tokens", 0) or 0,
            total=getattr(u, "total_tokens", 0) or 0,
        )
        return AgentResponse(
            text=str(result.output),
            usage=usage,
            raw=result,
            meta={"sdk": self.sdk, "model": self.model},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agents_pydantic.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/agents/pydantic_ai.py tests/test_agents_pydantic.py
git commit -m "feat: Pydantic AI agent adapter"
```

---

## Task 8: ClaudeAgentAdapter

**Files:**
- Create: `src/macollab/agents/claude.py`
- Test: `tests/test_agents_claude.py`

**Background (from API research):** `claude_agent_sdk.query(prompt=..., options=ClaudeAgentOptions(...))` is an **async generator**. Iterate it; the final `ResultMessage` carries `.result` (text, `None` if `is_error`), `.usage` (a dict with `input_tokens`/`output_tokens`), and `.total_cost_usd`. Assistant text streams as `AssistantMessage.content` blocks with `.text`. We extract via a pure, duck-typed helper (`_extract_claude_response`) so it is testable without the real SDK, and lazily import `query`/`ClaudeAgentOptions` inside `run`. Use `max_turns=1` and `permission_mode="bypassPermissions"` for single-turn automation.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agents_claude.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agents_claude.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.agents.claude'`.

- [ ] **Step 3: Write the implementation**

```python
# src/macollab/agents/claude.py
from __future__ import annotations

from typing import Any

from macollab.agents.base import AgentResponse, TokenUsage


def _extract_claude_response(messages: list[Any], *, model: str) -> AgentResponse:
    """Pure, duck-typed extraction of text + usage from a list of SDK messages.

    A ResultMessage is identified by having both ``result`` and ``total_cost_usd``;
    its ``usage`` is a dict (Anthropic Messages API shape). Otherwise we accumulate
    text from AssistantMessage-like objects whose ``content`` is a list of blocks
    with a ``text`` attribute.
    """
    result_text: str | None = None
    usage_dict: dict | None = None
    cost: float | None = None
    text_parts: list[str] = []

    for m in messages:
        if hasattr(m, "result") and hasattr(m, "total_cost_usd"):
            result_text = getattr(m, "result", None)
            usage_dict = getattr(m, "usage", None)
            cost = getattr(m, "total_cost_usd", None)
        elif isinstance(getattr(m, "content", None), list):
            for block in m.content:
                t = getattr(block, "text", None)
                if t:
                    text_parts.append(t)

    text = result_text if result_text else "".join(text_parts)
    u = usage_dict or {}
    inp = int(u.get("input_tokens", 0) or 0)
    out = int(u.get("output_tokens", 0) or 0)
    return AgentResponse(
        text=text or "",
        usage=TokenUsage(input=inp, output=out, total=inp + out, cost_usd=cost),
        raw=messages,
        meta={"sdk": "claude", "model": model},
    )


# Indirection points so tests can monkeypatch without importing the real SDK.
def _query(*, prompt: str, options):
    from claude_agent_sdk import query

    return query(prompt=prompt, options=options)


def _Options(**kwargs):
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(**kwargs)


class ClaudeAgentAdapter:
    sdk = "claude"

    def __init__(self, id: str, model: str = "claude-sonnet-4-6", *, max_turns: int = 1) -> None:
        self.id = id
        self.model = model
        self.max_turns = max_turns

    async def run(self, task: str, *, system: str | None = None) -> AgentResponse:
        options = _Options(
            model=self.model,
            max_turns=self.max_turns,
            system_prompt=system,
            permission_mode="bypassPermissions",
        )
        messages: list[Any] = []
        async for message in _query(prompt=task, options=options):
            messages.append(message)
        return _extract_claude_response(messages, model=self.model)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agents_claude.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/agents/claude.py tests/test_agents_claude.py
git commit -m "feat: Claude Agent SDK adapter with duck-typed extraction"
```

---

## Task 9: Agent registry / factory

**Files:**
- Create: `src/macollab/agents/registry.py`
- Test: `tests/test_agents_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agents_registry.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agents_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.agents.registry'`.

- [ ] **Step 3: Write the implementation**

```python
# src/macollab/agents/registry.py
from __future__ import annotations

from macollab.agents.base import Agent
from macollab.agents.claude import ClaudeAgentAdapter
from macollab.agents.fake import FakeAgent
from macollab.agents.pydantic_ai import PydanticAIAdapter


def make_agent(agent_id: str, sdk: str, model: str) -> Agent:
    if sdk == "fake":
        return FakeAgent(id=agent_id, model=model or "fake")
    if sdk == "pydantic":
        return PydanticAIAdapter(id=agent_id, model=model)
    if sdk == "claude":
        return ClaudeAgentAdapter(id=agent_id, model=model or "claude-sonnet-4-6")
    raise ValueError(f"unknown sdk: {sdk!r} (supported: fake, pydantic, claude)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agents_registry.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/agents/registry.py tests/test_agents_registry.py
git commit -m "feat: agent registry/factory (fake/pydantic/claude)"
```

---

## Task 10: Experiment config (YAML loader)

**Files:**
- Create: `src/macollab/experiment/config.py`
- Test: `tests/test_experiment_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiment_config.py
from pathlib import Path

from macollab.experiment.config import load_config

_YAML = """
experiment: demo
task_suite: gsm8k_mini
repetitions: 2
roster:
  A: {sdk: fake, model: drafter-bot}
  B: {sdk: fake, model: refiner-bot}
runs:
  - {pattern: single, bind: {solo: A}}
  - {pattern: draft_refine, bind: {drafter: A, refiner: B}, params: {note: x}}
"""


def test_load_config_parses_roster_and_runs(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(_YAML)
    cfg = load_config(p)

    assert cfg.experiment == "demo"
    assert cfg.task_suite == "gsm8k_mini"
    assert cfg.repetitions == 2
    assert cfg.roster["A"].sdk == "fake" and cfg.roster["A"].model == "drafter-bot"
    assert len(cfg.runs) == 2
    assert cfg.runs[0].pattern == "single"
    assert cfg.runs[1].bind == {"drafter": "A", "refiner": "B"}
    assert cfg.runs[1].params == {"note": "x"}


def test_repetitions_defaults_to_one(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "experiment: d\ntask_suite: gsm8k_mini\n"
        "roster: {A: {sdk: fake, model: m}}\n"
        "runs: [{pattern: single, bind: {solo: A}}]\n"
    )
    assert load_config(p).repetitions == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_experiment_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.experiment.config'`.

- [ ] **Step 3: Write the implementation**

```python
# src/macollab/experiment/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AgentSpec:
    sdk: str
    model: str


@dataclass
class RunSpec:
    pattern: str
    bind: dict[str, str]
    params: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    experiment: str
    task_suite: str
    roster: dict[str, AgentSpec]
    runs: list[RunSpec]
    repetitions: int = 1


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text())
    roster = {name: AgentSpec(**spec) for name, spec in data["roster"].items()}
    runs = [
        RunSpec(pattern=r["pattern"], bind=r["bind"], params=r.get("params", {}))
        for r in data["runs"]
    ]
    return ExperimentConfig(
        experiment=data["experiment"],
        task_suite=data["task_suite"],
        roster=roster,
        runs=runs,
        repetitions=int(data.get("repetitions", 1)),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_experiment_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/experiment/config.py tests/test_experiment_config.py
git commit -m "feat: experiment YAML config loader"
```

---

## Task 11: JSONL results store (with resume)

**Files:**
- Create: `src/macollab/experiment/store.py`
- Test: `tests/test_experiment_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiment_store.py
from pathlib import Path

from macollab.experiment.store import JsonlStore, run_key


def test_append_and_load_roundtrip(tmp_path: Path):
    store = JsonlStore(tmp_path / "runs.jsonl")
    store.append({"pattern": "single", "bind_key": "solo=A", "task_id": "t1", "rep": 0, "score": 1.0})
    store.append({"pattern": "single", "bind_key": "solo=A", "task_id": "t2", "rep": 0, "score": 0.0})
    rows = store.load_all()
    assert len(rows) == 2
    assert rows[0]["task_id"] == "t1"


def test_existing_keys_enables_resume(tmp_path: Path):
    store = JsonlStore(tmp_path / "runs.jsonl")
    store.append({"pattern": "single", "bind_key": "solo=A", "task_id": "t1", "rep": 0})
    keys = store.existing_keys()
    assert run_key("single", "solo=A", "t1", 0) in keys
    assert run_key("single", "solo=A", "t1", 1) not in keys


def test_load_all_missing_file_is_empty(tmp_path: Path):
    assert JsonlStore(tmp_path / "none.jsonl").load_all() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_experiment_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.experiment.store'`.

- [ ] **Step 3: Write the implementation**

```python
# src/macollab/experiment/store.py
from __future__ import annotations

import json
from pathlib import Path


def run_key(pattern: str, bind_key: str, task_id: str, rep: int) -> tuple:
    return (pattern, bind_key, task_id, int(rep))


class JsonlStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def existing_keys(self) -> set[tuple]:
        return {
            run_key(r["pattern"], r["bind_key"], r["task_id"], r["rep"])
            for r in self.load_all()
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_experiment_store.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/experiment/store.py tests/test_experiment_store.py
git commit -m "feat: JSONL results store with resume keys"
```

---

## Task 12: Experiment runner (async orchestration)

**Files:**
- Create: `src/macollab/experiment/runner.py`
- Test: `tests/test_experiment_runner.py`

**Design:** `run_experiment(cfg, store, *, concurrency=4)` expands `runs × tasks × repetitions`, builds the role→agent bindings via `make_agent`, runs each flow (`build_flow`), scores `shared["final"]`, sums `usage_log`, and appends one record per run. Jobs already in `store.existing_keys()` are skipped (resume). Exceptions per job are caught and recorded with an `error` field so one failure doesn't abort the batch.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiment_runner.py
import asyncio
from pathlib import Path

from macollab.experiment.config import load_config
from macollab.experiment.runner import run_experiment
from macollab.experiment.store import JsonlStore

_YAML = """
experiment: demo_fake
task_suite: gsm8k_mini
repetitions: 1
roster:
  A: {sdk: fake, model: drafter}
  B: {sdk: fake, model: refiner}
runs:
  - {pattern: single, bind: {solo: A}}
  - {pattern: draft_refine, bind: {drafter: A, refiner: B}}
"""


def test_runner_writes_one_record_per_run_and_is_resumable(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(_YAML)
    cfg = load_config(p)
    store = JsonlStore(tmp_path / "runs.jsonl")

    asyncio.run(run_experiment(cfg, store, concurrency=4))
    rows = store.load_all()

    # 2 patterns x 10 tasks x 1 rep = 20 records
    assert len(rows) == 20
    patterns = {r["pattern"] for r in rows}
    assert patterns == {"single", "draft_refine"}
    sample = rows[0]
    for key in ("experiment", "pattern", "bind_key", "task_id", "rep",
                "final_answer", "correct", "score", "total_tokens", "latency_s"):
        assert key in sample

    # Resume: running again adds nothing (all keys already present).
    asyncio.run(run_experiment(cfg, store, concurrency=4))
    assert len(store.load_all()) == 20


def test_runner_records_error_without_aborting(tmp_path: Path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text(_YAML)
    cfg = load_config(p)
    store = JsonlStore(tmp_path / "runs.jsonl")

    # Force the draft_refine flow builder to raise; single must still complete.
    import macollab.experiment.runner as runner_mod

    real_build = runner_mod.build_flow

    def flaky_build(pattern):
        if pattern == "draft_refine":
            raise RuntimeError("boom")
        return real_build(pattern)

    monkeypatch.setattr(runner_mod, "build_flow", flaky_build)
    asyncio.run(run_experiment(cfg, store, concurrency=2))

    rows = store.load_all()
    singles = [r for r in rows if r["pattern"] == "single"]
    errored = [r for r in rows if r.get("error")]
    assert len(singles) == 10 and all(r["error"] is None for r in singles)
    assert len(errored) == 10 and all(r["pattern"] == "draft_refine" for r in errored)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_experiment_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.experiment.runner'`.

- [ ] **Step 3: Write the implementation**

```python
# src/macollab/experiment/runner.py
from __future__ import annotations

import asyncio
import time

from macollab.agents.base import TokenUsage
from macollab.agents.registry import make_agent
from macollab.experiment.config import ExperimentConfig
from macollab.experiment.store import JsonlStore, run_key
from macollab.flows.patterns import build_flow
from macollab.scoring.numeric import NumericScorer
from macollab.tasks.reasoning_gsm8k import Gsm8kMiniSuite

SUITES = {"gsm8k_mini": Gsm8kMiniSuite}


def get_suite(name: str):
    return SUITES[name]()


def get_scorer(task_type: str):
    if task_type == "reasoning":
        return NumericScorer()
    raise ValueError(f"no scorer for task type {task_type!r}")


def bind_key_of(bind: dict[str, str]) -> str:
    return ",".join(f"{role}={name}" for role, name in sorted(bind.items()))


async def run_experiment(
    cfg: ExperimentConfig,
    store: JsonlStore,
    *,
    concurrency: int = 4,
) -> None:
    suite = get_suite(cfg.task_suite)
    tasks = suite.tasks()
    done = store.existing_keys()
    sem = asyncio.Semaphore(concurrency)

    async def run_one(run, task, rep) -> None:
        bk = bind_key_of(run.bind)
        if run_key(run.pattern, bk, task.id, rep) in done:
            return
        scorer = get_scorer(task.type)
        record = {
            "experiment": cfg.experiment,
            "pattern": run.pattern,
            "bind_key": bk,
            "bind": run.bind,
            "task_id": task.id,
            "rep": rep,
            "final_answer": None,
            "correct": False,
            "score": 0.0,
            "n_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": None,
            "latency_s": 0.0,
            "error": None,
        }
        async with sem:
            t0 = time.monotonic()
            try:
                roles = {
                    role: make_agent(name, cfg.roster[name].sdk, cfg.roster[name].model)
                    for role, name in run.bind.items()
                }
                shared = {
                    "task": task.prompt,
                    "ground_truth": task.ground_truth,
                    "roles": roles,
                    "responses": {},
                    "usage_log": [],
                    "final": None,
                }
                flow = build_flow(run.pattern)
                await flow.run_async(shared)
                final = shared.get("final") or ""
                sc = scorer.score(final, task.ground_truth)
                total = sum(shared["usage_log"], TokenUsage())
                record.update(
                    final_answer=final,
                    correct=sc.correct,
                    score=sc.score,
                    n_calls=len(shared["usage_log"]),
                    input_tokens=total.input,
                    output_tokens=total.output,
                    total_tokens=total.total,
                    cost_usd=total.cost_usd,
                )
            except Exception as exc:  # noqa: BLE001 - record and continue
                record["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                record["latency_s"] = round(time.monotonic() - t0, 4)
                store.append(record)

    jobs = [
        run_one(run, task, rep)
        for run in cfg.runs
        for task in tasks
        for rep in range(cfg.repetitions)
    ]
    await asyncio.gather(*jobs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_experiment_runner.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/experiment/runner.py tests/test_experiment_runner.py
git commit -m "feat: async experiment runner with resume and per-job error capture"
```

---

## Task 13: Metrics summary

**Files:**
- Create: `src/macollab/experiment/metrics.py`
- Test: `tests/test_experiment_metrics.py`

**Design:** `summarize(records)` groups by `(pattern, bind_key)`, computing `n`, `accuracy` (mean of `correct` over non-errored rows), `errors`, `mean_total_tokens`, `mean_latency_s`, and `delta_vs_single` (this group's accuracy minus the `single` pattern group's accuracy; `None` if there is no single group). `format_table(rows)` renders a fixed-width table string.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiment_metrics.py
from macollab.experiment.metrics import format_table, summarize


def _rec(pattern, bind_key, correct, tokens, error=None):
    return {
        "pattern": pattern, "bind_key": bind_key, "correct": correct,
        "total_tokens": tokens, "latency_s": 0.1, "error": error,
    }


def test_summarize_accuracy_and_delta_vs_single():
    records = [
        _rec("single", "solo=A", True, 100),
        _rec("single", "solo=A", False, 100),
        _rec("draft_refine", "drafter=A,refiner=B", True, 250),
        _rec("draft_refine", "drafter=A,refiner=B", True, 250),
    ]
    rows = {(r["pattern"], r["bind_key"]): r for r in summarize(records)}

    single = rows[("single", "solo=A")]
    p4 = rows[("draft_refine", "drafter=A,refiner=B")]
    assert single["n"] == 2 and single["accuracy"] == 0.5
    assert p4["accuracy"] == 1.0
    assert p4["delta_vs_single"] == 0.5
    assert p4["mean_total_tokens"] == 250


def test_summarize_excludes_errored_rows_from_accuracy():
    records = [
        _rec("single", "solo=A", True, 100),
        _rec("single", "solo=A", False, 100, error="Boom: x"),
    ]
    row = summarize(records)[0]
    assert row["n"] == 1          # errored row excluded from n/accuracy
    assert row["errors"] == 1
    assert row["accuracy"] == 1.0


def test_format_table_includes_headers_and_patterns():
    records = [_rec("single", "solo=A", True, 100)]
    table = format_table(summarize(records))
    assert "pattern" in table and "accuracy" in table and "single" in table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_experiment_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.experiment.metrics'`.

- [ ] **Step 3: Write the implementation**

```python
# src/macollab/experiment/metrics.py
from __future__ import annotations

from collections import defaultdict


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["pattern"], r["bind_key"])].append(r)

    rows: list[dict] = []
    for (pattern, bind_key), items in groups.items():
        ok = [r for r in items if not r.get("error")]
        errors = len(items) - len(ok)
        n = len(ok)
        accuracy = (sum(1 for r in ok if r["correct"]) / n) if n else 0.0
        mean_tokens = (sum(r["total_tokens"] for r in ok) / n) if n else 0.0
        mean_latency = (sum(r["latency_s"] for r in ok) / n) if n else 0.0
        rows.append({
            "pattern": pattern,
            "bind_key": bind_key,
            "n": n,
            "errors": errors,
            "accuracy": round(accuracy, 4),
            "mean_total_tokens": round(mean_tokens, 1),
            "mean_latency_s": round(mean_latency, 4),
            "delta_vs_single": None,
        })

    single_rows = [r for r in rows if r["pattern"] == "single"]
    if single_rows:
        baseline = single_rows[0]["accuracy"]
        for r in rows:
            r["delta_vs_single"] = round(r["accuracy"] - baseline, 4)

    rows.sort(key=lambda r: (r["pattern"] != "single", r["pattern"], r["bind_key"]))
    return rows


_COLS = [
    ("pattern", 16),
    ("bind_key", 28),
    ("n", 4),
    ("errors", 7),
    ("accuracy", 9),
    ("delta_vs_single", 16),
    ("mean_total_tokens", 18),
    ("mean_latency_s", 15),
]


def format_table(rows: list[dict]) -> str:
    header = "  ".join(name.ljust(width) for name, width in _COLS)
    lines = [header, "  ".join("-" * width for _, width in _COLS)]
    for r in rows:
        lines.append("  ".join(str(r.get(name, "")).ljust(width) for name, width in _COLS))
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_experiment_metrics.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/macollab/experiment/metrics.py tests/test_experiment_metrics.py
git commit -m "feat: metrics summary (accuracy, delta vs single, cost) + table"
```

---

## Task 14: CLI + configs + no-network demo run

**Files:**
- Create: `src/macollab/cli.py`
- Create: `configs/demo_fake.yaml`
- Create: `configs/p4_vs_baselines.yaml`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from pathlib import Path

from macollab.cli import main

_YAML = """
experiment: demo_fake
task_suite: gsm8k_mini
repetitions: 1
roster:
  A: {sdk: fake, model: drafter}
  B: {sdk: fake, model: refiner}
runs:
  - {pattern: single, bind: {solo: A}}
  - {pattern: draft_refine, bind: {drafter: A, refiner: B}}
"""


def test_cli_run_writes_results_and_prints_table(tmp_path: Path, capsys):
    cfg = tmp_path / "demo.yaml"
    cfg.write_text(_YAML)
    out = tmp_path / "runs.jsonl"

    rc = main(["run", str(cfg), "--out", str(out), "--concurrency", "4"])

    assert rc == 0
    assert out.exists()
    captured = capsys.readouterr().out
    assert "pattern" in captured and "accuracy" in captured
    assert "single" in captured and "draft_refine" in captured
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'macollab.cli'`.

- [ ] **Step 3: Write the implementation and configs**

```python
# src/macollab/cli.py
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from macollab.experiment.config import load_config
from macollab.experiment.metrics import format_table, summarize
from macollab.experiment.runner import run_experiment
from macollab.experiment.store import JsonlStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="macollab")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run an experiment config")
    run_p.add_argument("config", help="path to the experiment YAML")
    run_p.add_argument("--out", default=None, help="JSONL output path")
    run_p.add_argument("--concurrency", type=int, default=4)

    args = parser.parse_args(argv)

    if args.command == "run":
        cfg = load_config(args.config)
        out = Path(args.out) if args.out else Path("results") / cfg.experiment / "runs.jsonl"
        store = JsonlStore(out)
        asyncio.run(run_experiment(cfg, store, concurrency=args.concurrency))
        rows = summarize(store.load_all())
        print(f"\nExperiment: {cfg.experiment}  ->  {out}\n")
        print(format_table(rows))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

```yaml
# configs/demo_fake.yaml — no-network wiring demo (FakeAgent; proves the loop, not accuracy)
experiment: demo_fake
task_suite: gsm8k_mini
repetitions: 1
roster:
  A: {sdk: fake, model: drafter-bot}
  B: {sdk: fake, model: refiner-bot}
runs:
  - {pattern: single, bind: {solo: A}}
  - {pattern: draft_refine, bind: {drafter: A, refiner: B}}
```

```yaml
# configs/p4_vs_baselines.yaml — real demo (needs ANTHROPIC_API_KEY; Claude SDK needs Node)
experiment: p4_vs_baselines
task_suite: gsm8k_mini
repetitions: 3
roster:
  claude_sonnet: {sdk: claude,   model: claude-sonnet-4-6}
  pyd_sonnet:    {sdk: pydantic, model: anthropic:claude-sonnet-4-6}
runs:
  - {pattern: single, bind: {solo: claude_sonnet}}              # B0 baseline
  - {pattern: draft_refine, bind: {drafter: claude_sonnet, refiner: pyd_sonnet}}  # P4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full no-network demo end-to-end**

Run: `uv run macollab run configs/demo_fake.yaml --out results/demo_fake/runs.jsonl`
Expected: a metrics table prints with rows for `single` and `draft_refine` (accuracy likely `0.0` — FakeAgent echoes, it does not solve math; this run proves the *pipeline*, not capability). `results/` is gitignored.

- [ ] **Step 6: Run the whole test suite (no network)**

Run: `uv run pytest -m "not smoke" -q`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/macollab/cli.py configs/demo_fake.yaml configs/p4_vs_baselines.yaml tests/test_cli.py
git commit -m "feat: CLI runner + demo configs + no-network demo"
```

---

## Task 15: Real-provider smoke tests (marked, optional)

**Files:**
- Create: `tests/test_smoke_pydantic.py`
- Create: `tests/test_smoke_claude.py`

**Note:** these hit real providers and cost money; they are marked `smoke` and skip automatically when the relevant env var/runtime is missing. They are excluded from the default run via `-m "not smoke"`.

- [ ] **Step 1: Write the smoke tests**

```python
# tests/test_smoke_pydantic.py
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
```

```python
# tests/test_smoke_claude.py
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
```

- [ ] **Step 2: Verify they are skipped by default**

Run: `uv run pytest -m "not smoke" -q`
Expected: smoke tests are deselected; everything else passes.

- [ ] **Step 3: (Optional, costs money) Run the real demo if credentials are present**

Run (only with `ANTHROPIC_API_KEY` set and Node available):
```bash
uv run pytest -m smoke -v
uv run macollab run configs/p4_vs_baselines.yaml
```
Expected: smoke tests pass; the metrics table shows `single` (B0) vs `draft_refine` (P4) accuracy on GSM8K-mini with a `delta_vs_single` column — the v1 "prove the loop" result.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke_pydantic.py tests/test_smoke_claude.py
git commit -m "test: marked real-provider smoke tests for pydantic + claude"
```

- [ ] **Step 5: Push to GitHub**

```bash
git push
```
Expected: pushes to `origin/main` (the repo's `core.sshCommand` is pinned to the correct deploy key).

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Uniform `Agent` adapter (spec §4.1) → Tasks 1, 7, 8, 9 (+ Fake in 2).
- PocketFlow layer / shared store / `AgentNode` (spec §4.2) → Task 5.
- Patterns: P4 + B0 baseline (spec §5, §9) → Task 6. *(P1/P2/P5/P6/P7/P8/P9 and baselines B1/B2 are explicitly post-v1 per spec §9 — not in this plan.)*
- Tasks & scoring (spec §7.1, §7.2) → Tasks 3, 4. *(Agentic suite is post-v1; v1 is reasoning per §9.)*
- Experiment matrix / config / storage (spec §6) → Tasks 10, 11, 12. *(SQLite index is post-v1; v1 uses JSONL per §9 step 5.)*
- Metrics: accuracy, Δ vs baseline, tokens, latency (spec §7.3) → Task 13. *(Compute-matched Δ vs B1/B2, Pareto, agreement diagnostics, paired stats follow once B1/B2 land — §9.)*
- v1 milestone steps 1–6 (spec §9) → Tasks 1–14; real demo in Task 15 step 3.
- Risks: Python 3.11 venv, `max_turns=1` for Claude, marked smoke tests (spec §10) → Tasks 0, 8, 15.

**Placeholder scan:** No `TBD`/`TODO`/"handle edge cases"/"similar to" — every code and command step is complete and concrete.

**Type consistency (checked across tasks):** `TokenUsage(input/output/total/cost_usd)` and `__add__` (Task 1) used by Fake (2), adapters (7/8), runner (12). `AgentResponse(text/usage/structured/raw/meta)` consistent everywhere. `Agent.run(task, *, system)` matches Fake (2), Pydantic (7), Claude (8), and `AgentNode.exec_async` (5). `make_agent(agent_id, sdk, model)` (9) called by runner (12) with `cfg.roster[name].sdk/.model` (10). `build_flow(pattern)` (6) imported and called by runner (12) and monkeypatched in its test. Record fields written by runner (12) — `pattern`, `bind_key`, `task_id`, `rep`, `correct`, `total_tokens`, `latency_s`, `error` — match `run_key` (11), `summarize`/`format_table` consumption (13), and the resume check (12). `shared` keys (`task`/`ground_truth`/`roles`/`responses`/`usage_log`/`final`) consistent across nodes (5), patterns (6), and runner (12).

---

## Out of scope for v1 (tracked for the next plan)
Other patterns (P1/P2/P5/P6/P7/P8/P9) and compute-matched baselines (B1 self-consistency, B2 self-refine); OpenAI Agents SDK + Google ADK adapters; the agentic task suite + its checker; SQLite index; Pareto/agreement/paired-stat metrics; a global token/$ budget guard.
