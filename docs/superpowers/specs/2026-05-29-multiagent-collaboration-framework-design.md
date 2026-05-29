# Multi-Agent Collaboration Framework — Design Spec

**Date:** 2026-05-29
**Status:** Approved design → ready for implementation plan
**Repo:** `haibotong-pixel/multiagent-collab`

---

## 1. Motivation & research question

We are building a **research framework** to study how **two agents collaborating** — and *the manner in which they collaborate* — affects task capability, relative to a single agent.

The two primary independent variables:

1. **Agent identity** — the model and the **agent SDK** behind each of the two agents. The two agents may use the same or different models, and the same or different SDKs. Supported SDKs: **Claude Agent SDK**, **OpenAI Agents SDK**, **Google ADK**, **Pydantic AI**.
2. **Collaboration pattern** — *how* the two agents combine on one problem (aggregate, refine, debate, role-split, …).

The dependent variables: task accuracy and its lift over baselines, plus cost (tokens, $, latency, #LLM-calls) and diagnostics (inter-agent agreement/diversity).

**Central claim to be made defensible:** a 2-agent collaboration pattern "improves capability" only if it beats a **compute-matched single-agent baseline** (self-consistency / self-refine), not merely a single one-shot agent. The framework therefore treats those controls as first-class.

### Research questions
- **RQ1** — Does 2-agent collaboration improve accuracy over a single agent (B0)? Over compute-matched single-agent baselines (B1/B2)?
- **RQ2** — Which collaboration *pattern* helps most, and on which task type (reasoning vs agentic)?
- **RQ3** — Does heterogeneity help? (Different models / different SDKs in the two roles vs homogeneous pairs.)
- **RQ4** — **SDK effect vs model effect**: holding the underlying model fixed, does the *SDK/scaffolding* alone change capability?
- **RQ5** — What is the cost/accuracy trade-off (Pareto) of each pattern?

---

## 2. Scope

**In scope (project):** the four SDKs above as uniform adapters; a PocketFlow-based orchestration layer; the full pattern taxonomy (§5); a config-driven experiment matrix; reasoning + agentic task suites; scoring + metrics + storage.

**In scope (v1 milestone, §9):** scaffold + **one pattern (P4 Draft→Refine) end-to-end** + the **B0 single-agent baseline**, on **two SDKs first** (Pydantic AI + Claude Agent SDK), against a tiny reasoning suite, producing a metrics table. Everything else is wiring on top of the proven loop.

**Out of scope (for now):** >2 agents; human-in-the-loop; UI/dashboard; distributed/remote execution; training/fine-tuning.

---

## 3. Architecture

**Approach A — Adapter + PocketFlow flows + config-driven experiment matrix** (chosen over a blackboard-centric design B and per-pattern scripts C).

Key property: **all four SDKs ship Python packages**, so each becomes a thin **in-process async adapter** behind one uniform `Agent` interface. No service bus, no subprocess bridges *we* manage (the Claude Agent SDK manages its own Claude Code CLI subprocess internally). PocketFlow orchestrates **adapters**, never raw SDKs.

```
            ┌─────────────────────────────────────────────────────┐
            │  Experiment Runner  (config.yaml → run matrix)        │
            │  expands  runs × tasks × repetitions                  │
            └───────────────┬───────────────────────┬──────────────┘
                            │                        │
                   ┌────────▼────────┐      ┌────────▼────────┐
                   │  Pattern Flow   │      │   Scorer        │
                   │  (PocketFlow    │      │  (per suite)    │
                   │   AsyncFlow)    │      └─────────────────┘
                   └───┬─────────┬───┘
              role:drafter   role:refiner          → results store (JSONL + SQLite)
                   │             │
            ┌──────▼──────┐ ┌────▼────────┐
            │  Agent      │ │  Agent      │   ← uniform Agent adapters
            │  adapter    │ │  adapter    │
            └──────┬──────┘ └────┬────────┘
          claude / openai / google-adk / pydantic-ai
```

The framework is **async-native** because the Claude Agent SDK is async-only and the others all expose async entrypoints; PocketFlow provides `AsyncNode`/`AsyncFlow` and `AsyncParallelBatchNode` (parallel fan-out via `asyncio.gather`) which the fan-out patterns (P1/P2/P7) use directly.

---

## 4. Component design

### 4.1 Uniform `Agent` adapter

```python
@dataclass
class TokenUsage:
    input: int
    output: int
    total: int
    cost_usd: float | None = None      # only the Claude Agent SDK reports cost directly

@dataclass
class AgentResponse:
    text: str
    usage: TokenUsage
    structured: Any | None = None      # populated when a structured output_type is used
    raw: Any = None                    # the SDK's native result object, for the trace
    meta: dict = field(default_factory=dict)   # sdk, model_id, latency_s, n_calls

class Agent(Protocol):
    id: str
    sdk: str
    model: str
    async def run(
        self,
        task: str,
        *,
        system: str | None = None,
        history: list[dict] | None = None,
        tools: Any | None = None,
    ) -> AgentResponse: ...
```

Four adapters implement `Agent`, each normalizing the per-SDK quirks confirmed in the API research (Appendix A):

| Adapter | text from | usage normalization | model string example |
|---|---|---|---|
| `ClaudeAgentAdapter` | `ResultMessage.result` (fallback: concat `TextBlock`s) | `.usage` dict (`input_tokens`/`output_tokens`) + `.total_cost_usd`; guard `is_error` | `claude-opus-4-8` |
| `OpenAIAgentsAdapter` | `result.final_output` (`str(...)`) | `result.context_wrapper.usage` (**not** `result.usage`) | `gpt-4.1` / LiteLLM |
| `GoogleADKAdapter` | final-event `content.parts` text | **sum** `usage_metadata` over all events (`prompt_token_count`/`candidates_token_count`) | `gemini-2.5-flash` / `LiteLlm(...)` |
| `PydanticAIAdapter` | `result.output` | `result.usage()` → `RunUsage` (`input_tokens`/`output_tokens`/`total_tokens`) | `anthropic:claude-sonnet-4-6` |

**Registry + factory.** A config string resolves to an adapter instance:
- `"claude:claude-opus-4-8"`
- `"pydantic:anthropic:claude-sonnet-4-6"`
- `"openai:gpt-4.1"` or `"openai:litellm:anthropic/claude-sonnet-4-6"`
- `"google:gemini-2.5-flash"` or `"google:litellm:openai/gpt-4o"`

Because Pydantic AI (natively) and OpenAI/ADK (via LiteLLM) can wrap *arbitrary* models, the **same underlying model can be registered under different SDKs** — this is the mechanism enabling RQ4 (SDK-effect vs model-effect).

**Adapter contract notes (from research):**
- Adapter `run()` is `async`. A thin `run_sync()` convenience wrapper exists for tests/CLI but is never called inside flows.
- Claude/OpenAI/ADK are *agent runners* (loops with optional tools). For pure-reasoning tasks they are constrained to a single turn (`max_turns=1`, no tools) so they behave like one LLM call; for agentic tasks the loop + tools are enabled.
- Claude Agent SDK requires Node.js + the bundled Claude Code CLI at runtime, and `permission_mode="bypassPermissions"` for non-interactive automation.

### 4.2 PocketFlow orchestration layer

PocketFlow is pure orchestration: **no LLM client, no token tracking, no result object.** `flow.run(shared)` returns the terminal node's *action string* (often `None`); **all outputs travel through the mutated `shared` dict.** The design embraces this.

**Typed shared store (the blackboard):**
```python
shared = {
    "task": str,
    "ground_truth": Any,
    "roles": {"drafter": Agent, "refiner": Agent, "judge": Agent, ...},  # bound per experiment
    "messages": [],          # used by multi-round patterns (P6 debate / P7 MoA)
    "drafts": {},            # role → AgentResponse
    "final": None,           # final answer text
    "usage_log": [],         # every AgentResponse.usage; summed at the end
    "trace": [],             # structured event log for persistence
}
```

**`AgentNode`** — a generic `AsyncNode` parameterized by `role` (a key into `shared["roles"]`) and a prompt template. Its `exec_async` calls `await agent.run(...)`; its `post_async` writes the response into `shared` and appends usage. PocketFlow's built-in `max_retries`/`wait` gives per-node resilience for free.

Patterns are different **wirings** of `AgentNode`s (plus small aggregator/judge nodes). Adding new patterns is wiring, not new infrastructure.

---

## 5. Collaboration-pattern taxonomy

Each pattern is a small `AsyncFlow`. Roles are bound to concrete agents per experiment. `J` = judge/aggregator role (configurable as agent A, agent B, or a neutral third agent).

### Baselines / controls (no real collaboration)
- **B0 — Single agent.** Each SDK/model alone. Reference point for "improvement."
- **B1 — Self-consistency.** *One* model sampled N times + vote. Controls for "is it just more compute?"
- **B2 — Self-refine.** *One* model drafts then critiques/revises itself. Controls for "is it just a second pass?"

> A 2-agent pattern only demonstrates a *collaboration* gain if it beats B1/B2 at matched compute.

*(Pattern numbering note: the originally-sketched "P3 self-consistency" was promoted to baseline **B1**, so the pattern list runs P1, P2, P4–P9 — the gap at P3 is intentional.)*

### Family 1 — Parallel then aggregate (fan-out → fan-in)
- **P1 — Selection / Vote:** A and B answer independently; `J` *picks* the better. (User example 3.)
- **P2 — Synthesis:** A and B answer independently; `J` *merges* both into a new final answer. (User example 1, with `J = A`.)

### Family 2 — Sequential refinement (serial)
- **P4 — Draft→Refine:** A drafts; B improves it. (User example 2.) **← v1 pattern.**
- **P5 — Critique→Revise:** A drafts; B critiques *only*; A revises.

### Family 3 — Interactive / multi-round
- **P6 — Debate:** A and B argue for `k` rounds (each sees the peer's prior turn via `shared["messages"]`); `J` concludes.
- **P7 — Mixture-of-Agents (one layer):** A and B both answer, then both rewrite seeing the peer's answer; `J` aggregates.

### Family 4 — Role specialization (asymmetric; strongest for agentic tasks)
- **P8 — Planner–Executor:** A decomposes/plans; B executes steps (with tools).
- **P9 — Solver–Verifier:** A solves; B verifies and repairs.

### Cross-cutting variables (sweepable on any pattern)
who is judge (A / B / neutral) · number of rounds · whether a node sees the *task* or only the *peer's answer* · symmetric vs asymmetric roles · same vs different model/SDK per role.

---

## 6. Experiment matrix, config & storage

One YAML config drives a whole sweep:

```yaml
experiment: p4_vs_baselines
task_suite: gsm8k_mini          # or an agentic suite id
repetitions: 3                  # for mean ± std + paired stats
roster:                         # named agents → (sdk, model)
  A_claude:   {sdk: claude,  model: claude-sonnet-4-6}
  B_gpt:      {sdk: openai,  model: gpt-4.1}
runs:
  - {pattern: single,           bind: {solo: A_claude}}            # B0
  - {pattern: self_consistency, bind: {solo: A_claude}, params: {n: 5}}   # B1
  - {pattern: self_refine,      bind: {solo: A_claude}}            # B2
  - {pattern: draft_refine,     bind: {drafter: A_claude, refiner: B_gpt}} # P4
```

**Runner.** Expands `runs × tasks × repetitions`; executes flows with bounded concurrency; **resumable** (skips rows already present); **reproducible** (records seed + pinned model IDs + library versions per run).

**Storage.** One **JSONL row per run** — `{experiment, task_id, pattern, bindings, final_answer, correct, score, per_role_usage, total_usage, latency_s, n_calls, trace_ref, seed, versions}` — written to `results/<experiment>/runs.jsonl`, with a **SQLite** index for querying and a roll-up summary. `results/` is gitignored.

---

## 7. Tasks, scoring & metrics

### 7.1 Tasks
- **`TaskSuite`** protocol: `tasks() -> Iterable[Task{id, prompt, ground_truth, type}]`.
- v1 suites (tiny, for fast iteration; swappable):
  - **Reasoning:** GSM8K-style numeric word problems (clean exact/numeric-match signal).
  - **Agentic:** a tool-use task with a programmatic checker.

### 7.2 Scoring
- **`Scorer`** protocol: `score(answer, ground_truth) -> {correct: bool, score: float}`.
- Per-suite scorers: numeric-match, normalized string-match, task-specific checker.

### 7.3 Metrics
Per (pattern × binding):
- **Accuracy**, **Δ vs B0** (raw lift), and the headline **Δ vs compute-matched B1/B2**.
- **Cost:** total tokens, USD, latency, #LLM-calls.
- **Cost-adjusted Pareto:** accuracy per 1k tokens / per $.
- **Diagnostics:** A↔B answer agreement & diversity; whether disagreement predicts gain.
- **Statistics:** with `repetitions ≥ 3`, report mean ± std and a paired test vs the relevant baseline.

---

## 8. Repository layout

```
multiagent-collab/
  pyproject.toml                  # Python 3.11/3.12; deps pinned
  README.md
  .gitignore                      # results/, .venv/, __pycache__/, *.sqlite
  docs/superpowers/specs/         # this spec + future specs
  src/macollab/
    agents/
      base.py                     # Agent protocol, AgentResponse, TokenUsage
      registry.py                 # config-string → adapter factory
      claude.py  openai_agents.py  google_adk.py  pydantic_ai.py
    flows/
      nodes.py                    # AgentNode + judge/aggregator nodes
      patterns/
        single.py  self_consistency.py  self_refine.py
        draft_refine.py           # P4 (v1)
        select_vote.py  synthesis.py  debate.py  moa.py
        planner_executor.py  solver_verifier.py
    tasks/
      base.py  reasoning_gsm8k.py  agentic_*.py
    scoring/
      base.py  numeric.py  string_match.py
    experiment/
      config.py  runner.py  store.py  metrics.py
    cli.py                        # `macollab run configs/<x>.yaml`
  configs/
    p4_vs_baselines.yaml
  tests/
    test_adapters_fake.py         # adapters vs a fake/echo model (no network)
    test_flows_stub.py            # flows with stub agents
    test_smoke_<sdk>.py           # 1 real call per SDK (network marker)
  results/                        # gitignored outputs
```

---

## 9. v1 milestone — "prove the loop"

1. `Agent` base + `TokenUsage`/`AgentResponse` + registry/factory.
2. **Two adapters first:** **Pydantic AI** (simplest; multi-provider) + **Claude Agent SDK** (validates the async/subprocess path). Other two adapters follow immediately after.
3. `AgentNode` + the **P4 Draft→Refine** `AsyncFlow` + the **B0 single** baseline pattern.
4. **GSM8K-mini** `TaskSuite` + numeric `Scorer`.
5. Config + runner + JSONL store + a metrics summary (P4 vs B0).
6. **Demo run:** P4 (A drafts, B refines) vs B0 on the mini-suite, emitting a metrics table.

Then: add ADK + OpenAI adapters and the remaining patterns/baselines against the proven loop.

**Testing strategy:** adapters tested against a **fake/echo model** (no network); flows tested with **stub agents**; **one real smoke test per SDK** behind a network marker. TDD per the project's normal workflow.

---

## 10. Risks & open questions
- **Python version:** target a dedicated **3.11 or 3.12** venv (all four SDKs need ≥3.10; 3.14 is bleeding-edge for some transitive deps). Aligns with the existing `env-gen` 3.11 venv.
- **Cost:** real LLM calls cost money; v1 defaults to tiny suites + cheap models, with a global token/$ budget guard in the runner.
- **Claude Agent SDK overhead** on pure-reasoning tasks (agent loop + Node subprocess): mitigated with `max_turns=1` and no tools. Confirm Node.js + Claude Code CLI availability in the venv.
- **Usage accounting parity:** ADK requires summing per-event usage; LiteLLM-backed usage can be unreliable for some providers (may need `include_usage`/`ModelSettings`). Validate per provider in smoke tests.
- **Judge bias:** when `J ∈ {A, B}`, self-preference may bias P1/P2/P6. Mitigation: support a neutral judge and/or blind ordering; treat as a study variable.

---

## Appendix A — Grounded SDK/PocketFlow API facts (research 2026-05-29)

Captured from official docs/source so adapters are implemented against reality, not guesswork.

**PocketFlow** — install `pocketflow` (v0.0.3, zero deps, one ~100-line file). No LLM client, no usage tracking. `Node` lifecycle `prep/exec/post`; `Flow(start=node)`; transitions `a >> b` (default) and `a - "action" >> b`. `flow.run(shared)` returns the terminal **action string**, not output — read results from the mutated `shared`. Async: `AsyncNode`/`AsyncFlow` with `prep_async/exec_async/post_async` and `await flow.run_async(shared)`; `AsyncParallelBatchNode` runs items concurrently via `asyncio.gather`. `max_retries` = total attempts; `wait` = seconds between; `exec_fallback` fires on the final failed attempt. There are separate `pocketflow-typescript`/`pocketflow-go` packages — use the Python one.

**Claude Agent SDK** — install `claude-agent-sdk`, import `claude_agent_sdk` (Python ≥3.10). **Async-only.** `query(prompt=..., options=ClaudeAgentOptions(...))` is an async generator; iterate to a `ResultMessage` whose `.result` is the final text (`None` if `is_error`), `.usage` is a `dict` (`input_tokens`/`output_tokens`/cache fields), `.total_cost_usd` is cost. Spawns the bundled Claude Code CLI as a Node subprocess; use `permission_mode="bypassPermissions"` and `max_turns=1` for single-turn automation. Distinct from the low-level `anthropic` SDK. Current model IDs: `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5`.

**OpenAI Agents SDK** — install `openai-agents`, import `agents` (Python ≥3.10; needs `openai` v2.x). `Agent(name, instructions, model, tools=[...])`; `await Runner.run(agent, task)` or `Runner.run_sync(...)` → `RunResult`. Text: `result.final_output`. Usage: **`result.context_wrapper.usage`** (`input_tokens`/`output_tokens`/`total_tokens`) — *not* `result.usage`. Tools via `@function_tool`. Multi-provider via `LitellmModel` (`pip install "openai-agents[litellm]"`; may need `ModelSettings(include_usage=True)`).

**Google ADK** — install `google-adk`, import `google.adk` (Python ≥3.10; latest 2.1.0). `Agent(name, model, instruction, tools=[...])` (Agent = LlmAgent); `InMemoryRunner` / `Runner`; **must create a session first** (`await runner.session_service.create_session(...)`); iterate `runner.run_async(user_id=..., session_id=..., new_message=types.Content(...))` (keyword-only, async). Final text from the event where `event.is_final_response()`; **sum** `event.usage_metadata` (`prompt_token_count`/`candidates_token_count`/`total_token_count`) across events. Non-Gemini via `from google.adk.models.lite_llm import LiteLlm`.

**Pydantic AI** — install `pydantic-ai` (or `pydantic-ai-slim[provider]`), import `pydantic_ai` (Python ≥3.10; v1.x). `Agent("<provider>:<model>", instructions=...)`; `agent.run_sync(task)` or `await agent.run(task)`. Text: `result.output` (was `result.data`, now removed). Usage: `result.usage()` → `RunUsage` (`input_tokens`/`output_tokens`/`total_tokens`, `requests`). Provider-agnostic strings: `anthropic:...`, `openai:...`, `google-gla:...` (Gemini Dev API), `google-vertex:...`, etc. Tools via `@agent.tool` / `@agent.tool_plain`. `output_type=` forces structured output.

*Versions/model IDs current as of 2026-05-29; pin known-good versions in `pyproject.toml`.*
