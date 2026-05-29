# multiagent-collab

A research framework for studying how **two agents collaborating** — and *the manner in which they collaborate* — affects task capability, relative to a single agent.

Two independent variables under study:

1. **Agent identity** — the model and the **agent SDK** behind each agent (same or different).
   Supported SDKs: **Claude Agent SDK**, **OpenAI Agents SDK**, **Google ADK**, **Pydantic AI**.
2. **Collaboration pattern** — *how* the two agents combine on one problem
   (aggregate / vote, draft→refine, critique→revise, debate, mixture-of-agents,
   planner–executor, solver–verifier, …).

Orchestration uses [**PocketFlow**](https://github.com/the-pocket/PocketFlow): each collaboration
pattern is a small `AsyncFlow` of role-nodes; every agent sits behind a uniform async `Agent`
adapter so flows never touch a raw SDK. Experiments are config-driven (`configs/*.yaml`) and
evaluated against compute-matched single-agent baselines (self-consistency, self-refine) so a
measured "improvement" reflects *collaboration*, not just extra compute.

## Status

Design phase. See the full design spec:
[`docs/superpowers/specs/2026-05-29-multiagent-collaboration-framework-design.md`](docs/superpowers/specs/2026-05-29-multiagent-collaboration-framework-design.md).

## Planned layout

```
src/macollab/
  agents/      # uniform Agent adapters (claude / openai / google-adk / pydantic-ai) + registry
  flows/       # PocketFlow AgentNode + collaboration patterns
  tasks/       # reasoning + agentic task suites
  scoring/     # per-suite scorers
  experiment/  # config, runner, store, metrics
configs/       # experiment matrices
```
