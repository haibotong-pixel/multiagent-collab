# Related Work — Multi-Agent LLM Collaboration

*Draft related-work section for the multiagent-collab project. Compiled 2026-05-29 from a fan-out, adversarially fact-checked literature survey (5 search angles, 19 primary sources, 87 candidate claims, 25 verified, 18 confirmed / 7 refuted). Verification status is annotated where it matters; see **Provenance & caveats** at the end before citing specific numbers.*

Pattern codes (`P1`–`P9`) and baseline codes (`B0`/`B1`/`B2`) refer to the taxonomy in the design spec.

## 1. The central tension

Whether multi-agent collaboration "helps" is **entirely framing-dependent**, and the literature splits cleanly along this line:

- **vs. a single one-pass agent** — collaboration almost always wins (debate, mixture-of-agents, role frameworks, sampling+voting).
- **vs. a *compute-matched* single agent** — repeated sampling + majority vote (self-consistency / "More Agents"), or single-model self-refinement — collaboration **frequently ties or loses**, and reported gains are often attributable to the extra inference compute and to hyperparameter tuning rather than to the collaboration *structure* itself.

This is the gap the present project targets: it makes the compute-matched baselines (`B1` self-consistency, `B2` self-refine) first-class, scored identically to every collaboration pattern.

## 2. Collaboration patterns

**Multi-agent debate.** Du et al. (2023; ICML 2024) — multiple LLM instances independently answer, then critique/debate over rounds to converge. With 3 agents × 2 rounds they report Arithmetic 67→82%, GSM8K 77→85%, improved chess move quality, and reduced biography hallucination — **all measured against a single one-pass agent** (`P6`). Liang et al. (2023; EMNLP 2024) propose a separate "MAD" framework (tit-for-tat agents + a judge) to overcome *Degeneration-of-Thought*; critically, the benefit is **configuration-dependent and inverted-U** — only a *modest* level of disagreement and an adaptive stop help; too much or too little disagreement degrades results. (Implication for us: `#rounds`, judge identity, and disagreement level are real experimental variables, not free choices.)

**Sampling-and-voting ensembles.** Li et al., "More Agents Is All You Need" (2024; TMLR) — "Agent Forest": majority vote over independently sampled outputs. Performance scales with the number of samples across GSM8K/MATH/MMLU/Chess/HumanEval; an ensembled Llama2-13B (59% GSM8K) beats a single Llama2-70B (54%); and **standalone sampling+voting matches or exceeds more complex methods, including LLM-Debate, in most task-model cells**. This is the canonical operationalization of our `B1` and the bar collaboration must clear. (Related background, not independently re-verified here: self-consistency, Wang et al. 2022/ICLR 2023; LLM-Blender rank-and-fuse, Jiang et al. 2023/ACL — our `P1`/`P2`; Mixture-of-Agents, Wang et al. 2024 — our `P7`.)

**Single-model iterative refinement.** Self-Refine (Madaan et al. 2023; NeurIPS 2023) — one LLM acts as generator, feedback-provider, and refiner, with no extra training. This is exactly our `B2`, and the standard compute-matched control for critique-then-revise / generator-verifier collaboration (`P5`/`P9`). *(Its frequently-cited "~20% average improvement" figure did not pass our fact-check and should be cited cautiously.)*

**Role / conversation frameworks.** AutoGen (Wu et al. 2023; COLM 2024) — customizable conversable agents combining LLMs, humans, and tools — is the canonical open framework, alongside CAMEL, MetaGPT, and ChatDev (our `P8`/`P9` territory). In AutoGen's own MATH evaluation its two-agent setup reached 69.5% vs GPT-4's 55%, and — an early negative signal — Liang-style debate and LangChain-ReAct *underperformed* a vanilla single GPT-4 on the smaller set. (Author-reported; not independently reproduced.)

## 3. The compute-matched / skeptical line (the core of our motivation)

- **Smit et al., "Should we be going MAD?" (2023; ICML 2024)** — multi-agent debating systems, in their current form, **do not reliably outperform self-consistency / ensembling**; observed performance is governed largely by **hyperparameter sensitivity** rather than inherent superiority (with tuning, some MAD variants *can* become best-in-class — e.g. a Multi-Persona setup beating Medprompt/Self-Consistency on a USMLE subset).
- **Zhang et al. (2025)** — systematic evaluation of 5 representative MAD methods × 9 benchmarks × 4 models: MAD **often fails to beat Chain-of-Thought and Self-Consistency while spending significantly more inference compute**. Heterogeneous debate (Heter-MAD, mixing models) recovers **4–6 points** — a qualifier, not a refutation.
- **Tran & Kiela (Stanford, 2026 preprint)** — under **matched reasoning-token budgets**, a standard single-agent system is the best or statistically indistinguishable from the best multi-agent architecture (sequential, parallel-roles, debate, ensemble) on multi-hop QA (FRAMES, MuSiQue) at all but the lowest budget, across Qwen3 / DeepSeek-R1-Distill / Gemini 2.5. Argues (via a Data Processing Inequality) that many reported multi-agent gains are **artifacts of unaccounted compute and context effects**. (Scoped to idealized perfect-context multi-hop reasoning; the authors note MAS may regain advantage on long/noisy inputs or parallel tool use.)
- **Cemri et al., MAST (UC Berkeley, 2025; NeurIPS 2025 D&B)** — 1600+ annotated traces across 7 frameworks; finds MAS performance gains on popular benchmarks are **often minimal**, and builds a failure-mode taxonomy (inter-annotator Cohen's κ = 0.88). *(The specific "14 modes / 3 categories" breakdown did not pass our fact-check.)*

**Most pointed verified result:** using **Du et al.'s own numbers**, the claim that debate *substantially* beats a compute-matched **multi-agent majority vote** was **refuted (0/3)** — GSM8K debate 85% vs plain majority 81% vs single 77%; the debate *structure* adds little once you already aggregate multiple samples.

## 4. Model / agent heterogeneity

- **Bounded positive (corroborated):** mixing models (Heter-MAD) recovers ~4–6 points over homogeneous debate (Zhang et al. 2025).
- **Strong version refuted (1/2):** the claim that heterogeneity is a "universal antidote" / that diversity is *the* core driver of collaboration gains did **not** survive adversarial verification.
- **Open:** whether cross-model diversity does anything that single-model temperature sampling cannot. Directly addressable by our `model × SDK` axis.

## 5. Two-agent / small-N, and the framework effect

No verified result isolates the **two-agent / small-N** regime against compute-matched baselines — the small-N question is essentially **open**. Likewise, no work cleanly measures whether the **SDK / orchestration framework itself** (AutoGen vs CAMEL vs a custom harness) changes capability when pattern, model, and compute are held fixed.

## 6. How this project differs (the gaps we address)

1. **Structure vs. brute-force compute.** We score every pattern (`P1`–`P9`) against **strictly compute-matched** `B1`/`B2` on the *same* tasks — the comparison most of the positive literature omits and most of the skeptical literature performs only piecemeal.
2. **Reasoning *and* agentic.** The compute-matched negative results are established almost entirely on reasoning benchmarks (GSM8K/MATH/MMLU/multi-hop QA). Their validity on **agentic / tool-use** tasks (GAIA, SWE-bench, long-horizon tool execution) is the **largest open gap**; our eval spans both.
3. **Heterogeneity, isolated.** Holding the underlying model fixed across SDKs (Pydantic AI / LiteLLM-wrapped OpenAI / ADK) lets us separate the **model effect** from the **SDK/scaffolding effect**, and homogeneous vs heterogeneous pairs let us test whether cross-model diversity is causal.
4. **Two-agent, controlled.** A clean `pattern × model × SDK` matrix at the smallest collaborative unit, rather than large emergent agent societies.

## Provenance & caveats

- Compiled from an adversarially fact-checked survey; **18 of 25 verified claims confirmed, 7 refuted.** Only confirmed items are stated as findings above; refuted/unverified specifics are flagged inline.
- **Refuted or unverified (do not cite as fact):** "debate beats compute-matched majority vote" (refuted, Du et al. data); "heterogeneity is a universal antidote" (refuted strong form); Self-Refine "~20% average gain" (unverified); MAST "14 failure modes / 3 categories" (unverified); a claim that self-consistency yields only ~0.4% gains on modern LLMs (refuted — do **not** assume self-consistency is weak).
- The positive vs negative literatures are **largely not in direct contradiction** — they compare against *different* baselines (one-pass vs compute-matched). State the baseline explicitly whenever quoting a number.
- Tran & Kiela (2026) is an unrefereed preprint scoped to idealized multi-hop reasoning; do not over-generalize to agentic settings.

## References

| Cite | arXiv | Venue |
|---|---|---|
| Du et al., *Improving Factuality and Reasoning via Multiagent Debate* | [2305.14325](https://arxiv.org/abs/2305.14325) | ICML 2024 |
| Liang et al., *Encouraging Divergent Thinking … Multi-Agent Debate* | [2305.19118](https://arxiv.org/abs/2305.19118) | EMNLP 2024 |
| Wu et al., *AutoGen* | [2308.08155](https://arxiv.org/abs/2308.08155) | COLM 2024 |
| Li et al., *More Agents Is All You Need* | [2402.05120](https://arxiv.org/abs/2402.05120) | TMLR 2024 |
| Madaan et al., *Self-Refine* | [2303.17651](https://arxiv.org/abs/2303.17651) | NeurIPS 2023 |
| Smit et al., *Should we be going MAD?* | [2311.17371](https://arxiv.org/abs/2311.17371) | ICML 2024 |
| Zhang et al., *MAD systematic evaluation / Heter-MAD* | [2502.08788](https://arxiv.org/abs/2502.08788) | 2025 |
| Tran & Kiela, *compute-matched multi-agent vs single* | [2604.02460](https://arxiv.org/abs/2604.02460) | 2026 (preprint) |
| Cemri et al., *Why Do Multi-Agent LLM Systems Fail? (MAST)* | [2503.13657](https://arxiv.org/abs/2503.13657) | NeurIPS 2025 D&B |
| Wang et al., *Mixture-of-Agents* (background) | [2406.04692](https://arxiv.org/abs/2406.04692) | 2024 |
| Jiang et al., *LLM-Blender* (background) | [2306.02561](https://arxiv.org/abs/2306.02561) | ACL 2023 |
| Wang et al., *Self-Consistency* (background) | [2203.11171](https://arxiv.org/abs/2203.11171) | ICLR 2023 |
