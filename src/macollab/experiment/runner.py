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
