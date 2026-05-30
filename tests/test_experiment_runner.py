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
