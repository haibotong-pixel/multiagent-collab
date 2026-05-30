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
