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
