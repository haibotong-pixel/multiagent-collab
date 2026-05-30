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
