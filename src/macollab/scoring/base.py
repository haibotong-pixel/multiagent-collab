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
