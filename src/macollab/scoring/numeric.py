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
