from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Task:
    id: str
    prompt: str
    ground_truth: Any
    type: str = "reasoning"


class TaskSuite(Protocol):
    name: str

    def tasks(self) -> list[Task]:
        ...
