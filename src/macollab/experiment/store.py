from __future__ import annotations

import json
from pathlib import Path


def run_key(pattern: str, bind_key: str, task_id: str, rep: int) -> tuple:
    return (pattern, bind_key, task_id, int(rep))


class JsonlStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def existing_keys(self) -> set[tuple]:
        return {
            run_key(r["pattern"], r["bind_key"], r["task_id"], r["rep"])
            for r in self.load_all()
        }
