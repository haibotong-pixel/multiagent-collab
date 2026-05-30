from pathlib import Path

from macollab.experiment.store import JsonlStore, run_key


def test_append_and_load_roundtrip(tmp_path: Path):
    store = JsonlStore(tmp_path / "runs.jsonl")
    store.append({"pattern": "single", "bind_key": "solo=A", "task_id": "t1", "rep": 0, "score": 1.0})
    store.append({"pattern": "single", "bind_key": "solo=A", "task_id": "t2", "rep": 0, "score": 0.0})
    rows = store.load_all()
    assert len(rows) == 2
    assert rows[0]["task_id"] == "t1"


def test_existing_keys_enables_resume(tmp_path: Path):
    store = JsonlStore(tmp_path / "runs.jsonl")
    store.append({"pattern": "single", "bind_key": "solo=A", "task_id": "t1", "rep": 0})
    keys = store.existing_keys()
    assert run_key("single", "solo=A", "t1", 0) in keys
    assert run_key("single", "solo=A", "t1", 1) not in keys


def test_load_all_missing_file_is_empty(tmp_path: Path):
    assert JsonlStore(tmp_path / "none.jsonl").load_all() == []
