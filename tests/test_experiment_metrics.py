from macollab.experiment.metrics import format_table, summarize


def _rec(pattern, bind_key, correct, tokens, error=None):
    return {
        "pattern": pattern, "bind_key": bind_key, "correct": correct,
        "total_tokens": tokens, "latency_s": 0.1, "error": error,
    }


def test_summarize_accuracy_and_delta_vs_single():
    records = [
        _rec("single", "solo=A", True, 100),
        _rec("single", "solo=A", False, 100),
        _rec("draft_refine", "drafter=A,refiner=B", True, 250),
        _rec("draft_refine", "drafter=A,refiner=B", True, 250),
    ]
    rows = {(r["pattern"], r["bind_key"]): r for r in summarize(records)}

    single = rows[("single", "solo=A")]
    p4 = rows[("draft_refine", "drafter=A,refiner=B")]
    assert single["n"] == 2 and single["accuracy"] == 0.5
    assert p4["accuracy"] == 1.0
    assert p4["delta_vs_single"] == 0.5
    assert p4["mean_total_tokens"] == 250


def test_summarize_excludes_errored_rows_from_accuracy():
    records = [
        _rec("single", "solo=A", True, 100),
        _rec("single", "solo=A", False, 100, error="Boom: x"),
    ]
    row = summarize(records)[0]
    assert row["n"] == 1          # errored row excluded from n/accuracy
    assert row["errors"] == 1
    assert row["accuracy"] == 1.0


def test_format_table_includes_headers_and_patterns():
    records = [_rec("single", "solo=A", True, 100)]
    table = format_table(summarize(records))
    assert "pattern" in table and "accuracy" in table and "single" in table
