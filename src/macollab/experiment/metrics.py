from __future__ import annotations

from collections import defaultdict


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["pattern"], r["bind_key"])].append(r)

    rows: list[dict] = []
    for (pattern, bind_key), items in groups.items():
        ok = [r for r in items if not r.get("error")]
        errors = len(items) - len(ok)
        n = len(ok)
        accuracy = (sum(1 for r in ok if r["correct"]) / n) if n else 0.0
        mean_tokens = (sum(r["total_tokens"] for r in ok) / n) if n else 0.0
        mean_latency = (sum(r["latency_s"] for r in ok) / n) if n else 0.0
        rows.append({
            "pattern": pattern,
            "bind_key": bind_key,
            "n": n,
            "errors": errors,
            "accuracy": round(accuracy, 4),
            "mean_total_tokens": round(mean_tokens, 1),
            "mean_latency_s": round(mean_latency, 4),
            "delta_vs_single": None,
        })

    single_rows = [r for r in rows if r["pattern"] == "single"]
    if single_rows:
        baseline = single_rows[0]["accuracy"]
        for r in rows:
            r["delta_vs_single"] = round(r["accuracy"] - baseline, 4)

    rows.sort(key=lambda r: (r["pattern"] != "single", r["pattern"], r["bind_key"]))
    return rows


_COLS = [
    ("pattern", 16),
    ("bind_key", 28),
    ("n", 4),
    ("errors", 7),
    ("accuracy", 9),
    ("delta_vs_single", 16),
    ("mean_total_tokens", 18),
    ("mean_latency_s", 15),
]


def format_table(rows: list[dict]) -> str:
    header = "  ".join(name.ljust(width) for name, width in _COLS)
    lines = [header, "  ".join("-" * width for _, width in _COLS)]
    for r in rows:
        lines.append("  ".join(str(r.get(name, "")).ljust(width) for name, width in _COLS))
    return "\n".join(lines)
