from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from macollab.experiment.config import load_config
from macollab.experiment.metrics import format_table, summarize
from macollab.experiment.runner import run_experiment
from macollab.experiment.store import JsonlStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="macollab")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run an experiment config")
    run_p.add_argument("config", help="path to the experiment YAML")
    run_p.add_argument("--out", default=None, help="JSONL output path")
    run_p.add_argument("--concurrency", type=int, default=4)

    args = parser.parse_args(argv)

    if args.command == "run":
        cfg = load_config(args.config)
        out = Path(args.out) if args.out else Path("results") / cfg.experiment / "runs.jsonl"
        store = JsonlStore(out)
        asyncio.run(run_experiment(cfg, store, concurrency=args.concurrency))
        rows = summarize(store.load_all())
        print(f"\nExperiment: {cfg.experiment}  ->  {out}\n")
        print(format_table(rows))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
