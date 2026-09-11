"""Write the paired comparisons of spec C Section 12 from a completed Phase 2 main experiment."""

import argparse
import csv
import json
from pathlib import Path
import sys

from analysis.statistics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, PRIMARY_COMPARISONS, STATISTICS_COLUMNS, compare_methods


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase2-statistics")
    parser.add_argument("--results", type=Path, default=Path("results/phase2/main"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2/statistics.csv"))
    args = parser.parse_args(argv)
    manifest = args.results / "manifest.json"
    if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8"))["status"] != "complete":
        print(f"{args.results} holds no complete experiment; run experiments/run_phase2.py first", file=sys.stderr)
        return 1
    with (args.results / "method_realizations.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    results = compare_methods(rows, PRIMARY_COMPARISONS, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(STATISTICS_COLUMNS))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"comparisons": len(results), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
