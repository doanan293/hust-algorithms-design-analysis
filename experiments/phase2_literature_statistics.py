"""B1 anchor check and paired comparisons of TRAN with P and B2 (spec D Section 7.3)."""

import argparse
import csv
import json
from pathlib import Path
import sys

from analysis.statistics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, STATISTICS_COLUMNS, Comparison, compare_methods

ANCHOR_METHOD = "B1"
ANCHOR_COLUMNS = ("timely_count", "connected_count", "shortfall")
LITERATURE_COMPARISONS = tuple(
    Comparison(set_id, first, "TRAN") for set_id in ("v0", "v0-bh50") for first in ("P", "B2")
)


def _complete_rows(results: Path) -> list[dict[str, str]] | None:
    manifest = results / "manifest.json"
    if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8"))["status"] != "complete":
        return None
    with (results / "method_realizations.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def anchor_mismatches(main_rows: list[dict[str, str]], literature_rows: list[dict[str, str]]) -> list[str]:
    """Differences of B1 between the two experiments on (set, scenario, realization); empty when they agree."""
    def index(rows):
        return {(row["set_id"], row["scenario_id"], row["realization_id"]): row for row in rows if row["method"] == ANCHOR_METHOD}

    main, literature = index(main_rows), index(literature_rows)
    if not literature:
        return ["the literature results hold no B1 rows"]
    messages = [f"{key}: B1 row missing from the main results" for key in sorted(literature.keys() - main.keys())]
    for key in sorted(literature.keys() & main.keys()):
        for column in ANCHOR_COLUMNS:
            if float(main[key][column]) != float(literature[key][column]):
                messages.append(f"{key} {column}: main {main[key][column]}, literature {literature[key][column]}")
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase2-literature-statistics")
    parser.add_argument("--main", type=Path, default=Path("results/phase2/main"))
    parser.add_argument("--literature", type=Path, default=Path("results/phase2/literature"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2/statistics_literature.csv"))
    args = parser.parse_args(argv)
    main_rows, literature_rows = _complete_rows(args.main), _complete_rows(args.literature)
    if main_rows is None or literature_rows is None:
        print("both --main and --literature must hold complete experiments", file=sys.stderr)
        return 1
    mismatches = anchor_mismatches(main_rows, literature_rows)
    if mismatches:
        print("B1 anchor check failed; pairing with the main results is invalid:", file=sys.stderr)
        for message in mismatches[:5]:
            print(f"  {message}", file=sys.stderr)
        return 1
    rows = [row for row in main_rows if row["method"] in ("P", "B2")] + [row for row in literature_rows if row["method"] == "TRAN"]
    results = compare_methods(rows, LITERATURE_COMPARISONS, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(STATISTICS_COLUMNS))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"comparisons": len(results), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
