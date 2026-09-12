"""Write report CSVs and result macros for docs/report from the completed Phase 2 experiments."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Mapping, Sequence

from analysis.statistics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, bootstrap_mean_ci, topology_means
from report_tables import _read_rows, _write_semicolon, kbit, macro_name, num
from runner.provenance import source_hash

EXPERIMENTS = ("main", "budget", "sensitivity", "design")
SETS = (("v0", 1000000.0), ("v0-bh50", 50000.0))
MAIN_METHODS = ("B0", "B1", "B2", "B3", "P")
COMPARISON_LABELS = {"P - B2": "P $-$ B2", "B2 - B1": "B2 $-$ B1", "B2 - B3": "B2 $-$ B3", "P - B1": "P $-$ B1"}
ABLATIONS = (
    ("P_op1", "Chỉ thao tác 1"),
    ("P_op2", "Chỉ thao tác 2"),
    ("P_no_backlog", "Sửa lịch không dùng backlog"),
    ("P_fixed_paths", "Không chạy khối đường"),
    ("P_fixed_trajectory", "Không chạy khối quỹ đạo"),
    ("P_equal_bandwidth", "Không chạy khối băng thông"),
    ("P_expected_design", "Thiết kế trên tốc độ kỳ vọng"),
)
BUDGETS = (250, 500, 1000, 2000, 4000)
SENSITIVITY_PANELS = (
    ("lref", (("sens-lref125k", 125), ("v0", 250), ("sens-lref500k", 500), ("sens-lref1m", 1000))),
    ("backhaul", (("v0-bh50", 50), ("sens-bh100", 100), ("v0", 1000))),
    ("deadline", (("sens-deadline-short", 20), ("v0", 40), ("sens-deadline-long", 80))),
    ("btot", (("sens-btot05", 0.5), ("v0", 1), ("sens-btot2", 2))),
    ("alerts", (("sens-alerts20", 20), ("v0", 40), ("sens-alerts60", 60))),
    ("paths", (("sens-k2", 2), ("v0", 3), ("sens-k5", 5))),
)
DESIGN_POINTS = (("P_design_expected", "design", "Tốc độ kỳ vọng"), ("P_design1", "design", "1"), ("P", "sensitivity", "5"), ("P_design10", "design", "10"))
TERCILES = (("low", "Tối đa 1/3"), ("middle", "Từ 1/3 đến 2/3"), ("high", "Từ 2/3"))


def macro(*parts: str) -> str:
    return "ResultTwo" + macro_name(*parts)[len("Result"):]


def number(value: float | str, digits: int = 3) -> str:
    """siunitx number with fixed digits; nan becomes a dash and a value that rounds to zero loses its sign."""
    value = float(value)
    if value != value:
        return "---"
    return num(0.0 if round(value, digits) == 0 else value, digits)


def scenario_means(rows: Sequence[Mapping[str, str]], set_id: str, method: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["set_id"] == set_id and row["method"] == method:
            values[row["scenario_id"]].append(float(row["timely_ratio"]))
    return {scenario: statistics.fmean(items) for scenario, items in sorted(values.items())}


def compare_scenarios(first: Mapping[str, float], second: Mapping[str, float]) -> tuple[int, int, int]:
    higher = sum(first[key] > second[key] for key in first)
    equal = sum(first[key] == second[key] for key in first)
    return higher, equal, len(first) - higher - equal


def tercile(fraction: float) -> str:
    if fraction <= 1 / 3 + 1e-6:
        return "low"
    return "high" if fraction >= 2 / 3 - 1e-6 else "middle"


def compute_minutes(root: Path) -> float:
    rows = _read_rows(root / "method_realizations.csv")
    planning = {(row["set_id"], row["scenario_id"], row["method"]): float(row["planning_runtime_s"]) for row in rows}
    evaluation = sum(float(row["evaluation_runtime_s"]) for row in rows)
    bounds = sum(float(row["runtime_s"]) for row in _read_rows(root / "bounds.csv"))
    return (sum(planning.values()) + evaluation + bounds) / 60.0


def write_points(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="report-tables-phase2")
    parser.add_argument("--results", type=Path, default=Path("results/phase2"))
    parser.add_argument("--manifests", type=Path, default=Path("data/manifests"))
    parser.add_argument("--output", type=Path, default=Path("docs/report/data"))
    args = parser.parse_args(argv)

    current = source_hash()
    for name in EXPERIMENTS:
        manifest_path = args.results / name / "manifest.json"
        if not manifest_path.exists():
            print(f"missing {manifest_path}; run experiments/run_phase2.py first", file=sys.stderr)
            return 1
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["source_hash"] != current:
            print(f"{name} results are incomplete or were produced by different source code; rerun the experiment", file=sys.stderr)
            return 1
    summaries = {name: _read_rows(args.results / name / "summary.csv") for name in EXPERIMENTS}
    rows = {name: _read_rows(args.results / name / "method_realizations.csv") for name in EXPERIMENTS}
    summary = {name: {(row["set_id"], row["method"]): row for row in summaries[name]} for name in EXPERIMENTS}
    args.output.mkdir(parents=True, exist_ok=True)
    macros: list[tuple[str, str]] = []

    main_rows = []
    for set_id, capacity in SETS:
        for method in MAIN_METHODS:
            row = summary["main"][(set_id, method)]
            main_rows.append([
                set_id, kbit(capacity), method, number(row["timely_ratio_mean"]),
                f"[{number(row['timely_ratio_ci_low'])}, {number(row['timely_ratio_ci_high'])}]", number(row["conn_ratio_mean"]),
                number(row["gap_mean"]), number(row["union_gap_mean"]), number(row["planning_runtime_s_mean"], 1), number(row["evaluate_calls_mean"], 0),
            ])
            for metric, column, digits in (
                ("Timely", "timely_ratio_mean", 3), ("TimelyLow", "timely_ratio_ci_low", 3), ("TimelyHigh", "timely_ratio_ci_high", 3),
                ("Conn", "conn_ratio_mean", 3), ("Gap", "gap_mean", 3), ("UnionGap", "union_gap_mean", 3), ("UnionBound", "union_bound_ratio_mean", 3),
                ("Planning", "planning_runtime_s_mean", 1), ("Calls", "evaluate_calls_mean", 0),
            ):
                macros.append((macro(metric, set_id, method), number(row[column], digits)))
        means = {method: scenario_means(rows["main"], set_id, method) for method in MAIN_METHODS}
        for first, second in (("B2", "B0"), ("B2", "B1"), ("P", "B2"), ("B1", "B0")):
            higher, equal, lower = compare_scenarios(means[first], means[second])
            for word, value in (("Higher", higher), ("Equal", equal), ("Lower", lower)):
                macros.append((macro(word, set_id, first, second), str(value)))
    _write_semicolon(args.output / "phase2_main.csv", ["set", "backhaul", "method", "timely", "ci", "conn", "gap", "uniongap", "planning", "calls"], main_rows)
    macros.append(("ResultTwoScenarioCount", summary["main"][("v0", "B2")]["scenario_count"]))

    statistics_rows = []
    for row in _read_rows(args.results / "statistics.csv"):
        statistics_rows.append([
            row["set_id"], COMPARISON_LABELS[row["comparison"]], number(row["mean_difference"]),
            f"[{number(row['ci_low'])}, {number(row['ci_high'])}]", number(row["rank_biserial"], 2), number(row["p_value"], 4), number(row["p_holm"], 4),
        ])
        first, second = row["comparison"].split(" - ")
        for metric, column, digits in (("Diff", "mean_difference", 3), ("DiffLow", "ci_low", 3), ("DiffHigh", "ci_high", 3), ("Holm", "p_holm", 4), ("RankBiserial", "rank_biserial", 2)):
            macros.append((macro(metric, row["set_id"], first, second), number(row[column], digits)))
    _write_semicolon(args.output / "phase2_statistics.csv", ["set", "comparison", "diff", "ci", "rankbiserial", "p", "pholm"], statistics_rows)
    macros.append(("ResultTwoHolmFloor", number(min(float(row["p_holm"]) for row in _read_rows(args.results / "statistics.csv")), 4)))

    ablation_rows = []
    for method, label in (("P", "P đầy đủ"), *ABLATIONS):
        cells = [label]
        for set_id, _ in SETS:
            row = summary["main"][(set_id, method)]
            cells.append(number(row["timely_ratio_mean"]))
            if method == "P":
                cells.append("---")
                continue
            variant = topology_means(rows["main"], set_id, method)
            full = topology_means(rows["main"], set_id, "P")
            differences = [variant[topology] - full[topology] for topology in full]
            low, high = bootstrap_mean_ci(differences, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
            mean = statistics.fmean(differences)
            cells.append(f"{number(mean)} [{number(low)}, {number(high)}]")
            macros.append((macro("AblationDiff", set_id, method), number(mean)))
            macros.append((macro("AblationTimely", set_id, method), number(row["timely_ratio_mean"])))
        cells.append(number(summary["main"][("v0", method)]["evaluate_calls_mean"], 0))
        ablation_rows.append(cells)
    _write_semicolon(args.output / "phase2_ablation.csv", ["variant", "timely", "diff", "timelybh", "diffbh", "calls"], ablation_rows)

    connectivity_rows = []
    for set_id, _ in SETS:
        with (args.manifests / f"scenarios_{set_id}.csv").open(newline="", encoding="utf-8") as stream:
            fractions = {row["scenario_id"]: float(row["ground_connected_source_fraction"]) for row in csv.DictReader(stream) if row["split"] == "eval"}
        means = {method: scenario_means(rows["main"], set_id, method) for method in ("B0", "B1", "B2")}
        connectivity = defaultdict(list)
        for row in rows["main"]:
            if row["set_id"] == set_id and row["method"] == "B0":
                connectivity[row["scenario_id"]].append(float(row["conn_ratio"]))
        for key, label in TERCILES:
            group = [scenario for scenario in sorted(fractions) if tercile(fractions[scenario]) == key]
            b0, b1, b2 = (statistics.fmean(means[method][scenario] for scenario in group) for method in ("B0", "B1", "B2"))
            conn = statistics.fmean(statistics.fmean(connectivity[scenario]) for scenario in group)
            connectivity_rows.append([set_id, label, str(len(group)), number(conn), number(b0), number(b1), number(b2), number(b2 - b0)])
            macros.append((macro("TercileGain", set_id, key), number(b2 - b0)))
        b0 = [means["B0"][scenario] for scenario in sorted(fractions)]
        try:
            correlation = number(statistics.correlation([fractions[s] for s in sorted(fractions)], b0), 2)
        except statistics.StatisticsError:  # constant input: the correlation is undefined
            correlation = "---"
        macros.append((macro("GroundCorrelation", set_id), correlation))
    _write_semicolon(args.output / "phase2_connectivity.csv", ["set", "group", "scenarios", "conn", "b0", "b1", "b2", "gain"], connectivity_rows)

    for method in ("B2", "P"):
        points = []
        for budget in BUDGETS:
            row = summary["budget"][("v0", f"{method}_b{budget}")]
            points.append([budget, f"{float(row['timely_ratio_mean']):.6f}", f"{float(row['evaluate_calls_mean']):.1f}", f"{float(row['planning_runtime_s_mean']):.3f}"])
            macros.append((macro("Budget", method, str(budget)), number(row["timely_ratio_mean"])))
            macros.append((macro("BudgetCalls", method, str(budget)), number(row["evaluate_calls_mean"], 0)))
        write_points(args.output / f"phase2_budget_{method.lower()}.csv", ["budget", "timely", "calls", "planning"], points)

    for panel, points in SENSITIVITY_PANELS:
        lines = []
        for set_id, value in points:
            values = [float(summary["sensitivity"][(set_id, method)]["timely_ratio_mean"]) for method in ("B1", "B2", "P")]
            lines.append([value, *(f"{item:.6f}" for item in values)])
        write_points(args.output / f"phase2_sensitivity_{panel}.csv", ["x", "B1", "B2", "P"], lines)
    for set_id in sorted({key[0] for key in summary["sensitivity"]}):
        for method in ("B1", "B2", "P"):
            macros.append((macro("Sens", set_id, method), number(summary["sensitivity"][(set_id, method)]["timely_ratio_mean"])))

    design_rows = []
    for method, experiment, label in DESIGN_POINTS:
        row = summary[experiment][("v0", method)]
        design_rows.append([label, number(row["timely_ratio_mean"]), number(row["planning_runtime_s_mean"], 1), number(row["evaluate_calls_mean"], 0)])
        macros.append((macro("Design", label.replace("Tốc độ kỳ vọng", "expected")), number(row["timely_ratio_mean"])))
        macros.append((macro("DesignPlanning", label.replace("Tốc độ kỳ vọng", "expected")), number(row["planning_runtime_s_mean"], 1)))
    _write_semicolon(args.output / "phase2_design.csv", ["design", "timely", "planning", "calls"], design_rows)

    bound_rows = [row for name in ("main", "sensitivity") for row in _read_rows(args.results / name / "bounds.csv")]
    macros += [
        ("ResultTwoLpCount", str(len(bound_rows))),
        ("ResultTwoLpOptimalCount", str(sum(row["status"] == "optimal" for row in bound_rows))),
        ("ResultTwoComputeHours", number(sum(compute_minutes(args.results / name) for name in EXPERIMENTS) / 60.0, 1)),
        ("ResultTwoSensitivityScenarioCount", summary["sensitivity"][("v0", "P")]["scenario_count"]),
    ]
    lines = ["% Generated by experiments/report_tables_phase2.py from results/phase2; do not edit."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros]
    (args.output / "phase2_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"main": len(main_rows), "statistics": len(statistics_rows), "ablation": len(ablation_rows), "connectivity": len(connectivity_rows), "macros": len(macros)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
