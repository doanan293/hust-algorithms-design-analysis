"""Write report CSVs and result macros for the literature baseline, the RescueNet case study and the compute summary (spec D Section 13)."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

from report_tables import _read_rows, _write_semicolon, kbit, macro_name
from report_tables_phase2 import SETS, compute_minutes, number, write_points
from runner.provenance import result_provenance_problem

LITERATURE_METHODS = ("B1", "B2", "P", "TRAN")
CASE_METHODS = ("B0", "B1", "B2", "P", "TRAN")
TRACE_METHODS = ("B1", "P", "TRAN")
COMPARISON_LABELS = {"P - TRAN": "P $-$ TRAN", "B2 - TRAN": "B2 $-$ TRAN"}
COMPUTE_EXPERIMENTS = (
    ("phase1", "Pha 1: B0, B1, cận LP và MILP"),
    ("phase2/main", "So sánh chính và ablation"),
    ("phase2/budget", "Ngân sách đánh giá"),
    ("phase2/sensitivity", "Độ nhạy"),
    ("phase2/design", "Số realization thiết kế"),
    ("phase2/literature", "So sánh với TRAN"),
    ("phase2/case_study", "Case study RescueNet"),
)


def macro(*parts: str) -> str:
    return "ResultExt" + macro_name(*parts)[len("Result"):]


def km(value: str | float) -> str:
    return f"{float(value) / 1000.0:.4f}"


def provenance_problems(roots: list[Path]) -> list[str]:
    problems = []
    for root in roots:
        path = root / "manifest.json"
        if not path.exists():
            problems.append(f"missing {path}")
            continue
        problem = result_provenance_problem(json.loads(path.read_text(encoding="utf-8")))
        if problem:
            problems.append(f"{root}: {problem}")
    return problems


def edge_polyline(edges: list[dict[str, str]], failed: bool) -> list[list[str]]:
    """Edge segments separated by nan rows, so pgfplots draws them with unbounded coords=jump."""
    points: list[list[str]] = []
    for edge in edges:
        if (edge["failed"] == "1") == failed:
            points += [[km(edge["source_x_m"]), km(edge["source_y_m"])], [km(edge["target_x_m"]), km(edge["target_y_m"])], ["nan", "nan"]]
    return points or [["nan", "nan"]]


def pilot_jobs(runs: list[dict[str, str]]) -> dict[tuple[str, str, str, str], float]:
    """Planning time of each distinct pilot job; the grid reuses step-1 runs as rows with step grid."""
    jobs: dict[tuple[str, str, str, str], float] = {}
    for row in runs:
        jobs.setdefault((row["scenario_id"], row["theta"], row["mu"], row["block_slots"]), float(row["planning_runtime_s"]))
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="report-tables-extensions")
    parser.add_argument("--results", type=Path, default=Path("results/phase2"))
    parser.add_argument("--phase1", type=Path, default=Path("results/phase1"))
    parser.add_argument("--manifests", type=Path, default=Path("data/manifests"))
    parser.add_argument("--output", type=Path, default=Path("docs/report/data"))
    args = parser.parse_args(argv)

    roots = {name: args.results / name.split("/", 1)[1] if name.startswith("phase2/") else args.phase1 for name, _ in COMPUTE_EXPERIMENTS}
    problems = provenance_problems(list(roots.values()))
    if not (args.results / "tran_pilot" / "choice.json").exists():
        problems.append(f"missing {args.results / 'tran_pilot' / 'choice.json'}")
    if problems:
        print("\n".join(problems) + "\nrerun or refresh these experiments before writing report data", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    macros: list[tuple[str, str]] = []
    summary = {
        name: {(row["set_id"], row["method"]): row for row in _read_rows(args.results / name / "summary.csv")}
        for name in ("main", "literature", "case_study")
    }

    literature_rows = []
    for set_id, capacity in SETS:
        for method in LITERATURE_METHODS:
            row = summary["literature" if method == "TRAN" else "main"][(set_id, method)]
            literature_rows.append([
                set_id, kbit(capacity), method, number(row["timely_ratio_mean"]),
                f"[{number(row['timely_ratio_ci_low'])}, {number(row['timely_ratio_ci_high'])}]", number(row["gap_mean"]),
                number(row["planning_runtime_s_mean"], 1),
            ])
            for metric, column, digits in (("Timely", "timely_ratio_mean", 3), ("Gap", "gap_mean", 3), ("Planning", "planning_runtime_s_mean", 1), ("Conn", "conn_ratio_mean", 3)):
                macros.append((macro(metric, set_id, method), number(row[column], digits)))
    _write_semicolon(args.output / "ext_literature.csv", ["set", "backhaul", "method", "timely", "ci", "gap", "planning"], literature_rows)

    statistics_rows = []
    for row in _read_rows(args.results / "statistics_literature.csv"):
        statistics_rows.append([
            row["set_id"], COMPARISON_LABELS[row["comparison"]], number(row["mean_difference"]), f"[{number(row['ci_low'])}, {number(row['ci_high'])}]",
            number(row["rank_biserial"], 2), number(row["p_value"], 4), number(row["p_holm"], 4),
        ])
        first = row["comparison"].split(" - ")[0]
        for metric, column, digits in (("Diff", "mean_difference", 3), ("DiffLow", "ci_low", 3), ("DiffHigh", "ci_high", 3), ("PValue", "p_value", 4), ("Holm", "p_holm", 4), ("RankBiserial", "rank_biserial", 2)):
            macros.append((macro(metric, row["set_id"], first, "tran"), number(row[column], digits)))
    _write_semicolon(args.output / "ext_literature_statistics.csv", ["set", "comparison", "diff", "ci", "rankbiserial", "p", "pholm"], statistics_rows)

    quality = [["B1", summary["main"][("v0", "B1")]], ["TRAN", summary["literature"][("v0", "TRAN")]]]
    write_points(args.output / "ext_quality_points.csv", ["method", "planning", "timely"],
                 [[method, f"{float(row['planning_runtime_s_mean']):.3f}", f"{float(row['timely_ratio_mean']):.6f}"] for method, row in quality])
    literature_bounds = _read_rows(args.results / "literature" / "bounds.csv")
    environment = json.loads((args.results / "literature" / "manifest.json").read_text(encoding="utf-8"))["environment"]
    macros += [
        ("ResultExtLiteratureLpCount", str(len(literature_bounds))),
        ("ResultExtLiteratureLpOptimalCount", str(sum(row["status"] == "optimal" for row in literature_bounds))),
        ("ResultExtCvxpyVersion", environment["cvxpy"]),
        ("ResultExtClarabelVersion", environment["clarabel"]),
    ]

    choice = json.loads((args.results / "tran_pilot" / "choice.json").read_text(encoding="utf-8"))
    runs = _read_rows(args.results / "tran_pilot" / "runs.csv")
    grid: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in runs:
        if row["step"] == "grid":
            grid[(float(row["theta"]), float(row["mu"]))].append(float(row["timely_ratio"]))
    grid_means = [statistics.fmean(values) for values in grid.values()]
    fits = _read_rows(args.results / "tran_pilot" / "channel_fit.csv")
    macros += [
        ("ResultExtPilotTheta", number(choice["theta"])),
        ("ResultExtPilotMu", number(choice["mu"], 0)),
        ("ResultExtPilotBlockSlots", str(choice["block_slots"])),
        ("ResultExtPilotMedianRuntime", number(choice["median_runtime_step1_s"], 1)),
        ("ResultExtPilotScenarioCount", str(len(choice["scenarios"]))),
        ("ResultExtPilotBetterCount", str(choice["better_than_initial_count"])),
        ("ResultExtPilotTimely", number(choice["mean_timely_ratio"])),
        ("ResultExtPilotGridSpread", number(max(grid_means) - min(grid_means))),
        ("ResultExtFitAlphaMin", number(min(float(row["alpha"]) for row in fits), 2)),
        ("ResultExtFitAlphaMax", number(max(float(row["alpha"]) for row in fits), 2)),
        ("ResultExtFitRsquaredMin", number(min(float(row["r_squared"]) for row in fits), 4)),
    ]

    case_rows = []
    for method in CASE_METHODS:
        row = summary["case_study"][("rescuenet", method)]
        case_rows.append([
            method, number(row["timely_ratio_mean"]), f"[{number(row['timely_ratio_ci_low'])}, {number(row['timely_ratio_ci_high'])}]",
            number(row["conn_ratio_mean"]), number(row["gap_mean"]), number(row["planning_runtime_s_mean"], 1),
        ])
        for metric, column, digits in (("Timely", "timely_ratio_mean", 3), ("TimelyLow", "timely_ratio_ci_low", 3), ("TimelyHigh", "timely_ratio_ci_high", 3), ("Conn", "conn_ratio_mean", 3), ("Gap", "gap_mean", 3), ("Planning", "planning_runtime_s_mean", 1)):
            macros.append((macro("Case", metric, method), number(row[column], digits)))
    _write_semicolon(args.output / "ext_case_study.csv", ["method", "timely", "ci", "conn", "gap", "planning"], case_rows)
    case_bounds = _read_rows(args.results / "case_study" / "bounds.csv")
    macros += [
        ("ResultExtCaseLpCount", str(len(case_bounds))),
        ("ResultExtCaseLpOptimalCount", str(sum(row["status"] == "optimal" for row in case_bounds))),
        ("ResultExtCaseScenarioCount", summary["case_study"][("rescuenet", "P")]["scenario_count"]),
    ]

    targets = json.loads((args.manifests / "rescuenet_targets.json").read_text(encoding="utf-8"))
    distances = [pair["distance_m"] for pair in targets["merged_pairs"]]
    gsd_cm = [image["gsd_m_per_px"] * 100.0 for image in targets["images"]]
    fractions = {}
    for set_id in ("rescuenet", "v0"):
        rows = [row for row in _read_rows(args.manifests / f"scenarios_{set_id}.csv") if row["split"] == "eval"]
        fractions[set_id] = statistics.fmean(float(row["ground_connected_source_fraction"]) for row in rows)
    macros += [
        ("ResultExtImageCount", str(len(targets["images"]))),
        ("ResultExtImagesWithTargets", str(sum(image["targets"] > 0 for image in targets["images"]))),
        ("ResultExtRegionCount", str(targets["region_count"])),
        ("ResultExtTargetCount", str(targets["target_count"])),
        ("ResultExtDroppedCount", str(len(targets["dropped_regions"]))),
        ("ResultExtMergedCount", str(len(distances))),
        ("ResultExtMergeDistanceMin", number(min(distances), 1) if distances else "---"),
        ("ResultExtMergeDistanceMax", number(max(distances), 1) if distances else "---"),
        ("ResultExtGsdMin", number(min(gsd_cm), 2)),
        ("ResultExtGsdMax", number(max(gsd_cm), 2)),
        ("ResultExtFallbackPairs", ", ".join(image["pair_id"] for image in targets["images"] if image["focal_source"] == "fallback") or "---"),
        ("ResultExtNormalizationScale", number(targets["normalization"]["scale"])),
        ("ResultExtGroundFractionRescuenet", number(fractions["rescuenet"])),
        ("ResultExtGroundFractionVzero", number(fractions["v0"])),
    ]

    trace = args.results / "case_study" / "trace"
    trace_summary = json.loads((trace / "trace.json").read_text(encoding="utf-8"))
    scenario_id = trace_summary["scenario_id"]
    macros.append(("ResultExtTraceScenario", f"\\texttt{{{scenario_id}}}"))
    for method in TRACE_METHODS:
        macros.append((macro("Trace", method), number(trace_summary["methods"][method]["timely_ratio_trace"])))
    write_points(args.output / "ext_map_targets.csv", ["x", "y", "class"], [[km(row["x_m"]), km(row["y_m"]), row["damage_class"]] for row in _read_rows(trace / "targets.csv")])
    nodes = _read_rows(trace / "nodes.csv")
    write_points(args.output / "ext_map_relays.csv", ["x", "y"], [[km(row["x_m"]), km(row["y_m"])] for row in nodes if row["role"] == "relay"])
    write_points(args.output / "ext_map_center.csv", ["x", "y"], [[km(row["x_m"]), km(row["y_m"])] for row in nodes if row["role"] == "center"])
    edges = _read_rows(trace / "edges.csv")
    write_points(args.output / "ext_map_edges_alive.csv", ["x", "y"], edge_polyline(edges, failed=False))
    write_points(args.output / "ext_map_edges_failed.csv", ["x", "y"], edge_polyline(edges, failed=True))
    trajectories: dict[str, list[list[str]]] = defaultdict(list)
    for row in _read_rows(trace / "trajectories.csv"):
        trajectories[row["method"]].append([km(row["x_m"]), km(row["y_m"])])
    for method in TRACE_METHODS:
        write_points(args.output / f"ext_map_trajectory_{method.lower()}.csv", ["x", "y"], trajectories[method])
    alerts = int(next(row["alert_count"] for row in _read_rows(args.results / "case_study" / "method_realizations.csv") if row["scenario_id"] == scenario_id))
    progress: dict[str, dict[int, float]] = defaultdict(dict)
    for row in _read_rows(trace / "progress.csv"):
        progress[row["method"]][int(row["slot"])] = float(row["mean_timely_alerts"]) / alerts
    slots = sorted(progress[TRACE_METHODS[0]])
    write_points(args.output / "ext_progress.csv", ["slot", *TRACE_METHODS], [[slot, *(f"{progress[method][slot]:.6f}" for method in TRACE_METHODS)] for slot in slots])
    macros += [("ResultExtTraceAlertCount", str(alerts)), ("ResultExtTraceSlotCount", str(len(slots)))]

    compute_rows, total_hours = [], 0.0
    for name, label in COMPUTE_EXPERIMENTS:
        root = roots[name]
        minutes = compute_minutes(root)
        if (root / "milp_crosscheck.csv").exists():
            minutes += sum(float(row["milp_runtime_s"]) for row in _read_rows(root / "milp_crosscheck.csv")) / 60.0
        tasks = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["task_count"]
        compute_rows.append([label, str(tasks), number(minutes / 60.0, 1)])
        total_hours += minutes / 60.0
    jobs = pilot_jobs(runs)
    pilot_hours = sum(jobs.values()) / 3600.0
    compute_rows.insert(5, ["Pilot TRAN trên tập phát triển", str(len(jobs)), number(pilot_hours, 1)])
    total_hours += pilot_hours
    _write_semicolon(args.output / "ext_compute.csv", ["experiment", "tasks", "hours"], compute_rows)
    macros.append(("ResultExtComputeHours", number(total_hours, 1)))

    lines = ["% Generated by experiments/report_tables_extensions.py from results/phase1, results/phase2 and data/manifests; do not edit."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros]
    (args.output / "ext_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"literature": len(literature_rows), "statistics": len(statistics_rows), "case_study": len(case_rows), "compute": len(compute_rows), "macros": len(macros)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
