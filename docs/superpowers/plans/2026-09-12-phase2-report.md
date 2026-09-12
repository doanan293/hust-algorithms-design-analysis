# Phase 2 Report Implementation Plan (Sub-Project C, Plan C.3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `docs/report` to its Phase 2 state (spec C Section 14 and acceptance criterion 17.2.6): generate report data from the committed Phase 2 results, describe the search methods and the literature review, report the experiments with answers to RQ1–RQ5, and update the abstract, introduction, discussion, conclusion, and README.

**Architecture:** `experiments/report_tables_phase2.py`, modelled on `experiments/report_tables.py`, reads `results/phase2/{main,budget,sensitivity,design}` and `results/phase2/statistics.csv` and writes semicolon-separated CSV files for `csvsimple` tables, comma-separated CSV files for `pgfplots`, and `\ResultTwo...` macros into `docs/report/data/`. The sections read those files, so every number in the text comes from the committed results. The report is built with the `latex-document-skill` compile script, and a short check scans the final pdfLaTeX pass for warnings.

**Tech Stack:** Python 3.12 in the uv-managed `.venv` (`numpy`, `scipy` through `analysis.statistics`); pdfLaTeX with biber, `csvsimple`, `pgfplots` (with its `groupplots` library), `siunitx`, `cleveref`, `algorithmicx`, Vietnamese through T5 and babel; `.claude/skills/latex-document-skill/scripts/compile_latex.sh`; `pdfinfo` and `pdftotext` from poppler.

**Spec:** `docs/superpowers/specs/2026-09-11-phase2-core-design.md` (Sections 14, 16.2, 17.2.6, and Section 18 "Plan C.2 adjustments", which created this plan). Inputs: `results/phase2/` (plans C.1 and C.2), `docs/literature/review.md`, `docs/report/data/related_work.csv`, `docs/references/sensors-25-07443-v2.md`.

## Global Constraints

- Run project code through uv: tests with `uv run --extra test pytest`, scripts with `uv run python ...`. No Python dependency and no LaTeX package is added; `groupplots` is a library that ships with `pgfplots`.
- Follow `latex-document-skill`: build with its `compile_latex.sh`; tables read CSV with `csvsimple` (`separator=semicolon`); charts read CSV with `pgfplots`; prose paragraphs rather than bullet lists; figures at most `0.85\textwidth` with `[htbp]` placement.
- Numbers in prose come from `docs/report/data/phase1_results.tex` and `docs/report/data/phase2_results.tex`; Phase 2 macro names start with `ResultTwo` so they never clash with Phase 1 macros.
- `experiments/report_tables_phase2.py` refuses results whose manifest is not `complete` or whose `source_hash` differs from the current source, and writes nothing in that case.
- Statements about papers come only from `docs/literature/review.md`; statements about results only from the committed results. The text states no mechanism that the experiments did not measure.
- After each report task the final pdfLaTeX pass has no `Overfull`, `Underfull`, `LaTeX Warning`, `Package ... Warning`, or error lines, and the PDF text contains no `??`.
- `docs/report/build/` and `docs/report/main.pdf` stay Git-ignored; commit sources and the generated data under `docs/report/data/`.
- Test module basenames are unique across `tests/`; the generator follows test-driven development; every task ends with its own commit.

---
### Task 1: Report data from the Phase 2 results

**Files:**
- Create: `experiments/report_tables_phase2.py`
- Test: `tests/runner/test_report_tables_phase2.py`
- Create (generated): `docs/report/data/phase2_main.csv`, `phase2_statistics.csv`, `phase2_ablation.csv`, `phase2_connectivity.csv`, `phase2_design.csv`, `phase2_budget_b2.csv`, `phase2_budget_p.csv`, `phase2_sensitivity_{lref,backhaul,deadline,btot,alerts,paths}.csv`, `phase2_results.tex`

**Interfaces:**
- Consumes: `analysis.statistics.{BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, bootstrap_mean_ci, topology_means}`; `report_tables.{_read_rows, _write_semicolon, kbit, macro_name, num}` (the script runs with `experiments/` on `sys.path`); `runner.provenance.source_hash`; `results/phase2/{main,budget,sensitivity,design}/{summary.csv, method_realizations.csv, bounds.csv, manifest.json}`, `results/phase2/statistics.csv`, `data/manifests/scenarios_{v0,v0-bh50}.csv`.
- Produces: `uv run python experiments/report_tables_phase2.py [--results results/phase2] [--manifests data/manifests] [--output docs/report/data]` (exit 1 on missing, incomplete, or stale results); semicolon tables `phase2_main.csv` (`set;backhaul;method;timely;ci;conn;gap;uniongap;planning;calls`), `phase2_statistics.csv` (`set;comparison;diff;ci;rankbiserial;p;pholm`), `phase2_ablation.csv` (`variant;timely;diff;timelybh;diffbh;calls`), `phase2_connectivity.csv` (`set;group;scenarios;conn;b0;b1;b2;gain`), `phase2_design.csv` (`design;timely;planning;calls`); comma files `phase2_budget_{b2,p}.csv` (`budget,timely,calls,planning`) and `phase2_sensitivity_<panel>.csv` (`x,B1,B2,P`); macros `\ResultTwo{Timely,TimelyLow,TimelyHigh,Conn,Gap,UnionGap,UnionBound,Planning,Calls}{Vzero,VzeroBhfivezero}{Bzero,Bone,Btwo,Bthree,P}`, `\ResultTwo{Higher,Equal,Lower}<set>{BtwoBzero,BtwoBone,PBtwo,BoneBzero}`, `\ResultTwo{Diff,DiffLow,DiffHigh,Holm,RankBiserial}<set>{PBtwo,BtwoBone,BtwoBthree,PBone}`, `\ResultTwoHolmFloor`, `\ResultTwo{AblationDiff,AblationTimely}<set>{POpone,POptwo,PNoBacklog,PFixedPaths,PFixedTrajectory,PEqualBandwidth,PExpectedDesign}`, `\ResultTwoTercileGain<set>{Low,Middle,High}`, `\ResultTwoGroundCorrelation<set>`, `\ResultTwo{Budget,BudgetCalls}{Btwo,P}{Twofivezero,Fivezerozero,Onezerozerozero,Twozerozerozero,Fourzerozerozero}`, `\ResultTwoSens<set><method>` for every sensitivity set, `\ResultTwo{Design,DesignPlanning}{Expected,One,Five,Onezero}`, `\ResultTwoLpCount`, `\ResultTwoLpOptimalCount`, `\ResultTwoComputeHours`, `\ResultTwoScenarioCount`, `\ResultTwoSensitivityScenarioCount`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/runner/test_report_tables_phase2.py
import csv
import importlib.util
import json
from pathlib import Path
import sys

from runner.provenance import source_hash

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
SPEC = importlib.util.spec_from_file_location("report_tables_phase2", EXPERIMENTS / "report_tables_phase2.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

TOPOLOGIES = [f"T{index}" for index in range(10)]
MAIN = {"B0": 0.25, "B1": 0.375, "B2": 0.5, "B3": 0.125, "P": 0.5, "P_op1": 0.5, "P_op2": 0.5, "P_no_backlog": 0.5,
        "P_fixed_paths": 0.375, "P_fixed_trajectory": 0.375, "P_equal_bandwidth": 0.5, "P_expected_design": 0.4375}
SENSITIVITY_SETS = ["v0", "v0-bh50", "sens-bh100", "sens-lref125k", "sens-lref500k", "sens-lref1m", "sens-physical", "sens-deadline-short",
                    "sens-deadline-long", "sens-btot05", "sens-btot2", "sens-alerts20", "sens-alerts60", "sens-k2", "sens-k5"]
SUMMARY_COLUMNS = ["set_id", "method", "scenario_count", "timely_ratio_mean", "timely_ratio_ci_low", "timely_ratio_ci_high", "conn_ratio_mean",
                   "gap_mean", "union_gap_mean", "union_bound_ratio_mean", "planning_runtime_s_mean", "evaluate_calls_mean"]
ROW_COLUMNS = ["set_id", "scenario_id", "topology_id", "method", "realization_id", "timely_ratio", "conn_ratio", "planning_runtime_s", "evaluation_runtime_s"]


def _write(path: Path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _experiment(root: Path, name: str, values: dict, source: str):
    folder = root / "phase2" / name
    summary, rows = [], []
    for (set_id, method), timely in values.items():
        summary.append({"set_id": set_id, "method": method, "scenario_count": "10", "timely_ratio_mean": timely, "timely_ratio_ci_low": timely - 0.05,
                        "timely_ratio_ci_high": timely + 0.05, "conn_ratio_mean": 0.5 if method == "B0" else 1.0, "gap_mean": 0.2,
                        "union_gap_mean": 0.25, "union_bound_ratio_mean": 0.7, "planning_runtime_s_mean": 60.0, "evaluate_calls_mean": 2001})
        for topology in TOPOLOGIES:
            rows.append({"set_id": set_id, "scenario_id": f"{topology}-r0", "topology_id": topology, "method": method, "realization_id": "0",
                         "timely_ratio": timely, "conn_ratio": 0.5 if method == "B0" else 1.0, "planning_runtime_s": "60", "evaluation_runtime_s": "0.5"})
    _write(folder / "summary.csv", SUMMARY_COLUMNS, summary)
    _write(folder / "method_realizations.csv", ROW_COLUMNS, rows)
    _write(folder / "bounds.csv", ["set_id", "status", "runtime_s"], [{"set_id": "v0", "status": "optimal", "runtime_s": "120"}])
    (folder / "manifest.json").write_text(json.dumps({"status": "complete", "source_hash": source}), encoding="utf-8")


def _results(root: Path, source: str) -> tuple[Path, Path]:
    sets = ("v0", "v0-bh50")
    _experiment(root, "main", {(set_id, method): value for set_id in sets for method, value in MAIN.items()}, source)
    _experiment(root, "budget", {("v0", f"{method}_b{budget}"): 0.4 + budget / 100000 for method in ("B2", "P") for budget in (250, 500, 1000, 2000, 4000)}, source)
    _experiment(root, "sensitivity", {(set_id, method): {"B1": 0.3, "B2": 0.4, "P": 0.45}[method] for set_id in SENSITIVITY_SETS for method in ("B1", "B2", "P")}, source)
    _experiment(root, "design", {("v0", method): value for method, value in (("P_design_expected", 0.35), ("P_design1", 0.375), ("P_design10", 0.5))}, source)
    _write(root / "phase2" / "statistics.csv", ["set_id", "comparison", "mean_difference", "ci_low", "ci_high", "rank_biserial", "p_value", "p_holm", "topology_count"],
           [{"set_id": set_id, "comparison": label, "mean_difference": 0.125, "ci_low": 0.1, "ci_high": 0.15, "rank_biserial": 1.0, "p_value": 2 / 1024,
             "p_holm": 16 / 1024, "topology_count": 10} for set_id in sets for label in ("P - B2", "B2 - B1", "B2 - B3", "P - B1")])
    manifests = root / "manifests"
    fractions = [0.2, 0.2, 0.333333, 0.5, 0.5, 0.5, 0.6, 0.666667, 0.8, 0.8]
    for set_id in sets:
        rows = [{"scenario_id": f"{topology}-r0", "split": "eval", "ground_connected_source_fraction": fraction} for topology, fraction in zip(TOPOLOGIES, fractions)]
        rows.append({"scenario_id": "D0-r0", "split": "dev", "ground_connected_source_fraction": 0.1})
        _write(manifests / f"scenarios_{set_id}.csv", ["scenario_id", "split", "ground_connected_source_fraction"], rows)
    return root / "phase2", manifests


def _semicolon(path: Path) -> list[list[str]]:
    return [line.split(";") for line in path.read_text(encoding="utf-8").splitlines()]


def test_phase2_report_files_follow_the_results(tmp_path: Path):
    results, manifests = _results(tmp_path, source_hash())
    output = tmp_path / "data"
    assert report.main(["--results", str(results), "--manifests", str(manifests), "--output", str(output)]) == 0
    main = _semicolon(output / "phase2_main.csv")
    assert main[0] == ["set", "backhaul", "method", "timely", "ci", "conn", "gap", "uniongap", "planning", "calls"] and len(main) == 11
    assert main[3][:6] == ["v0", "\\qty{1000}{\\kilo\\bit\\per\\second}", "B2", "\\num{0.500}", "[\\num{0.450}, \\num{0.550}]", "\\num{1.000}"]
    assert main[6][0:3] == ["v0-bh50", "\\qty{50}{\\kilo\\bit\\per\\second}", "B0"]
    statistics_rows = _semicolon(output / "phase2_statistics.csv")
    assert statistics_rows[1] == ["v0", "P $-$ B2", "\\num{0.125}", "[\\num{0.100}, \\num{0.150}]", "\\num{1.00}", "\\num{0.0020}", "\\num{0.0156}"]
    ablation = _semicolon(output / "phase2_ablation.csv")
    assert [row[0] for row in ablation[1:]] == ["P đầy đủ", "Chỉ thao tác 1", "Chỉ thao tác 2", "Sửa lịch không dùng backlog", "Không chạy khối đường",
                                                 "Không chạy khối quỹ đạo", "Không chạy khối băng thông", "Thiết kế trên tốc độ kỳ vọng"]
    assert ablation[2][2] == "\\num{0.000} [\\num{0.000}, \\num{0.000}]"
    assert ablation[5][1:5] == ["\\num{0.375}", "\\num{-0.125} [\\num{-0.125}, \\num{-0.125}]", "\\num{0.375}", "\\num{-0.125} [\\num{-0.125}, \\num{-0.125}]"]
    connectivity = _semicolon(output / "phase2_connectivity.csv")
    assert [row[1:3] for row in connectivity[1:4]] == [["Tối đa 1/3", "3"], ["Từ 1/3 đến 2/3", "4"], ["Từ 2/3", "3"]]
    assert connectivity[1][3:] == ["\\num{0.500}", "\\num{0.250}", "\\num{0.375}", "\\num{0.500}", "\\num{0.250}"]
    assert (output / "phase2_budget_b2.csv").read_text(encoding="utf-8").splitlines()[:2] == ["budget,timely,calls,planning", "250,0.402500,2001.0,60.000"]
    assert (output / "phase2_sensitivity_lref.csv").read_text(encoding="utf-8").splitlines() == [
        "x,B1,B2,P", "125,0.300000,0.400000,0.450000", "250,0.300000,0.400000,0.450000", "500,0.300000,0.400000,0.450000", "1000,0.300000,0.400000,0.450000"]
    assert [row[0] for row in _semicolon(output / "phase2_design.csv")[1:]] == ["Tốc độ kỳ vọng", "1", "5", "10"]
    macros = (output / "phase2_results.tex").read_text(encoding="utf-8")
    for expected in (
        "\\newcommand{\\ResultTwoTimelyVzeroBtwo}{\\num{0.500}}", "\\newcommand{\\ResultTwoHigherVzeroBtwoBzero}{10}",
        "\\newcommand{\\ResultTwoEqualVzeroPBtwo}{10}", "\\newcommand{\\ResultTwoHolmFloor}{\\num{0.0156}}",
        "\\newcommand{\\ResultTwoAblationDiffVzeroPFixedPaths}{\\num{-0.125}}", "\\newcommand{\\ResultTwoTercileGainVzeroLow}{\\num{0.250}}",
        "\\newcommand{\\ResultTwoBudgetPFourzerozerozero}{\\num{0.440}}", "\\newcommand{\\ResultTwoSensSensLrefonemP}{\\num{0.450}}",
        "\\newcommand{\\ResultTwoDesignFive}{\\num{0.450}}", "\\newcommand{\\ResultTwoDesignOnezero}{\\num{0.500}}",
        "\\newcommand{\\ResultTwoGroundCorrelationVzero}{---}", "\\newcommand{\\ResultTwoLpCount}{2}", "\\newcommand{\\ResultTwoLpOptimalCount}{2}", "\\newcommand{\\ResultTwoScenarioCount}{10}",
    ):
        assert expected in macros, expected
    assert "\\ResultTwoTimelyVzeroBone}" in macros and "\\ResultTimelyVzeroBone}" not in macros


def test_phase2_report_tables_refuse_results_from_other_source_code(tmp_path: Path, capsys):
    results, manifests = _results(tmp_path, "0" * 64)
    assert report.main(["--results", str(results), "--manifests", str(manifests), "--output", str(tmp_path / "data")]) == 1
    assert "different source code" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()
```

The fixture gives every method the same value on all ten topologies, so differences, intervals, counts, and terciles have exact expected values; the constant B0 values also exercise the undefined-correlation branch.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/runner/test_report_tables_phase2.py -v`

Expected: collection fails with `FileNotFoundError` for `experiments/report_tables_phase2.py`.

- [ ] **Step 3: Implement the generator**

```python
# experiments/report_tables_phase2.py
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
```

Design-realization point 5 is P in the sensitivity experiment on `v0`, which runs the same ten scenarios and 30 realizations as the design experiment. A number that rounds to zero is written without a sign, so the tables never show `-0.000`.

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/runner/test_report_tables_phase2.py -v`

Expected: 2 passed.

Run: `uv run --extra test pytest -q`

Expected: 209 passed, 1 skipped.

- [ ] **Step 5: Generate the report data**

Run: `uv run python experiments/report_tables_phase2.py`

Expected: `{"main": 10, "statistics": 8, "ablation": 8, "connectivity": 6, "macros": 269}`.

The results are deterministic, so the small tables must match exactly:

`docs/report/data/phase2_main.csv`:

```text
set;backhaul;method;timely;ci;conn;gap;uniongap;planning;calls
v0;\qty{1000}{\kilo\bit\per\second};B0;\num{0.413};[\num{0.319}, \num{0.487}];\num{0.509};\num{0.171};\num{0.438};\num{0.0};\num{1}
v0;\qty{1000}{\kilo\bit\per\second};B1;\num{0.494};[\num{0.391}, \num{0.582}];\num{1.000};\num{0.238};\num{0.318};\num{0.1};\num{1}
v0;\qty{1000}{\kilo\bit\per\second};B2;\num{0.551};[\num{0.444}, \num{0.641}];\num{1.000};\num{0.210};\num{0.229};\num{150.1};\num{1933}
v0;\qty{1000}{\kilo\bit\per\second};B3;\num{0.367};[\num{0.274}, \num{0.451}];\num{0.999};\num{0.478};\num{0.497};\num{175.0};\num{2001}
v0;\qty{1000}{\kilo\bit\per\second};P;\num{0.553};[\num{0.445}, \num{0.644}];\num{1.000};\num{0.208};\num{0.227};\num{155.6};\num{1973}
v0-bh50;\qty{50}{\kilo\bit\per\second};B0;\num{0.370};[\num{0.283}, \num{0.437}];\num{0.509};\num{0.229};\num{0.467};\num{0.0};\num{1}
v0-bh50;\qty{50}{\kilo\bit\per\second};B1;\num{0.430};[\num{0.344}, \num{0.500}];\num{1.000};\num{0.287};\num{0.371};\num{0.1};\num{1}
v0-bh50;\qty{50}{\kilo\bit\per\second};B2;\num{0.509};[\num{0.417}, \num{0.585}];\num{1.000};\num{0.227};\num{0.245};\num{138.6};\num{1922}
v0-bh50;\qty{50}{\kilo\bit\per\second};B3;\num{0.331};[\num{0.270}, \num{0.382}];\num{1.000};\num{0.499};\num{0.512};\num{160.5};\num{2001}
v0-bh50;\qty{50}{\kilo\bit\per\second};P;\num{0.507};[\num{0.415}, \num{0.584}];\num{1.000};\num{0.228};\num{0.248};\num{138.8};\num{1901}
```

`docs/report/data/phase2_statistics.csv`:

```text
set;comparison;diff;ci;rankbiserial;p;pholm
v0;P $-$ B2;\num{0.002};[\num{-0.003}, \num{0.007}];\num{0.11};\num{0.5039};\num{1.0000}
v0;B2 $-$ B1;\num{0.057};[\num{0.043}, \num{0.075}];\num{1.00};\num{0.0020};\num{0.0156}
v0;B2 $-$ B3;\num{0.184};[\num{0.144}, \num{0.224}];\num{1.00};\num{0.0020};\num{0.0156}
v0;P $-$ B1;\num{0.059};[\num{0.043}, \num{0.079}];\num{1.00};\num{0.0020};\num{0.0156}
v0-bh50;P $-$ B2;\num{-0.002};[\num{-0.007}, \num{0.002}];\num{-0.38};\num{0.5312};\num{1.0000}
v0-bh50;B2 $-$ B1;\num{0.079};[\num{0.058}, \num{0.101}];\num{1.00};\num{0.0020};\num{0.0156}
v0-bh50;B2 $-$ B3;\num{0.179};[\num{0.137}, \num{0.214}];\num{1.00};\num{0.0020};\num{0.0156}
v0-bh50;P $-$ B1;\num{0.077};[\num{0.059}, \num{0.097}];\num{1.00};\num{0.0020};\num{0.0156}
```

`docs/report/data/phase2_ablation.csv`:

```text
variant;timely;diff;timelybh;diffbh;calls
P đầy đủ;\num{0.553};---;\num{0.507};---;\num{1973}
Chỉ thao tác 1;\num{0.553};\num{0.000} [\num{-0.006}, \num{0.008}];\num{0.510};\num{0.002} [\num{-0.002}, \num{0.007}];\num{1961}
Chỉ thao tác 2;\num{0.553};\num{0.000} [\num{-0.002}, \num{0.003}];\num{0.508};\num{0.000} [\num{-0.003}, \num{0.002}];\num{1961}
Sửa lịch không dùng backlog;\num{0.553};\num{-0.001} [\num{-0.002}, \num{0.001}];\num{0.509};\num{0.001} [\num{0.000}, \num{0.002}];\num{1982}
Không chạy khối đường;\num{0.504};\num{-0.050} [\num{-0.072}, \num{-0.032}];\num{0.443};\num{-0.065} [\num{-0.084}, \num{-0.048}];\num{597}
Không chạy khối quỹ đạo;\num{0.505};\num{-0.048} [\num{-0.069}, \num{-0.032}];\num{0.459};\num{-0.048} [\num{-0.063}, \num{-0.037}];\num{1648}
Không chạy khối băng thông;\num{0.555};\num{0.001} [\num{-0.007}, \num{0.011}];\num{0.510};\num{0.003} [\num{-0.003}, \num{0.010}];\num{958}
Thiết kế trên tốc độ kỳ vọng;\num{0.518};\num{-0.035} [\num{-0.046}, \num{-0.027}];\num{0.481};\num{-0.027} [\num{-0.031}, \num{-0.022}];\num{1921}
```

`docs/report/data/phase2_connectivity.csv`:

```text
set;group;scenarios;conn;b0;b1;b2;gain
v0;Tối đa 1/3;10;\num{0.260};\num{0.230};\num{0.276};\num{0.358};\num{0.128}
v0;Từ 1/3 đến 2/3;11;\num{0.527};\num{0.420};\num{0.514};\num{0.557};\num{0.137}
v0;Từ 2/3;9;\num{0.763};\num{0.608};\num{0.713};\num{0.760};\num{0.152}
v0-bh50;Tối đa 1/3;10;\num{0.260};\num{0.210};\num{0.260};\num{0.348};\num{0.138}
v0-bh50;Từ 1/3 đến 2/3;11;\num{0.527};\num{0.398};\num{0.456};\num{0.515};\num{0.117}
v0-bh50;Từ 2/3;9;\num{0.763};\num{0.514};\num{0.588};\num{0.682};\num{0.168}
```

`docs/report/data/phase2_design.csv`:

```text
design;timely;planning;calls
Tốc độ kỳ vọng;\num{0.467};\num{26.6};\num{1916}
1;\num{0.476};\num{29.2};\num{1989}
5;\num{0.506};\num{146.6};\num{2001}
10;\num{0.514};\num{236.9};\num{1982}
```

- [ ] **Step 6: Commit**

```bash
git add experiments/report_tables_phase2.py tests/runner/test_report_tables_phase2.py docs/report/data/phase2_*.csv docs/report/data/phase2_results.tex
git commit -m "feat: generate report data from phase 2 results"
```

### Task 2: Method section for the search methods

**Files:**
- Modify: `docs/report/sections/method.tex` (the planned Phase 2 algorithm subsection is replaced)

**Interfaces:**
- Consumes: existing labels `sec:bound`, `eq:objective`; citation `huang2025ddatsap`; spec C Sections 3 and 5–10.
- Produces: labels `sec:algorithm`, `eq:weights`, `eq:leximin`, `alg:bcd-repair`, used by Tasks 3–5.

- [ ] **Step 1: Replace the planned algorithm with the implemented methods**

In `docs/report/sections/method.tex` (1 edit):

Edit 1: replace

```latex

\subsection{Phương pháp đề xuất cho Pha 2}
\label{sec:algorithm}

\Cref{alg:bcd-repair} là phương pháp dự kiến cho Pha 2: phân rã theo ba khối như cấu trúc BCD của~\cite{huang2025ddatsap}, sau đó áp dụng hai thao tác sửa lịch cho các cảnh báo có độ dư thời gian thấp. Phương pháp này chưa được cài đặt và chưa được đánh giá trong Pha 1.

\begin{algorithm}[htbp]
\caption{Phân rã theo khối kết hợp sửa lịch theo độ dư thời gian (Pha 2, chưa đánh giá)}
\label{alg:bcd-repair}
\begin{algorithmic}[1]
\Require Scenario, tập candidate, số vòng tối đa $I_{\max}$
\Ensure Quỹ đạo $q$, lựa chọn đường $x$, băng thông $b$
\State Khởi tạo quỹ đạo khả thi $q^{(0)}$ và lựa chọn đường $x^{(0)}$
\For{$i = 1, \dots, I_{\max}$}
    \State Cập nhật $b$ với $(q, x)$ cố định
    \State Cập nhật $x$ với $(q, b)$ cố định \Comment{hoán đổi trong tập candidate}
    \State Cập nhật $q$ với $(x, b)$ cố định \Comment{tìm kiếm cục bộ}
    \State Tính độ dư thời gian của từng cảnh báo; chọn tập nguy cơ $A_{\mathrm{risk}}$
    \ForAll{$a \in A_{\mathrm{risk}}$}
        \State Thử chuyển băng thông từ luồng còn dư sang $a$
        \State Thử đổi đường của $a$ trong tập candidate
        \State Nhận thay đổi nếu mục tiêu từ điển không giảm
    \EndFor
    \If{không cải thiện trong $\kappa$ vòng liên tiếp} \textbf{break} \EndIf
\EndFor
\State \Return $(q, x, b)$
\end{algorithmic}
\end{algorithm}
```

with

```latex

\subsection{Phương pháp tìm kiếm theo khối và sửa lịch}
\label{sec:algorithm}

\paragraph{Vì sao các khối không là bài toán lồi.}
Cấu trúc ba khối của~\cite{huang2025ddatsap} được giữ lại, nhưng mỗi khối không được giải bằng một bài toán lồi. Ba thử nghiệm sơ bộ trên sáu scenario đánh giá với mười realization cho thấy lý do. Băng thông lấy từ nghiệm LP ở \cref{sec:bound}, dùng làm lịch tĩnh, làm tỷ lệ đúng hạn giảm từ \num{0.410} của B1 xuống \num{0.128}; dùng làm trọng số chia băng thông đạt \num{0.404}; đường chọn theo nghiệm LP đạt \num{0.412}. Như vậy chỉ dẫn từ LP không chuyển được sang bộ điều phối EDF, còn khối quỹ đạo dưới kênh PrLoS không có hàm thay thế lồi hợp lệ cho mục tiêu đếm cảnh báo. Ngược lại, tìm kiếm cục bộ được chấm điểm bằng chính bộ đánh giá nâng tỷ lệ này lên \num{0.427} khi chấm trên tốc độ kỳ vọng và \num{0.449} khi chấm trên năm realization lấy mẫu; chấm trên tốc độ kỳ vọng còn làm một scenario giảm từ \num{0.372} xuống \num{0.305}. Vì vậy mỗi khối ở đây là một bước tìm kiếm cục bộ có đánh giá.

\paragraph{Chia băng thông theo trọng số.}
Kế hoạch mang trọng số $w_{e,n}\ge 0$ cho mỗi liên kết vô tuyến $e$ và slot $n$. Trong mỗi nửa slot, gọi $\mathcal B_n$ là tập liên kết truy nhập có dữ liệu chờ, hoặc tập tối đa hai downlink có hạn đầu hàng đợi sớm nhất; băng thông được cấp là
\begin{equation}
\label{eq:weights}
b_{e,n}=B_{\mathrm{tot}}\frac{w_{e,n}}{\sum_{e'\in\mathcal B_n}w_{e',n}},\qquad e\in\mathcal B_n,
\end{equation}
\noindent và chia đều khi tổng trọng số bằng $0$. Quy tắc chia đều của B0, B1 là trường hợp mọi trọng số bằng $1$. Trọng số cố định trước nhiệm vụ, còn băng thông thực tế chỉ phụ thuộc hàng đợi hiện tại, nên chính sách không dùng thông tin tương lai.

\paragraph{Chấm điểm thiết kế.}
Phương pháp tìm kiếm không bao giờ dùng các realization đánh giá. Nó chấm một kế hoạch trên $M_d=5$ realization thiết kế, lấy từ một luồng số ngẫu nhiên khác với luồng đánh giá, theo khoá từ điển \eqref{eq:objective} cộng trên các realization đó. Mỗi lần chấm là một lần đánh giá, và tổng số lần đánh giá bị giới hạn bởi ngân sách $E=2000$; khi hết ngân sách, phương pháp trả về kế hoạch tốt nhất đang có. Một thay đổi chỉ được nhận khi khoá tăng chặt, nên khoá của kế hoạch hiện tại không bao giờ giảm. Khi chấm thiết kế, bộ kiểm tra độc lập được bỏ qua và $\mathrm{Conn}$ được ghi nhớ theo quỹ đạo và realization, vì cả hai không ảnh hưởng tới quyết định tìm kiếm; cách này giữ nguyên kế hoạch trả về và nhanh hơn khoảng \num{2.9} lần. Lần đánh giá cuối trên các realization đánh giá vẫn chạy bộ kiểm tra.

\paragraph{Ba khối của B2.}
B2 khởi tạo từ kế hoạch B1 với mọi trọng số bằng $1$ và lặp tối đa mười vòng. \emph{Khối đường} duyệt cảnh báo theo hạn tăng dần và thử lần lượt các lựa chọn không phục vụ và từng candidate. \emph{Khối băng thông} duyệt các liên kết vô tuyến đang được dùng theo tổng kích thước cảnh báo đi qua, giảm dần, và thử nhân toàn bộ trọng số của liên kết với $2$, rồi với $1/2$ nếu phép nhân đôi bị từ chối. \emph{Khối quỹ đạo} duyệt một họ hành trình khả thi theo cấu tạo: UAV rời trung tâm, bay thẳng với tốc độ $V_{\max}$ qua một tập con có thứ tự của các vùng hư hại, lơ lửng tại mỗi điểm dừng rồi quay về. Thời gian còn dư được chia cho các điểm dừng đều nhau, theo số cảnh báo của vùng, hoặc theo tổng $1/(d_a-r_a)$ của vùng; điểm dừng là tâm vùng hoặc trọng tâm các nguồn trong vùng. Với ba vùng, họ này có tối đa $90$ hành trình và chứa quỹ đạo của B1. Vòng lặp dừng khi một vòng không nhận thay đổi nào.

\paragraph{B3: mục tiêu leximin.}
B3 giống B2 nhưng dùng khoá khác. Với $\bar f_a$ là tỷ lệ bit của $a$ tới trung tâm trước hạn, lấy trung bình trên các realization thiết kế, tốc độ giao trung bình của $a$ chia cho tốc độ yêu cầu $L_a/((d_a-r_a)\Delta)$ bằng đúng $\bar f_a$, nên đây là phiên bản chuẩn hoá của mục tiêu max--min của~\cite{huang2025ddatsap}. Vì chỉ một cảnh báo không thể giao cũng làm $\min_a\bar f_a=0$ với mọi kế hoạch, B3 so sánh theo leximin:
\begin{equation}
\label{eq:leximin}
\bigl(\bar f_{(1)},\bar f_{(2)},\dots,\bar f_{(\abs{A})},\ \textstyle\sum_{\omega}\sum_a z_{a,\omega},\ \mathrm{Conn}\bigr),\qquad \bar f_{(1)}\le\bar f_{(2)}\le\dots\le\bar f_{(\abs{A})},
\end{equation}
\noindent trong đó cảnh báo không được phục vụ có $\bar f_a=0$. Chênh lệch giữa B2 và B3 đo cái giá của việc tối ưu công bằng thay cho số cảnh báo đúng hạn.

\paragraph{P: sửa lịch theo nguy cơ trễ hạn.}
P là B2 thêm một vòng sửa lịch sau khối quỹ đạo của mỗi vòng lặp. Một lần đánh giá có ghi sổ trên kế hoạch hiện tại cho biết, với mỗi cảnh báo và realization thiết kế, độ dư thời gian $d_a-1-t_a$ (bằng $-1$ nếu trễ, với $t_a$ là slot giao xong), chặng giữ dữ liệu của cảnh báo trong nhiều slot nhất và các liên kết đã dùng hết dung lượng. Nguy cơ của một cảnh báo là tỷ lệ realization thiết kế trong đó nó trễ hoặc có độ dư không quá $\theta=2$ slot; tối đa $R=10$ cảnh báo có nguy cơ cao nhất được xét. \emph{Thao tác 1} nhân đôi trọng số của chặng thắt cổ chai, nếu đó là liên kết vô tuyến, chỉ trong các slot $[r_a,d_a)$, nên băng thông được chuyển từ các liên kết khác cùng nửa slot sang cảnh báo này. \emph{Thao tác 2} thử các candidate khác của cảnh báo theo số liên kết nghẽn trong cửa sổ của nó, rồi theo chi phí ước lượng, và nhận candidate đầu tiên làm khoá tăng. Mọi thao tác dùng chung ngân sách với ba khối, nên P và B2 có cùng số lần đánh giá tối đa. \Cref{alg:bcd-repair} tóm tắt phương pháp.

\begin{algorithm}[htbp]
\caption{Tìm kiếm theo khối có đánh giá, kết hợp sửa lịch (B2 khi tắt sửa lịch, P khi bật)}
\label{alg:bcd-repair}
\begin{algorithmic}[1]
\Require Scenario, tập candidate, realization thiết kế, ngân sách $E$, số vòng tối đa $I_{\max}=10$
\Ensure Quỹ đạo $q$, lựa chọn đường $x$, trọng số băng thông $w$
\State $(q,x)\gets$ kế hoạch B1; $w\gets 1$; $\kappa\gets$ khoá của $(q,x,w)$
\For{$i=1,\dots,I_{\max}$}
    \State Khối đường: với từng cảnh báo theo hạn, thử không phục vụ và từng candidate
    \State Khối băng thông: với từng liên kết đang dùng, thử $w_e\gets 2w_e$, sau đó $w_e\gets w_e/2$
    \State Khối quỹ đạo: thử từng hành trình trong họ hành trình khả thi
    \If{sửa lịch được bật}
        \State Đánh giá có ghi sổ; chọn tối đa $R$ cảnh báo có nguy cơ trễ hạn cao nhất
        \ForAll{cảnh báo $a$ có nguy cơ}
            \State Thao tác 1: nhân đôi trọng số chặng thắt cổ chai trong $[r_a,d_a)$
            \State Thao tác 2: thử các candidate khác theo số liên kết nghẽn, rồi chi phí
        \EndFor
    \EndIf
    \State Mỗi phép thử được nhận khi khoá tăng chặt; dừng ngay khi hết ngân sách $E$
    \If{vòng này không nhận thay đổi nào} \textbf{break} \EndIf
\EndFor
\State \Return $(q,x,w)$
\end{algorithmic}
\end{algorithm}

\paragraph{Độ phức tạp.}
Một vòng lặp dùng tối đa $(K+1)\abs{A}+2\abs{\mathcal L_{\mathrm{used}}}+90$ lần đánh giá cho ba khối, cộng $1+RK$ lần cho vòng sửa lịch của P, với $\mathcal L_{\mathrm{used}}$ là các liên kết vô tuyến đang được dùng. Mỗi lần đánh giá tốn $O\bigl(M_dN(\abs{\mathcal L}+\abs{A}\log\abs{A})\bigr)$, còn phát hiện nguy cơ duyệt sổ ghi trong $O\bigl(M_d(N\abs{\mathcal L}+T)\bigr)$ với $T$ là số bản ghi truyền. Ngân sách chặn tổng số lần đánh giá, nên thời gian lập kế hoạch của B2, B3 và P là $O\bigl(EM_dN(\abs{\mathcal L}+\abs{A}\log\abs{A})\bigr)$, không phụ thuộc số vòng lặp thực tế. Phương pháp chỉ bảo đảm khoá của kế hoạch không giảm; nó không bảo đảm tối ưu, và cận LP ở \cref{sec:bound}, giải trên chính quỹ đạo mà mỗi phương pháp chọn, đo phần còn có thể cải thiện.
```

The numbers in the first paragraph are the spike results of spec C Section 3 (mean timely ratio on six scenarios with ten realizations); the 2.9-fold speed-up is the measurement recorded in spec C Section 18.

- [ ] **Step 2: Compile and check the report**

```bash
cd docs/report
bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --clean
rm -rf build main.pdf && mkdir build
bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --preview --preview-dir build/preview --verbose > build/compile-verbose.txt 2>&1
uv run python - <<'PY'
import re
import subprocess
import sys
from pathlib import Path

log = Path("build/compile-verbose.txt").read_text(encoding="utf-8", errors="replace")
final_pass = log.split("This is pdfTeX")[-1]
issues = re.findall(r"^.*(?:Overfull|Underfull|LaTeX Warning|Package \w+ Warning|^! ).*$", final_pass, re.M)
info = subprocess.run(["pdfinfo", "main.pdf"], capture_output=True, text=True, check=True).stdout
text = subprocess.run(["pdftotext", "main.pdf", "-"], capture_output=True, text=True, check=True).stdout
pages = re.search(r"Pages:\s+(\d+)", info).group(1)
print(f"final pass issues: {len(issues)}; pages: {pages}; unresolved ??: {text.count('??')}")
for line in issues:
    print("  ", line[:160])
sys.exit(1 if issues or "??" in text else 0)
PY
cd ../..
```

Expected: `final pass issues: 0; pages: 15; unresolved ??: 0` and exit code 0. The script scans only the last pdfLaTeX pass, because earlier passes always report undefined references before biber and the cross-reference pass have run. The `--clean` call first removes the Git-ignored auxiliary files of any earlier build; in a checkout that had been built before, stale `main.aux`, `main.bcf`, and `main.pdf` produced a corrupted PDF whose preview generation failed.

- [ ] **Step 3: Commit**

```bash
git add docs/report/sections/method.tex
git commit -m "docs: describe phase 2 search methods in the report"
```

### Task 3: Related work from the literature review

**Files:**
- Modify: `docs/report/sections/related-work.tex` (whole content), `docs/report/preamble.tex`
- Create: `docs/report/tables/related_work.tex`

**Interfaces:**
- Consumes: `docs/report/data/related_work.csv` (columns `key;year;venue;objective;uavs;relay_selection;deadline;connectivity;channel_uncertainty;bandwidth_power;method;guarantee`), bibliography keys from plan C.2 Task 6, labels `sec:method`, `sec:hardness`, `sec:bound`.
- Produces: label `tab:related`.

- [ ] **Step 1: Rewrite the related-work section**

Replace the whole content of `docs/report/sections/related-work.tex` with:

```latex
% docs/report/sections/related-work.tex
\section{Nghiên cứu liên quan}
\label{sec:related}

Phần này dựa trên một tổng quan có giao thức cố định. Sáu truy vấn theo các chủ đề UAV relay có hạn giao, lựa chọn relay với backhaul, luồng trên đồ thị mở rộng theo thời gian, giao dữ liệu đúng hạn dưới kênh ngẫu nhiên, đồng thiết kế quỹ đạo và băng thông, và phân rã kết hợp sửa lịch được chạy trên Crossref và arXiv, bổ sung mười truy vấn trên công cụ tìm kiếm web. Bài được sàng lọc theo tiêu đề rồi theo tóm tắt. Một bài được chọn nếu nghiên cứu UAV làm relay hoặc thu thập dữ liệu với ràng buộc thời gian theo từng yêu cầu (hạn, giới hạn trễ hoặc tuổi thông tin, trễ là mục tiêu, thời gian sống của tin, số yêu cầu được phục vụ trong một khoảng thời gian), hoặc có lựa chọn trạm hay ràng buộc dung lượng backhaul. Siêu dữ liệu được đối chiếu với Crossref hoặc arXiv; mọi nhận định chỉ lấy từ phần văn bản đã đọc, là tóm tắt, riêng bài của Huang và cộng sự là toàn văn. Mười sáu bài được chọn; \cref{tab:related} so sánh chúng theo các thuộc tính liên quan trực tiếp tới bài toán.

\paragraph{UAV relay và thu thập dữ liệu có hạn.}
Huang và cộng sự~\cite{huang2025ddatsap} dùng UAV làm relay giải mã và chuyển tiếp có bộ đệm cho nhiều cặp người dùng dưới kênh PrLoS, tối đa tốc độ trung bình nhỏ nhất bằng phân rã theo khối (BCD) kết hợp xấp xỉ lồi liên tiếp (SCA); đề tài này kế thừa mô hình kênh và cấu trúc ba khối của bài đó. Nhóm gần nhất với mục tiêu cảnh báo đúng hạn tối đa số thiết bị được phục vụ: Tran và cộng sự~\cite{tran2022relay} yêu cầu dữ liệu của thiết bị tới gateway mặt đất kịp thời qua bộ nhớ của UAV và tối ưu băng thông, công suất, quỹ đạo bằng nới lỏng biến nhị phân và xấp xỉ trong; Samir và cộng sự~\cite{samir2020time} gắn cho mỗi thiết bị một hạn cứng, tìm tối ưu toàn cục bằng branch, reduce and bound cho instance nhỏ và dùng SCA cho instance lớn; Du và cộng sự~\cite{du2023timeconstrained} đếm thiết bị được phục vụ trong thời gian thu thập hữu hạn. Các bài khác đặt thời gian vào ràng buộc hoặc mục tiêu theo cách khác: hạn thu thập chung với mô hình MILP và heuristic~\cite{albusalih2018deadline}, ngưỡng tuổi thông tin và lựa chọn trạm mặt đất nhận dữ liệu~\cite{liu2022aoi,liu2024interference}, trễ hàng đợi hữu hạn trong thiết kế trực tuyến dựa trên Lyapunov~\cite{tang2022delay}, trễ là mục tiêu của relay UAV hoặc của thu thập theo mức ưu tiên~\cite{he2022delay,cao2023priority,fadlullah2016dynamic}, và thời gian sống của tin trong mạng chịu trễ với UAV học tăng cường~\cite{wang2026juror}.

\paragraph{Backhaul và lựa chọn trạm.}
Khi UAV làm trạm gốc hoặc relay nối người dùng với mạng lõi, dung lượng backhaul giới hạn số người dùng phục vụ được. Kalantari và cộng sự~\cite{kalantari2017backhaul} đặt vị trí 3D của trạm bay theo loại backhaul; Selim và Kamal~\cite{selim2018postdisaster} dùng drone có dây làm backhaul sau thiên tai; Yu và cộng sự~\cite{yu2023backhaul} đồng tối ưu băng thông và vị trí dưới ràng buộc dung lượng backhaul quang không dây; Queiros và cộng sự~\cite{queiros2024predictive} cấp băng thông và đặt relay bay tầng hai khi đã biết trước quỹ đạo tầng một. Các bài này không có ràng buộc thời gian theo từng yêu cầu.

\paragraph{Phương pháp giải.}
Phân rã theo khối kết hợp SCA là cách giải phổ biến cho quỹ đạo và tài nguyên~\cite{huang2025ddatsap,wu2018multi,liu2024interference}; Wu và cộng sự~\cite{wu2018multi} chứng minh cách này hội tụ tới ít nhất một nghiệm tối ưu cục bộ cho bài toán thông lượng max--min. Các bài có đếm yêu cầu được phục vụ phải xử lý biến nhị phân bằng nới lỏng~\cite{tran2022relay} hoặc branch-and-bound~\cite{samir2020time}. Park và Lee~\cite{park2026expected} chỉ ra rằng xấp xỉ tốc độ kỳ vọng bằng tốc độ tại kênh trung bình ước lượng cao tốc độ thực do bất đẳng thức Jensen, là lý do đề tài này chấm điểm thiết kế trên các realization lấy mẫu. Tìm kiếm lân cận lớn với các toán tử phá và sửa~\cite{ropke2006alns} là tiền lệ cho cơ chế sửa lịch, còn định tuyến trên đồ thị thay đổi theo thời gian có động học biết trước~\cite{jain2004dtn} là tiền lệ cho cách đo khả năng kết nối theo thời gian.

\paragraph{Lập lịch với thời điểm phát sinh và hạn.}
Khi chỉ xét một liên kết, việc chọn tập cảnh báo giao đúng hạn gần với bài toán lập lịch một máy tối thiểu số công việc trễ. Phiên bản không ngắt quãng $1\,|\,r_j\,|\,\sum U_j$ là NP-khó mạnh~\cite{lenstra2020elements}, còn phiên bản có ngắt quãng $1\,|\,r_j,\mathrm{pmtn}\,|\,\sum U_j$ giải được bằng quy hoạch động trong thời gian đa thức~\cite{lawler1990preemptive}. Mô hình ở \cref{sec:method} cho phép chia dữ liệu theo slot, gần với trường hợp có ngắt quãng; độ khó của nó đến từ ràng buộc mỗi cảnh báo đi theo đúng một đường (\cref{sec:hardness}). Cận trên ở \cref{sec:bound} dựa trên tính lõm của hàm tốc độ theo băng thông và bao các tiếp tuyến, là công cụ chuẩn của tối ưu lồi~\cite{boyd2004convex}.

\paragraph{Khoảng trống và phương pháp so sánh mạnh.}
Không bài nào trong \cref{tab:related} kết hợp hạn theo từng yêu cầu với việc chọn relay mặt đất trên backhaul có dung lượng hữu hạn: các bài có hạn giả định một điểm đích, còn các bài có ràng buộc backhaul không có hạn theo yêu cầu. Để chọn một phương pháp mạnh từ tài liệu, các tiêu chí được xét theo thứ tự: mục tiêu gần với số cảnh báo đúng hạn, thiết kế ngoại tuyến với nhu cầu đã biết, đồng tối ưu quỹ đạo và tài nguyên vô tuyến, đủ chi tiết hoặc có mã để tái lập, và thích nghi được mà không đổi bản chất phương pháp. Ba bài của Tran, Samir và Du đạt tiêu chí đầu; tiêu chí cuối quyết định chọn Tran và cộng sự~\cite{tran2022relay} (phiên bản bán song công), vì chỉ bài này mô hình dữ liệu đi qua bộ đệm UAV tới gateway mặt đất, tương ứng với đường UAV thả dữ liệu vào mạng mặt đất ở đây. Bài của Samir và cộng sự là phương án dự phòng vì đã có mã cài đặt công khai. Việc thích nghi và đánh giá phương pháp này thuộc giai đoạn cuối của đề tài.

\input{tables/related_work}
```

- [ ] **Step 2: Add the comparison table**

```latex
% docs/report/tables/related_work.tex
% Bảng so sánh tài liệu — hàng đọc từ data/related_work.csv (tổng quan ở docs/literature/review.md).
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{So sánh các công trình được chọn trong tổng quan theo nội dung tóm tắt (riêng~\cite{huang2025ddatsap} theo toàn văn). ``Không nêu'' nghĩa là phần văn bản đã đọc không nói tới thuộc tính đó.}
\label{tab:related}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\begin{tabular}{@{}L{0.06\textwidth}L{0.17\textwidth}L{0.13\textwidth}L{0.14\textwidth}L{0.12\textwidth}L{0.13\textwidth}L{0.15\textwidth}@{}}
\toprule
\textbf{Bài} & \textbf{Mục tiêu} & \textbf{Chọn trạm, backhaul} & \textbf{Ràng buộc thời gian} & \textbf{Kênh, bất định} & \textbf{Tài nguyên} & \textbf{Phương pháp} \\
\midrule
\csvreader[separator=semicolon, late after line=\\, late after last line=\\\midrule]{data/related_work.csv}{1=\colkey, 4=\colobjective, 6=\colrelay, 7=\coldeadline, 9=\colchannel, 10=\colresource, 11=\colmethod}{\expandafter\cite\expandafter{\colkey} & \colobjective & \colrelay & \coldeadline & \colchannel & \colresource & \colmethod}
Đề tài này & Tỷ lệ cảnh báo đúng hạn & Có, relay mặt đất và backhaul hữu hạn & Có, hạn theo cảnh báo & PrLoS, thiết kế trên realization lấy mẫu & Băng thông theo trọng số & Tìm kiếm theo khối có đánh giá, sửa lịch \\
\bottomrule
\end{tabular}
\end{table}
```

Inside `\csvreader`, `\expandafter\cite\expandafter{\colkey}` expands the key before biblatex sees it; the ragged-right `L` columns avoid the wide inter-word gaps of justified narrow columns.

- [ ] **Step 3: Let long bibliography entries break**

In `docs/report/preamble.tex` (1 edit):

Edit 1: replace

```latex
\DeclareLanguageMapping{vietnamese}{english}
% \addbibresource nằm trong main.tex để script compile_latex.sh của skill nhận ra cần chạy biber
```

with

```latex
\DeclareLanguageMapping{vietnamese}{english}
\AtBeginBibliography{\raggedright}  % tên bài dài có từ nối gạch không ngắt được, căn trái để không tràn lề
% \addbibresource nằm trong main.tex để script compile_latex.sh của skill nhận ra cần chạy biber
```

Without it, two entries with hyphenated title words that TeX cannot break (`tang2022delay` and `wu2018multi`) overflow the margin by 10 pt and 25 pt.

- [ ] **Step 4: Compile and check the report**

```bash
cd docs/report
bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --clean
rm -rf build main.pdf && mkdir build
bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --preview --preview-dir build/preview --verbose > build/compile-verbose.txt 2>&1
uv run python - <<'PY'
import re
import subprocess
import sys
from pathlib import Path

log = Path("build/compile-verbose.txt").read_text(encoding="utf-8", errors="replace")
final_pass = log.split("This is pdfTeX")[-1]
issues = re.findall(r"^.*(?:Overfull|Underfull|LaTeX Warning|Package \w+ Warning|^! ).*$", final_pass, re.M)
info = subprocess.run(["pdfinfo", "main.pdf"], capture_output=True, text=True, check=True).stdout
text = subprocess.run(["pdftotext", "main.pdf", "-"], capture_output=True, text=True, check=True).stdout
pages = re.search(r"Pages:\s+(\d+)", info).group(1)
print(f"final pass issues: {len(issues)}; pages: {pages}; unresolved ??: {text.count('??')}")
for line in issues:
    print("  ", line[:160])
sys.exit(1 if issues or "??" in text else 0)
PY
cd ../..
```

Expected: `final pass issues: 0; pages: 18; unresolved ??: 0` and exit code 0. The script scans only the last pdfLaTeX pass, because earlier passes always report undefined references before biber and the cross-reference pass have run. The `--clean` call first removes the Git-ignored auxiliary files of any earlier build; in a checkout that had been built before, stale `main.aux`, `main.bcf`, and `main.pdf` produced a corrupted PDF whose preview generation failed.

Open the preview page that holds Table 1 and confirm that all seven columns fit the text width and the last row is "Đề tài này".

- [ ] **Step 5: Commit**

```bash
git add docs/report/sections/related-work.tex docs/report/tables/related_work.tex docs/report/preamble.tex
git commit -m "docs: report the literature review and baseline choice"
```

### Task 4: Experiments section with Phase 2 results

**Files:**
- Modify: `docs/report/sections/experiments.tex` (whole content), `docs/report/main.tex`, `docs/report/preamble.tex`
- Create: `docs/report/tables/phase2_main.tex`, `phase2_statistics.tex`, `phase2_ablation.tex`, `phase2_connectivity.tex`, `phase2_design.tex`
- Delete: `docs/report/tables/phase1_main.tex` (its B0 and B1 rows are part of `phase2_main.tex`)

**Interfaces:**
- Consumes: Task 1 data files and macros; Phase 1 macros (`\ResultPythonVersion`, `\ResultWorkers`, `\ResultMilpCount`, and others); labels from Tasks 2–3.
- Produces: labels `sec:setup`, `sec:results-main`, `sec:results-stats`, `sec:results-ablation`, `sec:results-connectivity`, `sec:results-budget`, `sec:results-sensitivity`, `sec:results-milp`, `sec:results-runtime`, `sec:rq`, `tab:phase2-main`, `tab:phase2-stats`, `tab:phase2-ablation`, `tab:phase2-connectivity`, `tab:phase2-design`, `fig:budget`, `fig:sensitivity`, `fig:lp-runtime`.

- [ ] **Step 1: Load the Phase 2 macros and the groupplots library**

In `docs/report/main.tex` (1 edit):

Edit 1: replace

```latex
\input{data/phase1_results}  % macro số liệu Pha 1, sinh bởi experiments/report_tables.py
\addbibresource{references.bib}
```

with

```latex
\input{data/phase1_results}  % macro số liệu Pha 1, sinh bởi experiments/report_tables.py
\input{data/phase2_results}  % macro số liệu Pha 2, sinh bởi experiments/report_tables_phase2.py
\addbibresource{references.bib}
```

In `docs/report/preamble.tex` (1 edit):

Edit 1: replace

```latex
\pgfplotsset{compat=1.18}
```

with

```latex
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}  % lưới biểu đồ độ nhạy
```

- [ ] **Step 2: Add the Phase 2 tables**

```latex
% docs/report/tables/phase2_main.tex
% So sánh chính Pha 2 — hàng đọc từ data/phase2_main.csv (sinh bởi experiments/report_tables_phase2.py).
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{3pt}
\caption{So sánh chính trên 30 scenario đánh giá mỗi họ, mỗi scenario 30 realization kênh. Khoảng tin cậy 95\% lấy bằng bootstrap theo topology; gap riêng tính so với cận trên cùng thông tin (B0: cận chỉ mặt đất; các phương pháp dùng UAV: cận trên chính quỹ đạo của phương pháp), gap hợp tính so với cận hợp (giá trị lớn nhất của các cận quỹ đạo theo từng realization); thời gian lập kế hoạch tính bằng giây; ĐG là số lần đánh giá trung bình.}
\label{tab:phase2-main}
\begin{tabular}{@{}llcccccrr@{}}
\toprule
\textbf{Backhaul} & \textbf{PP} & \textbf{Đúng hạn} & \textbf{KTC 95\%} & \textbf{Conn} & \textbf{Gap riêng} & \textbf{Gap hợp} & \textbf{Lập KH} & \textbf{ĐG} \\
\midrule
\csvreader[separator=semicolon, late after line=\\]{data/phase2_main.csv}{2=\colbackhaul, 3=\colmethod, 4=\coltimely, 5=\colci, 6=\colconn, 7=\colgap, 8=\coluniongap, 9=\colplanning, 10=\colcalls}{\colbackhaul & \colmethod & \coltimely & \colci & \colconn & \colgap & \coluniongap & \colplanning & \colcalls}
\bottomrule
\end{tabular}
\end{table}
```

```latex
% docs/report/tables/phase2_statistics.tex
% Kiểm định ghép cặp — hàng đọc từ data/phase2_statistics.csv (sinh bởi experiments/report_tables_phase2.py).
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Tám so sánh chính theo cặp ở mức topology (10 topology). Chênh lệch là hiệu tỷ lệ đúng hạn trung bình; KTC 95\% lấy bằng bootstrap trên topology; $r$ là tương quan hạng biserial ghép cặp; $p$ từ kiểm định đổi dấu chính xác và $p_{\mathrm{Holm}}$ sau hiệu chỉnh Holm trên tám so sánh.}
\label{tab:phase2-stats}
\begin{tabular}{@{}llccccc@{}}
\toprule
\textbf{Họ} & \textbf{So sánh} & \textbf{Chênh lệch} & \textbf{KTC 95\%} & \textbf{$r$} & \textbf{$p$} & \textbf{$p_{\mathrm{Holm}}$} \\
\midrule
\csvreader[separator=semicolon, late after line=\\]{data/phase2_statistics.csv}{1=\colset, 2=\colcomparison, 3=\coldiff, 4=\colci, 5=\colr, 6=\colp, 7=\colholm}{\colset & \colcomparison & \coldiff & \colci & \colr & \colp & \colholm}
\bottomrule
\end{tabular}
\end{table}
```

```latex
% docs/report/tables/phase2_ablation.tex
% Ablation của P — hàng đọc từ data/phase2_ablation.csv (sinh bởi experiments/report_tables_phase2.py).
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}
\caption{Ablation của P trên hai họ scenario. Cột $\Delta$ là trung bình theo topology của hiệu tỷ lệ đúng hạn giữa biến thể và P, kèm KTC 95\% bootstrap.}
\label{tab:phase2-ablation}
\begin{tabular}{@{}lcccc@{}}
\toprule
& \multicolumn{2}{c}{\textbf{v0}} & \multicolumn{2}{c}{\textbf{v0-bh50}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\textbf{Biến thể} & \textbf{Đúng hạn} & \textbf{$\Delta$ so với P [KTC]} & \textbf{Đúng hạn} & \textbf{$\Delta$ so với P [KTC]} \\
\midrule
\csvreader[separator=semicolon, late after line=\\]{data/phase2_ablation.csv}{1=\colvariant, 2=\coltimely, 3=\coldiff, 4=\coltimelybh, 5=\coldiffbh}{\colvariant & \coltimely & \coldiff & \coltimelybh & \coldiffbh}
\bottomrule
\end{tabular}
\end{table}
```

```latex
% docs/report/tables/phase2_connectivity.tex
% Theo mức kết nối mặt đất — hàng đọc từ data/phase2_connectivity.csv (sinh bởi experiments/report_tables_phase2.py).
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Kết quả theo tỷ lệ nguồn có relay mặt đất trong tầm của scenario. Conn B0 là tỷ lệ nguồn có đường theo thời gian tới trung tâm khi không có UAV; cột cuối là mức tăng tỷ lệ đúng hạn của B2 so với B0.}
\label{tab:phase2-connectivity}
\begin{tabular}{@{}llcccccc@{}}
\toprule
\textbf{Họ} & \textbf{Nguồn nối được mặt đất} & \textbf{Scenario} & \textbf{Conn B0} & \textbf{B0} & \textbf{B1} & \textbf{B2} & \textbf{B2 $-$ B0} \\
\midrule
\csvreader[separator=semicolon, late after line=\\]{data/phase2_connectivity.csv}{1=\colset, 2=\colgroup, 3=\colscenarios, 4=\colconn, 5=\colbzero, 6=\colbone, 7=\colbtwo, 8=\colgain}{\colset & \colgroup & \colscenarios & \colconn & \colbzero & \colbone & \colbtwo & \colgain}
\bottomrule
\end{tabular}
\end{table}
```

```latex
% docs/report/tables/phase2_design.tex
% Số realization thiết kế của P — hàng đọc từ data/phase2_design.csv (sinh bởi experiments/report_tables_phase2.py).
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{P với số realization thiết kế khác nhau trên replicate đầu của mười topology v0 (ngân sách 2000). Thời gian lập kế hoạch tính bằng giây.}
\label{tab:phase2-design}
\begin{tabular}{@{}lccr@{}}
\toprule
\textbf{Realization thiết kế} & \textbf{Đúng hạn} & \textbf{Lập KH} & \textbf{Đánh giá} \\
\midrule
\csvreader[separator=semicolon, late after line=\\]{data/phase2_design.csv}{1=\coldesign, 2=\coltimely, 3=\colplanning, 4=\colcalls}{\coldesign & \coltimely & \colplanning & \colcalls}
\bottomrule
\end{tabular}
\end{table}
```


The ablation table uses `\footnotesize` and two-point column separation; with the default spacing it is 116 pt wider than the text.

- [ ] **Step 3: Remove the superseded Phase 1 table**

```bash
git rm docs/report/tables/phase1_main.tex
```

- [ ] **Step 4: Rewrite the experiments section**

Replace the whole content of `docs/report/sections/experiments.tex` with:

```latex
% docs/report/sections/experiments.tex
\section{Kết quả thực nghiệm}
\label{sec:experiments}

\subsection{Thiết lập}
\label{sec:setup}

Bộ kịch bản v0 được sinh tất định từ 130 topology Topology Zoo có 15--40 nút, có toạ độ và liên thông. Các topology được chia tầng theo số nút; mười topology được dùng để đánh giá với ba replicate mỗi topology, ba topology khác dành cho phát triển với hai replicate. Kích thước cảnh báo lấy tỷ lệ tương đối từ demand của instance SNDlib \texttt{abilene}~\cite{orlowski2010sndlib}; vùng hư hại, điểm nguồn, thời điểm phát sinh, hạn và cạnh hỏng được sinh bằng các luồng số ngẫu nhiên có seed riêng. Họ v0-bh50 giữ nguyên mọi seed và chỉ giảm dung lượng backhaul, nên hai họ cùng topology, cảnh báo và mẫu cạnh hỏng. \Cref{tab:params} liệt kê các tham số. Mọi kết hợp là kịch bản tổng hợp từ dữ liệu công khai, không phải mạng cứu hộ đo thực địa.

Mỗi phương pháp lập kế hoạch một lần cho mỗi scenario và được đánh giá trên 30 realization kênh dùng chung số ngẫu nhiên. Các topology phát triển chỉ được dùng cho một lần chạy thử để xác nhận ngân sách $E=2000$ và $M_d=5$; không tham số nào được chỉnh trên tập đánh giá. Thí nghiệm chính chạy B0, B1, B2, B3, P và bảy biến thể ablation của P trên \ResultTwoScenarioCount{} scenario đánh giá của mỗi họ. Cận trên LP được giải cho từng realization trên chính quỹ đạo của B1, B2, B3 và P; \emph{cận hợp} của một realization là giá trị lớn nhất trong các cận đó. Thí nghiệm ngân sách chạy B2 và P với $E$ từ 250 đến 4000 trên v0. Thí nghiệm độ nhạy thay đổi từng tham số một so với v0 trên replicate đầu của mỗi topology đánh giá (\ResultTwoSensitivityScenarioCount{} scenario mỗi họ), và thí nghiệm số realization thiết kế chạy P trên cùng các scenario đó.

Kết quả được lấy trung bình theo realization rồi theo scenario; khoảng tin cậy 95\% dùng bootstrap theo cụm topology vì các replicate của một topology không độc lập. Tám so sánh chính (P $-$ B2, B2 $-$ B1, B2 $-$ B3 và P $-$ B1 trên mỗi họ) dùng mười cặp giá trị ở mức topology: kiểm định đổi dấu chính xác xét cả $2^{10}$ cách đổi dấu, giá trị $p$ được hiệu chỉnh Holm trên tám so sánh, khoảng tin cậy của chênh lệch dùng bootstrap trên topology với \num{10000} lần lấy mẫu, và cỡ ảnh hưởng là tương quan hạng biserial ghép cặp. Ablation và độ nhạy được báo cáo mô tả, không kiểm định thêm. Cả \ResultTwoLpOptimalCount{} trong \ResultTwoLpCount{} bài LP của thí nghiệm chính và độ nhạy được giải tối ưu; khi presolve của HiGHS không phân loại được một bài (hai bài ở lần chạy đầu tiên, do hệ số của ma trận trải từ khoảng $10^{-13}$ tới $10^{6}$), bài đó được giải lại khi tắt presolve. Thí nghiệm chạy bằng Python \ResultPythonVersion, \texttt{scipy} \ResultScipyVersion{} và HiGHS \ResultHighsVersion{} với \ResultWorkers{} tiến trình trên máy \ResultCpuCount{} CPU; tổng thời gian tính của các task Pha 2 là \ResultTwoComputeHours{} giờ.

\input{tables/params}

\subsection{So sánh chính}
\label{sec:results-main}

\input{tables/phase2_main}

\Cref{tab:phase2-main} tóm tắt so sánh chính. Trên họ v0, B2 giao đúng hạn \ResultTwoTimelyVzeroBtwo{} số cảnh báo, so với \ResultTwoTimelyVzeroBone{} của B1 và \ResultTwoTimelyVzeroBzero{} của B0; B2 cao hơn B0 trên \ResultTwoHigherVzeroBtwoBzero{} và cao hơn B1 trên \ResultTwoHigherVzeroBtwoBone{} trong \ResultTwoScenarioCount{} scenario. P đạt \ResultTwoTimelyVzeroP, gần như bằng B2: P cao hơn B2 trên \ResultTwoHigherVzeroPBtwo{} scenario, bằng trên \ResultTwoEqualVzeroPBtwo{} và thấp hơn trên \ResultTwoLowerVzeroPBtwo. B3 chỉ đạt \ResultTwoTimelyVzeroBthree, thấp hơn cả B1, dù mọi phương pháp dùng UAV đều đưa khả năng kết nối lên gần bằng một. Khi backhaul chỉ còn \qty{50}{\kilo\bit\per\second}, thứ tự không đổi: B2 đạt \ResultTwoTimelyVzeroBhfivezeroBtwo{} và P đạt \ResultTwoTimelyVzeroBhfivezeroP, so với \ResultTwoTimelyVzeroBhfivezeroBone{} của B1 và \ResultTwoTimelyVzeroBhfivezeroBzero{} của B0.

Gap của B2 so với cận trên chính quỹ đạo của nó là \ResultTwoGapVzeroBtwo, nhỏ hơn gap \ResultTwoGapVzeroBone{} của B1 trên quỹ đạo B1, vì B2 vừa giao được nhiều cảnh báo hơn vừa chọn quỹ đạo có cận cao hơn. So với cận hợp \ResultTwoUnionBoundVzeroBtwo, B2 còn cách \ResultTwoUnionGapVzeroBtwo{} và B1 cách \ResultTwoUnionGapVzeroBone. B3 có gap riêng \ResultTwoGapVzeroBthree{} dù cận trên quỹ đạo của nó gần bằng của B2, nên phần thua kém đến từ lựa chọn đường và băng thông theo mục tiêu leximin chứ không từ quỹ đạo.

\subsection{Kiểm định thống kê}
\label{sec:results-stats}

\input{tables/phase2_statistics}

\Cref{tab:phase2-stats} cho thấy B2 hơn B1 trung bình \ResultTwoDiffVzeroBtwoBone{} (KTC 95\% [\ResultTwoDiffLowVzeroBtwoBone, \ResultTwoDiffHighVzeroBtwoBone]) trên v0 và \ResultTwoDiffVzeroBhfivezeroBtwoBone{} trên v0-bh50, còn P hơn B1 \ResultTwoDiffVzeroPBone{} và \ResultTwoDiffVzeroBhfivezeroPBone. Các so sánh này và B2 $-$ B3 đều có $p_{\mathrm{Holm}}=\ResultTwoHolmFloor$, giá trị nhỏ nhất có thể với mười topology và tám so sánh ($8\cdot 2/2^{10}$): cả mười topology cùng chiều, và tương quan hạng biserial bằng một. Ngược lại, P $-$ B2 có chênh lệch \ResultTwoDiffVzeroPBtwo{} (KTC 95\% [\ResultTwoDiffLowVzeroPBtwo, \ResultTwoDiffHighVzeroPBtwo]) trên v0 và \ResultTwoDiffVzeroBhfivezeroPBtwo{} trên v0-bh50, với $p_{\mathrm{Holm}}=\ResultTwoHolmVzeroPBtwo$: dữ liệu không cho thấy cơ chế sửa lịch làm tăng tỷ lệ đúng hạn so với B2.

\subsection{Ablation}
\label{sec:results-ablation}

\input{tables/phase2_ablation}

Trong \cref{tab:phase2-ablation}, bỏ khối đường làm tỷ lệ đúng hạn thay đổi \ResultTwoAblationDiffVzeroPFixedPaths{} trên v0 và \ResultTwoAblationDiffVzeroBhfivezeroPFixedPaths{} trên v0-bh50, còn bỏ khối quỹ đạo thay đổi \ResultTwoAblationDiffVzeroPFixedTrajectory{} và \ResultTwoAblationDiffVzeroBhfivezeroPFixedTrajectory. Thiết kế trên tốc độ kỳ vọng thay cho năm realization lấy mẫu thay đổi \ResultTwoAblationDiffVzeroPExpectedDesign{} và \ResultTwoAblationDiffVzeroBhfivezeroPExpectedDesign, dù thời gian lập kế hoạch ngắn hơn nhiều. Các biến thể còn lại --- chỉ giữ một thao tác sửa lịch, sửa lịch không dùng thông tin backlog, và bỏ khối băng thông --- chỉ lệch khỏi P không quá vài phần nghìn. Lợi ích của tìm kiếm vì vậy đến từ việc đồng chọn đường và quỹ đạo trên các realization lấy mẫu; trọng số băng thông và hai thao tác sửa lịch hầu như không thay đổi kết quả khi bộ điều phối EDF đã chia băng thông theo backlog.

\subsection{Vai trò của UAV theo mức kết nối mặt đất}
\label{sec:results-connectivity}

\input{tables/phase2_connectivity}

\Cref{tab:phase2-connectivity} chia scenario theo tỷ lệ nguồn có relay mặt đất trong tầm. Tỷ lệ đúng hạn của B0 bám sát tỷ lệ này, với hệ số tương quan \ResultTwoGroundCorrelationVzero{} trên v0 và \ResultTwoGroundCorrelationVzeroBhfivezero{} trên v0-bh50, vì cảnh báo từ nguồn không có relay trong tầm không thể được giao khi không có UAV. UAV đưa khả năng kết nối của mọi nhóm lên gần bằng một, và mức tăng của B2 so với B0 gần như không đổi giữa các nhóm: \ResultTwoTercileGainVzeroLow, \ResultTwoTercileGainVzeroMiddle{} và \ResultTwoTercileGainVzeroHigh{} trên v0, \ResultTwoTercileGainVzeroBhfivezeroLow, \ResultTwoTercileGainVzeroBhfivezeroMiddle{} và \ResultTwoTercileGainVzeroBhfivezeroHigh{} trên v0-bh50. Thí nghiệm không tách riêng nguyên nhân của mức tăng ở từng nhóm, nên báo cáo không suy ra cơ chế từ bảng này.

\subsection{Ngân sách đánh giá và số realization thiết kế}
\label{sec:results-budget}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{semilogxaxis}[
    width=0.8\textwidth, height=5.5cm, log basis x=2,
    xlabel={Ngân sách đánh giá $E$}, ylabel={Tỷ lệ đúng hạn},
    xtick={250,500,1000,2000,4000}, xticklabels={250,500,1000,2000,4000},
    grid=major, legend pos=south east, /pgf/number format/use comma,
    yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=3},
]
\addplot[mark=*, color=blue!70!black] table[col sep=comma, x=budget, y=timely] {data/phase2_budget_b2.csv};
\addlegendentry{B2}
\addplot[mark=square*, dashed, color=red!70!black] table[col sep=comma, x=budget, y=timely] {data/phase2_budget_p.csv};
\addlegendentry{P}
\end{semilogxaxis}
\end{tikzpicture}
\caption{Tỷ lệ đúng hạn của B2 và P trên 30 scenario v0 theo ngân sách đánh giá.}
\label{fig:budget}
\end{figure}

\input{tables/phase2_design}

Theo \cref{fig:budget}, B2 đạt \ResultTwoBudgetBtwoTwofivezero{} với $E=250$ và \ResultTwoBudgetBtwoOnezerozerozero{} với $E=1000$, rồi gần như không đổi khi tăng tới $E=4000$ (\ResultTwoBudgetBtwoFourzerozerozero), vì phương pháp dừng sau trung bình \ResultTwoBudgetCallsBtwoFourzerozerozero{} lần đánh giá khi một vòng không còn nhận thay đổi. P có cùng xu hướng (\ResultTwoBudgetPTwofivezero, \ResultTwoBudgetPOnezerozerozero{} và \ResultTwoBudgetPFourzerozerozero). Ngân sách mặc định $E=2000$ vì vậy nằm ở vùng bão hoà, và kết luận P không hơn B2 không do thiếu ngân sách. \Cref{tab:phase2-design} cho thấy số realization thiết kế quan trọng hơn ngân sách: thiết kế trên tốc độ kỳ vọng đạt \ResultTwoDesignExpected, một realization đạt \ResultTwoDesignOne, năm realization đạt \ResultTwoDesignFive{} và mười realization đạt \ResultTwoDesignOnezero, trong khi thời gian lập kế hoạch tăng từ \ResultTwoDesignPlanningExpected{} lên \ResultTwoDesignPlanningOnezero{} giây.

\subsection{Độ nhạy}
\label{sec:results-sensitivity}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{groupplot}[
    group style={group size=3 by 2, horizontal sep=1.2cm, vertical sep=1.7cm},
    width=0.34\textwidth, height=4.3cm, ymin=0.2, ymax=0.65, grid=major,
    tick label style={font=\scriptsize}, label style={font=\footnotesize},
    /pgf/number format/use comma, yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1},
    cycle list={{blue!70!black, mark=*}, {green!50!black, mark=triangle*}, {red!70!black, dashed, mark=square*}},
]
\nextgroupplot[xlabel={$L_{\mathrm{ref}}$ (kbit)}, ylabel={Tỷ lệ đúng hạn}, xmode=log, log basis x=2, xtick={125,250,500,1000}, xticklabels={125,250,500,1000}, legend to name=senslegend, legend columns=3]
\addplot table[col sep=comma, x=x, y=B1] {data/phase2_sensitivity_lref.csv}; \addlegendentry{B1}
\addplot table[col sep=comma, x=x, y=B2] {data/phase2_sensitivity_lref.csv}; \addlegendentry{B2}
\addplot table[col sep=comma, x=x, y=P] {data/phase2_sensitivity_lref.csv}; \addlegendentry{P}
\nextgroupplot[xlabel={$C_e$ (kbit/s)}, xmode=log, xtick={50,100,1000}, xticklabels={50,100,1000}]
\addplot table[col sep=comma, x=x, y=B1] {data/phase2_sensitivity_backhaul.csv};
\addplot table[col sep=comma, x=x, y=B2] {data/phase2_sensitivity_backhaul.csv};
\addplot table[col sep=comma, x=x, y=P] {data/phase2_sensitivity_backhaul.csv};
\nextgroupplot[xlabel={Cửa sổ hạn trung bình (slot)}, xtick={20,40,80}]
\addplot table[col sep=comma, x=x, y=B1] {data/phase2_sensitivity_deadline.csv};
\addplot table[col sep=comma, x=x, y=B2] {data/phase2_sensitivity_deadline.csv};
\addplot table[col sep=comma, x=x, y=P] {data/phase2_sensitivity_deadline.csv};
\nextgroupplot[xlabel={$B_{\mathrm{tot}}$ (MHz)}, ylabel={Tỷ lệ đúng hạn}, xtick={0.5,1,2}]
\addplot table[col sep=comma, x=x, y=B1] {data/phase2_sensitivity_btot.csv};
\addplot table[col sep=comma, x=x, y=B2] {data/phase2_sensitivity_btot.csv};
\addplot table[col sep=comma, x=x, y=P] {data/phase2_sensitivity_btot.csv};
\nextgroupplot[xlabel={Số cảnh báo $\abs{A}$}, xtick={20,40,60}]
\addplot table[col sep=comma, x=x, y=B1] {data/phase2_sensitivity_alerts.csv};
\addplot table[col sep=comma, x=x, y=B2] {data/phase2_sensitivity_alerts.csv};
\addplot table[col sep=comma, x=x, y=P] {data/phase2_sensitivity_alerts.csv};
\nextgroupplot[xlabel={Số candidate $K$}, xtick={2,3,5}]
\addplot table[col sep=comma, x=x, y=B1] {data/phase2_sensitivity_paths.csv};
\addplot table[col sep=comma, x=x, y=B2] {data/phase2_sensitivity_paths.csv};
\addplot table[col sep=comma, x=x, y=P] {data/phase2_sensitivity_paths.csv};
\end{groupplot}
\end{tikzpicture}

\pgfplotslegendfromname{senslegend}
\caption{Tỷ lệ đúng hạn khi thay đổi từng tham số so với v0 (điểm giữa của mỗi hình, trừ backhaul, nơi v0 là \qty{1000}{\kilo\bit\per\second} và v0-bh50 là \qty{50}{\kilo\bit\per\second}), trên replicate đầu của mười topology đánh giá.}
\label{fig:sensitivity}
\end{figure}

\Cref{fig:sensitivity} cho thấy thứ tự B1 thấp hơn B2 và P giữ nguyên trên mọi họ không suy biến. Hạn ngắn thu hẹp lợi thế (B1 \ResultTwoSensSensDeadlineShortBone{} so với P \ResultTwoSensSensDeadlineShortP), còn hạn dài mở rộng lợi thế (B1 \ResultTwoSensSensDeadlineLongBone{} so với P \ResultTwoSensSensDeadlineLongP). Cảnh báo lớn hơn làm mọi phương pháp giảm mạnh: với $L_{\mathrm{ref}}=\qty{1}{\mega\bit}$, B1 đạt \ResultTwoSensSensLrefonemBone{} và P đạt \ResultTwoSensSensLrefonemP. Tăng số cảnh báo lên 60 hoặc giảm $B_{\mathrm{tot}}$ còn một nửa làm tỷ lệ đúng hạn giảm nhẹ (P \ResultTwoSensSensAlertssixzeroP{} và \ResultTwoSensSensBtotzerofiveP, so với \ResultTwoSensVzeroP{} của v0), còn số candidate $K$ ảnh hưởng ít. Họ dùng hằng số vật lý khác ($\beta_0=\qty{-40}{\dB}$, $N_0=\qty{-164}{\dBm\per\hertz}$) suy biến: mọi nguồn có relay mặt đất trong tầm và cả ba phương pháp đạt \ResultTwoSensSensPhysicalP, nên họ này không được dùng để kết luận về vai trò của UAV.

\subsection{Độ chặt của cận trên}
\label{sec:results-milp}

\input{tables/phase1_milp}

\Cref{tab:phase1-milp} đối chiếu cận LP với MILP trên \ResultMilpCount{} instance thu nhỏ của Pha 1. HiGHS giải tối ưu \ResultMilpOptimalCount{} trong \ResultMilpCount{} bài MILP. Trên các instance này, B1 đạt đúng giá trị tối ưu MILP ở \ResultMilpMethodAtOptimumCount{} instance và kế hoạch dựng từ nghiệm MILP đạt tối ưu ở \ResultMilpPlanAtOptimumCount{} instance, nên khoảng cách giữa cận LP và giá trị đạt được chủ yếu đến từ việc nới lỏng tính nguyên của lựa chọn đường. Khi dựng kế hoạch, băng thông lấy thẳng từ nghiệm MILP chỉ giao đúng hạn tổng cộng \ResultMilpStaticTotal{} cảnh báo trên năm instance, so với \ResultMilpEqualSplitTotal{} khi giữ các đường của nghiệm và chia đều băng thông theo backlog, vì lịch băng thông tĩnh lệch nhịp với cách bộ điều phối EDF phục vụ hàng đợi. Bao mười tiếp tuyến ước lượng tốc độ cao hơn thực tế tối đa \ResultTangentOverestimateMax{} (theo tỷ lệ tương đối) trên $[B_{\mathrm{tot}}/512, B_{\mathrm{tot}}]$. Vì vậy các gap ở \cref{tab:phase2-main} nên được hiểu là cận trên của phần có thể cải thiện, có thể lớn hơn đáng kể so với phần thực sự đạt được.

\subsection{Thời gian tính toán}
\label{sec:results-runtime}

Lập kế hoạch B0 gần như tức thời, còn B1 mất trung bình \ResultPlanningVzeroBone{} giây cho mỗi scenario v0 vì phải mô phỏng từng candidate. B2 và P mất trung bình \ResultTwoPlanningVzeroBtwo{} và \ResultTwoPlanningVzeroP{} giây với \ResultTwoCallsVzeroBtwo{} và \ResultTwoCallsVzeroP{} lần đánh giá khi chạy song song \ResultWorkers{} tiến trình; đánh giá một kế hoạch trên một realization mất khoảng \ResultEvaluationVzeroBone{} giây. \Cref{fig:lp-runtime} cho thấy thời gian giải của \ResultLpCount{} bài LP ở Pha 1: số biến có trung vị \ResultLpVariablesMedian{} và lớn nhất \ResultLpVariablesMax{}, thời gian giải có trung vị \ResultLpRuntimeMedian{} giây và lớn nhất \ResultLpRuntimeMax{} giây.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{loglogaxis}[
    width=0.8\textwidth, height=6cm,
    xlabel={Số biến của bài toán LP}, ylabel={Thời gian giải (s)},
    grid=major, /pgf/number format/use comma,
]
\addplot[only marks, mark=*, mark size=0.5pt, color=blue!60!black, opacity=0.4]
    table[col sep=comma, x=variables, y=runtime] {data/lp_runtime.csv};
\end{loglogaxis}
\end{tikzpicture}
\caption{Thời gian giải của \ResultLpCount{} bài toán LP cận trên ở Pha 1 (hai họ scenario, hai loại cận, 30 realization) theo số biến.}
\label{fig:lp-runtime}
\end{figure}

\subsection{Trả lời các câu hỏi nghiên cứu}
\label{sec:rq}

\textbf{RQ1 --- đồng tối ưu so với tối ưu từng phần.} Đồng tối ưu quỹ đạo, đường và băng thông (B2) tăng tỷ lệ đúng hạn so với quỹ đạo cố định (B1) trên cả hai họ, với chênh lệch \ResultTwoDiffVzeroBtwoBone{} và \ResultTwoDiffVzeroBhfivezeroBtwoBone{} cùng chiều trên cả mười topology. Ablation chỉ ra phần lợi ích này đến từ khối đường và khối quỹ đạo; khối băng thông gần như không đóng góp khi băng thông đã chia theo backlog.

\textbf{RQ2 --- sửa lịch theo nguy cơ trễ hạn.} Không đo được lợi ích: P $-$ B2 bằng \ResultTwoDiffVzeroPBtwo{} trên v0 và \ResultTwoDiffVzeroBhfivezeroPBtwo{} trên v0-bh50 với $p_{\mathrm{Holm}}=\ResultTwoHolmVzeroPBtwo$, các biến thể chỉ giữ một thao tác hoặc bỏ thông tin backlog cho kết quả như P, và kết luận không đổi khi tăng ngân sách.

\textbf{RQ3 --- khi nào UAV khôi phục việc chuyển dữ liệu.} UAV đưa khả năng kết nối từ \ResultTwoConnVzeroBzero{} lên gần một và tăng tỷ lệ đúng hạn ở mọi mức kết nối mặt đất, với mức tăng tương tự nhau giữa các nhóm; lợi ích biến mất khi mọi nguồn đã có relay mặt đất trong tầm, như ở họ dùng hằng số vật lý khác.

\textbf{RQ4 --- suy giảm theo điều kiện.} Tỷ lệ đúng hạn giảm mạnh nhất khi cảnh báo lớn hơn hoặc hạn ngắn hơn, giảm nhẹ khi tăng số cảnh báo hoặc giảm phổ; thứ tự giữa các phương pháp giữ nguyên. Sai lệch mô hình kênh khi thiết kế, thể hiện qua việc thiết kế trên tốc độ kỳ vọng thay cho realization lấy mẫu, làm P giảm \ResultTwoAblationDiffVzeroPExpectedDesign.

\textbf{RQ5 --- độ bền của kết luận.} Các chênh lệch có ý nghĩa đều cùng chiều trên cả mười topology, và thứ tự B1 thấp hơn B2, P giữ nguyên trên mọi họ độ nhạy không suy biến. Báo cáo chưa thay đổi quy tắc sinh vùng hư hại, điểm nguồn và cảnh báo, nên độ bền theo quy tắc tạo target chưa được kiểm chứng.
```

The section answers RQ1–RQ5 of `docs/issue.md` in its last subsection. Mechanisms that the experiments did not isolate (for example why a connectivity group gains a given amount) are not claimed.

- [ ] **Step 5: Compile and check the report**

```bash
cd docs/report
bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --clean
rm -rf build main.pdf && mkdir build
bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --preview --preview-dir build/preview --verbose > build/compile-verbose.txt 2>&1
uv run python - <<'PY'
import re
import subprocess
import sys
from pathlib import Path

log = Path("build/compile-verbose.txt").read_text(encoding="utf-8", errors="replace")
final_pass = log.split("This is pdfTeX")[-1]
issues = re.findall(r"^.*(?:Overfull|Underfull|LaTeX Warning|Package \w+ Warning|^! ).*$", final_pass, re.M)
info = subprocess.run(["pdfinfo", "main.pdf"], capture_output=True, text=True, check=True).stdout
text = subprocess.run(["pdftotext", "main.pdf", "-"], capture_output=True, text=True, check=True).stdout
pages = re.search(r"Pages:\s+(\d+)", info).group(1)
print(f"final pass issues: {len(issues)}; pages: {pages}; unresolved ??: {text.count('??')}")
for line in issues:
    print("  ", line[:160])
sys.exit(1 if issues or "??" in text else 0)
PY
cd ../..
```

Expected: `final pass issues: 0; pages: 22; unresolved ??: 0` and exit code 0. The script scans only the last pdfLaTeX pass, because earlier passes always report undefined references before biber and the cross-reference pass have run. The `--clean` call first removes the Git-ignored auxiliary files of any earlier build; in a checkout that had been built before, stale `main.aux`, `main.bcf`, and `main.pdf` produced a corrupted PDF whose preview generation failed.

Open the preview pages with Tables 3–7 and Figures 1–2: the ablation table fits the text width, the budget curve shows B2 and P rising to about 0.55 by $E=1000$, and the sensitivity grid has six panels with one shared legend.

- [ ] **Step 6: Commit**

```bash
git add docs/report/sections/experiments.tex docs/report/main.tex docs/report/preamble.tex docs/report/tables/phase2_main.tex docs/report/tables/phase2_statistics.tex docs/report/tables/phase2_ablation.tex docs/report/tables/phase2_connectivity.tex docs/report/tables/phase2_design.tex
git commit -m "docs: report phase 2 experiments and research answers"
```

### Task 5: Abstract, introduction, discussion, conclusion, and README

**Files:**
- Modify: `docs/report/sections/abstract.tex`, `docs/report/sections/discussion.tex`, `docs/report/sections/conclusion.tex` (whole content), `docs/report/sections/introduction.tex`, `README.md`

**Interfaces:**
- Consumes: Task 1 macros; labels from Tasks 2–4; citations `tran2022relay`, `park2026expected`, `rahnemoonfar2023rescuenet`.
- Produces: the Phase 2 report and README commands (spec acceptance criterion 17.2.6).

- [ ] **Step 1: Rewrite the abstract**

Replace the whole content of `docs/report/sections/abstract.tex` with:

```latex
% docs/report/sections/abstract.tex
\begin{abstract}
\noindent
Sau thiên tai, một phần mạng mặt đất bị hỏng và các cảnh báo phát sinh tại hiện trường cần được đưa về trung tâm cứu hộ trước hạn. Báo cáo này nghiên cứu bài toán đồng tối ưu quỹ đạo của một UAV làm relay di động, lựa chọn đường chuyển tiếp qua mạng relay mặt đất có backhaul hữu hạn và phân bổ băng thông, nhằm tối đa tỷ lệ cảnh báo giao đúng hạn dưới kênh không--đất theo mô hình xác suất tầm nhìn thẳng~\cite{huang2025ddatsap}. Chúng tôi chứng minh bài toán là NP-khó ngay cả khi quỹ đạo cố định, xây dựng cận trên quy hoạch tuyến tính hợp lệ theo từng realization kênh, và đề xuất phương pháp tìm kiếm theo ba khối được chấm điểm trên các realization kênh lấy mẫu (B2), cùng biến thể sửa lịch theo nguy cơ trễ hạn (P). Trên \ResultTwoScenarioCount{} kịch bản đánh giá tổng hợp từ Topology Zoo~\cite{knight2011zoo} và SNDlib~\cite{orlowski2010sndlib}, B2 giao đúng hạn \ResultTwoTimelyVzeroBtwo{} số cảnh báo, so với \ResultTwoTimelyVzeroBone{} khi UAV bay theo quỹ đạo cố định và \ResultTwoTimelyVzeroBzero{} khi không có UAV; chênh lệch với quỹ đạo cố định cùng chiều trên cả mười topology ($p_{\mathrm{Holm}}=\ResultTwoHolmVzeroBtwoBone$) và lớn hơn khi backhaul băng hẹp. Cơ chế sửa lịch không cải thiện thêm (P $-$ B2 bằng \ResultTwoDiffVzeroPBtwo), còn mục tiêu công bằng leximin làm tỷ lệ đúng hạn giảm \ResultTwoDiffVzeroBtwoBthree{} so với B2. Ablation cho thấy lợi ích đến từ việc đồng chọn đường và quỹ đạo, và từ việc thiết kế trên các realization lấy mẫu thay cho tốc độ kỳ vọng.

\vspace{0.5em}
\noindent
\textbf{Từ khoá:} UAV relay, quỹ đạo, lựa chọn relay, phân bổ băng thông, hạn giao, cận trên quy hoạch tuyến tính, tìm kiếm cục bộ, kiểm định ghép cặp.
\end{abstract}
```

- [ ] **Step 2: Update the contributions and outline of the introduction**

In `docs/report/sections/introduction.tex` (1 edit):

Edit 1: replace

```latex

\paragraph{Đóng góp của Pha 1.}
Thứ nhất, chúng tôi mô hình hoá bài toán với thời điểm phát sinh, hạn và kích thước của cảnh báo, mạng relay có backhaul hữu hạn, kênh PrLoS và quy ước nhân quả theo slot, cùng một bộ đánh giá độc lập với mọi phương pháp và bộ kịch bản tái lập được (\cref{sec:method}). Thứ hai, chúng tôi cài đặt hai phương pháp cơ sở B0 và B1, xây dựng cận trên quy hoạch tuyến tính theo từng realization kênh kèm chứng minh tính hợp lệ, và đối chiếu cận với MILP trên các instance thu nhỏ (\cref{sec:bound}). Thứ ba, chúng tôi chứng minh bài toán là NP-khó ngay cả khi quỹ đạo cố định và phân tích độ phức tạp tính toán (\cref{sec:hardness,sec:complexity}). Thứ tư, chúng tôi đánh giá hai phương pháp cơ sở và cận trên trên hai họ kịch bản, với backhaul thông thường và backhaul băng hẹp (\cref{sec:experiments}).

Phương pháp phân rã theo khối kết hợp sửa lịch theo độ dư thời gian, các phép ablation, kiểm định thống kê ghép cặp, so sánh với một phương pháp mạnh từ tài liệu và case study RescueNet thuộc Pha 2 và chưa được trình bày ở đây.

Phần còn lại của báo cáo được tổ chức như sau. \Cref{sec:related} điểm qua nghiên cứu liên quan. \Cref{sec:method} trình bày mô hình, phương pháp cơ sở, cận trên và phân tích độ khó. \Cref{sec:experiments} báo cáo kết quả Pha 1, \cref{sec:discussion} thảo luận và \cref{sec:conclusion} kết luận.
```

with

```latex

\paragraph{Đóng góp.}
Thứ nhất, chúng tôi mô hình hoá bài toán với thời điểm phát sinh, hạn và kích thước của cảnh báo, mạng relay có backhaul hữu hạn, kênh PrLoS và quy ước nhân quả theo slot, cùng một bộ đánh giá độc lập với mọi phương pháp và bộ kịch bản tái lập được (\cref{sec:method}). Thứ hai, chúng tôi xây dựng cận trên quy hoạch tuyến tính theo từng realization kênh kèm chứng minh tính hợp lệ, đối chiếu cận với MILP trên các instance thu nhỏ, chứng minh bài toán là NP-khó ngay cả khi quỹ đạo cố định và phân tích độ phức tạp (\cref{sec:bound,sec:hardness,sec:complexity}). Thứ ba, chúng tôi đề xuất phương pháp tìm kiếm theo ba khối --- đường, băng thông theo trọng số và quỹ đạo từ một họ hành trình khả thi --- được chấm điểm trên các realization kênh lấy mẫu dưới một ngân sách đánh giá, cùng biến thể dùng mục tiêu leximin và biến thể sửa lịch theo nguy cơ trễ hạn (\cref{sec:algorithm}). Thứ tư, chúng tôi đánh giá các phương pháp trên hai họ kịch bản với kiểm định ghép cặp ở mức topology, ablation, thí nghiệm ngân sách và độ nhạy, và trả lời năm câu hỏi nghiên cứu (\cref{sec:experiments}). Thứ năm, chúng tôi thực hiện một tổng quan tài liệu có giao thức và chọn phương pháp mạnh từ tài liệu để so sánh (\cref{sec:related}).

Kết quả chính là phương pháp tìm kiếm theo khối tăng tỷ lệ cảnh báo đúng hạn một cách nhất quán so với quỹ đạo cố định, trong khi cơ chế sửa lịch không cải thiện thêm; báo cáo trình bày kết quả này như đo được. Việc so sánh với phương pháp mạnh từ tài liệu và case study RescueNet thuộc giai đoạn cuối của đề tài.

Phần còn lại của báo cáo được tổ chức như sau. \Cref{sec:related} trình bày tổng quan tài liệu. \Cref{sec:method} trình bày mô hình, phương pháp cơ sở, cận trên, phân tích độ khó và phương pháp tìm kiếm. \Cref{sec:experiments} báo cáo kết quả thực nghiệm, \cref{sec:discussion} thảo luận và \cref{sec:conclusion} kết luận.
```

- [ ] **Step 3: Rewrite the discussion and the conclusion**

Replace the whole content of `docs/report/sections/discussion.tex` with:

```latex
% docs/report/sections/discussion.tex
\section{Thảo luận}
\label{sec:discussion}

\paragraph{Vai trò của UAV và của việc đồng tối ưu.}
Một quỹ đạo cố định đơn giản đã nối được mọi nguồn với trung tâm và tăng tỷ lệ đúng hạn so với mạng mặt đất đơn thuần, nhưng đồng tối ưu đường và quỹ đạo mới tạo ra chênh lệch nhất quán trên mọi topology. Ablation cho thấy khối đường và khối quỹ đạo là hai nguồn lợi ích chính, còn trọng số băng thông hầu như không đóng góp. Điều này phù hợp với kết quả đối chiếu MILP ở Pha 1: khi bộ điều phối EDF đã chia băng thông theo backlog, quyết định quan trọng là cảnh báo nào đi đường nào và UAV có mặt ở đâu, không phải cách chia băng thông trong từng nửa slot.

\paragraph{Vì sao sửa lịch không cải thiện thêm.}
P và B2 cho kết quả gần như bằng nhau trên cả hai họ, ở mọi mức ngân sách và trên hầu hết họ độ nhạy. Hai thao tác của P thay đổi đúng những biến mà ba khối của B2 đã tìm kiếm: thao tác đổi đường lặp lại khối đường, còn thao tác chuyển băng thông tác động lên trọng số, vốn ít ảnh hưởng tới kết quả. Một cách giải thích phù hợp với thí nghiệm ngân sách là tìm kiếm của B2 đã bão hoà, nên các thay đổi cục bộ quanh cảnh báo có nguy cơ hiếm khi còn tăng được khoá từ điển; báo cáo không đo trực tiếp tần suất các thao tác được nhận. Kết quả này là âm tính và được báo cáo như đo được; nó không loại trừ các thao tác sửa lịch mạnh hơn, chẳng hạn thay đổi đồng thời đường của nhiều cảnh báo cùng dùng một liên kết nghẽn.

\paragraph{Mục tiêu và thiết kế dưới bất định.}
Mục tiêu leximin của B3 làm tỷ lệ đúng hạn giảm mạnh so với B2 trên cùng loại quỹ đạo, cho thấy tối ưu phần giao được của cảnh báo khó nhất xung đột với việc giao đủ nhiều cảnh báo; mục tiêu max--min của bài tham chiếu vì vậy không phù hợp với bài toán cảnh báo có hạn. Thiết kế trên tốc độ kỳ vọng kém hơn thiết kế trên các realization lấy mẫu, và tăng số realization thiết kế từ năm lên mười tiếp tục cải thiện (\ResultTwoDesignFive{} lên \ResultTwoDesignOnezero) trong khi thời gian lập kế hoạch tăng từ \ResultTwoDesignPlanningFive{} lên \ResultTwoDesignPlanningOnezero{} giây. Kết quả này nhất quán với nhận xét của~\cite{park2026expected} rằng tốc độ tại kênh trung bình ước lượng cao tốc độ kỳ vọng thực.

\paragraph{Độ chặt của cận.}
So với cận hợp, P và B2 còn cách \ResultTwoUnionGapVzeroP{} và \ResultTwoUnionGapVzeroBtwo{} trên v0. Cận LP biết trước kênh, nới lỏng tính nguyên của lựa chọn đường và bỏ qua thứ tự EDF, và đối chiếu MILP ở Pha 1 cho thấy phần lớn khoảng cách trên instance nhỏ đến từ nới lỏng tính nguyên. Vì vậy gap hiện tại có thể phóng đại phần còn cải thiện được, và một cận chặt hơn, chẳng hạn MILP trên instance lớn hơn hoặc bất đẳng thức hợp lệ cho lựa chọn đường, là điều kiện để khẳng định phương pháp đã gần tối ưu hay chưa.

\paragraph{Giới hạn.}
Kết quả dựa trên mô phỏng và không chứng minh hiệu năng thực địa. Mạng backbone gốc được thu nhỏ về vùng \qty{2}{\kilo\metre}, nên chỉ còn mượn cấu trúc. Trạng thái kênh độc lập giữa các slot, kênh mặt đất tất định và công suất cố định. Kiểm định dựa trên mười topology, nên mức ý nghĩa nhỏ nhất đạt được bị giới hạn; độ nhạy chỉ dùng replicate đầu của mỗi topology và quy tắc sinh vùng hư hại, điểm nguồn, cảnh báo chưa được thay đổi. Các phương pháp là heuristic không có bảo đảm tối ưu, và cận chỉ hợp lệ với quỹ đạo và tập candidate đã cho.

\paragraph{Hướng tiếp theo.}
Giai đoạn cuối của đề tài sẽ cài đặt và so sánh với phương pháp của Tran và cộng sự~\cite{tran2022relay} đã chọn ở \cref{sec:related}, bổ sung case study với mục tiêu lấy từ bộ dữ liệu RescueNet~\cite{rahnemoonfar2023rescuenet}, và hoàn thiện gói tái lập.
```

Replace the whole content of `docs/report/sections/conclusion.tex` with:

```latex
% docs/report/sections/conclusion.tex
\section{Kết luận}
\label{sec:conclusion}

Báo cáo đã mô hình hoá bài toán đồng tối ưu quỹ đạo UAV, lựa chọn relay và phân bổ băng thông cho cảnh báo có hạn giao, chứng minh bài toán NP-khó ngay cả khi quỹ đạo cố định, xây dựng cận trên quy hoạch tuyến tính hợp lệ theo từng realization kênh, và đề xuất phương pháp tìm kiếm theo khối được chấm điểm trên các realization kênh lấy mẫu. Trên \ResultTwoScenarioCount{} kịch bản đánh giá với backhaul thông thường, phương pháp này giao đúng hạn \ResultTwoTimelyVzeroBtwo{} số cảnh báo, so với \ResultTwoTimelyVzeroBone{} khi UAV bay theo quỹ đạo cố định và \ResultTwoTimelyVzeroBzero{} khi không có UAV, với chênh lệch cùng chiều trên cả mười topology; khi backhaul băng hẹp, lợi ích còn lớn hơn. Cơ chế sửa lịch theo nguy cơ trễ hạn không cải thiện thêm, mục tiêu leximin làm giảm tỷ lệ đúng hạn, và việc thiết kế trên các realization lấy mẫu quan trọng hơn ngân sách tìm kiếm. Khoảng cách còn lại tới cận hợp cho thấy vẫn còn chỗ để cải thiện, hoặc để siết cận.
```

The discussion states plainly that P does not improve on B2 (spec C Section 14) and labels its explanation as an interpretation.

- [ ] **Step 4: Document the report data command**

Append to `README.md`:

````markdown
Sinh dữ liệu báo cáo Pha 2 (bảng so sánh chính, thống kê, ablation, kết nối, thiết kế, dữ liệu hình ngân sách và độ nhạy, macro số liệu) rồi biên dịch báo cáo bằng script của skill LaTeX:

```bash
uv run python experiments/report_tables.py
uv run python experiments/report_tables_phase2.py
cd docs/report && bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --preview --preview-dir build/preview
```
````

- [ ] **Step 5: Compile and check the report**

```bash
cd docs/report
bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --clean
rm -rf build main.pdf && mkdir build
bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --preview --preview-dir build/preview --verbose > build/compile-verbose.txt 2>&1
uv run python - <<'PY'
import re
import subprocess
import sys
from pathlib import Path

log = Path("build/compile-verbose.txt").read_text(encoding="utf-8", errors="replace")
final_pass = log.split("This is pdfTeX")[-1]
issues = re.findall(r"^.*(?:Overfull|Underfull|LaTeX Warning|Package \w+ Warning|^! ).*$", final_pass, re.M)
info = subprocess.run(["pdfinfo", "main.pdf"], capture_output=True, text=True, check=True).stdout
text = subprocess.run(["pdftotext", "main.pdf", "-"], capture_output=True, text=True, check=True).stdout
pages = re.search(r"Pages:\s+(\d+)", info).group(1)
print(f"final pass issues: {len(issues)}; pages: {pages}; unresolved ??: {text.count('??')}")
for line in issues:
    print("  ", line[:160])
sys.exit(1 if issues or "??" in text else 0)
PY
cd ../..
```

Expected: `final pass issues: 0; pages: 23; unresolved ??: 0` and exit code 0. The script scans only the last pdfLaTeX pass, because earlier passes always report undefined references before biber and the cross-reference pass have run. The `--clean` call first removes the Git-ignored auxiliary files of any earlier build; in a checkout that had been built before, stale `main.aux`, `main.bcf`, and `main.pdf` produced a corrupted PDF whose preview generation failed.

Open every preview page in `docs/report/build/preview/`: no placeholder text, no macro name printed literally, tables and figures inside the margins, and the bibliography ragged-right.

- [ ] **Step 6: Scan for placeholders and run the full suite**

```bash
uv run python - <<'PY'
import re
import sys
from pathlib import Path

PATTERN = re.compile(r"TODO|TBD|\[[^\]]*(?:Thay bằng|Điền|Đoạn|nhận xét về)[^\]]*\]")
hits = [
    f"{path}:{number}: {line.strip()}"
    for folder in ("docs/report/sections", "docs/report/tables")
    for path in sorted(Path(folder).glob("*.tex"))
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
    if PATTERN.search(line)
]
print("\n".join(hits) if hits else "no placeholders")
sys.exit(1 if hits else 0)
PY
```

Expected: `no placeholders` and exit code 0. The scan uses Python rather than `grep`, whose bracket expressions differ between implementations; it reports every `TODO`, `TBD`, or bracketed Vietnamese placeholder such as `[Thay bằng ...]` with its file and line.

Run: `uv run --extra test pytest -q`

Expected: 209 passed, 1 skipped.

- [ ] **Step 7: Commit**

```bash
git add docs/report/sections/abstract.tex docs/report/sections/introduction.tex docs/report/sections/discussion.tex docs/report/sections/conclusion.tex README.md
git commit -m "docs: update report front and back matter for phase 2"
```

---

## Spec Coverage

| Spec requirement | Task |
|---|---|
| 14 `method.tex`: weighted policy and design realizations, B2/P algorithm, leximin definition of B3, complexity and budget bound, reason the blocks are not convex | 2 |
| 14 `related-work.tex`: review table and baseline-selection rationale | 3 |
| 14 `experiments.tex`: main comparison table with intervals, own and union gaps, runtime, evaluate calls; statistics table; ablation table; quality–runtime figure; sensitivity figure grid from CSV; answers to RQ1–RQ5 | 1, 4 |
| 14 abstract, introduction, discussion, conclusion, including a plain statement that P does not beat B2 | 5 |
| 14 build and checks: skill `compile_latex.sh`, no warnings, no `??`, no placeholders, previews inspected | 2–5 |
| 16.2 Phase 2 report data: CSVs and macros match fixture inputs | 1 |
| 17.2.6 report compiles cleanly, answers RQ1–RQ5, contains no placeholders; README documents the Phase 2 commands | 4, 5 (run commands were added by plan C.2 Task 8) |
