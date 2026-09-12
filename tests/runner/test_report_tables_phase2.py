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
