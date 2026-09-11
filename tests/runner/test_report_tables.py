import csv
import importlib.util
import json
from pathlib import Path

from runner.provenance import source_hash

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "report_tables.py"
SPEC = importlib.util.spec_from_file_location("report_tables", SCRIPT)
report_tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report_tables)


def _write_results(root: Path, source: str) -> Path:
    results = root / "results"
    results.mkdir()
    manifest = {"status": "complete", "source_hash": source, "started_utc": "2026-09-11T00:00:00+00:00",
                "finished_utc": "2026-09-11T00:12:30+00:00", "workers": 12,
                "environment": {"cpu_count": 12, "python": "3.12.3", "scipy": "1.18.1", "highs": "1.12.0"}}
    (results / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = {"set_id": "v0", "method": "B1", "scenario_count": "30", "timely_ratio_mean": "0.25", "timely_ratio_ci_low": "0.2",
               "timely_ratio_ci_high": "0.3", "conn_ratio_mean": "0.9", "bound_ratio_mean": "0.5", "gap_mean": "0.5",
               "gap_undefined_count": "0", "shortfall_mean": "20", "planning_runtime_s_mean": "0.4",
               "evaluation_runtime_s_per_realization_mean": "0.02", "bound_runtime_s_mean": "1.5"}
    milp = {"set_id": "v0", "scenario_id": "Agis-r0", "alert_count": "10", "lp_ub": "4.5", "lp_status": "optimal",
            "milp_incumbent": "4.0", "milp_dual_bound": "4.0", "milp_status": "optimal", "milp_runtime_s": "3.2",
            "milp_plan_static_value": "1.0", "milp_plan_equal_split_value": "3.0", "milp_plan_value": "3.0", "method_value": "4.0",
            "tangent_max_overestimate": "0.0123"}
    method = {"set_id": "v0", "scenario_id": "Agis-r0", "topology_id": "Agis", "method": "B1", "realization_id": "0", "alert_count": "40",
              "timely_count": "10", "timely_ratio": "0.25", "source_count": "15", "connected_count": "15", "conn_ratio": "1.0",
              "shortfall": "20", "planning_runtime_s": "30", "evaluation_runtime_s": "1.55", "evaluate_calls": "1"}
    bound = {"set_id": "v0", "scenario_id": "Agis-r0", "topology_id": "Agis", "variant": "trajectory", "trajectory_method": "B1",
             "realization_id": "0", "value": "20", "ratio": "0.5", "status": "optimal", "runtime_s": "1.25",
             "variables": "30000", "rows": "40000", "nonzeros": "200000"}
    for name, row in (("summary.csv", summary), ("milp_crosscheck.csv", milp), ("bounds.csv", bound), ("method_realizations.csv", method)):
        with (results / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    return results


def _semicolon_rows(path: Path) -> list[list[str]]:
    return [line.split(";") for line in path.read_text(encoding="utf-8").splitlines()]


def test_report_files_cover_config_values_and_results(tmp_path: Path):
    results = _write_results(tmp_path, source_hash())
    output = tmp_path / "data"
    assert report_tables.main(["--results", str(results), "--output", str(output)]) == 0
    params = _semicolon_rows(output / "params.csv")
    assert params[0] == ["parameter", "symbol", "value"] and all(len(row) == 3 for row in params)
    text = (output / "params.csv").read_text(encoding="utf-8")
    for expected in (
        "\\qty{2000}{\\metre}", "10 / 3", "3 / 2", ";30\n", "\\qty{1}{\\second} / 150", "\\num{0.5}", "\\qty{100}{\\metre}",
        "\\qty{50}{\\metre\\per\\second}", "\\qty{0.1}{\\watt} / 2", "\\qty{1}{\\mega\\hertz}", "\\qty{-140}{\\dBm\\per\\hertz}",
        "\\num{9.61}", "\\num{0.16}", "\\num{0.2}", "\\num{2.2}", "\\num{3.3}", "\\qty{-50}{\\dB}", "\\num{2.8}",
        "\\qty{10}{\\kilo\\bit\\per\\second}", "3 (2)", "3 / \\qty{600}{\\metre} / \\qty{400}{\\metre}", "15 / \\qty{150}{\\metre}",
        "40 / 20--60 slot", "\\qty{250}{\\kilo\\bit}", "\\texttt{abilene}", "\\num{0.60} / \\num{0.05}",
        "\\qty{1000}{\\kilo\\bit\\per\\second} / \\qty{50}{\\kilo\\bit\\per\\second}", "10 / 10000",
    ):
        assert expected in text, expected
    main_rows = _semicolon_rows(output / "phase1_main.csv")
    assert main_rows[1][:5] == ["v0", "\\qty{1000}{\\kilo\\bit\\per\\second}", "B1", "\\num{0.250}", "[\\num{0.200}, \\num{0.300}]"]
    milp_rows = _semicolon_rows(output / "phase1_milp.csv")
    assert milp_rows[1][:5] == ["Agis-r0", "10", "\\num{4}", "\\num{3}", "\\num{4}"]
    assert milp_rows[1][7] == "tối ưu" and len(milp_rows[1]) == 10
    assert (output / "lp_runtime.csv").read_text(encoding="utf-8").splitlines() == ["variables,runtime", "30000,1.250000"]
    macros = (output / "phase1_results.tex").read_text(encoding="utf-8")
    assert "\\newcommand{\\ResultTimelyVzeroBone}{\\num{0.250}}" in macros
    assert "\\newcommand{\\ResultComputeMinutes}{\\num{1}}" in macros
    assert "\\newcommand{\\ResultMilpMethodAtOptimumCount}{1}" in macros
    assert "\\newcommand{\\ResultMilpPlanAtOptimumCount}{0}" in macros
    assert "\\newcommand{\\ResultHighsVersion}{1.12.0}" in macros
    assert "\\newcommand{\\ResultMilpStaticTotal}{1}" in macros
    assert "\\newcommand{\\ResultMilpEqualSplitTotal}{3}" in macros
    assert "\\newcommand{\\ResultBoneWinsVzero}" not in macros
    assert report_tables.macro_name("Gap", "v0-bh50", "B0") == "ResultGapVzeroBhfivezeroBzero"
    assert report_tables.macro_name("BoundTime", "v0", "B1") == "ResultBoundTimeVzeroBone"
    assert report_tables.macro_name("BoneWins", "v0-bh50") == "ResultBoneWinsVzeroBhfivezero"


def test_report_tables_refuse_results_from_other_source_code(tmp_path: Path, capsys):
    results = _write_results(tmp_path, "0" * 64)
    assert report_tables.main(["--results", str(results), "--output", str(tmp_path / "data")]) == 1
    assert "different source code" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()
