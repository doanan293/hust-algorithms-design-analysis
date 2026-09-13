import csv
import importlib.util
import json
from pathlib import Path
import sys

from experiment_fixtures import write_tiny_experiment
from runner.config import load_experiment_config
from runner.experiment import run_experiment

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
SPEC = importlib.util.spec_from_file_location("reproduce", EXPERIMENTS / "reproduce.py")
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


def _row(**values):
    base = {"set_id": "v0", "scenario_id": "A-r0", "method": "B1", "realization_id": "0", "timely_count": "3", "connected_count": "7",
            "evaluate_calls": "1", "shortfall": "0.5", "planning_runtime_s": "1.0"}
    return {**base, **values}


def test_method_rows_compare_non_runtime_columns_with_tolerances():
    rule = reproduce.RULES["method_realizations.csv"]
    reference = [_row()]
    assert reproduce.compare_rows(reference, [_row(planning_runtime_s="99", shortfall="0.5000000001")], rule) == []
    problems = reproduce.compare_rows(reference, [_row(timely_count="4", shortfall="0.51"), _row(scenario_id="B-r0")], rule)
    assert problems == ["('v0', 'A-r0', 'B1', '0') timely_count: 3 -> 4", "('v0', 'A-r0', 'B1', '0') shortfall: 0.5 -> 0.51",
                        "('v0', 'B-r0', 'B1', '0'): missing from the reference"]


def test_bounds_use_relative_tolerance_and_milp_incumbents_count_only_when_optimal():
    bounds = reproduce.RULES["bounds.csv"]
    key = {"set_id": "v0", "scenario_id": "A-r0", "variant": "trajectory", "trajectory_method": "B1", "realization_id": "0", "status": "optimal"}
    assert reproduce.compare_rows([{**key, "value": "1000.0"}], [{**key, "value": "1000.0005"}], bounds) == []
    assert len(reproduce.compare_rows([{**key, "value": "1000.0"}], [{**key, "value": "1000.01"}], bounds)) == 1
    milp = reproduce.RULES["milp_crosscheck.csv"]
    row = {"set_id": "v0", "scenario_id": "A-r0", "lp_status": "optimal", "lp_ub": "5.0", "milp_status": "time_limit", "milp_incumbent": "3.0"}
    assert reproduce.compare_rows([row], [{**row, "milp_incumbent": "2.0"}], milp) == []
    optimal = {**row, "milp_status": "optimal"}
    assert reproduce.compare_rows([optimal], [{**optimal, "milp_incumbent": "2.0"}], milp) == ["('v0', 'A-r0') milp_incumbent: 3.0 -> 2.0"]


def test_tables_level_reports_failed_commands_and_byte_differences(tmp_path: Path):
    committed = tmp_path / "committed"
    committed.mkdir()
    (committed / "same.csv").write_text("a\n", encoding="utf-8")
    (committed / "changed.csv").write_text("b\n", encoding="utf-8")
    writer = "import pathlib, sys; folder = pathlib.Path(sys.argv[1]); (folder / 'same.csv').write_text('a\\n'); (folder / 'changed.csv').write_text('c\\n')"
    commands = (("writer", ["-c", writer, "{data}"]), ("failing", ["-c", "import sys; sys.exit('broken')"]))
    checks = reproduce.tables_level(tmp_path / "work", commands, report_data=committed)
    by_name = {check.name: check for check in checks}
    assert by_name["writer"].passed and not by_name["failing"].passed and "broken" in by_name["failing"].detail
    assert by_name["report data/same.csv"].passed and not by_name["report data/changed.csv"].passed


def test_empty_reruns_fail_only_when_the_reference_has_rows(tmp_path: Path):
    header = "set_id,scenario_id,method,realization_id,timely_count,connected_count,evaluate_calls,shortfall\n"
    for name, body in (("reference", "v0,A-r0,B1,0,3,7,1,0.5\n"), ("candidate", "")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "method_realizations.csv").write_text(header + body, encoding="utf-8")
    [check] = reproduce.compare_result_dirs(tmp_path / "reference", tmp_path / "candidate")
    assert (check.passed, check.detail) == (False, "the rerun produced no rows")
    (tmp_path / "reference" / "method_realizations.csv").write_text(header, encoding="utf-8")
    [check] = reproduce.compare_result_dirs(tmp_path / "reference", tmp_path / "candidate")
    assert (check.passed, check.detail) == (True, "0 rows match")


def test_recorded_scenario_hashes_collect_every_manifest(tmp_path: Path):
    for name, digest in (("a", "1" * 64), ("b", "2" * 64)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps({"scenario_manifest_sha256": {"v0": digest, "rescuenet": "3" * 64}}), encoding="utf-8")
    assert reproduce.recorded_scenario_hashes(tmp_path) == {"v0": {"1" * 64, "2" * 64}, "rescuenet": {"3" * 64}}


def test_experiments_level_reruns_a_subset_and_detects_changed_results(tmp_path: Path, capsys):
    config_path = write_tiny_experiment(tmp_path, tmp_path / "reference", 1, ("B0", "B1"), crosscheck=False)
    assert run_experiment(load_experiment_config(config_path), config_path) == 0
    checks = reproduce.experiments_level(tmp_path / "work", [config_path], "TinyA", 1)
    assert checks and all(check.passed for check in checks)
    assert {check.name.rsplit("/", 1)[1] for check in checks} == {"method_realizations.csv", "bounds.csv"}
    assert [check.detail for check in checks][0] == "4 rows match"
    skipped = reproduce.experiments_level(tmp_path / "work-none", [config_path], "Missing", 1)
    assert [(check.passed, check.detail) for check in skipped] == [(True, "skipped: no eval scenario of topology Missing")]
    development = tmp_path / "development.yaml"
    development.write_text(config_path.read_text(encoding="utf-8").replace("split: eval", "split: dev"), encoding="utf-8")
    other_split = reproduce.experiments_level(tmp_path / "work-dev", [development], "TinyA", 1)
    assert [(check.passed, check.detail) for check in other_split] == [(True, "skipped: no dev scenario of topology TinyA")]
    rows = reproduce.read_rows(tmp_path / "reference" / "method_realizations.csv")
    rows[0]["timely_count"] = str(int(rows[0]["timely_count"]) + 1)
    with (tmp_path / "reference" / "method_realizations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert reproduce.main(["--compare", str(tmp_path / "reference"), str(tmp_path / "work" / "experiment-reference")]) == 1
    printed = capsys.readouterr().out
    assert "FAIL " in printed and "timely_count" in printed and "PASS " in printed
