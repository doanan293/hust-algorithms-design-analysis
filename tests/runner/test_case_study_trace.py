import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from experiment_fixtures import write_tiny_experiment
from optimization.bcd import SearchParams
from runner.config import load_experiment_config
from runner.experiment import run_experiment

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "case_study_trace.py"
SPEC = importlib.util.spec_from_file_location("case_study_trace", SCRIPT)
case_study_trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(case_study_trace)


def _rows(values):
    return [{"scenario_id": scenario_id, "method": method, "timely_ratio": str(value)} for scenario_id, method, value in values]


def test_case_study_configuration_uses_the_pilot_choice():
    config = load_experiment_config(Path("configs/experiments/phase2_case_study.yaml"))
    choice = json.loads(Path("results/phase2/tran_pilot/choice.json").read_text(encoding="utf-8"))
    assert [item.set_id for item in config.scenario_sets] == ["rescuenet"] and config.split == "eval"
    assert [(spec.id, spec.base, spec.bounds) for spec in config.methods] == [
        ("B0", "B0", False), ("B1", "B1", False), ("B2", "search", True), ("P", "search", True), ("TRAN", "TRAN", True),
    ]
    assert config.methods[3].search_params() == SearchParams(operation1=True, operation2=True)
    tran = config.methods[4].tran_params()
    assert (tran.theta, tran.mu, tran.block_slots) == (choice["theta"], choice["mu"], choice["block_slots"])
    assert (config.realization_ids, config.workers, config.output_dir) == (tuple(range(30)), 12, Path("results/phase2/case_study"))


def test_representative_scenario_is_closest_to_the_median_of_scenario_means():
    rows = _rows([("A-r0", "P", 0.2), ("A-r0", "P", 0.4), ("B-r0", "P", 0.5), ("C-r0", "P", 0.9), ("B-r0", "B1", 0.0)])
    assert case_study_trace.representative_scenario(rows, "P") == "B-r0"
    assert case_study_trace.representative_scenario(_rows([("B-r0", "P", 0.75), ("A-r0", "P", 0.25)]), "P") == "A-r0"


def test_progress_counts_timely_deliveries_by_slot():
    realizations = [
        SimpleNamespace(delivery_slot={"a": 1, "b": None, "c": 3}, timely={"a": True, "b": False, "c": True}),
        SimpleNamespace(delivery_slot={"a": 0, "b": 2, "c": None}, timely={"a": True, "b": True, "c": False}),
    ]
    assert case_study_trace.progress_counts(realizations, 5) == [0.5, 1.0, 1.5, 2.0, 2.0]


def test_trace_matches_the_runner_writes_map_data_and_stops_on_a_mismatch(tmp_path: Path, capsys):
    output = tmp_path / "out"
    methods = ("B1", {"id": "TRAN", "base": "TRAN", "params": {"max_iterations": 2}})
    config_path = write_tiny_experiment(tmp_path, output, 1, methods, crosscheck=False)
    assert run_experiment(load_experiment_config(config_path), config_path) == 0
    targets = tmp_path / "targets.csv"
    targets.write_text("target_id,x_m,y_m,damage_class\nt0,1230.0,1000.0,4\nt1,100.0,100.0,5\n", encoding="utf-8")
    trace = tmp_path / "trace"
    arguments = ["--config", str(config_path), "--results", str(output), "--targets", str(targets), "--output", str(trace),
                 "--methods", "B1", "TRAN", "--anchor", "B1"]
    assert case_study_trace.main(arguments) == 0
    summary = json.loads((trace / "trace.json").read_text(encoding="utf-8"))
    assert sorted(summary["methods"]) == ["B1", "TRAN"] and summary["realizations"] == 2
    counts = {}
    for name in ("targets", "nodes", "edges", "trajectories", "progress"):
        with (trace / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
            counts[name] = len(list(csv.DictReader(stream)))
    assert counts == {"targets": 2, "nodes": 4, "edges": 3, "trajectories": 42, "progress": 40}
    with (output / "method_realizations.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["timely_ratio"] = "0.123" if row["method"] == "B1" else row["timely_ratio"]
    with (output / "method_realizations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert case_study_trace.main(arguments) == 1
    assert "trace differs from the runner for ['B1']" in capsys.readouterr().err
