import csv
import json
from pathlib import Path

from experiment_fixtures import write_tiny_experiment
from runner.config import load_experiment_config
from runner.experiment import build_tasks, run_experiment, scenario_refs
from runner.tasks import BoundTask, MethodTask

TRAN_SPEC = {"id": "TRAN", "base": "TRAN", "params": {"max_iterations": 3}, "bounds": True}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_tran_gets_no_bound_tasks(tmp_path: Path):
    config = load_experiment_config(write_tiny_experiment(tmp_path, tmp_path / "out", 1, ("B1", TRAN_SPEC), crosscheck=False))
    tasks = build_tasks(config, scenario_refs(config))
    assert {(task.variant, task.trajectory_method) for task in tasks if isinstance(task, BoundTask)} == {("trajectory", "B1")}
    assert sum(isinstance(task, MethodTask) and task.spec.id == "TRAN" for task in tasks) == 2


def test_tran_runs_in_the_experiment_runner_with_in_task_bounds(tmp_path: Path):
    output = tmp_path / "out"
    config_path = write_tiny_experiment(tmp_path, output, 1, (TRAN_SPEC,), crosscheck=False)
    assert run_experiment(load_experiment_config(config_path), config_path) == 0
    rows, bounds = _rows(output / "method_realizations.csv"), _rows(output / "bounds.csv")
    assert {row["method"] for row in rows} == {"TRAN"} and len(rows) == 4
    assert {row["trajectory_method"] for row in bounds} == {"TRAN"} and len(bounds) == 4
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "complete"
