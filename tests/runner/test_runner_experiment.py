import csv
import json
from pathlib import Path

from runner.config import load_experiment_config
from runner.experiment import run_experiment
from experiment_fixtures import write_tiny_experiment

RUNTIME_COLUMNS = {"planning_runtime_s", "evaluation_runtime_s", "runtime_s", "milp_runtime_s"}


def _rows_without_runtime(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: value for key, value in row.items() if key not in RUNTIME_COLUMNS} for row in csv.DictReader(stream)]


def test_results_do_not_depend_on_worker_count(tmp_path: Path):
    one = load_experiment_config(write_tiny_experiment(tmp_path, tmp_path / "one"))
    two = load_experiment_config(write_tiny_experiment(tmp_path, tmp_path / "two"))
    assert run_experiment(one, Path("one.yaml"), workers=1) == 0
    assert run_experiment(two, Path("two.yaml"), workers=2) == 0
    for name in ("method_realizations.csv", "bounds.csv", "milp_crosscheck.csv"):
        assert _rows_without_runtime(tmp_path / "one" / name) == _rows_without_runtime(tmp_path / "two" / name)
    methods = _rows_without_runtime(tmp_path / "one" / "method_realizations.csv")
    assert len(methods) == 2 * 2 * 2
    bounds = _rows_without_runtime(tmp_path / "one" / "bounds.csv")
    assert {(row["variant"], row["trajectory_method"]) for row in bounds} == {("ground_only", ""), ("trajectory", "B1")}
    with (tmp_path / "one" / "summary.csv").open(newline="", encoding="utf-8") as stream:
        assert [(row["set_id"], row["method"]) for row in csv.DictReader(stream)] == [("tiny", "B0"), ("tiny", "B1")]
    manifest = json.loads((tmp_path / "one" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for field in (
        "experiment_id", "config_hash", "git_commit", "git_dirty", "source_hash", "scenario_manifest_sha256",
        "environment", "workers", "started_utc", "finished_utc", "command", "task_count",
    ):
        assert field in manifest
    assert set(manifest["environment"]) == {"python", "numpy", "scipy", "highs", "platform", "cpu_count"}


def test_rerun_skips_current_shards_and_recomputes_stale_ones(tmp_path: Path, capsys):
    config = load_experiment_config(write_tiny_experiment(tmp_path, tmp_path / "out"))
    assert run_experiment(config, Path("tiny.yaml")) == 0
    first = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert first["executed_task_count"] == first["task_count"]
    assert run_experiment(config, Path("tiny.yaml")) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["executed_task_count"] == 0
    shard = next((tmp_path / "out" / "shards").glob("bound__*.json"))
    payload = json.loads(shard.read_text(encoding="utf-8"))
    payload["task_hash"] = "stale"
    shard.write_text(json.dumps(payload), encoding="utf-8")
    assert run_experiment(config, Path("tiny.yaml")) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["executed_task_count"] == 1


def test_search_methods_bound_their_own_trajectories_and_join_the_union_bound(tmp_path: Path):
    methods = [
        "B1",
        {"id": "S", "base": "search", "params": {"budget": 20, "design_ids": [0]}, "bounds": True},
        {"id": "T", "base": "search", "params": {"budget": 10, "trajectory_block": False}},
    ]
    config = load_experiment_config(write_tiny_experiment(tmp_path, tmp_path / "out", methods=methods, crosscheck=False))
    assert run_experiment(config, Path("tiny.yaml"), workers=2) == 0
    output = tmp_path / "out"
    assert not (output / "milp_crosscheck.csv").exists()
    bounds = _rows_without_runtime(output / "bounds.csv")
    assert sorted({(row["variant"], row["trajectory_method"]) for row in bounds}) == [("trajectory", "B1"), ("trajectory", "S")]
    assert len([row for row in bounds if row["trajectory_method"] == "S"]) == 2 * 2
    assert not list((output / "shards").glob("bound__*__S__*.json"))
    calls = {row["method"]: int(row["evaluate_calls"]) for row in _rows_without_runtime(output / "method_realizations.csv")}
    assert calls["B1"] == 1 and 2 <= calls["S"] <= 21 and 2 <= calls["T"] <= 11
    with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
        summary = {row["method"]: row for row in csv.DictReader(stream)}
    assert summary["T"]["bound_ratio_mean"] == "nan" and summary["S"]["bound_ratio_mean"] != "nan"
    for method in ("B1", "S", "T"):
        assert float(summary[method]["union_bound_ratio_mean"]) >= float(summary["B1"]["bound_ratio_mean"]) - 1e-12
        assert float(summary[method]["union_bound_ratio_mean"]) >= float(summary["S"]["bound_ratio_mean"]) - 1e-12
