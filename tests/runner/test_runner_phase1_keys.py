import csv
from pathlib import Path

import pytest

from runner.config import load_experiment_config
from runner.experiment import build_tasks, scenario_refs

RESULTS = Path("results/phase1")


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_phase1_config_yields_the_phase1_task_keys():
    config = load_experiment_config(Path("configs/experiments/phase1.yaml"))
    if not all(item.manifest.exists() and item.scenario_dir.exists() for item in config.scenario_sets):
        pytest.skip("scenario sets are not generated in this checkout")
    expected = {f"method__{row['set_id']}__{row['scenario_id']}__{row['method']}" for row in _rows("method_realizations.csv")}
    expected |= {
        f"bound__{row['set_id']}__{row['scenario_id']}__{row['variant']}__{row['trajectory_method'] or 'none'}__r{int(row['realization_id']):03d}"
        for row in _rows("bounds.csv")
    }
    expected |= {f"milp__{row['set_id']}__{row['scenario_id']}" for row in _rows("milp_crosscheck.csv")}
    keys = [task.key for task in build_tasks(config, scenario_refs(config))]
    assert len(keys) == len(set(keys)) == len(expected) == 3725
    assert set(keys) == expected
