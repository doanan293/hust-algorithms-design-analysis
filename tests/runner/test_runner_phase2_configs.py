from pathlib import Path

import pytest

from optimization.bcd import SearchParams
from runner.config import load_experiment_config
from runner.experiment import build_tasks, scenario_refs

REPAIR = {"operation1": True, "operation2": True}
MAIN_SEARCH = {
    "B2": SearchParams(),
    "B3": SearchParams(objective="leximin"),
    "P": SearchParams(**REPAIR),
    "P_op1": SearchParams(operation1=True),
    "P_op2": SearchParams(operation2=True),
    "P_fixed_paths": SearchParams(path_block=False, operation1=True),
    "P_fixed_trajectory": SearchParams(trajectory_block=False, **REPAIR),
    "P_equal_bandwidth": SearchParams(bandwidth_block=False, operation2=True),
    "P_no_backlog": SearchParams(backlog=False, **REPAIR),
    "P_expected_design": SearchParams(design_ids=(), **REPAIR),
}
SENSITIVITY_SETS = [
    "v0", "v0-bh50", "sens-bh100", "sens-lref125k", "sens-lref500k", "sens-lref1m", "sens-physical", "sens-deadline-short",
    "sens-deadline-long", "sens-btot05", "sens-btot2", "sens-alerts20", "sens-alerts60", "sens-k2", "sens-k5",
]


def _search(config):
    return {spec.id: spec.search_params() for spec in config.methods if spec.base == "search"}


def test_main_config_runs_every_method_and_ablation_on_both_sets():
    config = load_experiment_config(Path("configs/experiments/phase2_main.yaml"))
    assert [(item.set_id, item.replicates) for item in config.scenario_sets] == [("v0", None), ("v0-bh50", None)]
    assert (config.split, config.realization_ids, config.crosscheck, config.output_dir) == ("eval", tuple(range(30)), None, Path("results/phase2/main"))
    assert [spec.id for spec in config.methods] == ["B0", "B1", *MAIN_SEARCH]
    assert _search(config) == MAIN_SEARCH
    assert {spec.id for spec in config.methods if spec.bounds} == {"B2", "B3", "P"}


def test_budget_config_sweeps_b2_and_p():
    config = load_experiment_config(Path("configs/experiments/phase2_budget.yaml"))
    budgets = (250, 500, 1000, 2000, 4000)
    expected = {f"B2_b{budget}": SearchParams(budget=budget) for budget in budgets}
    expected |= {f"P_b{budget}": SearchParams(budget=budget, **REPAIR) for budget in budgets}
    assert [item.set_id for item in config.scenario_sets] == ["v0"] and config.scenario_sets[0].replicates is None
    assert _search(config) == expected and not any(spec.bounds for spec in config.methods)


def test_sensitivity_and_design_configs_use_one_replicate_per_topology():
    sensitivity = load_experiment_config(Path("configs/experiments/phase2_sensitivity.yaml"))
    assert [(item.set_id, item.replicates) for item in sensitivity.scenario_sets] == [(name, (0,)) for name in SENSITIVITY_SETS]
    assert [spec.id for spec in sensitivity.methods] == ["B1", "B2", "P"]
    assert _search(sensitivity) == {"B2": SearchParams(), "P": SearchParams(**REPAIR)}
    design = load_experiment_config(Path("configs/experiments/phase2_design.yaml"))
    assert [(item.set_id, item.replicates) for item in design.scenario_sets] == [("v0", (0,))]
    assert _search(design) == {
        "P_design_expected": SearchParams(design_ids=(), **REPAIR),
        "P_design1": SearchParams(design_ids=(0,), **REPAIR),
        "P_design10": SearchParams(design_ids=tuple(range(10)), **REPAIR),
    }


@pytest.mark.parametrize(
    ("name", "tasks"),
    [("phase2_main", 60 * (12 + 30 + 30)), ("phase2_budget", 30 * 10), ("phase2_sensitivity", 15 * 10 * (3 + 30)), ("phase2_design", 10 * 3)],
)
def test_phase2_configs_expand_to_their_task_counts(name, tasks):
    config = load_experiment_config(Path(f"configs/experiments/{name}.yaml"))
    if not all(item.manifest.exists() and item.scenario_dir.exists() for item in config.scenario_sets):
        pytest.skip("scenario sets are not generated in this checkout")
    keys = [task.key for task in build_tasks(config, scenario_refs(config))]
    assert len(keys) == len(set(keys)) == tasks
