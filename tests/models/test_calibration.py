import importlib.util
from pathlib import Path

import numpy as np

from models.evaluate import REALIZED, evaluate
from models.paths import candidate_paths
from models.plan import validate_plan
from models.scenario import scenario_from_dict

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "calibrate_v0.py"
SPEC = importlib.util.spec_from_file_location("calibrate_v0", SCRIPT)
calibrate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibrate)


def test_zone_tour_visits_zones_and_is_feasible(raw_scenario):
    raw_scenario["time"]["num_slots"] = 120
    raw_scenario["damage_zones"] = [
        {"x_m": 1800.0, "y_m": 1000.0, "radius_m": 400.0},
        {"x_m": 200.0, "y_m": 200.0, "radius_m": 400.0},
    ]
    scenario = scenario_from_dict(raw_scenario)
    candidates = candidate_paths(scenario)
    plan = calibrate.zone_tour_plan(scenario, candidates)
    assert validate_plan(scenario, candidates, plan) == []
    for zone in scenario.damage_zones:
        assert np.min(np.linalg.norm(plan.trajectory - [zone.x_m, zone.y_m], axis=1)) < 1e-6


def test_zone_tour_drops_zones_that_do_not_fit(raw_scenario):
    raw_scenario["damage_zones"] = [
        {"x_m": 1200.0, "y_m": 1000.0, "radius_m": 400.0},
        {"x_m": 100.0, "y_m": 100.0, "radius_m": 400.0},
    ]
    scenario = scenario_from_dict(raw_scenario)
    trajectory = calibrate.zone_tour_trajectory(scenario)
    assert trajectory.shape == (21, 2)
    assert np.min(np.linalg.norm(trajectory - [1200.0, 1000.0], axis=1)) < 1e-6
    assert np.min(np.linalg.norm(trajectory - [100.0, 100.0], axis=1)) > 100.0


def test_calibration_plans_evaluate_feasibly(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    candidates = candidate_paths(scenario)
    for build in calibrate.PLANS.values():
        result = evaluate(scenario, build(scenario, candidates), REALIZED, (0,), candidates=candidates)
        assert result.feasible


def test_calibration_uses_the_baseline_zone_tour():
    from baselines import trajectories

    assert calibrate.zone_tour_trajectory is trajectories.zone_tour_trajectory
