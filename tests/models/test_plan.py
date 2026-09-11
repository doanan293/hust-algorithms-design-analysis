import numpy as np

from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, StaticSchedule, stationary_trajectory, validate_plan
from models.scenario import UAV_ID, scenario_from_dict


def _setup(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    return scenario, candidate_paths(scenario)


def test_stationary_equal_split_plan_is_valid(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0}, EqualSplitBacklogged())
    assert validate_plan(scenario, candidates, plan) == []


def test_trajectory_endpoint_and_speed_violations(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    trajectory = stationary_trajectory(scenario)
    trajectory[0] = [1000.0, 1001.0]
    trajectory[5] = [1100.0, 1000.0]
    violations = validate_plan(scenario, candidates, Plan(trajectory, {"alert-0000": 0}, EqualSplitBacklogged()))
    assert any("q[0] is not the start point" in item for item in violations)
    assert any("slot 4 moves" in item for item in violations)
    assert any("slot 5 moves" in item for item in violations)


def test_path_choice_violations(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    trajectory = stationary_trajectory(scenario)
    assert "path_choice: missing alert alert-0000" in validate_plan(
        scenario, candidates, Plan(trajectory, {}, EqualSplitBacklogged())
    )
    assert any(
        "invalid candidate index 3" in item
        for item in validate_plan(scenario, candidates, Plan(trajectory, {"alert-0000": 3}, EqualSplitBacklogged()))
    )


def test_static_schedule_violations(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    trajectory = stationary_trajectory(scenario)
    schedule = StaticSchedule(
        {
            ("access", "s00", "n1"): np.full(20, 6e5),
            ("access", "s00", "n0"): np.full(20, 6e5),
            ("down", UAV_ID, "n1"): np.full(20, -1.0),
            ("down", UAV_ID, "n9"): np.zeros(20),
        }
    )
    violations = validate_plan(scenario, candidates, Plan(trajectory, {"alert-0000": 0}, schedule))
    assert any("slot 0 access total" in item for item in violations)
    assert any("finite non-negative" in item for item in violations)
    assert any("unknown radio link" in item for item in violations)


def test_static_schedule_downlink_cap(raw_scenario):
    raw_scenario["uav"]["max_active_downlinks"] = 1
    raw_scenario["paths"]["max_ground"] = 0
    scenario, candidates = _setup(raw_scenario)
    downlinks = [link for path in candidates["alert-0000"] for link in path.links if link[0] == "down"]
    schedule = StaticSchedule({link: np.full(20, 1e5) for link in downlinks[:2]})
    violations = validate_plan(
        scenario, candidates, Plan(stationary_trajectory(scenario), {"alert-0000": 0}, schedule)
    )
    assert any("has 2 active downlinks" in item for item in violations)
