import numpy as np

from models.paths import CandidatePath, candidate_paths
from models.plan import EqualSplitBacklogged, Plan, StaticSchedule, stationary_trajectory, validate_plan
from models.scenario import UAV_ID, scenario_from_dict

ACCESS_N1 = ("access", "s00", "n1")
BACKHAUL_N1 = ("backhaul", "n1", "n0")


def _setup(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    return scenario, candidate_paths(scenario)


def _plan(scenario, path, bandwidth=None):
    return Plan(stationary_trajectory(scenario), {"alert-0000": path}, bandwidth or EqualSplitBacklogged())


def test_from_choice_matches_a_plan_built_from_the_same_paths(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    trajectory = stationary_trajectory(scenario)
    plan = Plan.from_choice(trajectory, candidates, {"alert-0000": 2}, EqualSplitBacklogged())
    assert plan.paths == {"alert-0000": candidates["alert-0000"][2]}
    assert Plan.from_choice(trajectory, candidates, {"alert-0000": None}, EqualSplitBacklogged()).paths == {"alert-0000": None}
    assert validate_plan(scenario, plan) == []


def test_stationary_equal_split_plan_is_valid(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    assert validate_plan(scenario, _plan(scenario, candidates["alert-0000"][0])) == []
    assert validate_plan(scenario, _plan(scenario, None)) == []


def test_trajectory_endpoint_and_speed_violations(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    trajectory = stationary_trajectory(scenario)
    trajectory[0] = [1000.0, 1001.0]
    trajectory[5] = [1100.0, 1000.0]
    violations = validate_plan(scenario, Plan(trajectory, {"alert-0000": candidates["alert-0000"][0]}, EqualSplitBacklogged()))
    assert any("q[0] is not the start point" in item for item in violations)
    assert any("slot 4 moves" in item for item in violations)
    assert any("slot 5 moves" in item for item in violations)


def test_missing_and_unknown_alerts(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    trajectory = stationary_trajectory(scenario)
    assert "paths: missing alert alert-0000" in validate_plan(scenario, Plan(trajectory, {}, EqualSplitBacklogged()))
    plan = Plan(trajectory, {"alert-0000": None, "alert-9999": None}, EqualSplitBacklogged())
    assert "paths: unknown alert alert-9999" in validate_plan(scenario, plan)


def test_path_structure_violations(raw_scenario):
    scenario, _ = _setup(raw_scenario)
    cases = {
        "wrong source": CandidatePath("ground", "n1", (("access", "s01", "n1"), BACKHAUL_N1), 1.0),
        "ground kind on a uav path": CandidatePath("ground", "n1", (("access", "s00", UAV_ID), ("down", UAV_ID, "n1"), BACKHAUL_N1), 1.0),
        "downlink to another node": CandidatePath("uav", "n1", (("access", "s00", UAV_ID), ("down", UAV_ID, "n2"), BACKHAUL_N1), 1.0),
        "out of ground range": CandidatePath("ground", "n3", (("access", "s00", "n3"), ("backhaul", "n3", "n1"), BACKHAUL_N1), 1.0),
        "broken chain": CandidatePath("ground", "n1", (ACCESS_N1, ("backhaul", "n2", "n0")), 1.0),
        "dead edge": CandidatePath("ground", "n1", (ACCESS_N1, ("backhaul", "n1", "n2"), ("backhaul", "n2", "n0")), 1.0),
        "repeated node": CandidatePath("ground", "n1", (ACCESS_N1, ("backhaul", "n1", "n3"), ("backhaul", "n3", "n1"), BACKHAUL_N1), 1.0),
        "does not end at the center": CandidatePath("ground", "n1", (ACCESS_N1,), 1.0),
    }
    for label, path in cases.items():
        violations = validate_plan(scenario, _plan(scenario, path))
        assert violations and all(item.startswith("paths: alert alert-0000") for item in violations), label


def test_entry_at_the_center_needs_no_backhaul(raw_scenario):
    scenario, _ = _setup(raw_scenario)
    direct = CandidatePath("ground", "n0", (("access", "s00", "n0"),), 1.0)
    via_uav = CandidatePath("uav", "n0", (("access", "s00", UAV_ID), ("down", UAV_ID, "n0")), 1.0)
    assert validate_plan(scenario, _plan(scenario, direct)) == []
    assert validate_plan(scenario, _plan(scenario, via_uav)) == []


def test_static_schedule_violations(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    schedule = StaticSchedule(
        {
            ("access", "s00", "n1"): np.full(20, 6e5),
            ("access", "s00", "n0"): np.full(20, 6e5),
            ("down", UAV_ID, "n1"): np.full(20, -1.0),
            ("down", UAV_ID, "n9"): np.zeros(20),
        }
    )
    violations = validate_plan(scenario, _plan(scenario, candidates["alert-0000"][0], schedule))
    assert any("slot 0 access total" in item for item in violations)
    assert any("finite non-negative" in item for item in violations)
    assert any("unknown radio link" in item for item in violations)


def test_static_schedule_downlink_cap(raw_scenario):
    raw_scenario["uav"]["max_active_downlinks"] = 1
    raw_scenario["paths"]["max_ground"] = 0
    scenario, candidates = _setup(raw_scenario)
    downlinks = [link for path in candidates["alert-0000"] for link in path.links if link[0] == "down"]
    schedule = StaticSchedule({link: np.full(20, 1e5) for link in downlinks[:2]})
    violations = validate_plan(scenario, _plan(scenario, candidates["alert-0000"][0], schedule))
    assert any("has 2 active downlinks" in item for item in violations)
