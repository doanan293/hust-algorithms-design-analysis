import copy

import numpy as np
import pytest

from baselines import METHODS, get_method
from baselines.b0 import B0
from baselines.b1 import B1, estimated_best_candidate
from baselines.trajectories import zone_tour_trajectory
from models.channel import build_link_channels
from models.evaluate import EvaluationCounter
from models.paths import GROUND, candidate_paths, radio_links
from models.plan import EqualSplitBacklogged, stationary_trajectory, validate_plan
from models.scenario import scenario_from_dict
from scenario_builders import BASE_SCENARIO, make_alert


def _scenario(alerts):
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = alerts
    return scenario_from_dict(raw)


def test_registry_names_methods_and_rejects_unknown_names():
    assert set(METHODS) == {"B0", "B1"}
    assert get_method("B1").uses_uav is True
    assert get_method("B0").uses_uav is False
    with pytest.raises(ValueError, match="unknown method 'B9'"):
        get_method("B9")


def test_b0_uses_first_ground_candidate_and_leaves_unreachable_sources_unserved():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 10, 100000), make_alert("alert-0001", "s01", 0, 10, 100000)])
    candidates = candidate_paths(scenario)
    counter = EvaluationCounter()
    plan = B0().plan(scenario, candidates, counter)
    assert plan.path_choice == {"alert-0000": 0, "alert-0001": None}
    assert candidates["alert-0000"][0].kind == GROUND
    assert isinstance(plan.bandwidth, EqualSplitBacklogged)
    assert np.array_equal(plan.trajectory, stationary_trajectory(scenario))
    assert validate_plan(scenario, candidates, plan) == []
    assert counter.calls == 0


def test_b1_chooses_earliest_estimated_delivery():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 20, 200000)])
    candidates = candidate_paths(scenario)
    trajectory = B1().trajectory(scenario)
    channels = build_link_channels(scenario, radio_links(candidates), trajectory, None)
    assert [(path.kind, path.entry) for path in candidates["alert-0000"]] == [("ground", "n1"), ("ground", "n0"), ("uav", "n1")]
    assert estimated_best_candidate(scenario, candidates, trajectory, channels, "alert-0000") == 0


def test_b1_leaves_hopeless_alerts_unserved_and_plans_validly():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 20, 200000), make_alert("alert-0001", "s01", 0, 2, 1000000)])
    candidates = candidate_paths(scenario)
    counter = EvaluationCounter()
    plan = B1().plan(scenario, candidates, counter)
    assert plan.path_choice == {"alert-0000": 0, "alert-0001": None}
    assert np.array_equal(plan.trajectory, zone_tour_trajectory(scenario))
    assert validate_plan(scenario, candidates, plan) == []
    assert counter.calls == 0
