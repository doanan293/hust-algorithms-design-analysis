import numpy as np
import pytest

from literature.tran import TranHD
from literature.tran_ia import initial_iterate
from literature.tran_model import TranParams, build_instance
from literature.tran_plan import expand_trajectory, keep_top_downlinks, plan_from_iterate
from models.channel import ACCESS
from models.evaluate import EvaluationCounter
from models.paths import candidate_paths
from models.plan import validate_plan
from models.scenario import scenario_from_dict
from tran_builders import tran_raw


def test_expand_trajectory_interpolates_inside_blocks():
    expanded = expand_trajectory(np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]), 5)
    assert expanded.shape == (11, 2)
    assert expanded[:6, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] and expanded[-1].tolist() == [5.0, 5.0]


def test_keep_top_downlinks_breaks_ties_by_link_order():
    links = [("down", "UAV", "a"), ("down", "UAV", "b"), ("down", "UAV", "c")]
    bandwidth = dict(zip(links, (np.array([3.0, 1.0]), np.array([3.0, 2.0]), np.array([1.0, 5.0]))))
    keep_top_downlinks(bandwidth, links, 2)
    assert [bandwidth[link].tolist() for link in links] == [[3.0, 0.0], [3.0, 2.0], [0.0, 5.0]]


@pytest.mark.parametrize("block_slots", [1, 5])
def test_initial_iterate_converts_to_a_valid_static_plan(block_slots):
    scenario = scenario_from_dict(tran_raw())
    candidates = candidate_paths(scenario)
    instance = build_instance(scenario, candidates, TranParams(block_slots=block_slots))
    plan = plan_from_iterate(instance, candidates, initial_iterate(instance))
    assert validate_plan(scenario, candidates, plan) == []
    assert plan.path_choice == {"alert-0000": 0, "alert-0001": 0, "alert-0002": 0}
    access = sum(values for link, values in plan.bandwidth.bandwidth_hz.items() if link[0] == ACCESS)
    assert np.all(access <= 1e6 * (1.0 + 1e-9))


def test_tran_method_plans_validly_and_has_no_fixed_trajectory():
    scenario = scenario_from_dict(tran_raw(uav_size=12000000))
    candidates = candidate_paths(scenario)
    method = TranHD(params=TranParams(max_iterations=3))
    outcome = method.solve(scenario, candidates)
    assert validate_plan(scenario, candidates, outcome.plan) == [] and validate_plan(scenario, candidates, outcome.initial_plan) == []
    assert outcome.penalty <= 0.0 and len(outcome.objectives) == outcome.iterations + 1
    assert np.array_equal(method.plan(scenario, candidates, EvaluationCounter()).trajectory, outcome.plan.trajectory)
    with pytest.raises(NotImplementedError):
        method.trajectory(scenario)
