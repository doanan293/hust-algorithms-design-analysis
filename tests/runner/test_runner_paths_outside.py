import copy

from models.paths import backhaul_routes, candidate_paths, uav_path
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import scenario_from_dict
from runner.tasks import paths_outside_candidates
from scenario_builders import BASE_SCENARIO


def test_paths_outside_candidates_counts_generated_routes_only():
    scenario = scenario_from_dict(copy.deepcopy(BASE_SCENARIO))
    candidates = candidate_paths(scenario)
    trajectory = stationary_trajectory(scenario)
    inside = Plan.from_choice(trajectory, candidates, {"alert-0000": 0}, EqualSplitBacklogged())
    assert paths_outside_candidates(inside, candidates) == 0
    assert paths_outside_candidates(Plan(trajectory, {"alert-0000": None}, EqualSplitBacklogged()), candidates) == 0
    alert = scenario.alerts[0]
    generated = uav_path(scenario, alert, "n3", backhaul_routes(scenario).route_links("n3"))
    assert generated not in candidates[alert.id]
    assert paths_outside_candidates(Plan(trajectory, {alert.id: generated}, EqualSplitBacklogged()), candidates) == 1
