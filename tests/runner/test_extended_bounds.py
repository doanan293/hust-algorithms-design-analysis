import copy
import importlib.util
from pathlib import Path

from models.evaluate import REALIZED, evaluate
from models.paths import backhaul_routes, candidate_paths, uav_path
from models.plan import EqualSplitBacklogged, Plan
from models.scenario import scenario_from_dict
from baselines.trajectories import zone_tour_trajectory
from scenario_builders import BASE_SCENARIO, make_alert

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "check_extended_bounds.py"
SPEC = importlib.util.spec_from_file_location("check_extended_bounds", SCRIPT)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def _scenario():
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = [make_alert("alert-0000", "s00", 0, 20, 400000), make_alert("alert-0001", "s01", 0, 20, 200000)]
    return scenario_from_dict(raw)


def test_extended_candidates_add_only_the_plan_paths_outside_the_set():
    scenario = _scenario()
    candidates = candidate_paths(scenario)
    alert = scenario.alert_by_id["alert-0001"]
    generated = uav_path(scenario, alert, "n3", backhaul_routes(scenario).route_links("n3"))
    plan = Plan(zone_tour_trajectory(scenario), {"alert-0000": candidates["alert-0000"][0], "alert-0001": generated}, EqualSplitBacklogged())
    extended = check.extended_candidates(candidates, plan)
    assert extended["alert-0000"] == candidates["alert-0000"]
    assert extended["alert-0001"] == (*candidates["alert-0001"], generated)


def test_exceeding_groups_pick_outside_rows_above_their_restricted_bound():
    rows = [
        {"set_id": "v0", "scenario_id": "A", "method": "P", "realization_id": "0", "timely_count": "5", "paths_outside_candidates": "2"},
        {"set_id": "v0", "scenario_id": "A", "method": "P", "realization_id": "1", "timely_count": "3", "paths_outside_candidates": "2"},
        {"set_id": "v0", "scenario_id": "B", "method": "P", "realization_id": "0", "timely_count": "9", "paths_outside_candidates": "0"},
    ]
    bounds = [
        {"set_id": "v0", "scenario_id": "A", "variant": "trajectory", "trajectory_method": "P", "realization_id": "0", "value": "4.5"},
        {"set_id": "v0", "scenario_id": "A", "variant": "trajectory", "trajectory_method": "P", "realization_id": "1", "value": "4.5"},
        {"set_id": "v0", "scenario_id": "B", "variant": "trajectory", "trajectory_method": "P", "realization_id": "0", "value": "4.0"},
    ]
    assert check.exceeding_groups(rows, bounds) == {("v0", "A", "P"): {0: (5, 4.5)}}


def test_extended_bound_holds_for_a_plan_with_a_generated_path():
    scenario = _scenario()
    candidates = candidate_paths(scenario)
    alert = scenario.alert_by_id["alert-0001"]
    generated = uav_path(scenario, alert, "n3", backhaul_routes(scenario).route_links("n3"))
    plan = Plan(zone_tour_trajectory(scenario), {"alert-0000": candidates["alert-0000"][0], "alert-0001": generated}, EqualSplitBacklogged())
    for realization_id in (0, 1):
        timely = sum(evaluate(scenario, plan, REALIZED, (realization_id,)).realizations[0].timely.values())
        value, status = check.extended_bound(scenario, candidates, plan, realization_id, tangents=10)
        assert status == "optimal" and timely <= value + 1e-6
