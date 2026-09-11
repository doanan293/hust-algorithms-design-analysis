import copy

import numpy as np
import pytest

from bounds.crosscheck import (
    flow_variable_count,
    model_snr_values,
    plan_from_solution,
    reduce_scenario,
    select_crosscheck_scenarios,
    select_positions,
    tangent_max_overestimate,
)
from bounds.formulation import build_bound_model
from bounds.solve import solve_lp, solve_milp
from baselines.trajectories import zone_tour_trajectory
from models.channel import build_link_channels, channel_uniforms
from models.evaluate import REALIZED, evaluate
from models.paths import candidate_paths, radio_links
from models.plan import validate_plan
from models.scenario import scenario_from_dict
from scenario_builders import BASE_SCENARIO, make_alert


def test_selection_positions_and_order():
    assert select_positions(30, 5) == [0, 7, 15, 22, 29]
    assert select_positions(3, 1) == [0]
    with pytest.raises(ValueError):
        select_positions(3, 4)
    assert select_crosscheck_scenarios([(9, "c"), (1, "b"), (1, "a")], 2) == ["a", "c"]


def test_reduce_scenario_keeps_smallest_alert_ids():
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = [make_alert(f"alert-000{index}", "s00", 0, 10, 1000) for index in (3, 1, 2)]
    reduced = reduce_scenario(scenario_from_dict(raw), 2)
    assert [alert.id for alert in reduced.alerts] == ["alert-0001", "alert-0002"]
    assert flow_variable_count(reduced, candidate_paths(reduced)) > 0


def test_milp_plan_is_valid_and_bounded_by_the_milp_bound():
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = [
        make_alert("alert-0000", "s00", 0, 12, 400000),
        make_alert("alert-0001", "s01", 0, 20, 60000),
        make_alert("alert-0002", "s00", 2, 15, 300000),
    ]
    scenario = scenario_from_dict(raw)
    candidates = candidate_paths(scenario)
    trajectory = zone_tour_trajectory(scenario)
    channels = build_link_channels(scenario, radio_links(candidates), trajectory, channel_uniforms(scenario, 0))
    model = build_bound_model(scenario, candidates, channels, tangents=10)
    lp = solve_lp(model)
    milp = solve_milp(model, time_limit_s=60)
    plan = plan_from_solution(scenario, candidates, model, milp.solution, trajectory)
    assert validate_plan(scenario, candidates, plan) == []
    evaluated = evaluate(scenario, plan, REALIZED, (0,), candidates=candidates)
    value = sum(evaluated.realizations[0].timely.values())
    assert value <= milp.dual_bound + 1e-6 <= lp.value + 2e-6
    overestimate = tangent_max_overestimate(model_snr_values(model, channels), scenario.spectrum.b_tot_hz, 10)
    assert 0.0 <= overestimate < 0.25
    assert tangent_max_overestimate([1e6], 1e6, 20) <= tangent_max_overestimate([1e6], 1e6, 10)
