from dataclasses import replace

import numpy as np
import pytest

from models.evaluate import REALIZED, EvaluationCounter, evaluate
from models.paths import candidate_paths
from models.scenario import scenario_from_dict
from optimization import bcd
from optimization.bcd import SearchMethod, SearchParams
from optimization.objectives import lexicographic_key
from optimization.scoring import DesignScorer
from scenario_builders import make_alert
from search_builders import three_zone_raw

SMALL = {"design_ids": (0, 1), "budget": 150, "max_iterations": 3}
VARIANTS = {
    "B2": SearchParams(**SMALL),
    "B3": SearchParams(objective="leximin", **SMALL),
    "P": SearchParams(operation1=True, operation2=True, **SMALL),
    "P_no_backlog": SearchParams(operation1=True, operation2=True, backlog=False, **SMALL),
    "P_expected_design": SearchParams(operation1=True, operation2=True, **{**SMALL, "design_ids": ()}),
}


def _scenario():
    raw = three_zone_raw(num_slots=50)
    raw["alerts"] += [make_alert("alert-0003", "s01", 2, 48, 400000), make_alert("alert-0004", "s00", 1, 8, 2500000)]
    return scenario_from_dict(raw)


def _same_plan(first, second):
    weights_a, weights_b = first.bandwidth.weights, second.bandwidth.weights
    return (
        np.array_equal(first.trajectory, second.trajectory)
        and first.path_choice == second.path_choice
        and weights_a.keys() == weights_b.keys()
        and all(np.array_equal(weights_a[link], weights_b[link]) for link in weights_a)
    )


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_search_is_monotone_within_budget_feasible_and_repeatable(name):
    scenario = _scenario()
    candidates = candidate_paths(scenario)
    params = VARIANTS[name]
    method = SearchMethod(name, params)
    outcome = method.search(scenario, candidates)
    assert list(outcome.iteration_keys) == sorted(outcome.iteration_keys)
    assert 1 <= outcome.calls <= params.budget
    if not outcome.budget_exhausted:
        assert outcome.iteration_keys[-1] == outcome.key
    rescorer = DesignScorer(scenario, candidates, params.design_ids, 1, bcd.OBJECTIVES[params.objective])
    assert rescorer.score(outcome.plan) == outcome.key
    assert evaluate(scenario, outcome.plan, REALIZED, (0, 1, 2), candidates=candidates).feasible
    assert _same_plan(outcome.plan, method.search(scenario, candidates).plan)


def test_budget_exhaustion_returns_the_feasible_incumbent():
    scenario = _scenario()
    candidates = candidate_paths(scenario)
    params = SearchParams(operation1=True, operation2=True, design_ids=(0,), budget=4)
    outcome = SearchMethod("P", params).search(scenario, candidates)
    assert outcome.budget_exhausted and outcome.calls == 4 and outcome.iteration_keys == ()
    assert evaluate(scenario, outcome.plan, REALIZED, (0,), candidates=candidates).feasible


def test_p_without_operations_is_b2_and_b3_differs_only_by_its_key(monkeypatch):
    scenario = _scenario()
    candidates = candidate_paths(scenario)
    b2 = SearchMethod("B2", VARIANTS["B2"]).search(scenario, candidates)
    p_off = SearchMethod("P", replace(VARIANTS["P"], operation1=False, operation2=False)).search(scenario, candidates)
    assert _same_plan(b2.plan, p_off.plan) and b2.calls == p_off.calls
    monkeypatch.setitem(bcd.OBJECTIVES, "leximin", lexicographic_key)
    b3 = SearchMethod("B3", VARIANTS["B3"]).search(scenario, candidates)
    assert _same_plan(b2.plan, b3.plan) and b3.key == b2.key


def test_plan_reports_calls_and_trajectory_needs_planning():
    scenario = _scenario()
    candidates = candidate_paths(scenario)
    method = SearchMethod("B2", VARIANTS["B2"])
    counter = EvaluationCounter(calls=1)
    method.plan(scenario, candidates, counter)
    assert counter.calls == 1 + method.search(scenario, candidates).calls
    with pytest.raises(NotImplementedError):
        method.trajectory(scenario)


def test_search_params_from_mapping_validates_values():
    assert SearchParams.from_mapping({}) == SearchParams()
    assert SearchParams.from_mapping({"design_ids": "expected", "operation1": True}).design_ids == ()
    assert SearchParams.from_mapping({"design_ids": [3, 4]}).design_ids == (3, 4)
    for raw, message in (
        ({"repair": True}, "unknown search parameters"),
        ({"objective": "maxmin"}, "objective"),
        ({"budget": 0}, "budget"),
        ({"operation1": "yes"}, "operation1"),
        ({"design_ids": []}, "design_ids"),
        ({"design_ids": [1, 1]}, "design_ids"),
    ):
        with pytest.raises(ValueError, match=message):
            SearchParams.from_mapping(raw)
