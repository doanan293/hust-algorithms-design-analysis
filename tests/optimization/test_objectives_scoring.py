from types import SimpleNamespace

import pytest

from baselines.b1 import B1
from models.evaluate import EXPECTED, evaluate, REALIZED
from models.paths import candidate_paths
from optimization.objectives import leximin_key, lexicographic_key
from optimization.scoring import INFEASIBLE_SCORE, BudgetExhausted, DesignScorer
from scenario_builders import make_alert
from search_builders import base_scenario


def _result(delivered, key):
    return SimpleNamespace(realizations=tuple(SimpleNamespace(delivered_bits=item) for item in delivered), key=key)


def test_lexicographic_key_is_the_evaluator_key():
    scenario = base_scenario([make_alert("alert-0000", "s00", 0, 10, 100)])
    assert lexicographic_key(scenario, _result([{"alert-0000": 100.0}], (1, 2, -0.0))) == (1, 2, -0.0)


def test_leximin_key_sorts_mean_fractions_then_counts():
    scenario = base_scenario([
        make_alert("alert-0000", "s00", 0, 10, 100),
        make_alert("alert-0001", "s00", 0, 10, 200),
        make_alert("alert-0002", "s00", 0, 10, 300),
    ])
    delivered = [
        {"alert-0000": 100.0, "alert-0001": 50.0, "alert-0002": 0.0},
        {"alert-0000": 50.0, "alert-0001": 200.0, "alert-0002": 100.0},
    ]
    key = leximin_key(scenario, _result(delivered, (3, 4, -2.5)))
    assert key == (round(1 / 6, 9), 0.625, 0.75, 3, 4)
    assert key > (round(1 / 6, 9), 0.6, 1.0, 6, 4)
    assert key < (round(1 / 6, 9), 0.625, 0.75, 4, 0)
    assert INFEASIBLE_SCORE < (0.0, 0.0, 0.0, 0, 0) and INFEASIBLE_SCORE < (0, 0, -40.0)


def _far_source_plan():
    scenario = base_scenario([make_alert("alert-0000", "s01", 0, 20, 1000000)])
    candidates = candidate_paths(scenario)
    plan = B1().plan(scenario, candidates, None)
    return scenario, candidates, type(plan)(plan.trajectory, {"alert-0000": 0}, plan.bandwidth)


def test_scorer_uses_the_design_namespace():
    scenario, candidates, plan = _far_source_plan()
    scorer = DesignScorer(scenario, candidates, design_ids=(0, 1, 2, 3, 4), budget=10)
    key, result = scorer.score_with_ledger(plan)
    design = evaluate(scenario, plan, REALIZED, (0, 1, 2, 3, 4), candidates=candidates, channel_namespace="design")
    public = evaluate(scenario, plan, REALIZED, (0, 1, 2, 3, 4), candidates=candidates)
    assert result.channel_namespace == "design"
    assert all(item.ledger is not None for item in result.realizations)
    assert key == design.key
    assert [item.delivered_bits for item in result.realizations] == [item.delivered_bits for item in design.realizations]
    assert [item.delivered_bits for item in result.realizations] != [item.delivered_bits for item in public.realizations]


def test_scorer_counts_calls_and_stops_at_the_budget():
    scenario, candidates, plan = _far_source_plan()
    scorer = DesignScorer(scenario, candidates, design_ids=(0,), budget=2)
    scorer.score(plan)
    scorer.score_with_ledger(plan)
    assert (scorer.calls, scorer.remaining) == (2, 0)
    with pytest.raises(BudgetExhausted):
        scorer.score(plan)
    assert scorer.calls == 2
    with pytest.raises(ValueError, match="budget"):
        DesignScorer(scenario, candidates, budget=0)


def test_empty_design_ids_score_expected_rates_and_infeasible_plans_score_lowest():
    scenario, candidates, plan = _far_source_plan()
    scorer = DesignScorer(scenario, candidates, design_ids=(), budget=5)
    key, result = scorer.score_with_ledger(plan)
    assert result.mode == EXPECTED and len(result.realizations) == 1
    assert key == evaluate(scenario, plan, EXPECTED, candidates=candidates).key
    broken = type(plan)(plan.trajectory, {"alert-0000": 7}, plan.bandwidth)
    assert scorer.score(broken) == INFEASIBLE_SCORE
