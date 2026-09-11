import copy

import numpy as np
import pytest

from channel_builders import constant_channels
from models import evaluate as evaluate_module
from models.channel import ACCESS, channel_uniforms
from models.checker import SimulatorInvariantError
from models.evaluate import REALIZED, evaluate
from models.paths import candidate_paths, radio_links
from models.plan import EqualSplitBacklogged, Plan, WeightedBacklogged, stationary_trajectory, validate_plan
from models.rng import named_rng
from models.scenario import scenario_from_dict
from models.simulator import simulate
from scenario_builders import BASE_SCENARIO, make_alert


def _scenario(alerts):
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = alerts
    return scenario_from_dict(raw)


def test_eval_namespace_keeps_previous_draws_and_design_differs():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 10, 100000)])
    expected = named_rng(scenario.scenario_seed, "channel-eval:3").random((6, scenario.time.num_slots))
    default = channel_uniforms(scenario, 3)
    assert np.array_equal(default["s00"], expected[0])
    assert np.array_equal(channel_uniforms(scenario, 3, "eval")["n3"], expected[5])
    assert not np.array_equal(channel_uniforms(scenario, 3, "design")["s00"], expected[0])
    with pytest.raises(ValueError, match="unknown channel namespace"):
        channel_uniforms(scenario, 3, "test")


def test_evaluate_records_the_channel_namespace():
    scenario = _scenario([make_alert("alert-0000", "s01", 0, 20, 20000)])
    candidates = candidate_paths(scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0}, EqualSplitBacklogged())
    assert evaluate(scenario, plan, REALIZED, (0,), candidates=candidates).channel_namespace == "eval"
    design = evaluate(scenario, plan, REALIZED, (0,), candidates=candidates, channel_namespace="design")
    assert design.channel_namespace == "design" and design.feasible


def test_weights_split_bandwidth_in_proportion():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 10, 900000), make_alert("alert-0001", "s00", 0, 10, 900000)])
    candidates = candidate_paths(scenario)
    to_n1, to_n0 = candidates["alert-0000"][0].links[0], candidates["alert-0000"][1].links[0]
    assert (to_n1[2], to_n0[2]) == ("n1", "n0")
    weights = {to_n1: np.full(20, 3.0), to_n0: np.full(20, 1.0)}
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0, "alert-0001": 1}, WeightedBacklogged(weights))
    assert validate_plan(scenario, candidates, plan) == []
    outcome = simulate(scenario, candidates, plan, constant_channels(scenario, candidates, 1e6))
    slot_zero = {link: hz for slot, link, hz in outcome.ledger.bandwidth if slot == 0 and link[0] == ACCESS}
    assert slot_zero == {to_n0: 250000.0, to_n1: 750000.0}


def test_unit_weights_reproduce_equal_split_exactly():
    scenario = _scenario([
        make_alert("alert-0000", "s00", 0, 12, 400000),
        make_alert("alert-0001", "s01", 1, 18, 50000),
        make_alert("alert-0002", "s00", 2, 15, 300000),
    ])
    candidates = candidate_paths(scenario)
    choice = {"alert-0000": 0, "alert-0001": 0, "alert-0002": 2}
    channels = constant_channels(scenario, candidates, 1e6)
    equal = simulate(scenario, candidates, Plan(stationary_trajectory(scenario), choice, EqualSplitBacklogged()), channels)
    ones = {link: np.ones(20) for link in radio_links(candidates)}
    weighted = simulate(scenario, candidates, Plan(stationary_trajectory(scenario), choice, WeightedBacklogged(ones)), channels)
    assert weighted.ledger == equal.ledger
    zeros = {link: np.zeros(20) for link in radio_links(candidates)}
    fallback = simulate(scenario, candidates, Plan(stationary_trajectory(scenario), choice, WeightedBacklogged(zeros)), channels)
    assert fallback.ledger == equal.ledger


def test_malformed_weights_are_rejected():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 10, 100000)])
    candidates = candidate_paths(scenario)
    link = candidates["alert-0000"][0].links[0]
    for weights, message in (
        ({link: np.full(20, -1.0)}, "finite non-negative"),
        ({link: np.ones(19)}, "finite non-negative"),
        ({(ACCESS, "s00", "n9"): np.ones(20)}, "unknown radio link"),
    ):
        plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0}, WeightedBacklogged(weights))
        assert any(message in item for item in validate_plan(scenario, candidates, plan))


def test_random_weighted_plans_pass_the_ledger_checker():
    scenario = _scenario([
        make_alert("alert-0000", "s00", 0, 12, 400000),
        make_alert("alert-0001", "s00", 3, 20, 700000),
        make_alert("alert-0002", "s01", 1, 18, 50000),
    ])
    candidates = candidate_paths(scenario)
    rng = np.random.default_rng(11)
    for _ in range(6):
        weights = {link: rng.uniform(0.0, 3.0, size=20) for link in radio_links(candidates)}
        choice = {alert.id: int(rng.integers(0, len(candidates[alert.id]))) for alert in scenario.alerts}
        plan = Plan(stationary_trajectory(scenario), choice, WeightedBacklogged(weights))
        result = evaluate(scenario, plan, REALIZED, (0, 1), candidates=candidates, channel_namespace="design")
        assert result.feasible


def test_unchecked_cached_evaluation_matches_the_checked_result(monkeypatch):
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 12, 400000), make_alert("alert-0001", "s01", 1, 18, 50000)])
    candidates = candidate_paths(scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0, "alert-0001": 0}, EqualSplitBacklogged())
    checked = evaluate(scenario, plan, REALIZED, (0, 1), candidates=candidates)
    real, calls = evaluate_module.connectivity, []
    monkeypatch.setattr(evaluate_module, "connectivity", lambda scenario, channels: calls.append(channels is None) or real(scenario, channels))
    cache = {}
    first = evaluate(scenario, plan, REALIZED, (0, 1), candidates=candidates, check=False, connectivity_cache=cache)
    second = evaluate(scenario, plan, REALIZED, (0, 1), candidates=candidates, check=False, connectivity_cache=cache)
    assert first.key == second.key == checked.key
    assert [item.connected for item in second.realizations] == [item.connected for item in checked.realizations]
    assert [item.delivered_bits for item in second.realizations] == [item.delivered_bits for item in checked.realizations]
    assert calls == [True, False, False] and len(cache) == 3


def test_check_false_skips_only_the_ledger_checker(monkeypatch):
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 12, 400000)])
    candidates = candidate_paths(scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0}, EqualSplitBacklogged())
    monkeypatch.setattr(evaluate_module, "check_ledger", lambda *args: ["forced problem"])
    with pytest.raises(SimulatorInvariantError, match="forced problem"):
        evaluate(scenario, plan, REALIZED, (0,), candidates=candidates)
    assert evaluate(scenario, plan, REALIZED, (0,), candidates=candidates, check=False).feasible
