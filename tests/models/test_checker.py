from dataclasses import replace

from models.checker import check_ledger
from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import scenario_from_dict
from models.simulator import simulate
from channel_builders import constant_channels


def _valid_run(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    candidates = candidate_paths(scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0}, EqualSplitBacklogged())
    channels = constant_channels(scenario, candidates, 1e6)
    outcome = simulate(scenario, candidates, plan, channels)
    return scenario, candidates, plan, channels, outcome


def _with_transfers(ledger, transfers):
    return replace(ledger, transfers=tuple(transfers))


def test_valid_ledger_has_no_violations(raw_scenario):
    scenario, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    assert check_ledger(scenario, candidates, plan, channels, outcome.ledger, outcome.delivered_bits) == []


def test_detects_over_capacity_transfer(raw_scenario):
    scenario, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    transfers = list(outcome.ledger.transfers)
    slot, link, alert, _ = transfers[0]
    transfers[0] = (slot, link, alert, 900000.0)
    violations = check_ledger(scenario, candidates, plan, channels, _with_transfers(outcome.ledger, transfers))
    assert any("over capacity" in item for item in violations)


def test_detects_drop_larger_than_buffer(raw_scenario):
    scenario, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    ledger = replace(outcome.ledger, drops=((12, "alert-0000", "n1", 1000000.0),))
    violations = check_ledger(scenario, candidates, plan, channels, ledger)
    assert any("drops 1000000.000 bits at hop 1 holding only 0.000" in item for item in violations)


def test_detects_off_path_flow(raw_scenario):
    scenario, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    transfers = list(outcome.ledger.transfers)
    slot, _, alert, bits = transfers[2]
    transfers[2] = (slot, ("backhaul", "n2", "n0"), alert, bits)
    violations = check_ledger(scenario, candidates, plan, channels, _with_transfers(outcome.ledger, transfers))
    assert any("outside its chosen path" in item for item in violations)


def test_detects_transmission_before_release(raw_scenario):
    _, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    raw_scenario["alerts"][0]["release_slot"] = 1
    shifted = scenario_from_dict(raw_scenario)
    violations = check_ledger(shifted, candidates, plan, channels, outcome.ledger)
    assert any("transmits before release" in item for item in violations)


def test_detects_same_slot_forwarding(raw_scenario):
    scenario, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    transfers = list(outcome.ledger.transfers)
    _, link, alert, bits = transfers[3]
    transfers[3] = (1, link, alert, bits)
    violations = check_ledger(scenario, candidates, plan, channels, _with_transfers(outcome.ledger, transfers))
    assert any("sends 1000000.000 bits at hop 1 holding only 500000.000" in item for item in violations)


def test_detects_reported_delivery_mismatch(raw_scenario):
    scenario, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    violations = check_ledger(scenario, candidates, plan, channels, outcome.ledger, {"alert-0000": 0.0})
    assert any("reported 0.000 delivered bits" in item for item in violations)
