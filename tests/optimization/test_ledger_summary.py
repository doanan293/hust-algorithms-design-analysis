import pytest

from models.evaluate import EXPECTED, evaluate
from models.ledger import Ledger
from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from optimization.ledger_summary import LedgerSummary, alert_window, congested_slots, residual_bps
from scenario_builders import make_alert
from search_builders import base_scenario

ACCESS_LINK = ("access", "s00", "n1")
BACKHAUL_LINK = ("backhaul", "n1", "n0")


def _ledger(transfers, capacity):
    return Ledger(realization_id=0, transfers=tuple(transfers), bandwidth=(), capacity_bits=tuple(capacity), drops=())


def test_congested_links_carry_their_whole_capacity():
    other = ("access", "s01", "n0")
    ledger = _ledger(
        [(0, ACCESS_LINK, "a", 6.0), (0, ACCESS_LINK, "b", 4.0), (1, ACCESS_LINK, "a", 3.0)],
        [(0, ACCESS_LINK, 10.0), (1, ACCESS_LINK, 10.0), (1, other, 0.0)],
    )
    assert congested_slots(ledger) == {ACCESS_LINK: {0}, other: {1}}


def test_alert_window_ends_at_the_deadline_or_the_horizon():
    scenario = base_scenario([make_alert("alert-0000", "s00", 3, 20, 100), make_alert("alert-0001", "s00", 2, 9, 100)])
    assert alert_window(scenario, scenario.alerts[1]) == range(2, 9)
    assert alert_window(scenario, scenario.alerts[0]) == range(3, 20)


def test_summary_counts_congestion_and_carried_bits_over_realizations():
    first = _ledger([(0, BACKHAUL_LINK, "a", 10.0), (1, BACKHAUL_LINK, "a", 10.0)], [(0, BACKHAUL_LINK, 10.0), (1, BACKHAUL_LINK, 10.0)])
    second = _ledger([(1, BACKHAUL_LINK, "a", 4.0)], [(1, BACKHAUL_LINK, 10.0)])
    summary = LedgerSummary(2, (congested_slots(first), congested_slots(second)), {(BACKHAUL_LINK, 0): 10.0, (BACKHAUL_LINK, 1): 14.0})
    assert summary.congestion_count(BACKHAUL_LINK, range(0, 2)) == 2
    assert summary.congestion_count(BACKHAUL_LINK, range(1, 2)) == 1
    assert summary.congested_realizations(BACKHAUL_LINK, 1) == 1
    assert summary.congested_realizations(ACCESS_LINK, 1) == 0
    assert summary.carried_bps(BACKHAUL_LINK, range(0, 2), 0.5) == pytest.approx(24.0 / (2 * 2 * 0.5))
    assert summary.carried_bps(BACKHAUL_LINK, range(5, 5), 0.5) == 0.0


def test_residual_is_capacity_minus_carried_and_floored():
    scenario = base_scenario([make_alert("alert-0000", "s00", 0, 10, 100)])
    capacity = scenario.backhaul_capacity_bps("n1", "n0")
    light = LedgerSummary(1, ({},), {(BACKHAUL_LINK, 0): 0.25 * capacity})
    assert residual_bps(scenario, light, BACKHAUL_LINK, range(0, 1)) == pytest.approx(0.75 * capacity)
    saturated = LedgerSummary(1, ({},), {(BACKHAUL_LINK, 0): capacity})
    assert residual_bps(scenario, saturated, BACKHAUL_LINK, range(0, 1)) == pytest.approx(capacity / 1000.0)
    assert residual_bps(scenario, light, ("backhaul", "n2", "n0"), range(0, 1)) == pytest.approx(scenario.backhaul_capacity_bps("n2", "n0"))


def test_from_result_needs_ledgers_and_reads_them():
    scenario = base_scenario([make_alert("alert-0000", "s00", 0, 10, 12000000)])
    candidates = candidate_paths(scenario)
    plan = Plan.from_choice(stationary_trajectory(scenario), candidates, {"alert-0000": 0}, EqualSplitBacklogged())
    with pytest.raises(ValueError, match="ledger"):
        LedgerSummary.from_result(evaluate(scenario, plan, EXPECTED))
    summary = LedgerSummary.from_result(evaluate(scenario, plan, EXPECTED, keep_ledger=True))
    assert summary.realization_count == 1
    assert summary.congestion_count(BACKHAUL_LINK, range(0, 10)) > 0
    assert summary.carried_bps(BACKHAUL_LINK, range(0, 10), 1.0) > 0.0
