import numpy as np
import pytest

from models.evaluate import EXPECTED, evaluate
from models.ledger import Ledger
from models.paths import CandidatePath, candidate_paths
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import scenario_from_dict
from optimization.blocks import SearchState, unit_weights
from optimization.repair import (
    AlertRisk,
    assess_risk,
    bottleneck_hop,
    congested_slots,
    path_change_order,
    repair_round,
    shift_bandwidth,
    slack_slots,
)
from optimization.scoring import DesignScorer
from scenario_builders import make_alert
from search_builders import base_scenario, hopeless_raw

ACCESS_LINK = ("access", "s00", "n1")
BACKHAUL_LINK = ("backhaul", "n1", "n0")


def test_slack_counts_spare_slots_and_marks_late_alerts():
    alert = base_scenario([make_alert("alert-0000", "s00", 0, 10, 100)]).alerts[0]
    assert slack_slots(alert, 7, True) == 2
    assert slack_slots(alert, 9, True) == 0
    assert slack_slots(alert, None, False) == -1


def test_bottleneck_is_the_hop_holding_bits_longest_with_ties_to_the_earliest():
    scenario = base_scenario([make_alert("alert-0000", "s00", 0, 6, 100)])
    alert = scenario.alerts[0]
    links = (ACCESS_LINK, BACKHAUL_LINK, ("backhaul", "n0", "n2"))
    path = CandidatePath("ground", "n1", links, 1.0)
    second_holds_longest = [(0, links[0], 100.0), (1, links[1], 50.0), (2, links[2], 50.0), (4, links[1], 50.0), (5, links[2], 50.0)]
    assert bottleneck_hop(scenario, alert, path, second_holds_longest) == 1
    one_slot_each = [(0, links[0], 100.0), (1, links[1], 100.0), (2, links[2], 100.0)]
    assert bottleneck_hop(scenario, alert, path, one_slot_each) == 0


def test_congested_links_carry_their_whole_capacity():
    other = ("access", "s01", "n0")
    ledger = Ledger(
        realization_id=0,
        transfers=((0, ACCESS_LINK, "a", 6.0), (0, ACCESS_LINK, "b", 4.0), (1, ACCESS_LINK, "a", 3.0)),
        bandwidth=(),
        capacity_bits=((0, ACCESS_LINK, 10.0), (1, ACCESS_LINK, 10.0), (1, other, 0.0)),
        drops=(),
    )
    assert congested_slots(ledger) == {ACCESS_LINK: {0}, other: {1}}


def test_operation1_doubles_weights_only_inside_the_alert_window():
    scenario = base_scenario([make_alert("alert-0000", "s00", 3, 8, 100)])
    weights = unit_weights(scenario, candidate_paths(scenario))
    shifted = shift_bandwidth(weights, ACCESS_LINK, scenario.alerts[0], scenario.time.num_slots)
    expected = np.ones(20)
    expected[3:8] = 2.0
    assert np.array_equal(shifted[ACCESS_LINK], expected)
    assert np.all(weights[ACCESS_LINK] == 1.0)
    assert all(shifted[link] is weights[link] for link in weights if link != ACCESS_LINK)


def test_operation2_orders_candidates_by_congestion_then_cost():
    downlink = ("downlink", "uav", "n1")
    paths = (
        CandidatePath("ground", "n1", (ACCESS_LINK, BACKHAUL_LINK), 0.1),
        CandidatePath("ground", "n1", (ACCESS_LINK, BACKHAUL_LINK), 1.0),
        CandidatePath("ground", "n2", (("access", "s00", "n2"), ("backhaul", "n2", "n0")), 5.0),
        CandidatePath("uav", "n1", (("access", "s00", "uav"), downlink, BACKHAUL_LINK), 0.5),
    )
    candidates = {"alert-0000": paths}
    risk = AlertRisk("alert-0000", 1.0, ACCESS_LINK, frozenset({ACCESS_LINK, BACKHAUL_LINK}))
    assert path_change_order(candidates, risk, 0) == [2, 3, 1]
    assert path_change_order(candidates, risk, None) == [2, 3, 0, 1]
    assert path_change_order(candidates, risk, 0, backlog=False) == [3, 1, 2]


def _hopeless_result(keep_ledger=True):
    scenario = scenario_from_dict(hopeless_raw())
    candidates = candidate_paths(scenario)
    choice = {"alert-0000": 0, "alert-0001": 0}
    plan = Plan(stationary_trajectory(scenario), choice, EqualSplitBacklogged())
    return scenario, candidates, choice, evaluate(scenario, plan, EXPECTED, candidates=candidates, keep_ledger=keep_ledger)


def test_risk_ranks_served_alerts_and_finds_backhaul_bottlenecks():
    scenario, candidates, choice, result = _hopeless_result()
    risks = assess_risk(scenario, candidates, choice, result)
    assert [(item.alert_id, item.risk, item.bottleneck) for item in risks] == [
        ("alert-0000", 1.0, BACKHAUL_LINK),
        ("alert-0001", 1.0, ACCESS_LINK),
    ]
    assert BACKHAUL_LINK in risks[0].congested_links
    assert [item.alert_id for item in assess_risk(scenario, candidates, choice, result, limit=1)] == ["alert-0000"]
    unserved = {"alert-0000": None, "alert-0001": 0}
    assert [item.alert_id for item in assess_risk(scenario, candidates, unserved, result)] == ["alert-0001"]


def test_no_backlog_risk_ignores_ledgers_and_targets_the_first_radio_hop():
    scenario, candidates, choice, result = _hopeless_result(keep_ledger=False)
    risks = assess_risk(scenario, candidates, choice, result, backlog=False)
    assert [(item.alert_id, item.bottleneck, item.congested_links) for item in risks] == [
        ("alert-0000", ACCESS_LINK, frozenset()),
        ("alert-0001", ACCESS_LINK, frozenset()),
    ]
    with pytest.raises(ValueError, match="ledgers"):
        assess_risk(scenario, candidates, choice, result)


def test_operation1_is_skipped_for_backhaul_bottlenecks():
    scenario = scenario_from_dict(hopeless_raw())
    candidates = candidate_paths(scenario)
    scorer = DesignScorer(scenario, candidates, design_ids=(), budget=50)
    state = SearchState(stationary_trajectory(scenario), {"alert-0000": 0, "alert-0001": 0}, unit_weights(scenario, candidates))
    state.key = scorer.score(state.plan())
    assert not repair_round(scenario, candidates, scorer, state, operation1=True, operation2=False)
    # initial score, one detection call, and operation 1 for alert-0001 only: alert-0000 waits on the backhaul
    assert scorer.calls == 3
