import pytest

from models.channel import LinkChannel
from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, StaticSchedule, stationary_trajectory
from models.scenario import UAV_ID, scenario_from_dict
from models.simulator import simulate
from channel_builders import constant_channels
from scenario_builders import make_alert


def _run(raw_scenario, choice, snr_per_hz=1e6, bandwidth=None, overrides=None):
    scenario = scenario_from_dict(raw_scenario)
    candidates = candidate_paths(scenario)
    plan = Plan(stationary_trajectory(scenario), choice, bandwidth or EqualSplitBacklogged())
    channels = constant_channels(scenario, candidates, snr_per_hz, overrides)
    return scenario, candidates, simulate(scenario, candidates, plan, channels)


def _transfers(outcome, alert_id):
    return [(slot, link, bits) for slot, link, alert, bits in outcome.ledger.transfers if alert == alert_id]


def test_ground_path_store_and_forward_meets_deadline_boundary(raw_scenario):
    _, candidates, outcome = _run(raw_scenario, {"alert-0000": 0})
    assert candidates["alert-0000"][0].entry == "n1"
    assert _transfers(outcome, "alert-0000") == [
        (0, ("access", "s00", "n1"), 500000.0),
        (1, ("access", "s00", "n1"), 500000.0),
        (1, ("backhaul", "n1", "n0"), 500000.0),
        (2, ("backhaul", "n1", "n0"), 500000.0),
    ]
    assert outcome.delivered_bits["alert-0000"] == 1000000.0
    assert outcome.delivery_slot["alert-0000"] == 2


def test_arrival_in_deadline_slot_is_late_and_expired_bits_drop(raw_scenario):
    raw_scenario["alerts"][0]["deadline_slot"] = 2
    _, _, outcome = _run(raw_scenario, {"alert-0000": 0})
    assert outcome.delivered_bits["alert-0000"] == 500000.0
    assert outcome.delivery_slot["alert-0000"] is None
    assert outcome.ledger.drops == ((2, "alert-0000", "n1", 500000.0),)
    assert all(slot < 2 for slot, _, _ in _transfers(outcome, "alert-0000"))


def test_zero_bandwidth_and_zero_power_deliver_nothing(raw_scenario):
    _, _, no_band = _run(raw_scenario, {"alert-0000": 0}, bandwidth=StaticSchedule({}))
    assert no_band.ledger.transfers == ()
    assert no_band.delivered_bits["alert-0000"] == 0.0
    _, _, no_power = _run(raw_scenario, {"alert-0000": 0}, snr_per_hz=0.0)
    assert no_power.ledger.transfers == ()


def test_no_transmission_before_release(raw_scenario):
    raw_scenario["alerts"][0].update(release_slot=5, deadline_slot=12)
    _, _, outcome = _run(raw_scenario, {"alert-0000": 0})
    assert min(slot for slot, _, _ in _transfers(outcome, "alert-0000")) == 5
    assert outcome.delivery_slot["alert-0000"] == 7


def test_unserved_alert_never_enters_the_network(raw_scenario):
    _, _, outcome = _run(raw_scenario, {"alert-0000": None})
    assert outcome.ledger.transfers == ()
    assert outcome.ledger.drops == ()


def test_backhaul_bottleneck_serves_earliest_deadline_first(raw_scenario):
    raw_scenario["network"]["edges"][0]["capacity_bps"] = 250000.0
    raw_scenario["alerts"] = [
        make_alert("alert-0000", "s00", 0, 10, 500000),
        make_alert("alert-0001", "s00", 0, 5, 500000),
    ]
    _, candidates, outcome = _run(raw_scenario, {"alert-0000": 0, "alert-0001": 0})
    assert candidates["alert-0001"][0].entry == "n1"
    assert outcome.delivery_slot == {"alert-0000": 4, "alert-0001": 2}
    assert _transfers(outcome, "alert-0001")[0] == (0, ("access", "s00", "n1"), 500000.0)
    assert _transfers(outcome, "alert-0000")[0] == (1, ("access", "s00", "n1"), 500000.0)


def test_disconnected_source_uses_uav_store_carry_forward(raw_scenario):
    raw_scenario["alerts"] = [make_alert("alert-0001", "s01", 0, 10, 1000000)]
    _, candidates, outcome = _run(raw_scenario, {"alert-0001": 0})
    assert candidates["alert-0001"][0].links == (("access", "s01", UAV_ID), ("down", UAV_ID, "n0"))
    assert _transfers(outcome, "alert-0001") == [
        (0, ("access", "s01", UAV_ID), 500000.0),
        (1, ("access", "s01", UAV_ID), 500000.0),
        (1, ("down", UAV_ID, "n0"), 500000.0),
        (2, ("down", UAV_ID, "n0"), 500000.0),
    ]
    assert outcome.delivered_bits["alert-0001"] == 1000000.0


def test_equal_split_serves_at_most_two_downlinks_by_deadline(raw_scenario):
    raw_scenario["alerts"] = [
        make_alert("alert-0000", "s01", 0, 15, 500000),
        make_alert("alert-0001", "s01", 0, 5, 500000),
        make_alert("alert-0002", "s01", 0, 10, 500000),
    ]
    scenario = scenario_from_dict(raw_scenario)
    candidates = candidate_paths(scenario)
    entries = [path.entry for path in candidates["alert-0000"]]
    assert entries == ["n0", "n1", "n2"]
    uplink = ("access", "s01", UAV_ID)
    _, _, outcome = _run(
        raw_scenario,
        {"alert-0000": 0, "alert-0001": 1, "alert-0002": 2},
        overrides={uplink: LinkChannel.constant(20, 1e9)},
    )
    slot_one = [(link, hz) for slot, link, hz in outcome.ledger.bandwidth if slot == 1 and link[0] == "down"]
    assert slot_one == [(("down", UAV_ID, "n1"), 500000.0), (("down", UAV_ID, "n2"), 500000.0)]
