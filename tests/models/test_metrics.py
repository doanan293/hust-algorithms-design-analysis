import numpy as np
import pytest

from models.channel import LinkChannel, all_uav_links
from models.metrics import connectivity, shortfall, timely_flags
from models.scenario import UAV_ID, scenario_from_dict
from scenario_builders import make_alert


def _uav_channels(scenario, usable):
    channels = {link: LinkChannel.constant(scenario.time.num_slots, 0.0) for link in all_uav_links(scenario)}
    for link, slot in usable:
        snr = np.zeros(scenario.time.num_slots)
        snr[slot] = 1e6
        channels[link] = LinkChannel(np.ones(scenario.time.num_slots), snr, snr.copy())
    return channels


def test_timely_flags_and_shortfall(raw_scenario):
    raw_scenario["alerts"].append(make_alert("alert-0001", "s01", 0, 10, 500000))
    scenario = scenario_from_dict(raw_scenario)
    delivered = {"alert-0000": 1000000.0 - 1e-9, "alert-0001": 125000.0}
    assert timely_flags(scenario, delivered) == {"alert-0000": True, "alert-0001": False}
    assert shortfall(scenario, delivered) == pytest.approx(0.75)


def test_source_connects_only_through_uav_visit_then_backhaul(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    channels = _uav_channels(scenario, [(("access", "s01", UAV_ID), 5), (("down", UAV_ID, "n2"), 7)])
    reach = connectivity(scenario, channels)
    assert reach.connected == {"s00": True, "s01": True}
    assert reach.disconnected_slots == {"s00": 0, "s01": 14}
    ground_only = connectivity(scenario, None)
    assert ground_only.connected == {"s00": True, "s01": False}
    assert ground_only.disconnected_slots["s01"] == 20


def test_downlink_before_uplink_does_not_connect(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    channels = _uav_channels(scenario, [(("access", "s01", UAV_ID), 8), (("down", UAV_ID, "n2"), 7)])
    assert connectivity(scenario, channels).connected["s01"] is False
