import math

import numpy as np
import pytest

from models.channel import (
    ACCESS,
    DOWNLINK,
    LinkChannel,
    build_link_channels,
    channel_uniforms,
    ground_range_m,
    ground_snr_per_hz,
    los_probability,
    rate_bps,
    uav_snr_per_hz,
)
from models.scenario import UAV_ID, scenario_from_dict


def test_rate_is_zero_without_bandwidth_or_signal():
    assert rate_bps(0.0, 1e6) == 0.0
    assert rate_bps(1e6, 0.0) == 0.0
    assert rate_bps(1e6, 1e6) == pytest.approx(1e6)


def test_los_probability_matches_hand_values():
    assert float(los_probability(0.0, 100.0, 9.61, 0.16)) == pytest.approx(0.999975, abs=1e-5)
    assert float(los_probability(100.0, 100.0, 9.61, 0.16)) == pytest.approx(0.9677, abs=1e-4)


def test_expected_rate_weights_rates_not_gains(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    los, nlos = uav_snr_per_hz(scenario, 300.0, 0.1)
    probability = float(los_probability(300.0, 100.0, 9.61, 0.16))
    channel = LinkChannel(np.array([probability]), np.array([los]), np.array([nlos]))
    expected = probability * rate_bps(1e6, los) + (1 - probability) * rate_bps(1e6, nlos)
    mean_gain_rate = rate_bps(1e6, probability * los + (1 - probability) * nlos)
    assert channel.rate_bps(0, 1e6) == pytest.approx(expected)
    assert channel.rate_bps(0, 1e6) < mean_gain_rate


def test_realized_state_is_los_exactly_when_uniform_below_probability(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    trajectory = np.tile([1000.0, 1000.0], (21, 1))
    link = (ACCESS, "s00", UAV_ID)
    probability = float(los_probability(230.0, 100.0, 9.61, 0.16))
    uniforms = {"s00": np.full(20, probability - 1e-9)}
    uniforms["s00"][3] = probability + 1e-9
    channel = build_link_channels(scenario, [link], trajectory, uniforms)[link]
    assert channel.los_weight[0] == 1.0
    assert channel.los_weight[3] == 0.0
    expected = build_link_channels(scenario, [link], trajectory, None)[link]
    assert expected.los_weight[0] == pytest.approx(probability)


def test_downlink_uses_per_link_uav_power(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    link = (DOWNLINK, UAV_ID, "n1")
    channel = build_link_channels(scenario, [link], np.tile([1000.0, 1000.0], (21, 1)), None)[link]
    los, _ = uav_snr_per_hz(scenario, 200.0, 0.05)
    assert channel.snr_los_per_hz[0] == pytest.approx(los)


def test_ground_range_matches_usable_rate_threshold(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    distance = ground_range_m(scenario)
    assert 340.0 < distance < 380.0
    assert rate_bps(1e6, ground_snr_per_hz(scenario, distance)) == pytest.approx(10000.0, rel=1e-9)


def test_channel_uniforms_are_reproducible_per_realization(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    first = channel_uniforms(scenario, 0)
    assert set(first) == {"s00", "s01", "n0", "n1", "n2", "n3"}
    assert first["s00"].shape == (20,)
    assert np.array_equal(first["n2"], channel_uniforms(scenario, 0)["n2"])
    assert not np.array_equal(first["n2"], channel_uniforms(scenario, 1)["n2"])
