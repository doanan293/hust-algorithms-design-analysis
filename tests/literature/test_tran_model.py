import math

import pytest

from literature.tran_model import TranParams, backhaul_slots, build_instance, fit_effective_channel
from models.channel import db_to_linear, ground_snr_per_hz, noise_psd_w_per_hz
from models.paths import candidate_paths
from models.scenario import scenario_from_dict
from scenario_builders import make_alert
from tran_builders import tran_raw


def _instance(raw, **params):
    scenario = scenario_from_dict(raw)
    candidates = candidate_paths(scenario)
    return scenario, candidates, build_instance(scenario, candidates, TranParams(**params))


def test_params_validate_and_reject_unknown_keys():
    assert TranParams() == TranParams(theta=0.5, mu=1.0, max_iterations=30, tolerance=1e-3, block_slots=1)
    assert TranParams.from_mapping({"theta": 0.25, "mu": 10}) == TranParams(theta=0.25, mu=10.0)
    with pytest.raises(ValueError, match="unknown TRAN parameters"):
        TranParams.from_mapping({"gamma": 1})
    for bad, message in ((dict(theta=1.0), "theta"), (dict(mu=0), "mu"), (dict(tolerance=-1.0), "tolerance"),
                         (dict(max_iterations=True), "max_iterations"), (dict(block_slots=0), "block_slots")):
        with pytest.raises(ValueError, match=message):
            TranParams(**bad)


def test_effective_channel_recovers_a_pure_los_power_law():
    raw = tran_raw()
    raw["channel"]["uav"]["plos_a"] = 1e-12
    channel = fit_effective_channel(scenario_from_dict(raw))
    assert channel.alpha == pytest.approx(2.2, rel=1e-6)
    assert channel.beta == pytest.approx(db_to_linear(-50.0), rel=1e-6)
    assert channel.r_squared == pytest.approx(1.0, abs=1e-9) and channel.max_log_error < 1e-6


def test_devices_follow_cost_choice_backhaul_and_theta_windows():
    scenario, candidates, instance = _instance(tran_raw())
    ground, first, second = instance.devices
    assert (ground.ground, ground.candidate, ground.uplink, ground.downlink) == (True, 0, (0, 18), (0, 0))
    assert backhaul_slots(scenario, candidates["alert-0000"][0], 200000) == 2
    distance = math.dist(scenario.source_by_id["s00"].xy, scenario.node_by_id["n1"].xy)
    assert ground.ground_efficiency == pytest.approx(math.log2(1.0 + ground_snr_per_hz(scenario, distance) / 1e6))
    assert (first.ground, first.gateway, first.uplink, first.downlink) == (False, "n1", (0, 19), (19, 38))
    assert (second.uplink, second.downlink) == ((5, 22), (22, 38))
    assert instance.points == ("s02", "n1")
    noise = noise_psd_w_per_hz(scenario) * 1e6
    assert instance.point_snr.tolist() == pytest.approx([0.1 * instance.channel.beta / noise, 0.05 * instance.channel.beta / noise])
    assert (len(instance.uplink_device), len(instance.downlink_device)) == (18 + 19 + 17, 19 + 16)
    assert set(instance.uplink_point[instance.uplink_device == 0]) == {-1}


def test_blocks_round_windows_inward_and_empty_windows_deactivate():
    raw = tran_raw()
    raw["alerts"].append(make_alert("alert-0003", "s02", 30, 33, 100000))
    _, _, instance = _instance(raw, block_slots=5)
    ground, first, second, tight = instance.devices
    assert (ground.uplink, first.uplink, first.downlink, second.uplink, second.downlink) == ((0, 3), (0, 3), (4, 7), (1, 4), (5, 7))
    assert instance.num_blocks == 8 and not tight.active and 3 not in set(instance.uplink_device)
    with pytest.raises(ValueError, match="must divide"):
        _instance(tran_raw(), block_slots=3)
