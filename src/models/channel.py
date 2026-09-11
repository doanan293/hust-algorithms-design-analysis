from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np

from .rng import named_rng
from .scenario import Scenario, UAV_ID

LinkId = tuple[str, str, str]
ACCESS = "access"
DOWNLINK = "down"
BACKHAUL = "backhaul"


def db_to_linear(value_db: float) -> float:
    return 10.0 ** (value_db / 10.0)


def noise_psd_w_per_hz(scenario: Scenario) -> float:
    return 10.0 ** ((scenario.spectrum.noise_psd_dbm_per_hz - 30.0) / 10.0)


def rate_bps(bandwidth_hz: float, snr_per_hz: float) -> float:
    """Shannon rate b*log2(1 + s/b), where s = p*g/N0 in Hz; zero bandwidth gives zero rate."""
    if bandwidth_hz <= 0.0 or snr_per_hz <= 0.0:
        return 0.0
    return bandwidth_hz * math.log2(1.0 + snr_per_hz / bandwidth_hz)


def los_probability(horizontal_m, altitude_m: float, plos_a: float, plos_b: float):
    elevation_deg = np.degrees(np.arctan2(altitude_m, horizontal_m))
    return 1.0 / (1.0 + plos_a * np.exp(-plos_b * (elevation_deg - plos_a)))


def uav_snr_per_hz(scenario: Scenario, horizontal_m, power_w: float):
    channel = scenario.uav_channel
    distance = np.hypot(horizontal_m, scenario.uav.altitude_m)
    scale = power_w * db_to_linear(channel.beta0_db) / noise_psd_w_per_hz(scenario)
    los = scale * distance ** (-channel.alpha_los)
    nlos = scale * channel.nlos_attenuation * distance ** (-channel.alpha_nlos)
    return los, nlos


def ground_snr_per_hz(scenario: Scenario, distance_m: float) -> float:
    channel = scenario.ground_channel
    gain = db_to_linear(channel.beta0_db) * max(distance_m, 1.0) ** (-channel.alpha)
    return scenario.source_power_w * gain / noise_psd_w_per_hz(scenario)


def ground_range_m(scenario: Scenario) -> float:
    bandwidth = scenario.spectrum.b_tot_hz
    required_snr = bandwidth * (2.0 ** (scenario.usable_rate_bps / bandwidth) - 1.0)
    channel = scenario.ground_channel
    reference = scenario.source_power_w * db_to_linear(channel.beta0_db) / noise_psd_w_per_hz(scenario)
    return (reference / required_snr) ** (1.0 / channel.alpha)


def in_ground_range(scenario: Scenario, distance_m: float) -> bool:
    return distance_m <= ground_range_m(scenario) * (1.0 + 1e-12)


@dataclass(frozen=True)
class LinkChannel:
    """Per-slot two-state channel: rate = w*R(b, s_los) + (1 - w)*R(b, s_nlos)."""

    los_weight: np.ndarray
    snr_los_per_hz: np.ndarray
    snr_nlos_per_hz: np.ndarray

    @classmethod
    def constant(cls, num_slots: int, snr_per_hz: float) -> "LinkChannel":
        values = np.full(num_slots, float(snr_per_hz))
        return cls(np.ones(num_slots), values, values.copy())

    def rate_bps(self, slot: int, bandwidth_hz: float) -> float:
        weight = float(self.los_weight[slot])
        los = rate_bps(bandwidth_hz, float(self.snr_los_per_hz[slot]))
        nlos = rate_bps(bandwidth_hz, float(self.snr_nlos_per_hz[slot]))
        return weight * los + (1.0 - weight) * nlos


def all_uav_links(scenario: Scenario) -> tuple[LinkId, ...]:
    uplinks = tuple((ACCESS, source.id, UAV_ID) for source in scenario.sources)
    downlinks = tuple((DOWNLINK, UAV_ID, node.id) for node in scenario.nodes)
    return uplinks + downlinks


def slot_midpoints(trajectory: np.ndarray) -> np.ndarray:
    points = np.asarray(trajectory, dtype=float)
    return 0.5 * (points[:-1] + points[1:])


def ground_point_ids(scenario: Scenario) -> tuple[str, ...]:
    return tuple(source.id for source in scenario.sources) + tuple(node.id for node in scenario.nodes)


def channel_uniforms(scenario: Scenario, realization_id: int) -> dict[str, np.ndarray]:
    rng = named_rng(scenario.scenario_seed, f"channel-eval:{realization_id}")
    point_ids = ground_point_ids(scenario)
    draws = rng.random((len(point_ids), scenario.time.num_slots))
    return {point_id: draws[index] for index, point_id in enumerate(point_ids)}


def build_link_channels(
    scenario: Scenario,
    links: Iterable[LinkId],
    trajectory: np.ndarray,
    uniforms: Mapping[str, np.ndarray] | None,
) -> dict[LinkId, LinkChannel]:
    """Build radio channels; uniforms=None yields expected-rate channels."""
    midpoints = slot_midpoints(trajectory)
    num_slots = scenario.time.num_slots
    channel = scenario.uav_channel
    channels: dict[LinkId, LinkChannel] = {}
    for link in links:
        kind, transmitter, receiver = link
        if kind == ACCESS and receiver != UAV_ID:
            distance = math.dist(scenario.position(transmitter), scenario.position(receiver))
            channels[link] = LinkChannel.constant(num_slots, ground_snr_per_hz(scenario, distance))
            continue
        if kind == ACCESS:
            ground_id, power_w = transmitter, scenario.source_power_w
        elif kind == DOWNLINK:
            ground_id, power_w = receiver, scenario.uav.downlink_power_w
        else:
            raise ValueError(f"link {link} has no radio channel")
        horizontal = np.linalg.norm(midpoints - np.asarray(scenario.position(ground_id)), axis=1)
        probability = los_probability(horizontal, scenario.uav.altitude_m, channel.plos_a, channel.plos_b)
        snr_los, snr_nlos = uav_snr_per_hz(scenario, horizontal, power_w)
        weight = probability if uniforms is None else (uniforms[ground_id] < probability).astype(float)
        channels[link] = LinkChannel(np.asarray(weight, dtype=float), snr_los, snr_nlos)
    return channels
