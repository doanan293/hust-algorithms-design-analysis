"""Tran et al. (2022) half-duplex instance built from a scenario (spec D Section 5.1)."""

from dataclasses import dataclass, fields
import math
from typing import Mapping

import numpy as np

from models.channel import BACKHAUL, db_to_linear, ground_snr_per_hz, los_probability, noise_psd_w_per_hz
from models.paths import GROUND, CandidatePath
from models.scenario import Scenario

FIT_STEP_M = 10.0
ROUNDING_TOLERANCE = 1e-9


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class TranParams:
    theta: float = 0.5
    mu: float = 1.0
    max_iterations: int = 30
    tolerance: float = 1e-3
    block_slots: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < _number(self.theta, "theta") < 1.0:
            raise ValueError(f"theta must lie in (0, 1), got {self.theta!r}")
        if _number(self.mu, "mu") <= 0.0:
            raise ValueError(f"mu must be positive, got {self.mu!r}")
        if _number(self.tolerance, "tolerance") <= 0.0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance!r}")
        _positive_integer(self.max_iterations, "max_iterations")
        _positive_integer(self.block_slots, "block_slots")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "TranParams":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown TRAN parameters {unknown}; known parameters: {sorted(known)}")
        return cls(**raw)


@dataclass(frozen=True)
class EffectiveChannel:
    """Power law beta * d^-alpha fitted to the expected PrLoS gain; r_squared and max_log_error judge the fit."""

    beta: float
    alpha: float
    r_squared: float
    max_log_error: float


@dataclass(frozen=True)
class TranDevice:
    """One alert as a Tran device; windows are [start, stop) in blocks, empty when stop == start."""

    alert_id: str
    candidate: int
    ground: bool
    source: str
    gateway: str | None
    size_bits: int
    uplink: tuple[int, int]
    downlink: tuple[int, int]
    ground_efficiency: float

    @property
    def active(self) -> bool:
        uplink = self.uplink[1] > self.uplink[0]
        return uplink if self.ground else uplink and self.downlink[1] > self.downlink[0]


@dataclass(frozen=True, eq=False)
class TranInstance:
    """Devices, radio points and per-entry index arrays of the half-duplex program.

    An uplink entry is one (active device, block) pair of the device's uplink window; `uplink_point` indexes
    `points` for UAV devices and is -1 for ground devices. Downlink entries exist for active UAV devices only.
    """

    scenario: Scenario
    devices: tuple[TranDevice, ...]
    channel: EffectiveChannel
    block_slots: int
    num_blocks: int
    points: tuple[str, ...]
    point_snr: np.ndarray
    uplink_device: np.ndarray
    uplink_block: np.ndarray
    uplink_point: np.ndarray
    downlink_device: np.ndarray
    downlink_block: np.ndarray
    downlink_point: np.ndarray


def expected_uav_gain(scenario: Scenario, horizontal_m: np.ndarray) -> np.ndarray:
    channel = scenario.uav_channel
    altitude = scenario.uav.altitude_m
    distance = np.hypot(horizontal_m, altitude)
    probability = los_probability(horizontal_m, altitude, channel.plos_a, channel.plos_b)
    los = distance ** (-channel.alpha_los)
    nlos = channel.nlos_attenuation * distance ** (-channel.alpha_nlos)
    return db_to_linear(channel.beta0_db) * (probability * los + (1.0 - probability) * nlos)


def ground_point_diameter(scenario: Scenario) -> float:
    points = np.array([node.xy for node in scenario.nodes] + [source.xy for source in scenario.sources])
    return float(max(np.linalg.norm(points - point, axis=1).max() for point in points))


def fit_effective_channel(scenario: Scenario) -> EffectiveChannel:
    """Least squares of ln(expected gain) on ln(distance) over horizontal distances 0..diameter in 10 m steps."""
    horizontal = np.arange(0.0, ground_point_diameter(scenario) + FIT_STEP_M, FIT_STEP_M)
    log_distance = np.log(np.hypot(horizontal, scenario.uav.altitude_m))
    log_gain = np.log(expected_uav_gain(scenario, horizontal))
    slope, intercept = np.polyfit(log_distance, log_gain, 1)
    residual = log_gain - (intercept + slope * log_distance)
    total = float(np.sum((log_gain - log_gain.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else 1.0
    return EffectiveChannel(float(np.exp(intercept)), float(-slope), r_squared, float(np.abs(residual).max()))


def backhaul_slots(scenario: Scenario, path: CandidatePath, size_bits: int) -> int:
    """Slots the backhaul route needs, with the per-link delay of models.paths (size/capacity + one slot)."""
    delay_s = sum(
        size_bits / scenario.backhaul_capacity_bps(link[1], link[2]) + scenario.time.slot_s
        for link in path.links
        if link[0] == BACKHAUL
    )
    return math.ceil(delay_s / scenario.time.slot_s - ROUNDING_TOLERANCE)


def _blocks(start_slot: int, stop_slot: int, block_slots: int) -> tuple[int, int]:
    start = -(-start_slot // block_slots)
    stop = stop_slot // block_slots
    return (start, max(start, stop))


def build_device(
    scenario: Scenario, candidates: tuple[CandidatePath, ...], alert_index: int, params: TranParams
) -> TranDevice:
    alert = scenario.alerts[alert_index]
    candidate = min(range(len(candidates)), key=lambda index: (candidates[index].cost_s, index))
    path = candidates[candidate]
    end = alert.deadline_slot - backhaul_slots(scenario, path, alert.size_bits)
    G = params.block_slots
    if path.kind == GROUND:
        distance = math.dist(scenario.source_by_id[alert.source].xy, scenario.node_by_id[path.entry].xy)
        efficiency = math.log2(1.0 + ground_snr_per_hz(scenario, distance) / scenario.spectrum.b_tot_hz)
        return TranDevice(
            alert.id, candidate, True, alert.source, None, alert.size_bits,
            _blocks(alert.release_slot, end, G), (0, 0), efficiency,
        )
    length = end - alert.release_slot
    split = alert.release_slot + (math.ceil(params.theta * length - ROUNDING_TOLERANCE) if length >= 1 else 0)
    return TranDevice(
        alert.id, candidate, False, alert.source, path.entry, alert.size_bits,
        _blocks(alert.release_slot, split, G), _blocks(split, end, G), 0.0,
    )


def build_instance(
    scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]], params: TranParams
) -> TranInstance:
    num_slots = scenario.time.num_slots
    if num_slots % params.block_slots:
        raise ValueError(f"block_slots {params.block_slots} must divide num_slots {num_slots}")
    devices = tuple(
        build_device(scenario, candidates[alert.id], index, params) for index, alert in enumerate(scenario.alerts)
    )
    channel = fit_effective_channel(scenario)
    noise_w = noise_psd_w_per_hz(scenario) * scenario.spectrum.b_tot_hz
    active_uav = [device for device in devices if device.active and not device.ground]
    sources = sorted({device.source for device in active_uav})
    gateways = sorted({device.gateway for device in active_uav})
    points = tuple(sources + gateways)
    point_snr = np.array(
        [scenario.source_power_w * channel.beta / noise_w] * len(sources)
        + [scenario.uav.downlink_power_w * channel.beta / noise_w] * len(gateways)
    )
    point_index = {point: index for index, point in enumerate(points)}
    uplink, downlink = [], []
    for index, device in enumerate(devices):
        if not device.active:
            continue
        uplink_point = -1 if device.ground else point_index[device.source]
        uplink.extend((index, block, uplink_point) for block in range(*device.uplink))
        if not device.ground:
            downlink.extend((index, block, point_index[device.gateway]) for block in range(*device.downlink))
    up = np.array(uplink, dtype=int).reshape(-1, 3)
    down = np.array(downlink, dtype=int).reshape(-1, 3)
    return TranInstance(
        scenario, devices, channel, params.block_slots, num_slots // params.block_slots, points, point_snr,
        up[:, 0], up[:, 1], up[:, 2], down[:, 0], down[:, 1], down[:, 2],
    )
