"""Conversion of a Tran iterate into a project Plan with a static schedule (spec D Section 5.4)."""

from typing import Mapping

import numpy as np

from models.channel import LinkId
from models.paths import CandidatePath
from models.plan import Plan, StaticSchedule, validate_plan

from .tran_ia import Iterate
from .tran_model import TranInstance


def expand_trajectory(block_trajectory: np.ndarray, block_slots: int) -> np.ndarray:
    """Slot-boundary trajectory interpolated linearly inside each block."""
    starts, stops = block_trajectory[:-1], block_trajectory[1:]
    fractions = np.arange(block_slots) / block_slots
    inner = starts[:, None, :] + (stops - starts)[:, None, :] * fractions[None, :, None]
    return np.vstack([inner.reshape(-1, 2), block_trajectory[-1:]])


def keep_top_downlinks(bandwidth: dict[LinkId, np.ndarray], links: list[LinkId], limit: int) -> None:
    """Per slot keep the `limit` downlinks with the most bandwidth (ties: link order); zero the others in place."""
    if not links:
        return
    matrix = np.vstack([bandwidth[link] for link in links])
    order = np.argsort(-matrix, axis=0, kind="stable")
    keep = np.zeros(matrix.shape, dtype=bool)
    np.put_along_axis(keep, order[:limit], True, axis=0)
    for index, link in enumerate(links):
        bandwidth[link] = np.where(keep[index], matrix[index], 0.0)


def plan_from_iterate(
    instance: TranInstance, candidates: Mapping[str, tuple[CandidatePath, ...]], iterate: Iterate
) -> Plan:
    scenario = instance.scenario
    G = instance.block_slots
    total = scenario.spectrum.b_tot_hz
    bandwidth: dict[LinkId, np.ndarray] = {}
    downlinks: set[LinkId] = set()
    for devices, blocks, shares, hop in (
        (instance.uplink_device, instance.uplink_block, iterate.uplink, 0),
        (instance.downlink_device, instance.downlink_block, iterate.downlink, 1),
    ):
        for device_index, block, share in zip(devices, blocks, shares):
            device = instance.devices[device_index]
            link = candidates[device.alert_id][device.candidate].links[hop]
            values = bandwidth.setdefault(link, np.zeros(scenario.time.num_slots))
            values[block * G : (block + 1) * G] += share * total
            if hop == 1:
                downlinks.add(link)
    keep_top_downlinks(bandwidth, sorted(downlinks), scenario.uav.max_active_downlinks)
    path_choice = {device.alert_id: device.candidate for device in instance.devices}
    plan = Plan(expand_trajectory(iterate.trajectory, G), path_choice, StaticSchedule(bandwidth))
    violations = validate_plan(scenario, candidates, plan)
    if violations:
        raise ValueError(f"TRAN produced an invalid plan: {violations[:3]}")
    return plan
