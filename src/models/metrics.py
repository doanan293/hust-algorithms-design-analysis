from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .channel import ACCESS, DOWNLINK, LinkChannel, LinkId, all_uav_links, in_ground_range
from .scenario import Scenario, UAV_ID

DELIVERY_TOLERANCE_BITS = 1e-6


def timely_flags(scenario: Scenario, delivered_bits: Mapping[str, float]) -> dict[str, bool]:
    return {
        alert.id: delivered_bits[alert.id] >= alert.size_bits - DELIVERY_TOLERANCE_BITS
        for alert in scenario.alerts
    }


def shortfall(scenario: Scenario, delivered_bits: Mapping[str, float]) -> float:
    return sum(
        max(0.0, alert.size_bits - delivered_bits[alert.id]) / alert.size_bits for alert in scenario.alerts
    )


@dataclass(frozen=True)
class Connectivity:
    connected: dict[str, bool]
    disconnected_slots: dict[str, int]


def connectivity(scenario: Scenario, uav_channels: Mapping[LinkId, LinkChannel] | None) -> Connectivity:
    """Backward reachability to the center on the time-expanded graph; None removes the UAV."""
    num_slots = scenario.time.num_slots
    neighbours: dict[str, list[str]] = {node.id: [] for node in scenario.nodes}
    for edge in scenario.edges:
        if edge.alive:
            neighbours[edge.source].append(edge.target)
            neighbours[edge.target].append(edge.source)
    entries = {
        source.id: [
            node.id for node in scenario.nodes if in_ground_range(scenario, math.dist(source.xy, node.xy))
        ]
        for source in scenario.sources
    }
    usable: dict[LinkId, np.ndarray] = {}
    if uav_channels is not None:
        for link in all_uav_links(scenario):
            channel = uav_channels[link]
            usable[link] = np.array(
                [
                    channel.rate_bps(slot, scenario.spectrum.b_tot_hz) >= scenario.usable_rate_bps
                    for slot in range(num_slots)
                ]
            )

    node_reach = {node.id: node.id == scenario.center for node in scenario.nodes}
    uav_reach = False
    source_reach = {source.id: False for source in scenario.sources}
    disconnected = {source.id: 0 for source in scenario.sources}
    for slot in range(num_slots - 1, -1, -1):
        next_nodes, next_uav, next_sources = node_reach, uav_reach, source_reach
        node_reach = {
            node_id: next_nodes[node_id] or any(next_nodes[other] for other in neighbours[node_id])
            for node_id in next_nodes
        }
        uav_reach = next_uav or any(
            next_nodes[node.id] and usable[(DOWNLINK, UAV_ID, node.id)][slot]
            for node in scenario.nodes
            if usable
        )
        source_reach = {}
        for source in scenario.sources:
            reach = (
                next_sources[source.id]
                or any(next_nodes[node_id] for node_id in entries[source.id])
                or (bool(usable) and next_uav and bool(usable[(ACCESS, source.id, UAV_ID)][slot]))
            )
            source_reach[source.id] = reach
            if not reach:
                disconnected[source.id] += 1
    return Connectivity(connected=source_reach, disconnected_slots=disconnected)
