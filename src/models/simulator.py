from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

from .channel import ACCESS, BACKHAUL, DOWNLINK, LinkChannel, LinkId
from .ledger import Ledger
from .paths import CandidatePath
from .plan import Plan, StaticSchedule, WeightedBacklogged, chosen_paths
from .scenario import Scenario

DELIVERY_TOLERANCE_BITS = 1e-6


@dataclass(frozen=True)
class SimulationOutcome:
    delivered_bits: dict[str, float]
    delivery_slot: dict[str, int | None]
    ledger: Ledger


def simulate(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    plan: Plan,
    channels: Mapping[LinkId, LinkChannel],
    realization_id: int | None = None,
) -> SimulationOutcome:
    slot_s = scenario.time.slot_s
    access_s = scenario.time.access_fraction * slot_s
    downlink_s = slot_s - access_s
    paths = chosen_paths(scenario, candidates, plan)
    served = [alert for alert in scenario.alerts if paths[alert.id] is not None]
    buffers: dict[tuple[str, int], float] = {}
    delivered = {alert.id: 0.0 for alert in scenario.alerts}
    delivery_slot: dict[str, int | None] = {alert.id: None for alert in scenario.alerts}
    transfers: list[tuple[int, LinkId, str, float]] = []
    bandwidth_log: list[tuple[int, LinkId, float]] = []
    capacity_log: list[tuple[int, LinkId, float]] = []
    drops: list[tuple[int, str, str, float]] = []

    for slot in range(scenario.time.num_slots):
        for alert in served:
            links = paths[alert.id].links
            if slot >= alert.deadline_slot:
                for hop in range(len(links)):
                    bits = buffers.pop((alert.id, hop), 0.0)
                    if bits > 0.0:
                        drops.append((slot, alert.id, links[hop][1], bits))
            elif slot == alert.release_slot:
                buffers[(alert.id, 0)] = float(alert.size_bits)

        queues: dict[LinkId, list[tuple[int, str, int]]] = defaultdict(list)
        for (alert_id, hop), bits in buffers.items():
            if bits > 0.0:
                deadline = scenario.alert_by_id[alert_id].deadline_slot
                queues[paths[alert_id].links[hop]].append((deadline, alert_id, hop))

        allocation = _allocate_bandwidth(scenario, plan, queues, slot)
        for link in sorted(allocation):
            if allocation[link] > 0.0:
                bandwidth_log.append((slot, link, allocation[link]))

        arrivals: list[tuple[str, int, float]] = []
        for link in sorted(queues):
            if link[0] == BACKHAUL:
                capacity = slot_s * scenario.backhaul_capacity_bps(link[1], link[2])
            else:
                duration = access_s if link[0] == ACCESS else downlink_s
                capacity = duration * channels[link].rate_bps(slot, allocation.get(link, 0.0))
            capacity_log.append((slot, link, capacity))
            remaining = capacity
            for _, alert_id, hop in sorted(queues[link]):
                if remaining <= 0.0:
                    break
                sent = min(buffers[(alert_id, hop)], remaining)
                buffers[(alert_id, hop)] -= sent
                remaining -= sent
                transfers.append((slot, link, alert_id, sent))
                arrivals.append((alert_id, hop + 1, sent))

        for alert_id, hop, bits in arrivals:
            if hop == len(paths[alert_id].links):
                delivered[alert_id] += bits
                size = scenario.alert_by_id[alert_id].size_bits
                if delivery_slot[alert_id] is None and delivered[alert_id] >= size - DELIVERY_TOLERANCE_BITS:
                    delivery_slot[alert_id] = slot
            else:
                buffers[(alert_id, hop)] = buffers.get((alert_id, hop), 0.0) + bits
        buffers = {key: bits for key, bits in buffers.items() if bits > 0.0}

    ledger = Ledger(
        realization_id=realization_id,
        transfers=tuple(transfers),
        bandwidth=tuple(bandwidth_log),
        capacity_bits=tuple(capacity_log),
        drops=tuple(drops),
    )
    return SimulationOutcome(delivered_bits=delivered, delivery_slot=delivery_slot, ledger=ledger)


def _allocate_bandwidth(
    scenario: Scenario,
    plan: Plan,
    queues: Mapping[LinkId, list[tuple[int, str, int]]],
    slot: int,
) -> dict[LinkId, float]:
    if isinstance(plan.bandwidth, StaticSchedule):
        return {link: plan.bandwidth.at(link, slot) for link in plan.bandwidth.bandwidth_hz}
    total = scenario.spectrum.b_tot_hz
    access = sorted(link for link in queues if link[0] == ACCESS)
    ranked_downlinks = sorted(
        (min(entry[0] for entry in queues[link]), link[2], link) for link in queues if link[0] == DOWNLINK
    )
    active = [item[2] for item in ranked_downlinks[: scenario.uav.max_active_downlinks]]
    if isinstance(plan.bandwidth, WeightedBacklogged):
        allocation = {}
        for group in (access, active):
            weights = [plan.bandwidth.weight(link, slot) for link in group]
            weight_sum = sum(weights)
            for link, weight in zip(group, weights):
                allocation[link] = total * weight / weight_sum if weight_sum > 0.0 else total / len(group)
        return allocation
    allocation = {link: total / len(access) for link in access}
    allocation.update({link: total / len(active) for link in active})
    return allocation
