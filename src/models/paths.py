from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping

from .channel import ACCESS, BACKHAUL, DOWNLINK, LinkId, ground_snr_per_hz, in_ground_range, rate_bps
from .scenario import Scenario, UAV_ID

GROUND = "ground"
VIA_UAV = "uav"


@dataclass(frozen=True)
class CandidatePath:
    kind: str
    entry: str
    links: tuple[LinkId, ...]
    cost_s: float

    @property
    def holders(self) -> tuple[str, ...]:
        return tuple(link[1] for link in self.links)


@dataclass(frozen=True)
class BackhaulRoutes:
    center: str
    next_hop: Mapping[str, str]
    hop_count: Mapping[str, int]

    def route_links(self, node_id: str) -> tuple[LinkId, ...]:
        links = []
        current = node_id
        while current != self.center:
            following = self.next_hop[current]
            links.append((BACKHAUL, current, following))
            current = following
        return tuple(links)


def backhaul_routes(scenario: Scenario) -> BackhaulRoutes:
    neighbours: dict[str, set[str]] = {node.id: set() for node in scenario.nodes}
    for edge in scenario.edges:
        if edge.alive:
            neighbours[edge.source].add(edge.target)
            neighbours[edge.target].add(edge.source)
    hop_count = {scenario.center: 0}
    queue = deque([scenario.center])
    while queue:
        current = queue.popleft()
        for other in sorted(neighbours[current]):
            if other not in hop_count:
                hop_count[other] = hop_count[current] + 1
                queue.append(other)
    length = {scenario.center: 0.0}
    next_hop: dict[str, str] = {}
    for node_id in sorted(hop_count, key=lambda item: (hop_count[item], item)):
        if node_id == scenario.center:
            continue
        options = [
            (math.dist(scenario.position(node_id), scenario.position(other)) + length[other], other)
            for other in neighbours[node_id]
            if hop_count.get(other) == hop_count[node_id] - 1
        ]
        length[node_id], next_hop[node_id] = min(options)
    return BackhaulRoutes(center=scenario.center, next_hop=next_hop, hop_count=hop_count)


def _backhaul_delay_s(scenario: Scenario, route: tuple[LinkId, ...], size_bits: int) -> float:
    return sum(
        size_bits / scenario.backhaul_capacity_bps(link[1], link[2]) + scenario.time.slot_s for link in route
    )


def candidate_paths(
    scenario: Scenario, routes: BackhaulRoutes | None = None
) -> dict[str, tuple[CandidatePath, ...]]:
    routes = routes or backhaul_routes(scenario)
    entries = sorted(routes.hop_count)
    bandwidth = scenario.spectrum.b_tot_hz
    candidates: dict[str, tuple[CandidatePath, ...]] = {}
    for alert in scenario.alerts:
        source = scenario.source_by_id[alert.source]
        ground: list[CandidatePath] = []
        aerial: list[CandidatePath] = []
        for node_id in entries:
            route = routes.route_links(node_id)
            backhaul_s = _backhaul_delay_s(scenario, route, alert.size_bits)
            distance = math.dist(source.xy, scenario.node_by_id[node_id].xy)
            if in_ground_range(scenario, distance):
                access_rate = rate_bps(bandwidth, ground_snr_per_hz(scenario, distance))
                ground.append(
                    CandidatePath(
                        kind=GROUND,
                        entry=node_id,
                        links=((ACCESS, source.id, node_id), *route),
                        cost_s=alert.size_bits / access_rate + backhaul_s,
                    )
                )
            aerial.append(
                CandidatePath(
                    kind=VIA_UAV,
                    entry=node_id,
                    links=((ACCESS, source.id, UAV_ID), (DOWNLINK, UAV_ID, node_id), *route),
                    cost_s=distance / scenario.uav.v_max_mps + backhaul_s,
                )
            )
        ground.sort(key=lambda path: (path.cost_s, path.entry))
        aerial.sort(key=lambda path: (path.cost_s, path.entry))
        chosen = ground[: scenario.paths.max_ground]
        candidates[alert.id] = tuple(chosen + aerial[: scenario.paths.k - len(chosen)])
    return candidates


def radio_links(candidates: Mapping[str, tuple[CandidatePath, ...]]) -> tuple[LinkId, ...]:
    return tuple(
        sorted(
            {
                link
                for paths in candidates.values()
                for path in paths
                for link in path.links
                if link[0] != BACKHAUL
            }
        )
    )


def ground_connected_source_fraction(scenario: Scenario, routes: BackhaulRoutes | None = None) -> float:
    routes = routes or backhaul_routes(scenario)
    connected = sum(
        1
        for source in scenario.sources
        if any(
            in_ground_range(scenario, math.dist(source.xy, scenario.node_by_id[node_id].xy))
            for node_id in routes.hop_count
        )
    )
    return connected / len(scenario.sources)
