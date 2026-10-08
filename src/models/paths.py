from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping

from .channel import ACCESS, BACKHAUL, DOWNLINK, LinkId, ground_snr_per_hz, in_ground_range, rate_bps
from .scenario import Alert, Scenario, UAV_ID

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


def backhaul_routes(
    scenario: Scenario,
    excluded_edges: frozenset[frozenset[str]] = frozenset(),
    excluded_nodes: frozenset[str] = frozenset(),
) -> BackhaulRoutes:
    """Fewest-hop routes to the center over alive edges, ties by Euclidean length then node id; exclusions prune the graph."""
    neighbours: dict[str, set[str]] = {node.id: set() for node in scenario.nodes}
    for edge in scenario.edges:
        ends = frozenset((edge.source, edge.target))
        if edge.alive and ends not in excluded_edges and not ends & excluded_nodes:
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


def ground_path(scenario: Scenario, alert: Alert, node_id: str, route: tuple[LinkId, ...]) -> CandidatePath | None:
    """Ground path entering at `node_id` and following `route`; None when the source is out of range of the node."""
    source = scenario.source_by_id[alert.source]
    distance = math.dist(source.xy, scenario.node_by_id[node_id].xy)
    if not in_ground_range(scenario, distance):
        return None
    access_rate = rate_bps(scenario.spectrum.b_tot_hz, ground_snr_per_hz(scenario, distance))
    backhaul_s = _backhaul_delay_s(scenario, route, alert.size_bits)
    return CandidatePath(GROUND, node_id, ((ACCESS, source.id, node_id), *route), alert.size_bits / access_rate + backhaul_s)


def uav_path(scenario: Scenario, alert: Alert, node_id: str, route: tuple[LinkId, ...]) -> CandidatePath:
    """Path via the UAV that downlinks at `node_id` and follows `route`; cost uses the flight-distance proxy."""
    source = scenario.source_by_id[alert.source]
    distance = math.dist(source.xy, scenario.node_by_id[node_id].xy)
    backhaul_s = _backhaul_delay_s(scenario, route, alert.size_bits)
    links = ((ACCESS, source.id, UAV_ID), (DOWNLINK, UAV_ID, node_id), *route)
    return CandidatePath(VIA_UAV, node_id, links, distance / scenario.uav.v_max_mps + backhaul_s)


def candidate_paths(
    scenario: Scenario, routes: BackhaulRoutes | None = None
) -> dict[str, tuple[CandidatePath, ...]]:
    routes = routes or backhaul_routes(scenario)
    entries = sorted(routes.hop_count)
    candidates: dict[str, tuple[CandidatePath, ...]] = {}
    for alert in scenario.alerts:
        ground: list[CandidatePath] = []
        aerial: list[CandidatePath] = []
        for node_id in entries:
            route = routes.route_links(node_id)
            path = ground_path(scenario, alert, node_id, route)
            if path is not None:
                ground.append(path)
            aerial.append(uav_path(scenario, alert, node_id, route))
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


def radio_links_of_paths(paths: Mapping[str, "CandidatePath | None"]) -> tuple[LinkId, ...]:
    """Sorted radio links of the paths a plan uses; unserved alerts contribute nothing."""
    return tuple(sorted({link for path in paths.values() if path is not None for link in path.links if link[0] != BACKHAUL}))


def path_violations(scenario: Scenario, source_id: str, path: CandidatePath) -> list[str]:
    """Why `path` is not a valid route of an alert of `source_id`: structure, range, alive edges, chain to the center."""
    links = path.links
    if not links or links[0][0] != ACCESS or links[0][1] != source_id:
        return [f"first link {links[0] if links else None} is not an access link of {source_id}"]
    receiver = links[0][2]
    if receiver == UAV_ID:
        if path.kind != VIA_UAV:
            return [f"uses the UAV but has kind {path.kind!r}"]
        if path.entry not in scenario.node_by_id:
            return [f"entry {path.entry} is not a node"]
        if len(links) < 2 or links[1] != (DOWNLINK, UAV_ID, path.entry):
            return [f"second link must be the downlink to entry {path.entry}"]
        backhaul = links[2:]
    else:
        if path.kind != GROUND:
            return [f"enters the ground network but has kind {path.kind!r}"]
        if receiver != path.entry or path.entry not in scenario.node_by_id:
            return [f"entry {path.entry} does not match the access link {links[0]}"]
        distance = math.dist(scenario.source_by_id[source_id].xy, scenario.node_by_id[path.entry].xy)
        if not in_ground_range(scenario, distance):
            return [f"entry {path.entry} is out of ground range ({distance:.1f} m)"]
        backhaul = links[1:]
    problems = []
    current = path.entry
    visited = {current}
    for link in backhaul:
        if link[0] != BACKHAUL or link[1] != current:
            problems.append(f"link {link} breaks the chain at {current}")
            break
        if scenario.backhaul_capacity_bps(link[1], link[2]) <= 0.0:
            problems.append(f"link {link} is not an alive edge")
        if link[2] in visited:
            problems.append(f"node {link[2]} repeats")
        visited.add(link[2])
        current = link[2]
    if current != scenario.center:
        problems.append(f"ends at {current}, not at the center {scenario.center}")
    return problems


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
