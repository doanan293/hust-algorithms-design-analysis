"""Neighbours of one alert's path, generated from the incumbent path and the incumbent ledger (spec E Section 5)."""

import math
from typing import Mapping, Sequence

import numpy as np

from models.channel import BACKHAUL, ground_snr_per_hz, in_ground_range, rate_bps, slot_midpoints
from models.paths import GROUND, VIA_UAV, BackhaulRoutes, CandidatePath, backhaul_routes, ground_path, uav_path
from models.scenario import Alert, Scenario, SourcePoint

from .ledger_summary import LedgerSummary, alert_window, residual_bps

MIN_TIME_S = 1e-3


def is_ground_connected(scenario: Scenario, source: SourcePoint, routes: BackhaulRoutes) -> bool:
    """The B0 test: some reachable node lies within ground range of the source."""
    return any(in_ground_range(scenario, math.dist(source.xy, scenario.node_by_id[node_id].xy)) for node_id in routes.hop_count)


def backhaul_links(path: CandidatePath) -> tuple[tuple[str, str, str], ...]:
    return tuple(link for link in path.links if link[0] == BACKHAUL)


def kind_flip(scenario: Scenario, alert: Alert, path: CandidatePath) -> CandidatePath | None:
    """Same entry and backhaul, the other kind; None when a ground entry would be out of range."""
    route = backhaul_links(path)
    if path.kind == GROUND:
        return uav_path(scenario, alert, path.entry, route)
    return ground_path(scenario, alert, path.entry, route)


def estimated_completion_s(
    scenario: Scenario, alert: Alert, path: CandidatePath, trajectory: np.ndarray, summary: LedgerSummary
) -> float:
    """Access time plus backhaul time at residual capacity; the UAV access time follows the incumbent trajectory."""
    source = scenario.source_by_id[alert.source]
    node = scenario.node_by_id[path.entry]
    window = alert_window(scenario, alert)
    if path.kind == GROUND:
        distance = math.dist(source.xy, node.xy)
        access_s = alert.size_bits / rate_bps(scenario.spectrum.b_tot_hz, ground_snr_per_hz(scenario, distance))
    else:
        midpoints = slot_midpoints(trajectory)[window.start : window.stop]
        if len(midpoints):
            distance = float(np.mean(np.linalg.norm(midpoints - np.asarray(node.xy), axis=1)))
        else:
            distance = math.dist(source.xy, node.xy)
        access_s = distance / scenario.uav.v_max_mps
    backhaul_s = sum(
        alert.size_bits / residual_bps(scenario, summary, link, window) + scenario.time.slot_s for link in backhaul_links(path)
    )
    return access_s + backhaul_s


def entry_switches(
    scenario: Scenario,
    alert: Alert,
    path: CandidatePath,
    routes: BackhaulRoutes,
    trajectory: np.ndarray,
    summary: LedgerSummary,
    rng: np.random.Generator,
    count: int,
) -> list[CandidatePath]:
    """Up to `count` paths of the same kind through other entries, drawn with probability ∝ 1/T, in increasing T."""
    options = []
    for node_id in sorted(routes.hop_count):
        if node_id == path.entry:
            continue
        route = routes.route_links(node_id)
        option = ground_path(scenario, alert, node_id, route) if path.kind == GROUND else uav_path(scenario, alert, node_id, route)
        if option is not None:
            options.append(option)
    if not options:
        return []
    times = np.array([estimated_completion_s(scenario, alert, option, trajectory, summary) for option in options])
    weights = 1.0 / np.maximum(times, MIN_TIME_S)
    drawn = rng.choice(len(options), size=min(count, len(options)), replace=False, p=weights / weights.sum())
    return [options[index] for index in sorted(drawn, key=lambda index: (times[index], options[index].entry))]


def backhaul_detour(scenario: Scenario, alert: Alert, path: CandidatePath, summary: LedgerSummary) -> CandidatePath | None:
    """Reroute from the head of the most congested backhaul link around that link; None without congestion or route."""
    window = alert_window(scenario, alert)
    links = backhaul_links(path)
    if not links:
        return None
    counts = [summary.congestion_count(link, window) for link in links]
    worst = max(range(len(links)), key=lambda index: (counts[index], -index))
    if counts[worst] == 0:
        return None
    _, head, tail = links[worst]
    prefix_nodes = frozenset(link[1] for link in links[:worst])
    routes = backhaul_routes(scenario, excluded_edges=frozenset({frozenset((head, tail))}), excluded_nodes=prefix_nodes)
    if head not in routes.hop_count:
        return None
    first_backhaul = path.links.index(links[0])
    new_links = (*path.links[: first_backhaul + worst], *routes.route_links(head))
    if new_links == path.links:
        return None
    constructor = ground_path if path.kind == GROUND else uav_path
    rebuilt = constructor(scenario, alert, path.entry, tuple(link for link in new_links if link[0] == BACKHAUL))
    return rebuilt


def path_neighbours(
    scenario: Scenario,
    alert: Alert,
    current: CandidatePath | None,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    routes: BackhaulRoutes,
    trajectory: np.ndarray,
    summary: LedgerSummary,
    rng: np.random.Generator,
    entry_neighbors: int,
) -> list[CandidatePath | None]:
    """Kind flip, entry switches, backhaul detour, unserve — distinct, valid, different from `current`."""
    if current is None:
        return list(candidates[alert.id])
    source = scenario.source_by_id[alert.source]
    ground_ok = is_ground_connected(scenario, source, routes)
    moves: list[CandidatePath | None] = []
    flipped = kind_flip(scenario, alert, current)
    if flipped is not None and (flipped.kind == VIA_UAV or ground_ok):
        moves.append(flipped)
    if current.kind == VIA_UAV or ground_ok:
        moves.extend(entry_switches(scenario, alert, current, routes, trajectory, summary, rng, entry_neighbors))
    detour = backhaul_detour(scenario, alert, current, summary)
    if detour is not None:
        moves.append(detour)
    moves.append(None)
    distinct: list[CandidatePath | None] = []
    for move in moves:
        if move != current and move not in distinct:
            distinct.append(move)
    return distinct


def urgency_order(alerts: Sequence[Alert], rng: np.random.Generator) -> list[Alert]:
    """Random order drawn without replacement with probability ∝ 1/(deadline − release): urgent alerts tend to come first."""
    if not alerts:
        return []
    weights = np.array([1.0 / max(alert.deadline_slot - alert.release_slot, 1) for alert in alerts])
    order = rng.choice(len(alerts), size=len(alerts), replace=False, p=weights / weights.sum())
    return [alerts[index] for index in order]
