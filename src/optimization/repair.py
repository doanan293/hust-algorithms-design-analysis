"""Risk detection and the two repair operations of method P (spec C Section 9)."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from models.channel import BACKHAUL, LinkId
from models.evaluate import EvaluationResult
from models.ledger import Ledger
from models.paths import BackhaulRoutes, CandidatePath
from models.scenario import Alert, Scenario

from .blocks import SearchState
from .ledger_summary import LedgerSummary, congested_slots  # noqa: F401 (congested_slots is re-exported)
from .path_moves import path_neighbours
from .scoring import DesignScorer

RISK_SLACK_SLOTS = 2
AT_RISK_LIMIT = 10
BUFFER_TOLERANCE_BITS = 1e-6


@dataclass(frozen=True)
class AlertRisk:
    alert_id: str
    risk: float
    bottleneck: LinkId
    congested_links: frozenset[LinkId]


def slack_slots(alert: Alert, delivery_slot: int | None, timely: bool) -> int:
    return alert.deadline_slot - 1 - delivery_slot if timely and delivery_slot is not None else -1


def bottleneck_hop(scenario: Scenario, alert: Alert, path: CandidatePath, transfers) -> int:
    """Hop whose buffer held bits of the alert at the start of the most slots in [r_a, d_a); ties go to the earliest hop."""
    num_slots = scenario.time.num_slots
    hop_of = {link: hop for hop, link in enumerate(path.links)}
    net = np.zeros((len(path.links), num_slots))
    for slot, link, bits in transfers:
        hop = hop_of[link]
        net[hop, slot] -= bits
        if hop + 1 < len(path.links):
            net[hop + 1, slot] += bits
    held = np.zeros_like(net)
    held[:, 1:] = np.cumsum(net, axis=1)[:, :-1]
    held[0, alert.release_slot:] += alert.size_bits
    window = held[:, alert.release_slot : min(alert.deadline_slot, num_slots)]
    return int(np.argmax((window > BUFFER_TOLERANCE_BITS).sum(axis=1)))


def assess_risk(
    scenario: Scenario,
    paths: Mapping[str, CandidatePath | None],
    result: EvaluationResult,
    backlog: bool = True,
    slack_limit: int = RISK_SLACK_SLOTS,
    limit: int = AT_RISK_LIMIT,
) -> list[AlertRisk]:
    """At-risk served alerts ordered by (-risk, deadline, id), truncated to `limit`."""
    realizations = result.realizations
    ranked = []
    for alert in scenario.alerts:
        if paths[alert.id] is None:
            continue
        risky = sum(
            slack_slots(alert, item.delivery_slot[alert.id], item.timely[alert.id]) <= slack_limit for item in realizations
        )
        if risky:
            ranked.append((-risky / len(realizations), alert.deadline_slot, alert.id))
    ranked = sorted(ranked)[:limit]
    if not backlog:
        return [
            AlertRisk(alert_id, -negative, paths[alert_id].links[0], frozenset())
            for negative, _, alert_id in ranked
        ]
    selected = {alert_id for _, _, alert_id in ranked}
    transfers = []
    congestion = []
    for item in realizations:
        if item.ledger is None:
            raise ValueError("risk detection with backlog information needs ledgers")
        by_alert: dict[str, list[tuple[int, LinkId, float]]] = defaultdict(list)
        for slot, link, alert_id, bits in item.ledger.transfers:
            if alert_id in selected:
                by_alert[alert_id].append((slot, link, bits))
        transfers.append(by_alert)
        congestion.append(congested_slots(item.ledger))
    risks = []
    for negative, _, alert_id in ranked:
        alert = scenario.alert_by_id[alert_id]
        path = paths[alert_id]
        hops = Counter(bottleneck_hop(scenario, alert, path, by_alert[alert_id]) for by_alert in transfers)
        hop = min(hops, key=lambda item: (-hops[item], item))
        window = set(range(alert.release_slot, alert.deadline_slot))
        congested = frozenset(link for slots_by_link in congestion for link, slots in slots_by_link.items() if slots & window)
        risks.append(AlertRisk(alert_id, -negative, path.links[hop], congested))
    return risks


def shift_bandwidth(
    weights: Mapping[LinkId, np.ndarray], link: LinkId, alert: Alert, num_slots: int
) -> dict[LinkId, np.ndarray]:
    """Operation 1: double the link's weights in slots [r_a, d_a) only."""
    values = weights[link].copy()
    values[alert.release_slot : min(alert.deadline_slot, num_slots)] *= 2.0
    return {**weights, link: values}


def repair_order(neighbours: Sequence[CandidatePath | None], risk: AlertRisk, backlog: bool = True) -> list[CandidatePath]:
    """Operation 2 order: served neighbours by (links in the congested set, cost_s); without backlog by cost_s only."""
    paths = [path for path in neighbours if path is not None]

    def order(path: CandidatePath) -> tuple:
        congested = sum(link in risk.congested_links for link in path.links) if backlog else 0
        return (congested, path.cost_s, path.links)

    return sorted(paths, key=order)


def repair_round(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    routes: BackhaulRoutes,
    scorer: DesignScorer,
    state: SearchState,
    params: "SearchParams",
    rng: np.random.Generator,
) -> bool:
    """One detection call, then operation 1 and operation 2 per at-risk alert."""
    _, result = scorer.score_with_ledger(state.plan())
    summary = LedgerSummary.from_result(result)
    accepted = False
    for risk in assess_risk(scenario, state.paths, result, params.backlog):
        alert = scenario.alert_by_id[risk.alert_id]
        if params.operation1 and risk.bottleneck[0] != BACKHAUL:
            weights = shift_bandwidth(state.weights, risk.bottleneck, alert, scenario.time.num_slots)
            accepted |= state.try_change(scorer, weights=weights)
        if params.operation2:
            neighbours = path_neighbours(
                scenario, alert, state.paths[alert.id], candidates, routes, state.trajectory, summary, rng, params.entry_neighbors
            )
            for path in repair_order(neighbours, risk, params.backlog):
                if state.try_change(scorer, paths={**state.paths, alert.id: path}):
                    accepted = True
                    break
    return accepted
