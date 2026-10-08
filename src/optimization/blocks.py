"""Search state and the path, bandwidth, and trajectory blocks of the evaluator-guided BCD (spec C Section 7.2, spec E Sections 5-6)."""

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from models.channel import BACKHAUL, LinkId
from models.paths import BackhaulRoutes, CandidatePath, radio_links, radio_links_of_paths
from models.plan import Plan, WeightedBacklogged
from models.scenario import Scenario

from .ledger_summary import LedgerSummary
from .objectives import Key
from .path_moves import path_neighbours, urgency_order
from .scoring import INFEASIBLE_SCORE, DesignScorer
from .tours import TourSpec
from .trajectory_moves import feasible_segment, sample_uav_segments, shift_segment


@dataclass
class SearchState:
    """The incumbent plan and its key; blocks change it only through try_change, so accepted work survives BudgetExhausted."""

    trajectory: np.ndarray
    paths: dict[str, CandidatePath | None]
    weights: dict[LinkId, np.ndarray]
    key: Key = INFEASIBLE_SCORE

    def plan(self) -> Plan:
        return Plan(self.trajectory, dict(self.paths), WeightedBacklogged(dict(self.weights)))

    def try_change(self, scorer: DesignScorer, **changes: object) -> bool:
        """Score the state with `changes` applied and adopt it only if the key increases strictly; new links get unit weights."""
        trial = replace(self, **changes)
        trial.weights = with_unit_weights(trial.weights, trial.paths, len(trial.trajectory) - 1)
        key = scorer.score(trial.plan())
        if key <= self.key:
            return False
        self.trajectory, self.paths, self.weights, self.key = trial.trajectory, trial.paths, trial.weights, key
        return True


def unit_weights(scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]]) -> dict[LinkId, np.ndarray]:
    return {link: np.ones(scenario.time.num_slots) for link in radio_links(candidates)}


def with_unit_weights(
    weights: Mapping[LinkId, np.ndarray], paths: Mapping[str, CandidatePath | None], num_slots: int
) -> dict[LinkId, np.ndarray]:
    """`weights` plus a unit vector for every radio link of `paths` that has none; the same mapping object when nothing is missing."""
    missing = [link for link in radio_links_of_paths(paths) if link not in weights]
    if not missing:
        return weights
    return {**weights, **{link: np.ones(num_slots) for link in missing}}


def links_by_load(scenario: Scenario, paths: Mapping[str, CandidatePath | None]) -> list[LinkId]:
    """Radio links of chosen paths by total routed alert size, descending; ties by link."""
    load: dict[LinkId, int] = defaultdict(int)
    for alert in scenario.alerts:
        path = paths[alert.id]
        if path is not None:
            for link in path.links:
                if link[0] != BACKHAUL:
                    load[link] += alert.size_bits
    return sorted(load, key=lambda link: (-load[link], link))


def bandwidth_block(scenario: Scenario, scorer: DesignScorer, state: SearchState) -> bool:
    """Per link: try doubling its weights, and halving them only if doubling was rejected."""
    accepted = False
    for link in links_by_load(scenario, state.paths):
        for factor in (2.0, 0.5):
            if state.try_change(scorer, weights={**state.weights, link: state.weights[link] * factor}):
                accepted = True
                break
    return accepted


def path_block(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    routes: BackhaulRoutes,
    scorer: DesignScorer,
    state: SearchState,
    params: "SearchParams",
    rng: np.random.Generator,
) -> bool:
    """One ledger call, then for each alert in urgency order every neighbour of its current path, first improvement."""
    _, result = scorer.score_with_ledger(state.plan())
    summary = LedgerSummary.from_result(result)
    accepted = False
    for alert in urgency_order(list(scenario.alerts), rng):
        neighbours = path_neighbours(
            scenario, alert, state.paths[alert.id], candidates, routes, state.trajectory, summary, rng, params.entry_neighbors
        )
        for path in neighbours:
            if state.try_change(scorer, paths={**state.paths, alert.id: path}):
                accepted = True
    return accepted


def catalog_trajectory_block(scorer: DesignScorer, state: SearchState, family: Sequence[tuple[TourSpec, np.ndarray]]) -> bool:
    """Try every pre-enumerated tour (the pre-local-search block, kept for the P_catalog_trajectory ablation)."""
    accepted = False
    for _, trajectory in family:
        if not np.array_equal(trajectory, state.trajectory) and state.try_change(scorer, trajectory=trajectory):
            accepted = True
    return accepted


def trajectory_block(
    scenario: Scenario,
    scorer: DesignScorer,
    state: SearchState,
    params: "SearchParams",
    rng: np.random.Generator,
    family: Sequence[tuple[TourSpec, np.ndarray]],
) -> bool:
    """One ledger call, W sampled segment shifts (full then half reach), and a rare catalogue tour."""
    _, result = scorer.score_with_ledger(state.plan())
    summary = LedgerSummary.from_result(result)
    step_m = scenario.uav.v_max_mps * scenario.time.slot_s
    accepted = False
    for segment in sample_uav_segments(scenario, state.plan(), summary, rng, params.trajectory_segments):
        bounds = feasible_segment(state.trajectory, segment.start, segment.stop, segment.direction, step_m, params.max_widening)
        if bounds is None:
            continue
        for fraction in (1.0, 0.5):
            trial = shift_segment(state.trajectory, *bounds, segment.direction, fraction, step_m)
            if trial is not None and state.try_change(scorer, trajectory=trial):
                accepted = True
                break
    if family and rng.random() < params.catalog_probability:
        tour = family[int(rng.integers(len(family)))][1]
        if not np.array_equal(tour, state.trajectory) and state.try_change(scorer, trajectory=tour):
            accepted = True
    return accepted
