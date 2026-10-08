"""Search state and the path, bandwidth, and trajectory blocks of the evaluator-guided BCD (spec C Section 7.2)."""

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from models.channel import BACKHAUL, LinkId
from models.paths import CandidatePath, radio_links
from models.plan import Plan, WeightedBacklogged
from models.scenario import Scenario

from .objectives import Key
from .scoring import INFEASIBLE_SCORE, DesignScorer
from .tours import TourSpec


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
        """Score the state with `changes` applied and adopt it only if the key increases strictly."""
        trial = replace(self, **changes)
        key = scorer.score(trial.plan())
        if key <= self.key:
            return False
        self.trajectory, self.paths, self.weights, self.key = trial.trajectory, trial.paths, trial.weights, key
        return True


def unit_weights(scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]]) -> dict[LinkId, np.ndarray]:
    return {link: np.ones(scenario.time.num_slots) for link in radio_links(candidates)}


def path_block(
    scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]], scorer: DesignScorer, state: SearchState
) -> bool:
    """Alerts by (deadline, id); options None and every candidate except the current one, each tried from the incumbent."""
    accepted = False
    for alert in sorted(scenario.alerts, key=lambda item: (item.deadline_slot, item.id)):
        current = state.paths[alert.id]
        for option in (None, *candidates[alert.id]):
            if option != current and state.try_change(scorer, paths={**state.paths, alert.id: option}):
                accepted = True
    return accepted


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


def trajectory_block(
    scorer: DesignScorer, state: SearchState, family: Sequence[tuple[TourSpec, np.ndarray]]
) -> bool:
    accepted = False
    for _, trajectory in family:
        if not np.array_equal(trajectory, state.trajectory) and state.try_change(scorer, trajectory=trajectory):
            accepted = True
    return accepted
