from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from models.channel import LinkChannel, LinkId, build_link_channels
from models.evaluate import EvaluationCounter
from models.paths import CandidatePath, radio_links
from models.plan import EqualSplitBacklogged, Plan
from models.scenario import Scenario
from models.simulator import simulate

from .trajectories import zone_tour_trajectory


@dataclass(frozen=True)
class B1:
    """Fixed zone-tour trajectory, path with the earliest estimated delivery, EDF, equal split."""

    name: str = "B1"
    uses_uav: bool = True

    def trajectory(self, scenario: Scenario) -> np.ndarray:
        return zone_tour_trajectory(scenario)

    def plan(
        self,
        scenario: Scenario,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        counter: EvaluationCounter,
    ) -> Plan:
        trajectory = self.trajectory(scenario)
        channels = build_link_channels(scenario, radio_links(candidates), trajectory, None)
        choice = {
            alert.id: estimated_best_candidate(scenario, candidates, trajectory, channels, alert.id)
            for alert in scenario.alerts
        }
        return Plan(trajectory, choice, EqualSplitBacklogged())


def estimated_best_candidate(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    trajectory: np.ndarray,
    channels: Mapping[LinkId, LinkChannel],
    alert_id: str,
) -> int | None:
    """Simulate the alert alone on expected-rate channels; earliest delivery wins, ties go to the lower index."""
    alone = replace(scenario, alerts=(scenario.alert_by_id[alert_id],))
    best: tuple[int, int] | None = None
    for index in range(len(candidates[alert_id])):
        plan = Plan(trajectory, {alert_id: index}, EqualSplitBacklogged())
        slot = simulate(alone, candidates, plan, channels).delivery_slot[alert_id]
        if slot is not None and (best is None or slot < best[0]):
            best = (slot, index)
    return None if best is None else best[1]
