from dataclasses import dataclass
from typing import Mapping

import numpy as np

from models.evaluate import EvaluationCounter
from models.paths import GROUND, CandidatePath
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import Scenario


@dataclass(frozen=True)
class B0:
    """No UAV: lowest-cost ground path per alert, EDF, equal bandwidth split."""

    name: str = "B0"
    uses_uav: bool = False

    def trajectory(self, scenario: Scenario) -> np.ndarray:
        return stationary_trajectory(scenario)

    def plan(
        self,
        scenario: Scenario,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        counter: EvaluationCounter,
    ) -> Plan:
        choice = {
            alert.id: next((index for index, path in enumerate(candidates[alert.id]) if path.kind == GROUND), None)
            for alert in scenario.alerts
        }
        return Plan(self.trajectory(scenario), choice, EqualSplitBacklogged())
