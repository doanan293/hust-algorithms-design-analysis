from typing import Mapping, Protocol

import numpy as np

from models.evaluate import EvaluationCounter
from models.paths import CandidatePath
from models.plan import Plan
from models.scenario import Scenario


class Method(Protocol):
    """A planning method; the runner scores every plan with models.evaluate."""

    name: str
    uses_uav: bool

    def trajectory(self, scenario: Scenario) -> np.ndarray: ...

    def plan(
        self,
        scenario: Scenario,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        counter: EvaluationCounter,
    ) -> Plan: ...
