"""Planning method TRAN: Tran et al. (2022) half-duplex design adapted to the project model (spec D Section 5)."""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from models.evaluate import EvaluationCounter
from models.paths import CandidatePath
from models.plan import Plan
from models.scenario import Scenario

from .tran_ia import initial_iterate, run_inner_approximation
from .tran_model import EffectiveChannel, TranParams, build_instance
from .tran_plan import plan_from_iterate


@dataclass(frozen=True, eq=False)
class TranOutcome:
    plan: Plan
    initial_plan: Plan
    channel: EffectiveChannel
    objectives: tuple[float, ...]
    status: str
    iterations: int
    penalty: float


@dataclass(frozen=True)
class TranHD:
    name: str = "TRAN"
    params: TranParams = field(default_factory=TranParams)
    uses_uav: bool = True

    def trajectory(self, scenario: Scenario) -> np.ndarray:
        raise NotImplementedError(f"{self.name}: the trajectory of TRAN exists only after planning")

    def plan(
        self,
        scenario: Scenario,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        counter: EvaluationCounter,
    ) -> Plan:
        return self.solve(scenario, candidates).plan

    def solve(self, scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]]) -> TranOutcome:
        instance = build_instance(scenario, candidates, self.params)
        initial = initial_iterate(instance)
        result = run_inner_approximation(instance, initial, self.params)
        lam = result.iterate.lam
        return TranOutcome(
            plan=plan_from_iterate(instance, candidates, result.iterate),
            initial_plan=plan_from_iterate(instance, candidates, initial),
            channel=instance.channel,
            objectives=result.objectives,
            status=result.status,
            iterations=result.iterations,
            penalty=float(np.sum(lam * (lam - 1.0))),
        )
