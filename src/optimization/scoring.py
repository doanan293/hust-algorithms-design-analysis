import math
from typing import Mapping, Sequence

from models.evaluate import EXPECTED, REALIZED, EvaluationResult, evaluate
from models.paths import CandidatePath
from models.plan import Plan
from models.scenario import Scenario

from .objectives import Key, Objective, lexicographic_key

DESIGN_NAMESPACE = "design"
DEFAULT_DESIGN_IDS = (0, 1, 2, 3, 4)
DEFAULT_BUDGET = 2000
INFEASIBLE_SCORE: Key = (-math.inf,)


class BudgetExhausted(RuntimeError):
    """The scorer has used its whole evaluation budget."""


class DesignScorer:
    """Scores plans on design realizations, or on expected rates when design_ids is empty, under a call budget."""

    def __init__(
        self,
        scenario: Scenario,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        design_ids: Sequence[int] = DEFAULT_DESIGN_IDS,
        budget: int = DEFAULT_BUDGET,
        objective: Objective = lexicographic_key,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be at least 1")
        self.scenario = scenario
        self.candidates = candidates
        self.design_ids = tuple(design_ids)
        self.budget = budget
        self.objective = objective
        self.calls = 0

    @property
    def remaining(self) -> int:
        return self.budget - self.calls

    def score(self, plan: Plan) -> Key:
        return self._evaluate(plan, keep_ledger=False)[0]

    def score_with_ledger(self, plan: Plan) -> tuple[Key, EvaluationResult]:
        return self._evaluate(plan, keep_ledger=True)

    def _evaluate(self, plan: Plan, keep_ledger: bool) -> tuple[Key, EvaluationResult]:
        if self.calls >= self.budget:
            raise BudgetExhausted(f"the evaluation budget of {self.budget} calls is exhausted")
        self.calls += 1
        result = evaluate(
            self.scenario,
            plan,
            REALIZED if self.design_ids else EXPECTED,
            self.design_ids,
            candidates=self.candidates,
            keep_ledger=keep_ledger,
            channel_namespace=DESIGN_NAMESPACE,
        )
        key = self.objective(self.scenario, result) if result.feasible else INFEASIBLE_SCORE
        return key, result
