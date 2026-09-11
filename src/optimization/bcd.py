"""SearchMethod: B2, B3, P and the ablation variants as parameter sets of one evaluator-guided BCD."""

from dataclasses import dataclass, field, fields
from typing import Mapping

import numpy as np

from baselines.b1 import B1
from models.evaluate import EvaluationCounter
from models.paths import CandidatePath
from models.plan import Plan
from models.scenario import Scenario

from .blocks import SearchState, bandwidth_block, path_block, trajectory_block, unit_weights
from .objectives import OBJECTIVES, Key
from .repair import repair_round
from .scoring import DEFAULT_BUDGET, DEFAULT_DESIGN_IDS, BudgetExhausted, DesignScorer
from .tours import tour_family

EXPECTED_DESIGN = "expected"


@dataclass(frozen=True)
class SearchParams:
    """Defaults are method B2; an empty design_ids tuple scores expected rates."""

    objective: str = "lex"
    path_block: bool = True
    bandwidth_block: bool = True
    trajectory_block: bool = True
    operation1: bool = False
    operation2: bool = False
    backlog: bool = True
    design_ids: tuple[int, ...] = DEFAULT_DESIGN_IDS
    budget: int = DEFAULT_BUDGET
    max_iterations: int = 10

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective: expected one of {sorted(OBJECTIVES)}, got {self.objective!r}")
        for name in ("path_block", "bandwidth_block", "trajectory_block", "operation1", "operation2", "backlog"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name}: expected true or false")
        for name in ("budget", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}: expected an integer of at least 1")
        ids = self.design_ids
        if not isinstance(ids, tuple) or len(set(ids)) != len(ids) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ids
        ):
            raise ValueError("design_ids: expected distinct non-negative integers or 'expected'")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "SearchParams":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown search parameters {unknown}")
        values = dict(raw)
        if "design_ids" in values:
            ids = values["design_ids"]
            if ids == EXPECTED_DESIGN:
                values["design_ids"] = ()
            elif isinstance(ids, (list, tuple)) and ids:
                values["design_ids"] = tuple(ids)
            else:
                raise ValueError("design_ids: expected a non-empty list or 'expected'")
        return cls(**values)


@dataclass(frozen=True)
class SearchOutcome:
    plan: Plan
    key: Key
    calls: int
    iteration_keys: tuple[Key, ...]
    budget_exhausted: bool


@dataclass(frozen=True)
class SearchMethod:
    """Starts from B1 with unit weights; iterates blocks and repair until an iteration accepts nothing."""

    name: str
    params: SearchParams = field(default_factory=SearchParams)
    uses_uav: bool = True

    def trajectory(self, scenario: Scenario) -> np.ndarray:
        raise NotImplementedError(f"{self.name}: the trajectory of a search method exists only after planning")

    def plan(
        self,
        scenario: Scenario,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        counter: EvaluationCounter,
    ) -> Plan:
        outcome = self.search(scenario, candidates)
        counter.calls += outcome.calls
        return outcome.plan

    def search(self, scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]]) -> SearchOutcome:
        params = self.params
        scorer = DesignScorer(scenario, candidates, params.design_ids, params.budget, OBJECTIVES[params.objective])
        initial = B1().plan(scenario, candidates, EvaluationCounter())
        state = SearchState(initial.trajectory, dict(initial.path_choice), unit_weights(scenario, candidates))
        family = tour_family(scenario) if params.trajectory_block else []
        iteration_keys: list[Key] = []
        exhausted = False
        try:
            state.key = scorer.score(state.plan())
            for _ in range(params.max_iterations):
                accepted = False
                if params.path_block:
                    accepted |= path_block(scenario, candidates, scorer, state)
                if params.bandwidth_block:
                    accepted |= bandwidth_block(scenario, candidates, scorer, state)
                if params.trajectory_block:
                    accepted |= trajectory_block(scorer, state, family)
                if params.operation1 or params.operation2:
                    accepted |= repair_round(
                        scenario, candidates, scorer, state, params.operation1, params.operation2, params.backlog
                    )
                iteration_keys.append(state.key)
                if not accepted:
                    break
        except BudgetExhausted:
            exhausted = True
        return SearchOutcome(state.plan(), state.key, scorer.calls, tuple(iteration_keys), exhausted)
