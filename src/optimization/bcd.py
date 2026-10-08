"""SearchMethod: B2, B3, P and the ablation variants as parameter sets of one evaluator-guided BCD."""

from dataclasses import dataclass, field, fields
from typing import Mapping

import numpy as np

from baselines.b1 import B1
from models.evaluate import EvaluationCounter
from models.paths import CandidatePath, backhaul_routes
from models.plan import Plan
from models.rng import named_rng
from models.scenario import Scenario

from .blocks import SearchState, bandwidth_block, catalog_trajectory_block, path_block, trajectory_block, unit_weights
from .objectives import OBJECTIVES, Key
from .repair import repair_round
from .scoring import DEFAULT_BUDGET, DEFAULT_DESIGN_IDS, BudgetExhausted, DesignScorer
from .tours import tour_family

EXPECTED_DESIGN = "expected"
LOCAL = "local"
CATALOG = "catalog"
TRAJECTORY_MODES = (LOCAL, CATALOG)


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
    trajectory_mode: str = LOCAL
    catalog_init: bool = False
    catalog_probability: float = 0.1
    entry_neighbors: int = 3
    trajectory_segments: int = 5
    max_widening: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective: expected one of {sorted(OBJECTIVES)}, got {self.objective!r}")
        for name in ("path_block", "bandwidth_block", "trajectory_block", "operation1", "operation2", "backlog", "catalog_init"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name}: expected true or false")
        if self.trajectory_mode not in TRAJECTORY_MODES:
            raise ValueError(f"trajectory_mode: expected one of {TRAJECTORY_MODES}, got {self.trajectory_mode!r}")
        probability = self.catalog_probability
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0.0 <= probability <= 1.0:
            raise ValueError("catalog_probability: expected a number in [0, 1]")
        for name in ("budget", "max_iterations", "entry_neighbors", "trajectory_segments"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}: expected an integer of at least 1")
        for name in ("max_widening", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name}: expected a non-negative integer")
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
    """Starts from B1 with unit weights; iterates the path, bandwidth, trajectory blocks and repair until an iteration accepts nothing."""

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
        scorer = DesignScorer(scenario, params.design_ids, params.budget, OBJECTIVES[params.objective])
        rng = named_rng(scenario.scenario_seed, f"search:{params.seed}")
        routes = backhaul_routes(scenario)
        initial = B1().plan(scenario, candidates, EvaluationCounter())
        state = SearchState(initial.trajectory, dict(initial.paths), unit_weights(scenario, candidates))
        family = tour_family(scenario) if params.trajectory_block else []
        iteration_keys: list[Key] = []
        exhausted = False
        try:
            state.key = scorer.score(state.plan())
            if params.trajectory_block and params.catalog_init and params.trajectory_mode == LOCAL:
                catalog_trajectory_block(scorer, state, family)
            for _ in range(params.max_iterations):
                accepted = False
                if params.path_block:
                    accepted |= path_block(scenario, candidates, routes, scorer, state, params, rng)
                if params.bandwidth_block:
                    accepted |= bandwidth_block(scenario, scorer, state)
                if params.trajectory_block and params.trajectory_mode == CATALOG:
                    accepted |= catalog_trajectory_block(scorer, state, family)
                elif params.trajectory_block:
                    accepted |= trajectory_block(scenario, scorer, state, params, rng, family)
                if params.operation1 or params.operation2:
                    accepted |= repair_round(scenario, candidates, routes, scorer, state, params, rng)
                iteration_keys.append(state.key)
                if not accepted:
                    break
        except BudgetExhausted:
            exhausted = True
        return SearchOutcome(state.plan(), state.key, scorer.calls, tuple(iteration_keys), exhausted)
