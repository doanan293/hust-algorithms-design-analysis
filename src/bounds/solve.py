from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .formulation import BoundModel

OPTIMAL = "optimal"
TIME_LIMIT = "time_limit"


@dataclass(frozen=True)
class BoundResult:
    value: float
    dual_bound: float
    status: str
    runtime_s: float
    variables: int
    rows: int
    nonzeros: int
    solution: np.ndarray | None

    @property
    def optimal(self) -> bool:
        return self.status == OPTIMAL


def solve_lp(model: BoundModel) -> BoundResult:
    """Solve with HiGHS defaults; if the result is not optimal, solve again without presolve.

    Presolve can leave badly scaled models (coefficients spanning about 1e-13 to 1e6) with model status Unknown,
    which HiGHS then solves to optimality without presolve.
    """
    started = time.perf_counter()
    arguments = dict(
        A_ub=model.a_ub,
        b_ub=model.b_ub,
        A_eq=model.a_eq if model.a_eq.shape[0] else None,
        b_eq=model.b_eq if model.a_eq.shape[0] else None,
        bounds=np.column_stack([np.zeros(model.variables), model.upper]),
        method="highs",
    )
    result = linprog(model.objective, **arguments)
    if result.status != 0:
        result = linprog(model.objective, **arguments, options={"presolve": False})
    runtime = time.perf_counter() - started
    solved = result.status == 0
    value = -float(result.fun) if solved else math.nan
    return BoundResult(
        value=value,
        dual_bound=value,
        status=OPTIMAL if solved else f"status {result.status}: {result.message}",
        runtime_s=runtime,
        variables=model.variables,
        rows=model.rows,
        nonzeros=model.nonzeros,
        solution=np.asarray(result.x) if solved else None,
    )


def solve_milp(model: BoundModel, time_limit_s: float) -> BoundResult:
    constraints = [LinearConstraint(model.a_ub, -np.inf, model.b_ub)]
    if model.a_eq.shape[0]:
        constraints.append(LinearConstraint(model.a_eq, model.b_eq, model.b_eq))
    started = time.perf_counter()
    result = milp(
        model.objective,
        constraints=constraints,
        integrality=model.integrality,
        bounds=Bounds(np.zeros(model.variables), model.upper),
        options={"time_limit": time_limit_s},
    )
    runtime = time.perf_counter() - started
    value = -float(result.fun) if result.x is not None else math.nan
    dual = getattr(result, "mip_dual_bound", None)
    if result.status == 0:
        status = OPTIMAL
    elif result.status == 1:
        status = TIME_LIMIT
    else:
        status = f"status {result.status}: {result.message}"
    return BoundResult(
        value=value,
        dual_bound=-float(dual) if dual is not None else value,
        status=status,
        runtime_s=runtime,
        variables=model.variables,
        rows=model.rows,
        nonzeros=model.nonzeros,
        solution=np.asarray(result.x) if result.x is not None else None,
    )


def highs_version() -> str:
    from scipy.optimize._highspy import _core

    return f"{_core.HIGHS_VERSION_MAJOR}.{_core.HIGHS_VERSION_MINOR}.{_core.HIGHS_VERSION_PATCH}"
