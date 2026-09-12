"""Planning methods and the bound rows that check them, built from experiment method specs."""

from baselines import get_method
from baselines.method import Method
from literature.tran import TranHD
from optimization.bcd import SearchMethod

from .config import PLANNED_TRAJECTORY_BASES, SEARCH_BASE, TRAN_BASE, MethodSpec

GROUND_ONLY = "ground_only"
TRAJECTORY = "trajectory"


def build_method(spec: MethodSpec) -> Method:
    if spec.base == SEARCH_BASE:
        return SearchMethod(spec.id, spec.search_params())
    if spec.base == TRAN_BASE:
        return TranHD(spec.id, spec.tran_params())
    return get_method(spec.base)


def bound_owner(spec: MethodSpec) -> tuple[str, str] | None:
    """(variant, trajectory_method) of the bound rows for the method; None for a planned-trajectory method without bounds."""
    if spec.base in PLANNED_TRAJECTORY_BASES:
        return (TRAJECTORY, spec.id) if spec.bounds else None
    return (TRAJECTORY, spec.base) if get_method(spec.base).uses_uav else (GROUND_ONLY, "")
