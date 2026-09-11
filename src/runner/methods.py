"""Planning methods and the bound rows that check them, built from experiment method specs."""

from baselines import get_method
from baselines.method import Method
from optimization.bcd import SearchMethod

from .config import SEARCH_BASE, MethodSpec

GROUND_ONLY = "ground_only"
TRAJECTORY = "trajectory"


def build_method(spec: MethodSpec) -> Method:
    if spec.base == SEARCH_BASE:
        return SearchMethod(spec.id, spec.search_params())
    return get_method(spec.base)


def bound_owner(spec: MethodSpec) -> tuple[str, str] | None:
    """(variant, trajectory_method) of the bound rows for the method; None for a search method without bounds."""
    if spec.base == SEARCH_BASE:
        return (TRAJECTORY, spec.id) if spec.bounds else None
    return (TRAJECTORY, spec.base) if get_method(spec.base).uses_uav else (GROUND_ONLY, "")
