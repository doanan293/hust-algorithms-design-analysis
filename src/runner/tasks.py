from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from typing import Mapping

import numpy as np

from baselines import get_method
from bounds.crosscheck import model_snr_values, plan_from_solution, reduce_scenario, tangent_max_overestimate
from bounds.formulation import build_bound_model, restrict_to_ground
from bounds.solve import solve_lp, solve_milp
from models.channel import build_link_channels, channel_uniforms
from models.evaluate import REALIZED, EvaluationCounter, evaluate
from models.paths import CandidatePath, candidate_paths, radio_links
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import Scenario, load_scenario

from .config import MethodSpec
from .methods import GROUND_ONLY, TRAJECTORY, build_method

CROSSCHECK_TRAJECTORY_METHOD = "B1"


@dataclass(frozen=True)
class MethodTask:
    set_id: str
    scenario_id: str
    scenario_path: str
    spec: MethodSpec
    realization_ids: tuple[int, ...]
    tangents: int

    @property
    def key(self) -> str:
        return f"method__{self.set_id}__{self.scenario_id}__{self.spec.id}"


@dataclass(frozen=True)
class BoundTask:
    set_id: str
    scenario_id: str
    scenario_path: str
    variant: str
    trajectory_method: str
    realization_id: int
    tangents: int

    @property
    def key(self) -> str:
        owner = self.trajectory_method or "none"
        return f"bound__{self.set_id}__{self.scenario_id}__{self.variant}__{owner}__r{self.realization_id:03d}"


@dataclass(frozen=True)
class CrosscheckTask:
    set_id: str
    scenario_id: str
    scenario_path: str
    trajectory_method: str
    alert_count: int
    realization_id: int
    time_limit_s: float
    tangents: int

    @property
    def key(self) -> str:
        return f"milp__{self.set_id}__{self.scenario_id}"


Task = MethodTask | BoundTask | CrosscheckTask


def task_hash(task: Task, scenario_sha256: str, source_hash: str) -> str:
    payload = {
        "type": type(task).__name__,
        "task": asdict(task),
        "scenario_sha256": scenario_sha256,
        "source_hash": source_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_method_task(task: MethodTask) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Method rows, plus trajectory bound rows for the planned trajectory when the spec asks for bounds."""
    scenario = load_scenario(Path(task.scenario_path))
    candidates = candidate_paths(scenario)
    method = build_method(task.spec)
    counter = EvaluationCounter()
    started = time.perf_counter()
    plan = method.plan(scenario, candidates, counter)
    planning_s = time.perf_counter() - started
    result = evaluate(scenario, plan, REALIZED, task.realization_ids, counter=counter, candidates=candidates)
    if not result.feasible:
        raise ValueError(f"{task.key}: infeasible plan: {list(result.violations[:3])}")
    sources = len(scenario.sources)
    without_uav = round(result.connectivity_ratio_without_uav * sources)
    rows = []
    for realization in result.realizations:
        timely = sum(realization.timely.values())
        connected = sum(realization.connected.values()) if method.uses_uav else without_uav
        rows.append(
            {
                "set_id": task.set_id,
                "scenario_id": task.scenario_id,
                "topology_id": scenario.network_id,
                "method": task.spec.id,
                "realization_id": realization.realization_id,
                "alert_count": len(scenario.alerts),
                "timely_count": timely,
                "timely_ratio": timely / len(scenario.alerts),
                "source_count": sources,
                "connected_count": connected,
                "conn_ratio": connected / sources,
                "shortfall": realization.shortfall,
                "planning_runtime_s": planning_s,
                "evaluation_runtime_s": result.runtime_s / len(task.realization_ids),
                "evaluate_calls": counter.calls,
            }
        )
    bound_rows = []
    if task.spec.bounds:
        bound_rows = [
            bound_row(
                task.set_id, task.scenario_id, scenario, candidates, TRAJECTORY, task.spec.id, plan.trajectory,
                realization_id, task.tangents,
            )
            for realization_id in task.realization_ids
        ]
    return rows, bound_rows


def bound_row(
    set_id: str,
    scenario_id: str,
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    variant: str,
    trajectory_method: str,
    trajectory: np.ndarray,
    realization_id: int,
    tangents: int,
) -> dict[str, object]:
    """Solve the per-realization LP bound for a fixed trajectory on the evaluation channel draws."""
    channels = build_link_channels(scenario, radio_links(candidates), trajectory, channel_uniforms(scenario, realization_id))
    result = solve_lp(build_bound_model(scenario, candidates, channels, tangents))
    return {
        "set_id": set_id,
        "scenario_id": scenario_id,
        "topology_id": scenario.network_id,
        "variant": variant,
        "trajectory_method": trajectory_method,
        "realization_id": realization_id,
        "value": result.value,
        "ratio": result.value / len(scenario.alerts),
        "status": result.status,
        "runtime_s": result.runtime_s,
        "variables": result.variables,
        "rows": result.rows,
        "nonzeros": result.nonzeros,
    }


def run_bound_task(task: BoundTask) -> list[dict[str, object]]:
    scenario = load_scenario(Path(task.scenario_path))
    candidates = candidate_paths(scenario)
    if task.variant == GROUND_ONLY:
        candidates = restrict_to_ground(candidates)
        trajectory = stationary_trajectory(scenario)
    else:
        trajectory = get_method(task.trajectory_method).trajectory(scenario)
    return [
        bound_row(
            task.set_id, task.scenario_id, scenario, candidates, task.variant, task.trajectory_method, trajectory,
            task.realization_id, task.tangents,
        )
    ]


def run_crosscheck_task(task: CrosscheckTask) -> list[dict[str, object]]:
    scenario = reduce_scenario(load_scenario(Path(task.scenario_path)), task.alert_count)
    candidates = candidate_paths(scenario)
    trajectory = get_method(task.trajectory_method).trajectory(scenario)
    channels = build_link_channels(
        scenario, radio_links(candidates), trajectory, channel_uniforms(scenario, task.realization_id)
    )
    model = build_bound_model(scenario, candidates, channels, task.tangents)
    lp = solve_lp(model)
    milp = solve_milp(model, task.time_limit_s)

    def timely_count(plan: Plan) -> float:
        evaluated = evaluate(scenario, plan, REALIZED, (task.realization_id,), candidates=candidates)
        return float(sum(evaluated.realizations[0].timely.values())) if evaluated.feasible else math.nan

    static_value = equal_split_value = math.nan
    if milp.solution is not None:
        static_plan = plan_from_solution(scenario, candidates, model, milp.solution, trajectory)
        static_value = timely_count(static_plan)
        equal_split_value = timely_count(Plan(trajectory, static_plan.path_choice, EqualSplitBacklogged()))
    method = get_method(task.trajectory_method)
    method_value = timely_count(method.plan(scenario, candidates, EvaluationCounter()))
    return [
        {
            "set_id": task.set_id,
            "scenario_id": task.scenario_id,
            "alert_count": len(scenario.alerts),
            "lp_ub": lp.value,
            "lp_status": lp.status,
            "milp_incumbent": milp.value,
            "milp_dual_bound": milp.dual_bound,
            "milp_status": milp.status,
            "milp_runtime_s": milp.runtime_s,
            "milp_plan_static_value": static_value,
            "milp_plan_equal_split_value": equal_split_value,
            "milp_plan_value": max(static_value, equal_split_value),
            "method_value": method_value,
            "tangent_max_overestimate": tangent_max_overestimate(
                model_snr_values(model, channels), scenario.spectrum.b_tot_hz, task.tangents
            ),
        }
    ]


RUNNERS = {BoundTask: run_bound_task, CrosscheckTask: run_crosscheck_task}


def execute(task: Task) -> dict[str, object]:
    started = time.perf_counter()
    bound_rows: list[dict[str, object]] = []
    try:
        if isinstance(task, MethodTask):
            rows, bound_rows = run_method_task(task)
        else:
            rows = RUNNERS[type(task)](task)
        error = None
    except Exception as exc:  # recorded in the shard; the experiment reports it and exits non-zero
        rows, bound_rows, error = [], [], "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return {"key": task.key, "rows": rows, "bound_rows": bound_rows, "error": error, "runtime_s": time.perf_counter() - started}
