"""TRAN development-set pilot logic (spec D Section 7.1): one pilot job, block and parameter rules, Samir rule."""

from collections import defaultdict
from pathlib import Path
import statistics
import time

from models.evaluate import REALIZED, evaluate
from models.paths import candidate_paths
from models.scenario import load_scenario

from .tran import TranHD
from .tran_model import TranParams

RUNTIME_LIMIT_S = 600.0
FALLBACK_BLOCK_SLOTS = 5
DEFAULT_THETA, DEFAULT_MU = 0.5, 1.0
THETAS = (1.0 / 3.0, 0.5, 2.0 / 3.0)
MUS = (1.0, 10.0)
TIE_TOLERANCE = 1e-12
RUN_COLUMNS = (
    "step", "scenario_id", "theta", "mu", "block_slots", "status", "iterations", "relaxed_objective", "penalty",
    "planning_runtime_s", "initial_timely_ratio", "timely_ratio",
)
FIT_COLUMNS = ("scenario_id", "beta", "alpha", "r_squared", "max_log_error")


def run_job(job: tuple[str, str, float, float, int, int]) -> dict[str, object]:
    step, scenario_path, theta, mu, block_slots, realizations = job
    scenario = load_scenario(Path(scenario_path))
    candidates = candidate_paths(scenario)
    started = time.perf_counter()
    outcome = TranHD(params=TranParams(theta=theta, mu=mu, block_slots=block_slots)).solve(scenario, candidates)
    planning_s = time.perf_counter() - started
    ids = tuple(range(realizations))
    initial = evaluate(scenario, outcome.initial_plan, REALIZED, ids, candidates=candidates)
    final = evaluate(scenario, outcome.plan, REALIZED, ids, candidates=candidates)
    if not (initial.feasible and final.feasible):
        raise ValueError(f"{scenario.scenario_id}: infeasible TRAN plan")
    return {
        "step": step, "scenario_id": scenario.scenario_id, "theta": theta, "mu": mu, "block_slots": block_slots,
        "status": outcome.status, "iterations": outcome.iterations, "relaxed_objective": outcome.objectives[-1],
        "penalty": outcome.penalty, "planning_runtime_s": planning_s, "initial_timely_ratio": initial.timely_ratio,
        "timely_ratio": final.timely_ratio, "beta": outcome.channel.beta, "alpha": outcome.channel.alpha,
        "r_squared": outcome.channel.r_squared, "max_log_error": outcome.channel.max_log_error,
    }


def choose_block_slots(rows: list[dict[str, object]]) -> int:
    median = statistics.median(float(row["planning_runtime_s"]) for row in rows)
    return FALLBACK_BLOCK_SLOTS if median > RUNTIME_LIMIT_S else 1


def choose_parameters(rows: list[dict[str, object]]) -> tuple[float, float]:
    """Highest mean timely ratio over scenarios; ties go to theta nearest 1/2, then the smaller mu."""
    groups: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in rows:
        groups[(float(row["theta"]), float(row["mu"]))].append(float(row["timely_ratio"]))
    means = {key: statistics.fmean(values) for key, values in groups.items()}
    best = max(means.values())
    return min((key for key, value in means.items() if value >= best - TIE_TOLERANCE), key=lambda key: (abs(key[0] - DEFAULT_THETA), key[1]))


def samir_fallback_fires(rows: list[dict[str, object]]) -> bool:
    """Section 5.8: the chosen TRAN is not better than its initial point on at least half of the scenarios."""
    not_better = sum(float(row["timely_ratio"]) <= float(row["initial_timely_ratio"]) for row in rows)
    return 2 * not_better >= len(rows)


def literature_config(choice: dict[str, object]) -> dict[str, object]:
    params = {"theta": choice["theta"], "mu": choice["mu"], "block_slots": choice["block_slots"]}
    return {
        "experiment_id": "phase2-literature",
        "scenario_sets": [
            {"set_id": "v0", "manifest": "data/manifests/scenarios_v0.csv", "scenario_dir": "data/processed/scenarios/v0"},
            {"set_id": "v0-bh50", "manifest": "data/manifests/scenarios_v0-bh50.csv", "scenario_dir": "data/processed/scenarios/v0-bh50"},
        ],
        "split": "eval",
        "realization_ids": {"start": 0, "stop": 30},
        "methods": ["B1", {"id": "TRAN", "base": "TRAN", "params": params, "bounds": True}],
        "bounds": {"tangents": 10},
        "bootstrap": {"resamples": 10000, "seed": 20260911, "confidence": 0.95},
        "workers": 12,
        "output_dir": "results/phase2/literature",
    }
