from collections import defaultdict
import math
from typing import Mapping, Sequence

import numpy as np

from .config import BootstrapConfig
from .tasks import bound_variant

VALIDITY_TOLERANCE = 1e-6


def cluster_bootstrap_ci(
    values_by_cluster: Mapping[str, Sequence[float]], resamples: int, seed: int, confidence: float
) -> tuple[float, float]:
    clusters = sorted(values_by_cluster)
    arrays = [np.asarray(values_by_cluster[cluster], dtype=float) for cluster in clusters]
    rng = np.random.default_rng(seed)
    statistics = np.empty(resamples)
    for index in range(resamples):
        picks = rng.integers(0, len(clusters), size=len(clusters))
        statistics[index] = np.concatenate([arrays[pick] for pick in picks]).mean()
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(statistics, tail)), float(np.quantile(statistics, 1.0 - tail))


def relative_gap(bound_mean: float, method_mean: float) -> float | None:
    return None if bound_mean <= 0.0 else (bound_mean - method_mean) / bound_mean


def _bound_lookup(bound_rows: Sequence[Mapping[str, object]]) -> dict[tuple, float]:
    return {
        (row["set_id"], row["scenario_id"], row["variant"], row["trajectory_method"], int(row["realization_id"])): float(row["value"])
        for row in bound_rows
    }


def validity_violations(
    method_rows: Sequence[Mapping[str, object]], bound_rows: Sequence[Mapping[str, object]]
) -> list[str]:
    bounds = _bound_lookup(bound_rows)
    violations = []
    for row in bound_rows:
        if row["status"] != "optimal":
            violations.append(f"bound {row['set_id']}/{row['scenario_id']}/{row['variant']}/r{row['realization_id']}: {row['status']}")
    for row in method_rows:
        variant, owner = bound_variant(str(row["method"]))
        key = (row["set_id"], row["scenario_id"], variant, owner, int(row["realization_id"]))
        label = f"{row['set_id']}/{row['scenario_id']}/{row['method']}/r{row['realization_id']}"
        if key not in bounds:
            violations.append(f"{label}: missing {variant} bound")
        elif int(row["timely_count"]) > bounds[key] + VALIDITY_TOLERANCE:
            violations.append(f"{label}: {row['timely_count']} timely alerts exceed the bound {bounds[key]:.6f}")
    return violations


def crosscheck_violations(rows: Sequence[Mapping[str, object]]) -> list[str]:
    violations = []
    for row in rows:
        label = f"milp {row['set_id']}/{row['scenario_id']}"
        keys = ("lp_ub", "milp_incumbent", "milp_dual_bound", "milp_plan_value", "method_value")
        values = [float(row[key]) for key in keys]
        if row["lp_status"] != "optimal" or any(math.isnan(value) for value in values):
            violations.append(f"{label}: incomplete cross-check ({row['lp_status']}, {row['milp_status']})")
            continue
        lp_ub, incumbent, dual, plan_value, method_value = values
        ordered = max(plan_value, method_value, incumbent) <= dual + VALIDITY_TOLERANCE and dual <= lp_ub + VALIDITY_TOLERANCE
        if not ordered:
            violations.append(
                f"{label}: chain broken (plan {plan_value}, method {method_value}, incumbent {incumbent}, dual {dual}, lp {lp_ub})"
            )
    return violations


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def summarize(
    method_rows: Sequence[Mapping[str, object]],
    bound_rows: Sequence[Mapping[str, object]],
    bootstrap: BootstrapConfig,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, list[Mapping[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in method_rows:
        grouped[(str(row["set_id"]), str(row["method"]))][str(row["scenario_id"])].append(row)
    bounds: dict[tuple, list[Mapping[str, object]]] = defaultdict(list)
    for row in bound_rows:
        bounds[(row["set_id"], row["scenario_id"], row["variant"], row["trajectory_method"])].append(row)
    summary = []
    for (set_id, method), scenarios in sorted(grouped.items()):
        variant, owner = bound_variant(method)
        clusters: dict[str, list[float]] = defaultdict(list)
        timely, conn, shortfall, planning, evaluation, bound_means, bound_runtime, gaps = [], [], [], [], [], [], [], []
        for scenario_id in sorted(scenarios):
            rows = scenarios[scenario_id]
            scenario_timely = _mean([float(row["timely_ratio"]) for row in rows])
            timely.append(scenario_timely)
            clusters[str(rows[0]["topology_id"])].append(scenario_timely)
            conn.append(_mean([float(row["conn_ratio"]) for row in rows]))
            shortfall.append(_mean([float(row["shortfall"]) for row in rows]))
            planning.append(_mean([float(row["planning_runtime_s"]) for row in rows]))
            evaluation.append(_mean([float(row["evaluation_runtime_s"]) for row in rows]))
            scenario_bounds = bounds[(set_id, scenario_id, variant, owner)]
            bound_mean = _mean([float(row["ratio"]) for row in scenario_bounds])
            bound_means.append(bound_mean)
            bound_runtime.append(_mean([float(row["runtime_s"]) for row in scenario_bounds]))
            gaps.append(relative_gap(bound_mean, scenario_timely))
        low, high = cluster_bootstrap_ci(clusters, bootstrap.resamples, bootstrap.seed, bootstrap.confidence)
        defined = [value for value in gaps if value is not None]
        summary.append(
            {
                "set_id": set_id,
                "method": method,
                "scenario_count": len(scenarios),
                "timely_ratio_mean": _mean(timely),
                "timely_ratio_ci_low": low,
                "timely_ratio_ci_high": high,
                "conn_ratio_mean": _mean(conn),
                "bound_ratio_mean": _mean(bound_means),
                "gap_mean": _mean(defined) if defined else math.nan,
                "gap_undefined_count": len(gaps) - len(defined),
                "shortfall_mean": _mean(shortfall),
                "planning_runtime_s_mean": _mean(planning),
                "evaluation_runtime_s_per_realization_mean": _mean(evaluation),
                "bound_runtime_s_mean": _mean(bound_runtime),
            }
        )
    return summary
