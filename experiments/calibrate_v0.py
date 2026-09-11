import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Callable, Mapping

import numpy as np

from models.evaluate import REALIZED, evaluate
from models.paths import GROUND, VIA_UAV, CandidatePath, candidate_paths
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import Scenario, load_scenario

LOWER_BOUND = 0.05
UPPER_BOUND = 0.95


def first_index(paths: tuple[CandidatePath, ...], kind: str) -> int | None:
    return next((index for index, path in enumerate(paths) if path.kind == kind), None)


def zone_tour_trajectory(scenario: Scenario) -> np.ndarray:
    num_slots = scenario.time.num_slots
    step_m = scenario.uav.v_max_mps * scenario.time.slot_s
    center = np.asarray(scenario.uav.start_xy_m, dtype=float)
    remaining = [np.array([zone.x_m, zone.y_m]) for zone in scenario.damage_zones]
    order: list[np.ndarray] = []
    current = center
    while remaining:
        index = min(range(len(remaining)), key=lambda item: (float(np.linalg.norm(remaining[item] - current)), item))
        current = remaining.pop(index)
        order.append(current)
    while True:
        stops = [center, *order, center]
        travel = [math.ceil(float(np.linalg.norm(stop - start)) / step_m) for start, stop in zip(stops, stops[1:])]
        if sum(travel) <= num_slots or not order:
            break
        order.pop()
    hover = (num_slots - sum(travel)) // len(order) if order else 0
    points = [center]
    for index, (start, stop, steps) in enumerate(zip(stops, stops[1:], travel)):
        points.extend(start + (stop - start) * step / steps for step in range(1, steps + 1))
        if index < len(order):
            points.extend([stop] * hover)
    points.extend([center] * (num_slots + 1 - len(points)))
    return np.asarray(points)


def ground_only_plan(scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]]) -> Plan:
    choice = {alert.id: first_index(candidates[alert.id], GROUND) for alert in scenario.alerts}
    return Plan(stationary_trajectory(scenario), choice, EqualSplitBacklogged())


def zone_tour_plan(scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]]) -> Plan:
    choice = {alert.id: first_index(candidates[alert.id], VIA_UAV) for alert in scenario.alerts}
    return Plan(zone_tour_trajectory(scenario), choice, EqualSplitBacklogged())


PLANS: dict[str, Callable[[Scenario, Mapping[str, tuple[CandidatePath, ...]]], Plan]] = {
    "ground_only": ground_only_plan,
    "zone_tour": zone_tour_plan,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="calibrate-v0")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/scenarios_v0.csv"))
    parser.add_argument("--scenario-dir", type=Path, default=Path("data/processed/scenarios/v0"))
    parser.add_argument("--realizations", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("results/calibration/v0.json"))
    args = parser.parse_args(argv)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        scenario_ids = sorted(row["scenario_id"] for row in csv.DictReader(stream) if row["split"] == "eval")
    realization_ids = tuple(range(args.realizations))
    ratios: dict[str, dict[str, float]] = {name: {} for name in PLANS}
    runtimes: dict[str, list[float]] = {name: [] for name in PLANS}
    feasible = {name: True for name in PLANS}
    for scenario_id in scenario_ids:
        scenario = load_scenario(args.scenario_dir / f"{scenario_id}.json")
        candidates = candidate_paths(scenario)
        for name, build in PLANS.items():
            result = evaluate(scenario, build(scenario, candidates), REALIZED, realization_ids, candidates=candidates)
            feasible[name] = feasible[name] and result.feasible
            ratios[name][scenario_id] = result.timely_ratio
            runtimes[name].append(result.runtime_s)
    plans = {
        name: {
            "feasible": feasible[name],
            "mean_timely_ratio": statistics.fmean(ratios[name].values()),
            "per_scenario_timely_ratio": ratios[name],
            "mean_runtime_s_per_evaluation": statistics.fmean(runtimes[name]),
        }
        for name in PLANS
    }
    passed = all(
        summary["feasible"] and LOWER_BOUND <= summary["mean_timely_ratio"] <= UPPER_BOUND
        for summary in plans.values()
    )
    report = {
        "scenario_count": len(scenario_ids),
        "realization_ids": list(realization_ids),
        "pass_bounds": [LOWER_BOUND, UPPER_BOUND],
        "plans": plans,
        "zone_tour_beats_ground_only": plans["zone_tour"]["mean_timely_ratio"] > plans["ground_only"]["mean_timely_ratio"],
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: round(summary["mean_timely_ratio"], 4) for name, summary in plans.items()} | {"passed": passed}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
