"""Trace of the representative case-study scenario (spec D Section 10.2): map and delivery-progress data.

The representative scenario has the anchor method's mean timely ratio closest to the median over scenarios (ties:
smallest scenario id). Each traced method is planned again and evaluated on the experiment's realizations; the
script stops without writing when a timely ratio differs from the runner's results.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Mapping, Sequence

from models.evaluate import REALIZED, EvaluationCounter, evaluate
from models.paths import candidate_paths
from models.scenario import load_scenario
from runner.config import load_experiment_config
from runner.methods import build_method

RATIO_TOLERANCE = 1e-12
RULE = "anchor mean timely ratio closest to the median over scenarios; ties go to the smallest scenario_id"


def scenario_means(rows: Sequence[Mapping[str, str]], method: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            values[row["scenario_id"]].append(float(row["timely_ratio"]))
    return {scenario_id: statistics.fmean(items) for scenario_id, items in sorted(values.items())}


def representative_scenario(rows: Sequence[Mapping[str, str]], method: str) -> str:
    means = scenario_means(rows, method)
    if not means:
        raise ValueError(f"no result rows for method {method}")
    median = statistics.median(means.values())
    return min(means, key=lambda scenario_id: (abs(means[scenario_id] - median), scenario_id))


def progress_counts(realizations: Sequence[object], num_slots: int) -> list[float]:
    """Mean over realizations of the alerts delivered on time by the end of each slot."""
    totals = [0.0] * num_slots
    for realization in realizations:
        for alert_id, slot in realization.delivery_slot.items():
            if slot is not None and realization.timely[alert_id]:
                for index in range(slot, num_slots):
                    totals[index] += 1.0
    return [total / len(realizations) for total in totals]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="case-study-trace")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/phase2_case_study.yaml"))
    parser.add_argument("--results", type=Path, default=Path("results/phase2/case_study"))
    parser.add_argument("--targets", type=Path, default=Path("data/manifests/rescuenet_targets.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2/case_study/trace"))
    parser.add_argument("--methods", nargs="+", default=["B1", "P", "TRAN"])
    parser.add_argument("--anchor", default="P")
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    with (args.results / "method_realizations.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    scenario_id = representative_scenario(rows, args.anchor)
    scenario = load_scenario(config.scenario_sets[0].scenario_dir / f"{scenario_id}.json")
    with args.targets.open(newline="", encoding="utf-8") as stream:
        targets = list(csv.DictReader(stream))
    if len(targets) != len(scenario.sources):
        raise ValueError(f"{args.targets}: {len(targets)} targets for {len(scenario.sources)} sources of {scenario_id}")
    candidates = candidate_paths(scenario)
    specs = {spec.id: spec for spec in config.methods}
    scenario_rows = [row for row in rows if row["scenario_id"] == scenario_id]
    checks, trajectories, progress = {}, [], []
    for method_id in args.methods:
        plan = build_method(specs[method_id]).plan(scenario, candidates, EvaluationCounter())
        result = evaluate(scenario, plan, REALIZED, config.realization_ids, candidates=candidates)
        checks[method_id] = {"timely_ratio_trace": result.timely_ratio, "timely_ratio_runner": scenario_means(scenario_rows, method_id)[scenario_id]}
        trajectories += [{"method": method_id, "slot": slot, "x_m": float(x), "y_m": float(y)} for slot, (x, y) in enumerate(plan.trajectory)]
        counts = progress_counts(result.realizations, scenario.time.num_slots)
        progress += [{"method": method_id, "slot": slot, "mean_timely_alerts": value} for slot, value in enumerate(counts)]
    differing = sorted(method for method, check in checks.items() if abs(check["timely_ratio_trace"] - check["timely_ratio_runner"]) > RATIO_TOLERANCE)
    if differing:
        print(f"trace differs from the runner for {differing}: {checks}", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    zone_of = {source.id: source.zone for source in scenario.sources}
    _write(args.output / "targets.csv", [
        {"source_id": source.id, "target_id": target["target_id"], "x_m": source.x_m, "y_m": source.y_m,
         "damage_class": target["damage_class"], "zone": zone_of[source.id]}
        for source, target in zip(scenario.sources, targets)
    ])
    _write(args.output / "nodes.csv", [
        {"node_id": node.id, "x_m": node.x_m, "y_m": node.y_m, "role": "center" if node.id == scenario.center else "relay"}
        for node in scenario.nodes
    ])
    _write(args.output / "edges.csv", [
        {"source": edge.source, "target": edge.target, "source_x_m": scenario.node_by_id[edge.source].x_m,
         "source_y_m": scenario.node_by_id[edge.source].y_m, "target_x_m": scenario.node_by_id[edge.target].x_m,
         "target_y_m": scenario.node_by_id[edge.target].y_m, "failed": int(edge.failed)}
        for edge in scenario.edges
    ])
    _write(args.output / "trajectories.csv", trajectories)
    _write(args.output / "progress.csv", progress)
    summary = {"scenario_id": scenario_id, "anchor": args.anchor, "rule": RULE, "realizations": len(config.realization_ids), "methods": checks}
    (args.output / "trace.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"scenario_id": scenario_id, "methods": sorted(checks), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
