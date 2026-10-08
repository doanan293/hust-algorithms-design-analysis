"""Validity check of search plans that leave the candidate set (spec E Section 8).

The LP bound (P2) of the runner covers plans whose paths are candidates. For every (set, scenario, method) in which a plan
routed outside the candidate set and beat that restricted bound on some realization, this script replans (the search is
deterministic), confirms the replanned plan reproduces the recorded timely counts, and solves (P2) on the plan's trajectory
with the candidate set extended by the plan's own paths. That bound covers the plan, so its timely count must stay below it.
Writes results/phase2/extended_bound_check.csv and exits non-zero on a violation or a mismatch.
"""

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
from typing import Mapping, Sequence

from bounds.formulation import build_bound_model
from bounds.solve import solve_lp
from models.channel import build_link_channels, channel_uniforms
from models.evaluate import REALIZED, EvaluationCounter, evaluate
from models.paths import CandidatePath, candidate_paths, radio_links
from models.plan import Plan
from models.scenario import Scenario, load_scenario
from runner.config import load_experiment_config
from runner.methods import TRAJECTORY, build_method

TOLERANCE = 1e-6
EXPERIMENTS = ("results/phase2/main", "results/phase2/sensitivity", "results/phase2/case_study")
COLUMNS = ("experiment", "set_id", "scenario_id", "method", "realization_id", "timely_count", "restricted_bound", "extended_bound", "status", "valid")


def extended_candidates(candidates: Mapping[str, tuple[CandidatePath, ...]], plan: Plan) -> dict[str, tuple[CandidatePath, ...]]:
    """Candidates of every alert, plus the plan's own path when it is not among them."""
    extended = {}
    for alert_id, paths in candidates.items():
        path = plan.paths.get(alert_id)
        extended[alert_id] = paths if path is None or path in paths else (*paths, path)
    return extended


def exceeding_groups(
    method_rows: Sequence[Mapping[str, str]], bound_rows: Sequence[Mapping[str, str]]
) -> dict[tuple[str, str, str], dict[int, tuple[int, float]]]:
    """(set, scenario, method) -> {realization: (timely count, restricted bound)} for outside-candidate rows above the bound."""
    bounds = {
        (row["set_id"], row["scenario_id"], row["trajectory_method"], int(row["realization_id"])): float(row["value"])
        for row in bound_rows
        if row["variant"] == TRAJECTORY
    }
    groups: dict[tuple[str, str, str], dict[int, tuple[int, float]]] = defaultdict(dict)
    for row in method_rows:
        if int(row.get("paths_outside_candidates", 0) or 0) == 0:
            continue
        realization = int(row["realization_id"])
        bound = bounds.get((row["set_id"], row["scenario_id"], row["method"], realization))
        if bound is not None and int(row["timely_count"]) > bound + TOLERANCE:
            groups[(row["set_id"], row["scenario_id"], row["method"])][realization] = (int(row["timely_count"]), bound)
    return dict(groups)


def extended_bound(
    scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]], plan: Plan, realization_id: int, tangents: int
) -> tuple[float, str]:
    extended = extended_candidates(candidates, plan)
    channels = build_link_channels(scenario, radio_links(extended), plan.trajectory, channel_uniforms(scenario, realization_id))
    result = solve_lp(build_bound_model(scenario, extended, channels, tangents))
    return result.value, result.status


def check_group(job: tuple[str, str, str, str, str, dict[int, tuple[int, float]]]) -> list[dict[str, object]]:
    experiment, config_path, set_id, scenario_id, method_id, realizations = job
    config = load_experiment_config(Path(config_path))
    scenario_dir = next(item.scenario_dir for item in config.scenario_sets if item.set_id == set_id)
    scenario = load_scenario(scenario_dir / f"{scenario_id}.json")
    candidates = candidate_paths(scenario)
    spec = next(item for item in config.methods if item.id == method_id)
    plan = build_method(spec).plan(scenario, candidates, EvaluationCounter())
    replayed = evaluate(scenario, plan, REALIZED, tuple(sorted(realizations)))
    rows = []
    for realization in replayed.realizations:
        timely = sum(realization.timely.values())
        recorded, restricted = realizations[realization.realization_id]
        value, status = extended_bound(scenario, candidates, plan, realization.realization_id, config.tangents)
        valid = timely == recorded and status == "optimal" and timely <= value + TOLERANCE
        rows.append({
            "experiment": experiment, "set_id": set_id, "scenario_id": scenario_id, "method": method_id,
            "realization_id": realization.realization_id, "timely_count": timely, "restricted_bound": restricted,
            "extended_bound": value, "status": status if timely == recorded else f"replay {timely} != recorded {recorded}",
            "valid": valid,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-extended-bounds")
    parser.add_argument("--experiments", nargs="+", type=Path, default=[Path(item) for item in EXPERIMENTS])
    parser.add_argument("--output", type=Path, default=Path("results/phase2/extended_bound_check.csv"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    jobs = []
    for directory in args.experiments:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        with (directory / "method_realizations.csv").open(newline="", encoding="utf-8") as stream:
            method_rows = list(csv.DictReader(stream))
        with (directory / "bounds.csv").open(newline="", encoding="utf-8") as stream:
            bound_rows = list(csv.DictReader(stream))
        for (set_id, scenario_id, method), realizations in sorted(exceeding_groups(method_rows, bound_rows).items()):
            jobs.append((directory.name, manifest["config_path"], set_id, scenario_id, method, realizations))
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in pool.map(check_group, jobs):
            rows.extend(result)
    rows.sort(key=lambda row: (row["experiment"], row["set_id"], row["scenario_id"], row["method"], row["realization_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    invalid = [row for row in rows if not row["valid"]]
    print(json.dumps({"groups": len(jobs), "rows": len(rows), "invalid": len(invalid)}))
    for row in invalid[:10]:
        print(row, file=sys.stderr)
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
