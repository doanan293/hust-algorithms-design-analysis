from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import sys
from typing import Sequence

from bounds.crosscheck import flow_variable_count, select_crosscheck_scenarios
from models.paths import candidate_paths
from models.scenario import load_scenario

from .aggregate import crosscheck_violations, summarize, validity_violations
from .config import ExperimentConfig
from .provenance import environment, file_sha256, git_state, source_hash
from .tasks import (
    CROSSCHECK_TRAJECTORY_METHOD,
    BoundTask,
    CrosscheckTask,
    MethodTask,
    Task,
    bound_variant,
    execute,
    task_hash,
)

METHOD_COLUMNS = (
    "set_id", "scenario_id", "topology_id", "method", "realization_id", "alert_count", "timely_count",
    "timely_ratio", "source_count", "connected_count", "conn_ratio", "shortfall", "planning_runtime_s",
    "evaluation_runtime_s", "evaluate_calls",
)
BOUND_COLUMNS = (
    "set_id", "scenario_id", "topology_id", "variant", "trajectory_method", "realization_id", "value", "ratio",
    "status", "runtime_s", "variables", "rows", "nonzeros",
)
CROSSCHECK_COLUMNS = (
    "set_id", "scenario_id", "alert_count", "lp_ub", "lp_status", "milp_incumbent", "milp_dual_bound",
    "milp_status", "milp_runtime_s", "milp_plan_static_value", "milp_plan_equal_split_value", "milp_plan_value",
    "method_value", "tangent_max_overestimate",
)
SUMMARY_COLUMNS = (
    "set_id", "method", "scenario_count", "timely_ratio_mean", "timely_ratio_ci_low", "timely_ratio_ci_high",
    "conn_ratio_mean", "bound_ratio_mean", "gap_mean", "gap_undefined_count", "shortfall_mean",
    "planning_runtime_s_mean", "evaluation_runtime_s_per_realization_mean", "bound_runtime_s_mean",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ScenarioRef:
    set_id: str
    scenario_id: str
    path: Path
    sha256: str


def scenario_refs(config: ExperimentConfig) -> list[ScenarioRef]:
    refs = []
    for scenario_set in config.scenario_sets:
        if not scenario_set.manifest.exists():
            raise FileNotFoundError(f"scenario manifest not found: {scenario_set.manifest}")
        with scenario_set.manifest.open(newline="", encoding="utf-8") as stream:
            rows = sorted(
                (row for row in csv.DictReader(stream) if row["split"] == config.split),
                key=lambda row: row["scenario_id"],
            )
        for row in rows:
            path = scenario_set.scenario_dir / f"{row['scenario_id']}.json"
            if not path.exists():
                raise FileNotFoundError(f"scenario file not found: {path}")
            refs.append(ScenarioRef(scenario_set.set_id, row["scenario_id"], path, row["sha256"]))
    return refs


def build_tasks(config: ExperimentConfig, refs: Sequence[ScenarioRef]) -> list[Task]:
    tasks: list[Task] = []
    variants = sorted({bound_variant(method) for method in config.methods})
    for ref in refs:
        for method in config.methods:
            tasks.append(MethodTask(ref.set_id, ref.scenario_id, str(ref.path), method, config.realization_ids))
        for variant, owner in variants:
            for realization_id in config.realization_ids:
                tasks.append(
                    BoundTask(ref.set_id, ref.scenario_id, str(ref.path), variant, owner, realization_id, config.tangents)
                )
    cross = config.crosscheck
    candidates_refs = [ref for ref in refs if ref.set_id == cross.set_id]
    scored = []
    for ref in candidates_refs:
        scenario = load_scenario(ref.path)
        scored.append((flow_variable_count(scenario, candidate_paths(scenario)), ref.scenario_id))
    selected = set(select_crosscheck_scenarios(scored, cross.scenario_count))
    for ref in candidates_refs:
        if ref.scenario_id in selected:
            tasks.append(
                CrosscheckTask(
                    ref.set_id, ref.scenario_id, str(ref.path), CROSSCHECK_TRAJECTORY_METHOD,
                    cross.alert_count, cross.realization_id, cross.time_limit_s, config.tangents,
                )
            )
    return tasks


def _shard_is_current(path: Path, expected_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        shard = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return shard.get("task_hash") == expected_hash and shard.get("error") is None


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_rows(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_experiment(
    config: ExperimentConfig, config_path: Path, workers: int | None = None, argv: Sequence[str] = ()
) -> int:
    started = _now()
    output = config.output_dir
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    refs = scenario_refs(config)
    scenario_sha = {(ref.set_id, ref.scenario_id): ref.sha256 for ref in refs}
    current_source = source_hash()
    tasks = build_tasks(config, refs)
    hashes = {task.key: task_hash(task, scenario_sha[(task.set_id, task.scenario_id)], current_source) for task in tasks}
    pending = [task for task in tasks if not _shard_is_current(shard_dir / f"{task.key}.json", hashes[task.key])]
    worker_count = workers or config.workers
    if pending:
        # spawn, not fork: HiGHS thread pools in the parent can deadlock forked children
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(execute, task) for task in pending]
            for future in as_completed(futures):
                shard = future.result()
                shard["task_hash"] = hashes[shard["key"]]
                _write_json(shard_dir / f"{shard['key']}.json", shard)

    shards = {task.key: json.loads((shard_dir / f"{task.key}.json").read_text(encoding="utf-8")) for task in tasks}
    failures = [f"{key}: {shard['error']}" for key, shard in sorted(shards.items()) if shard["error"]]
    method_rows = sorted(
        (row for task in tasks if isinstance(task, MethodTask) for row in shards[task.key]["rows"]),
        key=lambda row: (row["set_id"], row["scenario_id"], row["method"], row["realization_id"]),
    )
    bound_rows = sorted(
        (row for task in tasks if isinstance(task, BoundTask) for row in shards[task.key]["rows"]),
        key=lambda row: (row["set_id"], row["scenario_id"], row["variant"], row["trajectory_method"], row["realization_id"]),
    )
    crosscheck_rows = sorted(
        (row for task in tasks if isinstance(task, CrosscheckTask) for row in shards[task.key]["rows"]),
        key=lambda row: (row["set_id"], row["scenario_id"]),
    )
    _write_rows(output / "method_realizations.csv", METHOD_COLUMNS, method_rows)
    _write_rows(output / "bounds.csv", BOUND_COLUMNS, bound_rows)
    _write_rows(output / "milp_crosscheck.csv", CROSSCHECK_COLUMNS, crosscheck_rows)
    if not failures:
        failures += validity_violations(method_rows, bound_rows)
        failures += crosscheck_violations(crosscheck_rows)
    summary_path = output / "summary.csv"
    if failures:
        summary_path.unlink(missing_ok=True)
    else:
        _write_rows(summary_path, SUMMARY_COLUMNS, summarize(method_rows, bound_rows, config.bootstrap))

    commit, dirty = git_state(REPO_ROOT)
    manifest = {
        "experiment_id": config.experiment_id,
        "config_path": str(config_path),
        "config_hash": config.config_hash,
        "git_commit": commit,
        "git_dirty": dirty,
        "source_hash": current_source,
        "scenario_manifest_sha256": {item.set_id: file_sha256(item.manifest) for item in config.scenario_sets},
        "environment": environment(),
        "workers": worker_count,
        "started_utc": started,
        "finished_utc": _now(),
        "command": list(argv),
        "task_count": len(tasks),
        "executed_task_count": len(pending),
        "skipped_task_count": len(tasks) - len(pending),
        "status": "failed" if failures else "complete",
        "failures": failures,
    }
    _write_json(output / "manifest.json", manifest)
    for failure in failures[:20]:
        print(failure, file=sys.stderr)
    print(json.dumps({key: manifest[key] for key in ("task_count", "executed_task_count", "skipped_task_count", "status")}))
    return 1 if failures else 0
