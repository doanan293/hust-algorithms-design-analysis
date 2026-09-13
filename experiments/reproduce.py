"""Reproducibility checks of spec D Section 12.

Levels: `tables` regenerates statistics and report data into a temporary directory and compares bytes with the committed
files; `scenarios` verifies the raw data and regenerates the RescueNet targets and every scenario set under a temporary
data root, comparing manifest hashes; `experiments` reruns experiment configurations into a temporary directory and
compares result rows on their non-runtime columns. `--compare REFERENCE CANDIDATE` compares two result directories.
Each level writes results/reproducibility/<level>[-<subset>].json and exits non-zero when a check fails.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

from runner.config import ScenarioSetRef, load_experiment_config
from runner.experiment import run_experiment
from runner.provenance import environment, git_state

REPO_ROOT = Path(__file__).resolve().parents[1]
ABSOLUTE_TOLERANCE = 1e-9
RELATIVE_TOLERANCE = 1e-6
EXPERIMENTS = (
    "phase1", "phase2_pilot", "phase2_main", "phase2_budget", "phase2_sensitivity", "phase2_design", "phase2_literature", "phase2_case_study",
)
TABLE_COMMANDS = (
    ("phase2_statistics", ["experiments/phase2_statistics.py", "--output", "{statistics}/statistics.csv"]),
    ("phase2_literature_statistics", ["experiments/phase2_literature_statistics.py", "--output", "{statistics}/statistics_literature.csv"]),
    ("report_tables", ["experiments/report_tables.py", "--output", "{data}"]),
    ("report_tables_phase2", ["experiments/report_tables_phase2.py", "--output", "{data}"]),
    ("report_tables_extensions", ["experiments/report_tables_extensions.py", "--output", "{data}"]),
)
SCENARIO_INPUTS = ("topology_inventory.csv", "sources.json", "rescuenet_selection.csv")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class TableRule:
    """Row keys and non-runtime columns: `exact` must match, `absolute` within 1e-9, `relative` within 1e-6 of max(1, |x|),
    and `when_optimal` like `relative` but only where both rows report milp_status optimal."""

    keys: tuple[str, ...]
    exact: tuple[str, ...] = ()
    absolute: tuple[str, ...] = ()
    relative: tuple[str, ...] = ()
    when_optimal: tuple[str, ...] = ()


RULES = {
    "method_realizations.csv": TableRule(
        ("set_id", "scenario_id", "method", "realization_id"), exact=("timely_count", "connected_count", "evaluate_calls"), absolute=("shortfall",)
    ),
    "bounds.csv": TableRule(("set_id", "scenario_id", "variant", "trajectory_method", "realization_id"), exact=("status",), relative=("value",)),
    "milp_crosscheck.csv": TableRule(("set_id", "scenario_id"), exact=("lp_status", "milp_status"), relative=("lp_ub",), when_optimal=("milp_incumbent",)),
    "runs.csv": TableRule(("step", "scenario_id", "theta", "mu", "block_slots"), exact=("status", "iterations"), absolute=("timely_ratio",)),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _close(first: str, second: str, relative: bool) -> bool:
    x, y = float(first), float(second)
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    limit = RELATIVE_TOLERANCE * max(1.0, abs(x)) if relative else ABSOLUTE_TOLERANCE
    return abs(x - y) <= limit


def compare_rows(reference: Sequence[dict[str, str]], candidate: Sequence[dict[str, str]], rule: TableRule) -> list[str]:
    """Differences of candidate rows from the reference rows with the same keys; every candidate key must be in the reference."""
    index = {tuple(row[key] for key in rule.keys): row for row in reference}
    problems = []
    for row in candidate:
        key = tuple(row[name] for name in rule.keys)
        old = index.get(key)
        if old is None:
            problems.append(f"{key}: missing from the reference")
            continue
        differing = [column for column in rule.exact if old[column] != row[column]]
        differing += [column for column in rule.absolute if not _close(old[column], row[column], False)]
        differing += [column for column in rule.relative if not _close(old[column], row[column], True)]
        if old.get("milp_status") == "optimal" and row.get("milp_status") == "optimal":
            differing += [column for column in rule.when_optimal if not _close(old[column], row[column], True)]
        problems += [f"{key} {column}: {old[column]} -> {row[column]}" for column in differing]
    return problems


def compare_result_dirs(reference: Path, candidate: Path) -> list[Check]:
    checks = []
    for name, rule in RULES.items():
        if not (candidate / name).exists():
            continue
        label = f"{reference.as_posix()}/{name}"
        if not (reference / name).exists():
            checks.append(Check(label, False, "the reference file is missing"))
            continue
        rows, reference_rows = read_rows(candidate / name), read_rows(reference / name)
        if not rows and reference_rows:
            checks.append(Check(label, False, "the rerun produced no rows"))
            continue
        problems = compare_rows(reference_rows, rows, rule)
        detail = f"{len(rows)} rows match" if not problems else "; ".join(problems[:5]) + (f" (and {len(problems) - 5} more)" if len(problems) > 5 else "")
        checks.append(Check(label, not problems, detail))
    return checks


def _run(label: str, arguments: Sequence[str]) -> Check:
    result = subprocess.run([sys.executable, *arguments], cwd=REPO_ROOT, capture_output=True, text=True)
    detail = "exit 0" if result.returncode == 0 else f"exit {result.returncode}: {result.stderr.strip()[-400:]}"
    return Check(label, result.returncode == 0, detail)


def _same_bytes(label: str, generated: Path, committed: Path) -> Check:
    if not committed.exists():
        return Check(label, False, f"{committed} is not committed")
    same = generated.read_bytes() == committed.read_bytes()
    return Check(label, same, "identical" if same else f"differs from {committed}")


def tables_level(workdir: Path, commands=TABLE_COMMANDS, report_data: Path = Path("docs/report/data"), statistics: Path = Path("results/phase2")) -> list[Check]:
    paths = {"data": workdir / "data", "statistics": workdir / "statistics"}
    for folder in paths.values():
        folder.mkdir(parents=True, exist_ok=True)
    checks = [_run(name, [part.format(**{key: str(value) for key, value in paths.items()}) for part in arguments]) for name, arguments in commands]
    checks += [_same_bytes(f"statistics/{path.name}", path, REPO_ROOT / statistics / path.name) for path in sorted(paths["statistics"].iterdir())]
    checks += [_same_bytes(f"report data/{path.name}", path, REPO_ROOT / report_data / path.name) for path in sorted(paths["data"].iterdir())]
    return checks


def recorded_scenario_hashes(results: Path) -> dict[str, set[str]]:
    """Scenario-manifest SHA-256 values recorded by result manifests, by scenario set."""
    recorded: dict[str, set[str]] = {}
    for manifest in sorted(results.rglob("manifest.json")):
        for set_id, digest in json.loads(manifest.read_text(encoding="utf-8")).get("scenario_manifest_sha256", {}).items():
            recorded.setdefault(set_id, set()).add(digest)
    return recorded


def scenarios_level(workdir: Path) -> list[Check]:
    data = REPO_ROOT / "data"
    root = workdir / "data"
    (root / "processed").mkdir(parents=True)
    (root / "manifests").mkdir()
    (root / "raw").symlink_to(data / "raw")
    (root / "processed" / "networks").symlink_to(data / "processed" / "networks")
    for name in SCENARIO_INPUTS:
        shutil.copy2(data / "manifests" / name, root / "manifests" / name)
    checks = [
        _run("raw data verification", ["-m", "data.cli", "verify", "--profile", "configs/data/paper.yaml"]),
        _run("rescuenet targets", ["-m", "data.cli", "rescuenet-targets", "--config", "configs/data/rescuenet_targets.yaml", "--data-root", str(root)]),
    ]
    for name in ("rescuenet_targets.csv", "rescuenet_targets.json"):
        generated = root / "manifests" / name
        checks.append(_same_bytes(f"manifest {name}", generated, data / "manifests" / name) if generated.exists() else Check(f"manifest {name}", False, "not generated"))
    recorded = recorded_scenario_hashes(REPO_ROOT / "results")
    for config in sorted((REPO_ROOT / "configs" / "scenarios").glob("*.yaml")):
        set_id = config.stem
        run = _run(f"scenarios {set_id}", ["-m", "data.cli", "scenarios", "--config", str(config.relative_to(REPO_ROOT)), "--data-root", str(root)])
        checks.append(run)
        manifest = root / "manifests" / f"scenarios_{set_id}.csv"
        if not manifest.exists():
            continue
        checks.append(_same_bytes(f"manifest scenarios_{set_id}.csv", manifest, data / "manifests" / manifest.name))
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if set_id in recorded:
            checks.append(Check(f"recorded hash of {set_id}", recorded[set_id] == {digest}, f"regenerated {digest[:12]}, recorded {sorted(item[:12] for item in recorded[set_id])}"))
    return checks


def _subset_set(scenario_set: ScenarioSetRef, topology: str, split: str, folder: Path) -> tuple[ScenarioSetRef, int]:
    """Manifest copy with the rows of one topology; the count covers only rows the experiment runs (its split and replicates)."""
    rows = read_rows(REPO_ROOT / scenario_set.manifest)
    kept = [row for row in rows if row["topology_id"] == topology]
    runnable = [
        row for row in kept
        if row["split"] == split and (scenario_set.replicates is None or int(row["replicate"]) in scenario_set.replicates)
    ]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / scenario_set.manifest.name
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(kept)
    return replace(scenario_set, manifest=path), len(runnable)


def experiments_level(workdir: Path, config_paths: Sequence[Path], subset: str | None, workers: int | None) -> list[Check]:
    checks = []
    for config_path in config_paths:
        config = load_experiment_config(config_path)
        name = config_path.stem
        sets, crosscheck = config.scenario_sets, config.crosscheck
        if subset:
            filtered = [_subset_set(item, subset, config.split, workdir / "manifests" / name) for item in sets]
            if sum(count for _, count in filtered) == 0:
                checks.append(Check(name, True, f"skipped: no {config.split} scenario of topology {subset}"))
                continue
            sets, crosscheck = tuple(item for item, _ in filtered), None
        run_config = replace(config, scenario_sets=sets, crosscheck=crosscheck, output_dir=workdir / name)
        code = run_experiment(run_config, config_path, workers=workers, argv=["experiments/reproduce.py", str(config_path)])
        if code != 0:
            checks.append(Check(name, False, f"the runner exited with {code}"))
            continue
        checks += compare_result_dirs(config.output_dir, workdir / name)
    return checks


def pilot_check(workdir: Path, workers: int | None) -> list[Check]:
    from tran_pilot import main as run_pilot

    output = workdir / "tran_pilot"
    arguments = ["--output", str(output), "--config-output", str(workdir / "phase2_literature.yaml")]
    code = run_pilot(arguments + (["--workers", str(workers)] if workers else []))
    if code != 0:
        return [Check("tran_pilot", False, f"the pilot exited with {code}")]
    return compare_result_dirs(Path("results/phase2/tran_pilot"), output) + [
        _same_bytes("tran_pilot/choice.json", output / "choice.json", REPO_ROOT / "results" / "phase2" / "tran_pilot" / "choice.json")
    ]


def write_report(path: Path, level: str, subset: str | None, started: str, checks: Sequence[Check]) -> bool:
    commit, dirty = git_state(REPO_ROOT)
    passed = bool(checks) and all(check.passed for check in checks)
    payload = {
        "level": level, "subset": subset, "passed": passed, "git_commit": commit, "git_dirty": dirty, "environment": environment(),
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "checks": [asdict(check) for check in checks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reproduce")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--level", choices=("tables", "scenarios", "experiments"))
    mode.add_argument("--compare", nargs=2, type=Path, metavar=("REFERENCE", "CANDIDATE"))
    parser.add_argument("--subset", help="experiments level: keep only the scenarios of this topology and skip the MILP cross-check and the pilot")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--report-dir", type=Path, default=Path("results/reproducibility"))
    args = parser.parse_args(argv)
    if args.compare:
        checks = compare_result_dirs(*args.compare)
        for check in checks:
            print(f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.detail}")
        return 0 if checks and all(check.passed for check in checks) else 1
    started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="reproduce-") as temporary:
        workdir = Path(temporary)
        if args.level == "tables":
            checks = tables_level(workdir)
        elif args.level == "scenarios":
            checks = scenarios_level(workdir)
        else:
            configs = [Path(f"configs/experiments/{name}.yaml") for name in args.experiments]
            checks = experiments_level(workdir, configs, args.subset, args.workers)
            if not args.subset and "phase2_literature" in args.experiments:
                checks += pilot_check(workdir, args.workers)
            elif args.subset:
                checks.append(Check("tran_pilot", True, "skipped: the pilot uses the development split"))
    name = args.level + (f"-{args.subset}" if args.subset else "")
    passed = write_report(args.report_dir / f"{name}.json", args.level, args.subset, started, checks)
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.detail}")
    print(json.dumps({"level": args.level, "subset": args.subset, "checks": len(checks), "passed": passed}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
