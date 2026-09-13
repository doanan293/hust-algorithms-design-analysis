# Reproducibility Package and Final Report Implementation Plan (Sub-Project D, Plan D.3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every committed result checkable at three levels (tables, scenarios, experiments), refresh the Phase 1 results under the current code, and complete the final report with the SAA framing, the TRAN comparison, the RescueNet case study, the appendix and a reproducibility guide.

**Architecture:** `runner.provenance.result_provenance_problem` lets the report generators accept a result directory when its `source_hash` equals the current one or when it comes from a clean ancestor commit whose result code (`src/models`, `src/baselines`, `src/bounds`, `src/optimization`, `src/literature`) has not been modified, deleted or renamed since. `experiments/reproduce.py` regenerates statistics and report data, target and scenario manifests, and experiment results into temporary directories, compares them with the committed files (bytes, SHA-256, non-runtime result columns) and writes a JSON report to `results/reproducibility/`. `experiments/report_tables_extensions.py` writes `docs/report/data/ext_*`, which hand-written csvsimple tables and pgfplots figures read; `docs/reproducibility.md` maps every report item and every acceptance criterion of `docs/issue.md` Section 14 to its evidence.

**Tech Stack:** Python 3.12 in the uv-managed `.venv`; `pytest`; Git (`merge-base --is-ancestor`, `diff --diff-filter`); pdfLaTeX and biber through `.claude/skills/latex-document-skill/scripts/compile_latex.sh`; `csvsimple`, `pgfplots`, `siunitx`, `cleveref` with babel Vietnamese (T5). No new dependency.

**Spec:** `docs/superpowers/specs/2026-09-12-extensions-final-delivery-design.md`: Sections 11–13, Section 14 (reproduce), 15.3, acceptance criteria 16.3, and Section 17 "Plan D.3 adjustments" (committed with this plan). Builds on plans D.1 (`docs/superpowers/plans/2026-09-12-literature-baseline.md`) and D.2 (`docs/superpowers/plans/2026-09-12-rescuenet-case-study.md`), both merged on `develop`.

## Global Constraints

- Run project code through uv: tests with `uv run --extra test pytest`, scripts with `uv run python ...`. No dependency is added.
- Source changes are limited to `src/runner/provenance.py`; `src/models`, `src/baselines`, `src/bounds`, `src/optimization`, `src/literature`, `src/data` and `src/analysis` do not change; nothing under `src` imports `experiments`.
- **Memory caps.** pytest runs as `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest ...`; the Phase 1 rerun and `reproduce.py --level experiments` run under `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0`, because the experiments level solves TRAN programs. Never stop processes with `pkill -f`.
- Result provenance: a result directory is current when its manifest has `status` `complete` and either its `source_hash` equals `runner.provenance.source_hash()`, or its `git_commit` is recorded with `git_dirty` false, is an ancestor of `HEAD`, and `git diff --quiet --diff-filter=a <commit> -- src/models src/baselines src/bounds src/optimization src/literature` succeeds (added files are ignored; `src/runner` is not a result package).
- Reproduce comparisons: rows are matched by key; `timely_count`, `connected_count`, `evaluate_calls`, statuses and pilot iterations must be equal; `shortfall` and the pilot timely ratio within 1e-9; LP bound values and `lp_ub` within 1e-6 × max(1, |x|); `milp_incumbent` only where both runs report `optimal`; runtime columns are ignored; a rerun with no rows fails when the reference file has rows. Each level writes `results/reproducibility/<level>[-<subset>].json` and exits non-zero when any check fails; differences are reported, not filtered.
- `--subset TOPOLOGY` keeps the scenarios of one evaluation topology, drops the MILP cross-check, and skips `phase2_pilot` and `tran_pilot.py`, which run on the development split.
- Report compilation: never compile inside `docs/report` (the editor auto-builds it); copy `docs/report` to a directory outside the workspace and run the skill script there with `--verbose`. Accept only a final pass with no `Overfull`, `Underfull`, `LaTeX Warning`, `Package ... Warning` or `!` error lines, no `??` in the PDF text, and Vietnamese cross-reference words (no English "and" or "appendix" from `cleveref`).
- RescueNet images, masks and pictures derived from them are not committed; the case-study map is drawn from the normalized target coordinates of the committed manifests.
- Commit only the files each task lists; `docs/report/build/` and `docs/report/main.pdf` stay untracked. Test module basenames are unique across `tests/`; every code task follows test-driven development and ends with its own commit.

---

### Task 1: Result provenance rule

**Files:**
- Modify: `src/runner/provenance.py`, `experiments/report_tables.py`, `experiments/report_tables_phase2.py`
- Test: `tests/runner/test_runner_result_provenance.py`

**Interfaces:**
- Consumes: `runner.provenance.SRC_ROOT` and `source_hash(src_root) -> str`; Git on `PATH`; `experiments/report_tables.py` and `experiments/report_tables_phase2.py`, which until now compare `manifest["source_hash"]` with `source_hash()`.
- Produces: `runner.provenance.REPO_ROOT`; `RESULT_PACKAGES = ("models", "baselines", "bounds", "optimization", "literature")`; `result_provenance_problem(manifest: Mapping, repo_root: Path = REPO_ROOT, src_root: Path = SRC_ROOT) -> str | None`, which returns `None` for current results and otherwise the reason. Both report generators keep their refusal message and exit code 1, now decided by this function.

The tests build a temporary Git repository with its own `src/` tree and pass `repo_root` and `src_root`, so they do not depend on the history of this repository. `--diff-filter=a` ignores added files: a module added later (as plan D.1 added `src/literature`) cannot change results unless an existing file imports it, and that edit is caught.

- [ ] **Step 1: Write the failing tests**

Create `tests/runner/test_runner_result_provenance.py`:

```python
import subprocess
from pathlib import Path

from runner.provenance import result_provenance_problem, source_hash


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    for package, content in (("models", "x = 1\n"), ("runner", "y = 1\n")):
        (root / "src" / package).mkdir(parents=True)
        (root / "src" / package / "module.py").write_text(content, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.com")
    _git(root, "config", "user.name", "tests")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "results")
    return root, _git(root, "rev-parse", "HEAD")


def _manifest(commit: str, dirty: bool = False, source: str = "0" * 64) -> dict[str, object]:
    return {"status": "complete", "source_hash": source, "git_commit": commit, "git_dirty": dirty}


def test_results_with_the_current_source_hash_are_current(tmp_path: Path):
    root, commit = _repository(tmp_path)
    assert result_provenance_problem(_manifest(commit, source=source_hash(root / "src")), root, root / "src") is None


def test_runner_changes_after_a_clean_ancestor_commit_keep_results_current(tmp_path: Path):
    root, commit = _repository(tmp_path)
    (root / "src" / "runner" / "module.py").write_text("y = 2\n", encoding="utf-8")
    _git(root, "commit", "-qam", "runner")
    assert result_provenance_problem(_manifest(commit), root, root / "src") is None


def test_files_added_to_result_packages_after_the_commit_keep_results_current(tmp_path: Path):
    root, commit = _repository(tmp_path)
    (root / "src" / "literature").mkdir()
    (root / "src" / "literature" / "method.py").write_text("z = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "new package")
    assert result_provenance_problem(_manifest(commit), root, root / "src") is None


def test_committed_or_uncommitted_result_code_changes_are_rejected(tmp_path: Path):
    root, commit = _repository(tmp_path)
    (root / "src" / "models" / "module.py").write_text("x = 2\n", encoding="utf-8")
    assert f"changed after {commit}" in result_provenance_problem(_manifest(commit), root, root / "src")
    _git(root, "commit", "-qam", "models")
    assert f"changed after {commit}" in result_provenance_problem(_manifest(commit), root, root / "src")


def test_incomplete_dirty_and_foreign_results_are_rejected(tmp_path: Path):
    root, commit = _repository(tmp_path)
    assert result_provenance_problem({**_manifest(commit), "status": "failed"}, root, root / "src") == "the experiment is not complete"
    assert "record no clean commit" in result_provenance_problem(_manifest(commit, dirty=True), root, root / "src")
    assert "not an ancestor of HEAD" in result_provenance_problem(_manifest("f" * 40), root, root / "src")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/runner/test_runner_result_provenance.py tests/runner/test_report_tables.py tests/runner/test_report_tables_phase2.py`

Expected: collection fails (`1 error`) with `ImportError: cannot import name 'result_provenance_problem' from 'runner.provenance' (<repo>/src/runner/provenance.py)`.

- [ ] **Step 3: Implement**

In `src/runner/provenance.py` (3 edits):

Edit 1: replace

```python

import clarabel
```

with

```python
from typing import Mapping

import clarabel
```

Edit 2: replace

```python

SRC_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGES = ("models", "baselines", "bounds", "optimization", "literature", "runner")
```

with

```python

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
SOURCE_PACKAGES = ("models", "baselines", "bounds", "optimization", "literature", "runner")
RESULT_PACKAGES = ("models", "baselines", "bounds", "optimization", "literature")
```

Edit 3: replace

```python
        "cpu_count": os.cpu_count(),
    }
```

with

```python
        "cpu_count": os.cpu_count(),
    }


def result_provenance_problem(
    manifest: Mapping[str, object], repo_root: Path = REPO_ROOT, src_root: Path = SRC_ROOT
) -> str | None:
    """Why a results manifest does not describe the current result code, or None when it does (spec D, Plan D.3 adjustments).

    Results are current when their source hash equals the current one, or when they come from a clean commit that is an
    ancestor of HEAD and no file of the packages that determine results (RESULT_PACKAGES) was modified, deleted or renamed
    since that commit, including uncommitted changes. Files added later are ignored, because existing code must change to use
    them. The runner is left out: it orchestrates tasks, and the experiments level of experiments/reproduce.py checks it end
    to end.
    """
    if manifest.get("status") != "complete":
        return "the experiment is not complete"
    if manifest.get("source_hash") == source_hash(src_root):
        return None
    commit = manifest.get("git_commit")
    if not commit or manifest.get("git_dirty") is not False:
        return "the results were produced by different source code and record no clean commit"
    paths = [f"src/{package}" for package in RESULT_PACKAGES]
    try:
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", str(commit), "HEAD"], cwd=repo_root, capture_output=True)
        changed = subprocess.run(["git", "diff", "--quiet", "--diff-filter=a", str(commit), "--", *paths], cwd=repo_root, capture_output=True)
    except OSError:
        return "the results were produced by different source code and Git is unavailable"
    if ancestor.returncode != 0:
        return f"the results were produced by different source code at {commit}, which is not an ancestor of HEAD"
    if changed.returncode != 0:
        return f"the result code in {', '.join(paths)} changed after {commit}"
    return None
```


In `experiments/report_tables.py` (2 edits):

Edit 1: replace

```python
from runner.provenance import source_hash
```

with

```python
from runner.provenance import result_provenance_problem
```

Edit 2: replace

```python
    if manifest["status"] != "complete" or manifest["source_hash"] != source_hash():
```

with

```python
    if result_provenance_problem(manifest):
```


In `experiments/report_tables_phase2.py` (3 edits):

Edit 1: replace

```python
from runner.provenance import source_hash
```

with

```python
from runner.provenance import result_provenance_problem
```

Edit 2: replace

```python
    current = source_hash()
```

with

```python

```

Edit 3: replace

```python
        if manifest["status"] != "complete" or manifest["source_hash"] != current:
```

with

```python
        if result_provenance_problem(manifest):
```


- [ ] **Step 4: Run the tests to verify they pass**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/runner/test_runner_result_provenance.py tests/runner/test_report_tables.py tests/runner/test_report_tables_phase2.py`

Expected: `9 passed`.

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `275 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/runner/test_runner_result_provenance.py src/runner/provenance.py experiments/report_tables.py experiments/report_tables_phase2.py
git commit -m "feat: accept results whose result code is unchanged since their commit"
```

### Task 2: Reproducibility checks

**Files:**
- Create: `experiments/reproduce.py`
- Test: `tests/runner/test_reproduce.py`

**Interfaces:**
- Consumes: `runner.config.load_experiment_config(path) -> ExperimentConfig` (fields `scenario_sets`, `crosscheck`, `output_dir`, `split`) and `ScenarioSetRef` (`manifest: Path`, `replicates`); `runner.experiment.run_experiment(config, config_path, workers=..., argv=...) -> int`; `runner.provenance.environment()` and `git_state(repo_root) -> (commit, dirty)`; `experiments/tran_pilot.py` `main(argv)` with `--output`, `--config-output`, `--workers`; the `--output` option of `phase2_statistics.py`, `phase2_literature_statistics.py`, `report_tables.py`, `report_tables_phase2.py` and `report_tables_extensions.py` (Task 4; the `tables` level runs it for real only in Task 8); `python -m data.cli verify --profile PATH`, `rescuenet-targets --config PATH --data-root PATH`, `scenarios --config PATH --data-root PATH`; test helper `experiment_fixtures.write_tiny_experiment`.
- Produces: `experiments/reproduce.py`: `Check(name, passed, detail)`; `TableRule(keys, exact, absolute, relative, when_optimal)`; `RULES` for `method_realizations.csv`, `bounds.csv`, `milp_crosscheck.csv`, `runs.csv`; `compare_rows(reference, candidate, rule) -> list[str]`; `compare_result_dirs(reference, candidate) -> list[Check]`; `tables_level(workdir, commands=TABLE_COMMANDS, report_data, statistics) -> list[Check]`; `recorded_scenario_hashes(results) -> dict[str, set[str]]`; `scenarios_level(workdir) -> list[Check]`; `experiments_level(workdir, config_paths, subset, workers) -> list[Check]`; `pilot_check(workdir, workers) -> list[Check]`; `write_report(path, level, subset, started, checks) -> bool`; `main(argv) -> int`. CLI: `--level {tables,scenarios,experiments}` or `--compare REFERENCE CANDIDATE`, with `--subset TOPOLOGY`, `--experiments NAME ...`, `--workers N`, `--report-dir PATH` (default `results/reproducibility`); prints one `PASS name: detail` or `FAIL name: detail` line per check, then `{"level": ..., "subset": ..., "checks": ..., "passed": ...}`.

Runtime columns differ between any two runs and are never compared. `--subset` counts only the scenarios an experiment actually runs (its split and replicates), so a development-split experiment is skipped instead of producing an empty rerun, and a comparison fails when the rerun has no rows but the reference has some (`phase2_budget` and `phase2_design` compute no bounds, so their header-only `bounds.csv` files still match).

- [ ] **Step 1: Write the failing tests**

Create `tests/runner/test_reproduce.py`:

```python
import csv
import importlib.util
import json
from pathlib import Path
import sys

from experiment_fixtures import write_tiny_experiment
from runner.config import load_experiment_config
from runner.experiment import run_experiment

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
SPEC = importlib.util.spec_from_file_location("reproduce", EXPERIMENTS / "reproduce.py")
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


def _row(**values):
    base = {"set_id": "v0", "scenario_id": "A-r0", "method": "B1", "realization_id": "0", "timely_count": "3", "connected_count": "7",
            "evaluate_calls": "1", "shortfall": "0.5", "planning_runtime_s": "1.0"}
    return {**base, **values}


def test_method_rows_compare_non_runtime_columns_with_tolerances():
    rule = reproduce.RULES["method_realizations.csv"]
    reference = [_row()]
    assert reproduce.compare_rows(reference, [_row(planning_runtime_s="99", shortfall="0.5000000001")], rule) == []
    problems = reproduce.compare_rows(reference, [_row(timely_count="4", shortfall="0.51"), _row(scenario_id="B-r0")], rule)
    assert problems == ["('v0', 'A-r0', 'B1', '0') timely_count: 3 -> 4", "('v0', 'A-r0', 'B1', '0') shortfall: 0.5 -> 0.51",
                        "('v0', 'B-r0', 'B1', '0'): missing from the reference"]


def test_bounds_use_relative_tolerance_and_milp_incumbents_count_only_when_optimal():
    bounds = reproduce.RULES["bounds.csv"]
    key = {"set_id": "v0", "scenario_id": "A-r0", "variant": "trajectory", "trajectory_method": "B1", "realization_id": "0", "status": "optimal"}
    assert reproduce.compare_rows([{**key, "value": "1000.0"}], [{**key, "value": "1000.0005"}], bounds) == []
    assert len(reproduce.compare_rows([{**key, "value": "1000.0"}], [{**key, "value": "1000.01"}], bounds)) == 1
    milp = reproduce.RULES["milp_crosscheck.csv"]
    row = {"set_id": "v0", "scenario_id": "A-r0", "lp_status": "optimal", "lp_ub": "5.0", "milp_status": "time_limit", "milp_incumbent": "3.0"}
    assert reproduce.compare_rows([row], [{**row, "milp_incumbent": "2.0"}], milp) == []
    optimal = {**row, "milp_status": "optimal"}
    assert reproduce.compare_rows([optimal], [{**optimal, "milp_incumbent": "2.0"}], milp) == ["('v0', 'A-r0') milp_incumbent: 3.0 -> 2.0"]


def test_tables_level_reports_failed_commands_and_byte_differences(tmp_path: Path):
    committed = tmp_path / "committed"
    committed.mkdir()
    (committed / "same.csv").write_text("a\n", encoding="utf-8")
    (committed / "changed.csv").write_text("b\n", encoding="utf-8")
    writer = "import pathlib, sys; folder = pathlib.Path(sys.argv[1]); (folder / 'same.csv').write_text('a\\n'); (folder / 'changed.csv').write_text('c\\n')"
    commands = (("writer", ["-c", writer, "{data}"]), ("failing", ["-c", "import sys; sys.exit('broken')"]))
    checks = reproduce.tables_level(tmp_path / "work", commands, report_data=committed)
    by_name = {check.name: check for check in checks}
    assert by_name["writer"].passed and not by_name["failing"].passed and "broken" in by_name["failing"].detail
    assert by_name["report data/same.csv"].passed and not by_name["report data/changed.csv"].passed


def test_empty_reruns_fail_only_when_the_reference_has_rows(tmp_path: Path):
    header = "set_id,scenario_id,method,realization_id,timely_count,connected_count,evaluate_calls,shortfall\n"
    for name, body in (("reference", "v0,A-r0,B1,0,3,7,1,0.5\n"), ("candidate", "")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "method_realizations.csv").write_text(header + body, encoding="utf-8")
    [check] = reproduce.compare_result_dirs(tmp_path / "reference", tmp_path / "candidate")
    assert (check.passed, check.detail) == (False, "the rerun produced no rows")
    (tmp_path / "reference" / "method_realizations.csv").write_text(header, encoding="utf-8")
    [check] = reproduce.compare_result_dirs(tmp_path / "reference", tmp_path / "candidate")
    assert (check.passed, check.detail) == (True, "0 rows match")


def test_recorded_scenario_hashes_collect_every_manifest(tmp_path: Path):
    for name, digest in (("a", "1" * 64), ("b", "2" * 64)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps({"scenario_manifest_sha256": {"v0": digest, "rescuenet": "3" * 64}}), encoding="utf-8")
    assert reproduce.recorded_scenario_hashes(tmp_path) == {"v0": {"1" * 64, "2" * 64}, "rescuenet": {"3" * 64}}


def test_experiments_level_reruns_a_subset_and_detects_changed_results(tmp_path: Path, capsys):
    config_path = write_tiny_experiment(tmp_path, tmp_path / "reference", 1, ("B0", "B1"), crosscheck=False)
    assert run_experiment(load_experiment_config(config_path), config_path) == 0
    checks = reproduce.experiments_level(tmp_path / "work", [config_path], "TinyA", 1)
    assert checks and all(check.passed for check in checks)
    assert {check.name.rsplit("/", 1)[1] for check in checks} == {"method_realizations.csv", "bounds.csv"}
    assert [check.detail for check in checks][0] == "4 rows match"
    skipped = reproduce.experiments_level(tmp_path / "work-none", [config_path], "Missing", 1)
    assert [(check.passed, check.detail) for check in skipped] == [(True, "skipped: no eval scenario of topology Missing")]
    development = tmp_path / "development.yaml"
    development.write_text(config_path.read_text(encoding="utf-8").replace("split: eval", "split: dev"), encoding="utf-8")
    other_split = reproduce.experiments_level(tmp_path / "work-dev", [development], "TinyA", 1)
    assert [(check.passed, check.detail) for check in other_split] == [(True, "skipped: no dev scenario of topology TinyA")]
    rows = reproduce.read_rows(tmp_path / "reference" / "method_realizations.csv")
    rows[0]["timely_count"] = str(int(rows[0]["timely_count"]) + 1)
    with (tmp_path / "reference" / "method_realizations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert reproduce.main(["--compare", str(tmp_path / "reference"), str(tmp_path / "work" / "experiment-reference")]) == 1
    printed = capsys.readouterr().out
    assert "FAIL " in printed and "timely_count" in printed and "PASS " in printed
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/runner/test_reproduce.py`

Expected: collection fails (`1 error`) with `FileNotFoundError: [Errno 2] No such file or directory: '<repo>/experiments/reproduce.py'`.

- [ ] **Step 3: Implement**

Create `experiments/reproduce.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/runner/test_reproduce.py`

Expected: `6 passed`.

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `281 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/runner/test_reproduce.py experiments/reproduce.py
git commit -m "feat: add reproducibility checks for tables, scenarios and experiments"
```

### Task 3: Refresh the Phase 1 results

**Files:**
- Modify (generated): `results/phase1/method_realizations.csv`, `results/phase1/bounds.csv`, `results/phase1/milp_crosscheck.csv`, `results/phase1/manifest.json`, `docs/report/data/lp_runtime.csv`, `docs/report/data/phase1_milp.csv`, `docs/report/data/phase1_results.tex`

**Interfaces:**
- Consumes: `runner.provenance.result_provenance_problem` (Task 1); `experiments/reproduce.py --compare REFERENCE CANDIDATE` (Task 2); `experiments/run_phase1.py --config PATH --output-dir PATH`; `experiments/report_tables.py`.
- Produces: Phase 1 results whose manifest records the clean commit of Task 2, so `report_tables.py` (this task), `report_tables_extensions.py` (Task 4) and `reproduce.py --level tables` (Task 8) accept them.

The committed Phase 1 results were produced at commit `4433e38` ("feat: add parallel resumable experiment runner"). Sub-project B later changed `src/models/channel.py`, `src/models/evaluate.py`, `src/models/plan.py`, `src/models/simulator.py` and `src/bounds/solve.py`, and plan D.1 added `src/literature` to the source hash, so under the rule of Task 1 the results no longer describe the current code, and `report_tables.py` refuses them. Rerunning Phase 1 (about 17 minutes with 12 workers) and comparing its non-runtime columns with the committed rows shows whether those changes altered any result; the prototype run found none. Start this task with a clean working tree, because the manifest must record `git_dirty` false.

- [ ] **Step 1: Confirm that the committed results are rejected**

Run:

```bash
git status --short src configs experiments
uv run python -c "import json; from runner.provenance import result_provenance_problem; print(result_provenance_problem(json.load(open('results/phase1/manifest.json'))))"
```

Expected: `git status` prints nothing, then `the result code in src/models, src/baselines, src/bounds, src/optimization, src/literature changed after 4433e38de4f4c5cdaab6837fe85e2cdf72d8d778`.

- [ ] **Step 2: Rerun Phase 1 into a scratch directory**

Run:

```bash
WORK=${TMPDIR:-/tmp}/plan-d3-phase1
rm -rf "$WORK" && mkdir -p "$WORK/reference"
for name in method_realizations.csv bounds.csv milp_crosscheck.csv; do git show HEAD:results/phase1/$name > "$WORK/reference/$name"; done
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/run_phase1.py --config configs/experiments/phase1.yaml --output-dir "$WORK/rerun"
```

Expected: the last line is `{"task_count": 3725, "executed_task_count": 3725, "skipped_task_count": 0, "status": "complete"}`.

- [ ] **Step 3: Compare the rerun with the committed rows**

Run:

```bash
WORK=${TMPDIR:-/tmp}/plan-d3-phase1
uv run python experiments/reproduce.py --compare "$WORK/reference" "$WORK/rerun"
```

Expected (with `$REFERENCE` standing for `$WORK/reference`), exit code 0:

```text
PASS $REFERENCE/method_realizations.csv: 3600 rows match
PASS $REFERENCE/bounds.csv: 3600 rows match
PASS $REFERENCE/milp_crosscheck.csv: 5 rows match
```

If any line starts with `FAIL`, stop: the result code changes altered Phase 1 results, and the report text of sub-project B would need revisiting before these results replace the committed ones.

- [ ] **Step 4: Replace the committed results and regenerate the Phase 1 report data**

Run:

```bash
WORK=${TMPDIR:-/tmp}/plan-d3-phase1
cp "$WORK/rerun/method_realizations.csv" "$WORK/rerun/bounds.csv" "$WORK/rerun/milp_crosscheck.csv" "$WORK/rerun/manifest.json" results/phase1/
uv run python -c "import json; from runner.provenance import result_provenance_problem; print(result_provenance_problem(json.load(open('results/phase1/manifest.json'))))"
uv run python experiments/report_tables.py
git status --short results/phase1 docs/report/data
```

Expected: `None`, then the `report_tables.py` output

```text
{"params": 24, "summary": 4, "milp": 5, "lp": 3600}
```

and exactly these changed files:

```text
 M docs/report/data/lp_runtime.csv
 M docs/report/data/phase1_milp.csv
 M docs/report/data/phase1_results.tex
 M results/phase1/bounds.csv
 M results/phase1/manifest.json
 M results/phase1/method_realizations.csv
 M results/phase1/milp_crosscheck.csv
```

`git diff docs/report/data` shows changes only in LP and MILP runtimes (`lp_runtime.csv`, the time column of `phase1_milp.csv`, and the runtime macros of `phase1_results.tex`).

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `281 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add results/phase1/method_realizations.csv results/phase1/bounds.csv results/phase1/milp_crosscheck.csv results/phase1/manifest.json docs/report/data/lp_runtime.csv docs/report/data/phase1_milp.csv docs/report/data/phase1_results.tex
git commit -m "feat: refresh the Phase 1 results under the current code"
```

### Task 4: Report data for TRAN, the case study and compute time

**Files:**
- Create: `experiments/report_tables_extensions.py`
- Test: `tests/runner/test_report_tables_extensions.py`

**Interfaces:**
- Consumes: Task 1 `result_provenance_problem`; the refreshed `results/phase1/` of Task 3; `results/phase2/{main,budget,sensitivity,design,literature,case_study}/` with their `manifest.json`; `results/phase2/statistics_literature.csv`; `results/phase2/tran_pilot/` (`runs.csv`, `channel_fit.csv`, `choice.json`); `results/phase2/case_study/trace/` (`targets.csv`, `nodes.csv`, `edges.csv`, `trajectories.csv`, `progress.csv`, `trace.json`); `data/manifests/rescuenet_targets.json`.
- Produces: `experiments/report_tables_extensions.py` `main(argv) -> int` with `--results` (default `results/phase2`), `--phase1`, `--manifests`, `--output` (default `docs/report/data`). It writes `ext_literature.csv`, `ext_literature_statistics.csv`, `ext_quality_points.csv`, `ext_case_study.csv`, `ext_map_{targets,relays,center,edges_alive,edges_failed,trajectory_b1,trajectory_p,trajectory_tran}.csv`, `ext_progress.csv`, `ext_compute.csv` and `ext_results.tex` (macros `\ResultExt...`: timely ratio, gap and planning time per scenario set and method, the P $-$ TRAN and B2 $-$ TRAN statistics, pilot choice and channel fit, extraction counts, trace values, `\ResultExtComputeHours`), prints the counts as JSON, and exits with 1 when a result directory fails the provenance rule.

- [ ] **Step 1: Write the failing tests**

Create `tests/runner/test_report_tables_extensions.py`:

```python
import csv
import importlib.util
import json
from pathlib import Path
import sys

from runner.provenance import source_hash

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
SPEC = importlib.util.spec_from_file_location("report_tables_extensions", EXPERIMENTS / "report_tables_extensions.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

SUMMARY_COLUMNS = ["set_id", "method", "scenario_count", "timely_ratio_mean", "timely_ratio_ci_low", "timely_ratio_ci_high", "conn_ratio_mean", "gap_mean", "planning_runtime_s_mean"]
ROW_COLUMNS = ["set_id", "scenario_id", "method", "realization_id", "alert_count", "planning_runtime_s", "evaluation_runtime_s"]


def _write(path: Path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _experiment(folder: Path, values: dict, source: str, extra_manifest=None):
    _write(folder / "summary.csv", SUMMARY_COLUMNS, [
        {"set_id": set_id, "method": method, "scenario_count": "30", "timely_ratio_mean": timely, "timely_ratio_ci_low": timely - 0.05,
         "timely_ratio_ci_high": timely + 0.05, "conn_ratio_mean": 1.0, "gap_mean": 0.25, "planning_runtime_s_mean": 120.0}
        for (set_id, method), timely in values.items()
    ])
    _write(folder / "method_realizations.csv", ROW_COLUMNS, [
        {"set_id": set_id, "scenario_id": "Agis-r0", "method": method, "realization_id": "0", "alert_count": "40", "planning_runtime_s": "3600", "evaluation_runtime_s": "0"}
        for (set_id, method) in values
    ])
    _write(folder / "bounds.csv", ["status", "runtime_s"], [{"status": "optimal", "runtime_s": "1800"}])
    manifest = {"status": "complete", "source_hash": source, "task_count": 10, "environment": {"cvxpy": "1.9.2", "clarabel": "0.11.1"}}
    (folder / "manifest.json").write_text(json.dumps({**manifest, **(extra_manifest or {})}), encoding="utf-8")


def _results(root: Path, source: str):
    phase2, phase1 = root / "phase2", root / "phase1"
    sets = ("v0", "v0-bh50")
    _experiment(phase2 / "main", {(s, m): v for s in sets for m, v in (("B1", 0.5), ("B2", 0.55), ("P", 0.55))}, source)
    for name in ("budget", "sensitivity", "design"):
        _experiment(phase2 / name, {("v0", "P"): 0.5}, source)
    _experiment(phase1, {("v0", "B1"): 0.5}, source)
    _write(phase1 / "milp_crosscheck.csv", ["milp_runtime_s"], [{"milp_runtime_s": "3600"}])
    _experiment(phase2 / "literature", {(s, m): v for s in sets for m, v in (("B1", 0.5), ("TRAN", 0.53))}, source)
    _experiment(phase2 / "case_study", {("rescuenet", m): v for m, v in (("B0", 0.3), ("B1", 0.4), ("B2", 0.45), ("P", 0.45), ("TRAN", 0.44))}, source)
    _write(phase2 / "statistics_literature.csv", ["set_id", "comparison", "mean_difference", "ci_low", "ci_high", "rank_biserial", "p_value", "p_holm", "topology_count"], [
        {"set_id": s, "comparison": c, "mean_difference": 0.02, "ci_low": -0.01, "ci_high": 0.05, "rank_biserial": 0.4, "p_value": 0.36, "p_holm": 0.72, "topology_count": 10}
        for s in sets for c in ("P - TRAN", "B2 - TRAN")
    ])
    pilot = phase2 / "tran_pilot"
    pilot.mkdir(parents=True)
    (pilot / "choice.json").write_text(json.dumps({"theta": 1 / 3, "mu": 10.0, "block_slots": 1, "median_runtime_step1_s": 134.12, "scenarios": ["A-r0", "B-r0"],
                                                   "mean_timely_ratio": 0.5436, "better_than_initial_count": 2}), encoding="utf-8")
    run_columns = ["step", "scenario_id", "theta", "mu", "block_slots", "planning_runtime_s", "timely_ratio"]
    _write(pilot / "runs.csv", run_columns, [
        {"step": "runtime", "scenario_id": "A-r0", "theta": 0.5, "mu": 1.0, "block_slots": 1, "planning_runtime_s": 1800, "timely_ratio": 0.5},
        {"step": "grid", "scenario_id": "A-r0", "theta": 0.5, "mu": 1.0, "block_slots": 1, "planning_runtime_s": 1800, "timely_ratio": 0.5},
        {"step": "grid", "scenario_id": "A-r0", "theta": 0.3333333333333333, "mu": 10.0, "block_slots": 1, "planning_runtime_s": 1800, "timely_ratio": 0.6},
    ])
    _write(pilot / "channel_fit.csv", ["scenario_id", "beta", "alpha", "r_squared", "max_log_error"], [
        {"scenario_id": "A-r0", "beta": 0.002, "alpha": 3.35, "r_squared": 0.9937, "max_log_error": 0.5},
        {"scenario_id": "B-r0", "beta": 0.003, "alpha": 3.42, "r_squared": 0.9947, "max_log_error": 0.4},
    ])
    trace = phase2 / "case_study" / "trace"
    _write(trace / "targets.csv", ["source_id", "target_id", "x_m", "y_m", "damage_class", "zone"], [
        {"source_id": "s00", "target_id": "1-r01", "x_m": 1000, "y_m": 500, "damage_class": 4, "zone": 0}])
    _write(trace / "nodes.csv", ["node_id", "x_m", "y_m", "role"], [
        {"node_id": "n0", "x_m": 0, "y_m": 0, "role": "center"}, {"node_id": "n1", "x_m": 2000, "y_m": 0, "role": "relay"}])
    _write(trace / "edges.csv", ["source", "target", "source_x_m", "source_y_m", "target_x_m", "target_y_m", "failed"], [
        {"source": "n0", "target": "n1", "source_x_m": 0, "source_y_m": 0, "target_x_m": 2000, "target_y_m": 0, "failed": 1}])
    _write(trace / "trajectories.csv", ["method", "slot", "x_m", "y_m"], [
        {"method": method, "slot": slot, "x_m": 100 * slot, "y_m": 0} for method in ("B1", "P", "TRAN") for slot in range(3)])
    _write(trace / "progress.csv", ["method", "slot", "mean_timely_alerts"], [
        {"method": method, "slot": slot, "mean_timely_alerts": 10 * slot} for method in ("B1", "P", "TRAN") for slot in range(2)])
    (trace / "trace.json").write_text(json.dumps({"scenario_id": "Agis-r0", "methods": {m: {"timely_ratio_trace": 0.4} for m in ("B1", "P", "TRAN")}}), encoding="utf-8")
    manifests = root / "manifests"
    (manifests).mkdir()
    (manifests / "rescuenet_targets.json").write_text(json.dumps({
        "images": [{"pair_id": "1", "gsd_m_per_px": 0.0196, "focal_source": "calibrated", "targets": 2}, {"pair_id": "2", "gsd_m_per_px": 0.0246, "focal_source": "fallback", "targets": 0}],
        "region_count": 3, "target_count": 2, "dropped_regions": [], "merged_pairs": [{"distance_m": 2.7}, {"distance_m": 8.2}], "normalization": {"scale": 0.693323}}), encoding="utf-8")
    for set_id, fraction in (("rescuenet", 0.376), ("v0", 0.509)):
        _write(manifests / f"scenarios_{set_id}.csv", ["scenario_id", "split", "ground_connected_source_fraction"], [
            {"scenario_id": "Agis-r0", "split": "eval", "ground_connected_source_fraction": fraction},
            {"scenario_id": "Dev-r0", "split": "dev", "ground_connected_source_fraction": 0.0}])
    return phase2, phase1, manifests


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_extension_report_files_follow_the_results(tmp_path: Path):
    phase2, phase1, manifests = _results(tmp_path, source_hash())
    output = tmp_path / "data"
    arguments = ["--results", str(phase2), "--phase1", str(phase1), "--manifests", str(manifests), "--output", str(output)]
    assert report.main(arguments) == 0
    assert _lines(output / "ext_literature.csv")[:2] == ["set;backhaul;method;timely;ci;gap;planning",
                                                        "v0;\\qty{1000}{\\kilo\\bit\\per\\second};B1;\\num{0.500};[\\num{0.450}, \\num{0.550}];\\num{0.250};\\num{120.0}"]
    assert _lines(output / "ext_literature.csv")[4].split(";")[2:4] == ["TRAN", "\\num{0.530}"]
    assert _lines(output / "ext_literature_statistics.csv")[1] == "v0;P $-$ TRAN;\\num{0.020};[\\num{-0.010}, \\num{0.050}];\\num{0.40};\\num{0.3600};\\num{0.7200}"
    assert _lines(output / "ext_case_study.csv")[5] == "TRAN;\\num{0.440};[\\num{0.390}, \\num{0.490}];\\num{1.000};\\num{0.250};\\num{120.0}"
    assert _lines(output / "ext_quality_points.csv") == ["method,planning,timely", "B1,120.000,0.500000", "TRAN,120.000,0.530000"]
    assert _lines(output / "ext_map_targets.csv") == ["x,y,class", "1.0000,0.5000,4"]
    assert _lines(output / "ext_map_edges_alive.csv") == ["x,y", "nan,nan"]
    assert _lines(output / "ext_map_edges_failed.csv") == ["x,y", "0.0000,0.0000", "2.0000,0.0000", "nan,nan"]
    assert _lines(output / "ext_map_trajectory_tran.csv") == ["x,y", "0.0000,0.0000", "0.1000,0.0000", "0.2000,0.0000"]
    assert _lines(output / "ext_progress.csv") == ["slot,B1,P,TRAN", "0,0.000000,0.000000,0.000000", "1,0.250000,0.250000,0.250000"]
    compute = [line.split(";") for line in _lines(output / "ext_compute.csv")]
    assert [row[0] for row in compute[1:]][5] == "Pilot TRAN trên tập phát triển" and compute[6][1:] == ["2", "\\num{1.0}"]
    assert compute[1][1:] == ["10", "\\num{2.5}"] and len(compute) == 9
    macros = (output / "ext_results.tex").read_text(encoding="utf-8")
    for expected in (
        "\\newcommand{\\ResultExtTimelyVzeroTRAN}{\\num{0.530}}", "\\newcommand{\\ResultExtHolmVzeroBhfivezeroPTran}{\\num{0.7200}}",
        "\\newcommand{\\ResultExtPilotTheta}{\\num{0.333}}", "\\newcommand{\\ResultExtPilotMu}{\\num{10}}", "\\newcommand{\\ResultExtPilotGridSpread}{\\num{0.100}}",
        "\\newcommand{\\ResultExtCaseTimelyTRAN}{\\num{0.440}}", "\\newcommand{\\ResultExtMergeDistanceMax}{\\num{8.2}}",
        "\\newcommand{\\ResultExtGsdMax}{\\num{2.46}}", "\\newcommand{\\ResultExtFallbackPairs}{2}", "\\newcommand{\\ResultExtGroundFractionRescuenet}{\\num{0.376}}",
        "\\newcommand{\\ResultExtTraceScenario}{\\texttt{Agis-r0}}", "\\newcommand{\\ResultExtComputeHours}{\\num{24.5}}", "\\newcommand{\\ResultExtCvxpyVersion}{1.9.2}",
    ):
        assert expected in macros, expected


def test_extension_report_tables_refuse_results_from_other_source_code(tmp_path: Path, capsys):
    phase2, phase1, manifests = _results(tmp_path, source_hash())
    manifest = json.loads((phase2 / "case_study" / "manifest.json").read_text(encoding="utf-8"))
    (phase2 / "case_study" / "manifest.json").write_text(json.dumps({**manifest, "source_hash": "0" * 64}), encoding="utf-8")
    output = tmp_path / "data"
    assert report.main(["--results", str(phase2), "--phase1", str(phase1), "--manifests", str(manifests), "--output", str(output)]) == 1
    assert "case_study: the results were produced by different source code" in capsys.readouterr().err and not output.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/runner/test_report_tables_extensions.py`

Expected: collection fails (`1 error`) with `FileNotFoundError: [Errno 2] No such file or directory: '<repo>/experiments/report_tables_extensions.py'`.

- [ ] **Step 3: Implement**

Create `experiments/report_tables_extensions.py`:

```python
"""Write report CSVs and result macros for the literature baseline, the RescueNet case study and the compute summary (spec D Section 13)."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

from report_tables import _read_rows, _write_semicolon, kbit, macro_name
from report_tables_phase2 import SETS, compute_minutes, number, write_points
from runner.provenance import result_provenance_problem

LITERATURE_METHODS = ("B1", "B2", "P", "TRAN")
CASE_METHODS = ("B0", "B1", "B2", "P", "TRAN")
TRACE_METHODS = ("B1", "P", "TRAN")
COMPARISON_LABELS = {"P - TRAN": "P $-$ TRAN", "B2 - TRAN": "B2 $-$ TRAN"}
COMPUTE_EXPERIMENTS = (
    ("phase1", "Pha 1: B0, B1, cận LP và MILP"),
    ("phase2/main", "So sánh chính và ablation"),
    ("phase2/budget", "Ngân sách đánh giá"),
    ("phase2/sensitivity", "Độ nhạy"),
    ("phase2/design", "Số realization thiết kế"),
    ("phase2/literature", "So sánh với TRAN"),
    ("phase2/case_study", "Case study RescueNet"),
)


def macro(*parts: str) -> str:
    return "ResultExt" + macro_name(*parts)[len("Result"):]


def km(value: str | float) -> str:
    return f"{float(value) / 1000.0:.4f}"


def provenance_problems(roots: list[Path]) -> list[str]:
    problems = []
    for root in roots:
        path = root / "manifest.json"
        if not path.exists():
            problems.append(f"missing {path}")
            continue
        problem = result_provenance_problem(json.loads(path.read_text(encoding="utf-8")))
        if problem:
            problems.append(f"{root}: {problem}")
    return problems


def edge_polyline(edges: list[dict[str, str]], failed: bool) -> list[list[str]]:
    """Edge segments separated by nan rows, so pgfplots draws them with unbounded coords=jump."""
    points: list[list[str]] = []
    for edge in edges:
        if (edge["failed"] == "1") == failed:
            points += [[km(edge["source_x_m"]), km(edge["source_y_m"])], [km(edge["target_x_m"]), km(edge["target_y_m"])], ["nan", "nan"]]
    return points or [["nan", "nan"]]


def pilot_jobs(runs: list[dict[str, str]]) -> dict[tuple[str, str, str, str], float]:
    """Planning time of each distinct pilot job; the grid reuses step-1 runs as rows with step grid."""
    jobs: dict[tuple[str, str, str, str], float] = {}
    for row in runs:
        jobs.setdefault((row["scenario_id"], row["theta"], row["mu"], row["block_slots"]), float(row["planning_runtime_s"]))
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="report-tables-extensions")
    parser.add_argument("--results", type=Path, default=Path("results/phase2"))
    parser.add_argument("--phase1", type=Path, default=Path("results/phase1"))
    parser.add_argument("--manifests", type=Path, default=Path("data/manifests"))
    parser.add_argument("--output", type=Path, default=Path("docs/report/data"))
    args = parser.parse_args(argv)

    roots = {name: args.results / name.split("/", 1)[1] if name.startswith("phase2/") else args.phase1 for name, _ in COMPUTE_EXPERIMENTS}
    problems = provenance_problems(list(roots.values()))
    if not (args.results / "tran_pilot" / "choice.json").exists():
        problems.append(f"missing {args.results / 'tran_pilot' / 'choice.json'}")
    if problems:
        print("\n".join(problems) + "\nrerun or refresh these experiments before writing report data", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    macros: list[tuple[str, str]] = []
    summary = {
        name: {(row["set_id"], row["method"]): row for row in _read_rows(args.results / name / "summary.csv")}
        for name in ("main", "literature", "case_study")
    }

    literature_rows = []
    for set_id, capacity in SETS:
        for method in LITERATURE_METHODS:
            row = summary["literature" if method == "TRAN" else "main"][(set_id, method)]
            literature_rows.append([
                set_id, kbit(capacity), method, number(row["timely_ratio_mean"]),
                f"[{number(row['timely_ratio_ci_low'])}, {number(row['timely_ratio_ci_high'])}]", number(row["gap_mean"]),
                number(row["planning_runtime_s_mean"], 1),
            ])
            for metric, column, digits in (("Timely", "timely_ratio_mean", 3), ("Gap", "gap_mean", 3), ("Planning", "planning_runtime_s_mean", 1), ("Conn", "conn_ratio_mean", 3)):
                macros.append((macro(metric, set_id, method), number(row[column], digits)))
    _write_semicolon(args.output / "ext_literature.csv", ["set", "backhaul", "method", "timely", "ci", "gap", "planning"], literature_rows)

    statistics_rows = []
    for row in _read_rows(args.results / "statistics_literature.csv"):
        statistics_rows.append([
            row["set_id"], COMPARISON_LABELS[row["comparison"]], number(row["mean_difference"]), f"[{number(row['ci_low'])}, {number(row['ci_high'])}]",
            number(row["rank_biserial"], 2), number(row["p_value"], 4), number(row["p_holm"], 4),
        ])
        first = row["comparison"].split(" - ")[0]
        for metric, column, digits in (("Diff", "mean_difference", 3), ("DiffLow", "ci_low", 3), ("DiffHigh", "ci_high", 3), ("PValue", "p_value", 4), ("Holm", "p_holm", 4), ("RankBiserial", "rank_biserial", 2)):
            macros.append((macro(metric, row["set_id"], first, "tran"), number(row[column], digits)))
    _write_semicolon(args.output / "ext_literature_statistics.csv", ["set", "comparison", "diff", "ci", "rankbiserial", "p", "pholm"], statistics_rows)

    quality = [["B1", summary["main"][("v0", "B1")]], ["TRAN", summary["literature"][("v0", "TRAN")]]]
    write_points(args.output / "ext_quality_points.csv", ["method", "planning", "timely"],
                 [[method, f"{float(row['planning_runtime_s_mean']):.3f}", f"{float(row['timely_ratio_mean']):.6f}"] for method, row in quality])
    literature_bounds = _read_rows(args.results / "literature" / "bounds.csv")
    environment = json.loads((args.results / "literature" / "manifest.json").read_text(encoding="utf-8"))["environment"]
    macros += [
        ("ResultExtLiteratureLpCount", str(len(literature_bounds))),
        ("ResultExtLiteratureLpOptimalCount", str(sum(row["status"] == "optimal" for row in literature_bounds))),
        ("ResultExtCvxpyVersion", environment["cvxpy"]),
        ("ResultExtClarabelVersion", environment["clarabel"]),
    ]

    choice = json.loads((args.results / "tran_pilot" / "choice.json").read_text(encoding="utf-8"))
    runs = _read_rows(args.results / "tran_pilot" / "runs.csv")
    grid: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in runs:
        if row["step"] == "grid":
            grid[(float(row["theta"]), float(row["mu"]))].append(float(row["timely_ratio"]))
    grid_means = [statistics.fmean(values) for values in grid.values()]
    fits = _read_rows(args.results / "tran_pilot" / "channel_fit.csv")
    macros += [
        ("ResultExtPilotTheta", number(choice["theta"])),
        ("ResultExtPilotMu", number(choice["mu"], 0)),
        ("ResultExtPilotBlockSlots", str(choice["block_slots"])),
        ("ResultExtPilotMedianRuntime", number(choice["median_runtime_step1_s"], 1)),
        ("ResultExtPilotScenarioCount", str(len(choice["scenarios"]))),
        ("ResultExtPilotBetterCount", str(choice["better_than_initial_count"])),
        ("ResultExtPilotTimely", number(choice["mean_timely_ratio"])),
        ("ResultExtPilotGridSpread", number(max(grid_means) - min(grid_means))),
        ("ResultExtFitAlphaMin", number(min(float(row["alpha"]) for row in fits), 2)),
        ("ResultExtFitAlphaMax", number(max(float(row["alpha"]) for row in fits), 2)),
        ("ResultExtFitRsquaredMin", number(min(float(row["r_squared"]) for row in fits), 4)),
    ]

    case_rows = []
    for method in CASE_METHODS:
        row = summary["case_study"][("rescuenet", method)]
        case_rows.append([
            method, number(row["timely_ratio_mean"]), f"[{number(row['timely_ratio_ci_low'])}, {number(row['timely_ratio_ci_high'])}]",
            number(row["conn_ratio_mean"]), number(row["gap_mean"]), number(row["planning_runtime_s_mean"], 1),
        ])
        for metric, column, digits in (("Timely", "timely_ratio_mean", 3), ("TimelyLow", "timely_ratio_ci_low", 3), ("TimelyHigh", "timely_ratio_ci_high", 3), ("Conn", "conn_ratio_mean", 3), ("Gap", "gap_mean", 3), ("Planning", "planning_runtime_s_mean", 1)):
            macros.append((macro("Case", metric, method), number(row[column], digits)))
    _write_semicolon(args.output / "ext_case_study.csv", ["method", "timely", "ci", "conn", "gap", "planning"], case_rows)
    case_bounds = _read_rows(args.results / "case_study" / "bounds.csv")
    macros += [
        ("ResultExtCaseLpCount", str(len(case_bounds))),
        ("ResultExtCaseLpOptimalCount", str(sum(row["status"] == "optimal" for row in case_bounds))),
        ("ResultExtCaseScenarioCount", summary["case_study"][("rescuenet", "P")]["scenario_count"]),
    ]

    targets = json.loads((args.manifests / "rescuenet_targets.json").read_text(encoding="utf-8"))
    distances = [pair["distance_m"] for pair in targets["merged_pairs"]]
    gsd_cm = [image["gsd_m_per_px"] * 100.0 for image in targets["images"]]
    fractions = {}
    for set_id in ("rescuenet", "v0"):
        rows = [row for row in _read_rows(args.manifests / f"scenarios_{set_id}.csv") if row["split"] == "eval"]
        fractions[set_id] = statistics.fmean(float(row["ground_connected_source_fraction"]) for row in rows)
    macros += [
        ("ResultExtImageCount", str(len(targets["images"]))),
        ("ResultExtImagesWithTargets", str(sum(image["targets"] > 0 for image in targets["images"]))),
        ("ResultExtRegionCount", str(targets["region_count"])),
        ("ResultExtTargetCount", str(targets["target_count"])),
        ("ResultExtDroppedCount", str(len(targets["dropped_regions"]))),
        ("ResultExtMergedCount", str(len(distances))),
        ("ResultExtMergeDistanceMin", number(min(distances), 1) if distances else "---"),
        ("ResultExtMergeDistanceMax", number(max(distances), 1) if distances else "---"),
        ("ResultExtGsdMin", number(min(gsd_cm), 2)),
        ("ResultExtGsdMax", number(max(gsd_cm), 2)),
        ("ResultExtFallbackPairs", ", ".join(image["pair_id"] for image in targets["images"] if image["focal_source"] == "fallback") or "---"),
        ("ResultExtNormalizationScale", number(targets["normalization"]["scale"])),
        ("ResultExtGroundFractionRescuenet", number(fractions["rescuenet"])),
        ("ResultExtGroundFractionVzero", number(fractions["v0"])),
    ]

    trace = args.results / "case_study" / "trace"
    trace_summary = json.loads((trace / "trace.json").read_text(encoding="utf-8"))
    scenario_id = trace_summary["scenario_id"]
    macros.append(("ResultExtTraceScenario", f"\\texttt{{{scenario_id}}}"))
    for method in TRACE_METHODS:
        macros.append((macro("Trace", method), number(trace_summary["methods"][method]["timely_ratio_trace"])))
    write_points(args.output / "ext_map_targets.csv", ["x", "y", "class"], [[km(row["x_m"]), km(row["y_m"]), row["damage_class"]] for row in _read_rows(trace / "targets.csv")])
    nodes = _read_rows(trace / "nodes.csv")
    write_points(args.output / "ext_map_relays.csv", ["x", "y"], [[km(row["x_m"]), km(row["y_m"])] for row in nodes if row["role"] == "relay"])
    write_points(args.output / "ext_map_center.csv", ["x", "y"], [[km(row["x_m"]), km(row["y_m"])] for row in nodes if row["role"] == "center"])
    edges = _read_rows(trace / "edges.csv")
    write_points(args.output / "ext_map_edges_alive.csv", ["x", "y"], edge_polyline(edges, failed=False))
    write_points(args.output / "ext_map_edges_failed.csv", ["x", "y"], edge_polyline(edges, failed=True))
    trajectories: dict[str, list[list[str]]] = defaultdict(list)
    for row in _read_rows(trace / "trajectories.csv"):
        trajectories[row["method"]].append([km(row["x_m"]), km(row["y_m"])])
    for method in TRACE_METHODS:
        write_points(args.output / f"ext_map_trajectory_{method.lower()}.csv", ["x", "y"], trajectories[method])
    alerts = int(next(row["alert_count"] for row in _read_rows(args.results / "case_study" / "method_realizations.csv") if row["scenario_id"] == scenario_id))
    progress: dict[str, dict[int, float]] = defaultdict(dict)
    for row in _read_rows(trace / "progress.csv"):
        progress[row["method"]][int(row["slot"])] = float(row["mean_timely_alerts"]) / alerts
    slots = sorted(progress[TRACE_METHODS[0]])
    write_points(args.output / "ext_progress.csv", ["slot", *TRACE_METHODS], [[slot, *(f"{progress[method][slot]:.6f}" for method in TRACE_METHODS)] for slot in slots])
    macros += [("ResultExtTraceAlertCount", str(alerts)), ("ResultExtTraceSlotCount", str(len(slots)))]

    compute_rows, total_hours = [], 0.0
    for name, label in COMPUTE_EXPERIMENTS:
        root = roots[name]
        minutes = compute_minutes(root)
        if (root / "milp_crosscheck.csv").exists():
            minutes += sum(float(row["milp_runtime_s"]) for row in _read_rows(root / "milp_crosscheck.csv")) / 60.0
        tasks = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["task_count"]
        compute_rows.append([label, str(tasks), number(minutes / 60.0, 1)])
        total_hours += minutes / 60.0
    jobs = pilot_jobs(runs)
    pilot_hours = sum(jobs.values()) / 3600.0
    compute_rows.insert(5, ["Pilot TRAN trên tập phát triển", str(len(jobs)), number(pilot_hours, 1)])
    total_hours += pilot_hours
    _write_semicolon(args.output / "ext_compute.csv", ["experiment", "tasks", "hours"], compute_rows)
    macros.append(("ResultExtComputeHours", number(total_hours, 1)))

    lines = ["% Generated by experiments/report_tables_extensions.py from results/phase1, results/phase2 and data/manifests; do not edit."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros]
    (args.output / "ext_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"literature": len(literature_rows), "statistics": len(statistics_rows), "case_study": len(case_rows), "compute": len(compute_rows), "macros": len(macros)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/runner/test_report_tables_extensions.py`

Expected: `2 passed`.

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `283 passed, 1 skipped`.

- [ ] **Step 6: Generate the report data**

Run:

```bash
uv run python experiments/report_tables_extensions.py
git status --short --untracked-files=all docs/report/data
```

Expected: the script prints `{"literature": 8, "statistics": 4, "case_study": 5, "compute": 8, "macros": 125}` and `git status` lists only the new files:

```text
?? docs/report/data/ext_case_study.csv
?? docs/report/data/ext_compute.csv
?? docs/report/data/ext_literature.csv
?? docs/report/data/ext_literature_statistics.csv
?? docs/report/data/ext_map_center.csv
?? docs/report/data/ext_map_edges_alive.csv
?? docs/report/data/ext_map_edges_failed.csv
?? docs/report/data/ext_map_relays.csv
?? docs/report/data/ext_map_targets.csv
?? docs/report/data/ext_map_trajectory_b1.csv
?? docs/report/data/ext_map_trajectory_p.csv
?? docs/report/data/ext_map_trajectory_tran.csv
?? docs/report/data/ext_progress.csv
?? docs/report/data/ext_quality_points.csv
?? docs/report/data/ext_results.tex
```

- [ ] **Step 7: Commit**

```bash
git add tests/runner/test_report_tables_extensions.py experiments/report_tables_extensions.py docs/report/data/ext_case_study.csv docs/report/data/ext_compute.csv docs/report/data/ext_literature.csv docs/report/data/ext_literature_statistics.csv docs/report/data/ext_map_center.csv docs/report/data/ext_map_edges_alive.csv docs/report/data/ext_map_edges_failed.csv docs/report/data/ext_map_relays.csv docs/report/data/ext_map_targets.csv docs/report/data/ext_map_trajectory_b1.csv docs/report/data/ext_map_trajectory_p.csv docs/report/data/ext_map_trajectory_tran.csv docs/report/data/ext_progress.csv docs/report/data/ext_quality_points.csv docs/report/data/ext_results.tex
git commit -m "feat: write report data for TRAN, the case study and compute time"
```

### Task 5: Report method: SAA, TRAN and the RescueNet scenarios

**Files:**
- Create: `docs/report/tables/tran_adaptation.tex`
- Modify: `docs/report/main.tex`, `docs/report/preamble.tex`, `docs/report/sections/method.tex`

**Interfaces:**
- Consumes: `docs/report/data/ext_results.tex` (Task 4); spec D Sections 5, 8, 9 and 11; the report sections of sub-projects B and C.
- Produces: Labels `sec:tran`, `sec:rescuenet`, `tab:tran-adaptation`; the SAA paragraph in the search-method section; `main.tex` inputs `data/ext_results`; `preamble.tex` defines the Vietnamese `cleveref` conjunctions and the name "Phụ lục" for appendices.

`cleveref` has no Vietnamese conjunctions, so without the six conjunction commands it prints English "and" between references (already visible in the introduction before this plan). `\renewcommand` fails for them in the preamble with `Command ... undefined`, so they are defined with `\newcommand`.

- [ ] **Step 1: Edit the report sources**

In `docs/report/main.tex` (1 edit):

Edit 1: replace

```latex
\addbibresource{references.bib}
```

with

```latex
\input{data/ext_results}     % macro số liệu TRAN, case study và thời gian tính, sinh bởi experiments/report_tables_extensions.py
\addbibresource{references.bib}
```


In `docs/report/preamble.tex` (1 edit):

Edit 1: replace

```latex
% cleveref chưa có tiếng Việt, đặt tên bằng tay
```

with

```latex
% cleveref chưa có tiếng Việt, đặt tên và từ nối bằng tay
\newcommand{\crefpairconjunction}{ và }
\newcommand{\crefmiddleconjunction}{, }
\newcommand{\creflastconjunction}{ và }
\newcommand{\crefpairgroupconjunction}{ và }
\newcommand{\crefmiddlegroupconjunction}{, }
\newcommand{\creflastgroupconjunction}{ và }
\crefname{appendix}{Phụ lục}{Phụ lục}
\Crefname{appendix}{Phụ lục}{Phụ lục}
```


In `docs/report/sections/method.tex` (2 edits):

Edit 1: replace

```latex
\paragraph{Ba khối của B2.}
```

with

```latex
\paragraph{Chấm điểm như một xấp xỉ trung bình mẫu.}
Mục tiêu thực của bài toán là kỳ vọng của khoá từ điển theo phân phối kênh. Cộng khoá trên $M_d$ realization thiết kế là một xấp xỉ trung bình mẫu (sample average approximation, SAA) của kỳ vọng đó với cỡ mẫu $M_d=5$, còn chấm trên tốc độ kỳ vọng là bài toán tương đương tất định. Vì realization thiết kế và realization đánh giá lấy từ hai luồng số ngẫu nhiên khác nhau, tỷ lệ đúng hạn ở \cref{sec:experiments} là ước lượng ngoài mẫu. Thí nghiệm số realization thiết kế ở \cref{sec:results-budget} là khảo sát cỡ mẫu của SAA này; báo cáo không thêm phương pháp SAA nào khác.

\paragraph{Ba khối của B2.}
```

Edit 2: replace

```latex
Một vòng lặp dùng tối đa $(K+1)\abs{A}+2\abs{\mathcal L_{\mathrm{used}}}+90$ lần đánh giá cho ba khối, cộng $1+RK$ lần cho vòng sửa lịch của P, với $\mathcal L_{\mathrm{used}}$ là các liên kết vô tuyến đang được dùng. Mỗi lần đánh giá tốn $O\bigl(M_dN(\abs{\mathcal L}+\abs{A}\log\abs{A})\bigr)$, còn phát hiện nguy cơ duyệt sổ ghi trong $O\bigl(M_d(N\abs{\mathcal L}+T)\bigr)$ với $T$ là số bản ghi truyền. Ngân sách chặn tổng số lần đánh giá, nên thời gian lập kế hoạch của B2, B3 và P là $O\bigl(EM_dN(\abs{\mathcal L}+\abs{A}\log\abs{A})\bigr)$, không phụ thuộc số vòng lặp thực tế. Phương pháp chỉ bảo đảm khoá của kế hoạch không giảm; nó không bảo đảm tối ưu, và cận LP ở \cref{sec:bound}, giải trên chính quỹ đạo mà mỗi phương pháp chọn, đo phần còn có thể cải thiện.
```

with

```latex
Một vòng lặp dùng tối đa $(K+1)\abs{A}+2\abs{\mathcal L_{\mathrm{used}}}+90$ lần đánh giá cho ba khối, cộng $1+RK$ lần cho vòng sửa lịch của P, với $\mathcal L_{\mathrm{used}}$ là các liên kết vô tuyến đang được dùng. Mỗi lần đánh giá tốn $O\bigl(M_dN(\abs{\mathcal L}+\abs{A}\log\abs{A})\bigr)$, còn phát hiện nguy cơ duyệt sổ ghi trong $O\bigl(M_d(N\abs{\mathcal L}+T)\bigr)$ với $T$ là số bản ghi truyền. Ngân sách chặn tổng số lần đánh giá, nên thời gian lập kế hoạch của B2, B3 và P là $O\bigl(EM_dN(\abs{\mathcal L}+\abs{A}\log\abs{A})\bigr)$, không phụ thuộc số vòng lặp thực tế. Phương pháp chỉ bảo đảm khoá của kế hoạch không giảm; nó không bảo đảm tối ưu, và cận LP ở \cref{sec:bound}, giải trên chính quỹ đạo mà mỗi phương pháp chọn, đo phần còn có thể cải thiện.

\subsection{Phương pháp từ tài liệu: TRAN}
\label{sec:tran}

Phương pháp mạnh từ tài liệu là thuật toán xấp xỉ trong (inner approximation) của Tran và cộng sự~\cite{tran2022relay} cho chế độ bán song công, gọi là TRAN. Bài gốc tối đa số thiết bị được phục vụ: thiết bị $k$ tải $S_k$ bit lên UAV trong cửa sổ thu, UAV lưu dữ liệu rồi chuyển xuống gateway trong cửa sổ sau đó, và biến nhị phân $\lambda_k$ bằng một khi cả hai chặng đủ dữ liệu. Biến $\lambda_k$ được nới lỏng về $[0,1]$ kèm hàm phạt $\mu\sum_k\lambda_k(\lambda_k-1)$; mỗi vòng lặp tuyến tính hoá hàm phạt, thay tốc độ bằng cận dưới lõm quanh điểm hiện tại và giải một bài toán lồi theo quỹ đạo, băng thông và $\lambda$. \Cref{tab:tran-adaptation} liệt kê các điểm chuyển thể sang mô hình ở \cref{sec:model}. Các điểm này giữ nguyên cấu trúc thuật toán nhưng có thể làm phương pháp bất lợi, vì bài gốc không mô hình backhaul hữu hạn hay bộ điều phối EDF.

\input{tables/tran_adaptation}

Mỗi cảnh báo là một thiết bị và đi theo candidate có chi phí ước lượng nhỏ nhất; nút vào mạng mặt đất của candidate đó là gateway riêng của cảnh báo. Với $b_a$ slot trễ backhaul và $W_a=d_a-b_a-r_a$, cửa sổ thu là $[r_a,\,r_a+\lceil\theta W_a\rceil)$ và cửa sổ chuyển xuống là $[r_a+\lceil\theta W_a\rceil,\,d_a-b_a)$, để dữ liệu còn kịp qua backhaul trước hạn đầu--cuối. Cảnh báo đi đường mặt đất chỉ có chặng thu với tốc độ không phụ thuộc quỹ đạo. Công suất cố định như mọi phương pháp khác. Kênh Rician của bài gốc được thay bằng luật công suất $\beta d^{-\alpha}$ khớp bình phương tối thiểu với độ lợi kỳ vọng của mô hình PrLoS; trên các scenario phát triển, số mũ khớp nằm trong khoảng \ResultExtFitAlphaMin{}--\ResultExtFitAlphaMax{} với $R^2\ge\ResultExtFitRsquaredMin$.

Mỗi vòng lặp dựng lại bài toán nón bậc hai với điểm tuyến tính hoá là hằng số, xấp xỉ ràng buộc luỹ thừa bằng các nón bậc hai của \texttt{cvxpy} và giải bằng Clarabel với dung sai khoảng cách đối ngẫu $10^{-6}$ và dung sai khả thi $10^{-7}$. Thuật toán khởi tạo từ quỹ đạo của B1, băng thông chia đều và $\lambda=0$, là một điểm khả thi của bài toán lồi, và dừng khi mục tiêu nới lỏng thay đổi tương đối dưới $10^{-3}$ hoặc sau 30 vòng. Nghiệm được đổi thành lịch băng thông tĩnh, và mỗi slot chỉ giữ tối đa hai downlink có băng thông lớn nhất. Tham số được chọn trên \ResultExtPilotScenarioCount{} scenario phát triển từ lưới $\theta\in\{1/3,1/2,2/3\}$ và $\mu\in\{1,10\}$: cặp $\theta=\ResultExtPilotTheta$, $\mu=\ResultExtPilotMu$ đạt tỷ lệ đúng hạn trung bình cao nhất (\ResultExtPilotTimely) dù các ô lưới chỉ chênh nhau tối đa \ResultExtPilotGridSpread, và TRAN tốt hơn điểm khởi tạo trên cả \ResultExtPilotBetterCount{} scenario. Theo bài gốc, mỗi vòng lặp giải một bài toán lồi có $O(NK)$ biến và ràng buộc với $K$ thiết bị, nên một vòng tốn $O\bigl((NK)^{3.5}\bigr)$ bằng phương pháp điểm trong; dùng chung biến khoảng cách cho các cảnh báo cùng điểm mặt đất làm số biến thực tế nhỏ hơn nhiều.

\subsection{Kịch bản case study từ RescueNet}
\label{sec:rescuenet}

Case study thay quy tắc sinh điểm nguồn bằng vị trí công trình hư hại trong bộ dữ liệu RescueNet~\cite{rahnemoonfar2023rescuenet}, gồm ảnh UAV chụp sau bão Michael tại Mexico Beach kèm mask phân vùng. Trên \ResultExtImageCount{} cặp ảnh--mask đã chọn, các vùng liên thông tám hướng của ba lớp hư hại (nhẹ, nặng, sập hoàn toàn) có diện tích dưới \qty{10}{\metre\squared} bị loại; điểm đại diện là điểm ảnh của vùng gần tâm vùng nhất, nên luôn nằm trên công trình. Bài báo không cho độ phân giải mặt đất, nhưng metadata của từng ảnh có toạ độ GPS, độ cao tương đối, hướng gimbal và tiêu cự hiệu chuẩn. Kích thước một điểm ảnh bằng độ cao chia tiêu cự tính theo điểm ảnh, từ \ResultExtGsdMin{} đến \ResultExtGsdMax~\unit{\centi\metre}; ảnh duy nhất thiếu tiêu cự hiệu chuẩn (cặp \ResultExtFallbackPairs) dùng tiêu cự danh định của camera. Mỗi điểm đại diện được chiếu sang hệ UTM vùng 16N theo toạ độ tâm ảnh và hướng gimbal. Các ảnh chồng lên nhau, nên một công trình có thể xuất hiện ở hai ảnh với vị trí lệch nhau \ResultExtMergeDistanceMin{}--\ResultExtMergeDistanceMax~\unit{\metre}; các điểm cách nhau không quá \qty{10}{\metre} được gộp và giữ điểm của vùng lớn nhất. Từ \ResultExtRegionCount{} vùng hư hại thu được \ResultExtTargetCount{} target, được chuẩn hoá cùng tỷ lệ trên hai trục về vùng \qty{2}{\kilo\metre} (hệ số \ResultExtNormalizationScale).

Họ kịch bản \texttt{rescuenet} giống v0 ở mọi quy tắc khác: cùng mười topology đánh giá và ba replicate, cùng luồng số ngẫu nhiên cho cảnh báo và cạnh hỏng, nhưng mỗi target là một nguồn và ba vùng hư hại là ba cụm k-means của các target. Target trải khắp vùng thay vì tụ quanh ba tâm, nên tỷ lệ nguồn có relay mặt đất trong tầm của các kịch bản đánh giá giảm từ \ResultExtGroundFractionVzero{} ở v0 xuống \ResultExtGroundFractionRescuenet. Bố cục target là một bố cục địa lý duy nhất, lấy từ các ảnh không độc lập với nhau, nên kết quả case study mang tính mô tả. Ảnh và mask không được phân phối lại; kho mã chỉ lưu manifest target.
```


Create `docs/report/tables/tran_adaptation.tex`:

```latex
% Chuyển thể TRAN — bảng viết tay (spec D Mục 5.1).
\begin{table}[htbp]
\centering
\small
\caption{Chuyển thể thuật toán bán song công của Tran và cộng sự~\cite{tran2022relay} sang mô hình của đề tài.}
\label{tab:tran-adaptation}
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.38\textwidth}>{\raggedright\arraybackslash}X@{}}
\toprule
\textbf{Bài gốc} & \textbf{Trong đề tài} \\
\midrule
Thiết bị $k$ có $S_k$ bit, thời điểm bắt đầu và hạn thu & Cảnh báo $a$ có $L_a$ bit, thời điểm $r_a$ và cửa sổ thu $[r_a,\,r_a+\lceil\theta W_a\rceil)$ \\
Một gateway; chuyển xuống từ sau hạn thu tới hết nhiệm vụ & Gateway là nút vào của candidate có chi phí nhỏ nhất; chuyển xuống trong $[r_a+\lceil\theta W_a\rceil,\,d_a-b_a)$ \\
Tối ưu công suất của thiết bị và UAV & Công suất cố định như mọi phương pháp \\
Kênh Rician với suy hao $\omega_0 d^{-\alpha}$ & Luật công suất $\beta d^{-\alpha}$ khớp với độ lợi kỳ vọng PrLoS \\
Bộ nhớ UAV hữu hạn & Không có, vì bộ đệm UAV không giới hạn trong mô hình \\
Không giới hạn số downlink đồng thời & Mỗi slot giữ tối đa hai downlink có băng thông lớn nhất \\
Bài toán tìm điểm khả thi để khởi tạo & Khởi tạo từ quỹ đạo B1, băng thông chia đều và $\lambda=0$, luôn khả thi \\
Không cho giá trị $\mu$ và ngưỡng hội tụ & $\theta$ và $\mu$ chọn trên tập phát triển; dừng khi mục tiêu đổi dưới $10^{-3}$ hoặc sau 30 vòng \\
\bottomrule
\end{tabularx}
\end{table}
```

- [ ] **Step 2: Compile a copy outside the workspace and check it**

The editor rebuilds `docs/report` when a `.tex` file changes, so compile a copy in a temporary directory. From the repository root:

```bash
REPO=$(git rev-parse --show-toplevel)
CHECK=$(mktemp -d)
cp -r docs/report "$CHECK/report"
rm -rf "$CHECK/report/build" "$CHECK/report/main.pdf"
mkdir "$CHECK/report/build"
(cd "$CHECK/report" && bash "$REPO/.claude/skills/latex-document-skill/scripts/compile_latex.sh" main.tex --verbose > build/compile-verbose.txt 2>&1)
uv run python - "$CHECK/report" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

report = Path(sys.argv[1])
log = (report / "build" / "compile-verbose.txt").read_text(encoding="utf-8", errors="replace")
final_pass = log.split("This is pdfTeX")[-1]
issues = re.findall(r"^.*(?:Overfull|Underfull|LaTeX Warning|Package \w+ Warning|^! ).*$", final_pass, re.M)
info = subprocess.run(["pdfinfo", str(report / "main.pdf")], capture_output=True, text=True).stdout
text = subprocess.run(["pdftotext", str(report / "main.pdf"), "-"], capture_output=True, text=True).stdout
pages = re.search(r"Pages:\s+(\d+)", info).group(1) if info else "none"
english = re.findall(r"\d\s*,?\s+and\s+\d|\b[Aa]ppendix\b", " ".join(text.split()))
print(f"final pass issues: {len(issues)}; pages: {pages}; unresolved ??: {text.count('??')}; English cross-reference words: {len(english)}")
for line in issues[:5]:
    print(line)
PY
```

Expected: exactly one line, `final pass issues: 0; pages: 25; unresolved ??: 0; English cross-reference words: 0`. Any further line is a warning or error of the final pass; fix it in `docs/report` and repeat this step.

- [ ] **Step 3: Commit**

```bash
git add docs/report/main.tex docs/report/preamble.tex docs/report/sections/method.tex docs/report/tables/tran_adaptation.tex
git commit -m "docs: describe SAA, TRAN and the RescueNet scenarios in the report"
```

### Task 6: Report results: TRAN comparison and case study

**Files:**
- Create: `docs/report/tables/ext_literature.tex`, `docs/report/tables/ext_literature_statistics.tex`, `docs/report/tables/ext_case_study.tex`
- Modify: `docs/report/sections/experiments.tex`

**Interfaces:**
- Consumes: Task 4: `ext_literature.csv`, `ext_literature_statistics.csv`, `ext_quality_points.csv`, `ext_case_study.csv`, `ext_map_*.csv`, `ext_progress.csv` and the `\ResultExt...` macros; the existing `phase2_budget_*.csv`; Task 5 labels `sec:tran` and `sec:rescuenet`.
- Produces: Labels `sec:results-literature`, `tab:literature`, `tab:literature-stats`, `fig:quality-runtime`, `sec:results-case-study`, `tab:case-study`, `fig:case-map`, `fig:case-progress`; research-question answers with the final results.

`fig:quality-runtime` is a separate figure with the budget curves and the TRAN points of `ext_quality_points.csv`, so the budget figure of sub-project C stays unchanged.

- [ ] **Step 1: Edit the report sources**

In `docs/report/sections/experiments.tex` (4 edits):

Edit 1: replace

```latex

\input{tables/params}
```

with

```latex

Thí nghiệm so sánh với phương pháp từ tài liệu chạy B1 và TRAN trên cùng scenario, realization và loại cận của thí nghiệm chính. B1 của lần chạy này trùng từng realization với B1 của thí nghiệm chính, nên TRAN được ghép cặp trực tiếp với kết quả đã có của B2 và P; bốn so sánh P $-$ TRAN và B2 $-$ TRAN trên hai họ dùng cùng kiểm định đổi dấu, hiệu chỉnh Holm trên bốn so sánh. Case study chạy B0, B1, B2, P và TRAN trên \ResultExtCaseScenarioCount{} scenario đánh giá của họ \texttt{rescuenet} và chỉ được báo cáo mô tả. Bài toán TRAN được giải bằng \texttt{cvxpy}~\ResultExtCvxpyVersion{} với Clarabel~\ResultExtClarabelVersion; \ResultExtLiteratureLpOptimalCount{} trong \ResultExtLiteratureLpCount{} bài LP của thí nghiệm so sánh và \ResultExtCaseLpOptimalCount{} trong \ResultExtCaseLpCount{} bài LP của case study được giải tối ưu.

\input{tables/params}
```

Edit 2: replace

```latex

\Cref{tab:phase2-stats} cho thấy B2 hơn B1 trung bình \ResultTwoDiffVzeroBtwoBone{} (KTC 95\% [\ResultTwoDiffLowVzeroBtwoBone, \ResultTwoDiffHighVzeroBtwoBone]) trên v0 và \ResultTwoDiffVzeroBhfivezeroBtwoBone{} trên v0-bh50, còn P hơn B1 \ResultTwoDiffVzeroPBone{} và \ResultTwoDiffVzeroBhfivezeroPBone. Các so sánh này và B2 $-$ B3 đều có $p_{\mathrm{Holm}}=\ResultTwoHolmFloor$, giá trị nhỏ nhất có thể với mười topology và tám so sánh ($8\cdot 2/2^{10}$): cả mười topology cùng chiều, và tương quan hạng biserial bằng một. Ngược lại, P $-$ B2 có chênh lệch \ResultTwoDiffVzeroPBtwo{} (KTC 95\% [\ResultTwoDiffLowVzeroPBtwo, \ResultTwoDiffHighVzeroPBtwo]) trên v0 và \ResultTwoDiffVzeroBhfivezeroPBtwo{} trên v0-bh50, với $p_{\mathrm{Holm}}=\ResultTwoHolmVzeroPBtwo$: dữ liệu không cho thấy cơ chế sửa lịch làm tăng tỷ lệ đúng hạn so với B2.

\subsection{Ablation}
```

with

```latex

\Cref{tab:phase2-stats} cho thấy B2 hơn B1 trung bình \ResultTwoDiffVzeroBtwoBone{} (KTC 95\% [\ResultTwoDiffLowVzeroBtwoBone, \ResultTwoDiffHighVzeroBtwoBone]) trên v0 và \ResultTwoDiffVzeroBhfivezeroBtwoBone{} trên v0-bh50, còn P hơn B1 \ResultTwoDiffVzeroPBone{} và \ResultTwoDiffVzeroBhfivezeroPBone. Các so sánh này và B2 $-$ B3 đều có $p_{\mathrm{Holm}}=\ResultTwoHolmFloor$, giá trị nhỏ nhất có thể với mười topology và tám so sánh ($8\cdot 2/2^{10}$): cả mười topology cùng chiều, và tương quan hạng biserial bằng một. Ngược lại, P $-$ B2 có chênh lệch \ResultTwoDiffVzeroPBtwo{} (KTC 95\% [\ResultTwoDiffLowVzeroPBtwo, \ResultTwoDiffHighVzeroPBtwo]) trên v0 và \ResultTwoDiffVzeroBhfivezeroPBtwo{} trên v0-bh50, với $p_{\mathrm{Holm}}=\ResultTwoHolmVzeroPBtwo$: dữ liệu không cho thấy cơ chế sửa lịch làm tăng tỷ lệ đúng hạn so với B2.

\subsection{So sánh với phương pháp từ tài liệu}
\label{sec:results-literature}

\input{tables/ext_literature}

\Cref{tab:literature} đặt TRAN cạnh B1, B2 và P. Trên v0, TRAN giao đúng hạn \ResultExtTimelyVzeroTRAN{} số cảnh báo, cao hơn \ResultExtTimelyVzeroBone{} của B1 nhưng thấp hơn \ResultExtTimelyVzeroBtwo{} của B2 và \ResultExtTimelyVzeroP{} của P. Khi backhaul chỉ còn \qty{50}{\kilo\bit\per\second}, TRAN đạt \ResultExtTimelyVzeroBhfivezeroTRAN, gần bằng \ResultExtTimelyVzeroBhfivezeroBone{} của B1 và thấp hơn \ResultExtTimelyVzeroBhfivezeroBtwo{} của B2. Gap của TRAN so với cận trên chính quỹ đạo của nó là \ResultExtGapVzeroTRAN{} và \ResultExtGapVzeroBhfivezeroTRAN, lớn hơn gap của B2 trên cùng họ, và TRAN lập kế hoạch lâu hơn P (\ResultExtPlanningVzeroTRAN{} so với \ResultExtPlanningVzeroP{} giây trên v0).

\input{tables/ext_literature_statistics}

Theo \cref{tab:literature-stats}, P hơn TRAN trung bình \ResultExtDiffVzeroPTran{} trên v0 (KTC 95\% [\ResultExtDiffLowVzeroPTran, \ResultExtDiffHighVzeroPTran], $p_{\mathrm{Holm}}=\ResultExtHolmVzeroPTran$) và \ResultExtDiffVzeroBhfivezeroPTran{} trên v0-bh50 (KTC 95\% [\ResultExtDiffLowVzeroBhfivezeroPTran, \ResultExtDiffHighVzeroBhfivezeroPTran], $p_{\mathrm{Holm}}=\ResultExtHolmVzeroBhfivezeroPTran$); so sánh B2 $-$ TRAN cho kết quả tương tự. Trên v0-bh50, khoảng tin cậy bootstrap không chứa không và tương quan hạng biserial bằng \ResultExtRankBiserialVzeroBhfivezeroPTran, nhưng sau hiệu chỉnh Holm cho bốn so sánh, không chênh lệch nào có ý nghĩa ở mức 5\%. Kết quả phù hợp với cấu trúc của TRAN: phương pháp thiết kế trên một kênh hiệu dụng tất định với lịch băng thông tĩnh và không biết dung lượng backhaul, nên mất lợi thế rõ nhất ở họ có backhaul băng hẹp.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{semilogxaxis}[
    width=0.8\textwidth, height=5.5cm, xmin=0.05, xmax=1000,
    xlabel={Thời gian lập kế hoạch trung bình (s)}, ylabel={Tỷ lệ đúng hạn},
    grid=major, legend pos=north west, /pgf/number format/use comma,
    yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},
]
\addplot[mark=*, color=blue!70!black] table[col sep=comma, x=planning, y=timely] {data/phase2_budget_b2.csv};
\addlegendentry{B2, $E=250$--$4000$}
\addplot[mark=square*, dashed, color=red!70!black] table[col sep=comma, x=planning, y=timely] {data/phase2_budget_p.csv};
\addlegendentry{P, $E=250$--$4000$}
\addplot[only marks, mark=triangle*, mark size=3pt, color=green!50!black, nodes near coords, point meta=explicit symbolic,
    every node near coord/.append style={font=\footnotesize, anchor=south west}]
    table[col sep=comma, x=planning, y=timely, meta=method] {data/ext_quality_points.csv};
\addlegendentry{B1 và TRAN}
\end{semilogxaxis}
\end{tikzpicture}
\caption{Tỷ lệ đúng hạn theo thời gian lập kế hoạch trên 30 scenario v0: đường B2 và P ứng với ngân sách đánh giá từ 250 đến 4000, điểm B1 và TRAN ứng với cấu hình mặc định.}
\label{fig:quality-runtime}
\end{figure}

\Cref{fig:quality-runtime} đặt TRAN lên đường chất lượng--thời gian của B2 và P. Với ngân sách $E=500$, B2 đã đạt \ResultTwoBudgetBtwoFivezerozero, cao hơn TRAN, trong khi thời gian lập kế hoạch ngắn hơn nhiều; tăng ngân sách chỉ đưa B2 và P tới vùng bão hoà đã thấy ở \cref{sec:results-budget}.

\subsection{Ablation}
```

Edit 3: replace

```latex
\Cref{fig:sensitivity} cho thấy thứ tự B1 thấp hơn B2 và P giữ nguyên trên mọi họ không suy biến. Hạn ngắn thu hẹp lợi thế (B1 \ResultTwoSensSensDeadlineShortBone{} so với P \ResultTwoSensSensDeadlineShortP), còn hạn dài mở rộng lợi thế (B1 \ResultTwoSensSensDeadlineLongBone{} so với P \ResultTwoSensSensDeadlineLongP). Cảnh báo lớn hơn làm mọi phương pháp giảm mạnh: với $L_{\mathrm{ref}}=\qty{1}{\mega\bit}$, B1 đạt \ResultTwoSensSensLrefonemBone{} và P đạt \ResultTwoSensSensLrefonemP. Tăng số cảnh báo lên 60 hoặc giảm $B_{\mathrm{tot}}$ còn một nửa làm tỷ lệ đúng hạn giảm nhẹ (P \ResultTwoSensSensAlertssixzeroP{} và \ResultTwoSensSensBtotzerofiveP, so với \ResultTwoSensVzeroP{} của v0), còn số candidate $K$ ảnh hưởng ít. Họ dùng hằng số vật lý khác ($\beta_0=\qty{-40}{\dB}$, $N_0=\qty{-164}{\dBm\per\hertz}$) suy biến: mọi nguồn có relay mặt đất trong tầm và cả ba phương pháp đạt \ResultTwoSensSensPhysicalP, nên họ này không được dùng để kết luận về vai trò của UAV.

\subsection{Độ chặt của cận trên}
\label{sec:results-milp}
```

with

```latex
\Cref{fig:sensitivity} cho thấy thứ tự B1 thấp hơn B2 và P giữ nguyên trên mọi họ không suy biến. Hạn ngắn thu hẹp lợi thế (B1 \ResultTwoSensSensDeadlineShortBone{} so với P \ResultTwoSensSensDeadlineShortP), còn hạn dài mở rộng lợi thế (B1 \ResultTwoSensSensDeadlineLongBone{} so với P \ResultTwoSensSensDeadlineLongP). Cảnh báo lớn hơn làm mọi phương pháp giảm mạnh: với $L_{\mathrm{ref}}=\qty{1}{\mega\bit}$, B1 đạt \ResultTwoSensSensLrefonemBone{} và P đạt \ResultTwoSensSensLrefonemP. Tăng số cảnh báo lên 60 hoặc giảm $B_{\mathrm{tot}}$ còn một nửa làm tỷ lệ đúng hạn giảm nhẹ (P \ResultTwoSensSensAlertssixzeroP{} và \ResultTwoSensSensBtotzerofiveP, so với \ResultTwoSensVzeroP{} của v0), còn số candidate $K$ ảnh hưởng ít. Họ dùng hằng số vật lý khác ($\beta_0=\qty{-40}{\dB}$, $N_0=\qty{-164}{\dBm\per\hertz}$) suy biến: mọi nguồn có relay mặt đất trong tầm và cả ba phương pháp đạt \ResultTwoSensSensPhysicalP, nên họ này không được dùng để kết luận về vai trò của UAV.

\subsection{Case study RescueNet}
\label{sec:results-case-study}

\input{tables/ext_case_study}

\Cref{tab:case-study} tóm tắt case study. Không có UAV, B0 chỉ giao đúng hạn \ResultExtCaseTimelyBzero{} số cảnh báo và chỉ \ResultExtCaseConnBzero{} số nguồn có đường theo thời gian tới trung tâm. Quỹ đạo cố định của B1 nâng tỷ lệ đúng hạn lên \ResultExtCaseTimelyBone; B2, P và TRAN đạt \ResultExtCaseTimelyBtwo, \ResultExtCaseTimelyP{} và \ResultExtCaseTimelyTRAN, với khoảng tin cậy chồng lấn nhau. Thứ tự này giống thí nghiệm chính: đồng tối ưu quỹ đạo và đường tăng tỷ lệ đúng hạn so với quỹ đạo cố định, sửa lịch không thêm lợi ích, còn TRAN gần B2 nhưng có gap lớn hơn (\ResultExtCaseGapTRAN{} so với \ResultExtCaseGapBtwo) và lập kế hoạch lâu hơn (\ResultExtCasePlanningTRAN{} so với \ResultExtCasePlanningBtwo{} giây). Dù ít nguồn có relay mặt đất trong tầm hơn v0, mức tăng của B1 và B2 so với B0 gần bằng mức tăng ở thí nghiệm chính.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.72\textwidth, height=0.72\textwidth, axis equal image, xmin=0, xmax=2, ymin=0, ymax=2,
    xlabel={$x$ (km)}, ylabel={$y$ (km)}, /pgf/number format/use comma,
    legend style={font=\footnotesize, at={(0.5,-0.13)}, anchor=north, legend columns=3},
]
\addplot[gray!45, unbounded coords=jump] table[col sep=comma, x=x, y=y] {data/ext_map_edges_alive.csv};
\addlegendentry{Cạnh còn sống}
\addplot[red!60, dashed, unbounded coords=jump] table[col sep=comma, x=x, y=y] {data/ext_map_edges_failed.csv};
\addlegendentry{Cạnh hỏng}
\addplot[only marks, mark=*, mark size=1.3pt, black!55] table[col sep=comma, x=x, y=y] {data/ext_map_relays.csv};
\addlegendentry{Relay}
\addplot[only marks, mark=square*, mark size=3pt, black] table[col sep=comma, x=x, y=y] {data/ext_map_center.csv};
\addlegendentry{Trung tâm}
\addplot[black!60, dotted, thick] table[col sep=comma, x=x, y=y] {data/ext_map_trajectory_b1.csv};
\addlegendentry{Quỹ đạo B1}
\addplot[blue!70!black, thick] table[col sep=comma, x=x, y=y] {data/ext_map_trajectory_p.csv};
\addlegendentry{Quỹ đạo P}
\addplot[green!50!black, thick, dashed] table[col sep=comma, x=x, y=y] {data/ext_map_trajectory_tran.csv};
\addlegendentry{Quỹ đạo TRAN}
\addplot[only marks, mark=o, orange!85!black] table[col sep=comma, x=x, y=y, restrict expr to domain={\thisrow{class}}{3:3}] {data/ext_map_targets.csv};
\addlegendentry{Hư hại nhẹ}
\addplot[only marks, mark=triangle*, red!70!black] table[col sep=comma, x=x, y=y, restrict expr to domain={\thisrow{class}}{4:4}] {data/ext_map_targets.csv};
\addlegendentry{Hư hại nặng}
\addplot[only marks, mark=x, thick, violet] table[col sep=comma, x=x, y=y, restrict expr to domain={\thisrow{class}}{5:5}] {data/ext_map_targets.csv};
\addlegendentry{Sập hoàn toàn}
\end{axis}
\end{tikzpicture}
\caption{Target, mạng relay và quỹ đạo của B1, P và TRAN trên scenario đại diện \ResultExtTraceScenario{} của case study.}
\label{fig:case-map}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.8\textwidth, height=5.5cm, xmin=0, xmax=150, ymin=0,
    xlabel={Slot}, ylabel={Tỷ lệ cảnh báo đã giao đúng hạn},
    grid=major, legend pos=north west, /pgf/number format/use comma,
    yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},
]
\addplot[black!60, dotted, thick] table[col sep=comma, x=slot, y=B1] {data/ext_progress.csv};
\addlegendentry{B1}
\addplot[blue!70!black, thick] table[col sep=comma, x=slot, y=P] {data/ext_progress.csv};
\addlegendentry{P}
\addplot[green!50!black, thick, dashed] table[col sep=comma, x=slot, y=TRAN] {data/ext_progress.csv};
\addlegendentry{TRAN}
\end{axis}
\end{tikzpicture}
\caption{Tiến trình giao cảnh báo trên scenario \ResultExtTraceScenario: tỷ lệ trong \ResultExtTraceAlertCount{} cảnh báo đã tới trung tâm đúng hạn tính đến cuối mỗi slot, trung bình trên 30 realization.}
\label{fig:case-progress}
\end{figure}

\Cref{fig:case-map} vẽ scenario đại diện \ResultExtTraceScenario, là scenario có tỷ lệ đúng hạn của P gần trung vị nhất. P đi qua cùng các vùng như B1 nhưng dừng ở mỗi vùng theo lịch khác, còn TRAN bay theo một đường liên tục và vươn rộng hơn về hai phía của vùng target. Trên scenario này, B1, P và TRAN giao đúng hạn \ResultExtTraceBone, \ResultExtTraceP{} và \ResultExtTraceTRAN{} số cảnh báo. \Cref{fig:case-progress} cho thấy cả ba phương pháp giao theo từng đợt khi UAV tới gần các cụm nguồn; B1 gần như ngừng tăng sau khoảng slot 100, trong khi P và TRAN còn giao thêm nhiều cảnh báo có hạn muộn và kết thúc cao hơn.

\subsection{Độ chặt của cận trên}
\label{sec:results-milp}
```

Edit 4: replace

```latex
\label{sec:rq}

\textbf{RQ1 --- đồng tối ưu so với tối ưu từng phần.} Đồng tối ưu quỹ đạo, đường và băng thông (B2) tăng tỷ lệ đúng hạn so với quỹ đạo cố định (B1) trên cả hai họ, với chênh lệch \ResultTwoDiffVzeroBtwoBone{} và \ResultTwoDiffVzeroBhfivezeroBtwoBone{} cùng chiều trên cả mười topology. Ablation chỉ ra phần lợi ích này đến từ khối đường và khối quỹ đạo; khối băng thông gần như không đóng góp khi băng thông đã chia theo backlog.

\textbf{RQ2 --- sửa lịch theo nguy cơ trễ hạn.} Không đo được lợi ích: P $-$ B2 bằng \ResultTwoDiffVzeroPBtwo{} trên v0 và \ResultTwoDiffVzeroBhfivezeroPBtwo{} trên v0-bh50 với $p_{\mathrm{Holm}}=\ResultTwoHolmVzeroPBtwo$, các biến thể chỉ giữ một thao tác hoặc bỏ thông tin backlog cho kết quả như P, và kết luận không đổi khi tăng ngân sách.

\textbf{RQ3 --- khi nào UAV khôi phục việc chuyển dữ liệu.} UAV đưa khả năng kết nối từ \ResultTwoConnVzeroBzero{} lên gần một và tăng tỷ lệ đúng hạn ở mọi mức kết nối mặt đất, với mức tăng tương tự nhau giữa các nhóm; lợi ích biến mất khi mọi nguồn đã có relay mặt đất trong tầm, như ở họ dùng hằng số vật lý khác.

\textbf{RQ4 --- suy giảm theo điều kiện.} Tỷ lệ đúng hạn giảm mạnh nhất khi cảnh báo lớn hơn hoặc hạn ngắn hơn, giảm nhẹ khi tăng số cảnh báo hoặc giảm phổ; thứ tự giữa các phương pháp giữ nguyên. Sai lệch mô hình kênh khi thiết kế, thể hiện qua việc thiết kế trên tốc độ kỳ vọng thay cho realization lấy mẫu, làm tỷ lệ đúng hạn của P thay đổi \ResultTwoAblationDiffVzeroPExpectedDesign.

\textbf{RQ5 --- độ bền của kết luận.} Các chênh lệch có ý nghĩa đều cùng chiều trên cả mười topology, và thứ tự B1 thấp hơn B2, P giữ nguyên trên mọi họ độ nhạy không suy biến. Báo cáo chưa thay đổi quy tắc sinh vùng hư hại, điểm nguồn và cảnh báo, nên độ bền theo quy tắc tạo target chưa được kiểm chứng.
```

with

```latex
\label{sec:rq}

\textbf{RQ1 --- đồng tối ưu so với tối ưu từng phần.} Đồng tối ưu quỹ đạo, đường và băng thông (B2) tăng tỷ lệ đúng hạn so với quỹ đạo cố định (B1) trên cả hai họ, với chênh lệch \ResultTwoDiffVzeroBtwoBone{} và \ResultTwoDiffVzeroBhfivezeroBtwoBone{} cùng chiều trên cả mười topology. Ablation chỉ ra phần lợi ích này đến từ khối đường và khối quỹ đạo; khối băng thông gần như không đóng góp khi băng thông đã chia theo backlog. So với phương pháp mạnh từ tài liệu (TRAN), B2 và P cao hơn trên cả hai họ, rõ nhất khi backhaul băng hẹp, nhưng chênh lệch không có ý nghĩa sau hiệu chỉnh Holm.

\textbf{RQ2 --- sửa lịch theo nguy cơ trễ hạn.} Không đo được lợi ích: P $-$ B2 bằng \ResultTwoDiffVzeroPBtwo{} trên v0 và \ResultTwoDiffVzeroBhfivezeroPBtwo{} trên v0-bh50 với $p_{\mathrm{Holm}}=\ResultTwoHolmVzeroPBtwo$, các biến thể chỉ giữ một thao tác hoặc bỏ thông tin backlog cho kết quả như P, và kết luận không đổi khi tăng ngân sách.

\textbf{RQ3 --- khi nào UAV khôi phục việc chuyển dữ liệu.} UAV đưa khả năng kết nối từ \ResultTwoConnVzeroBzero{} lên gần một và tăng tỷ lệ đúng hạn ở mọi mức kết nối mặt đất, với mức tăng tương tự nhau giữa các nhóm; lợi ích biến mất khi mọi nguồn đã có relay mặt đất trong tầm, như ở họ dùng hằng số vật lý khác. Trong case study RescueNet, UAV nâng khả năng kết nối từ \ResultExtCaseConnBzero{} lên gần một.

\textbf{RQ4 --- suy giảm theo điều kiện.} Tỷ lệ đúng hạn giảm mạnh nhất khi cảnh báo lớn hơn hoặc hạn ngắn hơn, giảm nhẹ khi tăng số cảnh báo hoặc giảm phổ; thứ tự giữa các phương pháp giữ nguyên. Sai lệch mô hình kênh khi thiết kế, thể hiện qua việc thiết kế trên tốc độ kỳ vọng thay cho realization lấy mẫu, làm tỷ lệ đúng hạn của P thay đổi \ResultTwoAblationDiffVzeroPExpectedDesign.

\textbf{RQ5 --- độ bền của kết luận.} Các chênh lệch có ý nghĩa đều cùng chiều trên cả mười topology, và thứ tự B1 thấp hơn B2, P giữ nguyên trên mọi họ độ nhạy không suy biến. Case study RescueNet thay quy tắc sinh điểm nguồn bằng vị trí công trình hư hại thật và giữ nguyên thứ tự B0 thấp hơn B1, B1 thấp hơn B2, P và TRAN; kiểm chứng này chỉ dựa trên một bố cục địa lý, còn quy tắc sinh cảnh báo chưa được thay đổi.
```


Create `docs/report/tables/ext_literature.tex`:

```latex
% So sánh với TRAN — hàng đọc từ data/ext_literature.csv (sinh bởi experiments/report_tables_extensions.py).
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{So sánh với TRAN trên 30 scenario đánh giá mỗi họ, mỗi scenario 30 realization kênh. B1, B2 và P lấy từ thí nghiệm chính; khoảng tin cậy 95\% lấy bằng bootstrap theo topology; gap riêng tính so với cận trên chính quỹ đạo của phương pháp; thời gian lập kế hoạch tính bằng giây.}
\label{tab:literature}
\begin{tabular}{@{}llcccr@{}}
\toprule
\textbf{Backhaul} & \textbf{PP} & \textbf{Đúng hạn} & \textbf{KTC 95\%} & \textbf{Gap riêng} & \textbf{Lập KH} \\
\midrule
\csvreader[separator=semicolon, late after line=\\]{data/ext_literature.csv}{2=\colbackhaul, 3=\colmethod, 4=\coltimely, 5=\colci, 6=\colgap, 7=\colplanning}{\colbackhaul & \colmethod & \coltimely & \colci & \colgap & \colplanning}
\bottomrule
\end{tabular}
\end{table}
```

Create `docs/report/tables/ext_literature_statistics.tex`:

```latex
% Kiểm định so với TRAN — hàng đọc từ data/ext_literature_statistics.csv (sinh bởi experiments/report_tables_extensions.py).
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Bốn so sánh với TRAN theo cặp ở mức topology (10 topology), cùng quy ước với \cref{tab:phase2-stats}; $p_{\mathrm{Holm}}$ hiệu chỉnh trên bốn so sánh này.}
\label{tab:literature-stats}
\begin{tabular}{@{}llccccc@{}}
\toprule
\textbf{Họ} & \textbf{So sánh} & \textbf{Chênh lệch} & \textbf{KTC 95\%} & \textbf{$r$} & \textbf{$p$} & \textbf{$p_{\mathrm{Holm}}$} \\
\midrule
\csvreader[separator=semicolon, late after line=\\]{data/ext_literature_statistics.csv}{1=\colset, 2=\colcomparison, 3=\coldiff, 4=\colci, 5=\colr, 6=\colp, 7=\colholm}{\colset & \colcomparison & \coldiff & \colci & \colr & \colp & \colholm}
\bottomrule
\end{tabular}
\end{table}
```

Create `docs/report/tables/ext_case_study.tex`:

```latex
% Case study RescueNet — hàng đọc từ data/ext_case_study.csv (sinh bởi experiments/report_tables_extensions.py).
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Case study RescueNet trên 30 scenario đánh giá, mỗi scenario 30 realization kênh. Khoảng tin cậy 95\% lấy bằng bootstrap theo topology; gap riêng tính như ở \cref{tab:phase2-main}; thời gian lập kế hoạch tính bằng giây.}
\label{tab:case-study}
\begin{tabular}{@{}lccccr@{}}
\toprule
\textbf{PP} & \textbf{Đúng hạn} & \textbf{KTC 95\%} & \textbf{Conn} & \textbf{Gap riêng} & \textbf{Lập KH} \\
\midrule
\csvreader[separator=semicolon, late after line=\\]{data/ext_case_study.csv}{1=\colmethod, 2=\coltimely, 3=\colci, 4=\colconn, 5=\colgap, 6=\colplanning}{\colmethod & \coltimely & \colci & \colconn & \colgap & \colplanning}
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 2: Compile a copy outside the workspace and check it**

The editor rebuilds `docs/report` when a `.tex` file changes, so compile a copy in a temporary directory. From the repository root:

```bash
REPO=$(git rev-parse --show-toplevel)
CHECK=$(mktemp -d)
cp -r docs/report "$CHECK/report"
rm -rf "$CHECK/report/build" "$CHECK/report/main.pdf"
mkdir "$CHECK/report/build"
(cd "$CHECK/report" && bash "$REPO/.claude/skills/latex-document-skill/scripts/compile_latex.sh" main.tex --verbose > build/compile-verbose.txt 2>&1)
uv run python - "$CHECK/report" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

report = Path(sys.argv[1])
log = (report / "build" / "compile-verbose.txt").read_text(encoding="utf-8", errors="replace")
final_pass = log.split("This is pdfTeX")[-1]
issues = re.findall(r"^.*(?:Overfull|Underfull|LaTeX Warning|Package \w+ Warning|^! ).*$", final_pass, re.M)
info = subprocess.run(["pdfinfo", str(report / "main.pdf")], capture_output=True, text=True).stdout
text = subprocess.run(["pdftotext", str(report / "main.pdf"), "-"], capture_output=True, text=True).stdout
pages = re.search(r"Pages:\s+(\d+)", info).group(1) if info else "none"
english = re.findall(r"\d\s*,?\s+and\s+\d|\b[Aa]ppendix\b", " ".join(text.split()))
print(f"final pass issues: {len(issues)}; pages: {pages}; unresolved ??: {text.count('??')}; English cross-reference words: {len(english)}")
for line in issues[:5]:
    print(line)
PY
```

Expected: exactly one line, `final pass issues: 0; pages: 29; unresolved ??: 0; English cross-reference words: 0`. Any further line is a warning or error of the final pass; fix it in `docs/report` and repeat this step.

- [ ] **Step 3: Commit**

```bash
git add docs/report/sections/experiments.tex docs/report/tables/ext_literature.tex docs/report/tables/ext_literature_statistics.tex docs/report/tables/ext_case_study.tex
git commit -m "docs: report the TRAN comparison and the RescueNet case study"
```

### Task 7: Front matter, discussion, appendix and bibliography

**Files:**
- Create: `docs/report/sections/appendix.tex`, `docs/report/tables/ext_compute.tex`, `docs/report/tables/acceptance.tex`
- Modify: `docs/report/sections/abstract.tex`, `docs/report/sections/introduction.tex`, `docs/report/sections/related-work.tex`, `docs/report/sections/discussion.tex`, `docs/report/sections/conclusion.tex`, `docs/report/references.bib`, `docs/report/main.tex`

**Interfaces:**
- Consumes: The labels of Tasks 5 and 6; `ext_compute.csv` and `\ResultExtComputeHours` (Task 4); the acceptance items of `docs/issue.md` Section 14.
- Produces: Labels `sec:appendix`, `tab:compute`, `tab:acceptance`; the final abstract, introduction, related work, discussion and conclusion; the checked `rahnemoonfar2023rescuenet` record. `main.tex` ends with `\appendix` and `\input{sections/appendix}` after the bibliography.

- [ ] **Step 1: Edit the report sources**

In `docs/report/sections/abstract.tex` (2 edits):

Edit 1: replace

```latex
Sau thiên tai, một phần mạng mặt đất bị hỏng và các cảnh báo phát sinh tại hiện trường cần được đưa về trung tâm cứu hộ trước hạn. Báo cáo này nghiên cứu bài toán đồng tối ưu quỹ đạo của một UAV làm relay di động, lựa chọn đường chuyển tiếp qua mạng relay mặt đất có backhaul hữu hạn và phân bổ băng thông, nhằm tối đa tỷ lệ cảnh báo giao đúng hạn dưới kênh không--đất theo mô hình xác suất tầm nhìn thẳng~\cite{huang2025ddatsap}. Chúng tôi chứng minh bài toán là NP-khó ngay cả khi quỹ đạo cố định, xây dựng cận trên quy hoạch tuyến tính hợp lệ theo từng realization kênh, và đề xuất phương pháp tìm kiếm theo ba khối được chấm điểm trên các realization kênh lấy mẫu (B2), cùng biến thể sửa lịch theo nguy cơ trễ hạn (P). Trên \ResultTwoScenarioCount{} kịch bản đánh giá tổng hợp từ Topology Zoo~\cite{knight2011zoo} và SNDlib~\cite{orlowski2010sndlib}, B2 giao đúng hạn \ResultTwoTimelyVzeroBtwo{} số cảnh báo, so với \ResultTwoTimelyVzeroBone{} khi UAV bay theo quỹ đạo cố định và \ResultTwoTimelyVzeroBzero{} khi không có UAV; chênh lệch với quỹ đạo cố định cùng chiều trên cả mười topology ($p_{\mathrm{Holm}}=\ResultTwoHolmVzeroBtwoBone$) và lớn hơn khi backhaul băng hẹp. Cơ chế sửa lịch không cải thiện thêm (P $-$ B2 bằng \ResultTwoDiffVzeroPBtwo), còn mục tiêu công bằng leximin làm tỷ lệ đúng hạn giảm \ResultTwoDiffVzeroBtwoBthree{} so với B2. Ablation cho thấy lợi ích đến từ việc đồng chọn đường và quỹ đạo, và từ việc thiết kế trên các realization lấy mẫu thay cho tốc độ kỳ vọng.
```

with

```latex
Sau thiên tai, một phần mạng mặt đất bị hỏng và các cảnh báo phát sinh tại hiện trường cần được đưa về trung tâm cứu hộ trước hạn. Báo cáo này nghiên cứu bài toán đồng tối ưu quỹ đạo của một UAV làm relay di động, lựa chọn đường chuyển tiếp qua mạng relay mặt đất có backhaul hữu hạn và phân bổ băng thông, nhằm tối đa tỷ lệ cảnh báo giao đúng hạn dưới kênh không--đất theo mô hình xác suất tầm nhìn thẳng~\cite{huang2025ddatsap}. Chúng tôi chứng minh bài toán là NP-khó ngay cả khi quỹ đạo cố định, xây dựng cận trên quy hoạch tuyến tính hợp lệ theo từng realization kênh, và đề xuất phương pháp tìm kiếm theo ba khối được chấm điểm trên các realization kênh lấy mẫu (B2), cùng biến thể sửa lịch theo nguy cơ trễ hạn (P). Trên \ResultTwoScenarioCount{} kịch bản đánh giá tổng hợp từ Topology Zoo~\cite{knight2011zoo} và SNDlib~\cite{orlowski2010sndlib}, B2 giao đúng hạn \ResultTwoTimelyVzeroBtwo{} số cảnh báo, so với \ResultTwoTimelyVzeroBone{} khi UAV bay theo quỹ đạo cố định và \ResultTwoTimelyVzeroBzero{} khi không có UAV; chênh lệch với quỹ đạo cố định cùng chiều trên cả mười topology ($p_{\mathrm{Holm}}=\ResultTwoHolmVzeroBtwoBone$) và lớn hơn khi backhaul băng hẹp. Cơ chế sửa lịch không cải thiện thêm (P $-$ B2 bằng \ResultTwoDiffVzeroPBtwo), còn mục tiêu công bằng leximin làm tỷ lệ đúng hạn giảm \ResultTwoDiffVzeroBtwoBthree{} so với B2. Ablation cho thấy lợi ích đến từ việc đồng chọn đường và quỹ đạo, và từ việc thiết kế trên các realization lấy mẫu thay cho tốc độ kỳ vọng. So với phương pháp của Tran và cộng sự~\cite{tran2022relay} được chuyển thể sang cùng mô hình, B2 cao hơn \ResultExtDiffVzeroBtwoTran{} trên v0 và \ResultExtDiffVzeroBhfivezeroBtwoTran{} khi backhaul băng hẹp, dù chênh lệch không có ý nghĩa sau hiệu chỉnh Holm. Trong case study với \ResultExtTargetCount{} target lấy từ ảnh RescueNet~\cite{rahnemoonfar2023rescuenet}, thứ tự giữa các phương pháp giữ nguyên.
```

Edit 2: replace

```latex
\textbf{Từ khoá:} UAV relay, quỹ đạo, lựa chọn relay, phân bổ băng thông, hạn giao, cận trên quy hoạch tuyến tính, tìm kiếm cục bộ, kiểm định ghép cặp.
```

with

```latex
\textbf{Từ khoá:} UAV relay, quỹ đạo, lựa chọn relay, phân bổ băng thông, hạn giao, cận trên quy hoạch tuyến tính, tìm kiếm cục bộ, kiểm định ghép cặp, RescueNet.
```


In `docs/report/sections/introduction.tex` (4 edits):

Edit 1: replace

```latex
Báo cáo này nghiên cứu bài toán đồng tối ưu quỹ đạo của một UAV, lựa chọn relay hoặc đường chuyển tiếp cho từng cảnh báo và phân bổ băng thông, nhằm tối đa tỷ lệ cảnh báo giao đúng hạn; khả năng kết nối theo thời gian tới trung tâm là mục tiêu phụ theo thứ tự từ điển. Nghiên cứu hoàn toàn bằng mô phỏng trên các kịch bản tổng hợp từ dữ liệu công khai Topology Zoo~\cite{knight2011zoo} và SNDlib~\cite{orlowski2010sndlib}.
```

with

```latex
Báo cáo này nghiên cứu bài toán đồng tối ưu quỹ đạo của một UAV, lựa chọn relay hoặc đường chuyển tiếp cho từng cảnh báo và phân bổ băng thông, nhằm tối đa tỷ lệ cảnh báo giao đúng hạn; khả năng kết nối theo thời gian tới trung tâm là mục tiêu phụ theo thứ tự từ điển. Nghiên cứu hoàn toàn bằng mô phỏng trên các kịch bản tổng hợp từ dữ liệu công khai Topology Zoo~\cite{knight2011zoo}, SNDlib~\cite{orlowski2010sndlib} và RescueNet~\cite{rahnemoonfar2023rescuenet}.
```

Edit 2: replace

```latex
Thứ nhất, chúng tôi mô hình hoá bài toán với thời điểm phát sinh, hạn và kích thước của cảnh báo, mạng relay có backhaul hữu hạn, kênh PrLoS và quy ước nhân quả theo slot, cùng một bộ đánh giá độc lập với mọi phương pháp và bộ kịch bản tái lập được (\cref{sec:method}). Thứ hai, chúng tôi xây dựng cận trên quy hoạch tuyến tính theo từng realization kênh kèm chứng minh tính hợp lệ, đối chiếu cận với MILP trên các instance thu nhỏ, chứng minh bài toán là NP-khó ngay cả khi quỹ đạo cố định và phân tích độ phức tạp (\cref{sec:bound,sec:hardness,sec:complexity}). Thứ ba, chúng tôi đề xuất phương pháp tìm kiếm theo ba khối --- đường, băng thông theo trọng số và quỹ đạo từ một họ hành trình khả thi --- được chấm điểm trên các realization kênh lấy mẫu dưới một ngân sách đánh giá, cùng biến thể dùng mục tiêu leximin và biến thể sửa lịch theo nguy cơ trễ hạn (\cref{sec:algorithm}). Thứ tư, chúng tôi đánh giá các phương pháp trên hai họ kịch bản với kiểm định ghép cặp ở mức topology, ablation, thí nghiệm ngân sách và độ nhạy, và trả lời năm câu hỏi nghiên cứu (\cref{sec:experiments}). Thứ năm, chúng tôi thực hiện một tổng quan tài liệu có giao thức và chọn phương pháp mạnh từ tài liệu để so sánh (\cref{sec:related}).
```

with

```latex
Thứ nhất, chúng tôi mô hình hoá bài toán với thời điểm phát sinh, hạn và kích thước của cảnh báo, mạng relay có backhaul hữu hạn, kênh PrLoS và quy ước nhân quả theo slot, cùng một bộ đánh giá độc lập với mọi phương pháp và bộ kịch bản tái lập được (\cref{sec:method}). Thứ hai, chúng tôi xây dựng cận trên quy hoạch tuyến tính theo từng realization kênh kèm chứng minh tính hợp lệ, đối chiếu cận với MILP trên các instance thu nhỏ, chứng minh bài toán là NP-khó ngay cả khi quỹ đạo cố định và phân tích độ phức tạp (\cref{sec:bound,sec:hardness,sec:complexity}). Thứ ba, chúng tôi đề xuất phương pháp tìm kiếm theo ba khối --- đường, băng thông theo trọng số và quỹ đạo từ một họ hành trình khả thi --- được chấm điểm trên các realization kênh lấy mẫu dưới một ngân sách đánh giá, cùng biến thể dùng mục tiêu leximin và biến thể sửa lịch theo nguy cơ trễ hạn (\cref{sec:algorithm}). Thứ tư, chúng tôi đánh giá các phương pháp trên hai họ kịch bản với kiểm định ghép cặp ở mức topology, ablation, thí nghiệm ngân sách và độ nhạy, và trả lời năm câu hỏi nghiên cứu (\cref{sec:experiments}). Thứ năm, chúng tôi thực hiện một tổng quan tài liệu có giao thức, chuyển thể phương pháp mạnh được chọn (TRAN) sang mô hình của đề tài và so sánh với nó (\cref{sec:related,sec:tran,sec:results-literature}). Thứ sáu, chúng tôi xây dựng một case study với nguồn đặt tại các công trình hư hại trong ảnh RescueNet, được định vị bằng metadata của ảnh (\cref{sec:rescuenet,sec:results-case-study}), cùng một gói tái lập tự kiểm tra bảng, kịch bản và kết quả (\cref{sec:appendix}).
```

Edit 3: replace

```latex
Kết quả chính là phương pháp tìm kiếm theo khối tăng tỷ lệ cảnh báo đúng hạn một cách nhất quán so với quỹ đạo cố định, trong khi cơ chế sửa lịch không cải thiện thêm; báo cáo trình bày kết quả này như đo được. Việc so sánh với phương pháp mạnh từ tài liệu và case study RescueNet thuộc giai đoạn cuối của đề tài.
```

with

```latex
Kết quả chính là phương pháp tìm kiếm theo khối tăng tỷ lệ cảnh báo đúng hạn một cách nhất quán so với quỹ đạo cố định, trong khi cơ chế sửa lịch không cải thiện thêm; báo cáo trình bày kết quả này như đo được. B2 và P cũng cao hơn TRAN trên cả hai họ kịch bản, tuy chênh lệch không có ý nghĩa sau hiệu chỉnh Holm, và thứ tự giữa các phương pháp giữ nguyên trong case study RescueNet.
```

Edit 4: replace

```latex
Phần còn lại của báo cáo được tổ chức như sau. \Cref{sec:related} trình bày tổng quan tài liệu. \Cref{sec:method} trình bày mô hình, phương pháp cơ sở, cận trên, phân tích độ khó và phương pháp tìm kiếm. \Cref{sec:experiments} báo cáo kết quả thực nghiệm, \cref{sec:discussion} thảo luận và \cref{sec:conclusion} kết luận.
```

with

```latex
Phần còn lại của báo cáo được tổ chức như sau. \Cref{sec:related} trình bày tổng quan tài liệu. \Cref{sec:method} trình bày mô hình, phương pháp cơ sở, cận trên, phân tích độ khó và phương pháp tìm kiếm. \Cref{sec:experiments} báo cáo kết quả thực nghiệm, \cref{sec:discussion} thảo luận, \cref{sec:conclusion} kết luận, và phụ lục \ref{sec:appendix} mô tả gói tái lập.
```


In `docs/report/sections/related-work.tex` (1 edit):

Edit 1: replace

```latex
Không bài nào trong \cref{tab:related} kết hợp hạn theo từng yêu cầu với việc chọn relay mặt đất trên backhaul có dung lượng hữu hạn: các bài có hạn giả định một điểm đích, còn các bài có ràng buộc backhaul không có hạn theo yêu cầu. Để chọn một phương pháp mạnh từ tài liệu, các tiêu chí được xét theo thứ tự: mục tiêu gần với số cảnh báo đúng hạn, thiết kế ngoại tuyến với nhu cầu đã biết, đồng tối ưu quỹ đạo và tài nguyên vô tuyến, đủ chi tiết hoặc có mã để tái lập, và thích nghi được mà không đổi bản chất phương pháp. Ba bài của Tran, Samir và Du đạt tiêu chí đầu; tiêu chí cuối quyết định chọn Tran và cộng sự~\cite{tran2022relay} (phiên bản bán song công), vì chỉ bài này mô hình dữ liệu đi qua bộ đệm UAV tới gateway mặt đất, tương ứng với đường UAV thả dữ liệu vào mạng mặt đất ở đây. Bài của Samir và cộng sự là phương án dự phòng vì đã có mã cài đặt công khai. Việc thích nghi và đánh giá phương pháp này thuộc giai đoạn cuối của đề tài.
```

with

```latex
Không bài nào trong \cref{tab:related} kết hợp hạn theo từng yêu cầu với việc chọn relay mặt đất trên backhaul có dung lượng hữu hạn: các bài có hạn giả định một điểm đích, còn các bài có ràng buộc backhaul không có hạn theo yêu cầu. Để chọn một phương pháp mạnh từ tài liệu, các tiêu chí được xét theo thứ tự: mục tiêu gần với số cảnh báo đúng hạn, thiết kế ngoại tuyến với nhu cầu đã biết, đồng tối ưu quỹ đạo và tài nguyên vô tuyến, đủ chi tiết hoặc có mã để tái lập, và thích nghi được mà không đổi bản chất phương pháp. Ba bài của Tran, Samir và Du đạt tiêu chí đầu; tiêu chí cuối quyết định chọn Tran và cộng sự~\cite{tran2022relay} (phiên bản bán song công), vì chỉ bài này mô hình dữ liệu đi qua bộ đệm UAV tới gateway mặt đất, tương ứng với đường UAV thả dữ liệu vào mạng mặt đất ở đây. Bài của Samir và cộng sự là phương án dự phòng vì đã có mã cài đặt công khai. \Cref{sec:tran} trình bày cách chuyển thể phương pháp này và \cref{sec:results-literature} so sánh nó với các phương pháp của đề tài.
```


In `docs/report/sections/discussion.tex` (1 edit):

Edit 1: replace

```latex
\paragraph{Giới hạn.}
Kết quả dựa trên mô phỏng và không chứng minh hiệu năng thực địa. Mạng backbone gốc được thu nhỏ về vùng \qty{2}{\kilo\metre}, nên chỉ còn mượn cấu trúc. Trạng thái kênh độc lập giữa các slot, kênh mặt đất tất định và công suất cố định. Kiểm định dựa trên mười topology, nên mức ý nghĩa nhỏ nhất đạt được bị giới hạn; độ nhạy chỉ dùng replicate đầu của mỗi topology và quy tắc sinh vùng hư hại, điểm nguồn, cảnh báo chưa được thay đổi. Các phương pháp là heuristic không có bảo đảm tối ưu, và cận chỉ hợp lệ với quỹ đạo và tập candidate đã cho.

\paragraph{Hướng tiếp theo.}
Giai đoạn cuối của đề tài sẽ cài đặt và so sánh với phương pháp của Tran và cộng sự~\cite{tran2022relay} đã chọn ở \cref{sec:related}, bổ sung case study với mục tiêu lấy từ bộ dữ liệu RescueNet~\cite{rahnemoonfar2023rescuenet}, và hoàn thiện gói tái lập.
```

with

```latex
\paragraph{Phương pháp từ tài liệu.}
TRAN cải thiện rõ so với quỹ đạo cố định khi backhaul đủ rộng nhưng gần như mất lợi thế khi backhaul băng hẹp. Ba khác biệt với B2 giải thích điều này. TRAN thiết kế trên một kênh hiệu dụng tất định, còn B2 chấm điểm trên các realization lấy mẫu, và thí nghiệm số realization thiết kế cho thấy khác biệt này đáng kể. TRAN không biết dung lượng backhaul, vì bài gốc giả định gateway nhận được mọi dữ liệu UAV chuyển xuống. TRAN cũng cố định lịch băng thông, trong khi bộ điều phối EDF phục vụ theo backlog. Các điểm chuyển thể ở \cref{tab:tran-adaptation}, như cửa sổ thu cố định theo $\theta$ và một gateway cho mỗi cảnh báo, có thể làm phương pháp bất lợi hơn một phiên bản tối ưu hai chặng cùng lúc; kết quả vì vậy là so sánh với một chuyển thể hợp lý của TRAN, không phải với mọi biến thể của nó.

\paragraph{Case study RescueNet.}
Target lấy từ ảnh thật trải khắp vùng thay vì tụ quanh ba tâm, nên ít nguồn có relay mặt đất trong tầm hơn và ba vùng hư hại của họ hành trình chỉ là xấp xỉ thô của phân bố target. Dù vậy thứ tự giữa các phương pháp không đổi, cho thấy kết luận chính không phụ thuộc quy tắc sinh điểm nguồn tổng hợp. Case study chỉ có một bố cục địa lý, các ảnh chồng lên nhau, còn thời điểm, hạn và kích thước cảnh báo vẫn là giả định, nên nó bổ sung chứ không thay thế các kiểm định của thí nghiệm chính.

\paragraph{Giới hạn.}
Kết quả dựa trên mô phỏng và không chứng minh hiệu năng thực địa. Mạng backbone gốc được thu nhỏ về vùng \qty{2}{\kilo\metre}, nên chỉ còn mượn cấu trúc. Trạng thái kênh độc lập giữa các slot, kênh mặt đất tất định và công suất cố định. Kiểm định dựa trên mười topology, nên mức ý nghĩa nhỏ nhất đạt được bị giới hạn; độ nhạy chỉ dùng replicate đầu của mỗi topology, quy tắc sinh cảnh báo chưa được thay đổi và quy tắc sinh điểm nguồn chỉ được thay bằng một bố cục lấy từ RescueNet. Các phương pháp là heuristic không có bảo đảm tối ưu, và cận chỉ hợp lệ với quỹ đạo và tập candidate đã cho.

\paragraph{Hướng tiếp theo.}
Một cận chặt hơn, các thao tác sửa lịch thay đổi đồng thời nhiều cảnh báo, thiết kế với nhiều realization hơn trong cùng thời gian tính, và case study trên nhiều khu vực với vị trí nạn nhân hoặc lưu lượng đo thực địa là các hướng mở rộng tự nhiên của đề tài.
```


In `docs/report/sections/conclusion.tex` (1 edit):

Edit 1: replace

```latex
Báo cáo đã mô hình hoá bài toán đồng tối ưu quỹ đạo UAV, lựa chọn relay và phân bổ băng thông cho cảnh báo có hạn giao, chứng minh bài toán NP-khó ngay cả khi quỹ đạo cố định, xây dựng cận trên quy hoạch tuyến tính hợp lệ theo từng realization kênh, và đề xuất phương pháp tìm kiếm theo khối được chấm điểm trên các realization kênh lấy mẫu. Trên \ResultTwoScenarioCount{} kịch bản đánh giá với backhaul thông thường, phương pháp này giao đúng hạn \ResultTwoTimelyVzeroBtwo{} số cảnh báo, so với \ResultTwoTimelyVzeroBone{} khi UAV bay theo quỹ đạo cố định và \ResultTwoTimelyVzeroBzero{} khi không có UAV, với chênh lệch cùng chiều trên cả mười topology; khi backhaul băng hẹp, lợi ích còn lớn hơn. Cơ chế sửa lịch theo nguy cơ trễ hạn không cải thiện thêm, mục tiêu leximin làm giảm tỷ lệ đúng hạn, và việc thiết kế trên các realization lấy mẫu quan trọng hơn ngân sách tìm kiếm. Khoảng cách còn lại tới cận hợp cho thấy vẫn còn chỗ để cải thiện, hoặc để siết cận.
```

with

```latex
Báo cáo đã mô hình hoá bài toán đồng tối ưu quỹ đạo UAV, lựa chọn relay và phân bổ băng thông cho cảnh báo có hạn giao, chứng minh bài toán NP-khó ngay cả khi quỹ đạo cố định, xây dựng cận trên quy hoạch tuyến tính hợp lệ theo từng realization kênh, và đề xuất phương pháp tìm kiếm theo khối được chấm điểm trên các realization kênh lấy mẫu. Trên \ResultTwoScenarioCount{} kịch bản đánh giá với backhaul thông thường, phương pháp này giao đúng hạn \ResultTwoTimelyVzeroBtwo{} số cảnh báo, so với \ResultTwoTimelyVzeroBone{} khi UAV bay theo quỹ đạo cố định và \ResultTwoTimelyVzeroBzero{} khi không có UAV, với chênh lệch cùng chiều trên cả mười topology; khi backhaul băng hẹp, lợi ích còn lớn hơn. Cơ chế sửa lịch theo nguy cơ trễ hạn không cải thiện thêm, mục tiêu leximin làm giảm tỷ lệ đúng hạn, và việc thiết kế trên các realization lấy mẫu quan trọng hơn ngân sách tìm kiếm. Phương pháp tìm kiếm theo khối cũng cao hơn chuyển thể của phương pháp mạnh từ tài liệu trên cả hai họ kịch bản, rõ nhất khi backhaul băng hẹp, tuy chênh lệch không có ý nghĩa sau hiệu chỉnh Holm, và thứ tự giữa các phương pháp giữ nguyên trong case study với target lấy từ ảnh RescueNet. Khoảng cách còn lại tới cận hợp cho thấy vẫn còn chỗ để cải thiện, hoặc để siết cận. Gói tái lập kèm theo sinh lại và tự kiểm tra các bảng, kịch bản và kết quả của báo cáo.
```


In `docs/report/references.bib` (2 edits):

Edit 1: replace

```bibtex
% rahnemoonfar2023rescuenet chưa được đối chiếu; phải kiểm tra trước khi trích dẫn (Pha 2).

@article{huang2025ddatsap,
  author  = {Huang, Ping and Lin, Jie and Liu, Tong and Ning, Jin and Luo, Junsong and Duo, Bin},
```

with

```bibtex
% rahnemoonfar2023rescuenet: đối chiếu bản ghi Crossref của DOI 10.1038/s41597-023-02799-4 (tập 10, số 1, bài 913).

@article{huang2025ddatsap,
  author  = {Huang, Ping and Lin, Jie and Liu, Tong and Ning, Jin and Luo, Junsong and Duo, Bin},
```

Edit 2: replace

```bibtex
  journal = {Scientific Data},
  year    = {2023},
  volume  = {10},
```

with

```bibtex
  journal = {Scientific Data},
  year    = {2023},
  volume  = {10},
  number  = {1},
```


In `docs/report/main.tex` (1 edit):

Edit 1: replace

```latex
\printbibliography[title={Tài liệu tham khảo}]
```

with

```latex
\printbibliography[title={Tài liệu tham khảo}]

\appendix
\input{sections/appendix}
```


Create `docs/report/sections/appendix.tex`:

```latex
\section{Gói tái lập}
\label{sec:appendix}

Mã nguồn, cấu hình, manifest dữ liệu, kết quả và dữ liệu của báo cáo nằm trong cùng kho mã; tài liệu \texttt{docs/reproducibility.md} mô tả môi trường, dữ liệu và giấy phép, nguồn của từng bảng và hình, và thứ tự chạy lại toàn bộ. Script \texttt{experiments/reproduce.py} kiểm tra ba mức. Mức \texttt{tables} chạy lại thống kê và các script sinh dữ liệu báo cáo, rồi so từng byte với các file đã lưu. Mức \texttt{scenarios} kiểm tra checksum của dữ liệu thô, sinh lại manifest target RescueNet và mọi họ kịch bản trong một thư mục tạm, rồi so với manifest đã lưu và với mã băm ghi trong kết quả. Mức \texttt{experiments} chạy lại các thí nghiệm, trên toàn bộ hoặc trên các kịch bản của một topology, và so từng dòng kết quả trên các cột không phải thời gian chạy. Kết quả của mỗi mức được ghi trong thư mục \texttt{results/reproducibility}.

Mỗi thí nghiệm lưu cấu hình, mã băm của mã nguồn và của manifest kịch bản, commit, môi trường và thời gian chạy. Script sinh dữ liệu báo cáo chỉ đọc một thí nghiệm khi kết quả mô tả mã hiện tại: mã băm trùng, hoặc kết quả đến từ một commit sạch là tổ tiên của phiên bản hiện tại và không file nào của các package quyết định kết quả bị sửa kể từ commit đó. \Cref{tab:compute} liệt kê thời gian tính đã ghi của từng thí nghiệm, tổng cộng \ResultExtComputeHours{} giờ; \cref{tab:acceptance} đối chiếu tiêu chí nghiệm thu của đề bài với các mục của báo cáo.

\input{tables/ext_compute}

\input{tables/acceptance}
```

Create `docs/report/tables/ext_compute.tex`:

```latex
% Thời gian tính — hàng đọc từ data/ext_compute.csv (sinh bởi experiments/report_tables_extensions.py).
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Thời gian tính đã ghi của từng thí nghiệm, bằng tổng thời gian lập kế hoạch, đánh giá và giải cận của các task (với 12 tiến trình song song, thời gian thực ngắn hơn nhiều).}
\label{tab:compute}
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Thí nghiệm} & \textbf{Số task} & \textbf{Giờ tính} \\
\midrule
\csvreader[separator=semicolon, late after line=\\]{data/ext_compute.csv}{1=\colexperiment, 2=\coltasks, 3=\colhours}{\colexperiment & \coltasks & \colhours}
\bottomrule
\end{tabular}
\end{table}
```

Create `docs/report/tables/acceptance.tex`:

```latex
% Đối chiếu tiêu chí nghiệm thu — bảng viết tay (docs/issue.md Mục 14, docs/reproducibility.md Mục 6).
\begin{table}[htbp]
\centering
\small
\caption{Tiêu chí nghiệm thu của đề bài và bằng chứng tương ứng.}
\label{tab:acceptance}
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.45\textwidth}>{\raggedright\arraybackslash}X@{}}
\toprule
\textbf{Tiêu chí} & \textbf{Bằng chứng} \\
\midrule
Có quỹ đạo, lựa chọn relay và phân bổ băng thông & \Cref{sec:model,sec:problem} \\
Mục tiêu đúng hạn với thời điểm phát sinh, hạn, số bit và xác suất & \Cref{sec:problem}, \cref{eq:objective} \\
Kết nối theo thời gian và đường tới trung tâm & \Cref{sec:problem} \\
Bài của Huang dùng đúng vai trò, khác biệt được công khai & \Cref{sec:related,sec:algorithm} \\
Dữ liệu có nguồn gốc, đơn vị và phép chuyển đổi tái lập & \Cref{sec:setup,sec:rescuenet,sec:appendix} \\
Baseline chạy trên benchmark công khai; thiếu nguồn được báo đúng & \Cref{sec:baselines,sec:tran,sec:rescuenet} \\
Ít nhất một cơ chế cải tiến và ablation chứng minh tác động & \Cref{sec:algorithm,sec:results-ablation} \\
Luồng bảo toàn, buffer không âm, không dùng thông tin tương lai & Bộ kiểm tra độc lập (\cref{sec:complexity}) \\
Tài nguyên tổng và quỹ đạo khả thi; không tăng công suất ngầm & \Cref{sec:model,sec:tran} \\
20--30 seed, kiểm định, cỡ ảnh hưởng và độ bất định & \Cref{sec:setup,sec:results-stats,sec:results-literature} \\
Thời gian chạy, độ bền, tính khả thi, khả năng mở rộng và độ nhạy & \Cref{sec:results-runtime,sec:results-sensitivity,sec:results-case-study}, \cref{tab:compute} \\
Cận trên LP và gap trên mọi instance; đối chiếu MILP trên instance nhỏ & \Cref{sec:bound,sec:results-milp} \\
Kết nối trên đồ thị mở rộng theo thời gian là mục tiêu phụ từ điển & \Cref{sec:problem} \\
B2 kế thừa cấu trúc BCD của Huang; B3 tách tác dụng của mục tiêu & \Cref{sec:algorithm,sec:results-stats} \\
Gói tái lập và báo cáo theo cấu trúc môn học & \Cref{sec:appendix} \\
\bottomrule
\end{tabularx}
\end{table}
```

- [ ] **Step 2: Compile a copy outside the workspace and check it**

The editor rebuilds `docs/report` when a `.tex` file changes, so compile a copy in a temporary directory. From the repository root:

```bash
REPO=$(git rev-parse --show-toplevel)
CHECK=$(mktemp -d)
cp -r docs/report "$CHECK/report"
rm -rf "$CHECK/report/build" "$CHECK/report/main.pdf"
mkdir "$CHECK/report/build"
(cd "$CHECK/report" && bash "$REPO/.claude/skills/latex-document-skill/scripts/compile_latex.sh" main.tex --verbose > build/compile-verbose.txt 2>&1)
uv run python - "$CHECK/report" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

report = Path(sys.argv[1])
log = (report / "build" / "compile-verbose.txt").read_text(encoding="utf-8", errors="replace")
final_pass = log.split("This is pdfTeX")[-1]
issues = re.findall(r"^.*(?:Overfull|Underfull|LaTeX Warning|Package \w+ Warning|^! ).*$", final_pass, re.M)
info = subprocess.run(["pdfinfo", str(report / "main.pdf")], capture_output=True, text=True).stdout
text = subprocess.run(["pdftotext", str(report / "main.pdf"), "-"], capture_output=True, text=True).stdout
pages = re.search(r"Pages:\s+(\d+)", info).group(1) if info else "none"
english = re.findall(r"\d\s*,?\s+and\s+\d|\b[Aa]ppendix\b", " ".join(text.split()))
print(f"final pass issues: {len(issues)}; pages: {pages}; unresolved ??: {text.count('??')}; English cross-reference words: {len(english)}")
for line in issues[:5]:
    print(line)
PY
```

Expected: exactly one line, `final pass issues: 0; pages: 30; unresolved ??: 0; English cross-reference words: 0`. Any further line is a warning or error of the final pass; fix it in `docs/report` and repeat this step.

- [ ] **Step 3: Commit**

```bash
git add docs/report/sections/abstract.tex docs/report/sections/introduction.tex docs/report/sections/related-work.tex docs/report/sections/discussion.tex docs/report/sections/conclusion.tex docs/report/references.bib docs/report/main.tex docs/report/sections/appendix.tex docs/report/tables/ext_compute.tex docs/report/tables/acceptance.tex
git commit -m "docs: complete the final report"
```

### Task 8: Reproducibility guide and checks

**Files:**
- Create: `docs/reproducibility.md`; generated `results/reproducibility/tables.json`, `results/reproducibility/scenarios.json`, `results/reproducibility/experiments-Agis.json`
- Modify: `README.md`

**Interfaces:**
- Consumes: `experiments/reproduce.py` (Task 2); the refreshed Phase 1 results (Task 3); `docs/report/data/ext_*` (Task 4); the report labels `tab:tran-adaptation` (Task 5), `tab:literature`, `tab:literature-stats`, `fig:quality-runtime`, `tab:case-study`, `fig:case-map`, `fig:case-progress` (Task 6), `tab:compute`, `tab:acceptance`, `sec:appendix` (Task 7); the raw data under `data/raw/` and `data/processed/networks/` downloaded by `python -m data.cli all`.
- Produces: the reproducibility guide with its table–figure map and acceptance map, the README section "Tái lập", and the three committed JSON reports of acceptance criterion 16.3.2.

Run the three levels after the Task 7 commit. `runner.provenance.git_state` looks only at `src`, `configs` and `experiments`, so the uncommitted guide and reports do not mark the runs dirty, and every report records `git_dirty` false. `Agis` is an evaluation topology of `v0`, so `--subset Agis` reruns `phase1`, `phase2_main`, `phase2_budget`, `phase2_sensitivity`, `phase2_design`, `phase2_literature` and `phase2_case_study` on its scenarios and skips `phase2_pilot` and `tran_pilot.py`, which run on the development split.

- [ ] **Step 1: Write the guide and the README section**

Create `docs/reproducibility.md`:

````markdown
# Gói tái lập

Tài liệu này mô tả cách dựng lại môi trường, dữ liệu, kết quả, bảng và hình của báo cáo, và cách kiểm tra tự động rằng các file đã commit được sinh lại đúng. Thiết kế nằm ở spec D Mục 11–13 (`docs/superpowers/specs/2026-09-12-extensions-final-delivery-design.md`).

## 1. Môi trường

- Python 3.12 trong `.venv` do uv quản lý; `uv.lock` khoá phiên bản package. Tạo môi trường bằng `uv sync --extra test`.
- Thư viện chính: `numpy`, `scipy` (HiGHS cho quy hoạch tuyến tính), `cvxpy` với Clarabel (phương pháp TRAN), `networkx`, `pyproj`, Pillow, `PyYAML`.
- Báo cáo dùng pdfLaTeX, biber và script `compile_latex.sh` của `.claude/skills/latex-document-skill`.
- Kết quả đã commit được chạy trên WSL2 với 12 CPU và 20 GB RAM; môi trường cụ thể của từng thí nghiệm (Python, numpy, scipy, HiGHS, cvxpy, Clarabel, số CPU) nằm trong `manifest.json` của thí nghiệm đó.
- Các lệnh giải bài toán TRAN chạy trong `systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0`, để một lỗi tràn bộ nhớ chỉ dừng đúng tiến trình đó.

## 2. Dữ liệu và giấy phép

Dữ liệu thô không nằm trong Git. Lệnh sau tải Topology Zoo (revision cố định), archive XML của SNDlib và archive validation của RescueNet (khoảng 2,37 GB), kiểm tra checksum và sinh dữ liệu đã xử lý:

```bash
uv run python -m data.cli all --profile configs/data/paper.yaml
uv run python -m data.cli verify --profile configs/data/paper.yaml
```

`data/manifests/sources.json` ghi URL, phiên bản, dung lượng và SHA-256 của từng file tải về. README của tác giả RescueNet ghi giấy phép CC BY-NC-ND cho nội dung dataset, nên kho này không phân phối lại ảnh, mask hay hình dựng từ chúng; kho chỉ lưu manifest lựa chọn cặp ảnh, manifest target và mã chuyển đổi.

## 3. Ba mức kiểm tra

`experiments/reproduce.py` ghi kết quả từng mức vào `results/reproducibility/<mức>.json` (danh sách kiểm tra đạt hoặc không đạt, commit, môi trường) và trả mã khác 0 khi có kiểm tra không đạt.

| Mức | Lệnh | Kiểm tra | Thời gian |
|---|---|---|---|
| `tables` | `uv run python experiments/reproduce.py --level tables` | Chạy lại thống kê và ba script sinh dữ liệu báo cáo vào thư mục tạm, so từng byte với `results/phase2/statistics*.csv` và `docs/report/data/` | vài giây |
| `scenarios` | `uv run python experiments/reproduce.py --level scenarios` | Kiểm tra checksum dữ liệu thô, sinh lại manifest target RescueNet và mọi họ scenario dưới một thư mục dữ liệu tạm, so với manifest đã commit và với SHA-256 ghi trong manifest kết quả | khoảng 20 giây |
| `experiments` | `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/reproduce.py --level experiments --subset Agis` | Chạy lại mọi cấu hình thí nghiệm trên các scenario của một topology, so từng dòng kết quả trên các cột không phải thời gian chạy | khoảng một giờ |

Bỏ `--subset` để chạy lại toàn bộ thí nghiệm và pilot TRAN; tổng thời gian tính đã ghi của từng thí nghiệm nằm trong `docs/report/data/ext_compute.csv` (bảng tính toán ở phụ lục báo cáo). Với `--subset`, bước đối chiếu MILP của Pha 1 được bỏ qua vì nó chọn instance trên toàn tập, còn pilot TRAN và `phase2_pilot` được bỏ qua vì chúng chạy trên tập phát triển, trong khi `--subset` nhận một topology đánh giá. `--experiments` giới hạn các cấu hình được chạy, và `--compare REFERENCE CANDIDATE` so hai thư mục kết quả bất kỳ theo cùng quy tắc.

Quy tắc so sánh dòng: `timely_count`, `connected_count` và `evaluate_calls` phải bằng nhau, `shortfall` lệch tối đa $10^{-9}$; giá trị cận LP lệch tương đối tối đa $10^{-6}$ và trạng thái phải bằng nhau; giá trị MILP chỉ được so khi cả hai lần chạy đều tối ưu; với pilot TRAN, số vòng lặp, trạng thái và tỷ lệ đúng hạn phải khớp. Chênh lệch trên máy khác được báo cáo, không bị lọc.

## 4. Nguồn gốc của kết quả

Mỗi script sinh dữ liệu báo cáo chỉ đọc một thí nghiệm khi manifest của nó hoàn tất và mô tả mã hiện tại (`runner.provenance.result_provenance_problem`): hoặc `source_hash` bằng mã hiện tại, hoặc kết quả đến từ một commit sạch là tổ tiên của HEAD và không file nào của `src/models`, `src/baselines`, `src/bounds`, `src/optimization`, `src/literature` bị sửa, xoá hay đổi tên kể từ commit đó. Package `runner` không nằm trong danh sách vì chỉ điều phối task; mức `experiments` kiểm tra nó từ đầu đến cuối. Kết quả `results/phase2/pilot/` của sub-project C là bản ghi lịch sử của lần chạy thử và không được báo cáo dùng.

## 5. Bảng, hình và nguồn dữ liệu

| Mục báo cáo | Dữ liệu | Script sinh | Thí nghiệm |
|---|---|---|---|
| Bảng tham số (`tab:params`), bảng MILP (`tab:phase1-milp`), hình thời gian LP (`fig:lp-runtime`) | `params.csv`, `phase1_milp.csv`, `lp_runtime.csv`, `phase1_results.tex` | `experiments/report_tables.py` | `configs/experiments/phase1.yaml` → `results/phase1/` |
| So sánh chính, thống kê, ablation, kết nối, số realization thiết kế (`tab:phase2-*`), ngân sách (`fig:budget`), độ nhạy (`fig:sensitivity`) | `phase2_*.csv`, `phase2_results.tex` | `experiments/report_tables_phase2.py` (thống kê: `experiments/phase2_statistics.py`) | `phase2_main`, `phase2_budget`, `phase2_sensitivity`, `phase2_design` → `results/phase2/` |
| Bảng tổng quan tài liệu (`tab:related`) | `related_work.csv` | viết tay từ `docs/literature/review.md` | --- |
| So sánh với TRAN (`tab:literature`, `tab:literature-stats`), chất lượng–thời gian (`fig:quality-runtime`) | `ext_literature.csv`, `ext_literature_statistics.csv`, `ext_quality_points.csv`, `phase2_budget_*.csv` | `experiments/report_tables_extensions.py` (thống kê: `experiments/phase2_literature_statistics.py`) | `experiments/tran_pilot.py` → `results/phase2/tran_pilot/`; `phase2_literature` → `results/phase2/literature/` |
| Case study (`tab:case-study`), bản đồ (`fig:case-map`), tiến trình giao cảnh báo (`fig:case-progress`) | `ext_case_study.csv`, `ext_map_*.csv`, `ext_progress.csv` | `experiments/report_tables_extensions.py` | `data.cli rescuenet-targets`, `data.cli scenarios --config configs/scenarios/rescuenet.yaml`, `phase2_case_study` → `results/phase2/case_study/`, `experiments/case_study_trace.py` → `results/phase2/case_study/trace/` |
| Thời gian tính (`tab:compute`) | `ext_compute.csv`, `ext_results.tex` | `experiments/report_tables_extensions.py` | mọi thí nghiệm ở trên |

## 6. Đối chiếu tiêu chí nghiệm thu

| Tiêu chí (`docs/issue.md` Mục 14) | Bằng chứng |
|---|---|
| Đúng đề số 5: có trajectory, relay selection và bandwidth allocation | Báo cáo Mục Mô hình và phát biểu bài toán; `src/models/plan.py` (quỹ đạo, lựa chọn đường, băng thông) |
| Objective đúng hạn được định nghĩa với release, deadline, số bit và xác suất | Báo cáo công thức mục tiêu; `src/models/metrics.py`, `src/models/evaluate.py` |
| Connectivity được định nghĩa theo thời gian và đường đến trung tâm | Báo cáo Mục phát biểu bài toán; `src/models/metrics.py` |
| Bài Huang được dùng đúng vai trò, khác biệt giao thức/mục tiêu được công khai | Báo cáo Mục Nghiên cứu liên quan và Mục phương pháp tìm kiếm (vì sao các khối không lồi, B3) |
| Dữ liệu có provenance, đơn vị và phép chuyển đổi tái lập | `data/manifests/`, `configs/data/`, `experiments/reproduce.py --level scenarios` |
| Baseline chạy trên benchmark công khai; thiếu nguồn nào được báo đúng | B0, B1 và TRAN trên kịch bản từ Topology Zoo và SNDlib (`tab:phase2-main`, `tab:literature`); giả định tiêu cự của ảnh FC2103 và thiếu GSD trong bài RescueNet được ghi ở spec D Mục 17 |
| Có ít nhất một cơ chế cải tiến và ablation chứng minh tác động | P và bảy biến thể ablation (`tab:phase2-ablation`), kết quả âm tính được báo cáo |
| Flow bảo toàn, buffer không âm, không dùng thông tin tương lai ngoài giả định | Bộ kiểm tra độc lập `src/models/checker.py` chạy ở mọi lần đánh giá cuối; `tests/models/` |
| Tài nguyên tổng và quỹ đạo khả thi; không tăng công suất ngầm theo quy mô | `validate_plan` trong `src/models/plan.py`; công suất cố định cho mọi phương pháp, kể cả TRAN |
| Có 20–30 seed, kiểm định, effect size và uncertainty report | 30 realization cho mỗi scenario; `tab:phase2-stats`, `tab:literature-stats` (kiểm định đổi dấu chính xác, Holm, bootstrap, rank-biserial) |
| Có runtime, robustness, feasibility, scalability và sensitivity | Mục thời gian tính, `fig:sensitivity`, `fig:quality-runtime`, `tab:case-study`, `tab:compute` |
| Có cận trên LP và gap trên mọi instance; instance nhỏ đối chiếu với MILP chính xác | `bounds.csv` của mọi thí nghiệm, cột gap của các bảng kết quả, `tab:phase1-milp` |
| Connectivity trên time-expanded graph là mục tiêu phụ từ điển, dùng thống nhất | Khoá từ điển trong `src/models/evaluate.py` dùng cho mọi phương pháp |
| B2 kế thừa đúng cấu trúc BCD của bài Huang; B3 tách được tác dụng của objective | Báo cáo Mục phương pháp tìm kiếm; so sánh B2 $-$ B3 trong `tab:phase2-stats` |
| Có gói tái lập và báo cáo theo cấu trúc môn học | Tài liệu này, `experiments/reproduce.py`, `results/reproducibility/*.json`; báo cáo có Tóm tắt, Giới thiệu, Nghiên cứu liên quan, Phương pháp, Kết quả, Thảo luận, Kết luận |

## 7. Chạy lại toàn bộ theo thứ tự

```bash
uv sync --extra test
uv run python -m data.cli all --profile configs/data/paper.yaml
uv run python -m data.cli rescuenet-targets --config configs/data/rescuenet_targets.yaml
for config in configs/scenarios/*.yaml; do uv run python -m data.cli scenarios --config "$config"; done
uv run python experiments/run_phase1.py --config configs/experiments/phase1.yaml
for name in phase2_main phase2_budget phase2_sensitivity phase2_design; do uv run python experiments/run_phase2.py --config configs/experiments/$name.yaml; done
uv run python experiments/phase2_statistics.py
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/tran_pilot.py
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/run_phase2.py --config configs/experiments/phase2_literature.yaml
uv run python experiments/phase2_literature_statistics.py
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/run_phase2.py --config configs/experiments/phase2_case_study.yaml
systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run python experiments/case_study_trace.py
uv run python experiments/report_tables.py
uv run python experiments/report_tables_phase2.py
uv run python experiments/report_tables_extensions.py
```

Sau đó biên dịch báo cáo bằng script của skill LaTeX như trong README. Nếu editor tự biên dịch `docs/report` khi file `.tex` thay đổi, hãy tắt tính năng đó hoặc biên dịch trong một bản sao ngoài workspace, vì hai lần biên dịch đồng thời ghi đè cùng `main.pdf`.
````

Append to the end of `README.md`:

````markdown


## Tái lập

Gói tái lập được mô tả trong `docs/reproducibility.md`: môi trường, dữ liệu và giấy phép, bảng nguồn dữ liệu của từng bảng và hình, bảng đối chiếu tiêu chí nghiệm thu và thứ tự chạy lại toàn bộ. Ba mức kiểm tra tự động ghi kết quả vào `results/reproducibility/`:

```bash
uv run python experiments/reproduce.py --level tables
uv run python experiments/reproduce.py --level scenarios
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/reproduce.py --level experiments --subset Agis
```
````

- [ ] **Step 2: Run the `tables` level**

Run: `uv run python experiments/reproduce.py --level tables`

Expected: exit code 0 after a few seconds; every check line starts with `PASS` (five generator commands, then `statistics/...` and `report data/...` files reported `identical`), and the last line is `{"level": "tables", "subset": null, "checks": 41, "passed": true}`.

- [ ] **Step 3: Run the `scenarios` level**

Run: `uv run python experiments/reproduce.py --level scenarios`

Expected: exit code 0 after about 20 seconds; every check line starts with `PASS`, and the last line is `{"level": "scenarios", "subset": null, "checks": 52, "passed": true}`.

- [ ] **Step 4: Run the `experiments` level on one evaluation topology**

Run: `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/reproduce.py --level experiments --subset Agis`

Expected: exit code 0 after about 50 minutes with 12 workers. While it runs, the runner prints one JSON line per experiment that runs: `task_count` 372 (`phase1`), 432 (`phase2_main`), 30 (`phase2_budget`), 495 (`phase2_sensitivity`), 3 (`phase2_design`), 192 (`phase2_literature`) and 195 (`phase2_case_study`), each with `"status": "complete"`. The case-study run also prints numpy `RuntimeWarning: invalid value encountered in reduce` lines, as the committed literature and case-study runs did; they are not failures. The checks follow:

```text
PASS results/phase1/method_realizations.csv: 360 rows match
PASS results/phase1/bounds.csv: 360 rows match
PASS phase2_pilot: skipped: no dev scenario of topology Agis
PASS results/phase2/main/method_realizations.csv: 2160 rows match
PASS results/phase2/main/bounds.csv: 900 rows match
PASS results/phase2/budget/method_realizations.csv: 900 rows match
PASS results/phase2/budget/bounds.csv: 0 rows match
PASS results/phase2/sensitivity/method_realizations.csv: 1350 rows match
PASS results/phase2/sensitivity/bounds.csv: 450 rows match
PASS results/phase2/design/method_realizations.csv: 90 rows match
PASS results/phase2/design/bounds.csv: 0 rows match
PASS results/phase2/literature/method_realizations.csv: 360 rows match
PASS results/phase2/literature/bounds.csv: 360 rows match
PASS results/phase2/case_study/method_realizations.csv: 450 rows match
PASS results/phase2/case_study/bounds.csv: 450 rows match
PASS tran_pilot: skipped: the pilot uses the development split
{"level": "experiments", "subset": "Agis", "checks": 16, "passed": true}
```

`phase2_budget` and `phase2_design` compute no bounds, so their committed `bounds.csv` files hold only a header and the empty reruns match.

If a check fails, stop and report its detail line; do not commit a failing report.

- [ ] **Step 5: Run the full suite and confirm the files**

Run:

```bash
systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q
git status --short --untracked-files=all results/reproducibility docs README.md
```

Expected: `283 passed, 1 skipped`, then exactly:

```text
 M README.md
?? docs/reproducibility.md
?? results/reproducibility/experiments-Agis.json
?? results/reproducibility/scenarios.json
?? results/reproducibility/tables.json
```

- [ ] **Step 6: Commit**

```bash
git add docs/reproducibility.md README.md results/reproducibility/tables.json results/reproducibility/scenarios.json results/reproducibility/experiments-Agis.json
git commit -m "docs: add the reproducibility guide and check reports"
```

## Spec Coverage

| Spec section | Task |
|---|---|
| 11 SAA framing | 5 |
| 12.1 `reproduce.py` levels `tables`, `scenarios`, `experiments`, `--subset`, JSON reports | 2 (code), 8 (runs and committed reports) |
| 12.2 `docs/reproducibility.md`: setup, data and licenses, table–figure map, recorded compute, acceptance map; README section | 8 (guide), 4 (`ext_compute.csv`), 7 (`tab:compute`, `tab:acceptance`) |
| 13 `report_tables_extensions.py`, `ext_*.csv`, `ext_results.tex` | 4 |
| 13 Method: TRAN adaptation, RescueNet scenario construction, SAA paragraph | 5 |
| 13 Experiments: literature comparison with statistics, case-study table, map and delivery-progress figures, quality–runtime points, RQ answers | 6 |
| 13 Abstract, introduction, related work, discussion with limitations, conclusion, appendix, bibliography check | 7 |
| 13 Compilation with the skill script in a copy outside the workspace | 5, 6, 7 (Step 2 of each) |
| 14 Error handling: reproduce reports differences and exits non-zero | 2 |
| 15.3 Testing: reproduce comparisons, report data | 2, 4 |
| 16.3.1 Full test suite | 1, 2, 3, 4, 8 |
| 16.3.2 `tables`, `scenarios` and `experiments --subset` pass; JSON reports committed | 8 |
| 16.3.3 Guide and README section | 8 |
| 16.3.4 Report compiles cleanly with every required part, no placeholders, final RQ answers | 5, 6, 7 |
| 17 Plan D.3 adjustments | 1 (provenance rule), 2 (subset filter by split, empty reruns fail), 3 (Phase 1 refresh), 5 (Vietnamese `cleveref` conjunctions), 6 (separate quality–runtime figure), 7 (RescueNet record), 8 (measured check counts) |
