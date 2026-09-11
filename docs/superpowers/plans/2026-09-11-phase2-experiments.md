# Phase 2 Experiments and Literature Implementation Plan (Sub-Project C, Plan C.2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the Phase 2 experiments of spec C Section 11 (main comparison with ablations, quality–runtime, sensitivity, design-realization count), compute the paired statistics of Section 12, and deliver the literature review and strong-baseline decision of Section 13, after making design scoring fast enough for these runs to fit in about 3.5 hours with 12 workers.

**Architecture:** `models.evaluate` gains `check` and `connectivity_cache`, which `DesignScorer` uses to skip the ledger checker and reuse connectivity during search; `runner` scenario sets gain a `replicates` filter. Thirteen sensitivity scenario configurations derive from `v0.yaml`; four experiment configurations drive `experiments/run_phase2.py`. A new package `src/analysis` holds the topology-level paired statistics, written to `results/phase2/statistics.csv` by `experiments/phase2_statistics.py`. The literature review lives in `docs/literature/`, with the comparison table in `docs/report/data/related_work.csv` and verified entries in `docs/report/references.bib`. All report writing belongs to plan C.3.

**Tech Stack:** Python 3.12 in the uv-managed `.venv`, `numpy`, `scipy` 1.18 (HiGHS, `scipy.stats.rankdata`), `PyYAML`, `pytest`. No new dependency.

**Spec:** `docs/superpowers/specs/2026-09-11-phase2-core-design.md`: Sections 11–13, 16.2, acceptance criteria 17.2.1–17.2.5, and Section 18 "Plan C.2 adjustments" (committed with this plan). Builds on plan C.1 (`docs/superpowers/plans/2026-09-11-phase2-methods.md`) and its pilot note `results/phase2/pilot/pilot_note.md`.

## Global Constraints

- Run project code through uv: tests with `uv run --extra test pytest`, scripts with `uv run python ...`. No dependency is added.
- Import boundaries: `src/models` imports none of `baselines`, `bounds`, `optimization`, `runner`, `analysis`; `src/optimization` imports `models` and `baselines`; `src/runner` imports `models`, `baselines`, `bounds`, `optimization`; `src/analysis` imports only the standard library, `numpy`, and `scipy`; nothing under `src` imports `experiments`.
- Design scoring runs `evaluate(..., check=False, connectivity_cache=<one dict per scorer>)`; every other call keeps the defaults (`check=True`, no cache), so the runner's final evaluation still runs the ledger checker. Search decisions do not change.
- Experiments use split `eval`, realization ids 0–29, 10 tangents, bootstrap 10,000 resamples with seed 20260911, and 12 workers.
- Sensitivity and design-realization experiments run replicate 0 of each evaluation topology (10 scenarios per set); the main and budget experiments run all 30 evaluation scenarios per set.
- A sensitivity family configuration differs from `configs/scenarios/v0.yaml` only in `set_id` and the fields of spec Section 11.3.
- Statistics: exact two-sided sign-flip test over all $2^{10}$ assignments on the 10 topology differences, Holm over the eight primary comparisons, 95% bootstrap interval of the mean difference over topologies, matched-pairs rank-biserial correlation without zero differences; columns `set_id, comparison, mean_difference, ci_low, ci_high, rank_biserial, p_value, p_holm, topology_count`.
- `results/phase2/**/shards/` stays Git-ignored; each experiment commits `method_realizations.csv`, `bounds.csv`, `summary.csv`, `manifest.json`; generated scenario files under `data/processed/` stay ignored while their manifests under `data/manifests/` are committed.
- Literature statements come only from text that was read, and bibliography metadata only from Crossref or arXiv records.
- Test module basenames are unique across `tests/`; every code task follows test-driven development and ends with its own commit.

---
### Task 1: Faster design scoring

**Files:**
- Modify: `src/models/evaluate.py`, `src/optimization/scoring.py`
- Test: `tests/models/test_design_extensions.py`, `tests/optimization/test_objectives_scoring.py`

**Interfaces:**
- Consumes: plan C.1 `evaluate`, `DesignScorer`, `models.metrics.connectivity`, `models.checker.check_ledger`.
- Produces: `evaluate(scenario, plan, mode, realization_ids=(), counter=None, candidates=None, keep_ledger=False, channel_namespace="eval", check=True, connectivity_cache=None)`; with `check=False` the ledger checker is skipped; `connectivity_cache` (a dict) memoizes connectivity under the keys `("without-uav",)` and `(trajectory bytes, realization_id, channel_namespace)`. `DesignScorer` passes `check=False` and one private cache per scorer.

- [ ] **Step 1: Write the failing tests**

In `tests/models/test_design_extensions.py` (2 edits):

Edit 1: replace

```python
from channel_builders import constant_channels
from models.channel import ACCESS, channel_uniforms
from models.evaluate import REALIZED, evaluate
```

with

```python
from channel_builders import constant_channels
from models import evaluate as evaluate_module
from models.channel import ACCESS, channel_uniforms
from models.checker import SimulatorInvariantError
from models.evaluate import REALIZED, evaluate
```

Edit 2: replace

```python
        assert result.feasible
```

with

```python
        assert result.feasible


def test_unchecked_cached_evaluation_matches_the_checked_result(monkeypatch):
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 12, 400000), make_alert("alert-0001", "s01", 1, 18, 50000)])
    candidates = candidate_paths(scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0, "alert-0001": 0}, EqualSplitBacklogged())
    checked = evaluate(scenario, plan, REALIZED, (0, 1), candidates=candidates)
    real, calls = evaluate_module.connectivity, []
    monkeypatch.setattr(evaluate_module, "connectivity", lambda scenario, channels: calls.append(channels is None) or real(scenario, channels))
    cache = {}
    first = evaluate(scenario, plan, REALIZED, (0, 1), candidates=candidates, check=False, connectivity_cache=cache)
    second = evaluate(scenario, plan, REALIZED, (0, 1), candidates=candidates, check=False, connectivity_cache=cache)
    assert first.key == second.key == checked.key
    assert [item.connected for item in second.realizations] == [item.connected for item in checked.realizations]
    assert [item.delivered_bits for item in second.realizations] == [item.delivered_bits for item in checked.realizations]
    assert calls == [True, False, False] and len(cache) == 3


def test_check_false_skips_only_the_ledger_checker(monkeypatch):
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 12, 400000)])
    candidates = candidate_paths(scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0}, EqualSplitBacklogged())
    monkeypatch.setattr(evaluate_module, "check_ledger", lambda *args: ["forced problem"])
    with pytest.raises(SimulatorInvariantError, match="forced problem"):
        evaluate(scenario, plan, REALIZED, (0,), candidates=candidates)
    assert evaluate(scenario, plan, REALIZED, (0,), candidates=candidates, check=False).feasible
```

In `tests/optimization/test_objectives_scoring.py` (2 edits):

Edit 1: replace

```python
from models.paths import candidate_paths
from optimization.objectives import leximin_key, lexicographic_key
```

with

```python
from models.paths import candidate_paths
from optimization import scoring
from optimization.objectives import leximin_key, lexicographic_key
```

Edit 2: replace

```python
    assert scorer.score(broken) == INFEASIBLE_SCORE
```

with

```python
    assert scorer.score(broken) == INFEASIBLE_SCORE


def test_scorer_skips_the_checker_and_reuses_one_connectivity_cache(monkeypatch):
    scenario, candidates, plan = _far_source_plan()
    real, seen = scoring.evaluate, []

    def spy(*args, **kwargs):
        seen.append((kwargs["check"], id(kwargs["connectivity_cache"])))
        return real(*args, **kwargs)

    monkeypatch.setattr(scoring, "evaluate", spy)
    scorer = DesignScorer(scenario, candidates, design_ids=(0, 1), budget=5)
    first = scorer.score(plan)
    assert scorer.score_with_ledger(plan)[0] == first
    assert [check for check, _ in seen] == [False, False] and len({cache for _, cache in seen}) == 1
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/models/test_design_extensions.py tests/optimization/test_objectives_scoring.py -v`

Expected: the three new tests fail: the two evaluator tests with `TypeError: evaluate() got an unexpected keyword argument 'check'`, and the scorer test with `KeyError: 'check'`.

- [ ] **Step 3: Add the checker switch and the connectivity cache**

In `src/models/evaluate.py` (4 edits):

Edit 1: replace

```python
import time
from typing import Mapping, Sequence
```

with

```python
import time
from typing import Callable, Mapping, Sequence, TypeVar

import numpy as np
```

Edit 2: replace

```python
INFEASIBLE_KEY = (-1, -1, -math.inf)
```

with

```python
INFEASIBLE_KEY = (-1, -1, -math.inf)


T = TypeVar("T")


def _memoized(cache: dict | None, key: tuple, compute: Callable[[], T]) -> T:
    if cache is None:
        return compute()
    if key not in cache:
        cache[key] = compute()
    return cache[key]
```

Edit 3: replace

```python
    channel_namespace: str = "eval",
) -> EvaluationResult:
    started = time.perf_counter()
```

with

```python
    channel_namespace: str = "eval",
    check: bool = True,
    connectivity_cache: dict | None = None,
) -> EvaluationResult:
    """Score a plan; `check=False` skips the ledger checker and `connectivity_cache` memoizes connectivity per trajectory and draw."""
    started = time.perf_counter()
```

Edit 4: replace

```python
    links = set(radio_links(candidates)) | set(all_uav_links(scenario))
    without_uav = connectivity(scenario, None)
    results = []
    for realization_id in realization_ids if mode == REALIZED else (None,):
        uniforms = None if realization_id is None else channel_uniforms(scenario, realization_id, channel_namespace)
        channels = build_link_channels(scenario, links, plan.trajectory, uniforms)
        outcome = simulate(scenario, candidates, plan, channels, realization_id)
        problems = check_ledger(scenario, candidates, plan, channels, outcome.ledger, outcome.delivered_bits)
        if problems:
            raise SimulatorInvariantError(problems)
        reach = connectivity(scenario, channels)
        results.append(
```

with

```python
    links = set(radio_links(candidates)) | set(all_uav_links(scenario))
    without_uav = _memoized(connectivity_cache, ("without-uav",), lambda: connectivity(scenario, None))
    results = []
    for realization_id in realization_ids if mode == REALIZED else (None,):
        uniforms = None if realization_id is None else channel_uniforms(scenario, realization_id, channel_namespace)
        channels = build_link_channels(scenario, links, plan.trajectory, uniforms)
        outcome = simulate(scenario, candidates, plan, channels, realization_id)
        if check:
            problems = check_ledger(scenario, candidates, plan, channels, outcome.ledger, outcome.delivered_bits)
            if problems:
                raise SimulatorInvariantError(problems)
        key = (np.asarray(plan.trajectory, dtype=float).tobytes(), realization_id, channel_namespace)
        reach = _memoized(connectivity_cache, key, lambda: connectivity(scenario, channels))
        results.append(
```

- [ ] **Step 4: Score designs without the checker and with one cache**

In `src/optimization/scoring.py` (2 edits):

Edit 1: replace

```python
        self.calls = 0
```

with

```python
        self.calls = 0
        self._connectivity: dict = {}
```

Edit 2: replace

```python
            channel_namespace=DESIGN_NAMESPACE,
        )
```

with

```python
            channel_namespace=DESIGN_NAMESPACE,
            check=False,
            connectivity_cache=self._connectivity,
        )
```

On development scenarios `Bbnplanet-r0` and `Fccn-r1`, B2 and P with budget 300 returned the same plans and keys as the plan C.1 code and ran 2.9 times faster (20.3 s to 6.9 s for B2 on `Bbnplanet-r0`).

- [ ] **Step 5: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/models/test_design_extensions.py tests/optimization/test_objectives_scoring.py -v`

Expected: 14 passed.

Run: `uv run --extra test pytest -q`

Expected: 173 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add src/models/evaluate.py src/optimization/scoring.py tests/models/test_design_extensions.py tests/optimization/test_objectives_scoring.py
git commit -m "perf: score designs without the ledger checker and with cached connectivity"
```

### Task 2: Replicate filter for scenario sets

**Files:**
- Modify: `src/runner/config.py`, `src/runner/experiment.py`
- Test: `tests/runner/test_runner_config.py`; create `tests/runner/test_runner_scenario_refs.py`

**Interfaces:**
- Consumes: plan C.1 `runner.config.ScenarioSetRef`, `load_experiment_config`, `runner.experiment.scenario_refs`; scenario manifests with a `replicate` column.
- Produces: `ScenarioSetRef(set_id, manifest, scenario_dir, replicates: tuple[int, ...] | None = None)`; an experiment configuration's scenario set may carry `replicates: [0, ...]` (distinct non-negative integers; otherwise `ValueError("scenario_sets[i].replicates: ...")`); `scenario_refs` keeps rows of the configured split whose `replicate` is listed, or all rows when `replicates` is absent.

- [ ] **Step 1: Write the failing tests**

Append to `tests/runner/test_runner_config.py`:

```python
def test_scenario_sets_accept_a_replicate_filter(tmp_path: Path):
    def keep_first_replicate(raw):
        raw["scenario_sets"][1]["replicates"] = [0, 2]

    config = load_experiment_config(_write(tmp_path, keep_first_replicate))
    assert [item.replicates for item in config.scenario_sets] == [None, (0, 2)]
    for bad in ([], [-1], [1, 1], ["0"], True):
        with pytest.raises(ValueError, match=r"scenario_sets\[0\].replicates"):
            load_experiment_config(_write(tmp_path, lambda raw, bad=bad: raw["scenario_sets"][0].update(replicates=bad)))
```

```python
# tests/runner/test_runner_scenario_refs.py
import csv
from pathlib import Path
from types import SimpleNamespace

from runner.config import ScenarioSetRef
from runner.experiment import scenario_refs


def test_scenario_refs_keep_the_split_and_the_listed_replicates(tmp_path: Path):
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    rows = [("B-r0", "eval", 0), ("A-r1", "eval", 1), ("A-r0", "eval", 0), ("C-r0", "dev", 0)]
    for scenario_id, _, _ in rows:
        (scenario_dir / f"{scenario_id}.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "scenarios.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario_id", "split", "replicate", "sha256"])
        writer.writerows([scenario_id, split, replicate, "0" * 64] for scenario_id, split, replicate in rows)

    def refs(replicates):
        config = SimpleNamespace(scenario_sets=(ScenarioSetRef("s", manifest, scenario_dir, replicates),), split="eval")
        return [ref.scenario_id for ref in scenario_refs(config)]

    assert refs(None) == ["A-r0", "A-r1", "B-r0"]
    assert refs((0,)) == ["A-r0", "B-r0"]
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/runner/test_runner_config.py tests/runner/test_runner_scenario_refs.py -v`

Expected: `test_scenario_sets_accept_a_replicate_filter` fails with `AttributeError: 'ScenarioSetRef' object has no attribute 'replicates'` and `test_scenario_refs_keep_the_split_and_the_listed_replicates` fails with `TypeError` (the dataclass takes three fields).

- [ ] **Step 3: Parse the replicate filter**

In `src/runner/config.py` (3 edits):

Edit 1: replace

```python
class ScenarioSetRef:
    set_id: str
    manifest: Path
    scenario_dir: Path
```

with

```python
class ScenarioSetRef:
    """A generated scenario set; `replicates` keeps only the listed replicate numbers (all when None)."""

    set_id: str
    manifest: Path
    scenario_dir: Path
    replicates: tuple[int, ...] | None = None
```

Edit 2: replace

```python
    return tuple(_frozen(item) for item in value) if isinstance(value, (list, tuple)) else value
```

with

```python
    return tuple(_frozen(item) for item in value) if isinstance(value, (list, tuple)) else value


def _replicates(item: dict, index: int) -> tuple[int, ...] | None:
    raw = item.get("replicates")
    if raw is None:
        return None
    if (
        not isinstance(raw, list)
        or not raw
        or len(set(raw)) != len(raw)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw)
    ):
        raise ValueError(f"scenario_sets[{index}].replicates: expected a list of distinct non-negative integers")
    return tuple(raw)
```

Edit 3: replace

```python
    sets = tuple(
        ScenarioSetRef(str(item["set_id"]), Path(item["manifest"]), Path(item["scenario_dir"]))
        for item in raw["scenario_sets"]
    )
```

with

```python
    sets = tuple(
        ScenarioSetRef(str(item["set_id"]), Path(item["manifest"]), Path(item["scenario_dir"]), _replicates(item, index))
        for index, item in enumerate(raw["scenario_sets"])
    )
```

- [ ] **Step 4: Apply the filter when listing scenarios**

In `src/runner/experiment.py` (1 edit):

Edit 1: replace

```python
            rows = sorted(
                (row for row in csv.DictReader(stream) if row["split"] == config.split),
                key=lambda row: row["scenario_id"],
```

with

```python
            rows = sorted(
                (
                    row
                    for row in csv.DictReader(stream)
                    if row["split"] == config.split
                    and (scenario_set.replicates is None or int(row["replicate"]) in scenario_set.replicates)
                ),
                key=lambda row: row["scenario_id"],
```

Task keys do not depend on the filter, so the Phase 1 key test keeps passing.

- [ ] **Step 5: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/runner/test_runner_config.py tests/runner/test_runner_scenario_refs.py -v`

Expected: 16 passed.

Run: `uv run --extra test pytest -q`

Expected: 175 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add src/runner/config.py src/runner/experiment.py tests/runner/test_runner_config.py tests/runner/test_runner_scenario_refs.py
git commit -m "feat: filter scenario sets by replicate"
```

### Task 3: Sensitivity scenario families

**Files:**
- Create: `configs/scenarios/sens-lref125k.yaml`, `sens-lref500k.yaml`, `sens-lref1m.yaml`, `sens-bh100.yaml`, `sens-physical.yaml`, `sens-deadline-short.yaml`, `sens-deadline-long.yaml`, `sens-btot05.yaml`, `sens-btot2.yaml`, `sens-alerts20.yaml`, `sens-alerts60.yaml`, `sens-k2.yaml`, `sens-k5.yaml` (all under `configs/scenarios/`)
- Create (generated): `data/manifests/scenarios_sens-*.csv` (13 files); scenario files under `data/processed/scenarios/sens-*/` stay ignored
- Test: `tests/data/test_sensitivity_configs.py`

**Interfaces:**
- Consumes: `configs/scenarios/v0.yaml`; `python -m data.cli scenarios --config <path>` from sub-project A.
- Produces: 13 scenario sets `sens-*` with 30 evaluation and 6 development scenarios each, manifests `data/manifests/scenarios_<set_id>.csv`, scenario files `data/processed/scenarios/<set_id>/<scenario_id>.json`.

| Family | Changed fields (besides `set_id`) |
|---|---|
| `sens-lref125k`, `sens-lref500k`, `sens-lref1m` | `workload.size_ref_bits`: 125000, 500000, 1000000 |
| `sens-bh100` | `backhaul_capacity_bps`: 100000.0 |
| `sens-physical` | `channel.uav.beta0_db` and `channel.ground.beta0_db`: -40.0; `spectrum.noise_psd_dbm_per_hz`: -164.0 |
| `sens-deadline-short`, `sens-deadline-long` | `workload.deadline_min_slots`/`deadline_max_slots`: 10/30 and 40/120 |
| `sens-btot05`, `sens-btot2` | `spectrum.b_tot_hz`: 500000.0, 2000000.0 |
| `sens-alerts20`, `sens-alerts60` | `workload.alert_count`: 20, 60 |
| `sens-k2`, `sens-k5` | `paths.k`: 2 with `paths.max_ground`: 1; `paths.k`: 5 |

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_sensitivity_configs.py
from pathlib import Path

import pytest
import yaml

FAMILIES = {
    "sens-lref125k": {"workload.size_ref_bits": 125000},
    "sens-lref500k": {"workload.size_ref_bits": 500000},
    "sens-lref1m": {"workload.size_ref_bits": 1000000},
    "sens-bh100": {"backhaul_capacity_bps": 100000.0},
    "sens-physical": {"channel.uav.beta0_db": -40.0, "channel.ground.beta0_db": -40.0, "spectrum.noise_psd_dbm_per_hz": -164.0},
    "sens-deadline-short": {"workload.deadline_min_slots": 10, "workload.deadline_max_slots": 30},
    "sens-deadline-long": {"workload.deadline_min_slots": 40, "workload.deadline_max_slots": 120},
    "sens-btot05": {"spectrum.b_tot_hz": 500000.0},
    "sens-btot2": {"spectrum.b_tot_hz": 2000000.0},
    "sens-alerts20": {"workload.alert_count": 20},
    "sens-alerts60": {"workload.alert_count": 60},
    "sens-k2": {"paths.k": 2, "paths.max_ground": 1},
    "sens-k5": {"paths.k": 5},
}


def _flatten(raw, prefix=""):
    flat = {}
    for key, value in raw.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, name + "."))
        else:
            flat[name] = value
    return flat


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_sensitivity_family_differs_from_v0_only_in_its_fields(family):
    base = _flatten(yaml.safe_load(Path("configs/scenarios/v0.yaml").read_text(encoding="utf-8")))
    changed = _flatten(yaml.safe_load(Path(f"configs/scenarios/{family}.yaml").read_text(encoding="utf-8")))
    expected = {"set_id": family, **FAMILIES[family]}
    assert set(changed) == set(base)
    assert {key for key in base if base[key] != changed[key]} == set(expected)
    assert {key: changed[key] for key in expected} == expected


def test_every_sensitivity_config_is_listed():
    assert {path.stem for path in Path("configs/scenarios").glob("sens-*.yaml")} == set(FAMILIES)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `uv run --extra test pytest tests/data/test_sensitivity_configs.py -v`

Expected: every parametrized case fails with `FileNotFoundError` for its `configs/scenarios/sens-*.yaml`, and `test_every_sensitivity_config_is_listed` fails on the empty set.

- [ ] **Step 3: Write the family configurations**

Each file is `v0.yaml` with only the listed lines changed, so the text is derived rather than retyped:

```bash
uv run python - <<'PY'
from pathlib import Path

base = Path("configs/scenarios/v0.yaml").read_text(encoding="utf-8")
FAMILIES = {
    "sens-lref125k": [("  size_ref_bits: 250000\n", "  size_ref_bits: 125000\n")],
    "sens-lref500k": [("  size_ref_bits: 250000\n", "  size_ref_bits: 500000\n")],
    "sens-lref1m": [("  size_ref_bits: 250000\n", "  size_ref_bits: 1000000\n")],
    "sens-bh100": [("backhaul_capacity_bps: 1000000.0\n", "backhaul_capacity_bps: 100000.0\n")],
    "sens-physical": [("    beta0_db: -50.0\n", "    beta0_db: -40.0\n"), ("  noise_psd_dbm_per_hz: -140.0\n", "  noise_psd_dbm_per_hz: -164.0\n")],
    "sens-deadline-short": [("  deadline_min_slots: 20\n", "  deadline_min_slots: 10\n"), ("  deadline_max_slots: 60\n", "  deadline_max_slots: 30\n")],
    "sens-deadline-long": [("  deadline_min_slots: 20\n", "  deadline_min_slots: 40\n"), ("  deadline_max_slots: 60\n", "  deadline_max_slots: 120\n")],
    "sens-btot05": [("  b_tot_hz: 1000000.0\n", "  b_tot_hz: 500000.0\n")],
    "sens-btot2": [("  b_tot_hz: 1000000.0\n", "  b_tot_hz: 2000000.0\n")],
    "sens-alerts20": [("  alert_count: 40\n", "  alert_count: 20\n")],
    "sens-alerts60": [("  alert_count: 40\n", "  alert_count: 60\n")],
    "sens-k2": [("  k: 3\n", "  k: 2\n"), ("  max_ground: 2\n", "  max_ground: 1\n")],
    "sens-k5": [("  k: 3\n", "  k: 5\n")],
}
for family, changes in FAMILIES.items():
    text = base.replace("set_id: v0\n", f"set_id: {family}\n", 1)
    for old, new in changes:
        expected = 2 if old == "    beta0_db: -50.0\n" else 1
        assert text.count(old) == expected, (family, old)
        text = text.replace(old, new)
    Path(f"configs/scenarios/{family}.yaml").write_text(text, encoding="utf-8")
print(len(FAMILIES), "sensitivity configs written")
PY
```

Expected: `13 sensitivity configs written`.

- [ ] **Step 4: Run the test and the full suite**

Run: `uv run --extra test pytest tests/data/test_sensitivity_configs.py -v`

Expected: 14 passed.

Run: `uv run --extra test pytest -q`

Expected: 189 passed, 1 skipped.

- [ ] **Step 5: Generate the families**

```bash
for config in configs/scenarios/sens-*.yaml; do
  uv run python -m data.cli scenarios --config "$config" || { echo "generation failed for $(basename "$config" .yaml)"; break; }
done
uv run python - <<'PY'
import csv
from pathlib import Path

for path in sorted(Path("data/manifests").glob("scenarios_sens-*.csv")):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    splits = {split: sum(row["split"] == split for row in rows) for split in ("eval", "dev")}
    connected = sum(float(row["ground_connected_source_fraction"]) for row in rows) / len(rows)
    print(path.stem.removeprefix("scenarios_"), len(rows), splits, f"{connected:.3f}")
PY
```

Expected: no `generation failed` line (each family takes about a second), then:

```text
sens-alerts20 36 {'eval': 30, 'dev': 6} 0.506
sens-alerts60 36 {'eval': 30, 'dev': 6} 0.506
sens-bh100 36 {'eval': 30, 'dev': 6} 0.506
sens-btot05 36 {'eval': 30, 'dev': 6} 0.506
sens-btot2 36 {'eval': 30, 'dev': 6} 0.506
sens-deadline-long 36 {'eval': 30, 'dev': 6} 0.506
sens-deadline-short 36 {'eval': 30, 'dev': 6} 0.506
sens-k2 36 {'eval': 30, 'dev': 6} 0.506
sens-k5 36 {'eval': 30, 'dev': 6} 0.506
sens-lref125k 36 {'eval': 30, 'dev': 6} 0.506
sens-lref1m 36 {'eval': 30, 'dev': 6} 0.506
sens-lref500k 36 {'eval': 30, 'dev': 6} 0.506
sens-physical 36 {'eval': 30, 'dev': 6} 1.000
```

The last column is the mean ground-connected source fraction. `sens-physical` reaches 1.0: every source can reach a ground node directly, so the family is degenerate for the role of the UAV and is reported as such (spec Section 18).

- [ ] **Step 6: Commit**

```bash
git add configs/scenarios/sens-*.yaml tests/data/test_sensitivity_configs.py data/manifests/scenarios_sens-*.csv
git commit -m "feat: add sensitivity scenario families"
```

### Task 4: Phase 2 experiment configurations

**Files:**
- Create: `configs/experiments/phase2_main.yaml`, `configs/experiments/phase2_budget.yaml`, `configs/experiments/phase2_sensitivity.yaml`, `configs/experiments/phase2_design.yaml`
- Test: `tests/runner/test_runner_phase2_configs.py`

**Interfaces:**
- Consumes: plan C.1 method specs and `SearchParams`; Task 2 `replicates`; Task 3 scenario sets.
- Produces: four configurations for `experiments/run_phase2.py` with outputs `results/phase2/main`, `results/phase2/budget`, `results/phase2/sensitivity`, `results/phase2/design`, expanding to 4320, 300, 4950, and 30 tasks.

- [ ] **Step 1: Write the failing tests**

```python
# tests/runner/test_runner_phase2_configs.py
from pathlib import Path

import pytest

from optimization.bcd import SearchParams
from runner.config import load_experiment_config
from runner.experiment import build_tasks, scenario_refs

REPAIR = {"operation1": True, "operation2": True}
MAIN_SEARCH = {
    "B2": SearchParams(),
    "B3": SearchParams(objective="leximin"),
    "P": SearchParams(**REPAIR),
    "P_op1": SearchParams(operation1=True),
    "P_op2": SearchParams(operation2=True),
    "P_fixed_paths": SearchParams(path_block=False, operation1=True),
    "P_fixed_trajectory": SearchParams(trajectory_block=False, **REPAIR),
    "P_equal_bandwidth": SearchParams(bandwidth_block=False, operation2=True),
    "P_no_backlog": SearchParams(backlog=False, **REPAIR),
    "P_expected_design": SearchParams(design_ids=(), **REPAIR),
}
SENSITIVITY_SETS = [
    "v0", "v0-bh50", "sens-bh100", "sens-lref125k", "sens-lref500k", "sens-lref1m", "sens-physical", "sens-deadline-short",
    "sens-deadline-long", "sens-btot05", "sens-btot2", "sens-alerts20", "sens-alerts60", "sens-k2", "sens-k5",
]


def _search(config):
    return {spec.id: spec.search_params() for spec in config.methods if spec.base == "search"}


def test_main_config_runs_every_method_and_ablation_on_both_sets():
    config = load_experiment_config(Path("configs/experiments/phase2_main.yaml"))
    assert [(item.set_id, item.replicates) for item in config.scenario_sets] == [("v0", None), ("v0-bh50", None)]
    assert (config.split, config.realization_ids, config.crosscheck, config.output_dir) == ("eval", tuple(range(30)), None, Path("results/phase2/main"))
    assert [spec.id for spec in config.methods] == ["B0", "B1", *MAIN_SEARCH]
    assert _search(config) == MAIN_SEARCH
    assert {spec.id for spec in config.methods if spec.bounds} == {"B2", "B3", "P"}


def test_budget_config_sweeps_b2_and_p():
    config = load_experiment_config(Path("configs/experiments/phase2_budget.yaml"))
    budgets = (250, 500, 1000, 2000, 4000)
    expected = {f"B2_b{budget}": SearchParams(budget=budget) for budget in budgets}
    expected |= {f"P_b{budget}": SearchParams(budget=budget, **REPAIR) for budget in budgets}
    assert [item.set_id for item in config.scenario_sets] == ["v0"] and config.scenario_sets[0].replicates is None
    assert _search(config) == expected and not any(spec.bounds for spec in config.methods)


def test_sensitivity_and_design_configs_use_one_replicate_per_topology():
    sensitivity = load_experiment_config(Path("configs/experiments/phase2_sensitivity.yaml"))
    assert [(item.set_id, item.replicates) for item in sensitivity.scenario_sets] == [(name, (0,)) for name in SENSITIVITY_SETS]
    assert [spec.id for spec in sensitivity.methods] == ["B1", "B2", "P"]
    assert _search(sensitivity) == {"B2": SearchParams(), "P": SearchParams(**REPAIR)}
    design = load_experiment_config(Path("configs/experiments/phase2_design.yaml"))
    assert [(item.set_id, item.replicates) for item in design.scenario_sets] == [("v0", (0,))]
    assert _search(design) == {
        "P_design_expected": SearchParams(design_ids=(), **REPAIR),
        "P_design1": SearchParams(design_ids=(0,), **REPAIR),
        "P_design10": SearchParams(design_ids=tuple(range(10)), **REPAIR),
    }


@pytest.mark.parametrize(
    ("name", "tasks"),
    [("phase2_main", 60 * (12 + 30 + 30)), ("phase2_budget", 30 * 10), ("phase2_sensitivity", 15 * 10 * (3 + 30)), ("phase2_design", 10 * 3)],
)
def test_phase2_configs_expand_to_their_task_counts(name, tasks):
    config = load_experiment_config(Path(f"configs/experiments/{name}.yaml"))
    if not all(item.manifest.exists() and item.scenario_dir.exists() for item in config.scenario_sets):
        pytest.skip("scenario sets are not generated in this checkout")
    keys = [task.key for task in build_tasks(config, scenario_refs(config))]
    assert len(keys) == len(set(keys)) == tasks
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/runner/test_runner_phase2_configs.py -v`

Expected: every test fails with `FileNotFoundError` for `configs/experiments/phase2_*.yaml`.

- [ ] **Step 3: Write the main configuration**

```yaml
# configs/experiments/phase2_main.yaml
experiment_id: phase2-main
scenario_sets:
  - set_id: v0
    manifest: data/manifests/scenarios_v0.csv
    scenario_dir: data/processed/scenarios/v0
  - set_id: v0-bh50
    manifest: data/manifests/scenarios_v0-bh50.csv
    scenario_dir: data/processed/scenarios/v0-bh50
split: eval
realization_ids: {start: 0, stop: 30}
methods:
  - B0
  - B1
  - {id: B2, base: search, bounds: true}
  - {id: B3, base: search, params: {objective: leximin}, bounds: true}
  - {id: P, base: search, params: {operation1: true, operation2: true}, bounds: true}
  - {id: P_op1, base: search, params: {operation1: true}}
  - {id: P_op2, base: search, params: {operation2: true}}
  - {id: P_fixed_paths, base: search, params: {path_block: false, operation1: true}}
  - {id: P_fixed_trajectory, base: search, params: {trajectory_block: false, operation1: true, operation2: true}}
  - {id: P_equal_bandwidth, base: search, params: {bandwidth_block: false, operation2: true}}
  - {id: P_no_backlog, base: search, params: {operation1: true, operation2: true, backlog: false}}
  - {id: P_expected_design, base: search, params: {operation1: true, operation2: true, design_ids: expected}}
bounds: {tangents: 10}
bootstrap: {resamples: 10000, seed: 20260911, confidence: 0.95}
workers: 12
output_dir: results/phase2/main
```

The seven ablations run without bounds; B0 and B1 keep their Phase 1 bound tasks.

- [ ] **Step 4: Write the budget, sensitivity, and design configurations**

```yaml
# configs/experiments/phase2_budget.yaml
experiment_id: phase2-budget
scenario_sets:
  - set_id: v0
    manifest: data/manifests/scenarios_v0.csv
    scenario_dir: data/processed/scenarios/v0
split: eval
realization_ids: {start: 0, stop: 30}
methods:
  - {id: B2_b250, base: search, params: {budget: 250}}
  - {id: B2_b500, base: search, params: {budget: 500}}
  - {id: B2_b1000, base: search, params: {budget: 1000}}
  - {id: B2_b2000, base: search, params: {budget: 2000}}
  - {id: B2_b4000, base: search, params: {budget: 4000}}
  - {id: P_b250, base: search, params: {operation1: true, operation2: true, budget: 250}}
  - {id: P_b500, base: search, params: {operation1: true, operation2: true, budget: 500}}
  - {id: P_b1000, base: search, params: {operation1: true, operation2: true, budget: 1000}}
  - {id: P_b2000, base: search, params: {operation1: true, operation2: true, budget: 2000}}
  - {id: P_b4000, base: search, params: {operation1: true, operation2: true, budget: 4000}}
bounds: {tangents: 10}
bootstrap: {resamples: 10000, seed: 20260911, confidence: 0.95}
workers: 12
output_dir: results/phase2/budget
```

```yaml
# configs/experiments/phase2_sensitivity.yaml
experiment_id: phase2-sensitivity
scenario_sets:
  - set_id: v0
    manifest: data/manifests/scenarios_v0.csv
    scenario_dir: data/processed/scenarios/v0
    replicates: [0]
  - set_id: v0-bh50
    manifest: data/manifests/scenarios_v0-bh50.csv
    scenario_dir: data/processed/scenarios/v0-bh50
    replicates: [0]
  - set_id: sens-bh100
    manifest: data/manifests/scenarios_sens-bh100.csv
    scenario_dir: data/processed/scenarios/sens-bh100
    replicates: [0]
  - set_id: sens-lref125k
    manifest: data/manifests/scenarios_sens-lref125k.csv
    scenario_dir: data/processed/scenarios/sens-lref125k
    replicates: [0]
  - set_id: sens-lref500k
    manifest: data/manifests/scenarios_sens-lref500k.csv
    scenario_dir: data/processed/scenarios/sens-lref500k
    replicates: [0]
  - set_id: sens-lref1m
    manifest: data/manifests/scenarios_sens-lref1m.csv
    scenario_dir: data/processed/scenarios/sens-lref1m
    replicates: [0]
  - set_id: sens-physical
    manifest: data/manifests/scenarios_sens-physical.csv
    scenario_dir: data/processed/scenarios/sens-physical
    replicates: [0]
  - set_id: sens-deadline-short
    manifest: data/manifests/scenarios_sens-deadline-short.csv
    scenario_dir: data/processed/scenarios/sens-deadline-short
    replicates: [0]
  - set_id: sens-deadline-long
    manifest: data/manifests/scenarios_sens-deadline-long.csv
    scenario_dir: data/processed/scenarios/sens-deadline-long
    replicates: [0]
  - set_id: sens-btot05
    manifest: data/manifests/scenarios_sens-btot05.csv
    scenario_dir: data/processed/scenarios/sens-btot05
    replicates: [0]
  - set_id: sens-btot2
    manifest: data/manifests/scenarios_sens-btot2.csv
    scenario_dir: data/processed/scenarios/sens-btot2
    replicates: [0]
  - set_id: sens-alerts20
    manifest: data/manifests/scenarios_sens-alerts20.csv
    scenario_dir: data/processed/scenarios/sens-alerts20
    replicates: [0]
  - set_id: sens-alerts60
    manifest: data/manifests/scenarios_sens-alerts60.csv
    scenario_dir: data/processed/scenarios/sens-alerts60
    replicates: [0]
  - set_id: sens-k2
    manifest: data/manifests/scenarios_sens-k2.csv
    scenario_dir: data/processed/scenarios/sens-k2
    replicates: [0]
  - set_id: sens-k5
    manifest: data/manifests/scenarios_sens-k5.csv
    scenario_dir: data/processed/scenarios/sens-k5
    replicates: [0]
split: eval
realization_ids: {start: 0, stop: 30}
methods:
  - B1
  - {id: B2, base: search}
  - {id: P, base: search, params: {operation1: true, operation2: true}}
bounds: {tangents: 10}
bootstrap: {resamples: 10000, seed: 20260911, confidence: 0.95}
workers: 12
output_dir: results/phase2/sensitivity
```

```yaml
# configs/experiments/phase2_design.yaml
experiment_id: phase2-design
scenario_sets:
  - set_id: v0
    manifest: data/manifests/scenarios_v0.csv
    scenario_dir: data/processed/scenarios/v0
    replicates: [0]
split: eval
realization_ids: {start: 0, stop: 30}
methods:
  - {id: P_design_expected, base: search, params: {operation1: true, operation2: true, design_ids: expected}}
  - {id: P_design1, base: search, params: {operation1: true, operation2: true, design_ids: [0]}}
  - {id: P_design10, base: search, params: {operation1: true, operation2: true, design_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}}
bounds: {tangents: 10}
bootstrap: {resamples: 10000, seed: 20260911, confidence: 0.95}
workers: 12
output_dir: results/phase2/design
```

- [ ] **Step 5: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/runner/test_runner_phase2_configs.py -v`

Expected: 7 passed (the task-count tests run because Task 3 generated the families).

Run: `uv run --extra test pytest -q`

Expected: 196 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add configs/experiments/phase2_main.yaml configs/experiments/phase2_budget.yaml configs/experiments/phase2_sensitivity.yaml configs/experiments/phase2_design.yaml tests/runner/test_runner_phase2_configs.py
git commit -m "feat: add phase 2 experiment configurations"
```

### Task 5: Paired topology-level statistics

**Files:**
- Create: `src/analysis/__init__.py`, `src/analysis/statistics.py`, `experiments/phase2_statistics.py`
- Test: `tests/analysis/test_statistics.py`, `tests/analysis/test_phase2_statistics_script.py`

**Interfaces:**
- Consumes: `method_realizations.csv` rows with `set_id, scenario_id, topology_id, method, timely_ratio`.
- Produces: `analysis.statistics.BOOTSTRAP_RESAMPLES = 10000`, `BOOTSTRAP_SEED = 20260911`, `STATISTICS_COLUMNS`; frozen dataclass `Comparison(set_id, first, second)` with `label` (`"P - B2"`); `PRIMARY_COMPARISONS` (P − B2, B2 − B1, B2 − B3, P − B1 on `v0`, then on `v0-bh50`); `topology_means(rows, set_id, method) -> dict[str, float]`; `sign_flip_p_value(differences) -> float` (1 to 20 values); `holm_adjust(p_values) -> list[float]`; `rank_biserial(differences) -> float` (`nan` when all differences are zero); `bootstrap_mean_ci(differences, resamples, seed, confidence=0.95) -> tuple[float, float]`; `compare_methods(rows, comparisons, resamples, seed) -> list[dict]` (raises `ValueError` when the two methods cover different topologies). `experiments/phase2_statistics.py --results results/phase2/main --output results/phase2/statistics.csv` returns 1 unless the results manifest has status `complete`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/analysis/test_statistics.py
import math

import numpy as np
import pytest

from analysis.statistics import (
    PRIMARY_COMPARISONS,
    STATISTICS_COLUMNS,
    Comparison,
    bootstrap_mean_ci,
    compare_methods,
    holm_adjust,
    rank_biserial,
    sign_flip_p_value,
    topology_means,
)


def rows_for(set_id, method, values_by_scenario):
    return [
        {"set_id": set_id, "scenario_id": scenario, "topology_id": scenario.split("-r")[0], "method": method,
         "realization_id": str(index), "timely_ratio": str(value)}
        for scenario, values in values_by_scenario.items()
        for index, value in enumerate(values)
    ]


def test_sign_flip_test_is_exact():
    assert sign_flip_p_value([0.1] * 10) == 2 / 1024
    assert sign_flip_p_value([0.2, -0.2] * 5) == 1.0
    assert sign_flip_p_value([0.3, 0.1, 0.2]) == 0.25
    with pytest.raises(ValueError, match="between 1 and 20"):
        sign_flip_p_value([0.1] * 21)


def test_holm_adjustment_matches_hand_examples():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_rank_biserial_matches_hand_examples():
    assert rank_biserial([1.0, -2.0, 3.0]) == pytest.approx(1 / 3)
    assert rank_biserial([0.0, 2.0, -2.0, 1.0]) == pytest.approx(1 / 6)
    assert math.isnan(rank_biserial([0.0, 0.0]))


def test_bootstrap_interval_is_deterministic_for_its_seed():
    differences = np.array([0.1, 0.3, -0.05, 0.2, 0.0, 0.15, 0.4, -0.1, 0.05, 0.25])
    first = bootstrap_mean_ci(differences, 2000, 20260911)
    assert first == bootstrap_mean_ci(differences, 2000, 20260911)
    assert first != bootstrap_mean_ci(differences, 2000, 7)
    assert first[0] < differences.mean() < first[1]
    assert bootstrap_mean_ci(np.full(10, 0.2), 500, 1) == pytest.approx((0.2, 0.2))


def test_topology_means_average_scenarios_then_topologies():
    rows = rows_for("v0", "B1", {"A-r0": [0.2, 0.4], "A-r1": [0.6, 0.6], "B-r0": [0.1, 0.3]})
    rows += rows_for("v0", "B2", {"A-r0": [1.0]}) + rows_for("v0-bh50", "B1", {"A-r0": [1.0]})
    assert topology_means(rows, "v0", "B1") == pytest.approx({"A": 0.45, "B": 0.2})


def test_compare_methods_reports_every_column_with_holm():
    topologies = [f"T{index}" for index in range(10)]
    rows = rows_for("v0", "B2", {f"{name}-r0": [0.5] for name in topologies})
    rows += rows_for("v0", "P", {f"{name}-r0": [0.75] for name in topologies})
    rows += rows_for("v0", "B1", {f"{name}-r0": [0.375 if index % 2 == 0 else 0.625] for index, name in enumerate(topologies)})
    first, second = compare_methods(rows, (Comparison("v0", "P", "B2"), Comparison("v0", "B2", "B1")), 500, 3)
    assert tuple(first) == STATISTICS_COLUMNS
    assert (first["comparison"], first["mean_difference"], first["topology_count"]) == ("P - B2", 0.25, 10)
    assert (first["ci_low"], first["ci_high"], first["rank_biserial"]) == pytest.approx((0.25, 0.25, 1.0))
    assert (first["p_value"], first["p_holm"]) == pytest.approx((2 / 1024, 4 / 1024))
    assert (second["comparison"], second["mean_difference"], second["rank_biserial"]) == ("B2 - B1", 0.0, 0.0)
    assert (second["p_value"], second["p_holm"]) == (1.0, 1.0)
    with pytest.raises(ValueError, match="same non-empty topologies"):
        compare_methods(rows[:-1], (Comparison("v0", "B2", "B1"),), 10, 1)


def test_primary_comparisons_are_the_eight_of_the_spec():
    assert [(item.set_id, item.label) for item in PRIMARY_COMPARISONS] == [
        (set_id, label) for set_id in ("v0", "v0-bh50") for label in ("P - B2", "B2 - B1", "B2 - B3", "P - B1")
    ]
```

```python
# tests/analysis/test_phase2_statistics_script.py
import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "phase2_statistics.py"
SPEC = importlib.util.spec_from_file_location("phase2_statistics", SCRIPT)
phase2_statistics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase2_statistics)
VALUES = {"B1": 0.25, "B2": 0.5, "B3": 0.375, "P": 0.5}


def _results(root: Path, status: str) -> Path:
    results = root / "main"
    results.mkdir()
    (results / "manifest.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    with (results / "method_realizations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["set_id", "scenario_id", "topology_id", "method", "realization_id", "timely_ratio"])
        writer.writeheader()
        for set_id in ("v0", "v0-bh50"):
            for topology in range(10):
                for method, value in VALUES.items():
                    writer.writerow({"set_id": set_id, "scenario_id": f"T{topology}-r0", "topology_id": f"T{topology}",
                                     "method": method, "realization_id": "0", "timely_ratio": str(value)})
    return results


def test_statistics_script_writes_the_eight_primary_comparisons(tmp_path: Path):
    output = tmp_path / "statistics.csv"
    assert phase2_statistics.main(["--results", str(_results(tmp_path, "complete")), "--output", str(output)]) == 0
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["set_id"], row["comparison"]) for row in rows][:4] == [("v0", "P - B2"), ("v0", "B2 - B1"), ("v0", "B2 - B3"), ("v0", "P - B1")]
    assert len(rows) == 8 and float(rows[1]["mean_difference"]) == 0.25 and float(rows[0]["p_value"]) == 1.0
    assert float(rows[1]["p_holm"]) == min(1.0, 8 * 2 / 1024)


def test_statistics_script_refuses_incomplete_results(tmp_path: Path, capsys):
    output = tmp_path / "statistics.csv"
    assert phase2_statistics.main(["--results", str(_results(tmp_path, "failed")), "--output", str(output)]) == 1
    assert "no complete experiment" in capsys.readouterr().err and not output.exists()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/analysis -v`

Expected: collection fails: `ModuleNotFoundError: No module named 'analysis'` and `FileNotFoundError` for `experiments/phase2_statistics.py`.

- [ ] **Step 3: Implement the statistics**

```python
# src/analysis/__init__.py
"""Statistical analysis of experiment results; imports only numpy and scipy."""
```

```python
# src/analysis/statistics.py
"""Paired topology-level comparisons of spec C Section 12."""

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260911
MAX_EXACT_TOPOLOGIES = 20
TIE_TOLERANCE = 1e-12
STATISTICS_COLUMNS = (
    "set_id", "comparison", "mean_difference", "ci_low", "ci_high", "rank_biserial", "p_value", "p_holm", "topology_count",
)


@dataclass(frozen=True)
class Comparison:
    set_id: str
    first: str
    second: str

    @property
    def label(self) -> str:
        return f"{self.first} - {self.second}"


PRIMARY_COMPARISONS = tuple(
    Comparison(set_id, first, second)
    for set_id in ("v0", "v0-bh50")
    for first, second in (("P", "B2"), ("B2", "B1"), ("B2", "B3"), ("P", "B1"))
)


def topology_means(rows: Sequence[Mapping[str, str]], set_id: str, method: str) -> dict[str, float]:
    """Mean timely ratio per scenario over its realizations, then averaged over the scenarios of each topology."""
    per_scenario: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["set_id"] == set_id and row["method"] == method:
            per_scenario[(row["topology_id"], row["scenario_id"])].append(float(row["timely_ratio"]))
    per_topology: dict[str, list[float]] = defaultdict(list)
    for (topology, _), values in per_scenario.items():
        per_topology[topology].append(sum(values) / len(values))
    return {topology: sum(values) / len(values) for topology, values in sorted(per_topology.items())}


def sign_flip_p_value(differences: Sequence[float]) -> float:
    """Exact two-sided sign-flip test: share of all 2^n sign assignments whose |mean| reaches the observed |mean|."""
    values = np.asarray(differences, dtype=float)
    count = len(values)
    if not 1 <= count <= MAX_EXACT_TOPOLOGIES:
        raise ValueError(f"the exact test needs between 1 and {MAX_EXACT_TOPOLOGIES} paired values, got {count}")
    signs = ((np.arange(2**count)[:, None] >> np.arange(count)) & 1) * 2 - 1
    flipped = np.abs(signs @ values) / count
    return float(np.mean(flipped >= abs(values.sum()) / count - TIE_TOLERANCE))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    count = len(p_values)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(sorted(range(count), key=lambda item: p_values[item])):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def rank_biserial(differences: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation (W+ - W-)/(W+ + W-) without zero differences; nan when all are zero."""
    values = np.asarray([value for value in differences if value != 0.0], dtype=float)
    if len(values) == 0:
        return math.nan
    ranks = rankdata(np.abs(values))
    positive, negative = ranks[values > 0].sum(), ranks[values < 0].sum()
    return float((positive - negative) / (positive + negative))


def bootstrap_mean_ci(differences: Sequence[float], resamples: int, seed: int, confidence: float = 0.95) -> tuple[float, float]:
    """Percentile interval of the mean over topologies resampled with replacement."""
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def compare_methods(
    rows: Sequence[Mapping[str, str]], comparisons: Sequence[Comparison], resamples: int, seed: int
) -> list[dict[str, object]]:
    """One row per comparison with Holm adjustment across all given comparisons."""
    results = []
    for comparison in comparisons:
        first = topology_means(rows, comparison.set_id, comparison.first)
        second = topology_means(rows, comparison.set_id, comparison.second)
        if not first or first.keys() != second.keys():
            raise ValueError(f"{comparison.set_id} {comparison.label}: methods must cover the same non-empty topologies")
        differences = np.array([first[topology] - second[topology] for topology in first])
        low, high = bootstrap_mean_ci(differences, resamples, seed)
        results.append(
            {
                "set_id": comparison.set_id,
                "comparison": comparison.label,
                "mean_difference": float(differences.mean()),
                "ci_low": low,
                "ci_high": high,
                "rank_biserial": rank_biserial(differences),
                "p_value": sign_flip_p_value(differences),
                "p_holm": 0.0,
                "topology_count": len(differences),
            }
        )
    for row, adjusted in zip(results, holm_adjust([row["p_value"] for row in results])):
        row["p_holm"] = adjusted
    return results
```

- [ ] **Step 4: Add the statistics entry point**

```python
# experiments/phase2_statistics.py
"""Write the paired comparisons of spec C Section 12 from a completed Phase 2 main experiment."""

import argparse
import csv
import json
from pathlib import Path
import sys

from analysis.statistics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, PRIMARY_COMPARISONS, STATISTICS_COLUMNS, compare_methods


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase2-statistics")
    parser.add_argument("--results", type=Path, default=Path("results/phase2/main"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2/statistics.csv"))
    args = parser.parse_args(argv)
    manifest = args.results / "manifest.json"
    if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8"))["status"] != "complete":
        print(f"{args.results} holds no complete experiment; run experiments/run_phase2.py first", file=sys.stderr)
        return 1
    with (args.results / "method_realizations.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    results = compare_methods(rows, PRIMARY_COMPARISONS, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(STATISTICS_COLUMNS))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"comparisons": len(results), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/analysis -v`

Expected: 9 passed.

Run: `uv run --extra test pytest -q`

Expected: 205 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add src/analysis experiments/phase2_statistics.py tests/analysis
git commit -m "feat: add paired topology-level statistics"
```

### Task 6: Literature review and strong-baseline decision

**Files:**
- Create: `docs/literature/review.md`, `docs/literature/screening.csv`, `docs/report/data/related_work.csv`
- Modify: `docs/report/references.bib` (23 verified entries appended)

**Interfaces:**
- Consumes: spec Section 13 protocol; `docs/references/sensors-25-07443-v2.md`.
- Produces: 16 included papers with bibliography keys `huang2025ddatsap`, `tran2022relay`, `samir2020time`, `albusalih2018deadline`, `liu2022aoi`, `liu2024interference`, `tang2022delay`, `he2022delay`, `cao2023priority`, `fadlullah2016dynamic`, `du2023timeconstrained`, `wang2026juror`, `kalantari2017backhaul`, `selim2018postdisaster`, `yu2023backhaul`, `queiros2024predictive`; background keys `wu2018delay`, `park2026expected`, `wu2018multi`, `zeng2016mobile`, `na2020emergency`, `li2024flight`, `ropke2006alns`, `jain2004dtn`; the semicolon-separated table with columns `key;year;venue;objective;uavs;relay_selection;deadline;connectivity;channel_uncertainty;bandwidth_power;method;guarantee` for plan C.3; the decision that Tran et al. (half-duplex variant) is the strong literature baseline.

The review was carried out while writing this plan (searches, screening, and Crossref/arXiv checks on 2026-09-11); this task records it.

- [ ] **Step 1: Write the review**

```markdown
# Literature Review and Strong-Baseline Selection

**Date:** 2026-09-11
**Spec:** `docs/superpowers/specs/2026-09-11-phase2-core-design.md` Section 13
**Outputs:** comparison table `docs/report/data/related_work.csv`, screening record `docs/literature/screening.csv` (every returned record with its query, identifier, decision, and reason), bibliography entries in `docs/report/references.bib`

## 1. Protocol

### 1.1 Sources and queries

The six topics of spec Section 13 were searched on 2026-09-11 with fixed queries on Crossref (`query.bibliographic`, journal articles published from 2016, 15 rows each) and arXiv (conjunctive all-field terms with `UAV OR drone`, 15 results each). Every returned record is listed in `screening.csv`.

| Id | Topic | Query |
|---|---|---|
| Q1 | Deadline-aware UAV relaying | `deadline UAV relay trajectory` |
| Q2 | Relay selection with backhaul | `UAV relay selection backhaul` |
| Q3 | Time-expanded network flow | `time-expanded network flow UAV` |
| Q4 | Stochastic timely delivery | `UAV data collection deadline stochastic channel` |
| Q5 | Joint trajectory and bandwidth design | `joint UAV trajectory bandwidth allocation relay` |
| Q6 | Decomposition with repair | `UAV trajectory decomposition repair heuristic delay` |

A general web search engine was queried with:

| Id | Query |
|---|---|
| W1 | deadline-aware UAV relay trajectory optimization emergency communication timely delivery |
| W2 | UAV relay selection ground relay backhaul capacity joint trajectory disaster network |
| W3 | UAV data collection time-constrained IoT devices deadlines trajectory maximize served devices |
| W4 | joint UAV trajectory and bandwidth allocation relay network block coordinate descent |
| W5 | time-expanded graph UAV data ferrying store-carry-forward delay tolerant network flow |
| W6 | UAV-assisted network probabilistic line-of-sight robust trajectory stochastic channel deadline constrained |
| W7 | backhaul-aware UAV drone base station placement 5G disaster recovery ground relay |
| W8 | UAV mobile relaying store-and-forward information causality delay constraint throughput |
| W9 | UAV-assisted post-disaster emergency network connectivity restoration multi-hop ground nodes trajectory |
| W10 | UAV data collection hard deadline packets scheduling trajectory "deadline" wireless sensor network IEEE |

### 1.2 Screening and criteria

Candidates were screened first by title and then by abstract. The inclusion criterion of the spec — UAV relaying or data collection with time constraints, or relay selection with backhaul — was applied with these definitions:

- **Time constraint:** the abstract states per-demand timing: a deadline, a delay or Age-of-Information limit, delay as an objective, a message time-to-live, or a count of demands served within a stated period. Minimizing mission or flight time, or maximizing throughput over a horizon under buffer causality, does not qualify.
- **Relay selection with backhaul:** the abstract states a choice among ground stations or relays, or an explicit backhaul capacity for aerial nodes.
- **UAV relaying or data collection:** data moves from ground sources through a UAV to a destination, or a UAV collects data from ground nodes. UAV base stations serving users, computation offloading, and reflecting-surface access qualify only through the backhaul criterion.
- The reference paper of the course topic (Huang et al.) is included regardless of the criteria.

### 1.3 Verification

- **Metadata** comes from the Crossref record of the DOI, or from the arXiv record for preprints; every bibliography entry was generated from those records on 2026-09-11.
- **Statements** come only from text that was read: abstracts from arXiv, Semantic Scholar, OpenAlex, or Crossref, and the full text of Huang et al. (`docs/references/sensors-25-07443-v2.md`). A table cell reads *Không nêu* (not stated) when that text is silent.
- A candidate whose abstract could not be retrieved from any of these sources was excluded, because none of its statements could be checked.

### 1.4 Limitations

One reviewer screened all records; extraction is at abstract level except for the reference paper; web search results depend on the engine and date; the conjunctive arXiv queries returned few records.

## 2. Included Papers

Sixteen papers are included. Screening counts: 74 records excluded by title, 18 excluded by abstract, 15 included entries in `screening.csv` (the reference paper is added separately).

| Key | Reference | Criterion | Text read |
|---|---|---|---|
| `huang2025ddatsap` | Huang et al., “Dynamic Dual-Antenna Time-Slot Allocation Protocol for UAV-Aided Relaying System Under Probabilistic LoS-Channel”, *Sensors* 25(24):7443, 2025, DOI 10.3390/s25247443 | reference paper of the course topic | full text (docs/references/sensors-25-07443-v2.md) |
| `tran2022relay` | Tran et al., “UAV Relay-Assisted Emergency Communications in IoT Networks: Resource Allocation and Trajectory Optimization”, *IEEE Transactions on Wireless Communications* 21(3):1621–1637, 2022, DOI 10.1109/twc.2021.3105821 | time: per-device latency; counts devices served on time | arXiv abstract (2008.00218) |
| `samir2020time` | Samir et al., “UAV Trajectory Planning for Data Collection from Time-Constrained IoT Devices”, *IEEE Transactions on Wireless Communications* 19(1):34–46, 2020, DOI 10.1109/twc.2019.2940447 | time: per-device deadline | Semantic Scholar abstract |
| `albusalih2018deadline` | Albu-Salih and Seno, “Optimal UAV Deployment for Data Collection in Deadline-based IoT Applications”, *Baghdad Science Journal* 15(4), 2018, DOI 10.21123/bsj.2018.15.4.0484 | time: collection deadline | Crossref abstract |
| `liu2022aoi` | Liu and Zheng, “UAV Trajectory Optimization for Time-Constrained Data Collection in UAV-Enabled Environmental Monitoring Systems”, *IEEE Internet of Things Journal* 9(23):24300–24314, 2022, DOI 10.1109/jiot.2022.3189214 | time: AoI limit of monitored data | Semantic Scholar abstract |
| `liu2024interference` | Liu and Zheng, “UAV Trajectory Planning With Interference Awareness in UAV-Enabled Time-Constrained Data Collection Systems”, *IEEE Transactions on Vehicular Technology* 73(2):2799–2815, 2024, DOI 10.1109/tvt.2023.3320676 | time: data time constraint; station association | Semantic Scholar abstract |
| `tang2022delay` | Tang et al., “Delay-Tolerant UAV-Assisted Communication: Online Trajectory Design and User Association”, *IEEE Transactions on Vehicular Technology* 71(12):13137–13151, 2022, DOI 10.1109/tvt.2022.3195788 | time: finite queueing delay; user association | Semantic Scholar abstract |
| `he2022delay` | He et al., “Trajectory Optimization and Channel Allocation for Delay Sensitive Secure Transmission in UAV-Relayed VANETs”, *IEEE Transactions on Vehicular Technology* 71(4):4512–4517, 2022, DOI 10.1109/tvt.2022.3144178 | time: delay objective; UAV relaying | Semantic Scholar abstract |
| `cao2023priority` | Cao et al., “Energy-Delay Tradeoff for Dynamic Trajectory Planning in Priority-Oriented UAV-Aided IoT Networks”, *IEEE Transactions on Green Communications and Networking* 7(1):158–170, 2023, DOI 10.1109/tgcn.2022.3196670 | time: priority-weighted delay objective | OpenAlex abstract |
| `fadlullah2016dynamic` | Fadlullah et al., “A dynamic trajectory control algorithm for improving the communication throughput and delay in UAV-aided networks”, *IEEE Network* 30(1):100–105, 2016, DOI 10.1109/mnet.2016.7389838 | time: congestion delay objective; UAV relaying | OpenAlex abstract |
| `du2023timeconstrained` | Du et al., “Time-Constrained UAV-Aided Data Collection for IoT Networks with Energy Harvesting”, *IEEE INFOCOM 2023 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS)*, pp. 1–6, 2023, DOI 10.1109/infocomwkshps57453.2023.10226065 | time: devices served within the collection period | OpenAlex abstract |
| `wang2026juror` | Wang and Yang, “Joint UAV Flight and Opportunistic Routing under Reinforcement Learning for Delay-Tolerant Networks”, *arXiv preprint*, 2026, arXiv:2608.04590 | time: message time-to-live; store-carry-forward | arXiv abstract (2608.04590) |
| `kalantari2017backhaul` | Kalantari et al., “Backhaul-aware robust 3D drone placement in 5G+ wireless networks”, *2017 IEEE International Conference on Communications Workshops (ICC Workshops)*, pp. 109–114, 2017, DOI 10.1109/iccw.2017.7962642 | backhaul capacity | arXiv abstract (1702.08395) |
| `selim2018postdisaster` | Selim and Kamal, “Post-Disaster 4G/5G Network Rehabilitation Using Drones: Solving Battery and Backhaul Issues”, *2018 IEEE Globecom Workshops (GC Wkshps)*, pp. 1–6, 2018, DOI 10.1109/glocomw.2018.8644135 | backhaul capacity | arXiv abstract (1809.07859) |
| `yu2023backhaul` | Yu et al., “Backhaul-Aware Drone Base Station Placement and Resource Management for FSO-Based Drone-Assisted Mobile Networks”, *IEEE Transactions on Network Science and Engineering* 10(3):1659–1668, 2023, DOI 10.1109/tnse.2022.3233004 | backhaul capacity | arXiv abstract (2112.12883) |
| `queiros2024predictive` | Queiros et al., “Joint Channel Bandwidth Assignment and Relay Positioning for Predictive Flying Networks”, *2024 IEEE Globecom Workshops (GC Wkshps)*, pp. 1–6, 2024, DOI 10.1109/gcwkshp64532.2024.11100616 | backhaul relaying | arXiv abstract (2503.14248) |

## 3. Summaries

- **`huang2025ddatsap`.** A UAV decode-and-forward relay with an information buffer serves several two-way user pairs under the PrLoS channel. The DDATSAP protocol shares two antennas within a slot; the resource scheduling factor, transmit power, and trajectory are optimized by BCD with SCA to maximize the minimum average message rate. Section 5.4 gives a per-iteration interior-point complexity of O(R_max (KI)^3) and argues that relaxed auxiliary variables are tight at convergence.
- **`tran2022relay`.** A UAV collects data from latency-constrained IoT devices in an emergency and forwards it to a ground gateway, with limited on-board storage; full-duplex and half-duplex relaying are compared. A device counts as served only if its data reaches the gateway in time. The number of served devices is maximized over bandwidth, power, and trajectory by relaxing binary variables and applying an inner-approximation iterative algorithm; a second problem maximizes throughput for a given number of served devices.
- **`samir2020time`.** A UAV collects data from IoT devices that each have a hard upload deadline. Trajectory and radio resources are optimized to maximize the number of served devices; the mixed-integer non-convex problem is NP-hard. A branch, reduce and bound algorithm gives the global optimum for small instances and an SCA-based algorithm handles larger ones; greedy distance and deadline heuristics are the benchmarks.
- **`albusalih2018deadline`.** Multiple UAVs collect data from IoT nodes within a given deadline while minimizing the energy of nodes and UAVs. The deployment is written as a mixed integer linear program and a heuristic addresses its computational cost.
- **`liu2022aoi`.** A UAV collects time-constrained data in monitoring areas and delivers it to a ground base station. Mission completion time is minimized over speeds, hovering positions, and visiting order subject to Age-of-Information limits and on-board energy, by decomposing into speed (SCA) and path (genetic algorithm) subproblems.
- **`liu2024interference`.** A UAV collects data under data time constraints and delivers it to one of several ground base stations. Mission completion time is minimized over station association, speed, and path with energy and interference constraints, using SCA-based subproblems inside block coordinate descent.
- **`tang2022delay`.** In a delay-tolerant UAV-assisted network with random demand and mobility, UAV trajectories and user association are chosen online to keep users' queueing delay finite while minimizing UAV energy. Lyapunov optimization turns the stochastic problem into per-slot problems, which are decomposed using their graph structure.
- **`he2022delay`.** A UAV relays secure transmissions in a vehicular network. Total information delay is minimized over the relay trajectory (Newton method) and channel allocation (relax-and-round with sequential convex approximation) in an alternating framework.
- **`cao2023priority`.** A UAV collects time-sensitive data from mobile sensor nodes with different delay priorities. A weighted sum of UAV energy and nodes' average delay is minimized through an online reinforcement-learning-based heuristic trajectory planner with attention.
- **`fadlullah2016dynamic`.** Several UAVs form a relay chain for disaster-affected users; the distance between neighbouring trajectory centers drives delay and throughput. UAVs whose queue occupancy exceeds a threshold move their trajectory center and radius; simulations and field experiments evaluate the rule.
- **`du2023timeconstrained`.** Energy-harvesting IoT devices upload data to a UAV by TDMA within a finite gathering period. The number of served devices is maximized over the trajectory, time allocation, and transmit power, using a penalty reformulation with alternating optimization, SCA, and quadratic approximation to reach a sub-optimal solution.
- **`wang2026juror`.** In a delay-tolerant network with intermittent contacts, finite buffers, and message time-to-live, UAV headings and opportunistic routing are learned jointly under centralized training and decentralized execution with proximal policy optimization; the method is compared with PRoPHET and MaxProp.
- **`kalantari2017backhaul`.** The 3D placement of a drone base station is optimized with the capacity of different wireless backhaul types taken into account, maximizing served users (network-centric) or sum rate (user-centric); robustness to user displacement is examined.
- **`selim2018postdisaster`.** A grid of drones restores cellular coverage after a disaster, using tethered drones for high-capacity backhaul and powering drones for charging. Drone energy is minimized over placement while guaranteeing a minimum user rate.
- **`yu2023backhaul`.** A drone base station relays traffic between a macro base station and users over a free-space-optics backhaul whose capacity may be insufficient. Bandwidth allocation and drone placement are optimized jointly under the backhaul capacity constraint by the BROAD algorithm.
- **`queiros2024predictive`.** A second-tier flying relay forwards traffic of first-tier flying nodes whose trajectories are known in advance. Bandwidth assignment and relay position are optimized under positioning limits, finite bandwidth, and minimum rates by simulated annealing with penalty functions.

## 4. Comparison Table

`docs/report/data/related_work.csv` (semicolon-separated, Vietnamese cells for the report) has the columns `key, year, venue, objective, uavs, relay_selection, deadline, connectivity, channel_uncertainty, bandwidth_power, method, guarantee`. Cells follow Sections 2–3; *Không nêu* means the text read does not state it.

None of the included papers combines per-demand deadlines with a choice of ground relays over a finite-capacity backhaul, which is the combination studied in this project; papers with deadlines assume a single destination, and papers with backhaul constraints have no per-demand timing.

## 5. Excluded After Reading the Abstract

| Key or reference | Reason | Use |
|---|---|---|
| `wu2018delay`: Wu and Zhang, “Common Throughput Maximization in UAV-Enabled OFDMA Systems With Delay Consideration”, *IEEE Transactions on Communications* 66(12):6614–6627, 2018, DOI 10.1109/tcomm.2018.2865922 | UAV base station serving users (downlink OFDMA), not relaying or data collection | background: throughput–delay trade-off |
| `park2026expected`: Park and Lee, “Beyond Average-Channel-Based Rate Approximations: UAV Trajectory and Scheduling Optimization With Expected Rate Consideration”, *arXiv preprint*, 2026, arXiv:2602.17019 | communication service to ground nodes; relaying or data collection not stated; mission time objective | background: average-channel rates overestimate expected rates |
| `wu2018multi`: Wu et al., “Joint Trajectory and Communication Design for Multi-UAV Enabled Wireless Networks”, *IEEE Transactions on Wireless Communications* 17(3):2109–2121, 2018, DOI 10.1109/twc.2017.2789293 | UAV base stations maximizing max–min throughput; no time constraint | background: BCD with SCA for UAV trajectories |
| `zeng2016mobile`: Zeng et al., “Throughput Maximization for UAV-Enabled Mobile Relaying Systems”, *IEEE Transactions on Communications* 64(12):4983–4996, 2016, DOI 10.1109/tcomm.2016.2611512 | throughput over a horizon with information causality; no per-demand timing | background: mobile relaying with buffering |
| `na2020emergency`: Na et al., “Joint trajectory and power optimization for UAV-relay-assisted Internet of Things in emergency”, *Physical Communication* 41:101100, 2020, DOI 10.1016/j.phycom.2020.101100 | sum rate over a horizon with information causality; no per-demand timing | background |
| `li2024flight`: Li et al., “Flight time minimization of UAV for cooperative data collection in probabilistic LoS channel”, *China Communications* 21(2):210–226, 2024, DOI 10.23919/jcc.fa.2021-0823.202402 | flight-time minimization; no per-demand timing | background: PrLoS data collection |
| `ropke2006alns`: Ropke and Pisinger, “An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows”, *Transportation Science* 40(4):455–472, 2006, DOI 10.1287/trsc.1050.0135 | not UAV communication | background: large neighbourhood search with repair |
| `jain2004dtn`: Jain et al., “Routing in a delay tolerant network”, *Proceedings of the 2004 conference on Applications, technologies, architectures, and protocols for computer communications*, pp. 145–158, 2004, DOI 10.1145/1015467.1015484 | not UAV communication | background: routing on time-varying graphs with known dynamics |
| Zeng et al., “Trajectory Optimization and Resource Allocation for OFDMA UAV Relay Networks”, IEEE TWC 20(10):6634–6647, 2021, DOI 10.1109/TWC.2021.3075594 | amplify-and-forward relaying for throughput and fairness; no time or backhaul constraint | — |
| Zhou et al., “Delay-Aware UAV Computation Offloading and Communication Assistance for Post-Disaster Rescue”, IEEE TWC 23(12):19110–19125, 2024, DOI 10.1109/TWC.2024.3479709 | delay of task computation at aerial base stations, not delivery of data | — |
| Zhao et al., “UAV-Assisted Emergency Networks in Disasters”, IEEE Wireless Communications 26(1):45–51, 2019, DOI 10.1109/MWC.2018.1800160 | magazine framework; no formulated time or backhaul constraint in the abstract | — |
| Sharifi et al., “UDADT: An Energy- and Delay-Aware Trajectory Planning for UAV-Assisted Wireless Sensor Networks”, Trans. Emerging Telecommunications Technologies 37(6), 2026, DOI 10.1002/ett.70427 | delay is the tour duration; no per-demand timing | — |
| Zhang et al., “Joint Optimization of IRS and UAV-Trajectory ...”, IEEE Vehicular Technology Magazine 17(2):55–63, 2022, DOI 10.1109/MVT.2022.3158047 | IRS-assisted access; no UAV relaying or data collection stated | — |
| Mondal et al., “Deep reinforcement learning based multi-UAV assisted data collection for deadline-sensitive IoT networks”, Physical Communication 73:102880, 2025, DOI 10.1016/j.phycom.2025.102880 | abstract not retrievable (publisher page 403; absent from Crossref, OpenAlex, Semantic Scholar) | — |
| Hu, Chen, Chen, “Joint trajectory-resource optimization for UAV-enabled uplink communication networks with wireless backhaul”, Computer Networks 229:109779, 2023, DOI 10.1016/j.comnet.2023.109779 | abstract not retrievable (same sources) | — |
| Banerjee et al., “EDTP: Energy and Delay Optimized Trajectory Planning for UAV-IoT Environment”, Computer Networks 202:108623, 2022, DOI 10.1016/j.comnet.2021.108623 | abstract not retrievable (same sources) | — |
| “Delay performance of priority-queue equipped UAV-based mobile relay networks: Exploring the impact of trajectories”, Computer Networks 210:108856, 2022, DOI 10.1016/j.comnet.2022.108856 | abstract not retrievable (same sources) | — |
| “A buffer-aware dynamic UAV trajectory design for data collection in resource-constrained IoT frameworks”, Computers and Electrical Engineering 100:107934, 2022, DOI 10.1016/j.compeleceng.2022.107934 | abstract not retrievable (same sources) | — |

Records excluded by title are listed in `screening.csv` with their reason.

## 6. Strong-Baseline Selection

The spec criteria are applied in order. Only three included papers meet the first criterion, an objective close to maximizing the number of alerts delivered on time: `tran2022relay`, `samir2020time`, and `du2023timeconstrained`.

| Criterion | `tran2022relay` | `samir2020time` | `du2023timeconstrained` |
|---|---|---|---|
| 1. Objective close to timely alerts | Served devices whose data reaches the gateway in time | Served devices by their deadlines | Served devices within the gathering period |
| 2. Offline design with known demand | Yes | Yes | Yes |
| 3. Joint trajectory and radio resources | Bandwidth, power, trajectory | Radio resources, trajectory | Time allocation, power, trajectory |
| 4. Detail or code to reproduce | Full text on arXiv (2008.00218) | Full algorithms; unofficial MATLAB/CVX implementation under the MIT license (github.com/willyfh/uav-trajectory-planning) | Six-page workshop paper |
| 5. Adaptable without changing the method | Two hops with UAV storage and delivery to a ground gateway, like the source–UAV–entry-node path of this project | Collection at the UAV; no second hop in the abstract | Collection at the UAV with energy harvesting |

**Decision:** Tran et al. (half-duplex variant) is the strong literature baseline, confirming the default of spec Section 13. Criterion 5 decides between Tran and Samir: only Tran models delivery through the UAV buffer to a ground gateway, which maps onto the UAV paths that enter the ground network. Samir et al. remain the fallback if the Tran adaptation proves infeasible, because code is available. The adaptation — ground entry nodes as gateways with the fixed backhaul routes to the center, alerts as devices, and served as timely — is specified and implemented in sub-project D.
```

- [ ] **Step 2: Write the screening record**

Create `docs/literature/screening.csv` with exactly this content:

```text
query,source,id,year,first_author,title,stage,decision,reason
Q1,crossref,10.3390/photonics11121136,2024,Kang,Joint Divergence Angle of Free Space Optics (FSO) Link and UAV Trajectory Design in FSO-Based UAV-Enabled Wireless Power Transfer Relay Systems,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.3390/electronics13050828,2024,Lee,"Joint Optimization on Trajectory, Data Relay, and Wireless Power Transfer in UAV-Based Environmental Monitoring System",title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.12720/jcm.21.3.365-371,2026,Trang,Joint Trajectory and Power Optimization for UAV-Relay: A Constraint Handling Approach with the Bat Algorithm,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.23919/cje.2025.00.196,2026,Feng,When UAV Base Station Enabled System Meets RIS-UAV Relay Assisted System: Joint Trajectory and Resource Allocation Optimization,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.1109/access.2018.2867849,2018,Jiang,Power and Trajectory Optimization for UAV-Enabled Amplify-and-Forward Relay Networks,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.1109/jsyst.2021.3134305,2022,Ning,Secure UAV Relay Communication via Power Allocation and Trajectory Planning,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.1109/access.2020.3013319,2020,Yuan,Channel-Aware Potential Field Trajectory Planning for Solar-Powered Relay UAV in Near-Space,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.1109/lcomm.2017.2763135,2018,Zhang,Joint Trajectory and Power Optimization for UAV Relay Networks,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.23919/jcc.2021.06.015,2021,Wang,Joint 3D trajectory and resource optimization for a UAV relay-assisted cognitive radio network,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.1016/j.vehcom.2021.100357,2021,Zhang,Joint trajectory and power optimization for mobile jammer-aided secure UAV relay network,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.1016/j.comcom.2024.108007,2025,Sabuj,Trajectory design of UAV-aided energy-harvesting relay networks in the terahertz band,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.1016/j.dsp.2024.104626,2024,Nguyen,Joint resource and trajectory optimization for secure UAV-based two-way relay system,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.1016/j.cja.2021.12.001,2022,XIE,Energy-efficiency maximization for fixed-wing UAV-enabled relay network with circular trajectory,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q1,crossref,10.1016/j.phycom.2020.101100,2020,Na,Joint trajectory and power optimization for UAV-relay-assisted Internet of Things in emergency,abstract,excluded,no per-demand timing
Q1,crossref,10.1109/twc.2021.3075594,2021,Zeng,Trajectory Optimization and Resource Allocation for OFDMA UAV Relay Networks,abstract,excluded,no time or backhaul constraint
Q2,crossref,10.1109/twc.2022.3216797,2023,Bashir,Energy Optimization of a Laser-Powered Hovering-UAV Relay in Optical Wireless Backhaul,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.3390/s24061973,2024,Chen,An Adjustable Wireless Backhaul Link Selection Algorithm for LEO-UAV-Sensor-Based Internet of Remote Things Network,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.1109/lnet.2021.3080403,2021,Yanmaz,Dynamic Relay Selection and Positioning for Cooperative UAV Networks,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.1016/j.phycom.2018.11.005,2019,Deng,Outdated relay selection for UAV-enabled networks with cooperative NOMA,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.1109/tgcn.2023.3347567,2024,Li,"UAV Altitude, Relay Selection, and User Association Optimization for Cooperative Relay-Transmission in UAV-IRS-Based THz Networks",title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.1109/taes.2025.3526742,2025,Liu,3-D Position Optimization of Solar-Powered Hovering UAV Relay in Optical Wireless Backhaul,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.3390/su142316089,2022,Chen,A Method of Relay Node Selection for UAV Cluster Networks Based on Distance and Energy Constraints,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.3390/drones10030160,2026,Yang,Joint Altitude and Power Optimization for Multi-UAV-Aided Covert Communication with Relay Selection,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.3390/app10238762,2020,He,A Relay Selection Protocol for UAV-Assisted VANETs,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.1016/j.phycom.2019.01.007,2019,Zou,Impact of outdated antenna selection on cache-aided UAV relay networks,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.1155/2021/3929514,2021,Wang,Performance Analysis of Group Selection in UAV Networks with Backhaul‐Aware Biasing,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.1007/s11276-021-02644-9,2021,Fan,Optimal relay selection for UAV-assisted V2V communications,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.52783/jisem.v10i2s.198,2025,J.Vijaya Barathy,Relay Selection Technique for Energy-Efficient Data Transmission Policy for UAV-Assisted LoRaWAN,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.3390/drones7090579,2023,Liang,Hierarchical Matching Algorithm for Relay Selection in MEC-Aided Ultra-Dense UAV Networks,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,crossref,10.1016/j.dsp.2026.106256,2026,Mohamed,Digital twin-enabled neural bandit learning for two-hop UAV relay selection over delay–doppler channels,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,arxiv,2203.04377v1,2022,Dabiri,Long mmWave Backhaul Connectivity Using Fixed-Wing UAVs,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,arxiv,2401.09060v1,2024,Gures,Joint Route Selection and Power Allocation in Multi-hop Cache-enabled Networks,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q2,arxiv,2102.03040v1,2021,Popescu,Connecting flying backhauls of UAVs to enhance vehicular networks with fixed 5G NR infrastructure,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.3788/col202220.081101,2022,Liang,FAANet: feature-aligned attention network for real-time multiple object tracking in UAV videos,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.3390/drones8110671,2024,Huang,Mamba-UAV-SegNet: A Multi-Scale Adaptive Feature Fusion Network for Real-Time Semantic Segmentation of UAV Aerial Imagery,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.1016/j.dam.2024.06.033,2024,Figueroa González,A project and lift approach for a 2-commodity flow relocation model in a time expanded network,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.1109/mnet.008.2100472,2022,Bansal,Scalable Topologies for Time-Optimal Authentication of UAV Swarms,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.33969/j-nana.2021.010101,2021,,Generalized Traffic Flow Model for Multi-Services Oriented UAV System,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.1016/j.ejor.2022.08.044,2023,Xing,Joint tank container demurrage policy and flow optimisation using a progressive hedging algorithm with expanded time-space network,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.1007/s11554-026-01952-7,2026,Cui,EHR-Net: a real-time Edge-Enhanced Hierarchical Residual Network for UAV infrared small target detection,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.1049/ell2.70206,2025,Zhao,UAV‐Based Real‐Time Object Detection Network Using Structured Pruning Strategy,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.3390/drones9120812,2025,Mohamed,Real-Time Long-Range Control of an Autonomous UAV Using 4G LTE Network,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.1007/s11554-026-01977-y,2026,Dai,LSO-YOLO: a lightweight real-time object detection network for UAV small-object detection,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.1016/j.jnca.2025.104372,2026,Kumar,LiteWTAKA: Authenticating UAV-GCS and UAV–UAV communication using secure and lightweight mechanism based on PUF,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.1038/s41598-026-36440-2,2026,Yang,A collaborative multi-attention network for real-time small object detection in UAV imagery,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.13164/re.2025.0342,2025,Zafor,An Effective Routing Algorithm to Minimize the UAV Routing Time and Extend the Network Lifetime in Clustered IoT Network,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.21660/2024.124.4607,2024,Kanplumjit,ACCURACY ASSESSMENT OF THAILAND’S NETWORK REAL TIME KINEMATIC (NRTK) FOR UNMANNED AERIAL VEHICLE (UAV) PHOTOGRAMMETRY,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q3,crossref,10.1007/s11554-025-01803-x,2025,Dang,EMDANet: real-time road crack detection in UAV remote sensing images via multi-scale dual-attention network,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q4,crossref,10.21123/bsj.2018.15.4.0485,2018,,Optimal UAV Deployment for Data Collection in Deadline-based IoT Applications,title,excluded,duplicate record of albusalih2018deadline
Q4,crossref,10.21123/bsj.15.4.484-491,2018,Journal,Optimal UAV Deployment for Data Collection in Deadline-based IoT Applications,title,excluded,duplicate record of albusalih2018deadline
Q4,crossref,10.21123/bsj.2018.15.4.0484,2018,Albu-Salih,Optimal UAV Deployment for Data Collection in Deadline-based IoT Applications,abstract,included,
Q4,crossref,10.1587/transfun.e102.a.598,2019,LU,Stochastic Channel Selection for UAV-Aided Data Collection,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q4,crossref,10.1016/j.phycom.2025.102880,2025,Mondal,Deep reinforcement learning based multi-UAV assisted data collection for deadline-sensitive IoT networks,abstract,excluded,abstract not retrievable
Q4,crossref,10.23919/jcc.fa.2021-0823.202402,2024,Li,Flight time minimization of UAV for cooperative data collection in probabilistic LoS channel,abstract,excluded,flight-time minimization; no per-demand timing
Q4,crossref,10.14419/ijet.v7i3.14522,2018,Felix Arokya Jose,DBac: deadline based data collection using CSMA/CD and earliest deadline first (EDF) scheduling in wireless sensor network,title,excluded,not UAV
Q4,crossref,10.3390/info17020190,2026,Ma,DC-CSAP: An Edge-UAV-End Collaborative Data Collection Framework for UAV-Assisted IoT,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q4,crossref,10.3390/s22155839,2022,Li,Completion Time Minimization for UAV-UGV-Enabled Data Collection,title,excluded,mission completion time only
Q4,crossref,10.1109/tvt.2023.3320676,2024,Liu,UAV Trajectory Planning With Interference Awareness in UAV-Enabled Time-Constrained Data Collection Systems,abstract,included,
Q4,crossref,10.3997/1365-2397.fb2023092,2023,Dobrovolskiy,Water Bodies Data Collection Using UAV: UXO Search and Bathymetry,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q4,crossref,10.1109/jiot.2022.3189214,2022,Liu,UAV Trajectory Optimization for Time-Constrained Data Collection in UAV-Enabled Environmental Monitoring Systems,abstract,included,
Q4,crossref,10.1002/dac.70553,2026,Al‐Qurabat,IF‐UAV: An Intelligent Federated Framework for Energy‐Efficient Data Collection in UAV‐Assisted IoT Systems,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q4,crossref,10.1002/dac.5486,2023,Wala,DDC‐OMDC: Deadline‐based data collection using optimal mobile data collectors in Internet of Things,title,excluded,not UAV
Q4,crossref,10.1016/j.icte.2022.10.003,2023,Memos,Optimized UAV-based data collection from MWSNs,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q5,crossref,10.1007/s11277-024-11386-8,2024,Shi,"Joint User Scheduling, Trajectory, Beamforming and Power Allocation for Millimeter-Wave Full-Duplex UAV Relay",title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q5,crossref,10.1109/access.2024.3481438,2024,Wang,Joint Terminal-UAV Association and Power Allocation for Hybrid Satellite-UAV Relay Network With Rate Splitting Multiple Access,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q5,crossref,10.3390/drones9030160,2025,Li,Deep Reinforcement Learning-Enabled Trajectory and Bandwidth Allocation Optimization for UAV-Assisted Integrated Sensing and Covert Communication,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q5,crossref,10.1109/twc.2020.3000864,2020,Hu,Low-Complexity Joint Resource Allocation and Trajectory Design for UAV-Aided Relay Networks With the Segmented Ray-Tracing Channel Model,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q5,crossref,10.1016/j.phycom.2021.101328,2021,Deng,Joint UAV trajectory and power allocation optimization for NOMA in cognitive radio network,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q5,crossref,10.1016/j.phycom.2026.103355,2026,Gao,Joint optimization of 3D trajectory design and resource allocation for ARIS-assisted UAV,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q5,crossref,10.1109/tvt.2020.2985061,2020,Chen,Optimal Power and Bandwidth Allocation for Multiuser Video Streaming in UAV Relay Networks,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q5,crossref,10.31449/inf.v49i21.9892,2025,Liu,"Joint UAV Trajectory, IRS Beamforming, and Power Allocation Optimization for Secure Communication in IRS-Assisted Cognitive Networks",title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q5,crossref,10.23919/jcc.fa.2023-0107.202402,2024,Xie,Joint optimization of resource allocation and trajectory based on user trajectory for UAV-assisted backscatter communication system,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q5,arxiv,2503.14248v1,2025,Queiros,Joint Channel Bandwidth Assignment and Relay Positioning for Predictive Flying Networks,abstract,included,preprint of queiros2024predictive
Q5,arxiv,2601.01430v2,2026,Joshi,Context-Aware Information Transfer via Digital Semantic Communication in UAV-Based Networks,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q6,crossref,10.3390/drones8120777,2024,Fei,Heuristic Optimization-Based Trajectory Planning for UAV Swarms in Urban Target Strike Operations,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q6,crossref,10.1007/s10586-026-06112-x,2026,Mohamed,A delay-Doppler approach for joint mmWave beamforming and proactive UAV trajectory planning in UAV-vehicular communications,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q6,crossref,10.1016/j.comnet.2021.108623,2022,Banerjee,EDTP: Energy and Delay Optimized Trajectory Planning for UAV-IoT Environment,abstract,excluded,abstract not retrievable
Q6,crossref,10.1109/tvt.2022.3195788,2022,Tang,Delay-Tolerant UAV-Assisted Communication: Online Trajectory Design and User Association,abstract,included,
Q6,crossref,10.1109/mnet.2016.7389838,2016,Fadlullah,A dynamic trajectory control algorithm for improving the communication throughput and delay in UAV-aided networks,abstract,included,
Q6,crossref,10.31130/ud-jst.2025.23(10b).638e,2025,Binh,Energy minimization in wireless sensor networks: an efficient heuristic framework for data collection through UAV trajectory planning and power allocation,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q6,crossref,10.1002/ett.70427,2026,Sharifi,<scp>UDADT</scp> : An Energy‐ and Delay‐Aware Trajectory Planning for <scp>UAV</scp> ‐Assisted Wireless Sensor Networks,abstract,excluded,delay is the tour duration
Q6,crossref,10.1109/tvt.2022.3144178,2022,He,Trajectory Optimization and Channel Allocation for Delay Sensitive Secure Transmission in UAV-Relayed VANETs,abstract,included,
Q6,crossref,10.1002/asjc.3557,2025,Li,Predefined time trajectory tracking control and hardware in loop verification for quadrotor UAV under input delay and disturbances,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q6,crossref,10.1109/tgcn.2022.3196670,2023,Cao,Energy-Delay Tradeoff for Dynamic Trajectory Planning in Priority-Oriented UAV-Aided IoT Networks,abstract,included,
Q6,crossref,10.1109/tnse.2025.3638854,2026,Bao,Joint Trajectory and Resource Optimization for Delay Minimization of UAV-Enabled NOMA-MEC System With LWPT,title,excluded,"computation offloading (MEC), not data delivery"
Q6,crossref,10.1007/s10846-021-01467-2,2021,Nizar,Effective and Safe Trajectory Planning for an Autonomous UAV Using a Decomposition-Coordination Method,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q6,crossref,10.3390/drones10060434,2026,Cheng,Uncertainty-Calibrated UAV Trajectory Prediction for Beam Management in UAV-Assisted ISAC Scenarios,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q6,crossref,10.3390/drones9060446,2025,Chen,UAV Spiral Maneuvering Trajectory Intelligent Generation Method Based on Virtual Trajectory,title,excluded,"title shows no per-demand timing, relay or backhaul selection, or UAV data delivery"
Q6,crossref,10.1109/mvt.2022.3158047,2022,Zhang,Joint Optimization of IRS and UAV-Trajectory: For Supporting Statistical Delay and Error-Rate Bounded QoS Over mURLLC-Driven 6G Mobile Wireless Networks Using FBC,abstract,excluded,IRS-assisted access; no UAV relaying or data collection
web,web search,10.1109/TWC.2021.3105821,,tran2022relay,,abstract,included,
web,web search,10.1109/TWC.2019.2940447,,samir2020time,,abstract,included,
web,web search,10.1109/ICCW.2017.7962642,,kalantari2017backhaul,,abstract,included,
web,web search,10.1109/GLOCOMW.2018.8644135,,selim2018postdisaster,,abstract,included,
web,web search,10.1109/TNSE.2022.3233004,,yu2023backhaul,,abstract,included,
web,web search,2608.04590,,wang2026juror,,abstract,included,
web,web search,10.1109/INFOCOMWKSHPS57453.2023.10226065,,du2023timeconstrained,,abstract,included,
web,web search,10.1109/TCOMM.2018.2865922,,wu2018delay,,abstract,excluded,UAV base station downlink
web,web search,2602.17019,,park2026expected,,abstract,excluded,relaying or data collection not stated
web,web search,10.1109/TWC.2017.2789293,,wu2018multi,,abstract,excluded,no time constraint
web,web search,10.1109/TCOMM.2016.2611512,,zeng2016mobile,,abstract,excluded,no per-demand timing
web,web search,10.1109/TWC.2024.3479709,,zhou2024delayaware,,abstract,excluded,computation offloading
web,web search,10.1109/MWC.2018.1800160,,zhao2019emergency,,abstract,excluded,no formulated time or backhaul constraint
web,web search,10.1016/j.comnet.2023.109779,,hu2023uplink,,abstract,excluded,abstract not retrievable
web,web search,10.1016/j.comnet.2022.108856,,priorityqueue2022,,abstract,excluded,abstract not retrievable
web,web search,10.1016/j.compeleceng.2022.107934,,bufferaware2022,,abstract,excluded,abstract not retrievable
web,web search,10.1287/trsc.1050.0135,,ropke2006alns,,abstract,excluded,not UAV communication
web,web search,10.1145/1015467.1015484,,jain2004dtn,,abstract,excluded,not UAV communication
```

- [ ] **Step 3: Write the comparison table**

Create `docs/report/data/related_work.csv` with exactly this content:

```text
key;year;venue;objective;uavs;relay_selection;deadline;connectivity;channel_uncertainty;bandwidth_power;method;guarantee
huang2025ddatsap;2025;Sensors;Max–min tốc độ tin trung bình;1;Không;Không;Không nêu;PrLoS;Công suất, lập lịch khe;BCD, SCA;Không nêu
tran2022relay;2022;IEEE TWC;Tối đa số thiết bị phục vụ đúng hạn;1;Không nêu;Có, theo thiết bị;Không nêu;Không nêu;Băng thông, công suất, bộ nhớ UAV;Nới lỏng biến nhị phân, xấp xỉ trong;Không nêu
samir2020time;2020;IEEE TWC;Tối đa số thiết bị phục vụ trước hạn;1;Không nêu;Có, theo thiết bị;Không nêu;Không nêu;Tài nguyên vô tuyến;BRB, SCA;Tối ưu toàn cục với BRB
albusalih2018deadline;2018;Baghdad Sci. J.;Tối thiểu năng lượng thu thập;Nhiều;Không nêu;Có, hạn thu thập;Không nêu;Không nêu;Không nêu;MILP, heuristic;Tối ưu với MILP
liu2022aoi;2022;IEEE IoT J.;Tối thiểu thời gian nhiệm vụ;1;Không nêu;Có, ngưỡng AoI;Không nêu;Không nêu;Không nêu;Phân rã, SCA, GA;Không nêu
liu2024interference;2024;IEEE TVT;Tối thiểu thời gian nhiệm vụ;1;Có, liên kết UAV--trạm;Có, theo dữ liệu;Không nêu;Không nêu;Không nêu;BCD, SCA;Không nêu
tang2022delay;2022;IEEE TVT;Tối thiểu năng lượng UAV;Nhiều;Có, liên kết người dùng;Có, trễ hàng đợi hữu hạn;Không nêu;Nhu cầu, di chuyển ngẫu nhiên;Không nêu;Lyapunov, phân rã trực tuyến;Không nêu
he2022delay;2022;IEEE TVT;Tối thiểu tổng trễ thông tin;1;Không nêu;Mục tiêu trễ;Không nêu;Không nêu;Cấp phát kênh;Luân phiên Newton, relax-and-round, SCA;Không nêu
cao2023priority;2023;IEEE TGCN;Năng lượng UAV và trễ theo ưu tiên;1;Không nêu;Mục tiêu trễ theo ưu tiên;Không nêu;Nút cảm biến di động;Không nêu;Heuristic học tăng cường, trực tuyến;Không nêu
fadlullah2016dynamic;2016;IEEE Network;Giảm trễ, tăng thông lượng relay;Nhiều;Không nêu;Trễ do nghẽn hàng đợi;Chuỗi relay giữa các UAV;Không nêu;Không nêu;Điều khiển quỹ đạo theo ngưỡng hàng đợi;Không nêu
du2023timeconstrained;2023;IEEE INFOCOM Wkshps;Tối đa số thiết bị phục vụ trong thời gian thu thập;1;Không nêu;Có, thời gian thu thập;Không nêu;Không nêu;Thời gian, công suất phát;Luân phiên, SCA, xấp xỉ bậc hai;Không nêu
wang2026juror;2026;arXiv;Tăng phân phát tin trong DTN;Không nêu;Có, định tuyến cơ hội;Có, TTL của tin;Tiếp xúc gián đoạn;Không nêu;Không nêu;Học tăng cường PPO;Không nêu
kalantari2017backhaul;2017;IEEE ICC Wkshps;Tối đa số người dùng hoặc tổng tốc độ;1;Có, backhaul;Không nêu;Không nêu;Không nêu;Không nêu;Tìm vị trí 3D;Tối ưu vị trí đặt
selim2018postdisaster;2018;IEEE Globecom Wkshps;Tối thiểu năng lượng drone;Nhiều;Có, drone backhaul có dây;Không nêu;Không nêu;Không nêu;Không nêu;Tối ưu vị trí và năng lượng;Không nêu
yu2023backhaul;2023;IEEE TNSE;Cấp băng thông và vị trí trạm bay;1;Có, backhaul FSO;Không nêu;Không nêu;Không nêu;Băng thông;Thuật toán BROAD;Không nêu
queiros2024predictive;2024;IEEE Globecom Wkshps;Hiệu năng mạng với tốc độ tối thiểu;Nhiều tầng;Có, relay backhaul bay;Không nêu;Không nêu;Không nêu;Băng thông;Simulated annealing có hàm phạt;Không nêu
```

- [ ] **Step 4: Append the verified bibliography entries**

Append to `docs/report/references.bib`:

```bibtex
% Pha 2 (tổng quan tài liệu, docs/literature/review.md): metadata lấy từ Crossref theo DOI hoặc arXiv theo mã bài, ngày 11/09/2026.

@article{zeng2016mobile,
  author  = {Zeng, Yong and Zhang, Rui and Lim, Teng Joon},
  title   = {Throughput Maximization for {UAV}-Enabled Mobile Relaying Systems},
  journal = {IEEE Transactions on Communications},
  year    = {2016},
  volume  = {64},
  number  = {12},
  pages   = {4983--4996},
  doi     = {10.1109/tcomm.2016.2611512},
}

@article{na2020emergency,
  author  = {Na, Zhenyu and Mao, Beihang and Shi, Jingcheng and Wang, Jun and Gao, Zihe and Xiong, Mudi},
  title   = {Joint trajectory and power optimization for {UAV}-relay-assisted Internet of Things in emergency},
  journal = {Physical Communication},
  year    = {2020},
  volume  = {41},
  pages   = {101100},
  doi     = {10.1016/j.phycom.2020.101100},
}

@article{tran2022relay,
  author  = {Tran, Dinh-Hieu and Nguyen, Van-Dinh and Chatzinotas, Symeon and Vu, Thang X. and Ottersten, Bj{\"o}rn},
  title   = {{UAV} Relay-Assisted Emergency Communications in {IoT} Networks: Resource Allocation and Trajectory Optimization},
  journal = {IEEE Transactions on Wireless Communications},
  year    = {2022},
  volume  = {21},
  number  = {3},
  pages   = {1621--1637},
  doi     = {10.1109/twc.2021.3105821},
}

@article{samir2020time,
  author  = {Samir, Moataz and Sharafeddine, Sanaa and Assi, Chadi M. and Nguyen, Tri Minh and Ghrayeb, Ali},
  title   = {{UAV} Trajectory Planning for Data Collection from Time-Constrained {IoT} Devices},
  journal = {IEEE Transactions on Wireless Communications},
  year    = {2020},
  volume  = {19},
  number  = {1},
  pages   = {34--46},
  doi     = {10.1109/twc.2019.2940447},
}

@article{albusalih2018deadline,
  author  = {Albu-Salih, Alaa Taima and Seno, Seyed Amin Hosseini},
  title   = {Optimal {UAV} Deployment for Data Collection in Deadline-based {IoT} Applications},
  journal = {Baghdad Science Journal},
  year    = {2018},
  volume  = {15},
  number  = {4},
  doi     = {10.21123/bsj.2018.15.4.0484},
}

@article{liu2022aoi,
  author  = {Liu, Kai and Zheng, Jun},
  title   = {{UAV} Trajectory Optimization for Time-Constrained Data Collection in {UAV}-Enabled Environmental Monitoring Systems},
  journal = {IEEE Internet of Things Journal},
  year    = {2022},
  volume  = {9},
  number  = {23},
  pages   = {24300--24314},
  doi     = {10.1109/jiot.2022.3189214},
}

@article{liu2024interference,
  author  = {Liu, Kai and Zheng, Jun},
  title   = {{UAV} Trajectory Planning With Interference Awareness in {UAV}-Enabled Time-Constrained Data Collection Systems},
  journal = {IEEE Transactions on Vehicular Technology},
  year    = {2024},
  volume  = {73},
  number  = {2},
  pages   = {2799--2815},
  doi     = {10.1109/tvt.2023.3320676},
}

@article{li2024flight,
  author  = {Li, Yan and Xu, Shaoyi and Wu, Yunpu and Li, Dongji},
  title   = {Flight time minimization of {UAV} for cooperative data collection in probabilistic {LoS} channel},
  journal = {China Communications},
  year    = {2024},
  volume  = {21},
  number  = {2},
  pages   = {210--226},
  doi     = {10.23919/jcc.fa.2021-0823.202402},
}

@article{tang2022delay,
  author  = {Tang, Yao and Cheung, Man Hon and Lok, Tat-Ming},
  title   = {Delay-Tolerant {UAV}-Assisted Communication: Online Trajectory Design and User Association},
  journal = {IEEE Transactions on Vehicular Technology},
  year    = {2022},
  volume  = {71},
  number  = {12},
  pages   = {13137--13151},
  doi     = {10.1109/tvt.2022.3195788},
}

@article{he2022delay,
  author  = {He, Yixin and Wang, Dawei and Huang, Fanghui and Zhang, Ruonan and Pan, Jianping},
  title   = {Trajectory Optimization and Channel Allocation for Delay Sensitive Secure Transmission in {UAV}-Relayed {VANETs}},
  journal = {IEEE Transactions on Vehicular Technology},
  year    = {2022},
  volume  = {71},
  number  = {4},
  pages   = {4512--4517},
  doi     = {10.1109/tvt.2022.3144178},
}

@inproceedings{kalantari2017backhaul,
  author    = {Kalantari, Elham and Shakir, Muhammad Zeeshan and Yanikomeroglu, Halim and Yongacoglu, Abbas},
  title     = {Backhaul-aware robust {3D} drone placement in {5G}+ wireless networks},
  booktitle = {2017 IEEE International Conference on Communications Workshops (ICC Workshops)},
  year      = {2017},
  pages     = {109--114},
  doi       = {10.1109/iccw.2017.7962642},
}

@inproceedings{selim2018postdisaster,
  author    = {Selim, Mohamed Y. and Kamal, Ahmed E.},
  title     = {Post-Disaster {4G/5G} Network Rehabilitation Using Drones: Solving Battery and Backhaul Issues},
  booktitle = {2018 IEEE Globecom Workshops (GC Wkshps)},
  year      = {2018},
  pages     = {1--6},
  doi       = {10.1109/glocomw.2018.8644135},
}

@article{yu2023backhaul,
  author  = {Yu, Liangkun and Sun, Xiang and Shao, Sihua and Chen, Yougan and Albelaihi, Rana},
  title   = {Backhaul-Aware Drone Base Station Placement and Resource Management for {FSO}-Based Drone-Assisted Mobile Networks},
  journal = {IEEE Transactions on Network Science and Engineering},
  year    = {2023},
  volume  = {10},
  number  = {3},
  pages   = {1659--1668},
  doi     = {10.1109/tnse.2022.3233004},
}

@inproceedings{queiros2024predictive,
  author    = {Queiros, Ruben and Kaneko, Megumi and Fontes, Helder and Campos, Rui},
  title     = {Joint Channel Bandwidth Assignment and Relay Positioning for Predictive Flying Networks},
  booktitle = {2024 IEEE Globecom Workshops (GC Wkshps)},
  year      = {2024},
  pages     = {1--6},
  doi       = {10.1109/gcwkshp64532.2024.11100616},
}

@article{wu2018delay,
  author  = {Wu, Qingqing and Zhang, Rui},
  title   = {Common Throughput Maximization in {UAV}-Enabled {OFDMA} Systems With Delay Consideration},
  journal = {IEEE Transactions on Communications},
  year    = {2018},
  volume  = {66},
  number  = {12},
  pages   = {6614--6627},
  doi     = {10.1109/tcomm.2018.2865922},
}

@article{wu2018multi,
  author  = {Wu, Qingqing and Zeng, Yong and Zhang, Rui},
  title   = {Joint Trajectory and Communication Design for Multi-{UAV} Enabled Wireless Networks},
  journal = {IEEE Transactions on Wireless Communications},
  year    = {2018},
  volume  = {17},
  number  = {3},
  pages   = {2109--2121},
  doi     = {10.1109/twc.2017.2789293},
}

@article{ropke2006alns,
  author  = {Ropke, Stefan and Pisinger, David},
  title   = {An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows},
  journal = {Transportation Science},
  year    = {2006},
  volume  = {40},
  number  = {4},
  pages   = {455--472},
  doi     = {10.1287/trsc.1050.0135},
}

@inproceedings{jain2004dtn,
  author    = {Jain, Sushant and Fall, Kevin and Patra, Rabin},
  title     = {Routing in a delay tolerant network},
  booktitle = {Proceedings of the 2004 conference on Applications, technologies, architectures, and protocols for computer communications},
  year      = {2004},
  pages     = {145--158},
  doi       = {10.1145/1015467.1015484},
}

@misc{wang2026juror,
  author        = {Wang, Xiao and Yang, Shun-Ren},
  title         = {Joint {UAV} Flight and Opportunistic Routing under Reinforcement Learning for Delay-Tolerant Networks},
  year          = {2026},
  eprint        = {2608.04590},
  archiveprefix = {arXiv},
  primaryclass  = {cs.AI},
  doi           = {10.48550/arXiv.2608.04590},
}

@misc{park2026expected,
  author        = {Park, Gitae and Lee, Kisong},
  title         = {Beyond Average-Channel-Based Rate Approximations: {UAV} Trajectory and Scheduling Optimization With Expected Rate Consideration},
  year          = {2026},
  eprint        = {2602.17019},
  archiveprefix = {arXiv},
  primaryclass  = {eess.SY},
  doi           = {10.48550/arXiv.2602.17019},
}

@article{cao2023priority,
  author  = {Cao, Hailin and Zhu, Wang and Chen, Zhengchuan and Sun, Zhiwei and Wu, Dapeng Oliver},
  title   = {Energy-Delay Tradeoff for Dynamic Trajectory Planning in Priority-Oriented {UAV}-Aided {IoT} Networks},
  journal = {IEEE Transactions on Green Communications and Networking},
  year    = {2023},
  volume  = {7},
  number  = {1},
  pages   = {158--170},
  doi     = {10.1109/tgcn.2022.3196670},
}

@article{fadlullah2016dynamic,
  author  = {Fadlullah, Zubair Md. and Takaishi, Daisuke and Nishiyama, Hiroki and Kato, Nei and Miura, Ryu},
  title   = {A dynamic trajectory control algorithm for improving the communication throughput and delay in {UAV}-aided networks},
  journal = {IEEE Network},
  year    = {2016},
  volume  = {30},
  number  = {1},
  pages   = {100--105},
  doi     = {10.1109/mnet.2016.7389838},
}

@inproceedings{du2023timeconstrained,
  author    = {Du, Pengfei and Xie, Fan and Chen, Shijia and Zhang, Xuejun},
  title     = {Time-Constrained {UAV}-Aided Data Collection for {IoT} Networks with Energy Harvesting},
  booktitle = {IEEE INFOCOM 2023 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS)},
  year      = {2023},
  pages     = {1--6},
  doi       = {10.1109/infocomwkshps57453.2023.10226065},
}
```

- [ ] **Step 5: Check the bibliography**

```bash
uv run python - <<'PY'
import re
from pathlib import Path

text = Path("docs/report/references.bib").read_text(encoding="utf-8")
keys = re.findall(r"^@\w+\{([^,]+),", text, re.M)
assert len(keys) == len(set(keys)), "duplicate keys"
for match in re.finditer(r"^@\w+\{([^,]+),\n(.*?)^\}", text, re.M | re.S):
    body = match.group(2)
    assert body.count("{") == body.count("}"), match.group(1)
    assert re.search(r"^\s+title\s+=", body, re.M) and re.search(r"^\s+year\s+=", body, re.M), match.group(1)
print(len(keys), "bib entries, unique keys, balanced braces")
PY
uv run python - <<'PY'
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HEAD = {"User-Agent": "bibliography-check/1.0"}
text = Path("docs/report/references.bib").read_text(encoding="utf-8")
phase2 = text[text.index("% Pha 2 (tổng quan tài liệu"):]
normal = lambda value: re.sub(r"[^a-z0-9]", "", re.sub(r"\\.|[{}]", "", value).lower())
mismatches, checked = [], 0
for match in re.finditer(r"@\w+\{([^,]+),\n(.*?)\n\}", phase2, re.S):
    key, body = match.group(1), match.group(2)
    fields = dict(re.findall(r"^\s+(\w+)\s+= \{(.*)\},?$", body, re.M))
    if "eprint" in fields:
        feed = urllib.request.urlopen(urllib.request.Request("http://export.arxiv.org/api/query?id_list=" + fields["eprint"], headers=HEAD), timeout=60).read()
        entry = ET.fromstring(feed).find("{http://www.w3.org/2005/Atom}entry")
        source_title = " ".join(entry.find("{http://www.w3.org/2005/Atom}title").text.split())
        source_year = entry.find("{http://www.w3.org/2005/Atom}published").text[:4]
    else:
        record = json.loads(urllib.request.urlopen(urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(fields["doi"]), headers=HEAD), timeout=60).read())["message"]
        source_title, source_year = record["title"][0], str(record["issued"]["date-parts"][0][0])
    checked += 1
    if normal(fields["title"]) != normal(source_title) or fields["year"] != source_year:
        mismatches.append((key, fields["title"], source_title, fields["year"], source_year))
    time.sleep(1)
print(f"{checked} Phase 2 entries checked against Crossref or arXiv; {len(mismatches)} mismatches")
for item in mismatches:
    print("  ", item)
PY
```

Expected: `30 bib entries, unique keys, balanced braces`, then `23 Phase 2 entries checked against Crossref or arXiv; 0 mismatches` (the second check needs network access and takes about half a minute).

- [ ] **Step 6: Commit**

```bash
git add docs/literature docs/report/data/related_work.csv docs/report/references.bib
git commit -m "docs: review literature and select the strong baseline"
```

### Task 7: Main comparison and statistics

**Files:**
- Create (generated): `results/phase2/main/method_realizations.csv`, `bounds.csv`, `summary.csv`, `manifest.json`; `results/phase2/statistics.csv`

**Interfaces:**
- Consumes: Tasks 1, 4, 5; `experiments/run_phase2.py` from plan C.1.
- Produces: main results for plan C.3 and the eight paired comparisons (spec acceptance criteria 17.2.2 and 17.2.4).

- [ ] **Step 1: Run the main comparison**

Run: `uv run python experiments/run_phase2.py --config configs/experiments/phase2_main.yaml`

Expected: `{"task_count": 4320, "executed_task_count": 4320, "skipped_task_count": 0, "status": "complete"}` and exit code 0 after about two hours with 12 workers. Each of the 60 scenarios has 12 method tasks and 60 B0/B1 bound tasks; B2, B3, and P solve their 30 trajectory bounds inside their method tasks. In the smoke run a search task took 95–170 s under full load.

- [ ] **Step 2: Check the results**

```bash
uv run python - results/phase2/main <<'PY'
import csv
import json
import re
import sys

root = sys.argv[1]


def rows(name):
    with open(f"{root}/{name}", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


manifest = json.load(open(f"{root}/manifest.json", encoding="utf-8"))
methods, bounds, summary = rows("method_realizations.csv"), rows("bounds.csv"), rows("summary.csv")
print(manifest["status"], manifest["task_count"], "failures", len(manifest["failures"]))
print(len(methods), "method rows;", len(bounds), "bound rows; statuses", sorted({row["status"] for row in bounds}))
print("bounded trajectories", sorted({row["trajectory_method"] or "ground" for row in bounds}))


def call_limit(method):
    if method in ("B0", "B1"):
        return 1
    budget = re.search(r"_b(\d+)$", method)
    return (int(budget.group(1)) if budget else 2000) + 1


print(len(summary), "summary rows; evaluate calls within budget:", all(int(row["evaluate_calls"]) <= call_limit(row["method"]) for row in methods))
PY
```

Expected:

```text
complete 4320 failures 0
21600 method rows; 9000 bound rows; statuses ['optimal']
bounded trajectories ['B1', 'B2', 'B3', 'P', 'ground']
24 summary rows; evaluate calls within budget: True
```

The bound rows are 1800 ground-only (B0), 1800 on the B1 trajectory, and 1800 for each of B2, B3, and P. A search method may use its budget plus the runner's final evaluation (2001 calls).

- [ ] **Step 3: Confirm that a rerun skips every task**

Run: `uv run python experiments/run_phase2.py --config configs/experiments/phase2_main.yaml`

Expected: `{"task_count": 4320, "executed_task_count": 0, "skipped_task_count": 4320, "status": "complete"}`. The rerun rewrites `manifest.json` with its own timestamps; the result CSVs are unchanged.

- [ ] **Step 4: Compute the paired statistics**

Run: `uv run python experiments/phase2_statistics.py`

Expected: `{"comparisons": 8, "output": "results/phase2/statistics.csv"}`.

```bash
uv run python - <<'PY'
import csv

with open("results/phase2/statistics.csv", newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        print(row["set_id"], row["comparison"], row["topology_count"], f"{float(row['mean_difference']):+.3f}", f"p_holm={float(row['p_holm']):.4f}")
PY
```

Expected: eight lines in the order `v0 P - B2`, `v0 B2 - B1`, `v0 B2 - B3`, `v0 P - B1`, then the same four for `v0-bh50`, each with topology count 10. The values are reported whatever their significance.

- [ ] **Step 5: Confirm that only result files are new, then commit**

Run: `git status --short --untracked-files=all results/phase2`

Expected: exactly the five files listed under **Files**; no shard appears.

```bash
git add results/phase2/main/method_realizations.csv results/phase2/main/bounds.csv results/phase2/main/summary.csv results/phase2/main/manifest.json results/phase2/statistics.csv
git commit -m "feat: record phase 2 main results and statistics"
```

### Task 8: Budget, sensitivity, and design-realization experiments

**Files:**
- Create (generated): `results/phase2/budget/`, `results/phase2/sensitivity/`, `results/phase2/design/`, each with `method_realizations.csv`, `bounds.csv`, `summary.csv`, `manifest.json`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1–4; `experiments/run_phase2.py`.
- Produces: results for the quality–runtime figure, the sensitivity grid, and the design-realization comparison of plan C.3 (spec acceptance criterion 17.2.3); Phase 2 commands in `README.md`.

- [ ] **Step 1: Run and check the budget experiment**

Run: `uv run python experiments/run_phase2.py --config configs/experiments/phase2_budget.yaml`

Expected: `{"task_count": 300, "executed_task_count": 300, "skipped_task_count": 0, "status": "complete"}` and exit code 0 after about 30 minutes with 12 workers.

```bash
uv run python - results/phase2/budget <<'PY'
import csv
import json
import re
import sys

root = sys.argv[1]


def rows(name):
    with open(f"{root}/{name}", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


manifest = json.load(open(f"{root}/manifest.json", encoding="utf-8"))
methods, bounds, summary = rows("method_realizations.csv"), rows("bounds.csv"), rows("summary.csv")
print(manifest["status"], manifest["task_count"], "failures", len(manifest["failures"]))
print(len(methods), "method rows;", len(bounds), "bound rows; statuses", sorted({row["status"] for row in bounds}))
print("bounded trajectories", sorted({row["trajectory_method"] or "ground" for row in bounds}))


def call_limit(method):
    if method in ("B0", "B1"):
        return 1
    budget = re.search(r"_b(\d+)$", method)
    return (int(budget.group(1)) if budget else 2000) + 1


print(len(summary), "summary rows; evaluate calls within budget:", all(int(row["evaluate_calls"]) <= call_limit(row["method"]) for row in methods))
PY
```

Expected:

```text
complete 300 failures 0
9000 method rows; 0 bound rows; statuses []
bounded trajectories []
10 summary rows; evaluate calls within budget: True
```

The limit of a `<method>_b<budget>` id is its budget plus one. Large budgets are often not used up because a search stops after an iteration that accepts nothing (with budget 4000 the smoke run on `Agis-r0` stopped at 2372 and 2136 calls).

- [ ] **Step 2: Run and check the sensitivity experiment**

Run: `uv run python experiments/run_phase2.py --config configs/experiments/phase2_sensitivity.yaml`

Expected: `{"task_count": 4950, "executed_task_count": 4950, "skipped_task_count": 0, "status": "complete"}` and exit code 0 after about one hour with 12 workers.

```bash
uv run python - results/phase2/sensitivity <<'PY'
import csv
import json
import re
import sys

root = sys.argv[1]


def rows(name):
    with open(f"{root}/{name}", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


manifest = json.load(open(f"{root}/manifest.json", encoding="utf-8"))
methods, bounds, summary = rows("method_realizations.csv"), rows("bounds.csv"), rows("summary.csv")
print(manifest["status"], manifest["task_count"], "failures", len(manifest["failures"]))
print(len(methods), "method rows;", len(bounds), "bound rows; statuses", sorted({row["status"] for row in bounds}))
print("bounded trajectories", sorted({row["trajectory_method"] or "ground" for row in bounds}))


def call_limit(method):
    if method in ("B0", "B1"):
        return 1
    budget = re.search(r"_b(\d+)$", method)
    return (int(budget.group(1)) if budget else 2000) + 1


print(len(summary), "summary rows; evaluate calls within budget:", all(int(row["evaluate_calls"]) <= call_limit(row["method"]) for row in methods))
PY
```

Expected:

```text
complete 4950 failures 0
13500 method rows; 4500 bound rows; statuses ['optimal']
bounded trajectories ['B1']
45 summary rows; evaluate calls within budget: True
```

Fifteen sets (13 families, `v0`, `v0-bh50`) with 10 scenarios each; B1 keeps its 30 bound tasks per scenario, and B2 and P run without bounds.

- [ ] **Step 3: Run and check the design experiment**

Run: `uv run python experiments/run_phase2.py --config configs/experiments/phase2_design.yaml`

Expected: `{"task_count": 30, "executed_task_count": 30, "skipped_task_count": 0, "status": "complete"}` and exit code 0 after about five minutes with 12 workers.

```bash
uv run python - results/phase2/design <<'PY'
import csv
import json
import re
import sys

root = sys.argv[1]


def rows(name):
    with open(f"{root}/{name}", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


manifest = json.load(open(f"{root}/manifest.json", encoding="utf-8"))
methods, bounds, summary = rows("method_realizations.csv"), rows("bounds.csv"), rows("summary.csv")
print(manifest["status"], manifest["task_count"], "failures", len(manifest["failures"]))
print(len(methods), "method rows;", len(bounds), "bound rows; statuses", sorted({row["status"] for row in bounds}))
print("bounded trajectories", sorted({row["trajectory_method"] or "ground" for row in bounds}))


def call_limit(method):
    if method in ("B0", "B1"):
        return 1
    budget = re.search(r"_b(\d+)$", method)
    return (int(budget.group(1)) if budget else 2000) + 1


print(len(summary), "summary rows; evaluate calls within budget:", all(int(row["evaluate_calls"]) <= call_limit(row["method"]) for row in methods))
PY
```

Expected:

```text
complete 30 failures 0
900 method rows; 0 bound rows; statuses []
bounded trajectories []
3 summary rows; evaluate calls within budget: True
```

`P_design10` evaluates ten design realizations per call, so it is the slowest of the three variants.

- [ ] **Step 4: Document the Phase 2 commands**

Append to `README.md`:

````markdown
## Pha 2: phương pháp tìm kiếm, thí nghiệm và thống kê

Sinh 13 họ scenario độ nhạy; mỗi họ chỉ khác `configs/scenarios/v0.yaml` ở các trường liệt kê trong spec Pha 2, Mục 11.3:

```bash
for config in configs/scenarios/sens-*.yaml; do uv run python -m data.cli scenarios --config "$config"; done
```

Chạy các thí nghiệm Pha 2. Kết quả nằm trong `results/phase2/<tên>/`; thư mục `shards/` bên trong bị Git ignore và giúp chạy lại bỏ qua task đã xong:

```bash
uv run python experiments/run_phase2.py --config configs/experiments/phase2_pilot.yaml        # chạy thử B1, B2, B3, P trên tập phát triển
uv run python experiments/run_phase2.py --config configs/experiments/phase2_main.yaml         # B0, B1, B2, B3, P và bảy biến thể ablation trên v0, v0-bh50
uv run python experiments/run_phase2.py --config configs/experiments/phase2_budget.yaml       # B2 và P với ngân sách 250 đến 4000 lần đánh giá
uv run python experiments/run_phase2.py --config configs/experiments/phase2_sensitivity.yaml  # B1, B2, P trên replicate 0 của 15 họ scenario
uv run python experiments/run_phase2.py --config configs/experiments/phase2_design.yaml       # P thiết kế trên tốc độ kỳ vọng, 1 hoặc 10 realization
```

Tính kiểm định ghép cặp theo topology (tám so sánh, hiệu chỉnh Holm) từ thí nghiệm chính:

```bash
uv run python experiments/phase2_statistics.py
```

Tổng quan tài liệu, bảng so sánh và lý do chọn baseline từ tài liệu nằm ở `docs/literature/review.md`.
````

- [ ] **Step 5: Run the full suite**

Run: `uv run --extra test pytest -q`

Expected: 205 passed, 1 skipped.

- [ ] **Step 6: Confirm that only result files are new, then commit**

Run: `git status --short --untracked-files=all results/phase2 README.md`

Expected: the twelve result files and the modified `README.md`; no shard appears.

```bash
git add results/phase2/budget/method_realizations.csv results/phase2/budget/bounds.csv results/phase2/budget/summary.csv results/phase2/budget/manifest.json results/phase2/sensitivity/method_realizations.csv results/phase2/sensitivity/bounds.csv results/phase2/sensitivity/summary.csv results/phase2/sensitivity/manifest.json results/phase2/design/method_realizations.csv results/phase2/design/bounds.csv results/phase2/design/summary.csv results/phase2/design/manifest.json README.md
git commit -m "feat: record phase 2 budget, sensitivity, and design results"
```

---

## Spec Coverage

| Spec section | Task |
|---|---|
| 11.1 Main comparison and ablations | 4, 7 |
| 11.2 Quality–runtime | 4, 8 |
| 11.3 Sensitivity families and design-realization count | 2, 3, 4, 8 |
| 12 Statistics | 5, 7 |
| 13 Literature review and strong baseline | 6 |
| 14 Report updates | Plan C.3 (spec Section 18) |
| 15 Error handling: family generation stops with the family name | 3 |
| 16.2 Testing: statistics and sensitivity configurations | 3, 5; report data tests move to plan C.3 |
| 17.2.1 Full test suite | every task |
| 17.2.2 Main comparison complete, LPs optimal, rerun skips, results committed | 7 |
| 17.2.3 Budget and sensitivity complete, manifests committed | 3, 8 |
| 17.2.4 Eight primary comparisons with Holm | 7 |
| 17.2.5 Review, related-work table, verified bibliography, baseline decision | 6 |
| 17.2.6 Report and README | README: 8; report: plan C.3 |
| 18 Plan C.2 adjustments: faster scoring, replicate filter, experiment sizes | 1, 2, 4 |
