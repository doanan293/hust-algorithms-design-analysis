# Local Search Blocks — Plan E.1 (Methods) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the path and trajectory blocks of the evaluator-guided BCD into genuine local search (neighbourhoods generated from the incumbent plan and its ledger), keep every other method reproducing its current results, and run the development-split pilot that fixes the new parameters.

**Architecture:** A `Plan` stores its paths as values (`CandidatePath` objects), so the search can generate paths outside the K-candidate set; the simulator, checker and evaluator read paths from the plan. Two new modules generate neighbours from the incumbent: `path_moves.py` (kind flip, entry switch weighted by estimated completion time, backhaul detour around the most congested link, unserve) and `trajectory_moves.py` (feasible segment shifts toward congested UAV links, segments sampled by priority). A `LedgerSummary` turns the design-realization ledgers of one scoring call into congestion counts and residual backhaul capacity. All randomness flows through one seeded `numpy` generator per search.

**Tech Stack:** Python ≥ 3.11, numpy, pytest (`uv run --extra test pytest -q`), uv-managed `.venv`; experiments via `experiments/run_phase2.py` under `systemd-run` memory limits.

**Spec:** `docs/superpowers/specs/2026-10-08-local-search-blocks-design.md` (Sections 4–7 and 10–12 are implemented here; Sections 8–9 — the full rerun, report and slides — are **Plan E.2**, written after this plan's pilot numbers exist, following the C.1/C.2 convention of `2026-09-11-phase2-core-design.md`).

## Global Constraints

- Python ≥ 3.11, dependencies only from `pyproject.toml` (no new packages). Run everything with `uv run`; tests with `uv run --extra test pytest -q` from the repo root (`tests/support` is on the test path via `conftest`/`pythonpath`).
- Every random choice of the search uses the generator `named_rng(scenario.scenario_seed, f"search:{name}:{seed}")` (`src/models/rng.py`); nothing reads global random state (spec 7.1).
- Acceptance rule of every move stays "key increases strictly" via `SearchState.try_change`; every scoring call goes through `DesignScorer` so the evaluation budget counts it (spec 5.2, 6.2, 11.4).
- B0, B1, TRAN and the Phase 1 bounds must reproduce their committed results exactly on non-runtime columns (spec 4.3); Task 3 is the gate and nothing after it may change those methods.
- Memory-heavy runs (pilot, reproduction) use `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run ...` (WSL VM has 24 GB).
- Commit messages follow the repo style (`feat:`, `fix:`, `test:`, `docs:` prefixes, imperative, one line). **No `Co-Authored-By` or other attribution trailers** (user rule).
- Code style: match the surrounding modules — short module docstrings, docstrings only where they add information, no inline comment noise, `from models... import` absolute imports inside `src/`.
- Spec defaults until the pilot (Task 10) says otherwise: `entry_neighbors = 3`, `trajectory_segments = 5`, `catalog_probability = 0.1`, `max_widening = 10`, `seed = 0`, `trajectory_mode = "local"`.

## Review Focus

Inputs the spec implies but does not spell out; each line's test is added to the owning task.

1. **Entry node equal to the center (zero backhaul links).** A path `source → center` has no backhaul link: validation must accept it, the detour must return `None`, the kind flip must still work. — Task 1 (validation) and Task 6 (`backhaul_detour`, `kind_flip`).
2. **Detour that disconnects the node.** Removing the congested edge can leave `u` with no route to the center; the move must return `None`, not raise. — Task 6.
3. **Segment touching the fixed endpoints.** Shifting `q[1..j]` or `q[i..N−1]` must leave `q[0]` and `q[N]` untouched and stay feasible. — Task 7.
4. **Nothing to sample, or a zero estimated time.** All priorities zero (no congested UAV link) or all entry candidates filtered out: the samplers return an empty list rather than passing a zero-sum `p` to `rng.choice` (Task 7 test "handles no congestion"; Task 6 guards `if not options`). An entry at the UAV's hover point has `T(e) = 0`: the weight `1/T` is floored (`MIN_TIME_S`) so `p` stays finite. — Tasks 6 and 7.
5. **Alert window beyond the horizon or empty.** `deadline_slot > num_slots` must clamp; `estimated_completion_s` with an empty window must fall back to the source→entry distance. — Task 6 (`alert_window`).

---

### Task 1: `Plan.paths` and structural path validation

**Files:**
- Modify: `src/models/plan.py` (whole `Plan`, `chosen_paths`, `validate_plan`, `_weight_violations`, `_schedule_violations`)
- Modify: `src/models/paths.py` (add `path_violations`, `radio_links_of_paths`)
- Test: `tests/models/test_plan.py`, `tests/models/test_paths.py`

**Interfaces:**
- Produces: `Plan(trajectory, paths: Mapping[str, CandidatePath | None], bandwidth)`; `Plan.from_choice(trajectory, candidates, choice: Mapping[str, int | None], bandwidth) -> Plan`; `validate_plan(scenario, plan) -> list[str]`; `chosen_paths(plan) -> dict[str, CandidatePath | None]`; `path_violations(scenario, source_id, path) -> list[str]`; `radio_links_of_paths(paths: Mapping[str, CandidatePath | None]) -> tuple[LinkId, ...]`.
- Note: this task leaves callers (`simulate`, `evaluate`, B0/B1, …) broken until Task 2; run only the tests named here.

- [ ] **Step 1: Write the failing tests**

Replace the body of `tests/models/test_plan.py` with:

```python
import numpy as np

from models.paths import CandidatePath, candidate_paths
from models.plan import EqualSplitBacklogged, Plan, StaticSchedule, stationary_trajectory, validate_plan
from models.scenario import UAV_ID, scenario_from_dict

ACCESS_N1 = ("access", "s00", "n1")
BACKHAUL_N1 = ("backhaul", "n1", "n0")


def _setup(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    return scenario, candidate_paths(scenario)


def _plan(scenario, path, bandwidth=None):
    return Plan(stationary_trajectory(scenario), {"alert-0000": path}, bandwidth or EqualSplitBacklogged())


def test_from_choice_matches_a_plan_built_from_the_same_paths(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    trajectory = stationary_trajectory(scenario)
    plan = Plan.from_choice(trajectory, candidates, {"alert-0000": 2}, EqualSplitBacklogged())
    assert plan.paths == {"alert-0000": candidates["alert-0000"][2]}
    assert Plan.from_choice(trajectory, candidates, {"alert-0000": None}, EqualSplitBacklogged()).paths == {"alert-0000": None}
    assert validate_plan(scenario, plan) == []


def test_stationary_equal_split_plan_is_valid(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    assert validate_plan(scenario, _plan(scenario, candidates["alert-0000"][0])) == []
    assert validate_plan(scenario, _plan(scenario, None)) == []


def test_trajectory_endpoint_and_speed_violations(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    trajectory = stationary_trajectory(scenario)
    trajectory[0] = [1000.0, 1001.0]
    trajectory[5] = [1100.0, 1000.0]
    violations = validate_plan(scenario, Plan(trajectory, {"alert-0000": candidates["alert-0000"][0]}, EqualSplitBacklogged()))
    assert any("q[0] is not the start point" in item for item in violations)
    assert any("slot 4 moves" in item for item in violations)
    assert any("slot 5 moves" in item for item in violations)


def test_missing_and_unknown_alerts(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    trajectory = stationary_trajectory(scenario)
    assert "paths: missing alert alert-0000" in validate_plan(scenario, Plan(trajectory, {}, EqualSplitBacklogged()))
    plan = Plan(trajectory, {"alert-0000": None, "alert-9999": None}, EqualSplitBacklogged())
    assert "paths: unknown alert alert-9999" in validate_plan(scenario, plan)


def test_path_structure_violations(raw_scenario):
    scenario, _ = _setup(raw_scenario)
    cases = {
        "wrong source": CandidatePath("ground", "n1", (("access", "s01", "n1"), BACKHAUL_N1), 1.0),
        "ground kind on a uav path": CandidatePath("ground", "n1", (("access", "s00", UAV_ID), ("down", UAV_ID, "n1"), BACKHAUL_N1), 1.0),
        "downlink to another node": CandidatePath("uav", "n1", (("access", "s00", UAV_ID), ("down", UAV_ID, "n2"), BACKHAUL_N1), 1.0),
        "out of ground range": CandidatePath("ground", "n3", (("access", "s00", "n3"), ("backhaul", "n3", "n1"), BACKHAUL_N1), 1.0),
        "broken chain": CandidatePath("ground", "n1", (ACCESS_N1, ("backhaul", "n2", "n0")), 1.0),
        "dead edge": CandidatePath("ground", "n1", (ACCESS_N1, ("backhaul", "n1", "n2"), ("backhaul", "n2", "n0")), 1.0),
        "repeated node": CandidatePath("ground", "n1", (ACCESS_N1, ("backhaul", "n1", "n3"), ("backhaul", "n3", "n1"), BACKHAUL_N1), 1.0),
        "does not end at the center": CandidatePath("ground", "n1", (ACCESS_N1,), 1.0),
    }
    for label, path in cases.items():
        violations = validate_plan(scenario, _plan(scenario, path))
        assert violations and all(item.startswith("paths: alert alert-0000") for item in violations), label


def test_entry_at_the_center_needs_no_backhaul(raw_scenario):
    scenario, _ = _setup(raw_scenario)
    direct = CandidatePath("ground", "n0", (("access", "s00", "n0"),), 1.0)
    via_uav = CandidatePath("uav", "n0", (("access", "s00", UAV_ID), ("down", UAV_ID, "n0")), 1.0)
    assert validate_plan(scenario, _plan(scenario, direct)) == []
    assert validate_plan(scenario, _plan(scenario, via_uav)) == []


def test_static_schedule_violations(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    schedule = StaticSchedule(
        {
            ("access", "s00", "n1"): np.full(20, 6e5),
            ("access", "s00", "n0"): np.full(20, 6e5),
            ("down", UAV_ID, "n1"): np.full(20, -1.0),
            ("down", UAV_ID, "n9"): np.zeros(20),
        }
    )
    violations = validate_plan(scenario, _plan(scenario, candidates["alert-0000"][0], schedule))
    assert any("slot 0 access total" in item for item in violations)
    assert any("finite non-negative" in item for item in violations)
    assert any("unknown radio link" in item for item in violations)


def test_static_schedule_downlink_cap(raw_scenario):
    raw_scenario["uav"]["max_active_downlinks"] = 1
    raw_scenario["paths"]["max_ground"] = 0
    scenario, candidates = _setup(raw_scenario)
    downlinks = [link for path in candidates["alert-0000"] for link in path.links if link[0] == "down"]
    schedule = StaticSchedule({link: np.full(20, 1e5) for link in downlinks[:2]})
    violations = validate_plan(scenario, _plan(scenario, candidates["alert-0000"][0], schedule))
    assert any("has 2 active downlinks" in item for item in violations)
```

Append to `tests/models/test_paths.py`:

```python
def test_radio_links_of_paths_skips_backhaul_and_unserved(raw_scenario):
    from models.paths import radio_links_of_paths

    scenario = scenario_from_dict(raw_scenario)
    paths = candidate_paths(scenario)["alert-0000"]
    links = radio_links_of_paths({"alert-0000": paths[2], "alert-0001": None})
    assert links == (("access", "s00", "UAV"), ("down", "UAV", "n1"))
```

(Check the existing imports of `tests/models/test_paths.py`; it already imports `candidate_paths` and `scenario_from_dict`, otherwise add them.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra test pytest tests/models/test_plan.py tests/models/test_paths.py -q`
Expected: failures with `TypeError` (Plan has no `paths`/`from_choice`) and `ImportError: radio_links_of_paths`.

- [ ] **Step 3: Add `path_violations` and `radio_links_of_paths` to `src/models/paths.py`**

Append after `radio_links`:

```python
def radio_links_of_paths(paths: Mapping[str, "CandidatePath | None"]) -> tuple[LinkId, ...]:
    """Sorted radio links of the paths a plan uses; unserved alerts contribute nothing."""
    return tuple(sorted({link for path in paths.values() if path is not None for link in path.links if link[0] != BACKHAUL}))


def path_violations(scenario: Scenario, source_id: str, path: CandidatePath) -> list[str]:
    """Why `path` is not a valid route of an alert of `source_id`: structure, range, alive edges, chain to the center."""
    links = path.links
    if not links or links[0][0] != ACCESS or links[0][1] != source_id:
        return [f"first link {links[0] if links else None} is not an access link of {source_id}"]
    receiver = links[0][2]
    if receiver == UAV_ID:
        if path.kind != VIA_UAV:
            return [f"uses the UAV but has kind {path.kind!r}"]
        if path.entry not in scenario.node_by_id:
            return [f"entry {path.entry} is not a node"]
        if len(links) < 2 or links[1] != (DOWNLINK, UAV_ID, path.entry):
            return [f"second link must be the downlink to entry {path.entry}"]
        backhaul = links[2:]
    else:
        if path.kind != GROUND:
            return [f"enters the ground network but has kind {path.kind!r}"]
        if receiver != path.entry or path.entry not in scenario.node_by_id:
            return [f"entry {path.entry} does not match the access link {links[0]}"]
        distance = math.dist(scenario.source_by_id[source_id].xy, scenario.node_by_id[path.entry].xy)
        if not in_ground_range(scenario, distance):
            return [f"entry {path.entry} is out of ground range ({distance:.1f} m)"]
        backhaul = links[1:]
    problems = []
    current = path.entry
    visited = {current}
    for link in backhaul:
        if link[0] != BACKHAUL or link[1] != current:
            problems.append(f"link {link} breaks the chain at {current}")
            break
        if scenario.backhaul_capacity_bps(link[1], link[2]) <= 0.0:
            problems.append(f"link {link} is not an alive edge")
        if link[2] in visited:
            problems.append(f"node {link[2]} repeats")
        visited.add(link[2])
        current = link[2]
    if current != scenario.center:
        problems.append(f"ends at {current}, not at the center {scenario.center}")
    return problems
```

`paths.py` already imports `math`, `ACCESS`, `BACKHAUL`, `DOWNLINK`, `in_ground_range`, `UAV_ID`. `GROUND`/`VIA_UAV` are defined above in the same module.

- [ ] **Step 4: Rewrite `Plan`, `chosen_paths`, `validate_plan` in `src/models/plan.py`**

Replace the imports and the `Plan`/`chosen_paths`/`validate_plan`/`_weight_violations`/`_schedule_violations` definitions (keep `EqualSplitBacklogged`, `StaticSchedule`, `WeightedBacklogged`, `stationary_trajectory`, `_trajectory_violations`, the tolerances):

```python
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .channel import ACCESS, DOWNLINK, LinkId
from .paths import CandidatePath, path_violations
from .scenario import UAV_ID, Scenario
```

```python
@dataclass(frozen=True)
class Plan:
    """Trajectory at slot boundaries, the path of every alert (None = unserved), and the bandwidth policy."""

    trajectory: np.ndarray
    paths: Mapping[str, CandidatePath | None]
    bandwidth: EqualSplitBacklogged | StaticSchedule | WeightedBacklogged

    @classmethod
    def from_choice(
        cls,
        trajectory: np.ndarray,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        choice: Mapping[str, int | None],
        bandwidth: "EqualSplitBacklogged | StaticSchedule | WeightedBacklogged",
    ) -> "Plan":
        """Plan from candidate indices, for methods that choose within the candidate set."""
        paths = {alert_id: None if index is None else candidates[alert_id][index] for alert_id, index in choice.items()}
        return cls(trajectory, paths, bandwidth)


def chosen_paths(plan: Plan) -> dict[str, CandidatePath | None]:
    return dict(plan.paths)


def validate_plan(scenario: Scenario, plan: Plan) -> list[str]:
    violations = _trajectory_violations(scenario, plan.trajectory)
    alert_ids = {alert.id for alert in scenario.alerts}
    for alert_id in sorted(alert_ids - set(plan.paths)):
        violations.append(f"paths: missing alert {alert_id}")
    for alert_id in sorted(plan.paths):
        if alert_id not in alert_ids:
            violations.append(f"paths: unknown alert {alert_id}")
            continue
        path = plan.paths[alert_id]
        if path is not None:
            source_id = scenario.alert_by_id[alert_id].source
            violations.extend(f"paths: alert {alert_id} {problem}" for problem in path_violations(scenario, source_id, path))
    if isinstance(plan.bandwidth, StaticSchedule):
        violations.extend(_schedule_violations(scenario, plan.bandwidth))
    elif isinstance(plan.bandwidth, WeightedBacklogged):
        violations.extend(_weight_violations(scenario, plan.bandwidth))
    elif not isinstance(plan.bandwidth, EqualSplitBacklogged):
        violations.append("bandwidth: unknown policy")
    return violations


def _is_radio_link(scenario: Scenario, link: LinkId) -> bool:
    kind, transmitter, receiver = link
    if kind == ACCESS:
        return transmitter in scenario.source_by_id and (receiver == UAV_ID or receiver in scenario.node_by_id)
    return kind == DOWNLINK and transmitter == UAV_ID and receiver in scenario.node_by_id


def _weight_violations(scenario: Scenario, policy: WeightedBacklogged) -> list[str]:
    violations = []
    for link in sorted(policy.weights):
        if not _is_radio_link(scenario, link):
            violations.append(f"bandwidth: unknown radio link {link}")
            continue
        values = np.asarray(policy.weights[link], dtype=float)
        if values.shape != (scenario.time.num_slots,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            violations.append(f"bandwidth: weights of {link} need {scenario.time.num_slots} finite non-negative values")
    return violations


def _schedule_violations(scenario: Scenario, schedule: StaticSchedule) -> list[str]:
    num_slots = scenario.time.num_slots
    access_total = np.zeros(num_slots)
    downlink_total = np.zeros(num_slots)
    active_downlinks = np.zeros(num_slots, dtype=int)
    violations = []
    for link in sorted(schedule.bandwidth_hz):
        if not _is_radio_link(scenario, link):
            violations.append(f"bandwidth: unknown radio link {link}")
            continue
        values = np.asarray(schedule.bandwidth_hz[link], dtype=float)
        if values.shape != (num_slots,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            violations.append(f"bandwidth: link {link} needs {num_slots} finite non-negative values")
            continue
        if link[0] == ACCESS:
            access_total += values
        else:
            downlink_total += values
            active_downlinks += values > 0.0
    limit = scenario.spectrum.b_tot_hz * (1.0 + RELATIVE_TOLERANCE)
    for label, totals in (("access", access_total), ("downlink", downlink_total)):
        for slot in np.flatnonzero(totals > limit):
            violations.append(f"bandwidth: slot {slot} {label} total {totals[slot]:.1f} Hz exceeds B_tot")
    for slot in np.flatnonzero(active_downlinks > scenario.uav.max_active_downlinks):
        violations.append(f"bandwidth: slot {slot} has {active_downlinks[slot]} active downlinks")
    return violations
```

`_trajectory_violations` keeps its current body. `UAV_ID` lives in `models.scenario` (check: `from .scenario import Scenario, UAV_ID` is how `channel.py` imports it).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra test pytest tests/models/test_plan.py tests/models/test_paths.py -q`
Expected: all PASS (other test files are still broken; do not run the full suite yet).

- [ ] **Step 6: Commit**

```bash
git add src/models/plan.py src/models/paths.py tests/models/test_plan.py tests/models/test_paths.py
git commit -m "feat: store paths as values in Plan and validate their structure"
```

---

### Task 2: Plan-based simulator, checker, evaluator and every producer

**Files:**
- Modify: `src/models/simulator.py:20-33` (signature, `paths`), `src/models/checker.py:21-33`, `src/models/evaluate.py:67-116`
- Modify: `src/baselines/b0.py`, `src/baselines/b1.py`, `src/literature/tran_plan.py:35-60`, `src/bounds/crosscheck.py:40-70`, `src/runner/tasks.py:195-204`, `experiments/calibrate_v0.py:23-30`
- Modify: `src/optimization/scoring.py` (drop `candidates`), `src/optimization/blocks.py`, `src/optimization/repair.py`, `src/optimization/bcd.py` (index → path values, behaviour unchanged)
- Test: every test that calls `simulate`, `check_ledger`, `validate_plan`, `evaluate`, `DesignScorer`, or reads `path_choice` (list below)

**Interfaces:**
- Consumes: `Plan.paths`, `Plan.from_choice`, `validate_plan(scenario, plan)`, `radio_links_of_paths` from Task 1.
- Produces: `simulate(scenario, plan, channels, realization_id=None)`; `check_ledger(scenario, plan, channels, ledger, delivered_bits=None)`; `evaluate(scenario, plan, mode, realization_ids=(), counter=None, keep_ledger=False, channel_namespace="eval", check=True, connectivity_cache=None)`; `DesignScorer(scenario, design_ids=DEFAULT_DESIGN_IDS, budget=DEFAULT_BUDGET, objective=lexicographic_key)`; `SearchState(trajectory, paths: dict[str, CandidatePath | None], weights, key)`; `links_by_load(scenario, paths)`; `assess_risk(scenario, paths, result, backlog, ...)`; `path_change_order(candidates, risk, current: CandidatePath | None, backlog) -> list[CandidatePath]`.

- [ ] **Step 1: Rewrite the three evaluator entry points**

`src/models/simulator.py` — signature and the first body lines:

```python
def simulate(
    scenario: Scenario,
    plan: Plan,
    channels: Mapping[LinkId, LinkChannel],
    realization_id: int | None = None,
) -> SimulationOutcome:
    slot_s = scenario.time.slot_s
    access_s = scenario.time.access_fraction * slot_s
    downlink_s = slot_s - access_s
    paths = chosen_paths(plan)
```

Remove the `CandidatePath` import if unused. `src/models/checker.py`:

```python
def check_ledger(
    scenario: Scenario,
    plan: Plan,
    channels: Mapping[LinkId, LinkChannel],
    ledger: Ledger,
    delivered_bits: Mapping[str, float] | None = None,
) -> list[str]:
    """Replay the ledger without simulator code and list every broken invariant."""
    paths = {alert.id: plan.paths[alert.id] for alert in scenario.alerts}
```

`src/models/evaluate.py` — drop the `candidates` parameter and its default, validate and build channels from the plan:

```python
from .paths import radio_links_of_paths
...
def evaluate(
    scenario: Scenario,
    plan: Plan,
    mode: str,
    realization_ids: Sequence[int] = (),
    counter: EvaluationCounter | None = None,
    keep_ledger: bool = False,
    channel_namespace: str = "eval",
    check: bool = True,
    connectivity_cache: dict | None = None,
) -> EvaluationResult:
    ...
    violations = validate_plan(scenario, plan)
    ...
    links = set(radio_links_of_paths(plan.paths)) | set(all_uav_links(scenario))
    ...
        outcome = simulate(scenario, plan, channels, realization_id)
        if check:
            problems = check_ledger(scenario, plan, channels, outcome.ledger, outcome.delivered_bits)
```

Remove the now-unused imports `CandidatePath`, `candidate_paths`, `radio_links` from `evaluate.py`.

- [ ] **Step 2: Rewrite the index-based producers with `Plan.from_choice`**

`src/baselines/b0.py` `plan`: `return Plan.from_choice(self.trajectory(scenario), candidates, choice, EqualSplitBacklogged())`.

`src/baselines/b1.py`: in `plan`, `return Plan.from_choice(trajectory, candidates, choice, EqualSplitBacklogged())`; in `estimated_best_candidate`:

```python
        plan = Plan.from_choice(trajectory, candidates, {alert_id: index}, EqualSplitBacklogged())
        slot = simulate(alone, plan, channels).delivery_slot[alert_id]
```

`src/literature/tran_plan.py` tail of `plan_from_iterate`:

```python
    path_choice = {device.alert_id: device.candidate for device in instance.devices}
    plan = Plan.from_choice(expand_trajectory(iterate.trajectory, G), candidates, path_choice, StaticSchedule(bandwidth))
    violations = validate_plan(scenario, plan)
```

`src/bounds/crosscheck.py` `plan_from_solution`: end with `return Plan.from_choice(trajectory, candidates, choice, StaticSchedule(bandwidth))` (keep the bandwidth clean-up loops; the current `return Plan(trajectory, choice, StaticSchedule(bandwidth))` becomes that call).

`src/runner/tasks.py` `run_crosscheck_task`: `timely_count` calls `evaluate(scenario, plan, REALIZED, (task.realization_id,))`; the equal-split line becomes `timely_count(Plan(trajectory, static_plan.paths, EqualSplitBacklogged()))`. `run_method_task`: `evaluate(scenario, plan, REALIZED, task.realization_ids, counter=counter)`.

`experiments/calibrate_v0.py`: both builders return `Plan.from_choice(<trajectory>, candidates, choice, EqualSplitBacklogged())`; the `evaluate` call drops `candidates=candidates`. Same drop in `experiments/case_study_trace.py:85` and `src/literature/pilot.py:36-37`.

- [ ] **Step 3: Adapt the search modules to path values (same behaviour)**

`src/optimization/scoring.py`: remove the `candidates` constructor parameter and attribute; `evaluate(...)` no longer passes `candidates=`. New signature: `DesignScorer(scenario, design_ids=DEFAULT_DESIGN_IDS, budget=DEFAULT_BUDGET, objective=lexicographic_key)`.

`src/optimization/blocks.py`:

```python
@dataclass
class SearchState:
    trajectory: np.ndarray
    paths: dict[str, CandidatePath | None]
    weights: dict[LinkId, np.ndarray]
    key: Key = INFEASIBLE_SCORE

    def plan(self) -> Plan:
        return Plan(self.trajectory, dict(self.paths), WeightedBacklogged(dict(self.weights)))

    def try_change(self, scorer: DesignScorer, **changes: object) -> bool:
        trial = replace(self, **changes)
        key = scorer.score(trial.plan())
        if key <= self.key:
            return False
        self.trajectory, self.paths, self.weights, self.key = trial.trajectory, trial.paths, trial.weights, key
        return True


def path_block(scenario, candidates, scorer, state) -> bool:
    """Alerts by (deadline, id); options None and every candidate except the current one, each tried from the incumbent."""
    accepted = False
    for alert in sorted(scenario.alerts, key=lambda item: (item.deadline_slot, item.id)):
        current = state.paths[alert.id]
        for option in (None, *candidates[alert.id]):
            if option != current and state.try_change(scorer, paths={**state.paths, alert.id: option}):
                accepted = True
    return accepted


def links_by_load(scenario: Scenario, paths: Mapping[str, CandidatePath | None]) -> list[LinkId]:
    load: dict[LinkId, int] = defaultdict(int)
    for alert in scenario.alerts:
        path = paths[alert.id]
        if path is not None:
            for link in path.links:
                if link[0] != BACKHAUL:
                    load[link] += alert.size_bits
    return sorted(load, key=lambda link: (-load[link], link))


def bandwidth_block(scenario: Scenario, scorer: DesignScorer, state: SearchState) -> bool:
    accepted = False
    for link in links_by_load(scenario, state.paths):
        ...  # unchanged
```

`src/optimization/repair.py`: `assess_risk(scenario, paths, result, backlog=True, ...)` — replace `path_choice[alert.id] is None` with `paths[alert.id] is None`, and `candidates[alert_id][path_choice[alert_id]]` with `paths[alert_id]`; drop the `candidates` parameter. `path_change_order(candidates, risk, current, backlog=True) -> list[CandidatePath]` returns the candidate paths `!= current` sorted by the same key (congested count, `cost_s`, candidate index). `repair_round`: `assess_risk(scenario, state.paths, result, backlog)`; Operation 2 loop: `for path in path_change_order(candidates, risk, state.paths[alert.id], backlog): if state.try_change(scorer, paths={**state.paths, alert.id: path}): ...`.

`src/optimization/bcd.py` `search`: `scorer = DesignScorer(scenario, params.design_ids, params.budget, OBJECTIVES[params.objective])`; `state = SearchState(initial.trajectory, dict(initial.paths), unit_weights(scenario, candidates))`; `bandwidth_block(scenario, scorer, state)`.

- [ ] **Step 4: Sweep the tests**

Run `uv run --extra test pytest -q 2>&1 | tail -40` and fix every failing call site with these rewrite rules (grep: `grep -rn "candidates=\|path_choice\|simulate(\|check_ledger(\|validate_plan(\|DesignScorer(" tests`):

| Old | New |
|---|---|
| `evaluate(scenario, plan, MODE, ids, candidates=candidates)` | `evaluate(scenario, plan, MODE, ids)` |
| `simulate(scenario, candidates, plan, channels[, rid])` | `simulate(scenario, plan, channels[, rid])` |
| `check_ledger(scenario, candidates, plan, ...)` | `check_ledger(scenario, plan, ...)` |
| `validate_plan(scenario, candidates, plan)` | `validate_plan(scenario, plan)` |
| `Plan(trajectory, {"alert-0000": 0}, bw)` (index dict) | `Plan.from_choice(trajectory, candidates, {"alert-0000": 0}, bw)` |
| `DesignScorer(scenario, candidates, ...)` | `DesignScorer(scenario, ...)` |
| `SearchState(traj, dict(choice), weights)` in `tests/optimization/test_blocks.py::_start` | `SearchState(traj, {aid: None if i is None else candidates[aid][i] for aid, i in choice.items()}, weights)` |
| `state.path_choice == {...}` | `state.paths == {aid: candidates[aid][i] or None ...}` (build the expected dict from `candidates`) |
| `plan.path_choice` in `_same_plan` (test_search_methods) | `first.paths == second.paths` |
| `path_change_order(candidates, risk, 0) == [2, 3, 1]` | `path_change_order(candidates, risk, paths[0]) == [paths[2], paths[3], paths[1]]`; `None` case `== [paths[2], paths[3], paths[0], paths[1]]` |
| `assess_risk(scenario, candidates, choice, result, ...)` | `assess_risk(scenario, paths, result, ...)` |
| `tests/support/channel_builders.py::constant_channels(scenario, candidates, ...)` | leave as is (it only needs link ids) |

Files expected to need edits: `tests/models/test_evaluate.py`, `test_simulator.py`, `test_checker.py`, `test_calibration.py`, `test_design_extensions.py`, `tests/baselines/test_baselines.py`, `tests/literature/test_tran_plan.py`, `tests/literature/*` using `evaluate`, `tests/bounds/*`, `tests/optimization/test_blocks.py`, `test_repair.py`, `test_search_methods.py`, `test_objectives_scoring.py`, `tests/runner/*` that build plans.

- [ ] **Step 5: Run the whole suite**

Run: `uv run --extra test pytest -q`
Expected: all PASS, same test count as before Task 1 plus the new tests of Task 1.

- [ ] **Step 6: Commit**

```bash
git add -A src experiments tests
git commit -m "refactor: read paths from the plan in the simulator, checker and evaluator"
```

---

### Task 3: Reproduction gate

**Files:**
- No source changes. Output: `results/reproducibility/experiments-Agis.json`, `results/reproducibility/tables.json` (both already tracked; overwrite).

**Interfaces:**
- Consumes: the whole code base after Task 2.
- Produces: evidence that every method, including the still-catalogue-based B2/B3/P, reproduces its committed results.

- [ ] **Step 1: Run the experiments level on the Agis subset**

Run (foreground, ~30–45 min; use `run_in_background` and poll if the harness limits foreground time):

```bash
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 \
  uv run python experiments/reproduce.py --level experiments --subset Agis --workers 12
```

Expected: every line `PASS ...`, final JSON `"passed": true`. A `FAIL` on a `B0`, `B1`, `TRAN` or bound row means Task 1/2 changed behaviour — fix before continuing (likely culprits: channel link set in `evaluate`, `from_choice` index handling). A `FAIL` on `B2`/`B3`/`P` rows at this stage also indicates a Task 2 regression, since the blocks still choose within the candidate set.

- [ ] **Step 2: Run the tables level**

Run: `uv run python experiments/reproduce.py --level tables`
Expected: `"passed": true` (no result file changed, so the committed report data regenerates byte-identically).

- [ ] **Step 3: Commit the reproducibility reports**

```bash
git add results/reproducibility/experiments-Agis.json results/reproducibility/tables.json
git commit -m "test: reproduce the committed results with plan-stored paths"
```

---

### Task 4: New `SearchParams` fields

**Files:**
- Modify: `src/optimization/bcd.py:23-72` (`SearchParams`)
- Test: `tests/optimization/test_search_methods.py::test_search_params_from_mapping_validates_values`

**Interfaces:**
- Produces: `SearchParams.trajectory_mode: str ("local" | "catalog")`, `catalog_probability: float`, `entry_neighbors: int`, `trajectory_segments: int`, `max_widening: int`, `seed: int`; constants `LOCAL = "local"`, `CATALOG = "catalog"`, `TRAJECTORY_MODES = (LOCAL, CATALOG)` in `optimization/bcd.py`.

- [ ] **Step 1: Extend the failing test**

Replace `test_search_params_from_mapping_validates_values` with:

```python
def test_search_params_from_mapping_validates_values():
    assert SearchParams.from_mapping({}) == SearchParams()
    assert SearchParams.from_mapping({"design_ids": "expected", "operation1": True}).design_ids == ()
    assert SearchParams.from_mapping({"design_ids": [3, 4]}).design_ids == (3, 4)
    params = SearchParams.from_mapping(
        {"trajectory_mode": "catalog", "catalog_probability": 0.0, "entry_neighbors": 5, "trajectory_segments": 10, "max_widening": 0, "seed": 2}
    )
    assert (params.trajectory_mode, params.catalog_probability, params.entry_neighbors) == ("catalog", 0.0, 5)
    assert (params.trajectory_segments, params.max_widening, params.seed) == (10, 0, 2)
    assert (SearchParams().trajectory_mode, SearchParams().catalog_probability, SearchParams().seed) == ("local", 0.1, 0)
    for raw, message in (
        ({"repair": True}, "unknown search parameters"),
        ({"objective": "maxmin"}, "objective"),
        ({"budget": 0}, "budget"),
        ({"operation1": "yes"}, "operation1"),
        ({"design_ids": []}, "design_ids"),
        ({"design_ids": [1, 1]}, "design_ids"),
        ({"trajectory_mode": "random"}, "trajectory_mode"),
        ({"catalog_probability": 1.5}, "catalog_probability"),
        ({"catalog_probability": True}, "catalog_probability"),
        ({"entry_neighbors": 0}, "entry_neighbors"),
        ({"trajectory_segments": 0}, "trajectory_segments"),
        ({"max_widening": -1}, "max_widening"),
        ({"seed": -1}, "seed"),
    ):
        with pytest.raises(ValueError, match=message):
            SearchParams.from_mapping(raw)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra test pytest tests/optimization/test_search_methods.py::test_search_params_from_mapping_validates_values -q`
Expected: FAIL (`unknown search parameters ['trajectory_mode']`).

- [ ] **Step 3: Add the fields and validation**

In `src/optimization/bcd.py`, above `SearchParams`:

```python
LOCAL = "local"
CATALOG = "catalog"
TRAJECTORY_MODES = (LOCAL, CATALOG)
```

Add to the dataclass after `max_iterations`:

```python
    trajectory_mode: str = LOCAL
    catalog_probability: float = 0.1
    entry_neighbors: int = 3
    trajectory_segments: int = 5
    max_widening: int = 10
    seed: int = 0
```

Extend `__post_init__`:

```python
        if self.trajectory_mode not in TRAJECTORY_MODES:
            raise ValueError(f"trajectory_mode: expected one of {TRAJECTORY_MODES}, got {self.trajectory_mode!r}")
        probability = self.catalog_probability
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0.0 <= probability <= 1.0:
            raise ValueError("catalog_probability: expected a number in [0, 1]")
        for name in ("budget", "max_iterations", "entry_neighbors", "trajectory_segments"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}: expected an integer of at least 1")
        for name in ("max_widening", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name}: expected a non-negative integer")
```

(Replace the existing `for name in ("budget", "max_iterations")` loop with the two loops above.)

- [ ] **Step 4: Run the test file**

Run: `uv run --extra test pytest tests/optimization/test_search_methods.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/optimization/bcd.py tests/optimization/test_search_methods.py
git commit -m "feat: add local search parameters to SearchParams"
```

---

### Task 5: `LedgerSummary` — congestion and residual capacity from design ledgers

**Files:**
- Create: `src/optimization/ledger_summary.py`
- Modify: `src/optimization/repair.py` (import `congested_slots` from the new module; remove its local definition)
- Test: `tests/optimization/test_ledger_summary.py` (new); `tests/optimization/test_repair.py` keeps importing `congested_slots` from `optimization.repair` (re-export)

**Interfaces:**
- Consumes: `EvaluationResult.realizations[i].ledger` (`Ledger.transfers`, `Ledger.capacity_bits`).
- Produces:
  - `congested_slots(ledger) -> dict[LinkId, set[int]]` (moved).
  - `alert_window(scenario, alert) -> range` = `range(alert.release_slot, min(alert.deadline_slot, scenario.time.num_slots))`.
  - `LedgerSummary(realization_count: int, congested: tuple[dict[LinkId, set[int]], ...], carried_bits: dict[tuple[LinkId, int], float])` with `from_result(result)`, `congestion_count(link, window) -> int`, `congested_realizations(link, slot) -> int`, `carried_bps(link, window, slot_s) -> float`.
  - `residual_bps(scenario, summary, link, window) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/optimization/test_ledger_summary.py`:

```python
import pytest

from models.evaluate import EXPECTED, evaluate
from models.ledger import Ledger
from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from optimization.ledger_summary import LedgerSummary, alert_window, congested_slots, residual_bps
from scenario_builders import make_alert
from search_builders import base_scenario

ACCESS_LINK = ("access", "s00", "n1")
BACKHAUL_LINK = ("backhaul", "n1", "n0")


def _ledger(transfers, capacity):
    return Ledger(realization_id=0, transfers=tuple(transfers), bandwidth=(), capacity_bits=tuple(capacity), drops=())


def test_congested_links_carry_their_whole_capacity():
    other = ("access", "s01", "n0")
    ledger = _ledger(
        [(0, ACCESS_LINK, "a", 6.0), (0, ACCESS_LINK, "b", 4.0), (1, ACCESS_LINK, "a", 3.0)],
        [(0, ACCESS_LINK, 10.0), (1, ACCESS_LINK, 10.0), (1, other, 0.0)],
    )
    assert congested_slots(ledger) == {ACCESS_LINK: {0}, other: {1}}


def test_alert_window_clamps_to_the_horizon():
    scenario = base_scenario([make_alert("alert-0000", "s00", 3, 50, 100)])
    assert alert_window(scenario, scenario.alerts[0]) == range(3, 20)


def test_summary_counts_congestion_and_carried_bits_over_realizations():
    first = _ledger([(0, BACKHAUL_LINK, "a", 10.0), (1, BACKHAUL_LINK, "a", 10.0)], [(0, BACKHAUL_LINK, 10.0), (1, BACKHAUL_LINK, 10.0)])
    second = _ledger([(1, BACKHAUL_LINK, "a", 4.0)], [(1, BACKHAUL_LINK, 10.0)])
    summary = LedgerSummary(2, (congested_slots(first), congested_slots(second)), {(BACKHAUL_LINK, 0): 10.0, (BACKHAUL_LINK, 1): 14.0})
    assert summary.congestion_count(BACKHAUL_LINK, range(0, 2)) == 2
    assert summary.congestion_count(BACKHAUL_LINK, range(1, 2)) == 1
    assert summary.congested_realizations(BACKHAUL_LINK, 1) == 1
    assert summary.congested_realizations(ACCESS_LINK, 1) == 0
    assert summary.carried_bps(BACKHAUL_LINK, range(0, 2), 0.5) == pytest.approx(24.0 / (2 * 2 * 0.5))
    assert summary.carried_bps(BACKHAUL_LINK, range(5, 5), 0.5) == 0.0


def test_residual_is_capacity_minus_carried_and_floored():
    scenario = base_scenario([make_alert("alert-0000", "s00", 0, 10, 100)])
    capacity = scenario.backhaul_capacity_bps("n1", "n0")
    light = LedgerSummary(1, ({},), {(BACKHAUL_LINK, 0): 0.25 * capacity})
    assert residual_bps(scenario, light, BACKHAUL_LINK, range(0, 1)) == pytest.approx(0.75 * capacity)
    saturated = LedgerSummary(1, ({},), {(BACKHAUL_LINK, 0): capacity})
    assert residual_bps(scenario, saturated, BACKHAUL_LINK, range(0, 1)) == pytest.approx(capacity / 1000.0)
    assert residual_bps(scenario, light, ("backhaul", "n2", "n0"), range(0, 1)) == pytest.approx(scenario.backhaul_capacity_bps("n2", "n0"))


def test_from_result_needs_ledgers_and_reads_them():
    scenario = base_scenario([make_alert("alert-0000", "s00", 0, 10, 12000000)])
    candidates = candidate_paths(scenario)
    plan = Plan.from_choice(stationary_trajectory(scenario), candidates, {"alert-0000": 0}, EqualSplitBacklogged())
    with pytest.raises(ValueError, match="ledger"):
        LedgerSummary.from_result(evaluate(scenario, plan, EXPECTED))
    summary = LedgerSummary.from_result(evaluate(scenario, plan, EXPECTED, keep_ledger=True))
    assert summary.realization_count == 1
    assert summary.congestion_count(BACKHAUL_LINK, range(0, 10)) > 0
    assert summary.carried_bps(BACKHAUL_LINK, range(0, 10), 1.0) > 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra test pytest tests/optimization/test_ledger_summary.py -q`
Expected: `ImportError` on `optimization.ledger_summary`.

- [ ] **Step 3: Create the module and move `congested_slots`**

`src/optimization/ledger_summary.py`:

```python
"""Congestion and residual capacity of the incumbent plan, read from the ledgers of its design realizations."""

from collections import defaultdict
from dataclasses import dataclass

from models.channel import LinkId
from models.evaluate import EvaluationResult
from models.ledger import Ledger
from models.scenario import Alert, Scenario

CONGESTION_TOLERANCE = 1e-6
RESIDUAL_FLOOR = 1e-3


def congested_slots(ledger: Ledger) -> dict[LinkId, set[int]]:
    """Slots in which a link carried its whole recorded capacity (within 1e-6 relative)."""
    carried: dict[tuple[int, LinkId], float] = defaultdict(float)
    for slot, link, _, bits in ledger.transfers:
        carried[(slot, link)] += bits
    congested: dict[LinkId, set[int]] = defaultdict(set)
    for slot, link, capacity in ledger.capacity_bits:
        if carried.get((slot, link), 0.0) >= capacity * (1.0 - CONGESTION_TOLERANCE):
            congested[link].add(slot)
    return congested


def alert_window(scenario: Scenario, alert: Alert) -> range:
    """Slots in which the alert can still be delivered, clamped to the horizon."""
    return range(alert.release_slot, min(alert.deadline_slot, scenario.time.num_slots))


@dataclass(frozen=True)
class LedgerSummary:
    """Per-realization congested slots and carried bits per (link, slot) summed over the realizations."""

    realization_count: int
    congested: tuple[dict[LinkId, set[int]], ...]
    carried_bits: dict[tuple[LinkId, int], float]

    @classmethod
    def from_result(cls, result: EvaluationResult) -> "LedgerSummary":
        congested = []
        carried: dict[tuple[LinkId, int], float] = defaultdict(float)
        for item in result.realizations:
            if item.ledger is None:
                raise ValueError("a ledger summary needs ledgers; score the plan with keep_ledger=True")
            congested.append(congested_slots(item.ledger))
            for slot, link, _, bits in item.ledger.transfers:
                carried[(link, slot)] += bits
        return cls(len(result.realizations), tuple(congested), dict(carried))

    def congestion_count(self, link: LinkId, window: range) -> int:
        """Congested slots of the link inside the window, summed over the realizations."""
        return sum(len(slots.get(link, set()).intersection(window)) for slots in self.congested)

    def congested_realizations(self, link: LinkId, slot: int) -> int:
        return sum(slot in slots.get(link, ()) for slots in self.congested)

    def carried_bps(self, link: LinkId, window: range, slot_s: float) -> float:
        """Mean bits per second the link carried over the window and the realizations."""
        if len(window) == 0 or self.realization_count == 0:
            return 0.0
        total = sum(self.carried_bits.get((link, slot), 0.0) for slot in window)
        return total / (self.realization_count * len(window) * slot_s)


def residual_bps(scenario: Scenario, summary: LedgerSummary, link: LinkId, window: range) -> float:
    """Backhaul capacity left on the link during the window, floored at a thousandth of the capacity."""
    capacity = scenario.backhaul_capacity_bps(link[1], link[2])
    return max(capacity - summary.carried_bps(link, window, scenario.time.slot_s), capacity * RESIDUAL_FLOOR)
```

In `src/optimization/repair.py`: delete the local `congested_slots` and `CONGESTION_TOLERANCE`, add `from .ledger_summary import congested_slots  # noqa: F401 (re-exported for callers and tests)` — keep the name importable from `optimization.repair`.

- [ ] **Step 4: Run the tests**

Run: `uv run --extra test pytest tests/optimization -q`
Expected: PASS (including `test_repair.py::test_congested_links_carry_their_whole_capacity`).

- [ ] **Step 5: Commit**

```bash
git add src/optimization/ledger_summary.py src/optimization/repair.py tests/optimization/test_ledger_summary.py
git commit -m "feat: summarise congestion and residual capacity from design ledgers"
```

---

### Task 6: Path neighbourhood (`path_moves.py`)

**Files:**
- Modify: `src/models/paths.py` (`backhaul_routes` exclusions; factor `ground_path`/`uav_path` out of `candidate_paths`)
- Create: `src/optimization/path_moves.py`
- Test: `tests/optimization/test_path_moves.py` (new), `tests/models/test_paths.py` (routes with exclusions)

**Interfaces:**
- Consumes: `LedgerSummary`, `alert_window`, `residual_bps` (Task 5); `path_violations`, `radio_links_of_paths` (Task 1); `slot_midpoints` (`models/channel.py`).
- Produces (all in `optimization/path_moves.py` unless noted):
  - `models.paths.backhaul_routes(scenario, excluded_edges: frozenset[frozenset[str]] = frozenset(), excluded_nodes: frozenset[str] = frozenset()) -> BackhaulRoutes`
  - `models.paths.ground_path(scenario, alert, node_id, route) -> CandidatePath | None`, `models.paths.uav_path(scenario, alert, node_id, route) -> CandidatePath`
  - `is_ground_connected(scenario, source, routes) -> bool`
  - `kind_flip(scenario, alert, path) -> CandidatePath | None`
  - `estimated_completion_s(scenario, alert, path, trajectory, summary) -> float`
  - `entry_switches(scenario, alert, path, routes, trajectory, summary, rng, count) -> list[CandidatePath]`
  - `backhaul_detour(scenario, alert, path, summary) -> CandidatePath | None`
  - `path_neighbours(scenario, alert, current, candidates, routes, trajectory, summary, rng, entry_neighbors) -> list[CandidatePath | None]`
  - `urgency_order(alerts: Sequence[Alert], rng) -> list[Alert]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_paths.py`:

```python
def test_backhaul_routes_honour_exclusions(raw_scenario):
    from models.paths import backhaul_routes

    raw_scenario["network"]["edges"].append({"source": "n3", "target": "n2", "failed": False, "capacity_bps": 1000000.0})
    scenario = scenario_from_dict(raw_scenario)
    assert dict(backhaul_routes(scenario).next_hop) == {"n1": "n0", "n2": "n0", "n3": "n1"}
    detour = backhaul_routes(scenario, excluded_edges=frozenset({frozenset(("n1", "n0"))}))
    assert detour.route_links("n1") == (("backhaul", "n1", "n3"), ("backhaul", "n3", "n2"), ("backhaul", "n2", "n0"))
    blocked = backhaul_routes(scenario, excluded_edges=frozenset({frozenset(("n1", "n0"))}), excluded_nodes=frozenset({"n2"}))
    assert "n1" not in blocked.hop_count
```

Create `tests/optimization/test_path_moves.py`:

```python
import copy

import numpy as np
import pytest

from models.paths import GROUND, VIA_UAV, CandidatePath, backhaul_routes, candidate_paths, path_violations
from models.plan import stationary_trajectory
from models.scenario import UAV_ID, scenario_from_dict
from optimization.ledger_summary import LedgerSummary
from optimization.path_moves import (
    backhaul_detour,
    entry_switches,
    estimated_completion_s,
    is_ground_connected,
    kind_flip,
    path_neighbours,
    urgency_order,
)
from scenario_builders import BASE_SCENARIO, make_alert

ACCESS_N1 = ("access", "s00", "n1")
BACKHAUL_N1 = ("backhaul", "n1", "n0")
GROUND_N1 = CandidatePath(GROUND, "n1", (ACCESS_N1, BACKHAUL_N1), 2.3)
UAV_N1 = CandidatePath(VIA_UAV, "n1", (("access", "s00", UAV_ID), ("down", UAV_ID, "n1"), BACKHAUL_N1), 2.6)
EMPTY = LedgerSummary(1, ({},), {})


def _scenario(extra_edge=False, alerts=None):
    raw = copy.deepcopy(BASE_SCENARIO)
    if extra_edge:
        raw["network"]["edges"].append({"source": "n3", "target": "n2", "failed": False, "capacity_bps": 1000000.0})
    if alerts is not None:
        raw["alerts"] = alerts
    return scenario_from_dict(raw)


def _congested(link, slots):
    return LedgerSummary(1, ({link: set(slots)},), {})


def test_ground_connected_follows_the_b0_range_test():
    scenario = _scenario()
    routes = backhaul_routes(scenario)
    assert is_ground_connected(scenario, scenario.source_by_id["s00"], routes)
    assert not is_ground_connected(scenario, scenario.source_by_id["s01"], routes)


def test_kind_flip_keeps_entry_and_backhaul_and_respects_range():
    scenario = _scenario()
    alert = scenario.alerts[0]
    flipped = kind_flip(scenario, alert, GROUND_N1)
    assert flipped.kind == VIA_UAV and flipped.entry == "n1" and flipped.links[2:] == (BACKHAUL_N1,)
    assert kind_flip(scenario, alert, flipped).links == GROUND_N1.links
    far = CandidatePath(VIA_UAV, "n3", (("access", "s00", UAV_ID), ("down", UAV_ID, "n3"), ("backhaul", "n3", "n1"), BACKHAUL_N1), 9.0)
    assert kind_flip(scenario, alert, far) is None
    direct = CandidatePath(GROUND, "n0", (("access", "s00", "n0"),), 1.0)
    assert kind_flip(scenario, alert, direct).links == (("access", "s00", UAV_ID), ("down", UAV_ID, "n0"))


def test_estimated_completion_grows_with_load_and_distance():
    scenario = _scenario()
    alert = scenario.alerts[0]
    trajectory = stationary_trajectory(scenario)
    idle = estimated_completion_s(scenario, alert, GROUND_N1, trajectory, EMPTY)
    capacity = scenario.backhaul_capacity_bps("n1", "n0")
    busy = LedgerSummary(1, ({},), {(BACKHAUL_N1, slot): 0.9 * capacity for slot in range(20)})
    assert estimated_completion_s(scenario, alert, GROUND_N1, trajectory, busy) > idle
    near = estimated_completion_s(scenario, alert, UAV_N1, trajectory, EMPTY)
    far_trajectory = np.tile([100.0, 100.0], (21, 1))
    far_trajectory[0] = far_trajectory[-1] = scenario.uav.start_xy_m
    assert estimated_completion_s(scenario, alert, UAV_N1, far_trajectory, EMPTY) > near


def test_entry_switches_draw_distinct_entries_favouring_lower_times():
    scenario = _scenario()
    alert = scenario.alerts[0]
    routes = backhaul_routes(scenario)
    trajectory = stationary_trajectory(scenario)
    rng = np.random.default_rng(0)
    drawn = entry_switches(scenario, alert, UAV_N1, routes, trajectory, EMPTY, rng, 2)
    assert len(drawn) == 2 and len({path.entry for path in drawn}) == 2 and all(path.entry != "n1" for path in drawn)
    assert all(path.kind == VIA_UAV and path_violations(scenario, "s00", path) == [] for path in drawn)
    ground = entry_switches(scenario, alert, GROUND_N1, routes, trajectory, EMPTY, rng, 5)
    assert [path.entry for path in ground] == ["n0"]
    firsts = [entry_switches(scenario, alert, UAV_N1, routes, trajectory, EMPTY, np.random.default_rng(seed), 1)[0].entry for seed in range(200)]
    assert firsts.count("n0") > firsts.count("n3")


def test_detour_avoids_the_congested_link_and_prefix_nodes():
    scenario = _scenario(extra_edge=True)
    alert = scenario.alerts[0]
    assert backhaul_detour(scenario, alert, GROUND_N1, EMPTY) is None
    detour = backhaul_detour(scenario, alert, GROUND_N1, _congested(BACKHAUL_N1, range(0, 10)))
    assert detour.links == (ACCESS_N1, ("backhaul", "n1", "n3"), ("backhaul", "n3", "n2"), ("backhaul", "n2", "n0"))
    assert detour.kind == GROUND and detour.entry == "n1" and path_violations(scenario, "s00", detour) == []
    long_path = CandidatePath(GROUND, "n1", (ACCESS_N1, ("backhaul", "n1", "n3"), ("backhaul", "n3", "n2"), ("backhaul", "n2", "n0")), 5.0)
    assert backhaul_detour(scenario, alert, long_path, _congested(("backhaul", "n2", "n0"), range(0, 10))) is None
    assert backhaul_detour(scenario, _scenario().alerts[0], GROUND_N1, _congested(BACKHAUL_N1, range(0, 10))) is None
    direct = CandidatePath(GROUND, "n0", (("access", "s00", "n0"),), 1.0)
    assert backhaul_detour(scenario, alert, direct, _congested(BACKHAUL_N1, range(0, 10))) is None


def test_neighbours_are_valid_distinct_and_only_uav_for_unconnected_sources():
    scenario = _scenario(alerts=[make_alert("alert-0000", "s00", 0, 10, 1000000), make_alert("alert-0001", "s01", 0, 20, 1000000)])
    candidates = candidate_paths(scenario)
    routes = backhaul_routes(scenario)
    trajectory = stationary_trajectory(scenario)
    rng = np.random.default_rng(1)
    for alert in scenario.alerts:
        current = candidates[alert.id][0]
        neighbours = path_neighbours(scenario, alert, current, candidates, routes, trajectory, EMPTY, rng, 3)
        assert neighbours[-1] is None and current not in neighbours
        paths = [path for path in neighbours if path is not None]
        assert len(paths) == len(set(paths))
        assert all(path_violations(scenario, alert.source, path) == [] for path in paths)
        if alert.source == "s01":
            assert all(path.kind == VIA_UAV for path in paths)
    assert path_neighbours(scenario, scenario.alerts[0], None, candidates, routes, trajectory, EMPTY, rng, 3) == list(candidates["alert-0000"])


def test_urgency_order_is_a_permutation_biased_to_short_windows():
    scenario = _scenario(alerts=[make_alert("alert-0000", "s00", 0, 2, 1), make_alert("alert-0001", "s00", 0, 20, 1), make_alert("alert-0002", "s00", 0, 20, 1)])
    alerts = list(scenario.alerts)
    firsts = []
    for seed in range(300):
        order = urgency_order(alerts, np.random.default_rng(seed))
        assert sorted(alert.id for alert in order) == [alert.id for alert in alerts]
        firsts.append(order[0].id)
    assert firsts.count("alert-0000") > firsts.count("alert-0001") + firsts.count("alert-0002")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra test pytest tests/optimization/test_path_moves.py tests/models/test_paths.py -q`
Expected: `ImportError`/`TypeError` (no `path_moves`, `backhaul_routes` takes no exclusions).

- [ ] **Step 3: Generalise `backhaul_routes` and factor the path constructors in `src/models/paths.py`**

```python
def backhaul_routes(
    scenario: Scenario,
    excluded_edges: frozenset[frozenset[str]] = frozenset(),
    excluded_nodes: frozenset[str] = frozenset(),
) -> BackhaulRoutes:
    """Fewest-hop routes to the center over alive edges, ties by Euclidean length then node id; exclusions prune the graph."""
    neighbours: dict[str, set[str]] = {node.id: set() for node in scenario.nodes}
    for edge in scenario.edges:
        ends = frozenset((edge.source, edge.target))
        if edge.alive and ends not in excluded_edges and not ends & excluded_nodes:
            neighbours[edge.source].add(edge.target)
            neighbours[edge.target].add(edge.source)
    ...  # BFS and next_hop exactly as today
```

Then split `candidate_paths`:

```python
def ground_path(scenario: Scenario, alert: Alert, node_id: str, route: tuple[LinkId, ...]) -> CandidatePath | None:
    """Ground path entering at `node_id` and following `route`; None when the source is out of range of the node."""
    source = scenario.source_by_id[alert.source]
    distance = math.dist(source.xy, scenario.node_by_id[node_id].xy)
    if not in_ground_range(scenario, distance):
        return None
    access_rate = rate_bps(scenario.spectrum.b_tot_hz, ground_snr_per_hz(scenario, distance))
    cost_s = alert.size_bits / access_rate + _backhaul_delay_s(scenario, route, alert.size_bits)
    return CandidatePath(GROUND, node_id, ((ACCESS, source.id, node_id), *route), cost_s)


def uav_path(scenario: Scenario, alert: Alert, node_id: str, route: tuple[LinkId, ...]) -> CandidatePath:
    """Path via the UAV that downlinks at `node_id` and follows `route`; cost uses the flight-distance proxy."""
    source = scenario.source_by_id[alert.source]
    distance = math.dist(source.xy, scenario.node_by_id[node_id].xy)
    cost_s = distance / scenario.uav.v_max_mps + _backhaul_delay_s(scenario, route, alert.size_bits)
    return CandidatePath(VIA_UAV, node_id, ((ACCESS, source.id, UAV_ID), (DOWNLINK, UAV_ID, node_id), *route), cost_s)


def candidate_paths(scenario, routes=None):
    routes = routes or backhaul_routes(scenario)
    entries = sorted(routes.hop_count)
    candidates = {}
    for alert in scenario.alerts:
        ground = [path for path in (ground_path(scenario, alert, node_id, routes.route_links(node_id)) for node_id in entries) if path is not None]
        aerial = [uav_path(scenario, alert, node_id, routes.route_links(node_id)) for node_id in entries]
        ground.sort(key=lambda path: (path.cost_s, path.entry))
        aerial.sort(key=lambda path: (path.cost_s, path.entry))
        chosen = ground[: scenario.paths.max_ground]
        candidates[alert.id] = tuple(chosen + aerial[: scenario.paths.k - len(chosen)])
    return candidates
```

Import `Alert` from `.scenario`. `candidate_paths` must produce byte-identical `cost_s` values (same arithmetic order) — the existing `tests/models/test_paths.py` and the Task 3 gate protect this; rerun `uv run --extra test pytest tests/models -q` after this step.

- [ ] **Step 4: Create `src/optimization/path_moves.py`**

```python
"""Neighbours of one alert's path, generated from the incumbent path and the incumbent ledger (spec E Section 5)."""

import math
from typing import Mapping, Sequence

import numpy as np

from models.channel import ACCESS, BACKHAUL, DOWNLINK, ground_snr_per_hz, in_ground_range, rate_bps, slot_midpoints
from models.paths import GROUND, VIA_UAV, BackhaulRoutes, CandidatePath, backhaul_routes, ground_path, uav_path
from models.scenario import UAV_ID, Alert, Scenario, SourcePoint

from .ledger_summary import LedgerSummary, alert_window, residual_bps

MIN_TIME_S = 1e-3


def is_ground_connected(scenario: Scenario, source: SourcePoint, routes: BackhaulRoutes) -> bool:
    """The B0 test: some reachable node lies within ground range of the source."""
    return any(in_ground_range(scenario, math.dist(source.xy, scenario.node_by_id[node_id].xy)) for node_id in routes.hop_count)


def backhaul_links(path: CandidatePath) -> tuple[tuple[str, str, str], ...]:
    return tuple(link for link in path.links if link[0] == BACKHAUL)


def kind_flip(scenario: Scenario, alert: Alert, path: CandidatePath) -> CandidatePath | None:
    """Same entry and backhaul, the other kind; None when a ground entry would be out of range."""
    route = backhaul_links(path)
    if path.kind == GROUND:
        return uav_path(scenario, alert, path.entry, route)
    return ground_path(scenario, alert, path.entry, route)


def estimated_completion_s(
    scenario: Scenario, alert: Alert, path: CandidatePath, trajectory: np.ndarray, summary: LedgerSummary
) -> float:
    """Access time plus backhaul time at residual capacity; the UAV access time follows the incumbent trajectory."""
    source = scenario.source_by_id[alert.source]
    node = scenario.node_by_id[path.entry]
    window = alert_window(scenario, alert)
    if path.kind == GROUND:
        distance = math.dist(source.xy, node.xy)
        access_s = alert.size_bits / rate_bps(scenario.spectrum.b_tot_hz, ground_snr_per_hz(scenario, distance))
    else:
        midpoints = slot_midpoints(trajectory)[window.start : window.stop]
        if len(midpoints):
            distance = float(np.mean(np.linalg.norm(midpoints - np.asarray(node.xy), axis=1)))
        else:
            distance = math.dist(source.xy, node.xy)
        access_s = distance / scenario.uav.v_max_mps
    backhaul_s = sum(
        alert.size_bits / residual_bps(scenario, summary, link, window) + scenario.time.slot_s for link in backhaul_links(path)
    )
    return access_s + backhaul_s


def entry_switches(
    scenario: Scenario,
    alert: Alert,
    path: CandidatePath,
    routes: BackhaulRoutes,
    trajectory: np.ndarray,
    summary: LedgerSummary,
    rng: np.random.Generator,
    count: int,
) -> list[CandidatePath]:
    """Up to `count` paths of the same kind through other entries, drawn with probability ∝ 1/T, in increasing T."""
    options = []
    for node_id in sorted(routes.hop_count):
        if node_id == path.entry:
            continue
        route = routes.route_links(node_id)
        option = ground_path(scenario, alert, node_id, route) if path.kind == GROUND else uav_path(scenario, alert, node_id, route)
        if option is not None:
            options.append(option)
    if not options:
        return []
    times = np.array([estimated_completion_s(scenario, alert, option, trajectory, summary) for option in options])
    weights = 1.0 / np.maximum(times, MIN_TIME_S)
    drawn = rng.choice(len(options), size=min(count, len(options)), replace=False, p=weights / weights.sum())
    return [options[index] for index in sorted(drawn, key=lambda index: (times[index], options[index].entry))]


def backhaul_detour(scenario: Scenario, alert: Alert, path: CandidatePath, summary: LedgerSummary) -> CandidatePath | None:
    """Reroute from the head of the most congested backhaul link around that link; None without congestion or route."""
    window = alert_window(scenario, alert)
    links = backhaul_links(path)
    if not links:
        return None
    counts = [summary.congestion_count(link, window) for link in links]
    worst = max(range(len(links)), key=lambda index: (counts[index], -index))
    if counts[worst] == 0:
        return None
    _, head, tail = links[worst]
    prefix_nodes = frozenset(link[1] for link in links[:worst])
    routes = backhaul_routes(scenario, excluded_edges=frozenset({frozenset((head, tail))}), excluded_nodes=prefix_nodes)
    if head not in routes.hop_count:
        return None
    first_backhaul = path.links.index(links[0])
    new_links = (*path.links[: first_backhaul + worst], *routes.route_links(head))
    if new_links == path.links:
        return None
    constructor = ground_path if path.kind == GROUND else uav_path
    rebuilt = constructor(scenario, alert, path.entry, tuple(link for link in new_links if link[0] == BACKHAUL))
    return rebuilt


def path_neighbours(
    scenario: Scenario,
    alert: Alert,
    current: CandidatePath | None,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    routes: BackhaulRoutes,
    trajectory: np.ndarray,
    summary: LedgerSummary,
    rng: np.random.Generator,
    entry_neighbors: int,
) -> list[CandidatePath | None]:
    """Kind flip, entry switches, backhaul detour, unserve — distinct, valid, different from `current`."""
    if current is None:
        return list(candidates[alert.id])
    source = scenario.source_by_id[alert.source]
    ground_ok = is_ground_connected(scenario, source, routes)
    moves: list[CandidatePath | None] = []
    flipped = kind_flip(scenario, alert, current)
    if flipped is not None and (flipped.kind == VIA_UAV or ground_ok):
        moves.append(flipped)
    if current.kind == VIA_UAV or ground_ok:
        moves.extend(entry_switches(scenario, alert, current, routes, trajectory, summary, rng, entry_neighbors))
    detour = backhaul_detour(scenario, alert, current, summary)
    if detour is not None:
        moves.append(detour)
    moves.append(None)
    distinct: list[CandidatePath | None] = []
    for move in moves:
        if move != current and move not in distinct:
            distinct.append(move)
    return distinct


def urgency_order(alerts: Sequence[Alert], rng: np.random.Generator) -> list[Alert]:
    """Random order drawn without replacement with probability ∝ 1/(deadline − release): urgent alerts tend to come first."""
    if not alerts:
        return []
    weights = np.array([1.0 / max(alert.deadline_slot - alert.release_slot, 1) for alert in alerts])
    order = rng.choice(len(alerts), size=len(alerts), replace=False, p=weights / weights.sum())
    return [alerts[index] for index in order]
```

Check `SourcePoint` is the source dataclass name in `models/scenario.py` (`source_by_id -> dict[str, SourcePoint]`); import whatever name it has. The unused imports `ACCESS`, `DOWNLINK`, `UAV_ID` may be dropped if the linter complains.

Note on `backhaul_detour`: the prefix kept is `path.links[: first_backhaul + worst]` — the radio links plus the backhaul links before the congested one — and the detour route from `head` is appended; rebuilding through `ground_path`/`uav_path` recomputes `cost_s` with the nominal capacities (spec 5.1).

- [ ] **Step 5: Run the tests**

Run: `uv run --extra test pytest tests/optimization/test_path_moves.py tests/models -q`
Expected: PASS. If `test_entry_switches...` fails on the `n0 > n3` count, check that `times` for `n0` (distance 230 m, zero backhaul) is smaller than for `n3` (470 m plus two backhaul hops) — it must be, so the failure would be in the sampling code.

- [ ] **Step 6: Commit**

```bash
git add src/models/paths.py src/optimization/path_moves.py tests/optimization/test_path_moves.py tests/models/test_paths.py
git commit -m "feat: generate path neighbours from the incumbent path and ledger"
```

---

### Task 7: Trajectory moves (`trajectory_moves.py`)

**Files:**
- Create: `src/optimization/trajectory_moves.py`
- Test: `tests/optimization/test_trajectory_moves.py` (new)

**Interfaces:**
- Consumes: `LedgerSummary`, `alert_window` (Task 5); `radio_links_of_paths` (Task 1); `Plan`.
- Produces: `max_shift(anchor, point, direction, step_m) -> float`; `shift_segment(trajectory, start, stop, direction, fraction, step_m) -> np.ndarray | None`; `feasible_segment(trajectory, start, stop, direction, step_m, max_widening) -> tuple[int, int] | None`; `Segment(start: int, stop: int, direction: np.ndarray, priority: float)`; `sample_uav_segments(scenario, plan, summary, rng, count) -> list[Segment]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/optimization/test_trajectory_moves.py`:

```python
import numpy as np
import pytest

from baselines.trajectories import zone_tour_trajectory
from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, _trajectory_violations
from models.scenario import UAV_ID, scenario_from_dict
from optimization.ledger_summary import LedgerSummary
from optimization.trajectory_moves import Segment, feasible_segment, max_shift, sample_uav_segments, shift_segment
from scenario_builders import make_alert
from search_builders import three_zone_raw

STEP = 50.0
EAST = np.array([1.0, 0.0])


def _hover(num_slots=20, point=(1000.0, 1000.0)):
    return np.tile(np.asarray(point, dtype=float), (num_slots + 1, 1))


def test_max_shift_solves_the_boundary_circle():
    anchor = np.array([0.0, 0.0])
    assert max_shift(anchor, np.array([0.0, 0.0]), EAST, STEP) == pytest.approx(STEP)
    assert max_shift(anchor, np.array([30.0, 0.0]), EAST, STEP) == pytest.approx(20.0)
    assert max_shift(anchor, np.array([0.0, 30.0]), EAST, STEP) == pytest.approx(40.0)
    assert max_shift(anchor, np.array([50.0, 0.0]), EAST, STEP) == pytest.approx(0.0)
    assert max_shift(anchor, np.array([60.0, 0.0]), EAST, STEP) == 0.0


def test_shift_segment_keeps_feasibility_and_endpoints():
    scenario = scenario_from_dict(three_zone_raw(num_slots=20))
    trajectory = _hover()
    shifted = shift_segment(trajectory, 1, 19, EAST, 1.0, STEP)
    assert np.allclose(shifted[1:20], [1050.0, 1000.0]) and np.array_equal(shifted[0], trajectory[0]) and np.array_equal(shifted[20], trajectory[20])
    assert _trajectory_violations(scenario, shifted) == []
    half = shift_segment(trajectory, 5, 7, EAST, 0.5, STEP)
    assert np.allclose(half[5:8], [1025.0, 1000.0]) and np.array_equal(half[4], trajectory[4]) and np.array_equal(half[8], trajectory[8])
    rng = np.random.default_rng(3)
    for _ in range(50):
        start = int(rng.integers(1, 19))
        stop = int(rng.integers(start, 20))
        direction = rng.normal(size=2)
        out = shift_segment(trajectory, start, stop, direction, float(rng.uniform(0.1, 1.0)), STEP)
        assert out is not None and _trajectory_violations(scenario, out) == []


def test_shift_segment_refuses_a_cruise_boundary_and_widening_finds_room():
    scenario = scenario_from_dict(three_zone_raw(num_slots=20))
    cruise = np.array(
        [[1000.0, 1000.0]] * 3                                              # q[0..2] hover at the start
        + [[1000.0 + STEP * index, 1000.0] for index in range(1, 6)]        # q[3..7] cruise east at V_max
        + [[1250.0, 1000.0]] * 8                                            # q[8..15] hover
        + [[1250.0 - STEP * index, 1000.0] for index in range(1, 6)]        # q[16..20] cruise back
    )
    assert cruise.shape == (21, 2) and _trajectory_violations(scenario, cruise) == []
    north = np.array([0.0, 1.0])
    assert shift_segment(cruise, 4, 5, north, 1.0, STEP) is None
    assert feasible_segment(cruise, 4, 5, north, STEP, 0) is None
    assert feasible_segment(cruise, 4, 5, north, STEP, 10) == (2, 7)
    shifted = shift_segment(cruise, 2, 7, north, 1.0, STEP)
    assert shifted is not None and _trajectory_violations(scenario, shifted) == []
    with pytest.raises(ValueError):
        shift_segment(cruise, 0, 3, north, 1.0, STEP)
    with pytest.raises(ValueError):
        shift_segment(cruise, 3, 3, np.zeros(2), 1.0, STEP)


def _uav_plan(scenario):
    candidates = candidate_paths(scenario)
    uav = {alert.id: next(path for path in candidates[alert.id] if path.kind == "uav") for alert in scenario.alerts}
    return Plan(zone_tour_trajectory(scenario), uav, EqualSplitBacklogged())


def test_sample_uav_segments_follows_priority_and_handles_no_congestion():
    raw = three_zone_raw(num_slots=20)
    raw["alerts"] = [make_alert("alert-0000", "s00", 0, 10, 1000000), make_alert("alert-0001", "s01", 2, 20, 1000000)]
    scenario = scenario_from_dict(raw)
    plan = _uav_plan(scenario)
    uplink_a, uplink_b = ("access", "s00", UAV_ID), ("access", "s01", UAV_ID)
    assert sample_uav_segments(scenario, plan, LedgerSummary(1, ({},), {}), np.random.default_rng(0), 5) == []
    summary = LedgerSummary(2, ({uplink_a: {3, 4, 5, 9}, uplink_b: {12}}, {uplink_a: {4, 5}}), {})
    segments = sample_uav_segments(scenario, plan, summary, np.random.default_rng(0), 5)
    assert [(segment.start, segment.stop) for segment in segments] == [(3, 6), (9, 10), (12, 13)]
    assert segments[0].priority > segments[1].priority > segments[2].priority > 0.0
    source = np.asarray(scenario.source_by_id["s00"].xy)
    centroid = plan.trajectory[3:7].mean(axis=0)
    assert np.allclose(segments[0].direction, (source - centroid) / np.linalg.norm(source - centroid))
    assert len(sample_uav_segments(scenario, plan, summary, np.random.default_rng(0), 2)) == 2
    firsts = [sample_uav_segments(scenario, plan, summary, np.random.default_rng(seed), 1)[0].start for seed in range(100)]
    assert firsts.count(3) > firsts.count(12)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra test pytest tests/optimization/test_trajectory_moves.py -q`
Expected: `ImportError` on `optimization.trajectory_moves`.

- [ ] **Step 3: Create `src/optimization/trajectory_moves.py`**

```python
"""Segment shifts of the incumbent trajectory toward congested UAV links (spec E Section 6)."""

from dataclasses import dataclass
import math

import numpy as np

from models.channel import ACCESS
from models.paths import radio_links_of_paths
from models.plan import Plan
from models.scenario import UAV_ID, Scenario

from .ledger_summary import LedgerSummary, alert_window

MIN_SHIFT_M = 1e-6
STEP_MARGIN = 1.0 - 1e-9


def max_shift(anchor: np.ndarray, point: np.ndarray, direction: np.ndarray, step_m: float) -> float:
    """Largest λ ≥ 0 with ‖point + λ·direction − anchor‖ ≤ step_m for a unit direction; 0 when none exists."""
    offset = np.asarray(point, dtype=float) - np.asarray(anchor, dtype=float)
    b = float(np.dot(offset, direction))
    c = float(np.dot(offset, offset)) - step_m * step_m
    discriminant = b * b - c
    if discriminant < 0.0:
        return 0.0
    return max(-b + math.sqrt(discriminant), 0.0)


def shift_segment(
    trajectory: np.ndarray, start: int, stop: int, direction: np.ndarray, fraction: float, step_m: float
) -> np.ndarray | None:
    """q[start..stop] moved by fraction·λ* along `direction`; None when λ* is (numerically) zero."""
    last = len(trajectory) - 1
    if not 1 <= start <= stop <= last - 1:
        raise ValueError(f"segment [{start}, {stop}] must lie inside [1, {last - 1}]")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("direction must be non-zero")
    unit = np.asarray(direction, dtype=float) / norm
    limit = step_m * STEP_MARGIN
    reach = min(max_shift(trajectory[start - 1], trajectory[start], unit, limit), max_shift(trajectory[stop + 1], trajectory[stop], unit, limit))
    if reach < MIN_SHIFT_M:
        return None
    shifted = np.array(trajectory, dtype=float, copy=True)
    shifted[start : stop + 1] += fraction * reach * unit
    return shifted


def feasible_segment(
    trajectory: np.ndarray, start: int, stop: int, direction: np.ndarray, step_m: float, max_widening: int
) -> tuple[int, int] | None:
    """Widen [start, stop] by one point per side, at most `max_widening` times, until a shift is possible."""
    last = len(trajectory) - 1
    for _ in range(max_widening + 1):
        if shift_segment(trajectory, start, stop, direction, 1.0, step_m) is not None:
            return start, stop
        widened = (max(start - 1, 1), min(stop + 1, last - 1))
        if widened == (start, stop):
            return None
        start, stop = widened
    return None


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int
    direction: np.ndarray
    priority: float


def sample_uav_segments(
    scenario: Scenario, plan: Plan, summary: LedgerSummary, rng: np.random.Generator, count: int
) -> list[Segment]:
    """Runs of congested slots on UAV links, drawn with probability ∝ priority, returned in decreasing priority."""
    num_slots = scenario.time.num_slots
    last = num_slots
    candidates: list[Segment] = []
    for link in radio_links_of_paths(plan.paths):
        if UAV_ID not in (link[1], link[2]):
            continue
        urgency = np.zeros(num_slots)
        for alert in scenario.alerts:
            path = plan.paths[alert.id]
            if path is not None and link in path.links:
                window = alert_window(scenario, alert)
                urgency[window.start : window.stop] += alert.size_bits / (alert.deadline_slot - alert.release_slot)
        congested = np.array([summary.congested_realizations(link, slot) for slot in range(num_slots)], dtype=float)
        priority = congested * urgency
        ground_id = link[1] if link[0] == ACCESS else link[2]
        ground = np.asarray(scenario.position(ground_id), dtype=float)
        slot = 0
        while slot < num_slots:
            if priority[slot] <= 0.0:
                slot += 1
                continue
            run_start = slot
            while slot < num_slots and priority[slot] > 0.0:
                slot += 1
            start, stop = max(run_start, 1), min(slot, last - 1)
            if start > stop:
                continue
            centroid = plan.trajectory[start : stop + 1].mean(axis=0)
            direction = ground - centroid
            norm = float(np.linalg.norm(direction))
            if norm > 0.0:
                candidates.append(Segment(start, stop, direction / norm, float(priority[run_start:slot].sum())))
    if not candidates:
        return []
    weights = np.array([segment.priority for segment in candidates])
    drawn = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False, p=weights / weights.sum())
    return sorted((candidates[index] for index in drawn), key=lambda segment: (-segment.priority, segment.start))
```

Mapping check against the test: congested slots `{3,4,5}` → run `[3, 6)` → `start = 3`, `stop = min(6, 19) = 6` ✓; slot `9` → `(9, 10)` ✓; slot `12` → `(12, 13)` ✓. `_trajectory_violations` is private in `plan.py`; the test imports it deliberately (same package test style as `tests/models/test_plan.py` uses its public wrapper) — if the linter objects, expose `trajectory_violations` as a public alias in `plan.py` and import that instead.

- [ ] **Step 4: Run the tests**

Run: `uv run --extra test pytest tests/optimization/test_trajectory_moves.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/optimization/trajectory_moves.py tests/optimization/test_trajectory_moves.py
git commit -m "feat: shift trajectory segments toward congested UAV links"
```

---

### Task 8: Wire the local-search blocks into the BCD

**Files:**
- Modify: `src/optimization/blocks.py` (`SearchState.try_change` unit weights for new links; new `path_block`; `catalog_trajectory_block`; new `trajectory_block`)
- Modify: `src/optimization/repair.py` (Operation 2 via `path_neighbours`)
- Modify: `src/optimization/bcd.py` (`search`: RNG, routes, mode dispatch)
- Test: `tests/optimization/test_blocks.py`, `tests/optimization/test_repair.py`, `tests/optimization/test_search_methods.py`

**Interfaces:**
- Consumes: Tasks 4–7.
- Produces: `path_block(scenario, candidates, routes, scorer, state, params, rng) -> bool`; `catalog_trajectory_block(scorer, state, family) -> bool` (today's `trajectory_block`); `trajectory_block(scenario, scorer, state, params, rng, family) -> bool`; `repair_round(scenario, candidates, routes, scorer, state, params, rng) -> bool`; `SearchMethod.search` unchanged signature.

- [ ] **Step 1: Update the tests**

In `tests/optimization/test_blocks.py`:

```python
import numpy as np

from models.paths import backhaul_routes, candidate_paths, uav_path
from models.plan import stationary_trajectory, validate_plan
from models.scenario import scenario_from_dict
from optimization.bcd import SearchParams
from optimization.blocks import (
    SearchState,
    bandwidth_block,
    catalog_trajectory_block,
    links_by_load,
    path_block,
    trajectory_block,
    unit_weights,
    with_unit_weights,
)
from optimization.scoring import DesignScorer
from optimization.tours import tour_family
from scenario_builders import make_alert
from search_builders import bandwidth_raw, hopeless_raw, three_zone_raw

FIRST_ACCESS = ("access", "s00", "n1")
SECOND_ACCESS = ("access", "s02", "n2")
PARAMS = SearchParams(design_ids=(), budget=500)


def _start(raw, choice, trajectory=None):
    scenario = scenario_from_dict(raw)
    candidates = candidate_paths(scenario)
    scorer = DesignScorer(scenario, design_ids=(), budget=500)
    paths = {alert_id: None if index is None else candidates[alert_id][index] for alert_id, index in choice.items()}
    state = SearchState(stationary_trajectory(scenario) if trajectory is None else trajectory, paths, unit_weights(scenario, candidates))
    state.key = scorer.score(state.plan())
    return scenario, candidates, scorer, state


def test_try_change_adopts_only_strict_improvements():
    scenario, candidates, scorer, state = _start(hopeless_raw(), {"alert-0000": 0, "alert-0001": 0})
    before = state.key
    assert not state.try_change(scorer, paths=dict(state.paths))
    assert state.key == before and scorer.calls == 2
    assert state.try_change(scorer, paths={"alert-0000": None, "alert-0001": candidates["alert-0001"][0]})
    assert state.paths == {"alert-0000": None, "alert-0001": candidates["alert-0001"][0]} and state.key > before


def test_with_unit_weights_adds_missing_radio_links_only():
    scenario, candidates, _, state = _start(hopeless_raw(), {"alert-0000": 0, "alert-0001": 0})
    new_path = uav_path(scenario, scenario.alert_by_id["alert-0001"], "n1", (("backhaul", "n1", "n0"),))
    completed = with_unit_weights(state.weights, {"alert-0000": None, "alert-0001": new_path}, 20)
    uplink, downlink = ("access", "s00", "UAV"), ("down", "UAV", "n1")
    assert set(completed) == set(state.weights) | {uplink, downlink}
    assert np.all(completed[uplink] == 1.0) and completed[uplink].shape == (20,)
    assert completed[FIRST_ACCESS] is state.weights[FIRST_ACCESS]
    assert with_unit_weights(state.weights, dict(state.paths), 20) == state.weights


def test_path_block_unblocks_the_alert_held_by_a_hopeless_one():
    scenario, candidates, scorer, state = _start(hopeless_raw(), {"alert-0000": 0, "alert-0001": 0})
    assert state.key[0] == 0
    rng = np.random.default_rng(0)
    assert path_block(scenario, candidates, backhaul_routes(scenario), scorer, state, PARAMS, rng)
    assert state.key[0] == 1
    assert validate_plan(scenario, state.plan()) == []


def test_path_block_only_scores_through_the_scorer_and_stays_valid():
    raw = three_zone_raw(num_slots=50)
    scenario, candidates, scorer, state = _start(raw, {"alert-0000": 0, "alert-0001": 0, "alert-0002": 0})
    rng = np.random.default_rng(1)
    before = scorer.calls
    path_block(scenario, candidates, backhaul_routes(scenario), scorer, state, PARAMS, rng)
    assert scorer.calls > before and validate_plan(scenario, state.plan()) == []
    assert all(link in state.weights for path in state.paths.values() if path is not None for link in path.links if link[0] != "backhaul")


def test_catalog_block_flies_to_a_source_only_the_uav_reaches():
    raw = three_zone_raw(num_slots=150)
    raw["alerts"] = [make_alert("alert-0000", "s01", 0, 150, 1000000)]
    scenario, candidates, scorer, state = _start(raw, {"alert-0000": 0})
    assert [path.kind for path in candidates["alert-0000"]] == ["uav", "uav", "uav"]
    assert state.key[0] == 0
    assert catalog_trajectory_block(scorer, state, tour_family(scenario))
    assert state.key[0] == 1
    assert np.min(np.linalg.norm(state.trajectory - [100.0, 100.0], axis=1)) < 1e-6
    assert validate_plan(scenario, state.plan()) == []


def test_local_trajectory_block_moves_only_by_feasible_shifts_within_its_budget():
    raw = three_zone_raw(num_slots=150)
    raw["alerts"] = [make_alert("alert-0000", "s01", 0, 150, 1000000), make_alert("alert-0001", "s00", 0, 150, 4000000)]
    scenario, candidates, scorer, state = _start(raw, {"alert-0000": 0, "alert-0001": 2})
    params = SearchParams(design_ids=(), budget=500, catalog_probability=0.0)
    rng = np.random.default_rng(0)
    before_calls, before_key = scorer.calls, state.key
    trajectory_block(scenario, scorer, state, params, rng, tour_family(scenario))
    assert scorer.calls - before_calls <= 1 + 2 * params.trajectory_segments
    assert state.key >= before_key and validate_plan(scenario, state.plan()) == []
    assert np.array_equal(state.trajectory[0], stationary_trajectory(scenario)[0])
    always_catalog = SearchParams(design_ids=(), budget=500, catalog_probability=1.0)
    calls = scorer.calls
    trajectory_block(scenario, scorer, state, always_catalog, np.random.default_rng(0), tour_family(scenario))
    assert scorer.calls - calls <= 2 + 2 * always_catalog.trajectory_segments


def test_bandwidth_block_raises_the_weight_of_the_congested_link():
    scenario, candidates, scorer, state = _start(bandwidth_raw(), {"alert-0000": 0, "alert-0001": 0})
    assert links_by_load(scenario, state.paths) == [FIRST_ACCESS, SECOND_ACCESS]
    assert state.key[0] == 1
    assert bandwidth_block(scenario, scorer, state)
    assert state.key[0] == 2
    assert np.all(state.weights[FIRST_ACCESS] == 2.0) and np.all(state.weights[SECOND_ACCESS] == 1.0)


def test_links_by_load_breaks_ties_by_link_and_skips_unserved_alerts():
    raw = bandwidth_raw()
    raw["alerts"][0]["size_bits"] = 4000000
    scenario = scenario_from_dict(raw)
    candidates = candidate_paths(scenario)
    paths = {"alert-0000": candidates["alert-0000"][0], "alert-0001": candidates["alert-0001"][0]}
    assert links_by_load(scenario, paths) == [FIRST_ACCESS, SECOND_ACCESS]
    assert links_by_load(scenario, {**paths, "alert-0000": None}) == [SECOND_ACCESS]
```

In `tests/optimization/test_repair.py`: replace `test_operation2_orders_candidates_by_congestion_then_cost` with a test of the new ordering helper and keep the rest:

```python
def test_operation2_orders_neighbours_by_congestion_then_cost():
    from optimization.repair import repair_order

    downlink = ("down", "UAV", "n1")
    paths = [
        CandidatePath("ground", "n1", (ACCESS_LINK, BACKHAUL_LINK), 0.1),
        CandidatePath("ground", "n2", (("access", "s00", "n2"), ("backhaul", "n2", "n0")), 5.0),
        CandidatePath("uav", "n1", (("access", "s00", "UAV"), downlink, BACKHAUL_LINK), 0.5),
        None,
    ]
    risk = AlertRisk("alert-0000", 1.0, ACCESS_LINK, frozenset({ACCESS_LINK, BACKHAUL_LINK}))
    assert repair_order(paths, risk, backlog=True) == [paths[1], paths[2], paths[0]]
    assert repair_order(paths, risk, backlog=False) == [paths[0], paths[2], paths[1]]
```

Update the `repair_round` tests in that file (if any call it) to the new signature `repair_round(scenario, candidates, routes, scorer, state, params, rng)` with `params = SearchParams(operation1=True, operation2=True, design_ids=(), budget=...)` and `rng = np.random.default_rng(0)`.

In `tests/optimization/test_search_methods.py`: add to `VARIANTS`:

```python
    "P_catalog_trajectory": SearchParams(operation1=True, operation2=True, trajectory_mode="catalog", **SMALL),
    "P_seed1": SearchParams(operation1=True, operation2=True, seed=1, **SMALL),
```

and add:

```python
def test_catalog_mode_only_visits_catalogue_tours():
    scenario = _scenario()
    candidates = candidate_paths(scenario)
    catalog = SearchMethod("P", replace(VARIANTS["P"], trajectory_mode="catalog", path_block=False, operation1=False, operation2=False, bandwidth_block=False))
    outcome = catalog.search(scenario, candidates)
    family = {trajectory.tobytes() for _, trajectory in tour_family(scenario)} | {B1().plan(scenario, candidates, EvaluationCounter()).trajectory.tobytes()}
    assert outcome.plan.trajectory.tobytes() in family


def test_search_streams_are_seeded_per_scenario_and_seed():
    scenario = _scenario()
    first = named_rng(scenario.scenario_seed, "search:0")
    assert first.random() == named_rng(scenario.scenario_seed, "search:0").random()
    assert first.random() != named_rng(scenario.scenario_seed, "search:1").random()
```

(imports: `from baselines.b1 import B1`, `from models.rng import named_rng`, `from optimization.tours import tour_family`.) Repeatability of every variant, including `P_seed1`, is already asserted by `test_search_is_monotone_within_budget_feasible_and_repeatable`. The existing `test_p_without_operations_is_b2_and_b3_differs_only_by_its_key` must keep passing: B2 and P-with-operations-off share name-independent behaviour only if the RNG namespace does not include the method name. **Decision:** the RNG namespace is `f"search:{params.seed}"` — the same stream for every method id — so that this test, and the spec's intent that B2 equals P without operations, hold. Update spec Section 7.1 accordingly in Step 5 of this task.

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra test pytest tests/optimization -q`
Expected: `ImportError` (`catalog_trajectory_block`, `repair_order`) and signature `TypeError`s.

- [ ] **Step 3: Implement the blocks**

`src/optimization/blocks.py` — imports and `SearchState.try_change`:

```python
from models.paths import CandidatePath, BackhaulRoutes, radio_links, radio_links_of_paths
from .ledger_summary import LedgerSummary
from .path_moves import path_neighbours, urgency_order
from .trajectory_moves import feasible_segment, sample_uav_segments, shift_segment
from .tours import TourSpec
```

```python
    def try_change(self, scorer: DesignScorer, **changes: object) -> bool:
        """Score the state with `changes` applied and adopt it only if the key increases strictly; new links get unit weights."""
        trial = replace(self, **changes)
        trial.weights = with_unit_weights(trial.weights, trial.paths, len(trial.trajectory) - 1)
        key = scorer.score(trial.plan())
        if key <= self.key:
            return False
        self.trajectory, self.paths, self.weights, self.key = trial.trajectory, trial.paths, trial.weights, key
        return True
```

Add the helper next to `unit_weights`:

```python
def with_unit_weights(
    weights: Mapping[LinkId, np.ndarray], paths: Mapping[str, CandidatePath | None], num_slots: int
) -> dict[LinkId, np.ndarray]:
    """`weights` plus a unit vector for every radio link of `paths` that has none; the same mapping object when nothing is missing."""
    missing = [link for link in radio_links_of_paths(paths) if link not in weights]
    if not missing:
        return weights
    return {**weights, **{link: np.ones(num_slots) for link in missing}}
```

Replace `path_block` and rename/add the trajectory blocks (`SearchParams` is not imported, to avoid the `bcd` ↔ `blocks` cycle — `params` is annotated with the string `"SearchParams"`):

```python
def path_block(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    routes: BackhaulRoutes,
    scorer: DesignScorer,
    state: SearchState,
    params: "SearchParams",
    rng: np.random.Generator,
) -> bool:
    """One ledger call, then for each alert in urgency order every neighbour of its current path, first improvement."""
    _, result = scorer.score_with_ledger(state.plan())
    summary = LedgerSummary.from_result(result)
    accepted = False
    for alert in urgency_order(list(scenario.alerts), rng):
        neighbours = path_neighbours(
            scenario, alert, state.paths[alert.id], candidates, routes, state.trajectory, summary, rng, params.entry_neighbors
        )
        for path in neighbours:
            if state.try_change(scorer, paths={**state.paths, alert.id: path}):
                accepted = True
    return accepted


def catalog_trajectory_block(scorer: DesignScorer, state: SearchState, family: Sequence[tuple[TourSpec, np.ndarray]]) -> bool:
    """Try every pre-enumerated tour (the pre-local-search block, kept for the P_catalog_trajectory ablation)."""
    accepted = False
    for _, trajectory in family:
        if not np.array_equal(trajectory, state.trajectory) and state.try_change(scorer, trajectory=trajectory):
            accepted = True
    return accepted


def trajectory_block(
    scenario: Scenario,
    scorer: DesignScorer,
    state: SearchState,
    params: "SearchParams",
    rng: np.random.Generator,
    family: Sequence[tuple[TourSpec, np.ndarray]],
) -> bool:
    """One ledger call, W sampled segment shifts (full then half reach), and a rare catalogue tour."""
    _, result = scorer.score_with_ledger(state.plan())
    summary = LedgerSummary.from_result(result)
    step_m = scenario.uav.v_max_mps * scenario.time.slot_s
    accepted = False
    for segment in sample_uav_segments(scenario, state.plan(), summary, rng, params.trajectory_segments):
        bounds = feasible_segment(state.trajectory, segment.start, segment.stop, segment.direction, step_m, params.max_widening)
        if bounds is None:
            continue
        for fraction in (1.0, 0.5):
            trial = shift_segment(state.trajectory, *bounds, segment.direction, fraction, step_m)
            if trial is not None and state.try_change(scorer, trajectory=trial):
                accepted = True
                break
    if family and rng.random() < params.catalog_probability:
        tour = family[int(rng.integers(len(family)))][1]
        if not np.array_equal(tour, state.trajectory) and state.try_change(scorer, trajectory=tour):
            accepted = True
    return accepted
```

`src/optimization/repair.py`:

```python
def repair_order(neighbours: Sequence[CandidatePath | None], risk: AlertRisk, backlog: bool = True) -> list[CandidatePath]:
    """Operation 2 order: served neighbours by (links in the congested set, cost_s); without backlog by cost_s only."""
    paths = [path for path in neighbours if path is not None]

    def order(path: CandidatePath) -> tuple:
        congested = sum(link in risk.congested_links for link in path.links) if backlog else 0
        return (congested, path.cost_s, path.links)

    return sorted(paths, key=order)


def repair_round(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    routes: BackhaulRoutes,
    scorer: DesignScorer,
    state: SearchState,
    params: "SearchParams",
    rng: np.random.Generator,
) -> bool:
    """One detection call, then operation 1 and operation 2 per at-risk alert."""
    _, result = scorer.score_with_ledger(state.plan())
    summary = LedgerSummary.from_result(result)
    accepted = False
    for risk in assess_risk(scenario, state.paths, result, params.backlog):
        alert = scenario.alert_by_id[risk.alert_id]
        if params.operation1 and risk.bottleneck[0] != BACKHAUL:
            weights = shift_bandwidth(state.weights, risk.bottleneck, alert, scenario.time.num_slots)
            accepted |= state.try_change(scorer, weights=weights)
        if params.operation2:
            neighbours = path_neighbours(
                scenario, alert, state.paths[alert.id], candidates, routes, state.trajectory, summary, rng, params.entry_neighbors
            )
            for path in repair_order(neighbours, risk, params.backlog):
                if state.try_change(scorer, paths={**state.paths, alert.id: path}):
                    accepted = True
                    break
    return accepted
```

Delete `path_change_order`. Import `path_neighbours`, `LedgerSummary`, `BackhaulRoutes`, `Sequence`.

`src/optimization/bcd.py` `search`:

```python
    def search(self, scenario, candidates) -> SearchOutcome:
        params = self.params
        scorer = DesignScorer(scenario, params.design_ids, params.budget, OBJECTIVES[params.objective])
        rng = named_rng(scenario.scenario_seed, f"search:{params.seed}")
        routes = backhaul_routes(scenario)
        initial = B1().plan(scenario, candidates, EvaluationCounter())
        state = SearchState(initial.trajectory, dict(initial.paths), unit_weights(scenario, candidates))
        family = tour_family(scenario) if params.trajectory_block else []
        iteration_keys: list[Key] = []
        exhausted = False
        try:
            state.key = scorer.score(state.plan())
            for _ in range(params.max_iterations):
                accepted = False
                if params.path_block:
                    accepted |= path_block(scenario, candidates, routes, scorer, state, params, rng)
                if params.bandwidth_block:
                    accepted |= bandwidth_block(scenario, scorer, state)
                if params.trajectory_block and params.trajectory_mode == CATALOG:
                    accepted |= catalog_trajectory_block(scorer, state, family)
                elif params.trajectory_block:
                    accepted |= trajectory_block(scenario, scorer, state, params, rng, family)
                if params.operation1 or params.operation2:
                    accepted |= repair_round(scenario, candidates, routes, scorer, state, params, rng)
                iteration_keys.append(state.key)
                if not accepted:
                    break
        except BudgetExhausted:
            exhausted = True
        return SearchOutcome(state.plan(), state.key, scorer.calls, tuple(iteration_keys), exhausted)
```

Imports: `from models.paths import CandidatePath, backhaul_routes`, `from models.rng import named_rng`, `from .blocks import SearchState, bandwidth_block, catalog_trajectory_block, path_block, trajectory_block, unit_weights`. Update the `SearchMethod` docstring: "Starts from B1 with unit weights; iterates the path, bandwidth, trajectory blocks and repair until an iteration accepts nothing."

- [ ] **Step 4: Run the whole suite**

Run: `uv run --extra test pytest -q`
Expected: all PASS. `test_search_is_monotone_within_budget_feasible_and_repeatable` covers determinism (same seed → same plan) for every variant.

- [ ] **Step 5: Record the RNG-namespace decision in the spec**

In `docs/superpowers/specs/2026-10-08-local-search-blocks-design.md` Section 7.1, replace `named_rng(scenario.scenario_seed, f"search:{self.name}:{params.seed}")` with `named_rng(scenario.scenario_seed, f"search:{params.seed}")` and replace the sentence "Different method ids (B2, P, ablations) get different streams, which is intended: they are different methods." with "Every method id shares the stream for a given `seed`, so B2 and P with both operations off make the same choices (the existing equivalence test keeps holding) and the ablations differ from P only by the block they switch."

- [ ] **Step 6: Commit**

```bash
git add src/optimization docs/superpowers/specs/2026-10-08-local-search-blocks-design.md tests/optimization
git commit -m "feat: local search path and trajectory blocks in the evaluator-guided BCD"
```

---

### Task 9: Configuration, ablation label, pilot config, docs

**Files:**
- Modify: `configs/experiments/phase2_main.yaml` (add `P_catalog_trajectory`)
- Create: `configs/experiments/phase2_pilot_local.yaml`
- Modify: `experiments/report_tables_phase2.py:20-28` (`ABLATIONS`)
- Modify: `README.md` (Pha 2 section: pilot command), `tests/runner/test_runner_phase2_configs.py`
- Test: `tests/runner/test_runner_phase2_configs.py`, `tests/runner/test_report_tables_phase2.py` (if it enumerates ablations)

**Interfaces:**
- Consumes: `SearchParams` fields of Task 4.
- Produces: method id `P_catalog_trajectory`; experiment id `phase2-pilot-local` writing `results/phase2/pilot_local/`.

- [ ] **Step 1: Extend the config test**

In `tests/runner/test_runner_phase2_configs.py` add to `MAIN_SEARCH`:

```python
    "P_catalog_trajectory": SearchParams(trajectory_mode="catalog", **REPAIR),
```

and add:

```python
PILOT_LOCAL = {
    "P": SearchParams(**REPAIR),
    "P_m2": SearchParams(entry_neighbors=2, **REPAIR),
    "P_m5": SearchParams(entry_neighbors=5, **REPAIR),
    "P_w3": SearchParams(trajectory_segments=3, **REPAIR),
    "P_w10": SearchParams(trajectory_segments=10, **REPAIR),
    "P_cat0": SearchParams(catalog_probability=0.0, **REPAIR),
    "P_cat03": SearchParams(catalog_probability=0.3, **REPAIR),
    "P_seed1": SearchParams(seed=1, **REPAIR),
    "P_seed2": SearchParams(seed=2, **REPAIR),
    "P_catalog_trajectory": SearchParams(trajectory_mode="catalog", **REPAIR),
}


def test_local_pilot_config_sweeps_one_parameter_at_a_time_on_the_development_split():
    config = load_experiment_config(Path("configs/experiments/phase2_pilot_local.yaml"))
    assert (config.split, config.output_dir) == ("dev", Path("results/phase2/pilot_local"))
    assert [item.set_id for item in config.scenario_sets] == ["v0"]
    assert [spec.id for spec in config.methods] == ["B1", "B2", *PILOT_LOCAL]
    assert _search(config) == {"B2": SearchParams(), **PILOT_LOCAL}
    assert not any(spec.bounds for spec in config.methods)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra test pytest tests/runner/test_runner_phase2_configs.py -q`
Expected: FAIL (`P_catalog_trajectory` missing; pilot config file missing).

- [ ] **Step 3: Write the configs and the label**

Append to `configs/experiments/phase2_main.yaml` methods list, after `P_expected_design`:

```yaml
  - {id: P_catalog_trajectory, base: search, params: {operation1: true, operation2: true, trajectory_mode: catalog}}
```

Create `configs/experiments/phase2_pilot_local.yaml`:

```yaml
experiment_id: phase2-pilot-local
scenario_sets:
  - set_id: v0
    manifest: data/manifests/scenarios_v0.csv
    scenario_dir: data/processed/scenarios/v0
split: dev
realization_ids: {start: 0, stop: 30}
methods:
  - B1
  - {id: B2, base: search}
  - {id: P, base: search, params: {operation1: true, operation2: true}}
  - {id: P_m2, base: search, params: {operation1: true, operation2: true, entry_neighbors: 2}}
  - {id: P_m5, base: search, params: {operation1: true, operation2: true, entry_neighbors: 5}}
  - {id: P_w3, base: search, params: {operation1: true, operation2: true, trajectory_segments: 3}}
  - {id: P_w10, base: search, params: {operation1: true, operation2: true, trajectory_segments: 10}}
  - {id: P_cat0, base: search, params: {operation1: true, operation2: true, catalog_probability: 0.0}}
  - {id: P_cat03, base: search, params: {operation1: true, operation2: true, catalog_probability: 0.3}}
  - {id: P_seed1, base: search, params: {operation1: true, operation2: true, seed: 1}}
  - {id: P_seed2, base: search, params: {operation1: true, operation2: true, seed: 2}}
  - {id: P_catalog_trajectory, base: search, params: {operation1: true, operation2: true, trajectory_mode: catalog}}
bounds: {tangents: 10}
bootstrap: {resamples: 10000, seed: 20260911, confidence: 0.95}
workers: 12
output_dir: results/phase2/pilot_local
```

In `experiments/report_tables_phase2.py` `ABLATIONS`, append `("P_catalog_trajectory", "Catalogue trajectory block"),` after the `P_expected_design` entry. If `tests/runner/test_report_tables_phase2.py` fixes the ablation row count or labels, update it to include the new row (run the test to see).

In `README.md`, section "Pha 2: phương pháp tìm kiếm, thí nghiệm và thống kê", add before the `phase2_main` line of the command block:

```bash
uv run python experiments/run_phase2.py --config configs/experiments/phase2_pilot_local.yaml  # pilot tham số local search (M, W, xác suất catalogue, seed) trên tập phát triển
```

- [ ] **Step 4: Run the runner tests**

Run: `uv run --extra test pytest tests/runner -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/experiments/phase2_main.yaml configs/experiments/phase2_pilot_local.yaml experiments/report_tables_phase2.py README.md tests/runner
git commit -m "feat: add the catalogue-trajectory ablation and the local search pilot config"
```

---

### Task 10: Run the pilot, fix the defaults, record Section 12

**Files:**
- Output: `results/phase2/pilot_local/{summary.csv,method_realizations.csv,manifest.json}` (commit; `shards/` is git-ignored)
- Modify: `src/optimization/bcd.py` (defaults, only if the pilot changes them), `configs/experiments/phase2_pilot_local.yaml` (no change expected), `docs/superpowers/specs/2026-10-08-local-search-blocks-design.md` Section 12
- Test: `tests/optimization/test_search_methods.py::test_search_params_from_mapping_validates_values` (defaults assertion), `tests/runner/test_runner_phase2_configs.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the parameter defaults Plan E.2 runs with, and the pilot table the report cites.

- [ ] **Step 1: Run the pilot**

Run (background; about 12 search variants × 6 development scenarios ≈ 72 tasks at 2–4 min each on 12 workers → 20–40 min):

```bash
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 \
  uv run python experiments/run_phase2.py --config configs/experiments/phase2_pilot_local.yaml
```

Expected: exit code 0, `results/phase2/pilot_local/manifest.json` with `"status": "complete"` and `"failures": []`. An infeasible-plan `ValueError` from `run_method_task` means a move produced an invalid path or trajectory — reproduce on that scenario with `evaluate(..., check=True)` and fix the move before rerunning.

- [ ] **Step 2: Tabulate the pilot**

Run:

```bash
uv run python - <<'EOF'
import csv
rows = list(csv.DictReader(open("results/phase2/pilot_local/summary.csv")))
print("| Method | timely mean | CI low | CI high | planning s | calls |")
print("|---|---|---|---|---|---|")
for row in rows:
    print(f"| {row['method']} | {float(row['timely_ratio_mean']):.3f} | {float(row['timely_ratio_ci_low']):.3f} | {float(row['timely_ratio_ci_high']):.3f} | {float(row['planning_runtime_s_mean']):.0f} | {float(row['evaluate_calls_mean']):.0f} |")
EOF
```

Decision rule (write it down with the numbers): for each of M, W and `catalog_probability`, keep the spec default unless a neighbouring value beats it by more than the half-width of the default's bootstrap CI; among ties prefer the cheaper setting (fewer evaluations per pass: smaller M, smaller W, lower probability). The seed rows report the spread (max − min of `timely_ratio_mean` over seeds 0, 1, 2); the `P_catalog_trajectory` row is the first local-vs-catalogue comparison.

- [ ] **Step 3: Apply the chosen defaults**

If a default changes: edit the field default in `SearchParams` (`src/optimization/bcd.py`), the defaults assertion in `test_search_params_from_mapping_validates_values`, and the "Spec defaults" line of this plan's Global Constraints; spec Section 7.2 table too. Run `uv run --extra test pytest -q` → PASS.

- [ ] **Step 4: Write Section 12 of the spec**

Replace the body of Section 12 with: the chosen values of M, W, `catalog_probability`; the Markdown table from Step 2; the seed spread; the sentence comparing `P` and `P_catalog_trajectory` on the development split; any deviation from the spec found during Tasks 1–9 (at least the RNG-namespace change of Task 8 Step 5). Also note the measured planning runtime per search task, which Plan E.2 uses to size the rerun.

- [ ] **Step 5: Commit**

```bash
git add results/phase2/pilot_local/summary.csv results/phase2/pilot_local/method_realizations.csv results/phase2/pilot_local/manifest.json \
        docs/superpowers/specs/2026-10-08-local-search-blocks-design.md src/optimization/bcd.py tests/optimization/test_search_methods.py
git commit -m "docs: record the local search pilot and fix the search defaults"
```

---

## After this plan

Plan E.2 (to be written with the writing-plans skill once Task 10's numbers exist) covers spec Sections 8–9: deleting the stale search shards, rerunning `phase2_main`, `phase2_budget`, `phase2_sensitivity`, `phase2_design`, `phase2_case_study`, `case_study_trace`, the statistics scripts and `report_tables*.py`; updating `docs/reproducibility.md`; revising the report sections, tables and `main.pdf`; editing the slides; and the final `reproduce.py` levels.
