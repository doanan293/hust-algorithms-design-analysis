# Phase 2 Methods Implementation Plan (Sub-Project C, Plan C.1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement spec C Sections 5–10 (design realizations, weighted bandwidth, design scoring with a budget, the search methods B2, B3, P and their ablation parameters, runner method specs with in-task trajectory bounds and the union bound), confirm that Phase 1 still reproduces, and run the development-set pilot of Section 17.1.

**Architecture:** `src/models` gains a `"design"` channel namespace and the `WeightedBacklogged` policy. A new package `src/optimization` holds `objectives` (lexicographic and leximin keys), `scoring` (`DesignScorer` with a call budget), `tours` (feasible-by-construction tour family), `blocks` (`SearchState` and the path, bandwidth, trajectory blocks), `repair` (risk detection and operations 1–2), and `bcd` (`SearchParams`, `SearchMethod`). `src/runner` accepts method specs, builds every method in `runner/methods.py`, solves search-method trajectory bounds inside method tasks, and reports the union bound; `experiments/run_phase2.py` runs `configs/experiments/phase2_pilot.yaml`.

**Tech Stack:** Python 3.12 in the uv-managed `.venv`, `numpy`, `scipy` 1.18 (HiGHS), `PyYAML`, `pytest`. No new dependency.

**Spec:** `docs/superpowers/specs/2026-09-11-phase2-core-design.md` (Sections 5–10, 15, 16.1, 17.1). Builds on release `v0.1.0`: `docs/superpowers/specs/2026-09-11-model-evaluator-design.md`, `docs/superpowers/specs/2026-09-11-phase1-completion-design.md`, and `docs/superpowers/plans/2026-09-11-phase1-completion.md`. Plan C.2 (Sections 11–14) is written from the pilot evidence of Task 9.

## Global Constraints

- Run project code through uv: tests with `uv run --extra test pytest`, scripts with `uv run python ...`. No dependency is added. Skill-script dependencies stay installed globally and never go into `.venv`.
- Import boundaries: `src/models` imports none of `baselines`, `bounds`, `optimization`, `runner`; `src/baselines` is unchanged and does not import `optimization`; `src/optimization` imports `models` and `baselines` only; `src/runner` imports `models`, `baselines`, `bounds`, `optimization`; nothing under `src` imports `experiments`.
- Backward compatibility: `channel_uniforms(scenario, realization_id)` still draws from `named_rng(scenario_seed, f"channel-eval:{realization_id}")`; `evaluate` defaults to `channel_namespace="eval"`; `EqualSplitBacklogged` and `StaticSchedule` behave exactly as before; the Phase 1 configuration yields the same task keys and the same non-runtime values in every committed `results/phase1` column.
- Planners never touch the `"eval"` namespace: search methods score plans only through `DesignScorer` (namespace `"design"`, or expected rates when `design_ids` is empty); the runner's final evaluation uses `"eval"`.
- Budget: every `DesignScorer` call is one evaluation whatever the number of design realizations; once `calls == budget` the next call raises `BudgetExhausted` and the method returns its incumbent. The `evaluate_calls` column of a search method is its scorer calls plus the runner's final evaluation.
- Acceptance: a change is adopted only if its key is strictly greater; ties keep the incumbent; an infeasible plan scores `(-inf,)`, below every feasible key.
- Defaults: `design_ids = (0, 1, 2, 3, 4)`, `budget = 2000`, `max_iterations = 10`, slack threshold θ = 2 slots, at-risk limit R = 10, congestion tolerance `1e-6` relative, buffer tolerance `1e-6` bits.
- Task hashes cover the source of `src/models`, `src/baselines`, `src/bounds`, `src/optimization`, `src/runner`.
- `results/phase2/**/shards/` is Git-ignored; every other file in `results/phase2/pilot/` is committed. Evaluation-split scenarios are never used for tuning; the pilot uses split `dev`.
- Test module basenames are unique across `tests/`; shared builders live in `tests/support/`.
- Every code task follows test-driven development and ends with its own commit.

## Spec Clarifications

The spec leaves these details open; this plan fixes them, and the tests encode them.

1. **Tour family (7.3).** A segment of length $d$ takes $\lceil d/(V_{\max}\Delta)\rceil$ steps (zero for $d = 0$). Hover slots are $\lfloor \mathrm{spare}\cdot w_i/\sum w\rfloor$ in exact rational arithmetic. A tour identical to an earlier one in enumeration order is dropped, because single-zone tours coincide across hover splits. The B1 zone tour belongs to the family.
2. **Path block (7.2).** An alert's option list is fixed when its turn starts (the current option excluded); each later option is compared with the updated incumbent.
3. **Bandwidth block (7.2).** Halving a link's weights is tried only when doubling was rejected, since halving an accepted doubling restores the rejected state.
4. **Repair round (9).** Detection runs once per round. For each at-risk alert, operation 1 is tried first, then operation 2 on the updated incumbent.
5. **Bottleneck hop (9.1).** A hop holds bits in a slot when its buffer at the start of the slot (after the release) exceeds `1e-6` bits, counted over slots $[r_a, \min(d_a, N))$. The most frequent bottleneck across design realizations wins; ties go to the earliest hop.
6. **Congestion (9.1).** A backlogged link whose recorded capacity is zero is congested (it carried 0 of 0 bits).
7. **Search trajectory.** `SearchMethod.trajectory()` raises `NotImplementedError`; the runner never calls it for search specs.
8. **Method specs (10).** A B0 or B1 entry in mapping form must have `id` equal to `base` and no `params` or `bounds`; its bounds come from the Phase 1 bound tasks. Search ids match `^[A-Za-z0-9.-]+(_[A-Za-z0-9.-]+)*$` and are not `B0` or `B1`. `params` are validated by `SearchParams`; `design_ids` is a non-empty list or `"expected"`. `milp_crosscheck` is optional; without it no cross-check task runs and `milp_crosscheck.csv` is not written.
9. **Summary (10).** Three columns are appended: `union_bound_ratio_mean`, `union_gap_mean`, `evaluate_calls_mean`. A method without its own bound gets `nan` for `bound_ratio_mean`, `gap_mean`, `bound_runtime_s_mean`, and `gap_undefined_count` equal to its scenario count. A scenario's union bound is defined only when every realization has at least one `trajectory` bound row. Existing columns keep their order and values.
10. **Shards.** Method-task shards gain a `bound_rows` list, which is empty for B0 and B1.

---
### Task 1: Design realizations and weighted bandwidth policy

**Files:**
- Modify: `src/models/channel.py`, `src/models/evaluate.py`, `src/models/plan.py`, `src/models/simulator.py`
- Test: `tests/models/test_design_extensions.py`

**Interfaces:**
- Consumes (sub-project A): `models.rng.named_rng`, `models.paths.radio_links`, `models.channel.{ACCESS, LinkId}`, test helpers `scenario_builders.{BASE_SCENARIO, make_alert}` and `channel_builders.constant_channels`.
- Produces: `models.channel.CHANNEL_NAMESPACES = ("eval", "design")`; `channel_uniforms(scenario, realization_id, namespace="eval") -> dict[str, np.ndarray]` (raises `ValueError("unknown channel namespace ...")`); `evaluate(scenario, plan, mode, realization_ids=(), counter=None, candidates=None, keep_ledger=False, channel_namespace="eval")`; `EvaluationResult.channel_namespace: str = "eval"` (last field); frozen dataclass `models.plan.WeightedBacklogged(weights: Mapping[LinkId, np.ndarray])` with `weight(link, slot) -> float` (0 for a missing link); `Plan.bandwidth` accepts it; `validate_plan` reports `bandwidth: unknown radio link {link}` and `bandwidth: weights of {link} need {N} finite non-negative values`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_design_extensions.py
import copy

import numpy as np
import pytest

from channel_builders import constant_channels
from models.channel import ACCESS, channel_uniforms
from models.evaluate import REALIZED, evaluate
from models.paths import candidate_paths, radio_links
from models.plan import EqualSplitBacklogged, Plan, WeightedBacklogged, stationary_trajectory, validate_plan
from models.rng import named_rng
from models.scenario import scenario_from_dict
from models.simulator import simulate
from scenario_builders import BASE_SCENARIO, make_alert


def _scenario(alerts):
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = alerts
    return scenario_from_dict(raw)


def test_eval_namespace_keeps_previous_draws_and_design_differs():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 10, 100000)])
    expected = named_rng(scenario.scenario_seed, "channel-eval:3").random((6, scenario.time.num_slots))
    default = channel_uniforms(scenario, 3)
    assert np.array_equal(default["s00"], expected[0])
    assert np.array_equal(channel_uniforms(scenario, 3, "eval")["n3"], expected[5])
    assert not np.array_equal(channel_uniforms(scenario, 3, "design")["s00"], expected[0])
    with pytest.raises(ValueError, match="unknown channel namespace"):
        channel_uniforms(scenario, 3, "test")


def test_evaluate_records_the_channel_namespace():
    scenario = _scenario([make_alert("alert-0000", "s01", 0, 20, 20000)])
    candidates = candidate_paths(scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0}, EqualSplitBacklogged())
    assert evaluate(scenario, plan, REALIZED, (0,), candidates=candidates).channel_namespace == "eval"
    design = evaluate(scenario, plan, REALIZED, (0,), candidates=candidates, channel_namespace="design")
    assert design.channel_namespace == "design" and design.feasible


def test_weights_split_bandwidth_in_proportion():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 10, 900000), make_alert("alert-0001", "s00", 0, 10, 900000)])
    candidates = candidate_paths(scenario)
    to_n1, to_n0 = candidates["alert-0000"][0].links[0], candidates["alert-0000"][1].links[0]
    assert (to_n1[2], to_n0[2]) == ("n1", "n0")
    weights = {to_n1: np.full(20, 3.0), to_n0: np.full(20, 1.0)}
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0, "alert-0001": 1}, WeightedBacklogged(weights))
    assert validate_plan(scenario, candidates, plan) == []
    outcome = simulate(scenario, candidates, plan, constant_channels(scenario, candidates, 1e6))
    slot_zero = {link: hz for slot, link, hz in outcome.ledger.bandwidth if slot == 0 and link[0] == ACCESS}
    assert slot_zero == {to_n0: 250000.0, to_n1: 750000.0}


def test_unit_weights_reproduce_equal_split_exactly():
    scenario = _scenario([
        make_alert("alert-0000", "s00", 0, 12, 400000),
        make_alert("alert-0001", "s01", 1, 18, 50000),
        make_alert("alert-0002", "s00", 2, 15, 300000),
    ])
    candidates = candidate_paths(scenario)
    choice = {"alert-0000": 0, "alert-0001": 0, "alert-0002": 2}
    channels = constant_channels(scenario, candidates, 1e6)
    equal = simulate(scenario, candidates, Plan(stationary_trajectory(scenario), choice, EqualSplitBacklogged()), channels)
    ones = {link: np.ones(20) for link in radio_links(candidates)}
    weighted = simulate(scenario, candidates, Plan(stationary_trajectory(scenario), choice, WeightedBacklogged(ones)), channels)
    assert weighted.ledger == equal.ledger
    zeros = {link: np.zeros(20) for link in radio_links(candidates)}
    fallback = simulate(scenario, candidates, Plan(stationary_trajectory(scenario), choice, WeightedBacklogged(zeros)), channels)
    assert fallback.ledger == equal.ledger


def test_malformed_weights_are_rejected():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 10, 100000)])
    candidates = candidate_paths(scenario)
    link = candidates["alert-0000"][0].links[0]
    for weights, message in (
        ({link: np.full(20, -1.0)}, "finite non-negative"),
        ({link: np.ones(19)}, "finite non-negative"),
        ({(ACCESS, "s00", "n9"): np.ones(20)}, "unknown radio link"),
    ):
        plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0}, WeightedBacklogged(weights))
        assert any(message in item for item in validate_plan(scenario, candidates, plan))


def test_random_weighted_plans_pass_the_ledger_checker():
    scenario = _scenario([
        make_alert("alert-0000", "s00", 0, 12, 400000),
        make_alert("alert-0001", "s00", 3, 20, 700000),
        make_alert("alert-0002", "s01", 1, 18, 50000),
    ])
    candidates = candidate_paths(scenario)
    rng = np.random.default_rng(11)
    for _ in range(6):
        weights = {link: rng.uniform(0.0, 3.0, size=20) for link in radio_links(candidates)}
        choice = {alert.id: int(rng.integers(0, len(candidates[alert.id]))) for alert in scenario.alerts}
        plan = Plan(stationary_trajectory(scenario), choice, WeightedBacklogged(weights))
        result = evaluate(scenario, plan, REALIZED, (0, 1), candidates=candidates, channel_namespace="design")
        assert result.feasible
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/models/test_design_extensions.py -v`

Expected: collection fails with `ImportError: cannot import name 'WeightedBacklogged' from 'models.plan'`.

- [ ] **Step 3: Add the channel namespace**

In `src/models/channel.py` (2 edits):

Edit 1: replace

```python
LinkId = tuple[str, str, str]
ACCESS = "access"
```

with

```python
LinkId = tuple[str, str, str]
CHANNEL_NAMESPACES = ("eval", "design")
ACCESS = "access"
```

Edit 2: replace

```python

def channel_uniforms(scenario: Scenario, realization_id: int) -> dict[str, np.ndarray]:
    rng = named_rng(scenario.scenario_seed, f"channel-eval:{realization_id}")
    point_ids = ground_point_ids(scenario)
```

with

```python

def channel_uniforms(scenario: Scenario, realization_id: int, namespace: str = "eval") -> dict[str, np.ndarray]:
    """Uniform draws per ground point and slot; "eval" scores results, "design" is reserved for planners."""
    if namespace not in CHANNEL_NAMESPACES:
        raise ValueError(f"unknown channel namespace {namespace!r}; expected one of {CHANNEL_NAMESPACES}")
    rng = named_rng(scenario.scenario_seed, f"channel-{namespace}:{realization_id}")
    point_ids = ground_point_ids(scenario)
```

- [ ] **Step 4: Pass the namespace through evaluate**

In `src/models/evaluate.py` (4 edits):

Edit 1: replace

```python
    runtime_s: float
```

with

```python
    runtime_s: float
    channel_namespace: str = "eval"
```

Edit 2: replace

```python
    keep_ledger: bool = False,
) -> EvaluationResult:
```

with

```python
    keep_ledger: bool = False,
    channel_namespace: str = "eval",
) -> EvaluationResult:
```

Edit 3: replace

```python
            runtime_s=time.perf_counter() - started,
        )

    links = set(radio_links(candidates)) | set(all_uav_links(scenario))
    without_uav = connectivity(scenario, None)
    results = []
    for realization_id in realization_ids if mode == REALIZED else (None,):
        uniforms = None if realization_id is None else channel_uniforms(scenario, realization_id)
        channels = build_link_channels(scenario, links, plan.trajectory, uniforms)
```

with

```python
            runtime_s=time.perf_counter() - started,
            channel_namespace=channel_namespace,
        )

    links = set(radio_links(candidates)) | set(all_uav_links(scenario))
    without_uav = connectivity(scenario, None)
    results = []
    for realization_id in realization_ids if mode == REALIZED else (None,):
        uniforms = None if realization_id is None else channel_uniforms(scenario, realization_id, channel_namespace)
        channels = build_link_channels(scenario, links, plan.trajectory, uniforms)
```

Edit 4: replace

```python
        runtime_s=time.perf_counter() - started,
    )
```

with

```python
        runtime_s=time.perf_counter() - started,
        channel_namespace=channel_namespace,
    )
```

- [ ] **Step 5: Add the weighted policy and its validation**

In `src/models/plan.py` (3 edits):

Edit 1: replace

```python
@dataclass(frozen=True)
class Plan:
    trajectory: np.ndarray
    path_choice: Mapping[str, int | None]
    bandwidth: EqualSplitBacklogged | StaticSchedule
```

with

```python
@dataclass(frozen=True)
class WeightedBacklogged:
    """Split B_tot among backlogged links of a half-slot in proportion to per-slot weights; missing links weigh 0."""

    weights: Mapping[LinkId, np.ndarray]

    def weight(self, link: LinkId, slot: int) -> float:
        values = self.weights.get(link)
        return 0.0 if values is None else float(values[slot])


@dataclass(frozen=True)
class Plan:
    trajectory: np.ndarray
    path_choice: Mapping[str, int | None]
    bandwidth: EqualSplitBacklogged | StaticSchedule | WeightedBacklogged
```

Edit 2: replace

```python
        violations.extend(_schedule_violations(scenario, candidates, plan.bandwidth))
    elif not isinstance(plan.bandwidth, EqualSplitBacklogged):
```

with

```python
        violations.extend(_schedule_violations(scenario, candidates, plan.bandwidth))
    elif isinstance(plan.bandwidth, WeightedBacklogged):
        violations.extend(_weight_violations(scenario, candidates, plan.bandwidth))
    elif not isinstance(plan.bandwidth, EqualSplitBacklogged):
```

Edit 3: replace

```python
        violations.append(f"trajectory: slot {slot} moves {steps[slot]:.3f} m, limit {limit:.3f} m")
    return violations
```

with

```python
        violations.append(f"trajectory: slot {slot} moves {steps[slot]:.3f} m, limit {limit:.3f} m")
    return violations


def _weight_violations(
    scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]], policy: WeightedBacklogged
) -> list[str]:
    known = set(radio_links(candidates))
    violations = []
    for link in sorted(policy.weights):
        if link not in known:
            violations.append(f"bandwidth: unknown radio link {link}")
            continue
        values = np.asarray(policy.weights[link], dtype=float)
        if values.shape != (scenario.time.num_slots,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            violations.append(f"bandwidth: weights of {link} need {scenario.time.num_slots} finite non-negative values")
    return violations
```

- [ ] **Step 6: Allocate bandwidth by weight in the simulator**

In `src/models/simulator.py` (2 edits):

Edit 1: replace

```python
from .paths import CandidatePath
from .plan import Plan, StaticSchedule, chosen_paths
from .scenario import Scenario
```

with

```python
from .paths import CandidatePath
from .plan import Plan, StaticSchedule, WeightedBacklogged, chosen_paths
from .scenario import Scenario
```

Edit 2: replace

```python
    active = [item[2] for item in ranked_downlinks[: scenario.uav.max_active_downlinks]]
    allocation = {link: total / len(access) for link in access}
```

with

```python
    active = [item[2] for item in ranked_downlinks[: scenario.uav.max_active_downlinks]]
    if isinstance(plan.bandwidth, WeightedBacklogged):
        allocation = {}
        for group in (access, active):
            weights = [plan.bandwidth.weight(link, slot) for link in group]
            weight_sum = sum(weights)
            for link, weight in zip(group, weights):
                allocation[link] = total * weight / weight_sum if weight_sum > 0.0 else total / len(group)
        return allocation
    allocation = {link: total / len(access) for link in access}
```

Within each group (backlogged access links; the active downlinks), a positive weight sum gives $b_e = B_{\mathrm{tot}} w_e/\sum w$ and a zero sum gives an equal split. The ledger checker needs no change.

- [ ] **Step 7: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/models/test_design_extensions.py -v`

Expected: 6 passed.

Run: `uv run --extra test pytest -q`

Expected: 125 passed, 1 skipped.

- [ ] **Step 8: Commit**

```bash
git add src/models tests/models/test_design_extensions.py
git commit -m "feat: add design realizations and weighted bandwidth policy"
```

### Task 2: Design scorer and objectives

**Files:**
- Create: `src/optimization/__init__.py`, `src/optimization/objectives.py`, `src/optimization/scoring.py`
- Create: `tests/support/search_builders.py`
- Test: `tests/optimization/test_objectives_scoring.py`

**Interfaces:**
- Consumes: Task 1 `evaluate(..., channel_namespace)`; `models.evaluate.{EXPECTED, REALIZED, EvaluationResult}`; `baselines.b1.B1` (tests only).
- Produces: `optimization.objectives.Key = tuple`; `Objective = Callable[[Scenario, EvaluationResult], Key]`; `lexicographic_key(scenario, result) -> Key` (the evaluator key `(timely_count, connected_count, -round(shortfall, 9))`); `leximin_key(scenario, result) -> Key` (mean delivered fraction per alert, capped at 1, rounded to 9 digits, sorted ascending, then `timely_count`, `connected_count`); `OBJECTIVES = {"lex": ..., "leximin": ...}`. `optimization.scoring.DESIGN_NAMESPACE = "design"`, `DEFAULT_DESIGN_IDS = (0, 1, 2, 3, 4)`, `DEFAULT_BUDGET = 2000`, `INFEASIBLE_SCORE = (-math.inf,)`, `BudgetExhausted(RuntimeError)`, `DesignScorer(scenario, candidates, design_ids=DEFAULT_DESIGN_IDS, budget=DEFAULT_BUDGET, objective=lexicographic_key)` with attributes `calls`, `budget`, `design_ids`, property `remaining`, `score(plan) -> Key`, `score_with_ledger(plan) -> tuple[Key, EvaluationResult]` (ledgers kept); `ValueError` for `budget < 1`. Test builders `search_builders.base_scenario(alerts, **changes)`, `three_zone_raw(num_slots=50)`, `hopeless_raw()`, `bandwidth_raw()`.

- [ ] **Step 1: Add the shared search test builders**

```python
# tests/support/search_builders.py
import copy

from models.scenario import scenario_from_dict
from scenario_builders import BASE_SCENARIO, make_alert


def base_scenario(alerts, **changes):
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = alerts
    for key, value in changes.items():
        raw[key] = value
    return scenario_from_dict(raw)


def three_zone_raw(num_slots=50):
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["damage_zones"] = [
        {"x_m": 1230.0, "y_m": 1000.0, "radius_m": 400.0},
        {"x_m": 1000.0, "y_m": 1400.0, "radius_m": 300.0},
        {"x_m": 100.0, "y_m": 100.0, "radius_m": 200.0},
    ]
    raw["sources"] = [
        {"id": "s00", "x_m": 1230.0, "y_m": 1000.0, "zone": 0},
        {"id": "s01", "x_m": 100.0, "y_m": 100.0, "zone": 2},
        {"id": "s02", "x_m": 1000.0, "y_m": 1500.0, "zone": 1},
        {"id": "s03", "x_m": 1100.0, "y_m": 1300.0, "zone": 1},
    ]
    raw["alerts"] = [
        make_alert("alert-0000", "s00", 0, 10, 100000),
        make_alert("alert-0001", "s02", 0, 20, 100000),
        make_alert("alert-0002", "s03", 5, 45, 100000),
    ]
    raw["time"]["num_slots"] = num_slots
    return raw


def hopeless_raw():
    """One ground candidate per alert; a hopeless 12 Mbit alert holds the 1 Mbit/s backhaul until its deadline."""
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["paths"] = {"k": 1, "max_ground": 1, "version": 1}
    raw["alerts"] = [make_alert("alert-0000", "s00", 0, 10, 12000000), make_alert("alert-0001", "s00", 0, 11, 1500000)]
    return raw


def bandwidth_raw():
    """Two access links share B_tot; alert-0000 is timely only with more than an equal share."""
    raw = copy.deepcopy(BASE_SCENARIO)
    for edge in raw["network"]["edges"]:
        edge["capacity_bps"] = 1e8
    raw["sources"].append({"id": "s02", "x_m": 1000.0, "y_m": 1430.0, "zone": 0})
    raw["time"]["num_slots"] = 40
    raw["alerts"] = [make_alert("alert-0000", "s00", 0, 5, 4500000), make_alert("alert-0001", "s02", 0, 40, 4000000)]
    return raw
```

`three_zone_raw` places zone 2 1273 m from the center, so with 50 slots every tour through zone 2 exceeds the mission; `hopeless_raw` and `bandwidth_raw` are the block and repair instances of Tasks 4–5.

- [ ] **Step 2: Write the failing tests**

```python
# tests/optimization/test_objectives_scoring.py
from types import SimpleNamespace

import pytest

from baselines.b1 import B1
from models.evaluate import EXPECTED, evaluate, REALIZED
from models.paths import candidate_paths
from optimization.objectives import leximin_key, lexicographic_key
from optimization.scoring import INFEASIBLE_SCORE, BudgetExhausted, DesignScorer
from scenario_builders import make_alert
from search_builders import base_scenario


def _result(delivered, key):
    return SimpleNamespace(realizations=tuple(SimpleNamespace(delivered_bits=item) for item in delivered), key=key)


def test_lexicographic_key_is_the_evaluator_key():
    scenario = base_scenario([make_alert("alert-0000", "s00", 0, 10, 100)])
    assert lexicographic_key(scenario, _result([{"alert-0000": 100.0}], (1, 2, -0.0))) == (1, 2, -0.0)


def test_leximin_key_sorts_mean_fractions_then_counts():
    scenario = base_scenario([
        make_alert("alert-0000", "s00", 0, 10, 100),
        make_alert("alert-0001", "s00", 0, 10, 200),
        make_alert("alert-0002", "s00", 0, 10, 300),
    ])
    delivered = [
        {"alert-0000": 100.0, "alert-0001": 50.0, "alert-0002": 0.0},
        {"alert-0000": 50.0, "alert-0001": 200.0, "alert-0002": 100.0},
    ]
    key = leximin_key(scenario, _result(delivered, (3, 4, -2.5)))
    assert key == (round(1 / 6, 9), 0.625, 0.75, 3, 4)
    assert key > (round(1 / 6, 9), 0.6, 1.0, 6, 4)
    assert key < (round(1 / 6, 9), 0.625, 0.75, 4, 0)
    assert INFEASIBLE_SCORE < (0.0, 0.0, 0.0, 0, 0) and INFEASIBLE_SCORE < (0, 0, -40.0)


def _far_source_plan():
    scenario = base_scenario([make_alert("alert-0000", "s01", 0, 20, 1000000)])
    candidates = candidate_paths(scenario)
    plan = B1().plan(scenario, candidates, None)
    return scenario, candidates, type(plan)(plan.trajectory, {"alert-0000": 0}, plan.bandwidth)


def test_scorer_uses_the_design_namespace():
    scenario, candidates, plan = _far_source_plan()
    scorer = DesignScorer(scenario, candidates, design_ids=(0, 1, 2, 3, 4), budget=10)
    key, result = scorer.score_with_ledger(plan)
    design = evaluate(scenario, plan, REALIZED, (0, 1, 2, 3, 4), candidates=candidates, channel_namespace="design")
    public = evaluate(scenario, plan, REALIZED, (0, 1, 2, 3, 4), candidates=candidates)
    assert result.channel_namespace == "design"
    assert all(item.ledger is not None for item in result.realizations)
    assert key == design.key
    assert [item.delivered_bits for item in result.realizations] == [item.delivered_bits for item in design.realizations]
    assert [item.delivered_bits for item in result.realizations] != [item.delivered_bits for item in public.realizations]


def test_scorer_counts_calls_and_stops_at_the_budget():
    scenario, candidates, plan = _far_source_plan()
    scorer = DesignScorer(scenario, candidates, design_ids=(0,), budget=2)
    scorer.score(plan)
    scorer.score_with_ledger(plan)
    assert (scorer.calls, scorer.remaining) == (2, 0)
    with pytest.raises(BudgetExhausted):
        scorer.score(plan)
    assert scorer.calls == 2
    with pytest.raises(ValueError, match="budget"):
        DesignScorer(scenario, candidates, budget=0)


def test_empty_design_ids_score_expected_rates_and_infeasible_plans_score_lowest():
    scenario, candidates, plan = _far_source_plan()
    scorer = DesignScorer(scenario, candidates, design_ids=(), budget=5)
    key, result = scorer.score_with_ledger(plan)
    assert result.mode == EXPECTED and len(result.realizations) == 1
    assert key == evaluate(scenario, plan, EXPECTED, candidates=candidates).key
    broken = type(plan)(plan.trajectory, {"alert-0000": 7}, plan.bandwidth)
    assert scorer.score(broken) == INFEASIBLE_SCORE
```

- [ ] **Step 3: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/optimization/test_objectives_scoring.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'optimization'`.

- [ ] **Step 4: Implement the objectives and the scorer**

```python
# src/optimization/__init__.py
"""Evaluator-guided search methods B2, B3, P and their ablation variants."""
```

```python
# src/optimization/objectives.py
"""Acceptance keys of search methods; a larger key is a better plan."""

from typing import Callable

from models.evaluate import EvaluationResult
from models.scenario import Scenario

Key = tuple
Objective = Callable[[Scenario, EvaluationResult], Key]
FRACTION_DIGITS = 9


def lexicographic_key(scenario: Scenario, result: EvaluationResult) -> Key:
    """(timely_count, connected_count, -round(shortfall, 9)) summed over realizations."""
    return result.key


def leximin_key(scenario: Scenario, result: EvaluationResult) -> Key:
    """Mean delivered fraction per alert sorted ascending, then timely_count and connected_count."""
    count = len(result.realizations)
    fractions = sorted(
        round(
            sum(min(item.delivered_bits[alert.id] / alert.size_bits, 1.0) for item in result.realizations) / count,
            FRACTION_DIGITS,
        )
        for alert in scenario.alerts
    )
    return (*fractions, result.key[0], result.key[1])


OBJECTIVES: dict[str, Objective] = {"lex": lexicographic_key, "leximin": leximin_key}
```

```python
# src/optimization/scoring.py
import math
from typing import Mapping, Sequence

from models.evaluate import EXPECTED, REALIZED, EvaluationResult, evaluate
from models.paths import CandidatePath
from models.plan import Plan
from models.scenario import Scenario

from .objectives import Key, Objective, lexicographic_key

DESIGN_NAMESPACE = "design"
DEFAULT_DESIGN_IDS = (0, 1, 2, 3, 4)
DEFAULT_BUDGET = 2000
INFEASIBLE_SCORE: Key = (-math.inf,)


class BudgetExhausted(RuntimeError):
    """The scorer has used its whole evaluation budget."""


class DesignScorer:
    """Scores plans on design realizations, or on expected rates when design_ids is empty, under a call budget."""

    def __init__(
        self,
        scenario: Scenario,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        design_ids: Sequence[int] = DEFAULT_DESIGN_IDS,
        budget: int = DEFAULT_BUDGET,
        objective: Objective = lexicographic_key,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be at least 1")
        self.scenario = scenario
        self.candidates = candidates
        self.design_ids = tuple(design_ids)
        self.budget = budget
        self.objective = objective
        self.calls = 0

    @property
    def remaining(self) -> int:
        return self.budget - self.calls

    def score(self, plan: Plan) -> Key:
        return self._evaluate(plan, keep_ledger=False)[0]

    def score_with_ledger(self, plan: Plan) -> tuple[Key, EvaluationResult]:
        return self._evaluate(plan, keep_ledger=True)

    def _evaluate(self, plan: Plan, keep_ledger: bool) -> tuple[Key, EvaluationResult]:
        if self.calls >= self.budget:
            raise BudgetExhausted(f"the evaluation budget of {self.budget} calls is exhausted")
        self.calls += 1
        result = evaluate(
            self.scenario,
            plan,
            REALIZED if self.design_ids else EXPECTED,
            self.design_ids,
            candidates=self.candidates,
            keep_ledger=keep_ledger,
            channel_namespace=DESIGN_NAMESPACE,
        )
        key = self.objective(self.scenario, result) if result.feasible else INFEASIBLE_SCORE
        return key, result
```

- [ ] **Step 5: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/optimization/test_objectives_scoring.py -v`

Expected: 5 passed.

Run: `uv run --extra test pytest -q`

Expected: 130 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add src/optimization tests/support/search_builders.py tests/optimization/test_objectives_scoring.py
git commit -m "feat: add design scorer and search objectives"
```

### Task 3: Tour family

**Files:**
- Create: `src/optimization/tours.py`
- Test: `tests/optimization/test_tours.py`

**Interfaces:**
- Consumes: `models.scenario.Scenario` (`damage_zones`, `sources[].zone`, `uav.start_xy_m`, `uav.v_max_mps`, `time`); Task 2 `search_builders.three_zone_raw`; `baselines.trajectories.zone_tour_trajectory` (tests only).
- Produces: `optimization.tours.HOVER_SPLITS = ("uniform", "load", "urgency")`; `HOVER_POINTS = ("zone_center", "source_centroid")`; frozen dataclass `TourSpec(zones: tuple[int, ...], split: str, hover: str)`; `zone_orders(zone_count) -> list[tuple[int, ...]]`; `hover_point(scenario, zone, hover) -> np.ndarray`; `split_weights(scenario, zones, split) -> list[float]`; `hover_slots(spare, weights) -> list[int]`; `tour_trajectory(scenario, spec) -> np.ndarray | None` (`None` when travel exceeds $N$ slots); `tour_family(scenario) -> list[tuple[TourSpec, np.ndarray]]` in enumeration order (orders, then splits, then hover points), without duplicate trajectories.

- [ ] **Step 1: Write the failing tests**

```python
# tests/optimization/test_tours.py
import numpy as np

from baselines.trajectories import zone_tour_trajectory
from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, validate_plan
from models.scenario import scenario_from_dict
from optimization.tours import (
    TourSpec,
    hover_point,
    hover_slots,
    split_weights,
    tour_family,
    tour_trajectory,
    zone_orders,
)
from search_builders import three_zone_raw


def test_zone_orders_go_by_size_then_lexicographic():
    orders = zone_orders(3)
    assert len(orders) == 15
    assert orders[:7] == [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (0, 1)]
    assert orders[-3:] == [(0,), (1,), (2,)]


def test_family_skips_tours_that_exceed_the_mission_and_every_tour_is_valid():
    scenario = scenario_from_dict(three_zone_raw(num_slots=50))
    candidates = candidate_paths(scenario)
    family = tour_family(scenario)
    assert {spec.zones for spec, _ in family} == {(0, 1), (1, 0), (0,), (1,)}
    assert tour_trajectory(scenario, TourSpec((2,), "uniform", "zone_center")) is None
    for _, trajectory in family:
        plan = Plan(trajectory, {alert.id: None for alert in scenario.alerts}, EqualSplitBacklogged())
        assert validate_plan(scenario, candidates, plan) == []
    assert len({spec for spec, _ in family}) == len(family) <= 90


def test_hover_splits_follow_load_and_urgency():
    scenario = scenario_from_dict(three_zone_raw(num_slots=50))
    assert split_weights(scenario, (0, 1), "uniform") == [1.0, 1.0]
    assert split_weights(scenario, (0, 1), "load") == [1.0, 2.0]
    assert split_weights(scenario, (0, 1), "urgency") == [0.1, 0.05 + 0.025]
    assert split_weights(scenario, (2,), "load") == [1.0]
    assert hover_slots(27, [1.0, 1.0]) == [13, 13]
    assert hover_slots(27, [1.0, 2.0]) == [9, 18]
    assert hover_slots(27, [0.1, 0.075]) == [15, 11]
    trajectory = tour_trajectory(scenario, TourSpec((0, 1), "load", "zone_center"))
    at_first = np.all(np.isclose(trajectory, [1230.0, 1000.0]), axis=1).sum()
    at_second = np.all(np.isclose(trajectory, [1000.0, 1400.0]), axis=1).sum()
    assert (at_first, at_second) == (9 + 1, 18 + 1)


def test_hover_points_use_source_centroid_or_zone_center():
    raw = three_zone_raw()
    scenario = scenario_from_dict(raw)
    assert np.allclose(hover_point(scenario, 1, "source_centroid"), [1050.0, 1400.0])
    assert np.allclose(hover_point(scenario, 1, "zone_center"), [1000.0, 1400.0])
    for source in raw["sources"]:
        if source["zone"] == 1:
            source["zone"] = 0
    moved = scenario_from_dict(raw)
    assert np.allclose(hover_point(moved, 1, "source_centroid"), [1000.0, 1400.0])


def test_b1_tour_belongs_to_the_family():
    scenario = scenario_from_dict(three_zone_raw(num_slots=150))
    assert any(np.array_equal(trajectory, zone_tour_trajectory(scenario)) for _, trajectory in tour_family(scenario))
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/optimization/test_tours.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'optimization.tours'`.

- [ ] **Step 3: Implement the tour family**

```python
# src/optimization/tours.py
"""Tour family of the trajectory block: feasible by construction (spec C Section 7.3)."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
import math

import numpy as np

from models.scenario import Scenario

HOVER_SPLITS = ("uniform", "load", "urgency")
HOVER_POINTS = ("zone_center", "source_centroid")


@dataclass(frozen=True)
class TourSpec:
    zones: tuple[int, ...]
    split: str
    hover: str


def zone_orders(zone_count: int) -> list[tuple[int, ...]]:
    """Every non-empty subset in every order: subset size descending, then lexicographic."""
    orders = [order for size in range(1, zone_count + 1) for subset in combinations(range(zone_count), size) for order in permutations(subset)]
    return sorted(orders, key=lambda order: (-len(order), order))


def hover_point(scenario: Scenario, zone: int, hover: str) -> np.ndarray:
    if hover == "source_centroid":
        points = [source.xy for source in scenario.sources if source.zone == zone]
        if points:
            return np.mean(np.asarray(points, dtype=float), axis=0)
    elif hover != "zone_center":
        raise ValueError(f"unknown hover point {hover!r}")
    return np.array([scenario.damage_zones[zone].x_m, scenario.damage_zones[zone].y_m])


def split_weights(scenario: Scenario, zones: tuple[int, ...], split: str) -> list[float]:
    """Per-stop hover weights; a zero weight sum falls back to uniform."""
    if split == "uniform":
        return [1.0] * len(zones)
    if split not in HOVER_SPLITS:
        raise ValueError(f"unknown hover split {split!r}")
    zone_of = {source.id: source.zone for source in scenario.sources}
    weights = []
    for zone in zones:
        alerts = [alert for alert in scenario.alerts if zone_of[alert.source] == zone]
        if split == "load":
            weights.append(float(len(alerts)))
        else:
            weights.append(sum(1.0 / (alert.deadline_slot - alert.release_slot) for alert in alerts))
    return weights if sum(weights) > 0.0 else [1.0] * len(zones)


def hover_slots(spare: int, weights: list[float]) -> list[int]:
    total = sum(Fraction(weight) for weight in weights)
    return [math.floor(spare * Fraction(weight) / total) for weight in weights]


def tour_trajectory(scenario: Scenario, spec: TourSpec) -> np.ndarray | None:
    """Straight segments at V_max, hover at each stop, remaining slots at the center; None if travel exceeds N."""
    num_slots = scenario.time.num_slots
    step_m = scenario.uav.v_max_mps * scenario.time.slot_s
    center = np.asarray(scenario.uav.start_xy_m, dtype=float)
    stops = [center, *(hover_point(scenario, zone, spec.hover) for zone in spec.zones), center]
    travel = [math.ceil(float(np.linalg.norm(stop - start)) / step_m) for start, stop in zip(stops, stops[1:])]
    if sum(travel) > num_slots:
        return None
    hovers = hover_slots(num_slots - sum(travel), split_weights(scenario, spec.zones, spec.split))
    points = [center]
    for index, (start, stop, steps) in enumerate(zip(stops, stops[1:], travel)):
        points.extend(start + (stop - start) * step / steps for step in range(1, steps + 1))
        if index < len(spec.zones):
            points.extend([stop] * hovers[index])
    points.extend([center] * (num_slots + 1 - len(points)))
    return np.asarray(points)


def tour_family(scenario: Scenario) -> list[tuple[TourSpec, np.ndarray]]:
    """Feasible tours in enumeration order; a tour identical to an earlier one is dropped."""
    family: list[tuple[TourSpec, np.ndarray]] = []
    for zones in zone_orders(len(scenario.damage_zones)):
        for split in HOVER_SPLITS:
            for hover in HOVER_POINTS:
                spec = TourSpec(zones, split, hover)
                trajectory = tour_trajectory(scenario, spec)
                if trajectory is not None and not any(np.array_equal(trajectory, other) for _, other in family):
                    family.append((spec, trajectory))
    return family
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/optimization/test_tours.py -v`

Expected: 5 passed.

Run: `uv run --extra test pytest -q`

Expected: 135 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/optimization/tours.py tests/optimization/test_tours.py
git commit -m "feat: add feasible tour family"
```

### Task 4: Search state and blocks

**Files:**
- Create: `src/optimization/blocks.py`
- Test: `tests/optimization/test_blocks.py`

**Interfaces:**
- Consumes: Task 1 `WeightedBacklogged`; Task 2 `DesignScorer`, `INFEASIBLE_SCORE`, `Key`, `search_builders.{hopeless_raw, bandwidth_raw, three_zone_raw}`; Task 3 `TourSpec`, `tour_family`; `models.paths.radio_links`; `models.channel.BACKHAUL`.
- Produces: mutable dataclass `optimization.blocks.SearchState(trajectory, path_choice: dict[str, int | None], weights: dict[LinkId, np.ndarray], key=INFEASIBLE_SCORE)` with `plan() -> Plan` (a `WeightedBacklogged` plan) and `try_change(scorer, **changes) -> bool` (scores the changed state, adopts it only on a strictly greater key; never mutates arrays or dictionaries in place); `unit_weights(scenario, candidates) -> dict[LinkId, np.ndarray]` (ones for every radio link of the candidate set); `path_block(scenario, candidates, scorer, state) -> bool`; `links_by_load(scenario, candidates, path_choice) -> list[LinkId]`; `bandwidth_block(scenario, candidates, scorer, state) -> bool`; `trajectory_block(scorer, state, family) -> bool`. Blocks return whether they accepted anything and let `BudgetExhausted` propagate; accepted changes are already in `state`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/optimization/test_blocks.py
import numpy as np

from models.paths import candidate_paths
from models.plan import stationary_trajectory, validate_plan
from models.scenario import scenario_from_dict
from optimization.blocks import SearchState, bandwidth_block, links_by_load, path_block, trajectory_block, unit_weights
from optimization.scoring import DesignScorer
from optimization.tours import tour_family
from scenario_builders import make_alert
from search_builders import bandwidth_raw, hopeless_raw, three_zone_raw

FIRST_ACCESS = ("access", "s00", "n1")
SECOND_ACCESS = ("access", "s02", "n2")


def _start(raw, choice):
    scenario = scenario_from_dict(raw)
    candidates = candidate_paths(scenario)
    scorer = DesignScorer(scenario, candidates, design_ids=(), budget=500)
    state = SearchState(stationary_trajectory(scenario), dict(choice), unit_weights(scenario, candidates))
    state.key = scorer.score(state.plan())
    return scenario, candidates, scorer, state


def test_try_change_adopts_only_strict_improvements():
    _, _, scorer, state = _start(hopeless_raw(), {"alert-0000": 0, "alert-0001": 0})
    before = state.key
    assert not state.try_change(scorer, path_choice={"alert-0000": 0, "alert-0001": 0})
    assert state.key == before and scorer.calls == 2
    assert state.try_change(scorer, path_choice={"alert-0000": None, "alert-0001": 0})
    assert state.path_choice == {"alert-0000": None, "alert-0001": 0} and state.key > before


def test_path_block_drops_a_hopeless_alert_that_blocks_another():
    scenario, candidates, scorer, state = _start(hopeless_raw(), {"alert-0000": 0, "alert-0001": 0})
    assert state.key[0] == 0
    assert path_block(scenario, candidates, scorer, state)
    assert state.path_choice == {"alert-0000": None, "alert-0001": 0}
    assert state.key[0] == 1
    assert scorer.calls == 3


def test_trajectory_block_flies_to_a_source_only_the_uav_reaches():
    raw = three_zone_raw(num_slots=150)
    raw["alerts"] = [make_alert("alert-0000", "s01", 0, 150, 1000000)]
    scenario, candidates, scorer, state = _start(raw, {"alert-0000": 0})
    assert [path.kind for path in candidates["alert-0000"]] == ["uav", "uav", "uav"]
    assert state.key[0] == 0
    assert trajectory_block(scorer, state, tour_family(scenario))
    assert state.key[0] == 1
    assert np.min(np.linalg.norm(state.trajectory - [100.0, 100.0], axis=1)) < 1e-6
    assert validate_plan(scenario, candidates, state.plan()) == []


def test_bandwidth_block_raises_the_weight_of_the_congested_link():
    scenario, candidates, scorer, state = _start(bandwidth_raw(), {"alert-0000": 0, "alert-0001": 0})
    assert links_by_load(scenario, candidates, state.path_choice) == [FIRST_ACCESS, SECOND_ACCESS]
    assert state.key[0] == 1
    assert bandwidth_block(scenario, candidates, scorer, state)
    assert state.key[0] == 2
    assert np.all(state.weights[FIRST_ACCESS] == 2.0) and np.all(state.weights[SECOND_ACCESS] == 1.0)


def test_links_by_load_breaks_ties_by_link_and_skips_unserved_alerts():
    raw = bandwidth_raw()
    raw["alerts"][0]["size_bits"] = 4000000
    scenario = scenario_from_dict(raw)
    candidates = candidate_paths(scenario)
    assert links_by_load(scenario, candidates, {"alert-0000": 0, "alert-0001": 0}) == [FIRST_ACCESS, SECOND_ACCESS]
    assert links_by_load(scenario, candidates, {"alert-0000": None, "alert-0001": 0}) == [SECOND_ACCESS]
```

The instances were checked on expected rates: in `hopeless_raw` dropping `alert-0000` makes `alert-0001` timely (three scorer calls in total); in the three-zone instance only a tour over zone 2 delivers the far alert; in `bandwidth_raw` equal split leaves `alert-0000` late and doubling the weights of `(access, s00, n1)` makes both alerts timely.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/optimization/test_blocks.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'optimization.blocks'`.

- [ ] **Step 3: Implement the search state and blocks**

```python
# src/optimization/blocks.py
"""Search state and the path, bandwidth, and trajectory blocks of the evaluator-guided BCD (spec C Section 7.2)."""

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from models.channel import BACKHAUL, LinkId
from models.paths import CandidatePath, radio_links
from models.plan import Plan, WeightedBacklogged
from models.scenario import Scenario

from .objectives import Key
from .scoring import INFEASIBLE_SCORE, DesignScorer
from .tours import TourSpec


@dataclass
class SearchState:
    """The incumbent plan and its key; blocks change it only through try_change, so accepted work survives BudgetExhausted."""

    trajectory: np.ndarray
    path_choice: dict[str, int | None]
    weights: dict[LinkId, np.ndarray]
    key: Key = INFEASIBLE_SCORE

    def plan(self) -> Plan:
        return Plan(self.trajectory, dict(self.path_choice), WeightedBacklogged(dict(self.weights)))

    def try_change(self, scorer: DesignScorer, **changes: object) -> bool:
        """Score the state with `changes` applied and adopt it only if the key increases strictly."""
        trial = replace(self, **changes)
        key = scorer.score(trial.plan())
        if key <= self.key:
            return False
        self.trajectory, self.path_choice, self.weights, self.key = trial.trajectory, trial.path_choice, trial.weights, key
        return True


def unit_weights(scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]]) -> dict[LinkId, np.ndarray]:
    return {link: np.ones(scenario.time.num_slots) for link in radio_links(candidates)}


def path_block(
    scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]], scorer: DesignScorer, state: SearchState
) -> bool:
    """Alerts by (deadline, id); options None, 0, ..., K-1 except the current one, each tried from the incumbent."""
    accepted = False
    for alert in sorted(scenario.alerts, key=lambda item: (item.deadline_slot, item.id)):
        current = state.path_choice[alert.id]
        for option in (None, *range(len(candidates[alert.id]))):
            if option != current and state.try_change(scorer, path_choice={**state.path_choice, alert.id: option}):
                accepted = True
    return accepted


def links_by_load(
    scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]], path_choice: Mapping[str, int | None]
) -> list[LinkId]:
    """Radio links of chosen paths by total routed alert size, descending; ties by link."""
    load: dict[LinkId, int] = defaultdict(int)
    for alert in scenario.alerts:
        index = path_choice[alert.id]
        if index is not None:
            for link in candidates[alert.id][index].links:
                if link[0] != BACKHAUL:
                    load[link] += alert.size_bits
    return sorted(load, key=lambda link: (-load[link], link))


def bandwidth_block(
    scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]], scorer: DesignScorer, state: SearchState
) -> bool:
    """Per link: try doubling its weights, and halving them only if doubling was rejected."""
    accepted = False
    for link in links_by_load(scenario, candidates, state.path_choice):
        for factor in (2.0, 0.5):
            if state.try_change(scorer, weights={**state.weights, link: state.weights[link] * factor}):
                accepted = True
                break
    return accepted


def trajectory_block(
    scorer: DesignScorer, state: SearchState, family: Sequence[tuple[TourSpec, np.ndarray]]
) -> bool:
    accepted = False
    for _, trajectory in family:
        if not np.array_equal(trajectory, state.trajectory) and state.try_change(scorer, trajectory=trajectory):
            accepted = True
    return accepted
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/optimization/test_blocks.py -v`

Expected: 5 passed.

Run: `uv run --extra test pytest -q`

Expected: 140 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/optimization/blocks.py tests/optimization/test_blocks.py
git commit -m "feat: add search state and BCD blocks"
```

### Task 5: Deadline-aware repair

**Files:**
- Create: `src/optimization/repair.py`
- Test: `tests/optimization/test_repair.py`

**Interfaces:**
- Consumes: Task 2 `DesignScorer.score_with_ledger`; Task 4 `SearchState`, `unit_weights`; `models.ledger.Ledger` (`transfers` as `(slot, link, alert_id, bits)`, `capacity_bits` as `(slot, link, bits)`); `models.evaluate.EvaluationResult` (`realizations[].delivery_slot`, `timely`, `ledger`).
- Produces: `optimization.repair.RISK_SLACK_SLOTS = 2`, `AT_RISK_LIMIT = 10`, `CONGESTION_TOLERANCE = 1e-6`, `BUFFER_TOLERANCE_BITS = 1e-6`; frozen dataclass `AlertRisk(alert_id, risk: float, bottleneck: LinkId, congested_links: frozenset[LinkId])`; `slack_slots(alert, delivery_slot, timely) -> int`; `bottleneck_hop(scenario, alert, path, transfers) -> int` (`transfers` are the alert's `(slot, link, bits)`); `congested_slots(ledger) -> dict[LinkId, set[int]]`; `assess_risk(scenario, candidates, path_choice, result, backlog=True, slack_limit=2, limit=10) -> list[AlertRisk]` (ordered by `(-risk, deadline_slot, id)`; with `backlog=False` the bottleneck is the first radio hop, congestion is empty, and no ledger is read; with `backlog=True` missing ledgers raise `ValueError("... needs ledgers")`); `shift_bandwidth(weights, link, alert, num_slots) -> dict[LinkId, np.ndarray]`; `path_change_order(candidates, risk, current, backlog=True) -> list[int]`; `repair_round(scenario, candidates, scorer, state, operation1, operation2, backlog=True) -> bool` (at most $1 + RK$ evaluations).

- [ ] **Step 1: Write the failing tests**

```python
# tests/optimization/test_repair.py
import numpy as np
import pytest

from models.evaluate import EXPECTED, evaluate
from models.ledger import Ledger
from models.paths import CandidatePath, candidate_paths
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import scenario_from_dict
from optimization.blocks import SearchState, unit_weights
from optimization.repair import (
    AlertRisk,
    assess_risk,
    bottleneck_hop,
    congested_slots,
    path_change_order,
    repair_round,
    shift_bandwidth,
    slack_slots,
)
from optimization.scoring import DesignScorer
from scenario_builders import make_alert
from search_builders import base_scenario, hopeless_raw

ACCESS_LINK = ("access", "s00", "n1")
BACKHAUL_LINK = ("backhaul", "n1", "n0")


def test_slack_counts_spare_slots_and_marks_late_alerts():
    alert = base_scenario([make_alert("alert-0000", "s00", 0, 10, 100)]).alerts[0]
    assert slack_slots(alert, 7, True) == 2
    assert slack_slots(alert, 9, True) == 0
    assert slack_slots(alert, None, False) == -1


def test_bottleneck_is_the_hop_holding_bits_longest_with_ties_to_the_earliest():
    scenario = base_scenario([make_alert("alert-0000", "s00", 0, 6, 100)])
    alert = scenario.alerts[0]
    links = (ACCESS_LINK, BACKHAUL_LINK, ("backhaul", "n0", "n2"))
    path = CandidatePath("ground", "n1", links, 1.0)
    second_holds_longest = [(0, links[0], 100.0), (1, links[1], 50.0), (2, links[2], 50.0), (4, links[1], 50.0), (5, links[2], 50.0)]
    assert bottleneck_hop(scenario, alert, path, second_holds_longest) == 1
    one_slot_each = [(0, links[0], 100.0), (1, links[1], 100.0), (2, links[2], 100.0)]
    assert bottleneck_hop(scenario, alert, path, one_slot_each) == 0


def test_congested_links_carry_their_whole_capacity():
    other = ("access", "s01", "n0")
    ledger = Ledger(
        realization_id=0,
        transfers=((0, ACCESS_LINK, "a", 6.0), (0, ACCESS_LINK, "b", 4.0), (1, ACCESS_LINK, "a", 3.0)),
        bandwidth=(),
        capacity_bits=((0, ACCESS_LINK, 10.0), (1, ACCESS_LINK, 10.0), (1, other, 0.0)),
        drops=(),
    )
    assert congested_slots(ledger) == {ACCESS_LINK: {0}, other: {1}}


def test_operation1_doubles_weights_only_inside_the_alert_window():
    scenario = base_scenario([make_alert("alert-0000", "s00", 3, 8, 100)])
    weights = unit_weights(scenario, candidate_paths(scenario))
    shifted = shift_bandwidth(weights, ACCESS_LINK, scenario.alerts[0], scenario.time.num_slots)
    expected = np.ones(20)
    expected[3:8] = 2.0
    assert np.array_equal(shifted[ACCESS_LINK], expected)
    assert np.all(weights[ACCESS_LINK] == 1.0)
    assert all(shifted[link] is weights[link] for link in weights if link != ACCESS_LINK)


def test_operation2_orders_candidates_by_congestion_then_cost():
    downlink = ("downlink", "uav", "n1")
    paths = (
        CandidatePath("ground", "n1", (ACCESS_LINK, BACKHAUL_LINK), 0.1),
        CandidatePath("ground", "n1", (ACCESS_LINK, BACKHAUL_LINK), 1.0),
        CandidatePath("ground", "n2", (("access", "s00", "n2"), ("backhaul", "n2", "n0")), 5.0),
        CandidatePath("uav", "n1", (("access", "s00", "uav"), downlink, BACKHAUL_LINK), 0.5),
    )
    candidates = {"alert-0000": paths}
    risk = AlertRisk("alert-0000", 1.0, ACCESS_LINK, frozenset({ACCESS_LINK, BACKHAUL_LINK}))
    assert path_change_order(candidates, risk, 0) == [2, 3, 1]
    assert path_change_order(candidates, risk, None) == [2, 3, 0, 1]
    assert path_change_order(candidates, risk, 0, backlog=False) == [3, 1, 2]


def _hopeless_result(keep_ledger=True):
    scenario = scenario_from_dict(hopeless_raw())
    candidates = candidate_paths(scenario)
    choice = {"alert-0000": 0, "alert-0001": 0}
    plan = Plan(stationary_trajectory(scenario), choice, EqualSplitBacklogged())
    return scenario, candidates, choice, evaluate(scenario, plan, EXPECTED, candidates=candidates, keep_ledger=keep_ledger)


def test_risk_ranks_served_alerts_and_finds_backhaul_bottlenecks():
    scenario, candidates, choice, result = _hopeless_result()
    risks = assess_risk(scenario, candidates, choice, result)
    assert [(item.alert_id, item.risk, item.bottleneck) for item in risks] == [
        ("alert-0000", 1.0, BACKHAUL_LINK),
        ("alert-0001", 1.0, ACCESS_LINK),
    ]
    assert BACKHAUL_LINK in risks[0].congested_links
    assert [item.alert_id for item in assess_risk(scenario, candidates, choice, result, limit=1)] == ["alert-0000"]
    unserved = {"alert-0000": None, "alert-0001": 0}
    assert [item.alert_id for item in assess_risk(scenario, candidates, unserved, result)] == ["alert-0001"]


def test_no_backlog_risk_ignores_ledgers_and_targets_the_first_radio_hop():
    scenario, candidates, choice, result = _hopeless_result(keep_ledger=False)
    risks = assess_risk(scenario, candidates, choice, result, backlog=False)
    assert [(item.alert_id, item.bottleneck, item.congested_links) for item in risks] == [
        ("alert-0000", ACCESS_LINK, frozenset()),
        ("alert-0001", ACCESS_LINK, frozenset()),
    ]
    with pytest.raises(ValueError, match="ledgers"):
        assess_risk(scenario, candidates, choice, result)


def test_operation1_is_skipped_for_backhaul_bottlenecks():
    scenario = scenario_from_dict(hopeless_raw())
    candidates = candidate_paths(scenario)
    scorer = DesignScorer(scenario, candidates, design_ids=(), budget=50)
    state = SearchState(stationary_trajectory(scenario), {"alert-0000": 0, "alert-0001": 0}, unit_weights(scenario, candidates))
    state.key = scorer.score(state.plan())
    assert not repair_round(scenario, candidates, scorer, state, operation1=True, operation2=False)
    # initial score, one detection call, and operation 1 for alert-0001 only: alert-0000 waits on the backhaul
    assert scorer.calls == 3
```

In `hopeless_raw` the access link carries 1.5 Mbit per slot and the backhaul 1 Mbit per slot: the 12 Mbit alert waits longest on the backhaul, while `alert-0001` waits behind it on the access link for its whole window.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/optimization/test_repair.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'optimization.repair'`.

- [ ] **Step 3: Implement risk detection and the repair operations**

```python
# src/optimization/repair.py
"""Risk detection and the two repair operations of method P (spec C Section 9)."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from models.channel import BACKHAUL, LinkId
from models.evaluate import EvaluationResult
from models.ledger import Ledger
from models.paths import CandidatePath
from models.scenario import Alert, Scenario

from .blocks import SearchState
from .scoring import DesignScorer

RISK_SLACK_SLOTS = 2
AT_RISK_LIMIT = 10
CONGESTION_TOLERANCE = 1e-6
BUFFER_TOLERANCE_BITS = 1e-6


@dataclass(frozen=True)
class AlertRisk:
    alert_id: str
    risk: float
    bottleneck: LinkId
    congested_links: frozenset[LinkId]


def slack_slots(alert: Alert, delivery_slot: int | None, timely: bool) -> int:
    return alert.deadline_slot - 1 - delivery_slot if timely and delivery_slot is not None else -1


def bottleneck_hop(scenario: Scenario, alert: Alert, path: CandidatePath, transfers) -> int:
    """Hop whose buffer held bits of the alert at the start of the most slots in [r_a, d_a); ties go to the earliest hop."""
    num_slots = scenario.time.num_slots
    hop_of = {link: hop for hop, link in enumerate(path.links)}
    net = np.zeros((len(path.links), num_slots))
    for slot, link, bits in transfers:
        hop = hop_of[link]
        net[hop, slot] -= bits
        if hop + 1 < len(path.links):
            net[hop + 1, slot] += bits
    held = np.zeros_like(net)
    held[:, 1:] = np.cumsum(net, axis=1)[:, :-1]
    held[0, alert.release_slot:] += alert.size_bits
    window = held[:, alert.release_slot : min(alert.deadline_slot, num_slots)]
    return int(np.argmax((window > BUFFER_TOLERANCE_BITS).sum(axis=1)))


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


def assess_risk(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    path_choice: Mapping[str, int | None],
    result: EvaluationResult,
    backlog: bool = True,
    slack_limit: int = RISK_SLACK_SLOTS,
    limit: int = AT_RISK_LIMIT,
) -> list[AlertRisk]:
    """At-risk served alerts ordered by (-risk, deadline, id), truncated to `limit`."""
    realizations = result.realizations
    ranked = []
    for alert in scenario.alerts:
        if path_choice[alert.id] is None:
            continue
        risky = sum(
            slack_slots(alert, item.delivery_slot[alert.id], item.timely[alert.id]) <= slack_limit for item in realizations
        )
        if risky:
            ranked.append((-risky / len(realizations), alert.deadline_slot, alert.id))
    ranked = sorted(ranked)[:limit]
    if not backlog:
        return [
            AlertRisk(alert_id, -negative, candidates[alert_id][path_choice[alert_id]].links[0], frozenset())
            for negative, _, alert_id in ranked
        ]
    selected = {alert_id for _, _, alert_id in ranked}
    transfers = []
    congestion = []
    for item in realizations:
        if item.ledger is None:
            raise ValueError("risk detection with backlog information needs ledgers")
        by_alert: dict[str, list[tuple[int, LinkId, float]]] = defaultdict(list)
        for slot, link, alert_id, bits in item.ledger.transfers:
            if alert_id in selected:
                by_alert[alert_id].append((slot, link, bits))
        transfers.append(by_alert)
        congestion.append(congested_slots(item.ledger))
    risks = []
    for negative, _, alert_id in ranked:
        alert = scenario.alert_by_id[alert_id]
        path = candidates[alert_id][path_choice[alert_id]]
        hops = Counter(bottleneck_hop(scenario, alert, path, by_alert[alert_id]) for by_alert in transfers)
        hop = min(hops, key=lambda item: (-hops[item], item))
        window = set(range(alert.release_slot, alert.deadline_slot))
        congested = frozenset(link for slots_by_link in congestion for link, slots in slots_by_link.items() if slots & window)
        risks.append(AlertRisk(alert_id, -negative, path.links[hop], congested))
    return risks


def shift_bandwidth(
    weights: Mapping[LinkId, np.ndarray], link: LinkId, alert: Alert, num_slots: int
) -> dict[LinkId, np.ndarray]:
    """Operation 1: double the link's weights in slots [r_a, d_a) only."""
    values = weights[link].copy()
    values[alert.release_slot : min(alert.deadline_slot, num_slots)] *= 2.0
    return {**weights, link: values}


def path_change_order(
    candidates: Mapping[str, tuple[CandidatePath, ...]], risk: AlertRisk, current: int | None, backlog: bool = True
) -> list[int]:
    """Operation 2: other candidates by (congested links, cost_s, index); without backlog by (cost_s, index)."""
    paths = candidates[risk.alert_id]

    def order(index: int) -> tuple:
        congested = sum(link in risk.congested_links for link in paths[index].links) if backlog else 0
        return (congested, paths[index].cost_s, index)

    return sorted((index for index in range(len(paths)) if index != current), key=order)


def repair_round(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    scorer: DesignScorer,
    state: SearchState,
    operation1: bool,
    operation2: bool,
    backlog: bool = True,
) -> bool:
    """One detection call, then operation 1 and operation 2 per at-risk alert; at most 1 + R*K evaluations."""
    _, result = scorer.score_with_ledger(state.plan())
    accepted = False
    for risk in assess_risk(scenario, candidates, state.path_choice, result, backlog):
        alert = scenario.alert_by_id[risk.alert_id]
        if operation1 and risk.bottleneck[0] != BACKHAUL:
            weights = shift_bandwidth(state.weights, risk.bottleneck, alert, scenario.time.num_slots)
            accepted |= state.try_change(scorer, weights=weights)
        if operation2:
            for index in path_change_order(candidates, risk, state.path_choice[alert.id], backlog):
                if state.try_change(scorer, path_choice={**state.path_choice, alert.id: index}):
                    accepted = True
                    break
    return accepted
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/optimization/test_repair.py -v`

Expected: 8 passed.

Run: `uv run --extra test pytest -q`

Expected: 148 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/optimization/repair.py tests/optimization/test_repair.py
git commit -m "feat: add deadline-aware repair operations"
```

### Task 6: Search methods B2, B3, P and ablation variants

**Files:**
- Create: `src/optimization/bcd.py`
- Test: `tests/optimization/test_search_methods.py`

**Interfaces:**
- Consumes: `baselines.b1.B1` (initial plan); Task 2 `DesignScorer`, `BudgetExhausted`, `DEFAULT_BUDGET`, `DEFAULT_DESIGN_IDS`, `OBJECTIVES`; Task 3 `tour_family`; Task 4 `SearchState`, `unit_weights`, blocks; Task 5 `repair_round`; `models.evaluate.EvaluationCounter`.
- Produces: `optimization.bcd.EXPECTED_DESIGN = "expected"`; frozen dataclass `SearchParams(objective="lex", path_block=True, bandwidth_block=True, trajectory_block=True, operation1=False, operation2=False, backlog=True, design_ids=DEFAULT_DESIGN_IDS, budget=DEFAULT_BUDGET, max_iterations=10)` validated in `__post_init__` (messages name the field) with `from_mapping(raw) -> SearchParams` (`"expected"` maps to `()`; unknown keys raise `ValueError("unknown search parameters ...")`); frozen dataclass `SearchOutcome(plan, key, calls, iteration_keys: tuple[Key, ...], budget_exhausted: bool)`; frozen dataclass `SearchMethod(name, params=SearchParams(), uses_uav=True)` with `search(scenario, candidates) -> SearchOutcome`, `plan(scenario, candidates, counter) -> Plan` (adds the scorer calls to `counter.calls`), and `trajectory(scenario)` raising `NotImplementedError`.

Spec Section 9.4 variants as `SearchParams` (defaults are B2); plan C.2 writes them into its configurations:

| Variant | Parameters |
|---|---|
| B2 | none |
| B3 | `objective: leximin` |
| P | `operation1: true, operation2: true` |
| `P_op1` | `operation1: true` |
| `P_op2` | `operation2: true` |
| `P_fixed_paths` | `path_block: false, operation1: true` |
| `P_fixed_trajectory` | `trajectory_block: false, operation1: true, operation2: true` |
| `P_equal_bandwidth` | `bandwidth_block: false, operation2: true` |
| `P_no_backlog` | `operation1: true, operation2: true, backlog: false` |
| `P_expected_design` | `operation1: true, operation2: true, design_ids: expected` |

- [ ] **Step 1: Write the failing tests**

```python
# tests/optimization/test_search_methods.py
from dataclasses import replace

import numpy as np
import pytest

from models.evaluate import REALIZED, EvaluationCounter, evaluate
from models.paths import candidate_paths
from models.scenario import scenario_from_dict
from optimization import bcd
from optimization.bcd import SearchMethod, SearchParams
from optimization.objectives import lexicographic_key
from optimization.scoring import DesignScorer
from scenario_builders import make_alert
from search_builders import three_zone_raw

SMALL = {"design_ids": (0, 1), "budget": 150, "max_iterations": 3}
VARIANTS = {
    "B2": SearchParams(**SMALL),
    "B3": SearchParams(objective="leximin", **SMALL),
    "P": SearchParams(operation1=True, operation2=True, **SMALL),
    "P_no_backlog": SearchParams(operation1=True, operation2=True, backlog=False, **SMALL),
    "P_expected_design": SearchParams(operation1=True, operation2=True, **{**SMALL, "design_ids": ()}),
}


def _scenario():
    raw = three_zone_raw(num_slots=50)
    raw["alerts"] += [make_alert("alert-0003", "s01", 2, 48, 400000), make_alert("alert-0004", "s00", 1, 8, 2500000)]
    return scenario_from_dict(raw)


def _same_plan(first, second):
    weights_a, weights_b = first.bandwidth.weights, second.bandwidth.weights
    return (
        np.array_equal(first.trajectory, second.trajectory)
        and first.path_choice == second.path_choice
        and weights_a.keys() == weights_b.keys()
        and all(np.array_equal(weights_a[link], weights_b[link]) for link in weights_a)
    )


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_search_is_monotone_within_budget_feasible_and_repeatable(name):
    scenario = _scenario()
    candidates = candidate_paths(scenario)
    params = VARIANTS[name]
    method = SearchMethod(name, params)
    outcome = method.search(scenario, candidates)
    assert list(outcome.iteration_keys) == sorted(outcome.iteration_keys)
    assert 1 <= outcome.calls <= params.budget
    if not outcome.budget_exhausted:
        assert outcome.iteration_keys[-1] == outcome.key
    rescorer = DesignScorer(scenario, candidates, params.design_ids, 1, bcd.OBJECTIVES[params.objective])
    assert rescorer.score(outcome.plan) == outcome.key
    assert evaluate(scenario, outcome.plan, REALIZED, (0, 1, 2), candidates=candidates).feasible
    assert _same_plan(outcome.plan, method.search(scenario, candidates).plan)


def test_budget_exhaustion_returns_the_feasible_incumbent():
    scenario = _scenario()
    candidates = candidate_paths(scenario)
    params = SearchParams(operation1=True, operation2=True, design_ids=(0,), budget=4)
    outcome = SearchMethod("P", params).search(scenario, candidates)
    assert outcome.budget_exhausted and outcome.calls == 4 and outcome.iteration_keys == ()
    assert evaluate(scenario, outcome.plan, REALIZED, (0,), candidates=candidates).feasible


def test_p_without_operations_is_b2_and_b3_differs_only_by_its_key(monkeypatch):
    scenario = _scenario()
    candidates = candidate_paths(scenario)
    b2 = SearchMethod("B2", VARIANTS["B2"]).search(scenario, candidates)
    p_off = SearchMethod("P", replace(VARIANTS["P"], operation1=False, operation2=False)).search(scenario, candidates)
    assert _same_plan(b2.plan, p_off.plan) and b2.calls == p_off.calls
    monkeypatch.setitem(bcd.OBJECTIVES, "leximin", lexicographic_key)
    b3 = SearchMethod("B3", VARIANTS["B3"]).search(scenario, candidates)
    assert _same_plan(b2.plan, b3.plan) and b3.key == b2.key


def test_plan_reports_calls_and_trajectory_needs_planning():
    scenario = _scenario()
    candidates = candidate_paths(scenario)
    method = SearchMethod("B2", VARIANTS["B2"])
    counter = EvaluationCounter(calls=1)
    method.plan(scenario, candidates, counter)
    assert counter.calls == 1 + method.search(scenario, candidates).calls
    with pytest.raises(NotImplementedError):
        method.trajectory(scenario)


def test_search_params_from_mapping_validates_values():
    assert SearchParams.from_mapping({}) == SearchParams()
    assert SearchParams.from_mapping({"design_ids": "expected", "operation1": True}).design_ids == ()
    assert SearchParams.from_mapping({"design_ids": [3, 4]}).design_ids == (3, 4)
    for raw, message in (
        ({"repair": True}, "unknown search parameters"),
        ({"objective": "maxmin"}, "objective"),
        ({"budget": 0}, "budget"),
        ({"operation1": "yes"}, "operation1"),
        ({"design_ids": []}, "design_ids"),
        ({"design_ids": [1, 1]}, "design_ids"),
    ):
        with pytest.raises(ValueError, match=message):
            SearchParams.from_mapping(raw)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/optimization/test_search_methods.py -v`

Expected: collection fails with `ImportError: cannot import name 'bcd' from 'optimization'`.

- [ ] **Step 3: Implement the search method**

```python
# src/optimization/bcd.py
"""SearchMethod: B2, B3, P and the ablation variants as parameter sets of one evaluator-guided BCD."""

from dataclasses import dataclass, field, fields
from typing import Mapping

import numpy as np

from baselines.b1 import B1
from models.evaluate import EvaluationCounter
from models.paths import CandidatePath
from models.plan import Plan
from models.scenario import Scenario

from .blocks import SearchState, bandwidth_block, path_block, trajectory_block, unit_weights
from .objectives import OBJECTIVES, Key
from .repair import repair_round
from .scoring import DEFAULT_BUDGET, DEFAULT_DESIGN_IDS, BudgetExhausted, DesignScorer
from .tours import tour_family

EXPECTED_DESIGN = "expected"


@dataclass(frozen=True)
class SearchParams:
    """Defaults are method B2; an empty design_ids tuple scores expected rates."""

    objective: str = "lex"
    path_block: bool = True
    bandwidth_block: bool = True
    trajectory_block: bool = True
    operation1: bool = False
    operation2: bool = False
    backlog: bool = True
    design_ids: tuple[int, ...] = DEFAULT_DESIGN_IDS
    budget: int = DEFAULT_BUDGET
    max_iterations: int = 10

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective: expected one of {sorted(OBJECTIVES)}, got {self.objective!r}")
        for name in ("path_block", "bandwidth_block", "trajectory_block", "operation1", "operation2", "backlog"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name}: expected true or false")
        for name in ("budget", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}: expected an integer of at least 1")
        ids = self.design_ids
        if not isinstance(ids, tuple) or len(set(ids)) != len(ids) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ids
        ):
            raise ValueError("design_ids: expected distinct non-negative integers or 'expected'")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "SearchParams":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown search parameters {unknown}")
        values = dict(raw)
        if "design_ids" in values:
            ids = values["design_ids"]
            if ids == EXPECTED_DESIGN:
                values["design_ids"] = ()
            elif isinstance(ids, (list, tuple)) and ids:
                values["design_ids"] = tuple(ids)
            else:
                raise ValueError("design_ids: expected a non-empty list or 'expected'")
        return cls(**values)


@dataclass(frozen=True)
class SearchOutcome:
    plan: Plan
    key: Key
    calls: int
    iteration_keys: tuple[Key, ...]
    budget_exhausted: bool


@dataclass(frozen=True)
class SearchMethod:
    """Starts from B1 with unit weights; iterates blocks and repair until an iteration accepts nothing."""

    name: str
    params: SearchParams = field(default_factory=SearchParams)
    uses_uav: bool = True

    def trajectory(self, scenario: Scenario) -> np.ndarray:
        raise NotImplementedError(f"{self.name}: the trajectory of a search method exists only after planning")

    def plan(
        self,
        scenario: Scenario,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        counter: EvaluationCounter,
    ) -> Plan:
        outcome = self.search(scenario, candidates)
        counter.calls += outcome.calls
        return outcome.plan

    def search(self, scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]]) -> SearchOutcome:
        params = self.params
        scorer = DesignScorer(scenario, candidates, params.design_ids, params.budget, OBJECTIVES[params.objective])
        initial = B1().plan(scenario, candidates, EvaluationCounter())
        state = SearchState(initial.trajectory, dict(initial.path_choice), unit_weights(scenario, candidates))
        family = tour_family(scenario) if params.trajectory_block else []
        iteration_keys: list[Key] = []
        exhausted = False
        try:
            state.key = scorer.score(state.plan())
            for _ in range(params.max_iterations):
                accepted = False
                if params.path_block:
                    accepted |= path_block(scenario, candidates, scorer, state)
                if params.bandwidth_block:
                    accepted |= bandwidth_block(scenario, candidates, scorer, state)
                if params.trajectory_block:
                    accepted |= trajectory_block(scorer, state, family)
                if params.operation1 or params.operation2:
                    accepted |= repair_round(
                        scenario, candidates, scorer, state, params.operation1, params.operation2, params.backlog
                    )
                iteration_keys.append(state.key)
                if not accepted:
                    break
        except BudgetExhausted:
            exhausted = True
        return SearchOutcome(state.plan(), state.key, scorer.calls, tuple(iteration_keys), exhausted)
```

One iteration applies the enabled blocks in the order path, bandwidth, trajectory, then a repair round when either operation is on; the search stops after an iteration that accepts nothing, after `max_iterations`, or when the budget is exhausted.

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/optimization/test_search_methods.py -v`

Expected: 9 passed.

Run: `uv run --extra test pytest -q`

Expected: 157 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/optimization/bcd.py tests/optimization/test_search_methods.py
git commit -m "feat: add search methods B2, B3, P"
```

### Task 7: Runner method specs, in-task bounds, and union bound

**Files:**
- Create: `src/runner/methods.py`, `experiments/run_phase2.py`, `configs/experiments/phase2_pilot.yaml`
- Modify: `src/runner/config.py`, `src/runner/tasks.py`, `src/runner/aggregate.py`, `src/runner/experiment.py`, `src/runner/provenance.py`, `.gitignore`
- Modify tests: `tests/runner/test_runner_config.py`, `tests/runner/test_runner_aggregate.py`, `tests/runner/experiment_fixtures.py`, `tests/runner/test_runner_experiment.py`
- Test: `tests/runner/test_runner_phase1_keys.py`

**Interfaces:**
- Consumes: Task 6 `SearchMethod`, `SearchParams.from_mapping`; `baselines.{METHODS, get_method}`; `bounds.formulation.build_bound_model`, `bounds.solve.solve_lp`; sub-project B runner.
- Produces: `runner.config.SEARCH_BASE = "search"`; frozen dataclass `MethodSpec(id, base, params: tuple[tuple[str, object], ...] = (), bounds: bool = False)` with `search_params() -> SearchParams`; `parse_method_spec(raw, index) -> MethodSpec`; `ExperimentConfig.methods: tuple[MethodSpec, ...]`; `ExperimentConfig.crosscheck: CrosscheckConfig | None`. `runner.methods.GROUND_ONLY = "ground_only"`, `TRAJECTORY = "trajectory"`, `build_method(spec) -> Method`, `bound_owner(spec) -> tuple[str, str] | None`. `runner.tasks.MethodTask(set_id, scenario_id, scenario_path, spec: MethodSpec, realization_ids, tangents)` with the unchanged key `method__{set_id}__{scenario_id}__{spec.id}`; `run_method_task(task) -> tuple[list[dict], list[dict]]`; `bound_row(set_id, scenario_id, scenario, candidates, variant, trajectory_method, trajectory, realization_id, tangents) -> dict`; `execute(task)` shards carry `bound_rows`; `runner.tasks.bound_variant` is removed. `runner.aggregate.validity_violations(method_rows, bound_rows, owners)`, `union_bound_ratios(bound_rows) -> dict[tuple[str, str, int], float]`, `summarize(method_rows, bound_rows, bootstrap, owners)`; `runner.experiment.SUMMARY_COLUMNS` ends with `union_bound_ratio_mean, union_gap_mean, evaluate_calls_mean`. `write_tiny_experiment(root, output_dir, workers=2, methods=("B0", "B1"), crosscheck=True)`.

- [ ] **Step 1: Update and add the runner tests**

Replace the whole content of `tests/runner/test_runner_config.py` with:

```python
# tests/runner/test_runner_config.py
from pathlib import Path

import pytest
import yaml

from optimization.bcd import SearchParams
from runner.config import MethodSpec, load_experiment_config
from runner.methods import bound_owner, build_method


def test_phase1_config_values():
    config = load_experiment_config(Path("configs/experiments/phase1.yaml"))
    assert [item.set_id for item in config.scenario_sets] == ["v0", "v0-bh50"]
    assert config.realization_ids == tuple(range(30))
    assert config.methods == (MethodSpec("B0", "B0"), MethodSpec("B1", "B1"))
    assert (config.tangents, config.workers, config.crosscheck.alert_count, config.crosscheck.time_limit_s) == (10, 12, 10, 300.0)
    assert (config.bootstrap.resamples, config.bootstrap.seed) == (10000, 20260911)
    assert len(config.config_hash) == 64


def test_phase2_pilot_config_values():
    config = load_experiment_config(Path("configs/experiments/phase2_pilot.yaml"))
    assert (config.split, config.crosscheck, config.output_dir) == ("dev", None, Path("results/phase2/pilot"))
    assert [item.set_id for item in config.scenario_sets] == ["v0"]
    assert [(spec.id, spec.base, spec.bounds) for spec in config.methods] == [
        ("B1", "B1", False), ("B2", "search", True), ("B3", "search", True), ("P", "search", True),
    ]
    assert [spec.search_params() for spec in config.methods[1:]] == [
        SearchParams(), SearchParams(objective="leximin"), SearchParams(operation1=True, operation2=True),
    ]


def _write(tmp_path: Path, change) -> Path:
    raw = yaml.safe_load(Path("configs/experiments/phase1.yaml").read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_method_specs_accept_names_and_search_mappings(tmp_path: Path):
    methods = [
        "B0",
        {"id": "B1", "base": "B1"},
        {"id": "P_expected_design", "base": "search", "params": {"operation2": True, "operation1": True, "design_ids": "expected"}, "bounds": True},
        {"id": "P_design10", "base": "search", "params": {"design_ids": list(range(10))}},
    ]
    config = load_experiment_config(_write(tmp_path, lambda raw: raw.update(methods=methods)))
    assert config.methods[:2] == (MethodSpec("B0", "B0"), MethodSpec("B1", "B1"))
    expected = config.methods[2]
    assert expected == MethodSpec("P_expected_design", "search", (("design_ids", "expected"), ("operation1", True), ("operation2", True)), True)
    assert expected.search_params() == SearchParams(operation1=True, operation2=True, design_ids=())
    assert config.methods[3].params == (("design_ids", tuple(range(10))),) and config.methods[3].bounds is False
    assert build_method(expected).name == "P_expected_design"
    assert [bound_owner(spec) for spec in config.methods] == [
        ("ground_only", ""), ("trajectory", "B1"), ("trajectory", "P_expected_design"), None,
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(methods=["B9"]), "methods"),
        (lambda raw: raw.update(methods=["B1", "B1"]), "unique"),
        (lambda raw: raw.update(methods=[{"id": "X", "base": "B7"}]), "unknown base"),
        (lambda raw: raw.update(methods=[{"id": "B1", "base": "B1", "params": {"budget": 5}}]), "takes no"),
        (lambda raw: raw.update(methods=[{"id": "B1", "base": "search"}]), "not name a baseline"),
        (lambda raw: raw.update(methods=[{"id": "P__x", "base": "search"}]), "single '_'"),
        (lambda raw: raw.update(methods=[{"id": "P", "base": "search", "params": {"repair": True}}]), "unknown search parameters"),
        (lambda raw: raw.update(methods=[{"id": "P", "base": "search", "bounds": "yes"}]), "bounds"),
        (lambda raw: raw["scenario_sets"].append(dict(raw["scenario_sets"][0])), "unique"),
        (lambda raw: raw["milp_crosscheck"].update(set_id="missing"), "milp_crosscheck.set_id"),
        (lambda raw: raw.update(realization_ids={"start": 3, "stop": 3}), "realization_ids"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, change, message):
    with pytest.raises(ValueError, match=message):
        load_experiment_config(_write(tmp_path, change))
```

In `tests/runner/test_runner_aggregate.py` (2 edits):

Edit 1: replace

```python
    summarize,
    validity_violations,
)
from runner.config import BootstrapConfig
```

with

```python
    summarize,
    union_bound_ratios,
    validity_violations,
)
from runner.config import BootstrapConfig

OWNERS = {"B0": ("ground_only", "")}
```

Edit 2: replace

```python
    bound_rows = [_bound_row("A", 0, 4.0), _bound_row("A", 1, 4.0), _bound_row("B", 0, 0.0), _bound_row("B", 1, 0.0)]
    [row] = summarize(method_rows, bound_rows, BootstrapConfig(100, 3, 0.95))
    assert row["timely_ratio_mean"] == pytest.approx((0.75 + 0.0) / 2)
    assert row["bound_ratio_mean"] == pytest.approx(0.5)
    assert row["gap_mean"] == pytest.approx(0.25)
    assert row["gap_undefined_count"] == 1
    assert row["shortfall_mean"] == pytest.approx((1.0 + 4.0) / 2)
    assert validity_violations(method_rows, bound_rows) == []


def test_validity_and_crosscheck_violations_are_reported():
    method_rows = [_method_row("A", "T1", 0, 3)]
    assert "exceed the bound" in validity_violations(method_rows, [_bound_row("A", 0, 2.0)])[0]
    assert "missing ground_only bound" in validity_violations(method_rows, [])[0]
    bad_status = dict(_bound_row("A", 0, 5.0), status="status 2: infeasible")
    assert any("infeasible" in item for item in validity_violations(method_rows, [bad_status]))
    good = {"set_id": "s", "scenario_id": "A", "lp_ub": 5.0, "lp_status": "optimal", "milp_incumbent": 4.0,
            "milp_dual_bound": 4.5, "milp_status": "time_limit", "milp_plan_value": 3.0, "method_value": 2.0}
    assert crosscheck_violations([good]) == []
    assert "chain broken" in crosscheck_violations([dict(good, milp_plan_value=5.0)])[0]
    assert "chain broken" in crosscheck_violations([dict(good, method_value=4.6)])[0]
    assert "incomplete" in crosscheck_violations([dict(good, milp_plan_value=math.nan)])[0]
```

with

```python
    bound_rows = [_bound_row("A", 0, 4.0), _bound_row("A", 1, 4.0), _bound_row("B", 0, 0.0), _bound_row("B", 1, 0.0)]
    [row] = summarize(method_rows, bound_rows, BootstrapConfig(100, 3, 0.95), OWNERS)
    assert row["timely_ratio_mean"] == pytest.approx((0.75 + 0.0) / 2)
    assert row["bound_ratio_mean"] == pytest.approx(0.5)
    assert row["gap_mean"] == pytest.approx(0.25)
    assert row["gap_undefined_count"] == 1
    assert row["shortfall_mean"] == pytest.approx((1.0 + 4.0) / 2)
    assert validity_violations(method_rows, bound_rows, OWNERS) == []


def test_validity_and_crosscheck_violations_are_reported():
    method_rows = [_method_row("A", "T1", 0, 3)]
    assert "exceed the bound" in validity_violations(method_rows, [_bound_row("A", 0, 2.0)], OWNERS)[0]
    assert "missing ground_only bound" in validity_violations(method_rows, [], OWNERS)[0]
    bad_status = dict(_bound_row("A", 0, 5.0), status="status 2: infeasible")
    assert any("infeasible" in item for item in validity_violations(method_rows, [bad_status], OWNERS))
    good = {"set_id": "s", "scenario_id": "A", "lp_ub": 5.0, "lp_status": "optimal", "milp_incumbent": 4.0,
            "milp_dual_bound": 4.5, "milp_status": "time_limit", "milp_plan_value": 3.0, "method_value": 2.0}
    assert crosscheck_violations([good]) == []
    assert "chain broken" in crosscheck_violations([dict(good, milp_plan_value=5.0)])[0]
    assert "chain broken" in crosscheck_violations([dict(good, method_value=4.6)])[0]
    assert "incomplete" in crosscheck_violations([dict(good, milp_plan_value=math.nan)])[0]


def _trajectory_row(scenario, realization, owner, value):
    return dict(_bound_row(scenario, realization, value), variant="trajectory", trajectory_method=owner)


def test_union_bound_is_the_per_realization_maximum_over_trajectories():
    bound_rows = [
        _trajectory_row("A", 0, "B1", 2.0), _trajectory_row("A", 0, "P", 3.0),
        _trajectory_row("A", 1, "B1", 4.0), _trajectory_row("A", 1, "P", 3.0),
        _bound_row("A", 0, 4.0),
    ]
    assert union_bound_ratios(bound_rows) == {("s", "A", 0): 0.75, ("s", "A", 1): 1.0}
    method_rows = [dict(_method_row("A", "T1", realization, 2), method="P") for realization in (0, 1)]
    owners = {"P": ("trajectory", "P")}
    [row] = summarize(method_rows, bound_rows, BootstrapConfig(50, 3, 0.95), owners)
    assert row["bound_ratio_mean"] == pytest.approx(0.75)
    assert row["union_bound_ratio_mean"] == pytest.approx(0.875)
    assert row["union_gap_mean"] == pytest.approx((0.875 - 0.5) / 0.875)
    assert row["evaluate_calls_mean"] == 1.0
    assert validity_violations(method_rows, bound_rows, owners) == []


def test_methods_without_bounds_are_summarized_without_gaps_or_validity_checks():
    method_rows = [dict(_method_row("A", "T1", 0, 4), method="P_op1")]
    [row] = summarize(method_rows, [], BootstrapConfig(50, 3, 0.95), {"P_op1": None})
    assert math.isnan(row["bound_ratio_mean"]) and math.isnan(row["gap_mean"]) and math.isnan(row["union_gap_mean"])
    assert row["gap_undefined_count"] == 1
    assert validity_violations(method_rows, [], {"P_op1": None}) == []
```

In `tests/runner/experiment_fixtures.py` (2 edits):

Edit 1: replace

```python

def write_tiny_experiment(root: Path, output_dir: Path, workers: int = 2) -> Path:
    scenario_dir = root / "scenarios"
```

with

```python

def write_tiny_experiment(root: Path, output_dir: Path, workers: int = 2, methods=("B0", "B1"), crosscheck: bool = True) -> Path:
    scenario_dir = root / "scenarios"
```

Edit 2: replace

```python
        "realization_ids": {"start": 0, "stop": 2},
        "methods": ["B0", "B1"],
        "bounds": {"tangents": 10},
        "milp_crosscheck": {"set_id": "tiny", "scenario_count": 2, "alert_count": 2, "realization_id": 0, "time_limit_s": 30},
        "bootstrap": {"resamples": 200, "seed": 5, "confidence": 0.95},
        "workers": workers,
        "output_dir": str(output_dir),
    }
    path = root / f"experiment-{output_dir.name}.yaml"
```

with

```python
        "realization_ids": {"start": 0, "stop": 2},
        "methods": list(methods),
        "bounds": {"tangents": 10},
        "bootstrap": {"resamples": 200, "seed": 5, "confidence": 0.95},
        "workers": workers,
        "output_dir": str(output_dir),
    }
    if crosscheck:
        config["milp_crosscheck"] = {"set_id": "tiny", "scenario_count": 2, "alert_count": 2, "realization_id": 0, "time_limit_s": 30}
    path = root / f"experiment-{output_dir.name}.yaml"
```

Append to `tests/runner/test_runner_experiment.py`:

```python
def test_search_methods_bound_their_own_trajectories_and_join_the_union_bound(tmp_path: Path):
    methods = [
        "B1",
        {"id": "S", "base": "search", "params": {"budget": 20, "design_ids": [0]}, "bounds": True},
        {"id": "T", "base": "search", "params": {"budget": 10, "trajectory_block": False}},
    ]
    config = load_experiment_config(write_tiny_experiment(tmp_path, tmp_path / "out", methods=methods, crosscheck=False))
    assert run_experiment(config, Path("tiny.yaml"), workers=2) == 0
    output = tmp_path / "out"
    assert not (output / "milp_crosscheck.csv").exists()
    bounds = _rows_without_runtime(output / "bounds.csv")
    assert sorted({(row["variant"], row["trajectory_method"]) for row in bounds}) == [("trajectory", "B1"), ("trajectory", "S")]
    assert len([row for row in bounds if row["trajectory_method"] == "S"]) == 2 * 2
    assert not list((output / "shards").glob("bound__*__S__*.json"))
    calls = {row["method"]: int(row["evaluate_calls"]) for row in _rows_without_runtime(output / "method_realizations.csv")}
    assert calls["B1"] == 1 and 2 <= calls["S"] <= 21 and 2 <= calls["T"] <= 11
    with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
        summary = {row["method"]: row for row in csv.DictReader(stream)}
    assert summary["T"]["bound_ratio_mean"] == "nan" and summary["S"]["bound_ratio_mean"] != "nan"
    for method in ("B1", "S", "T"):
        assert float(summary[method]["union_bound_ratio_mean"]) >= float(summary["B1"]["bound_ratio_mean"]) - 1e-12
        assert float(summary[method]["union_bound_ratio_mean"]) >= float(summary["S"]["bound_ratio_mean"]) - 1e-12
```

Create the Phase 1 key test (it skips only when the scenario sets are not generated):

```python
# tests/runner/test_runner_phase1_keys.py
import csv
from pathlib import Path

import pytest

from runner.config import load_experiment_config
from runner.experiment import build_tasks, scenario_refs

RESULTS = Path("results/phase1")


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_phase1_config_yields_the_phase1_task_keys():
    config = load_experiment_config(Path("configs/experiments/phase1.yaml"))
    if not all(item.manifest.exists() and item.scenario_dir.exists() for item in config.scenario_sets):
        pytest.skip("scenario sets are not generated in this checkout")
    expected = {f"method__{row['set_id']}__{row['scenario_id']}__{row['method']}" for row in _rows("method_realizations.csv")}
    expected |= {
        f"bound__{row['set_id']}__{row['scenario_id']}__{row['variant']}__{row['trajectory_method'] or 'none'}__r{int(row['realization_id']):03d}"
        for row in _rows("bounds.csv")
    }
    expected |= {f"milp__{row['set_id']}__{row['scenario_id']}" for row in _rows("milp_crosscheck.csv")}
    keys = [task.key for task in build_tasks(config, scenario_refs(config))]
    assert len(keys) == len(set(keys)) == len(expected) == 3725
    assert set(keys) == expected
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/runner -v`

Expected: collection fails for `test_runner_aggregate.py` (`cannot import name 'union_bound_ratios'`) and `test_runner_config.py` (`cannot import name 'MethodSpec'`), and pytest stops at these collection errors.

Run: `uv run --extra test pytest tests/runner/test_runner_experiment.py -v`

Expected: `test_search_methods_bound_their_own_trajectories_and_join_the_union_bound` fails with `ValueError: methods: unknown or missing methods`, because the old loader accepts method names only.

- [ ] **Step 3: Parse method specs and build methods**

Replace the whole content of `src/runner/config.py` with:

```python
# src/runner/config.py
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import yaml

from baselines import METHODS
from optimization.bcd import SearchParams

SEARCH_BASE = "search"
METHOD_ID_PATTERN = re.compile(r"^[A-Za-z0-9.-]+(_[A-Za-z0-9.-]+)*$")


@dataclass(frozen=True)
class ScenarioSetRef:
    set_id: str
    manifest: Path
    scenario_dir: Path


@dataclass(frozen=True)
class CrosscheckConfig:
    set_id: str
    scenario_count: int
    alert_count: int
    realization_id: int
    time_limit_s: float


@dataclass(frozen=True)
class BootstrapConfig:
    resamples: int
    seed: int
    confidence: float


@dataclass(frozen=True)
class MethodSpec:
    """B0 or B1 by name, or a search variant with sorted parameter pairs and in-task trajectory bounds."""

    id: str
    base: str
    params: tuple[tuple[str, object], ...] = ()
    bounds: bool = False

    def search_params(self) -> SearchParams:
        return SearchParams.from_mapping(dict(self.params))


def _frozen(value: object) -> object:
    return tuple(_frozen(item) for item in value) if isinstance(value, (list, tuple)) else value


def parse_method_spec(raw: object, index: int) -> MethodSpec:
    label = f"methods[{index}]"
    if isinstance(raw, str):
        if raw not in METHODS:
            raise ValueError(f"{label}: unknown method {raw!r}; name one of {sorted(METHODS)} or give a search mapping")
        return MethodSpec(raw, raw)
    if not isinstance(raw, dict):
        raise ValueError(f"{label}: expected a method name or a mapping")
    unknown = sorted(set(raw) - {"id", "base", "params", "bounds"})
    if unknown:
        raise ValueError(f"{label}: unknown keys {unknown}")
    method_id, base = raw.get("id"), raw.get("base")
    if base in METHODS:
        if method_id != base or "params" in raw or "bounds" in raw:
            raise ValueError(f"{label}: {base} takes no other id, params, or bounds; its bounds come from bound tasks")
        return MethodSpec(base, base)
    if base != SEARCH_BASE:
        raise ValueError(f"{label}: unknown base {base!r}; expected one of {[*sorted(METHODS), SEARCH_BASE]}")
    if not isinstance(method_id, str) or not METHOD_ID_PATTERN.match(method_id) or method_id in METHODS:
        raise ValueError(f"{label}: id {method_id!r} must use letters, digits, '.', '-', single '_' and not name a baseline")
    params, bounds = raw.get("params", {}), raw.get("bounds", False)
    if not isinstance(params, dict):
        raise ValueError(f"{label}.params: expected a mapping")
    if not isinstance(bounds, bool):
        raise ValueError(f"{label}.bounds: expected true or false")
    spec = MethodSpec(method_id, base, tuple(sorted((str(key), _frozen(value)) for key, value in params.items())), bounds)
    try:
        spec.search_params()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}.params: {error}") from error
    return spec


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    scenario_sets: tuple[ScenarioSetRef, ...]
    split: str
    realization_ids: tuple[int, ...]
    methods: tuple[MethodSpec, ...]
    tangents: int
    crosscheck: CrosscheckConfig | None
    bootstrap: BootstrapConfig
    workers: int
    output_dir: Path
    config_hash: str


def load_experiment_config(path: Path) -> ExperimentConfig:
    payload = path.read_bytes()
    raw = yaml.safe_load(payload)
    sets = tuple(
        ScenarioSetRef(str(item["set_id"]), Path(item["manifest"]), Path(item["scenario_dir"]))
        for item in raw["scenario_sets"]
    )
    set_ids = [item.set_id for item in sets]
    if not sets or len(set(set_ids)) != len(set_ids):
        raise ValueError("scenario_sets: set_id values must be present and unique")
    span = raw["realization_ids"]
    realization_ids = tuple(range(int(span["start"]), int(span["stop"])))
    if not realization_ids:
        raise ValueError("realization_ids: the range is empty")
    methods = tuple(parse_method_spec(item, index) for index, item in enumerate(raw.get("methods") or ()))
    method_ids = [spec.id for spec in methods]
    if not methods or len(set(method_ids)) != len(method_ids):
        raise ValueError("methods: at least one method is required and ids must be unique")
    cross = raw.get("milp_crosscheck")
    crosscheck = None
    if cross is not None:
        crosscheck = CrosscheckConfig(
            set_id=str(cross["set_id"]),
            scenario_count=int(cross["scenario_count"]),
            alert_count=int(cross["alert_count"]),
            realization_id=int(cross["realization_id"]),
            time_limit_s=float(cross["time_limit_s"]),
        )
        if crosscheck.set_id not in set_ids:
            raise ValueError(f"milp_crosscheck.set_id: {crosscheck.set_id!r} is not a configured scenario set")
    boot = raw["bootstrap"]
    bootstrap = BootstrapConfig(int(boot["resamples"]), int(boot["seed"]), float(boot["confidence"]))
    if bootstrap.resamples < 1 or not 0.0 < bootstrap.confidence < 1.0:
        raise ValueError("bootstrap: resamples must be positive and confidence must lie in (0, 1)")
    workers = int(raw["workers"])
    tangents = int(raw["bounds"]["tangents"])
    if workers < 1 or tangents < 1:
        raise ValueError("workers and bounds.tangents must be at least 1")
    return ExperimentConfig(
        experiment_id=str(raw["experiment_id"]),
        scenario_sets=sets,
        split=str(raw["split"]),
        realization_ids=realization_ids,
        methods=methods,
        tangents=tangents,
        crosscheck=crosscheck,
        bootstrap=bootstrap,
        workers=workers,
        output_dir=Path(raw["output_dir"]),
        config_hash=hashlib.sha256(payload).hexdigest(),
    )
```

```python
# src/runner/methods.py
"""Planning methods and the bound rows that check them, built from experiment method specs."""

from baselines import get_method
from baselines.method import Method
from optimization.bcd import SearchMethod

from .config import SEARCH_BASE, MethodSpec

GROUND_ONLY = "ground_only"
TRAJECTORY = "trajectory"


def build_method(spec: MethodSpec) -> Method:
    if spec.base == SEARCH_BASE:
        return SearchMethod(spec.id, spec.search_params())
    return get_method(spec.base)


def bound_owner(spec: MethodSpec) -> tuple[str, str] | None:
    """(variant, trajectory_method) of the bound rows for the method; None for a search method without bounds."""
    if spec.base == SEARCH_BASE:
        return (TRAJECTORY, spec.id) if spec.bounds else None
    return (TRAJECTORY, spec.base) if get_method(spec.base).uses_uav else (GROUND_ONLY, "")
```

- [ ] **Step 4: Run method specs and their bounds in tasks**

Replace the whole content of `src/runner/tasks.py` with:

```python
# src/runner/tasks.py
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from typing import Mapping

import numpy as np

from baselines import get_method
from bounds.crosscheck import model_snr_values, plan_from_solution, reduce_scenario, tangent_max_overestimate
from bounds.formulation import build_bound_model, restrict_to_ground
from bounds.solve import solve_lp, solve_milp
from models.channel import build_link_channels, channel_uniforms
from models.evaluate import REALIZED, EvaluationCounter, evaluate
from models.paths import CandidatePath, candidate_paths, radio_links
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import Scenario, load_scenario

from .config import MethodSpec
from .methods import GROUND_ONLY, TRAJECTORY, build_method

CROSSCHECK_TRAJECTORY_METHOD = "B1"


@dataclass(frozen=True)
class MethodTask:
    set_id: str
    scenario_id: str
    scenario_path: str
    spec: MethodSpec
    realization_ids: tuple[int, ...]
    tangents: int

    @property
    def key(self) -> str:
        return f"method__{self.set_id}__{self.scenario_id}__{self.spec.id}"


@dataclass(frozen=True)
class BoundTask:
    set_id: str
    scenario_id: str
    scenario_path: str
    variant: str
    trajectory_method: str
    realization_id: int
    tangents: int

    @property
    def key(self) -> str:
        owner = self.trajectory_method or "none"
        return f"bound__{self.set_id}__{self.scenario_id}__{self.variant}__{owner}__r{self.realization_id:03d}"


@dataclass(frozen=True)
class CrosscheckTask:
    set_id: str
    scenario_id: str
    scenario_path: str
    trajectory_method: str
    alert_count: int
    realization_id: int
    time_limit_s: float
    tangents: int

    @property
    def key(self) -> str:
        return f"milp__{self.set_id}__{self.scenario_id}"


Task = MethodTask | BoundTask | CrosscheckTask


def task_hash(task: Task, scenario_sha256: str, source_hash: str) -> str:
    payload = {
        "type": type(task).__name__,
        "task": asdict(task),
        "scenario_sha256": scenario_sha256,
        "source_hash": source_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_method_task(task: MethodTask) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Method rows, plus trajectory bound rows for the planned trajectory when the spec asks for bounds."""
    scenario = load_scenario(Path(task.scenario_path))
    candidates = candidate_paths(scenario)
    method = build_method(task.spec)
    counter = EvaluationCounter()
    started = time.perf_counter()
    plan = method.plan(scenario, candidates, counter)
    planning_s = time.perf_counter() - started
    result = evaluate(scenario, plan, REALIZED, task.realization_ids, counter=counter, candidates=candidates)
    if not result.feasible:
        raise ValueError(f"{task.key}: infeasible plan: {list(result.violations[:3])}")
    sources = len(scenario.sources)
    without_uav = round(result.connectivity_ratio_without_uav * sources)
    rows = []
    for realization in result.realizations:
        timely = sum(realization.timely.values())
        connected = sum(realization.connected.values()) if method.uses_uav else without_uav
        rows.append(
            {
                "set_id": task.set_id,
                "scenario_id": task.scenario_id,
                "topology_id": scenario.network_id,
                "method": task.spec.id,
                "realization_id": realization.realization_id,
                "alert_count": len(scenario.alerts),
                "timely_count": timely,
                "timely_ratio": timely / len(scenario.alerts),
                "source_count": sources,
                "connected_count": connected,
                "conn_ratio": connected / sources,
                "shortfall": realization.shortfall,
                "planning_runtime_s": planning_s,
                "evaluation_runtime_s": result.runtime_s / len(task.realization_ids),
                "evaluate_calls": counter.calls,
            }
        )
    bound_rows = []
    if task.spec.bounds:
        bound_rows = [
            bound_row(
                task.set_id, task.scenario_id, scenario, candidates, TRAJECTORY, task.spec.id, plan.trajectory,
                realization_id, task.tangents,
            )
            for realization_id in task.realization_ids
        ]
    return rows, bound_rows


def bound_row(
    set_id: str,
    scenario_id: str,
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    variant: str,
    trajectory_method: str,
    trajectory: np.ndarray,
    realization_id: int,
    tangents: int,
) -> dict[str, object]:
    """Solve the per-realization LP bound for a fixed trajectory on the evaluation channel draws."""
    channels = build_link_channels(scenario, radio_links(candidates), trajectory, channel_uniforms(scenario, realization_id))
    result = solve_lp(build_bound_model(scenario, candidates, channels, tangents))
    return {
        "set_id": set_id,
        "scenario_id": scenario_id,
        "topology_id": scenario.network_id,
        "variant": variant,
        "trajectory_method": trajectory_method,
        "realization_id": realization_id,
        "value": result.value,
        "ratio": result.value / len(scenario.alerts),
        "status": result.status,
        "runtime_s": result.runtime_s,
        "variables": result.variables,
        "rows": result.rows,
        "nonzeros": result.nonzeros,
    }


def run_bound_task(task: BoundTask) -> list[dict[str, object]]:
    scenario = load_scenario(Path(task.scenario_path))
    candidates = candidate_paths(scenario)
    if task.variant == GROUND_ONLY:
        candidates = restrict_to_ground(candidates)
        trajectory = stationary_trajectory(scenario)
    else:
        trajectory = get_method(task.trajectory_method).trajectory(scenario)
    return [
        bound_row(
            task.set_id, task.scenario_id, scenario, candidates, task.variant, task.trajectory_method, trajectory,
            task.realization_id, task.tangents,
        )
    ]


def run_crosscheck_task(task: CrosscheckTask) -> list[dict[str, object]]:
    scenario = reduce_scenario(load_scenario(Path(task.scenario_path)), task.alert_count)
    candidates = candidate_paths(scenario)
    trajectory = get_method(task.trajectory_method).trajectory(scenario)
    channels = build_link_channels(
        scenario, radio_links(candidates), trajectory, channel_uniforms(scenario, task.realization_id)
    )
    model = build_bound_model(scenario, candidates, channels, task.tangents)
    lp = solve_lp(model)
    milp = solve_milp(model, task.time_limit_s)

    def timely_count(plan: Plan) -> float:
        evaluated = evaluate(scenario, plan, REALIZED, (task.realization_id,), candidates=candidates)
        return float(sum(evaluated.realizations[0].timely.values())) if evaluated.feasible else math.nan

    static_value = equal_split_value = math.nan
    if milp.solution is not None:
        static_plan = plan_from_solution(scenario, candidates, model, milp.solution, trajectory)
        static_value = timely_count(static_plan)
        equal_split_value = timely_count(Plan(trajectory, static_plan.path_choice, EqualSplitBacklogged()))
    method = get_method(task.trajectory_method)
    method_value = timely_count(method.plan(scenario, candidates, EvaluationCounter()))
    return [
        {
            "set_id": task.set_id,
            "scenario_id": task.scenario_id,
            "alert_count": len(scenario.alerts),
            "lp_ub": lp.value,
            "lp_status": lp.status,
            "milp_incumbent": milp.value,
            "milp_dual_bound": milp.dual_bound,
            "milp_status": milp.status,
            "milp_runtime_s": milp.runtime_s,
            "milp_plan_static_value": static_value,
            "milp_plan_equal_split_value": equal_split_value,
            "milp_plan_value": max(static_value, equal_split_value),
            "method_value": method_value,
            "tangent_max_overestimate": tangent_max_overestimate(
                model_snr_values(model, channels), scenario.spectrum.b_tot_hz, task.tangents
            ),
        }
    ]


RUNNERS = {BoundTask: run_bound_task, CrosscheckTask: run_crosscheck_task}


def execute(task: Task) -> dict[str, object]:
    started = time.perf_counter()
    bound_rows: list[dict[str, object]] = []
    try:
        if isinstance(task, MethodTask):
            rows, bound_rows = run_method_task(task)
        else:
            rows = RUNNERS[type(task)](task)
        error = None
    except Exception as exc:  # recorded in the shard; the experiment reports it and exits non-zero
        rows, bound_rows, error = [], [], "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return {"key": task.key, "rows": rows, "bound_rows": bound_rows, "error": error, "runtime_s": time.perf_counter() - started}
```

- [ ] **Step 5: Add owners and the union bound to aggregation**

Replace the whole content of `src/runner/aggregate.py` with:

```python
# src/runner/aggregate.py
from collections import defaultdict
import math
from typing import Mapping, Sequence

import numpy as np

from .config import BootstrapConfig
from .methods import TRAJECTORY

VALIDITY_TOLERANCE = 1e-6
BoundOwners = Mapping[str, tuple[str, str] | None]


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
    method_rows: Sequence[Mapping[str, object]], bound_rows: Sequence[Mapping[str, object]], owners: BoundOwners
) -> list[str]:
    """Every bound optimal; every method row with its own bound stays below it."""
    bounds = _bound_lookup(bound_rows)
    violations = []
    for row in bound_rows:
        if row["status"] != "optimal":
            violations.append(f"bound {row['set_id']}/{row['scenario_id']}/{row['variant']}/r{row['realization_id']}: {row['status']}")
    for row in method_rows:
        owner = owners[str(row["method"])]
        if owner is None:
            continue
        variant, trajectory_method = owner
        key = (row["set_id"], row["scenario_id"], variant, trajectory_method, int(row["realization_id"]))
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


def union_bound_ratios(bound_rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, int], float]:
    """Per (set, scenario, realization): the largest trajectory bound ratio over every trajectory with bounds."""
    union: dict[tuple[str, str, int], float] = {}
    for row in bound_rows:
        if row["variant"] == TRAJECTORY:
            key = (str(row["set_id"]), str(row["scenario_id"]), int(row["realization_id"]))
            union[key] = max(union.get(key, -math.inf), float(row["ratio"]))
    return union


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _mean_or_nan(values: Sequence[float]) -> float:
    return _mean(values) if values else math.nan


def summarize(
    method_rows: Sequence[Mapping[str, object]],
    bound_rows: Sequence[Mapping[str, object]],
    bootstrap: BootstrapConfig,
    owners: BoundOwners,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, list[Mapping[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in method_rows:
        grouped[(str(row["set_id"]), str(row["method"]))][str(row["scenario_id"])].append(row)
    bounds: dict[tuple, list[Mapping[str, object]]] = defaultdict(list)
    for row in bound_rows:
        bounds[(row["set_id"], row["scenario_id"], row["variant"], row["trajectory_method"])].append(row)
    union = union_bound_ratios(bound_rows)
    summary = []
    for (set_id, method), scenarios in sorted(grouped.items()):
        owner = owners[method]
        clusters: dict[str, list[float]] = defaultdict(list)
        timely, conn, shortfall, planning, evaluation, calls = [], [], [], [], [], []
        bound_means, bound_runtime, gaps, union_means, union_gaps = [], [], [], [], []
        for scenario_id in sorted(scenarios):
            rows = scenarios[scenario_id]
            scenario_timely = _mean([float(row["timely_ratio"]) for row in rows])
            timely.append(scenario_timely)
            clusters[str(rows[0]["topology_id"])].append(scenario_timely)
            conn.append(_mean([float(row["conn_ratio"]) for row in rows]))
            shortfall.append(_mean([float(row["shortfall"]) for row in rows]))
            planning.append(_mean([float(row["planning_runtime_s"]) for row in rows]))
            evaluation.append(_mean([float(row["evaluation_runtime_s"]) for row in rows]))
            calls.append(_mean([float(row["evaluate_calls"]) for row in rows]))
            if owner is not None:
                scenario_bounds = bounds[(set_id, scenario_id, *owner)]
                bound_mean = _mean([float(row["ratio"]) for row in scenario_bounds])
                bound_means.append(bound_mean)
                bound_runtime.append(_mean([float(row["runtime_s"]) for row in scenario_bounds]))
                gaps.append(relative_gap(bound_mean, scenario_timely))
            union_keys = [(set_id, scenario_id, int(row["realization_id"])) for row in rows]
            if all(key in union for key in union_keys):
                union_mean = _mean([union[key] for key in union_keys])
                union_means.append(union_mean)
                union_gaps.append(relative_gap(union_mean, scenario_timely))
        low, high = cluster_bootstrap_ci(clusters, bootstrap.resamples, bootstrap.seed, bootstrap.confidence)
        defined = [value for value in gaps if value is not None]
        union_defined = [value for value in union_gaps if value is not None]
        summary.append(
            {
                "set_id": set_id,
                "method": method,
                "scenario_count": len(scenarios),
                "timely_ratio_mean": _mean(timely),
                "timely_ratio_ci_low": low,
                "timely_ratio_ci_high": high,
                "conn_ratio_mean": _mean(conn),
                "bound_ratio_mean": _mean_or_nan(bound_means),
                "gap_mean": _mean_or_nan(defined),
                "gap_undefined_count": len(scenarios) - len(defined),
                "shortfall_mean": _mean(shortfall),
                "planning_runtime_s_mean": _mean(planning),
                "evaluation_runtime_s_per_realization_mean": _mean(evaluation),
                "bound_runtime_s_mean": _mean_or_nan(bound_runtime),
                "union_bound_ratio_mean": _mean_or_nan(union_means),
                "union_gap_mean": _mean_or_nan(union_defined),
                "evaluate_calls_mean": _mean(calls),
            }
        )
    return summary
```

- [ ] **Step 6: Collect in-task bounds and make the cross-check optional**

Replace the whole content of `src/runner/experiment.py` with:

```python
# src/runner/experiment.py
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
from .config import SEARCH_BASE, ExperimentConfig
from .methods import bound_owner
from .provenance import environment, file_sha256, git_state, source_hash
from .tasks import CROSSCHECK_TRAJECTORY_METHOD, BoundTask, CrosscheckTask, MethodTask, Task, execute, task_hash

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
    "union_bound_ratio_mean", "union_gap_mean", "evaluate_calls_mean",
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
    """Method tasks for every spec; bound tasks for B0/B1 trajectories (search methods bound inside their tasks)."""
    tasks: list[Task] = []
    variants = sorted({bound_owner(spec) for spec in config.methods if spec.base != SEARCH_BASE})
    for ref in refs:
        for spec in config.methods:
            tasks.append(MethodTask(ref.set_id, ref.scenario_id, str(ref.path), spec, config.realization_ids, config.tangents))
        for variant, owner in variants:
            for realization_id in config.realization_ids:
                tasks.append(
                    BoundTask(ref.set_id, ref.scenario_id, str(ref.path), variant, owner, realization_id, config.tangents)
                )
    cross = config.crosscheck
    if cross is None:
        return tasks
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


def _bound_rows(task: Task, shard: dict[str, object]) -> list[dict[str, object]]:
    if isinstance(task, BoundTask):
        return shard["rows"]
    return shard.get("bound_rows", []) if isinstance(task, MethodTask) else []


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
        (row for task in tasks for row in _bound_rows(task, shards[task.key])),
        key=lambda row: (row["set_id"], row["scenario_id"], row["variant"], row["trajectory_method"], row["realization_id"]),
    )
    crosscheck_rows = sorted(
        (row for task in tasks if isinstance(task, CrosscheckTask) for row in shards[task.key]["rows"]),
        key=lambda row: (row["set_id"], row["scenario_id"]),
    )
    owners = {spec.id: bound_owner(spec) for spec in config.methods}
    _write_rows(output / "method_realizations.csv", METHOD_COLUMNS, method_rows)
    _write_rows(output / "bounds.csv", BOUND_COLUMNS, bound_rows)
    if config.crosscheck is not None:
        _write_rows(output / "milp_crosscheck.csv", CROSSCHECK_COLUMNS, crosscheck_rows)
    if not failures:
        failures += validity_violations(method_rows, bound_rows, owners)
        failures += crosscheck_violations(crosscheck_rows)
    summary_path = output / "summary.csv"
    if failures:
        summary_path.unlink(missing_ok=True)
    else:
        _write_rows(summary_path, SUMMARY_COLUMNS, summarize(method_rows, bound_rows, config.bootstrap, owners))

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
```

- [ ] **Step 7: Hash the optimization sources and ignore Phase 2 shards**

In `src/runner/provenance.py` (1 edit):

Edit 1: replace

```python
SRC_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGES = ("models", "baselines", "bounds", "runner")
```

with

```python
SRC_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGES = ("models", "baselines", "bounds", "optimization", "runner")
```

In `.gitignore` (1 edit):

Edit 1: replace

```text
results/phase1/shards/
```

with

```text
results/phase1/shards/
results/phase2/**/shards/
```

- [ ] **Step 8: Add the Phase 2 entry point and pilot configuration**

```python
# experiments/run_phase2.py
import argparse
from pathlib import Path
import sys

from runner.config import load_experiment_config
from runner.experiment import run_experiment


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="run-phase2")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/phase2_pilot.yaml"))
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(arguments)
    config = load_experiment_config(args.config)
    return run_experiment(config, args.config, workers=args.workers, argv=["experiments/run_phase2.py", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
```

```yaml
# configs/experiments/phase2_pilot.yaml
experiment_id: phase2-pilot
scenario_sets:
  - set_id: v0
    manifest: data/manifests/scenarios_v0.csv
    scenario_dir: data/processed/scenarios/v0
split: dev
realization_ids: {start: 0, stop: 30}
methods:
  - B1
  - {id: B2, base: search, bounds: true}
  - {id: B3, base: search, params: {objective: leximin}, bounds: true}
  - {id: P, base: search, params: {operation1: true, operation2: true}, bounds: true}
bounds: {tangents: 10}
bootstrap: {resamples: 10000, seed: 20260911, confidence: 0.95}
workers: 12
output_dir: results/phase2/pilot
```

- [ ] **Step 9: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/runner -v`

Expected: 27 passed, counting the unchanged report-table and provenance tests (the Phase 1 key test runs because `data/manifests` and `data/processed/scenarios` exist in this checkout).

Run: `uv run --extra test pytest -q`

Expected: 170 passed, 1 skipped.

- [ ] **Step 10: Commit**

```bash
git add src/runner experiments/run_phase2.py configs/experiments/phase2_pilot.yaml .gitignore tests/runner
git commit -m "feat: run search methods with in-task bounds and union bound"
```

### Task 8: Phase 1 reproduction check

**Files:** none (verification only; nothing to commit).

**Interfaces:**
- Consumes: Task 7 runner and `experiments/run_phase1.py`; `configs/experiments/phase1.yaml`; committed `results/phase1/{method_realizations,bounds,milp_crosscheck,summary}.csv`.
- Produces: evidence for spec acceptance criterion 17.1.2.

- [ ] **Step 1: Re-run Phase 1 into a temporary directory and compare**

Run the whole block in one shell so `REPRO` stays set. The source hash changed, so every task is recomputed; the prototype run took 15.5 minutes with 12 workers.

```bash
REPRO=$(mktemp -d)
sed "s#^output_dir: .*#output_dir: $REPRO#" configs/experiments/phase1.yaml > "$REPRO/phase1.yaml"
uv run python experiments/run_phase1.py --config "$REPRO/phase1.yaml"
uv run python - results/phase1 "$REPRO" <<'PY'
import csv
from pathlib import Path
import sys

committed, rerun = Path(sys.argv[1]), Path(sys.argv[2])
RUNTIME = {
    "planning_runtime_s", "evaluation_runtime_s", "runtime_s", "milp_runtime_s",
    "planning_runtime_s_mean", "evaluation_runtime_s_per_realization_mean", "bound_runtime_s_mean",
}
reproduced = True
for name in ("method_realizations.csv", "bounds.csv", "milp_crosscheck.csv", "summary.csv"):
    with (committed / name).open(newline="", encoding="utf-8") as stream:
        old = list(csv.DictReader(stream))
    with (rerun / name).open(newline="", encoding="utf-8") as stream:
        new = list(csv.DictReader(stream))
    columns = [column for column in old[0] if column not in RUNTIME]
    differences = [
        (index, column, before[column], after.get(column))
        for index, (before, after) in enumerate(zip(old, new))
        for column in columns
        if before[column] != after.get(column)
    ]
    same = len(old) == len(new) and not differences
    print(f"{name}: {len(old)} rows, {len(columns)} non-runtime columns, {'identical' if same else 'DIFFERENT'}")
    for difference in differences[:5]:
        print("  ", difference)
    reproduced = reproduced and same
print("PHASE 1 REPRODUCED" if reproduced else "PHASE 1 DIFFERS")
sys.exit(0 if reproduced else 1)
PY
echo "compare exit $?"
```

Expected: the runner prints `{"task_count": 3725, "executed_task_count": 3725, "skipped_task_count": 0, "status": "complete"}`, then the comparison prints

```text
method_realizations.csv: 3600 rows, 13 non-runtime columns, identical
bounds.csv: 3600 rows, 12 non-runtime columns, identical
milp_crosscheck.csv: 5 rows, 13 non-runtime columns, identical
summary.csv: 4 rows, 11 non-runtime columns, identical
PHASE 1 REPRODUCED
compare exit 0
```

The re-run `summary.csv` also has the three new columns; the comparison checks the committed columns only.

- [ ] **Step 2: Remove the temporary run**

```bash
rm -rf "$REPRO"
```

If any file prints `DIFFERENT`, stop: criterion 17.1.2 fails. Find the change that alters Phase 1 behaviour before continuing with Task 9.

### Task 9: Development-set pilot

**Files:**
- Create: `results/phase2/pilot/method_realizations.csv`, `results/phase2/pilot/bounds.csv`, `results/phase2/pilot/summary.csv`, `results/phase2/pilot/manifest.json`, `results/phase2/pilot/pilot_note.md` (`results/phase2/pilot/shards/` stays ignored)

**Interfaces:**
- Consumes: Task 7 `experiments/run_phase2.py` and `configs/experiments/phase2_pilot.yaml`; generated scenario set `v0` (split `dev`: `Bbnplanet-r0`, `Bbnplanet-r1`, `Canerie-r0`, `Canerie-r1`, `Fccn-r0`, `Fccn-r1`).
- Produces: evidence for spec acceptance criterion 17.1.3, the budget and design-realization decision, and runtime figures that plan C.2 uses to size its experiments.

- [ ] **Step 1: Run the pilot**

Run: `uv run python experiments/run_phase2.py`

Expected: `{"task_count": 204, "executed_task_count": 204, "skipped_task_count": 0, "status": "complete"}` and exit code 0. The 204 tasks are, per scenario, four method tasks and 30 B1 bound tasks; B2, B3, P solve their 30 bounds inside their method tasks. The prototype run took 14 minutes with 12 workers.

- [ ] **Step 2: Check the outputs**

```bash
uv run python - <<'PY'
import csv


def rows(name):
    with open(f"results/phase2/pilot/{name}", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


bounds, methods, summary = rows("bounds.csv"), rows("method_realizations.csv"), rows("summary.csv")
print(len(methods), "method rows;", len(bounds), "bound rows; statuses", sorted({row["status"] for row in bounds}))
print("bounded trajectories", sorted({row["trajectory_method"] for row in bounds}))
for row in summary:
    print(row["method"], *(f"{float(row[column]):.3f}" for column in ("timely_ratio_mean", "gap_mean", "union_gap_mean")), row["evaluate_calls_mean"])
PY
```

Expected:

```text
720 method rows; 720 bound rows; statuses ['optimal']
bounded trajectories ['B1', 'B2', 'B3', 'P']
B1 0.461 0.290 0.383 1.0
B2 0.545 0.225 0.264 2001.0
B3 0.396 0.466 0.479 2001.0
P 0.542 0.230 0.268 2001.0
```

Results are deterministic, so every number above must match. A search method's 2001 evaluate calls are its whole budget plus the runner's final evaluation. If a value differs, stop and find the source of non-determinism before writing the note.

- [ ] **Step 3: Write the pilot note**

The note reads the result files, so its numbers always match them; only the timestamps and runtimes change from run to run. Run it before any re-run, because a re-run rewrites `manifest.json`.

```bash
uv run python - <<'PY'
import csv
from collections import defaultdict
import json
from pathlib import Path

PILOT = Path("results/phase2/pilot")
BUDGET = 2000
SEARCH = ("B2", "B3", "P")


def rows(name):
    with (PILOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(value, digits=3):
    return "—" if value == "nan" else f"{float(value):.{digits}f}"


summary = {row["method"]: row for row in rows("summary.csv")}
methods = [method for method in ("B1", *SEARCH) if method in summary]
realizations = rows("method_realizations.csv")
manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
per_scenario = defaultdict(lambda: defaultdict(list))
calls = defaultdict(dict)
planning = defaultdict(dict)
for row in realizations:
    per_scenario[row["scenario_id"]][row["method"]].append(float(row["timely_ratio"]))
    calls[row["method"]][row["scenario_id"]] = int(row["evaluate_calls"])
    planning[row["method"]][row["scenario_id"]] = float(row["planning_runtime_s"])
scenarios = sorted(per_scenario)
means = {scenario: {method: sum(values) / len(values) for method, values in per_scenario[scenario].items()} for scenario in scenarios}

lines = [
    "# Phase 2 pilot on the development split",
    "",
    f"Configuration `configs/experiments/phase2_pilot.yaml`: set `v0`, split `dev` ({len(scenarios)} scenarios), "
    f"{len(realizations) // (len(scenarios) * len(methods))} evaluation realizations, methods {', '.join(methods)}. "
    "Search methods use the spec C defaults (design realizations 0–4, budget 2000, at most 10 iterations) and solve their "
    "trajectory bounds inside their method tasks. "
    f"Run: {manifest['task_count']} tasks, {manifest['workers']} workers, status `{manifest['status']}`, "
    f"{manifest['started_utc'][:19]}Z to {manifest['finished_utc'][:19]}Z (UTC).",
    "",
    "## Results by method",
    "",
    "| Method | Timely ratio | Gap to own bound | Gap to union bound | Evaluate calls | Planning time (s) | LP time per bound (s) |",
    "|---|---|---|---|---|---|---|",
]
for method in methods:
    row = summary[method]
    lines.append(
        f"| {method} | {number(row['timely_ratio_mean'])} | {number(row['gap_mean'])} | {number(row['union_gap_mean'])} | "
        f"{float(row['evaluate_calls_mean']):.0f} | {number(row['planning_runtime_s_mean'], 1)} | {number(row['bound_runtime_s_mean'], 2)} |"
    )
lines += ["", "## Timely ratio by scenario", "", "| Scenario | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
for scenario in scenarios:
    lines.append(f"| {scenario} | " + " | ".join(f"{means[scenario][method]:.3f}" for method in methods) + " |")
lines += ["", "## Budget and runtime of search methods", ""]
for method in SEARCH:
    if method in calls:
        exhausted = sum(value == BUDGET + 1 for value in calls[method].values())
        times = planning[method].values()
        lines.append(
            f"- {method}: budget exhausted on {exhausted} of {len(scenarios)} scenarios; planning time {min(times):.0f}–{max(times):.0f} s per scenario."
        )
def compare(first, second):
    higher = sum(means[scenario][first] > means[scenario][second] for scenario in scenarios)
    equal = sum(means[scenario][first] == means[scenario][second] for scenario in scenarios)
    return higher, equal, len(scenarios) - higher - equal


search_planning = [value for method in SEARCH for value in planning[method].values()]
mean_planning = sum(search_planning) / len(search_planning)
mean_lp = sum(float(summary[method]["bound_runtime_s_mean"]) for method in SEARCH) / len(SEARCH)
main_hours = (60 * 10 * mean_planning + 60 * 3 * 30 * mean_lp) / 12 / 3600
budget_hours = 30 * 2 * (250 + 500 + 1000 + 2000 + 4000) / BUDGET * mean_planning / 12 / 3600
p_b2, b2_b1, b3_b1 = compare("P", "B2"), compare("B2", "B1"), compare("B3", "B1")
lines += [
    "",
    "## Decision",
    "",
    "- **Budget stays at 2000 and design realizations stay at 0–4 for plan C.2.** Every search task used its whole budget, so "
    "the pilot compares methods at equal evaluation cost, not at convergence. The C.2 quality–runtime experiment (budgets 250 "
    "to 4000) measures what a larger budget buys, and the design-realization sensitivity (expected, 1, 5, 10) measures the "
    "effect of the design sample; raising either default now would multiply the runtime of every C.2 experiment.",
    f"- **Observed on the development split, not tuned on:** B2 is above B1 on {b2_b1[0]} scenarios, equal on {b2_b1[1]}, below on "
    f"{b2_b1[2]}; P is above B2 on {p_b2[0]}, equal on {p_b2[1]}, below on {p_b2[2]}; B3 is above B1 on {b3_b1[0]}, equal on "
    f"{b3_b1[1]}, below on {b3_b1[2]}. No method parameter was changed after the pilot.",
    f"- **Runtime for plan C.2:** a search task planned for {mean_planning:.0f} s on average with 12 workers "
    f"(range {min(search_planning):.0f}–{max(search_planning):.0f} s) and a trajectory LP took {mean_lp:.1f} s. At that rate "
    f"the main comparison with ablations (60 scenario instances, 10 search methods, bounds for 3) needs about {main_hours:.1f} h "
    f"and the budget experiment about {budget_hours:.1f} h, instead of the 2 h and 0.5 h estimated in spec C Section 11; "
    "plan C.2 sizes its experiments from these numbers and records any change in spec Section 18.",
]
(PILOT / "pilot_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((PILOT / "pilot_note.md").read_text(encoding="utf-8"))
PY
```

Expected: the note is printed and written to `results/phase2/pilot/pilot_note.md`. Its method table shows timely ratios 0.461 (B1), 0.545 (B2), 0.396 (B3), 0.542 (P), and its second decision bullet reads: B2 is above B1 on 5 scenarios, equal on 0, below on 1; P is above B2 on 0, equal on 3, below on 3; B3 is above B1 on 1, equal on 0, below on 5.

- [ ] **Step 4: Confirm that only result files are new, then commit**

Run: `git status --short --untracked-files=all results/phase2`

Expected: exactly the five files listed under **Files**; no shard appears.

```bash
git add results/phase2/pilot/method_realizations.csv results/phase2/pilot/bounds.csv results/phase2/pilot/summary.csv results/phase2/pilot/manifest.json results/phase2/pilot/pilot_note.md
git commit -m "feat: record phase 2 pilot results"
```

---

## Spec Coverage

| Spec section | Task |
|---|---|
| 4 Architecture, file layout, import rules | Global constraints, 1–7 |
| 5.1 Design realizations | 1, 2 |
| 5.2 Weighted bandwidth policy | 1 |
| 6 Design scoring and budget | 2 |
| 7.1–7.2 B2 state, initialization, iteration, blocks | 4, 6 |
| 7.3 Tour family | 3 |
| 7.4 Complexity (budget bound) | 6; the written analysis belongs to plan C.2 (Section 14) |
| 8 B3 leximin key | 2, 6 |
| 9.1 Risk detection | 5 |
| 9.2 Operation 1 | 5 |
| 9.3 Operation 2 | 5 |
| 9.4 Acceptance, shared budget, variants | 4, 5, 6 |
| 10 Runner: method specs, in-task bounds, union bound, compatibility | 7, 8 |
| 15 Error handling | 1 (namespaces, weights), 2 (`BudgetExhausted`, infeasible score), 6 (incumbent return), 7 (validity checks on in-task bounds) |
| 16.1 Testing | 1–7 |
| 17.1 Acceptance criteria 1–3 | 1: every task; 2: Task 8; 3: Task 9 |
| 11–14, 16.2, 17.2 | Plan C.2 |
