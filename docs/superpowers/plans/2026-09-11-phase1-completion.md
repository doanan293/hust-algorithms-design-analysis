# Phase 1 Completion Implementation Plan (Sub-Project B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver course Phase 1: baselines B0 and B1, a valid per-realization LP upper bound with an MILP cross-check, the `v0-bh50` backhaul-stressed family, a resumable parallel runner with committed results, and the Phase 1 state of the report in `docs/report`.

**Architecture:** New packages `src/baselines` (planning methods behind a `Method` protocol), `src/bounds` (sparse LP/MILP model solved by HiGHS through `scipy`), and `src/runner` (configuration, tasks, spawn-based parallel execution with hash-checked shards, aggregation, provenance) sit on top of the sub-project A `models` package. `experiments/run_phase1.py` runs the experiment; `experiments/report_tables.py` turns results into CSVs and macros that the report reads through the LaTeX skill's `csvsimple` and `pgfplots` patterns.

**Tech Stack:** Python 3.12 in the uv-managed `.venv`, `numpy`, `scipy` 1.18 (HiGHS), `networkx`, `PyYAML`, `pytest`; LaTeX with `pdflatex`, `biber`, `csvsimple`, `pgfplots`, `siunitx`, built by `.claude/skills/latex-document-skill/scripts/compile_latex.sh`.

**Spec:** `docs/superpowers/specs/2026-09-11-phase1-completion-design.md` (roadmap: `docs/superpowers/specs/2026-09-11-implementation-roadmap.md`; sub-project A: `docs/superpowers/specs/2026-09-11-model-evaluator-design.md`)

## Global Constraints

- Run project code through uv: tests with `uv run --extra test pytest`, scripts with `uv run python ...`. No dependency is added; `scipy>=1.14,<2` is already in `pyproject.toml`. Skill-script dependencies stay installed globally and never go into `.venv`.
- Import boundaries: `src/models` imports none of `baselines`, `bounds`, `runner`; `src/baselines` and `src/bounds` import `models` only (`bounds.crosscheck` may use `baselines` only in tests); `src/runner` imports `models`, `baselines`, `bounds`; nothing under `src` imports `experiments`.
- Planning uses expected-rate channels only and never reads evaluation realizations.
- The bound is the per-realization LP of spec Section 6.1 with ten tangents at `b_k = B_tot * 2**-k`; every LP must end with status `optimal`; validity tolerance is `1e-6` alerts.
- B0 uses the `ground_only` bound and connectivity without the UAV; B1 uses the `trajectory` bound on the B1 trajectory.
- MILP cross-check: set `v0`, five scenarios at positions 0, 7, 15, 22, 29 of the `(flow_variable_count, scenario_id)` order, the ten smallest alert IDs, B1 trajectory, realization 0, time limit 300 s.
- `configs/scenarios/v0-bh50.yaml` equals `configs/scenarios/v0.yaml` except `set_id: v0-bh50` and `backhaul_capacity_bps: 50000.0`.
- Worker processes use the `spawn` start method. A task hash covers the task type and fields, the scenario `sha256`, and the source hash of `src/models`, `src/baselines`, `src/bounds`, `src/runner`.
- `results/phase1/shards/` is Git-ignored; every other file in `results/phase1/` is committed.
- Test module basenames are unique across `tests/` (there are no `__init__.py` files); shared builders live in `tests/support/`, which is on the pytest `pythonpath`.
- The report follows `latex-document-skill`: tables read semicolon-separated CSV with `csvsimple`, the runtime figure reads CSV with `pgfplots`, the build uses the skill's `compile_latex.sh`, decimals use a comma, and numbers in prose come from `docs/report/data/phase1_results.tex` macros.
- Only verified references are added: `lawler1990preemptive` and `lenstra2020elements`.
- Every task follows test-driven development where code is involved and ends with its own commit.

---
### Task 1: Baselines B0 and B1

**Files:**
- Move: `tests/models/scenario_builders.py` → `tests/support/scenario_builders.py`
- Move: `tests/models/channel_builders.py` → `tests/support/channel_builders.py`
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]` `pythonpath`)
- Create: `src/baselines/__init__.py`, `src/baselines/method.py`, `src/baselines/trajectories.py`, `src/baselines/b0.py`, `src/baselines/b1.py`
- Modify: `experiments/calibrate_v0.py` (import the zone tour instead of defining it)
- Test: `tests/baselines/test_baselines.py`; modify `tests/models/test_calibration.py`

**Interfaces:**
- Consumes (sub-project A): `models.evaluate.EvaluationCounter`, `models.paths.{CandidatePath, GROUND, candidate_paths, radio_links}`, `models.plan.{Plan, EqualSplitBacklogged, stationary_trajectory}`, `models.channel.{LinkChannel, LinkId, build_link_channels}`, `models.simulator.simulate`, `models.scenario.Scenario`.
- Produces: `baselines.method.Method` protocol with `name: str`, `uses_uav: bool`, `trajectory(scenario) -> np.ndarray`, `plan(scenario, candidates, counter) -> Plan`; `baselines.METHODS: dict[str, Method]` with keys `"B0"`, `"B1"`; `baselines.get_method(name) -> Method` (raises `ValueError("unknown method ...")`); frozen dataclasses `baselines.b0.B0` (`uses_uav=False`) and `baselines.b1.B1` (`uses_uav=True`); `baselines.b1.estimated_best_candidate(scenario, candidates, trajectory, channels, alert_id) -> int | None`; `baselines.trajectories.zone_tour_trajectory(scenario) -> np.ndarray` (moved verbatim from `experiments/calibrate_v0.py`).

- [ ] **Step 1: Move the shared test builders onto the pytest path**

```bash
mkdir -p tests/support
git mv tests/models/scenario_builders.py tests/support/scenario_builders.py
git mv tests/models/channel_builders.py tests/support/channel_builders.py
```

In `pyproject.toml`, replace `pythonpath = ["src"]` with `pythonpath = ["src", "tests/support"]`. Existing imports such as `from scenario_builders import make_alert` keep working from any test directory.

Run: `uv run --extra test pytest -q`

Expected: 88 passed, 1 skipped.

- [ ] **Step 2: Write the failing baseline tests**

```python
# tests/baselines/test_baselines.py
import copy

import numpy as np
import pytest

from baselines import METHODS, get_method
from baselines.b0 import B0
from baselines.b1 import B1, estimated_best_candidate
from baselines.trajectories import zone_tour_trajectory
from models.channel import build_link_channels
from models.evaluate import EvaluationCounter
from models.paths import GROUND, candidate_paths, radio_links
from models.plan import EqualSplitBacklogged, stationary_trajectory, validate_plan
from models.scenario import scenario_from_dict
from scenario_builders import BASE_SCENARIO, make_alert


def _scenario(alerts):
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = alerts
    return scenario_from_dict(raw)


def test_registry_names_methods_and_rejects_unknown_names():
    assert set(METHODS) == {"B0", "B1"}
    assert get_method("B1").uses_uav is True
    assert get_method("B0").uses_uav is False
    with pytest.raises(ValueError, match="unknown method 'B9'"):
        get_method("B9")


def test_b0_uses_first_ground_candidate_and_leaves_unreachable_sources_unserved():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 10, 100000), make_alert("alert-0001", "s01", 0, 10, 100000)])
    candidates = candidate_paths(scenario)
    counter = EvaluationCounter()
    plan = B0().plan(scenario, candidates, counter)
    assert plan.path_choice == {"alert-0000": 0, "alert-0001": None}
    assert candidates["alert-0000"][0].kind == GROUND
    assert isinstance(plan.bandwidth, EqualSplitBacklogged)
    assert np.array_equal(plan.trajectory, stationary_trajectory(scenario))
    assert validate_plan(scenario, candidates, plan) == []
    assert counter.calls == 0


def test_b1_chooses_earliest_estimated_delivery():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 20, 200000)])
    candidates = candidate_paths(scenario)
    trajectory = B1().trajectory(scenario)
    channels = build_link_channels(scenario, radio_links(candidates), trajectory, None)
    assert [(path.kind, path.entry) for path in candidates["alert-0000"]] == [("ground", "n1"), ("ground", "n0"), ("uav", "n1")]
    assert estimated_best_candidate(scenario, candidates, trajectory, channels, "alert-0000") == 0


def test_b1_leaves_hopeless_alerts_unserved_and_plans_validly():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 20, 200000), make_alert("alert-0001", "s01", 0, 2, 1000000)])
    candidates = candidate_paths(scenario)
    counter = EvaluationCounter()
    plan = B1().plan(scenario, candidates, counter)
    assert plan.path_choice == {"alert-0000": 0, "alert-0001": None}
    assert np.array_equal(plan.trajectory, zone_tour_trajectory(scenario))
    assert validate_plan(scenario, candidates, plan) == []
    assert counter.calls == 0
```

Append to `tests/models/test_calibration.py`:

```python
def test_calibration_uses_the_baseline_zone_tour():
    from baselines import trajectories

    assert calibrate.zone_tour_trajectory is trajectories.zone_tour_trajectory
```

- [ ] **Step 3: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/baselines/test_baselines.py tests/models/test_calibration.py -v`

Expected: `tests/baselines/test_baselines.py` fails to collect with `ModuleNotFoundError: No module named 'baselines'`; `test_calibration_uses_the_baseline_zone_tour` fails with the same error.

- [ ] **Step 4: Implement the baselines package**

```python
# src/baselines/method.py
from typing import Mapping, Protocol

import numpy as np

from models.evaluate import EvaluationCounter
from models.paths import CandidatePath
from models.plan import Plan
from models.scenario import Scenario


class Method(Protocol):
    """A planning method; the runner scores every plan with models.evaluate."""

    name: str
    uses_uav: bool

    def trajectory(self, scenario: Scenario) -> np.ndarray: ...

    def plan(
        self,
        scenario: Scenario,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        counter: EvaluationCounter,
    ) -> Plan: ...
```

```python
# src/baselines/trajectories.py
import math

import numpy as np

from models.scenario import Scenario


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
```

```python
# src/baselines/b0.py
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from models.evaluate import EvaluationCounter
from models.paths import GROUND, CandidatePath
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import Scenario


@dataclass(frozen=True)
class B0:
    """No UAV: lowest-cost ground path per alert, EDF, equal bandwidth split."""

    name: str = "B0"
    uses_uav: bool = False

    def trajectory(self, scenario: Scenario) -> np.ndarray:
        return stationary_trajectory(scenario)

    def plan(
        self,
        scenario: Scenario,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        counter: EvaluationCounter,
    ) -> Plan:
        choice = {
            alert.id: next((index for index, path in enumerate(candidates[alert.id]) if path.kind == GROUND), None)
            for alert in scenario.alerts
        }
        return Plan(self.trajectory(scenario), choice, EqualSplitBacklogged())
```

```python
# src/baselines/b1.py
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from models.channel import LinkChannel, LinkId, build_link_channels
from models.evaluate import EvaluationCounter
from models.paths import CandidatePath, radio_links
from models.plan import EqualSplitBacklogged, Plan
from models.scenario import Scenario
from models.simulator import simulate

from .trajectories import zone_tour_trajectory


@dataclass(frozen=True)
class B1:
    """Fixed zone-tour trajectory, path with the earliest estimated delivery, EDF, equal split."""

    name: str = "B1"
    uses_uav: bool = True

    def trajectory(self, scenario: Scenario) -> np.ndarray:
        return zone_tour_trajectory(scenario)

    def plan(
        self,
        scenario: Scenario,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        counter: EvaluationCounter,
    ) -> Plan:
        trajectory = self.trajectory(scenario)
        channels = build_link_channels(scenario, radio_links(candidates), trajectory, None)
        choice = {
            alert.id: estimated_best_candidate(scenario, candidates, trajectory, channels, alert.id)
            for alert in scenario.alerts
        }
        return Plan(trajectory, choice, EqualSplitBacklogged())


def estimated_best_candidate(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    trajectory: np.ndarray,
    channels: Mapping[LinkId, LinkChannel],
    alert_id: str,
) -> int | None:
    """Simulate the alert alone on expected-rate channels; earliest delivery wins, ties go to the lower index."""
    alone = replace(scenario, alerts=(scenario.alert_by_id[alert_id],))
    best: tuple[int, int] | None = None
    for index in range(len(candidates[alert_id])):
        plan = Plan(trajectory, {alert_id: index}, EqualSplitBacklogged())
        slot = simulate(alone, candidates, plan, channels).delivery_slot[alert_id]
        if slot is not None and (best is None or slot < best[0]):
            best = (slot, index)
    return None if best is None else best[1]
```

```python
# src/baselines/__init__.py
"""Baseline planning methods B0 and B1."""

from .b0 import B0
from .b1 import B1
from .method import Method

METHODS: dict[str, Method] = {method.name: method for method in (B0(), B1())}


def get_method(name: str) -> Method:
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}; known methods: {sorted(METHODS)}")
    return METHODS[name]
```

- [ ] **Step 5: Make the calibration script import the zone tour**

Replace the whole content of `experiments/calibrate_v0.py` with:

```python
# experiments/calibrate_v0.py
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Callable, Mapping

from baselines.trajectories import zone_tour_trajectory
from models.evaluate import REALIZED, evaluate
from models.paths import GROUND, VIA_UAV, CandidatePath, candidate_paths
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import Scenario, load_scenario

LOWER_BOUND = 0.05
UPPER_BOUND = 0.95


def first_index(paths: tuple[CandidatePath, ...], kind: str) -> int | None:
    return next((index for index, path in enumerate(paths) if path.kind == kind), None)


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
```

- [ ] **Step 6: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/baselines/test_baselines.py tests/models/test_calibration.py -v`

Expected: 8 passed.

Run: `uv run --extra test pytest -q`

Expected: 93 passed, 1 skipped.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml tests/support tests/models tests/baselines src/baselines experiments/calibrate_v0.py
git commit -m "feat: add baselines B0 and B1"
```

### Task 2: Per-realization LP bound and HiGHS solvers

**Files:**
- Create: `src/bounds/__init__.py`, `src/bounds/formulation.py`, `src/bounds/solve.py`
- Test: `tests/bounds/test_formulation.py`

**Interfaces:**
- Consumes: `models.channel.{ACCESS, BACKHAUL, LinkChannel, LinkId, build_link_channels, channel_uniforms, rate_bps}`, `models.paths.{GROUND, CandidatePath, candidate_paths, radio_links}`, `models.evaluate.{REALIZED, evaluate}`, test helper `channel_builders.constant_channels`.
- Produces: `bounds.formulation.rate_tangent(bandwidth_hz, snr_per_hz) -> (value, slope)`; `tangent_points(b_tot_hz, tangents) -> tuple[float, ...]`; `restrict_to_ground(candidates) -> dict[str, tuple[CandidatePath, ...]]`; `realized_snr(channel, slot) -> float` (raises `ValueError` for fractional LoS weights); frozen dataclass `BoundModel(objective, a_ub, b_ub, a_eq, b_eq, upper, integrality, columns)` with properties `variables`, `rows`, `nonzeros`, where `columns` maps `("z", alert)`, `("x", alert, index)`, `("f", alert, index, hop, slot)`, `("B", alert, index, hop, slot)`, `("b", link, slot)` to column indices; `build_bound_model(scenario, candidates, channels, tangents) -> BoundModel` (a minimization of `-sum z`). `bounds.solve.OPTIMAL = "optimal"`, `TIME_LIMIT = "time_limit"`; frozen dataclass `BoundResult(value, dual_bound, status, runtime_s, variables, rows, nonzeros, solution)` with `optimal`; `solve_lp(model) -> BoundResult`; `solve_milp(model, time_limit_s) -> BoundResult`; `highs_version() -> str`.

- [ ] **Step 1: Write the failing bound tests**

```python
# tests/bounds/test_formulation.py
import copy

import numpy as np
import pytest

from bounds.formulation import build_bound_model, rate_tangent, restrict_to_ground, tangent_points
from bounds.solve import OPTIMAL, highs_version, solve_lp, solve_milp
from models.channel import LinkChannel, build_link_channels, channel_uniforms, rate_bps
from models.evaluate import REALIZED, evaluate
from models.paths import candidate_paths, radio_links
from models.plan import EqualSplitBacklogged, Plan, StaticSchedule, stationary_trajectory
from models.scenario import UAV_ID, scenario_from_dict
from channel_builders import constant_channels
from scenario_builders import BASE_SCENARIO, make_alert


def _scenario(alerts):
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = alerts
    return scenario_from_dict(raw)


def test_tangent_envelope_lies_above_the_rate():
    for snr in (1e3, 1e6, 1e9):
        for point in tangent_points(1e6, 10):
            value, slope = rate_tangent(point, snr)
            assert value == pytest.approx(rate_bps(point, snr))
            for bandwidth in np.geomspace(1e2, 1e6, 60):
                assert value + slope * (bandwidth - point) >= rate_bps(bandwidth, snr) - 1e-6


@pytest.mark.parametrize(("deadline", "size", "expected"), [(2, 1000000, 0.5), (3, 1000000, 1.0), (2, 500000, 1.0)])
def test_single_ground_path_bound_matches_store_and_forward_capacity(deadline, size, expected):
    scenario = _scenario([make_alert("alert-0000", "s00", 0, deadline, size)])
    only_n1 = {"alert-0000": (candidate_paths(scenario)["alert-0000"][0],)}
    assert only_n1["alert-0000"][0].entry == "n1"
    model = build_bound_model(scenario, only_n1, constant_channels(scenario, only_n1, 1e6), tangents=10)
    result = solve_lp(model)
    assert result.status == OPTIMAL
    assert result.value == pytest.approx(expected, abs=1e-7)


def test_ground_only_bound_uses_no_uav_link_and_milp_stays_below_lp():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 6, 900000), make_alert("alert-0001", "s00", 0, 6, 900000)])
    candidates = restrict_to_ground(candidate_paths(scenario))
    model = build_bound_model(scenario, candidates, constant_channels(scenario, candidates, 1e6), tangents=10)
    assert not any(UAV_ID in key[1] for key in model.columns if key[0] == "b")
    lp = solve_lp(model)
    milp = solve_milp(model, time_limit_s=30)
    assert milp.status == OPTIMAL
    assert milp.value <= lp.value + 1e-6
    assert milp.dual_bound >= milp.value - 1e-6


def test_bound_rejects_expected_rate_channels():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 6, 100000)])
    candidates = candidate_paths(scenario)
    channels = build_link_channels(scenario, radio_links(candidates), stationary_trajectory(scenario), None)
    with pytest.raises(ValueError, match="realized channels"):
        build_bound_model(scenario, candidates, channels, tangents=10)


def test_bound_is_at_least_every_evaluated_random_plan():
    scenario = _scenario(
        [
            make_alert("alert-0000", "s00", 0, 12, 400000),
            make_alert("alert-0001", "s00", 3, 20, 700000),
            make_alert("alert-0002", "s01", 1, 18, 50000),
            make_alert("alert-0003", "s01", 6, 20, 90000),
        ]
    )
    candidates = candidate_paths(scenario)
    links = radio_links(candidates)
    rng = np.random.default_rng(7)
    for trial in range(8):
        steps = rng.uniform(-35.0, 35.0, size=(10, 2))
        outward = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
        trajectory = np.asarray(scenario.uav.start_xy_m) + np.vstack([outward, outward[-2::-1]])
        choice = {alert.id: int(rng.integers(0, len(candidates[alert.id]))) for alert in scenario.alerts}
        policy = EqualSplitBacklogged() if trial % 2 == 0 else StaticSchedule({link: np.full(20, 1e6 / len(links)) for link in links})
        if isinstance(policy, StaticSchedule):
            downlinks = [link for link in links if link[0] == "down"]
            for link in downlinks[2:]:
                policy.bandwidth_hz[link][:] = 0.0
        realization = int(rng.integers(0, 5))
        result = evaluate(scenario, Plan(trajectory, choice, policy), REALIZED, (realization,), candidates=candidates)
        assert result.feasible, result.violations
        channels = build_link_channels(scenario, links, trajectory, channel_uniforms(scenario, realization))
        bound = solve_lp(build_bound_model(scenario, candidates, channels, tangents=10))
        assert bound.value >= sum(result.realizations[0].timely.values()) - 1e-6


def test_highs_version_is_reported():
    major, minor, patch = highs_version().split(".")
    assert int(major) >= 1
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/bounds/test_formulation.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'bounds'`.

- [ ] **Step 3: Implement the formulation and solvers**

```python
# src/bounds/__init__.py
"""Per-realization LP upper bound and MILP cross-check."""
```

```python
# src/bounds/formulation.py
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from models.channel import ACCESS, BACKHAUL, LinkChannel, LinkId
from models.paths import GROUND, CandidatePath
from models.scenario import Scenario


def rate_tangent(bandwidth_hz: float, snr_per_hz: float) -> tuple[float, float]:
    """Value and slope of R(b) = b*log2(1 + s/b) at b > 0."""
    ratio = snr_per_hz / bandwidth_hz
    value = bandwidth_hz * math.log2(1.0 + ratio)
    slope = math.log2(1.0 + ratio) - ratio / ((1.0 + ratio) * math.log(2.0))
    return value, slope


def tangent_points(b_tot_hz: float, tangents: int) -> tuple[float, ...]:
    return tuple(b_tot_hz * 2.0 ** (-index) for index in range(tangents))


def restrict_to_ground(
    candidates: Mapping[str, tuple[CandidatePath, ...]],
) -> dict[str, tuple[CandidatePath, ...]]:
    return {alert_id: tuple(path for path in paths if path.kind == GROUND) for alert_id, paths in candidates.items()}


def realized_snr(channel: LinkChannel, slot: int) -> float:
    weight = float(channel.los_weight[slot])
    if weight not in (0.0, 1.0):
        raise ValueError("the bound needs realized channels with LoS weight 0 or 1")
    return float(channel.snr_los_per_hz[slot] if weight == 1.0 else channel.snr_nlos_per_hz[slot])


@dataclass(frozen=True)
class BoundModel:
    """minimize objective @ v  s.t.  a_ub v <= b_ub, a_eq v = b_eq, 0 <= v <= upper."""

    objective: np.ndarray
    a_ub: csr_matrix
    b_ub: np.ndarray
    a_eq: csr_matrix
    b_eq: np.ndarray
    upper: np.ndarray
    integrality: np.ndarray
    columns: Mapping[tuple, int]

    @property
    def variables(self) -> int:
        return len(self.objective)

    @property
    def rows(self) -> int:
        return self.a_ub.shape[0] + self.a_eq.shape[0]

    @property
    def nonzeros(self) -> int:
        return self.a_ub.nnz + self.a_eq.nnz


class _Rows:
    def __init__(self) -> None:
        self.row_index: list[int] = []
        self.column_index: list[int] = []
        self.values: list[float] = []
        self.rhs: list[float] = []

    def add(self, coefficients: Mapping[int, float], rhs: float) -> None:
        row = len(self.rhs)
        for column, value in coefficients.items():
            self.row_index.append(row)
            self.column_index.append(column)
            self.values.append(value)
        self.rhs.append(rhs)

    def matrix(self, width: int) -> tuple[csr_matrix, np.ndarray]:
        shape = (len(self.rhs), width)
        matrix = coo_matrix((self.values, (self.row_index, self.column_index)), shape=shape).tocsr()
        return matrix, np.asarray(self.rhs, dtype=float)


def build_bound_model(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    channels: Mapping[LinkId, LinkChannel],
    tangents: int,
) -> BoundModel:
    slot_s = scenario.time.slot_s
    access_s = scenario.time.access_fraction * slot_s
    columns: dict[tuple, int] = {}

    def column(key: tuple) -> int:
        if key not in columns:
            columns[key] = len(columns)
        return columns[key]

    upper_rows, equal_rows = _Rows(), _Rows()
    link_flows: dict[tuple[LinkId, int], list[int]] = {}
    for alert in scenario.alerts:
        paths = candidates[alert.id]
        z = column(("z", alert.id))
        if paths:
            upper_rows.add({column(("x", alert.id, index)): 1.0 for index in range(len(paths))}, 1.0)
        delivered = {z: float(alert.size_bits)}
        for index, path in enumerate(paths):
            hops = len(path.links)
            for hop in range(hops):
                for slot in range(alert.release_slot + hop, alert.deadline_slot):
                    link_flows.setdefault((path.links[hop], slot), []).append(column(("f", alert.id, index, hop, slot)))
            for hop in range(hops):
                start = alert.release_slot + hop
                if start >= alert.deadline_slot:
                    break
                for slot in range(start, alert.deadline_slot):
                    flow = columns[("f", alert.id, index, hop, slot)]
                    buffer = column(("B", alert.id, index, hop, slot))
                    upper_rows.add({flow: 1.0, buffer: -1.0}, 0.0)
                    if slot + 1 < alert.deadline_slot:
                        balance = {column(("B", alert.id, index, hop, slot + 1)): 1.0, buffer: -1.0, flow: 1.0}
                        if hop > 0:
                            balance[columns[("f", alert.id, index, hop - 1, slot)]] = -1.0
                        equal_rows.add(balance, 0.0)
                first = columns[("B", alert.id, index, hop, start)]
                if hop == 0:
                    equal_rows.add({first: 1.0, columns[("x", alert.id, index)]: -float(alert.size_bits)}, 0.0)
                else:
                    equal_rows.add({first: 1.0, columns[("f", alert.id, index, hop - 1, start - 1)]: -1.0}, 0.0)
            last = hops - 1
            for slot in range(alert.release_slot + last, alert.deadline_slot):
                delivered[columns[("f", alert.id, index, last, slot)]] = -1.0
        upper_rows.add(delivered, 0.0)

    spectrum: dict[tuple[str, int], list[int]] = {}
    points = tangent_points(scenario.spectrum.b_tot_hz, tangents)
    for link, slot in sorted(link_flows):
        load = {flow: 1.0 for flow in link_flows[(link, slot)]}
        if link[0] == BACKHAUL:
            upper_rows.add(load, slot_s * scenario.backhaul_capacity_bps(link[1], link[2]))
            continue
        duration = access_s if link[0] == ACCESS else slot_s - access_s
        snr = realized_snr(channels[link], slot)
        if snr <= 0.0:
            upper_rows.add(load, 0.0)
            continue
        bandwidth = column(("b", link, slot))
        spectrum.setdefault((link[0], slot), []).append(bandwidth)
        for point in points:
            value, slope = rate_tangent(point, snr)
            upper_rows.add({**load, bandwidth: -duration * slope}, duration * (value - slope * point))
    for key in sorted(spectrum):
        upper_rows.add({index: 1.0 for index in spectrum[key]}, scenario.spectrum.b_tot_hz)

    width = len(columns)
    objective = np.zeros(width)
    upper = np.full(width, np.inf)
    integrality = np.zeros(width)
    for key, index in columns.items():
        if key[0] == "z":
            objective[index] = -1.0
        if key[0] in ("x", "z"):
            upper[index] = 1.0
            integrality[index] = 1
    a_ub, b_ub = upper_rows.matrix(width)
    a_eq, b_eq = equal_rows.matrix(width)
    return BoundModel(objective, a_ub, b_ub, a_eq, b_eq, upper, integrality, columns)
```

```python
# src/bounds/solve.py
from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .formulation import BoundModel

OPTIMAL = "optimal"
TIME_LIMIT = "time_limit"


@dataclass(frozen=True)
class BoundResult:
    value: float
    dual_bound: float
    status: str
    runtime_s: float
    variables: int
    rows: int
    nonzeros: int
    solution: np.ndarray | None

    @property
    def optimal(self) -> bool:
        return self.status == OPTIMAL


def solve_lp(model: BoundModel) -> BoundResult:
    started = time.perf_counter()
    result = linprog(
        model.objective,
        A_ub=model.a_ub,
        b_ub=model.b_ub,
        A_eq=model.a_eq if model.a_eq.shape[0] else None,
        b_eq=model.b_eq if model.a_eq.shape[0] else None,
        bounds=np.column_stack([np.zeros(model.variables), model.upper]),
        method="highs",
    )
    runtime = time.perf_counter() - started
    solved = result.status == 0
    value = -float(result.fun) if solved else math.nan
    return BoundResult(
        value=value,
        dual_bound=value,
        status=OPTIMAL if solved else f"status {result.status}: {result.message}",
        runtime_s=runtime,
        variables=model.variables,
        rows=model.rows,
        nonzeros=model.nonzeros,
        solution=np.asarray(result.x) if solved else None,
    )


def solve_milp(model: BoundModel, time_limit_s: float) -> BoundResult:
    constraints = [LinearConstraint(model.a_ub, -np.inf, model.b_ub)]
    if model.a_eq.shape[0]:
        constraints.append(LinearConstraint(model.a_eq, model.b_eq, model.b_eq))
    started = time.perf_counter()
    result = milp(
        model.objective,
        constraints=constraints,
        integrality=model.integrality,
        bounds=Bounds(np.zeros(model.variables), model.upper),
        options={"time_limit": time_limit_s},
    )
    runtime = time.perf_counter() - started
    value = -float(result.fun) if result.x is not None else math.nan
    dual = getattr(result, "mip_dual_bound", None)
    if result.status == 0:
        status = OPTIMAL
    elif result.status == 1:
        status = TIME_LIMIT
    else:
        status = f"status {result.status}: {result.message}"
    return BoundResult(
        value=value,
        dual_bound=-float(dual) if dual is not None else value,
        status=status,
        runtime_s=runtime,
        variables=model.variables,
        rows=model.rows,
        nonzeros=model.nonzeros,
        solution=np.asarray(result.x) if result.x is not None else None,
    )


def highs_version() -> str:
    from scipy.optimize._highspy import _core

    return f"{_core.HIGHS_VERSION_MAJOR}.{_core.HIGHS_VERSION_MINOR}.{_core.HIGHS_VERSION_PATCH}"
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/bounds/test_formulation.py -v`

Expected: 8 passed.

Run: `uv run --extra test pytest -q`

Expected: 101 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/bounds tests/bounds/test_formulation.py
git commit -m "feat: add per-realization LP upper bound"
```

### Task 3: MILP cross-check utilities

**Files:**
- Create: `src/bounds/crosscheck.py`
- Test: `tests/bounds/test_crosscheck.py`

**Interfaces:**
- Consumes: Task 2 `BoundModel`, `rate_tangent`, `realized_snr`, `tangent_points`, `solve_lp`, `solve_milp`; `models.plan.{Plan, StaticSchedule, validate_plan}`; Task 1 `zone_tour_trajectory` (tests only).
- Produces: `flow_variable_count(scenario, candidates) -> int`; `select_positions(available, count) -> list[int]` (`floor(i*(available-1)/(count-1) + 0.5)`, `[0]` for one); `select_crosscheck_scenarios(scored: Sequence[tuple[int, str]], count) -> list[str]`; `reduce_scenario(scenario, alert_count) -> Scenario` (keeps the smallest alert IDs); `plan_from_solution(scenario, candidates, model, solution, trajectory) -> Plan` (paths with `x >= 0.5`, `StaticSchedule` from `b`, rescaled to `B_tot` per half-slot, at most `max_active_downlinks` downlinks per slot); `model_snr_values(model, channels) -> list[float]`; `tangent_max_overestimate(snr_values, b_tot_hz, tangents, grid_points=2000) -> float`.

- [ ] **Step 1: Write the failing cross-check tests**

```python
# tests/bounds/test_crosscheck.py
import copy

import numpy as np
import pytest

from bounds.crosscheck import (
    flow_variable_count,
    model_snr_values,
    plan_from_solution,
    reduce_scenario,
    select_crosscheck_scenarios,
    select_positions,
    tangent_max_overestimate,
)
from bounds.formulation import build_bound_model
from bounds.solve import solve_lp, solve_milp
from baselines.trajectories import zone_tour_trajectory
from models.channel import build_link_channels, channel_uniforms
from models.evaluate import REALIZED, evaluate
from models.paths import candidate_paths, radio_links
from models.plan import validate_plan
from models.scenario import scenario_from_dict
from scenario_builders import BASE_SCENARIO, make_alert


def test_selection_positions_and_order():
    assert select_positions(30, 5) == [0, 7, 15, 22, 29]
    assert select_positions(3, 1) == [0]
    with pytest.raises(ValueError):
        select_positions(3, 4)
    assert select_crosscheck_scenarios([(9, "c"), (1, "b"), (1, "a")], 2) == ["a", "c"]


def test_reduce_scenario_keeps_smallest_alert_ids():
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = [make_alert(f"alert-000{index}", "s00", 0, 10, 1000) for index in (3, 1, 2)]
    reduced = reduce_scenario(scenario_from_dict(raw), 2)
    assert [alert.id for alert in reduced.alerts] == ["alert-0001", "alert-0002"]
    assert flow_variable_count(reduced, candidate_paths(reduced)) > 0


def test_milp_plan_is_valid_and_bounded_by_the_milp_bound():
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = [
        make_alert("alert-0000", "s00", 0, 12, 400000),
        make_alert("alert-0001", "s01", 0, 20, 60000),
        make_alert("alert-0002", "s00", 2, 15, 300000),
    ]
    scenario = scenario_from_dict(raw)
    candidates = candidate_paths(scenario)
    trajectory = zone_tour_trajectory(scenario)
    channels = build_link_channels(scenario, radio_links(candidates), trajectory, channel_uniforms(scenario, 0))
    model = build_bound_model(scenario, candidates, channels, tangents=10)
    lp = solve_lp(model)
    milp = solve_milp(model, time_limit_s=60)
    plan = plan_from_solution(scenario, candidates, model, milp.solution, trajectory)
    assert validate_plan(scenario, candidates, plan) == []
    evaluated = evaluate(scenario, plan, REALIZED, (0,), candidates=candidates)
    value = sum(evaluated.realizations[0].timely.values())
    assert value <= milp.dual_bound + 1e-6 <= lp.value + 2e-6
    overestimate = tangent_max_overestimate(model_snr_values(model, channels), scenario.spectrum.b_tot_hz, 10)
    assert 0.0 <= overestimate < 0.25
    assert tangent_max_overestimate([1e6], 1e6, 20) <= tangent_max_overestimate([1e6], 1e6, 10)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/bounds/test_crosscheck.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'bounds.crosscheck'`.

- [ ] **Step 3: Implement the cross-check utilities**

```python
# src/bounds/crosscheck.py
from dataclasses import replace
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from models.channel import ACCESS, DOWNLINK, LinkChannel, LinkId
from models.paths import CandidatePath, radio_links
from models.plan import Plan, StaticSchedule
from models.scenario import Scenario

from .formulation import BoundModel, rate_tangent, realized_snr, tangent_points


def flow_variable_count(scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]]) -> int:
    return sum(
        len(path.links) * (alert.deadline_slot - alert.release_slot)
        for alert in scenario.alerts
        for path in candidates[alert.id]
    )


def select_positions(available: int, count: int) -> list[int]:
    if not 1 <= count <= available:
        raise ValueError(f"cannot select {count} of {available} scenarios")
    if count == 1:
        return [0]
    return [math.floor(index * (available - 1) / (count - 1) + 0.5) for index in range(count)]


def select_crosscheck_scenarios(scored: Sequence[tuple[int, str]], count: int) -> list[str]:
    ordered = sorted(scored)
    return [ordered[position][1] for position in select_positions(len(ordered), count)]


def reduce_scenario(scenario: Scenario, alert_count: int) -> Scenario:
    kept = tuple(sorted(scenario.alerts, key=lambda alert: alert.id)[:alert_count])
    return replace(scenario, alerts=kept)


def plan_from_solution(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    model: BoundModel,
    solution: np.ndarray,
    trajectory: np.ndarray,
) -> Plan:
    choice: dict[str, int | None] = {}
    for alert in scenario.alerts:
        chosen = [
            index for index in range(len(candidates[alert.id])) if solution[model.columns[("x", alert.id, index)]] >= 0.5
        ]
        choice[alert.id] = chosen[0] if chosen else None
    num_slots = scenario.time.num_slots
    bandwidth = {link: np.zeros(num_slots) for link in radio_links(candidates)}
    for key, index in model.columns.items():
        if key[0] == "b":
            bandwidth[key[1]][key[2]] = max(0.0, float(solution[index]))
    total = scenario.spectrum.b_tot_hz
    cap = scenario.uav.max_active_downlinks
    for slot in range(num_slots):
        for kind in (ACCESS, DOWNLINK):
            group = [link for link in bandwidth if link[0] == kind]
            used = sum(bandwidth[link][slot] for link in group)
            if used > total:
                for link in group:
                    bandwidth[link][slot] *= total / used
        active = sorted((bandwidth[link][slot], link) for link in bandwidth if link[0] == DOWNLINK and bandwidth[link][slot] > 0.0)
        for _, link in active[: max(0, len(active) - cap)]:
            bandwidth[link][slot] = 0.0
    return Plan(trajectory, choice, StaticSchedule(bandwidth))


def model_snr_values(model: BoundModel, channels: Mapping[LinkId, LinkChannel]) -> list[float]:
    return [realized_snr(channels[key[1]], key[2]) for key in model.columns if key[0] == "b"]


def tangent_max_overestimate(snr_values: Iterable[float], b_tot_hz: float, tangents: int, grid_points: int = 2000) -> float:
    """Largest relative overestimate of the tangent envelope over b in [B_tot/512, B_tot]."""
    grid = np.geomspace(b_tot_hz / 512.0, b_tot_hz, grid_points)
    points = tangent_points(b_tot_hz, tangents)
    worst = 0.0
    for snr in sorted({float(value) for value in snr_values if value > 0.0}):
        true_rate = grid * np.log2(1.0 + snr / grid)
        envelope = np.min(
            [value + slope * (grid - point) for point in points for value, slope in [rate_tangent(point, snr)]],
            axis=0,
        )
        worst = max(worst, float(np.max((envelope - true_rate) / true_rate)))
    return worst
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/bounds/test_crosscheck.py -v`

Expected: 3 passed.

Run: `uv run --extra test pytest -q`

Expected: 104 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/bounds/crosscheck.py tests/bounds/test_crosscheck.py
git commit -m "feat: add MILP cross-check utilities"
```

### Task 4: Backhaul-stressed scenario family `v0-bh50`

**Files:**
- Create: `configs/scenarios/v0-bh50.yaml`
- Test: `tests/data/test_scenario_configs.py`
- Create (generated, committed): `data/manifests/scenarios_v0-bh50.csv`, `results/calibration/v0-bh50.json`

**Interfaces:**
- Consumes: the sub-project A `scenarios` stage and `experiments/calibrate_v0.py` (Task 1 version).
- Produces: scenario set `v0-bh50` in `data/processed/scenarios/v0-bh50/` (ignored) with manifest `data/manifests/scenarios_v0-bh50.csv`, used by Tasks 7 and 8.

Steps 5–7 need the pinned data from the `paper` profile and the committed `v0` scenario files under `data/processed/scenarios/v0/`.

- [ ] **Step 1: Write the failing config test**

```python
# tests/data/test_scenario_configs.py
from pathlib import Path

import yaml


def test_backhaul_stressed_family_differs_only_in_set_id_and_capacity():
    base = yaml.safe_load(Path("configs/scenarios/v0.yaml").read_text(encoding="utf-8"))
    stressed = yaml.safe_load(Path("configs/scenarios/v0-bh50.yaml").read_text(encoding="utf-8"))
    changed = ("set_id", "backhaul_capacity_bps")
    assert (stressed["set_id"], stressed["backhaul_capacity_bps"]) == ("v0-bh50", 50000.0)
    assert {key: value for key, value in stressed.items() if key not in changed} == {
        key: value for key, value in base.items() if key not in changed
    }
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `uv run --extra test pytest tests/data/test_scenario_configs.py -v`

Expected: FAIL with `FileNotFoundError: [Errno 2] No such file or directory: 'configs/scenarios/v0-bh50.yaml'`.

- [ ] **Step 3: Create the stressed configuration**

```bash
sed -e 's/^set_id: v0$/set_id: v0-bh50/' -e 's/^backhaul_capacity_bps: 1000000.0$/backhaul_capacity_bps: 50000.0/' configs/scenarios/v0.yaml > configs/scenarios/v0-bh50.yaml
```

Run: `uv run --extra test pytest tests/data/test_scenario_configs.py -v`

Expected: 1 passed.

Run: `uv run --extra test pytest -q`

Expected: 105 passed, 1 skipped.

- [ ] **Step 4: Generate `v0-bh50` twice and confirm reproducibility**

```bash
uv run python -m data.cli scenarios --config configs/scenarios/v0-bh50.yaml
cp data/manifests/scenarios_v0-bh50.csv /tmp/scenarios_v0-bh50.first.csv
uv run python -m data.cli scenarios --config configs/scenarios/v0-bh50.yaml
cmp /tmp/scenarios_v0-bh50.first.csv data/manifests/scenarios_v0-bh50.csv && echo REPRODUCIBLE
ls data/processed/scenarios/v0-bh50 | wc -l
```
Expected: `REPRODUCIBLE` and `36`.

- [ ] **Step 5: Confirm the moved zone tour reproduces the committed `v0` calibration (spec acceptance 2)**

```bash
uv run python experiments/calibrate_v0.py --output /tmp/v0-calibration-check.json
python3 -c "import json; a=json.load(open('/tmp/v0-calibration-check.json')); b=json.load(open('results/calibration/v0.json')); assert all(a['plans'][n]['per_scenario_timely_ratio']==b['plans'][n]['per_scenario_timely_ratio'] for n in b['plans']); print('IDENTICAL')"
```
Expected: the script prints `{"ground_only": 0.4133, "zone_tour": 0.1036, "passed": true}` and the check prints `IDENTICAL`.

- [ ] **Step 6: Run the calibration check on `v0-bh50`**

Run: `uv run python experiments/calibrate_v0.py --manifest data/manifests/scenarios_v0-bh50.csv --scenario-dir data/processed/scenarios/v0-bh50 --output results/calibration/v0-bh50.json`

Expected: exit code 0 printing `{"ground_only": 0.37, "zone_tour": 0.0878, "passed": true}`.

- [ ] **Step 7: Commit**

```bash
git add configs/scenarios/v0-bh50.yaml tests/data/test_scenario_configs.py data/manifests/scenarios_v0-bh50.csv results/calibration/v0-bh50.json
git commit -m "feat: add backhaul-stressed scenario family"
```

### Task 5: Runner configuration and provenance

**Files:**
- Create: `src/runner/__init__.py`, `src/runner/config.py`, `src/runner/provenance.py`, `configs/experiments/phase1.yaml`
- Test: `tests/runner/test_runner_config.py`, `tests/runner/test_runner_provenance.py`

**Interfaces:**
- Consumes: `baselines.METHODS` (Task 1); `bounds.solve.highs_version` (Task 2).
- Produces: frozen dataclasses `ScenarioSetRef(set_id, manifest: Path, scenario_dir: Path)`, `CrosscheckConfig(set_id, scenario_count, alert_count, realization_id, time_limit_s)`, `BootstrapConfig(resamples, seed, confidence)`, `ExperimentConfig(experiment_id, scenario_sets, split, realization_ids: tuple[int, ...], methods: tuple[str, ...], tangents, crosscheck, bootstrap, workers, output_dir: Path, config_hash)`; `load_experiment_config(path) -> ExperimentConfig` (raises `ValueError` naming `methods`, `unique`, `milp_crosscheck.set_id`, or `realization_ids`). `runner.provenance.SRC_ROOT`, `SOURCE_PACKAGES = ("models", "baselines", "bounds", "runner")`, `source_hash(src_root=SRC_ROOT, packages=SOURCE_PACKAGES) -> str`, `file_sha256(path) -> str`, `git_state(repo_root) -> (commit | None, dirty | None)`, `environment() -> dict` with keys `python, numpy, scipy, highs, platform, cpu_count`.

- [ ] **Step 1: Write the failing configuration and provenance tests**

```python
# tests/runner/test_runner_config.py
from pathlib import Path

import pytest
import yaml

from runner.config import load_experiment_config


def test_phase1_config_values():
    config = load_experiment_config(Path("configs/experiments/phase1.yaml"))
    assert [item.set_id for item in config.scenario_sets] == ["v0", "v0-bh50"]
    assert config.realization_ids == tuple(range(30))
    assert config.methods == ("B0", "B1")
    assert (config.tangents, config.workers, config.crosscheck.alert_count, config.crosscheck.time_limit_s) == (10, 12, 10, 300.0)
    assert (config.bootstrap.resamples, config.bootstrap.seed) == (10000, 20260911)
    assert len(config.config_hash) == 64


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(methods=["B9"]), "methods"),
        (lambda raw: raw["scenario_sets"].append(dict(raw["scenario_sets"][0])), "unique"),
        (lambda raw: raw["milp_crosscheck"].update(set_id="missing"), "milp_crosscheck.set_id"),
        (lambda raw: raw.update(realization_ids={"start": 3, "stop": 3}), "realization_ids"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, change, message):
    raw = yaml.safe_load(Path("configs/experiments/phase1.yaml").read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)
```

```python
# tests/runner/test_runner_provenance.py
from pathlib import Path

from runner.provenance import environment, git_state, source_hash


def test_source_hash_tracks_file_content(tmp_path: Path):
    package = tmp_path / "models"
    package.mkdir()
    module = package / "a.py"
    module.write_text("x = 1\n", encoding="utf-8")
    first = source_hash(tmp_path, ("models",))
    assert first == source_hash(tmp_path, ("models",))
    module.write_text("x = 2\n", encoding="utf-8")
    assert source_hash(tmp_path, ("models",)) != first


def test_environment_and_git_state_outside_a_repository(tmp_path: Path):
    assert set(environment()) == {"python", "numpy", "scipy", "highs", "platform", "cpu_count"}
    assert git_state(tmp_path) == (None, None)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/runner/test_runner_config.py tests/runner/test_runner_provenance.py -v`

Expected: both modules fail to collect with `ModuleNotFoundError: No module named 'runner'`.

- [ ] **Step 3: Implement configuration, provenance, and the Phase 1 config**

```python
# src/runner/__init__.py
"""Experiment runner: tasks, resumable shards, aggregation, and provenance."""
```

```python
# src/runner/config.py
from dataclasses import dataclass
import hashlib
from pathlib import Path

import yaml

from baselines import METHODS


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
class ExperimentConfig:
    experiment_id: str
    scenario_sets: tuple[ScenarioSetRef, ...]
    split: str
    realization_ids: tuple[int, ...]
    methods: tuple[str, ...]
    tangents: int
    crosscheck: CrosscheckConfig
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
    methods = tuple(str(name) for name in raw["methods"])
    unknown = [name for name in methods if name not in METHODS]
    if not methods or unknown:
        raise ValueError(f"methods: unknown or missing methods {unknown}")
    cross = raw["milp_crosscheck"]
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
# src/runner/provenance.py
import hashlib
import os
from pathlib import Path
import platform
import subprocess

import numpy
import scipy

from bounds.solve import highs_version

SRC_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGES = ("models", "baselines", "bounds", "runner")


def source_hash(src_root: Path = SRC_ROOT, packages: tuple[str, ...] = SOURCE_PACKAGES) -> str:
    digest = hashlib.sha256()
    for package in packages:
        for path in sorted((src_root / package).rglob("*.py")):
            digest.update(path.relative_to(src_root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "configs", "experiments"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "highs": highs_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
```

```yaml
# configs/experiments/phase1.yaml
experiment_id: phase1
scenario_sets:
  - set_id: v0
    manifest: data/manifests/scenarios_v0.csv
    scenario_dir: data/processed/scenarios/v0
  - set_id: v0-bh50
    manifest: data/manifests/scenarios_v0-bh50.csv
    scenario_dir: data/processed/scenarios/v0-bh50
split: eval
realization_ids: {start: 0, stop: 30}
methods: [B0, B1]
bounds: {tangents: 10}
milp_crosscheck: {set_id: v0, scenario_count: 5, alert_count: 10, realization_id: 0, time_limit_s: 300}
bootstrap: {resamples: 10000, seed: 20260911, confidence: 0.95}
workers: 12
output_dir: results/phase1
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/runner/test_runner_config.py tests/runner/test_runner_provenance.py -v`

Expected: 7 passed.

Run: `uv run --extra test pytest -q`

Expected: 112 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/runner/__init__.py src/runner/config.py src/runner/provenance.py configs/experiments/phase1.yaml tests/runner/test_runner_config.py tests/runner/test_runner_provenance.py
git commit -m "feat: add experiment configuration and provenance"
```

### Task 6: Experiment tasks and aggregation

**Files:**
- Create: `src/runner/tasks.py`, `src/runner/aggregate.py`
- Test: `tests/runner/test_runner_aggregate.py`

**Interfaces:**
- Consumes: Tasks 1–3 and 5; `models.evaluate.{REALIZED, EvaluationCounter, evaluate}`, `models.scenario.load_scenario`, `models.plan.stationary_trajectory`.
- Produces: `runner.tasks.GROUND_ONLY = "ground_only"`, `TRAJECTORY = "trajectory"`, `CROSSCHECK_TRAJECTORY_METHOD = "B1"`; frozen dataclasses `MethodTask(set_id, scenario_id, scenario_path, method, realization_ids)`, `BoundTask(set_id, scenario_id, scenario_path, variant, trajectory_method, realization_id, tangents)`, `CrosscheckTask(set_id, scenario_id, scenario_path, trajectory_method, alert_count, realization_id, time_limit_s, tangents)`, each with a `key` property (`method__…`, `bound__…__r000`, `milp__…`); `bound_variant(method_name) -> (variant, trajectory_method)`; `task_hash(task, scenario_sha256, source_hash) -> str`; `run_method_task`, `run_bound_task`, `run_crosscheck_task` returning row dicts with the spec Section 8.3 columns (the cross-check row evaluates the MILP path choice with the static bandwidth of the solution and with `EqualSplitBacklogged`, stores both as `milp_plan_static_value` and `milp_plan_equal_split_value`, their maximum as `milp_plan_value`, and B1 on the same reduced instance as `method_value`); `execute(task) -> {"key", "rows", "error", "runtime_s"}`. `runner.aggregate.VALIDITY_TOLERANCE = 1e-6`; `cluster_bootstrap_ci(values_by_cluster, resamples, seed, confidence) -> (low, high)`; `relative_gap(bound_mean, method_mean) -> float | None`; `validity_violations(method_rows, bound_rows) -> list[str]`; `crosscheck_violations(rows) -> list[str]` (requires `max(milp_plan_value, method_value, milp_incumbent) <= milp_dual_bound <= lp_ub` within the tolerance); `summarize(method_rows, bound_rows, bootstrap) -> list[dict]` with the spec `summary.csv` columns.

The task runners are exercised end to end by the experiment tests in Task 7.

- [ ] **Step 1: Write the failing aggregation tests**

```python
# tests/runner/test_runner_aggregate.py
import math

import pytest

from runner.aggregate import (
    cluster_bootstrap_ci,
    crosscheck_violations,
    relative_gap,
    summarize,
    validity_violations,
)
from runner.config import BootstrapConfig


def _method_row(scenario, topology, realization, timely):
    return {
        "set_id": "s", "scenario_id": scenario, "topology_id": topology, "method": "B0", "realization_id": realization,
        "alert_count": 4, "timely_count": timely, "timely_ratio": timely / 4, "source_count": 2, "connected_count": 1,
        "conn_ratio": 0.5, "shortfall": 4 - timely, "planning_runtime_s": 0.1, "evaluation_runtime_s": 0.2, "evaluate_calls": 1,
    }


def _bound_row(scenario, realization, value):
    return {
        "set_id": "s", "scenario_id": scenario, "topology_id": "T", "variant": "ground_only", "trajectory_method": "",
        "realization_id": realization, "value": value, "ratio": value / 4, "status": "optimal", "runtime_s": 0.3,
        "variables": 1, "rows": 1, "nonzeros": 1,
    }


def test_bootstrap_interval_and_gap_helpers():
    assert cluster_bootstrap_ci({"a": [0.4, 0.4], "b": [0.4]}, 50, 1, 0.95) == pytest.approx((0.4, 0.4))
    low, high = cluster_bootstrap_ci({"a": [0.0], "b": [1.0]}, 400, 1, 0.9)
    assert (low, high) == cluster_bootstrap_ci({"a": [0.0], "b": [1.0]}, 400, 1, 0.9)
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    assert relative_gap(0.5, 0.25) == 0.5
    assert relative_gap(0.0, 0.0) is None


def test_summary_means_gaps_and_undefined_gap_count():
    method_rows = [_method_row("A", "T1", 0, 2), _method_row("A", "T1", 1, 4), _method_row("B", "T2", 0, 0), _method_row("B", "T2", 1, 0)]
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

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/runner/test_runner_aggregate.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'runner.aggregate'`.

- [ ] **Step 3: Implement tasks and aggregation**

```python
# src/runner/tasks.py
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

from baselines import get_method
from bounds.crosscheck import model_snr_values, plan_from_solution, reduce_scenario, tangent_max_overestimate
from bounds.formulation import build_bound_model, restrict_to_ground
from bounds.solve import solve_lp, solve_milp
from models.channel import build_link_channels, channel_uniforms
from models.evaluate import REALIZED, EvaluationCounter, evaluate
from models.paths import candidate_paths, radio_links
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import load_scenario

GROUND_ONLY = "ground_only"
TRAJECTORY = "trajectory"
CROSSCHECK_TRAJECTORY_METHOD = "B1"


@dataclass(frozen=True)
class MethodTask:
    set_id: str
    scenario_id: str
    scenario_path: str
    method: str
    realization_ids: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"method__{self.set_id}__{self.scenario_id}__{self.method}"


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


def bound_variant(method_name: str) -> tuple[str, str]:
    method = get_method(method_name)
    return (TRAJECTORY, method.name) if method.uses_uav else (GROUND_ONLY, "")


def task_hash(task: Task, scenario_sha256: str, source_hash: str) -> str:
    payload = {
        "type": type(task).__name__,
        "task": asdict(task),
        "scenario_sha256": scenario_sha256,
        "source_hash": source_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_method_task(task: MethodTask) -> list[dict[str, object]]:
    scenario = load_scenario(Path(task.scenario_path))
    candidates = candidate_paths(scenario)
    method = get_method(task.method)
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
                "method": task.method,
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
    return rows


def run_bound_task(task: BoundTask) -> list[dict[str, object]]:
    scenario = load_scenario(Path(task.scenario_path))
    candidates = candidate_paths(scenario)
    if task.variant == GROUND_ONLY:
        candidates = restrict_to_ground(candidates)
        trajectory = stationary_trajectory(scenario)
    else:
        trajectory = get_method(task.trajectory_method).trajectory(scenario)
    channels = build_link_channels(
        scenario, radio_links(candidates), trajectory, channel_uniforms(scenario, task.realization_id)
    )
    result = solve_lp(build_bound_model(scenario, candidates, channels, task.tangents))
    return [
        {
            "set_id": task.set_id,
            "scenario_id": task.scenario_id,
            "topology_id": scenario.network_id,
            "variant": task.variant,
            "trajectory_method": task.trajectory_method,
            "realization_id": task.realization_id,
            "value": result.value,
            "ratio": result.value / len(scenario.alerts),
            "status": result.status,
            "runtime_s": result.runtime_s,
            "variables": result.variables,
            "rows": result.rows,
            "nonzeros": result.nonzeros,
        }
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


RUNNERS = {MethodTask: run_method_task, BoundTask: run_bound_task, CrosscheckTask: run_crosscheck_task}


def execute(task: Task) -> dict[str, object]:
    started = time.perf_counter()
    try:
        rows, error = RUNNERS[type(task)](task), None
    except Exception as exc:  # recorded in the shard; the experiment reports it and exits non-zero
        rows, error = [], "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return {"key": task.key, "rows": rows, "error": error, "runtime_s": time.perf_counter() - started}
```

```python
# src/runner/aggregate.py
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
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/runner/test_runner_aggregate.py -v`

Expected: 3 passed.

Run: `uv run --extra test pytest -q`

Expected: 115 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/runner/tasks.py src/runner/aggregate.py tests/runner/test_runner_aggregate.py
git commit -m "feat: add experiment tasks and aggregation"
```

### Task 7: Experiment orchestration and entry point

**Files:**
- Create: `src/runner/experiment.py`, `experiments/run_phase1.py`
- Modify: `.gitignore` (append `results/phase1/shards/`)
- Test: `tests/runner/experiment_fixtures.py`, `tests/runner/test_runner_experiment.py`

**Interfaces:**
- Consumes: Tasks 3, 5, 6.
- Produces: `runner.experiment.METHOD_COLUMNS`, `BOUND_COLUMNS`, `CROSSCHECK_COLUMNS`, `SUMMARY_COLUMNS`; frozen dataclass `ScenarioRef(set_id, scenario_id, path, sha256)`; `scenario_refs(config) -> list[ScenarioRef]`; `build_tasks(config, refs) -> list[Task]`; `run_experiment(config, config_path, workers=None, argv=()) -> int` (0 on success; writes `method_realizations.csv`, `bounds.csv`, `milp_crosscheck.csv`, `summary.csv` when no failure, `manifest.json`; prints one JSON line with `task_count`, `executed_task_count`, `skipped_task_count`, `status`). `experiments/run_phase1.py` exposes `main(argv=None) -> int` with `--config` (default `configs/experiments/phase1.yaml`) and `--workers`.

Workers use `multiprocessing.get_context("spawn")`: with `fork`, the tests deadlocked once the parent pytest process had already called HiGHS.

- [ ] **Step 1: Write the fixture and the failing experiment tests**

```python
# tests/runner/experiment_fixtures.py
import copy
import csv
import json
from pathlib import Path

import yaml

from models.scenario import load_scenario
from scenario_builders import BASE_SCENARIO, make_alert


def write_tiny_experiment(root: Path, output_dir: Path, workers: int = 2) -> Path:
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, topology in enumerate(("TinyA", "TinyB")):
        raw = copy.deepcopy(BASE_SCENARIO)
        raw["scenario_id"] = f"{topology}-r0"
        raw["network"]["network_id"] = topology
        raw["provenance"]["scenario_seed"] = 11 + index
        raw["alerts"] = [
            make_alert("alert-0000", "s00", 0, 12, 300000 + 100000 * index),
            make_alert("alert-0001", "s01", 1, 20, 40000),
            make_alert("alert-0002", "s00", 2, 16, 250000),
        ]
        path = scenario_dir / f"{raw['scenario_id']}.json"
        path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append({"scenario_id": raw["scenario_id"], "split": "eval", "topology_id": topology, "sha256": load_scenario(path).sha256})
    manifest = root / "scenarios.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario_id", "split", "topology_id", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "experiment_id": "tiny",
        "scenario_sets": [{"set_id": "tiny", "manifest": str(manifest), "scenario_dir": str(scenario_dir)}],
        "split": "eval",
        "realization_ids": {"start": 0, "stop": 2},
        "methods": ["B0", "B1"],
        "bounds": {"tangents": 10},
        "milp_crosscheck": {"set_id": "tiny", "scenario_count": 2, "alert_count": 2, "realization_id": 0, "time_limit_s": 30},
        "bootstrap": {"resamples": 200, "seed": 5, "confidence": 0.95},
        "workers": workers,
        "output_dir": str(output_dir),
    }
    path = root / f"experiment-{output_dir.name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
```

```python
# tests/runner/test_runner_experiment.py
import csv
import json
from pathlib import Path

from runner.config import load_experiment_config
from runner.experiment import run_experiment
from experiment_fixtures import write_tiny_experiment

RUNTIME_COLUMNS = {"planning_runtime_s", "evaluation_runtime_s", "runtime_s", "milp_runtime_s"}


def _rows_without_runtime(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: value for key, value in row.items() if key not in RUNTIME_COLUMNS} for row in csv.DictReader(stream)]


def test_results_do_not_depend_on_worker_count(tmp_path: Path):
    one = load_experiment_config(write_tiny_experiment(tmp_path, tmp_path / "one"))
    two = load_experiment_config(write_tiny_experiment(tmp_path, tmp_path / "two"))
    assert run_experiment(one, Path("one.yaml"), workers=1) == 0
    assert run_experiment(two, Path("two.yaml"), workers=2) == 0
    for name in ("method_realizations.csv", "bounds.csv", "milp_crosscheck.csv"):
        assert _rows_without_runtime(tmp_path / "one" / name) == _rows_without_runtime(tmp_path / "two" / name)
    methods = _rows_without_runtime(tmp_path / "one" / "method_realizations.csv")
    assert len(methods) == 2 * 2 * 2
    bounds = _rows_without_runtime(tmp_path / "one" / "bounds.csv")
    assert {(row["variant"], row["trajectory_method"]) for row in bounds} == {("ground_only", ""), ("trajectory", "B1")}
    with (tmp_path / "one" / "summary.csv").open(newline="", encoding="utf-8") as stream:
        assert [(row["set_id"], row["method"]) for row in csv.DictReader(stream)] == [("tiny", "B0"), ("tiny", "B1")]
    manifest = json.loads((tmp_path / "one" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for field in (
        "experiment_id", "config_hash", "git_commit", "git_dirty", "source_hash", "scenario_manifest_sha256",
        "environment", "workers", "started_utc", "finished_utc", "command", "task_count",
    ):
        assert field in manifest
    assert set(manifest["environment"]) == {"python", "numpy", "scipy", "highs", "platform", "cpu_count"}


def test_rerun_skips_current_shards_and_recomputes_stale_ones(tmp_path: Path, capsys):
    config = load_experiment_config(write_tiny_experiment(tmp_path, tmp_path / "out"))
    assert run_experiment(config, Path("tiny.yaml")) == 0
    first = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert first["executed_task_count"] == first["task_count"]
    assert run_experiment(config, Path("tiny.yaml")) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["executed_task_count"] == 0
    shard = next((tmp_path / "out" / "shards").glob("bound__*.json"))
    payload = json.loads(shard.read_text(encoding="utf-8"))
    payload["task_hash"] = "stale"
    shard.write_text(json.dumps(payload), encoding="utf-8")
    assert run_experiment(config, Path("tiny.yaml")) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["executed_task_count"] == 1
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/runner/test_runner_experiment.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'runner.experiment'`.

- [ ] **Step 3: Implement orchestration and the entry point**

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
```

```python
# experiments/run_phase1.py
import argparse
from pathlib import Path
import sys

from runner.config import load_experiment_config
from runner.experiment import run_experiment


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="run-phase1")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/phase1.yaml"))
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(arguments)
    config = load_experiment_config(args.config)
    return run_experiment(config, args.config, workers=args.workers, argv=["experiments/run_phase1.py", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
printf 'results/phase1/shards/\n' >> .gitignore
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/runner/test_runner_experiment.py -v`

Expected: 2 passed.

Run: `uv run --extra test pytest -q`

Expected: 117 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/runner/experiment.py experiments/run_phase1.py .gitignore tests/runner/experiment_fixtures.py tests/runner/test_runner_experiment.py
git commit -m "feat: add parallel resumable experiment runner"
```

### Task 8: Phase 1 acceptance run

**Files:**
- Create (generated, committed): `results/phase1/method_realizations.csv`, `results/phase1/bounds.csv`, `results/phase1/milp_crosscheck.csv`, `results/phase1/summary.csv`, `results/phase1/manifest.json`

**Interfaces:**
- Consumes: Tasks 1–7; scenario sets `v0` and `v0-bh50`.
- Produces: the committed Phase 1 results used by Task 9. Their `manifest.json` `source_hash` must equal the source hash of the committed code, so no file under `src/` may change after this task.

- [ ] **Step 1: Run the experiment**

Run: `uv run python experiments/run_phase1.py --config configs/experiments/phase1.yaml`

Expected: exit code 0 and a final JSON line with `"task_count": 3725` and `"status": "complete"`. With 12 workers the prototype run took 20 minutes. If the command stops or exits non-zero, rerun it: completed shards are skipped. A non-zero exit that persists lists the failing tasks on stderr; stop and investigate instead of committing.

- [ ] **Step 2: Confirm resumability**

Run: `uv run python experiments/run_phase1.py --config configs/experiments/phase1.yaml`

Expected: exit code 0 and `"executed_task_count": 0`.

- [ ] **Step 3: Check the deterministic results**

Runtime columns vary between machines; every other value is deterministic. `results/phase1/summary.csv` must match these values to four decimals:

| set | method | timely mean | CI low | CI high | Conn | bound | gap | undefined gaps |
|---|---|---|---|---|---|---|---|---|
| v0 | B0 | 0.4133 | 0.3192 | 0.4867 | 0.5089 | 0.4832 | 0.1708 | 0 |
| v0 | B1 | 0.4943 | 0.3912 | 0.5816 | 1.0000 | 0.6213 | 0.2376 | 0 |
| v0-bh50 | B0 | 0.3700 | 0.2833 | 0.4367 | 0.5089 | 0.4678 | 0.2289 | 0 |
| v0-bh50 | B1 | 0.4301 | 0.3438 | 0.5002 | 1.0000 | 0.5855 | 0.2870 | 0 |

`results/phase1/milp_crosscheck.csv` must contain:

| scenario | B1 | MILP plan (static / equal split) | incumbent | dual bound | LP bound | status |
|---|---|---|---|---|---|---|
| Agis-r2 | 9 | 9 (3 / 9) | 9 | 9.00 | 9.51 | optimal |
| Belnet2008-r0 | 5 | 5 (0 / 5) | 5 | 5.00 | 6.41 | optimal |
| BtNorthAmerica-r0 | 2 | 2 (0 / 2) | 2 | 2.00 | 3.89 | optimal |
| Digex-r1 | 1 | 1 (0 / 1) | 1 | 1.00 | 2.38 | optimal |
| Digex-r2 | 3 | 3 (1 / 3) | 4 | 4.00 | 5.32 | optimal |

- [ ] **Step 4: Commit**

```bash
git add results/phase1/method_realizations.csv results/phase1/bounds.csv results/phase1/milp_crosscheck.csv results/phase1/summary.csv results/phase1/manifest.json
git commit -m "feat: record phase 1 results"
```

### Task 9: Report data from Phase 1 results

**Files:**
- Create: `experiments/report_tables.py`
- Test: `tests/runner/test_report_tables.py`
- Create (generated, committed): `docs/report/data/params.csv`, `docs/report/data/phase1_main.csv`, `docs/report/data/phase1_milp.csv`, `docs/report/data/lp_runtime.csv`, `docs/report/data/phase1_results.tex`

**Interfaces:**
- Consumes: Task 8 results; `runner.provenance.source_hash`; `configs/scenarios/v0.yaml`, `configs/scenarios/v0-bh50.yaml`, `configs/experiments/phase1.yaml`, `configs/data/paper.yaml`.
- Produces: semicolon-separated `params.csv` (`parameter;symbol;value`), `phase1_main.csv` (`set;backhaul;method;timely;ci;conn;bound;gap;planning;evaluation;boundtime`), `phase1_milp.csv` (`scenario;alerts;method;plan;incumbent;dual;lp;status;runtime;tangent`), comma-separated `lp_runtime.csv` (`variables,runtime`), and `phase1_results.tex` with `\newcommand` macros named `\Result<Metric><Set><Method>` (for example `\ResultTimelyVzeroBone`, `\ResultGapVzeroBhfivezeroBzero`) plus `\ResultLpCount`, `\ResultLpRuntimeMedian`, `\ResultLpRuntimeMax`, `\ResultLpVariablesMedian`, `\ResultLpVariablesMax`, `\ResultMilpCount`, `\ResultMilpOptimalCount`, `\ResultMilpMethodAtOptimumCount`, `\ResultMilpPlanAtOptimumCount`, `\ResultTangentOverestimateMax`, `\ResultComputeMinutes` (task runtimes summed from the result CSVs), `\ResultWorkers`, `\ResultCpuCount`, `\ResultScenarioCount`, `\ResultPythonVersion`, `\ResultScipyVersion`, `\ResultHighsVersion`, and per set `\ResultBoneWins<Set>` and `\ResultBoneTies<Set>` (scenarios where B1's mean timely ratio is above or equal to B0's). Numbers are written as `\num{...}` so `siunitx` prints decimal commas.

- [ ] **Step 1: Write the failing report-data tests**

```python
# tests/runner/test_report_tables.py
import csv
import importlib.util
import json
from pathlib import Path

from runner.provenance import source_hash

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "report_tables.py"
SPEC = importlib.util.spec_from_file_location("report_tables", SCRIPT)
report_tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report_tables)


def _write_results(root: Path, source: str) -> Path:
    results = root / "results"
    results.mkdir()
    manifest = {"status": "complete", "source_hash": source, "started_utc": "2026-09-11T00:00:00+00:00",
                "finished_utc": "2026-09-11T00:12:30+00:00", "workers": 12,
                "environment": {"cpu_count": 12, "python": "3.12.3", "scipy": "1.18.1", "highs": "1.12.0"}}
    (results / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = {"set_id": "v0", "method": "B1", "scenario_count": "30", "timely_ratio_mean": "0.25", "timely_ratio_ci_low": "0.2",
               "timely_ratio_ci_high": "0.3", "conn_ratio_mean": "0.9", "bound_ratio_mean": "0.5", "gap_mean": "0.5",
               "gap_undefined_count": "0", "shortfall_mean": "20", "planning_runtime_s_mean": "0.4",
               "evaluation_runtime_s_per_realization_mean": "0.02", "bound_runtime_s_mean": "1.5"}
    milp = {"set_id": "v0", "scenario_id": "Agis-r0", "alert_count": "10", "lp_ub": "4.5", "lp_status": "optimal",
            "milp_incumbent": "4.0", "milp_dual_bound": "4.0", "milp_status": "optimal", "milp_runtime_s": "3.2",
            "milp_plan_static_value": "1.0", "milp_plan_equal_split_value": "3.0", "milp_plan_value": "3.0", "method_value": "4.0",
            "tangent_max_overestimate": "0.0123"}
    method = {"set_id": "v0", "scenario_id": "Agis-r0", "topology_id": "Agis", "method": "B1", "realization_id": "0", "alert_count": "40",
              "timely_count": "10", "timely_ratio": "0.25", "source_count": "15", "connected_count": "15", "conn_ratio": "1.0",
              "shortfall": "20", "planning_runtime_s": "30", "evaluation_runtime_s": "1.55", "evaluate_calls": "1"}
    bound = {"set_id": "v0", "scenario_id": "Agis-r0", "topology_id": "Agis", "variant": "trajectory", "trajectory_method": "B1",
             "realization_id": "0", "value": "20", "ratio": "0.5", "status": "optimal", "runtime_s": "1.25",
             "variables": "30000", "rows": "40000", "nonzeros": "200000"}
    for name, row in (("summary.csv", summary), ("milp_crosscheck.csv", milp), ("bounds.csv", bound), ("method_realizations.csv", method)):
        with (results / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    return results


def _semicolon_rows(path: Path) -> list[list[str]]:
    return [line.split(";") for line in path.read_text(encoding="utf-8").splitlines()]


def test_report_files_cover_config_values_and_results(tmp_path: Path):
    results = _write_results(tmp_path, source_hash())
    output = tmp_path / "data"
    assert report_tables.main(["--results", str(results), "--output", str(output)]) == 0
    params = _semicolon_rows(output / "params.csv")
    assert params[0] == ["parameter", "symbol", "value"] and all(len(row) == 3 for row in params)
    text = (output / "params.csv").read_text(encoding="utf-8")
    for expected in (
        "\\qty{2000}{\\metre}", "10 / 3", "3 / 2", ";30\n", "\\qty{1}{\\second} / 150", "\\num{0.5}", "\\qty{100}{\\metre}",
        "\\qty{50}{\\metre\\per\\second}", "\\qty{0.1}{\\watt} / 2", "\\qty{1}{\\mega\\hertz}", "\\qty{-140}{\\dBm\\per\\hertz}",
        "\\num{9.61}", "\\num{0.16}", "\\num{0.2}", "\\num{2.2}", "\\num{3.3}", "\\qty{-50}{\\dB}", "\\num{2.8}",
        "\\qty{10}{\\kilo\\bit\\per\\second}", "3 (2)", "3 / \\qty{600}{\\metre} / \\qty{400}{\\metre}", "15 / \\qty{150}{\\metre}",
        "40 / 20--60 slot", "\\qty{250}{\\kilo\\bit}", "\\texttt{abilene}", "\\num{0.60} / \\num{0.05}",
        "\\qty{1000}{\\kilo\\bit\\per\\second} / \\qty{50}{\\kilo\\bit\\per\\second}", "10 / 10000",
    ):
        assert expected in text, expected
    main_rows = _semicolon_rows(output / "phase1_main.csv")
    assert main_rows[1][:5] == ["v0", "\\qty{1000}{\\kilo\\bit\\per\\second}", "B1", "\\num{0.250}", "[\\num{0.200}, \\num{0.300}]"]
    milp_rows = _semicolon_rows(output / "phase1_milp.csv")
    assert milp_rows[1][:5] == ["Agis-r0", "10", "\\num{4}", "\\num{3}", "\\num{4}"]
    assert milp_rows[1][7] == "tối ưu" and len(milp_rows[1]) == 10
    assert (output / "lp_runtime.csv").read_text(encoding="utf-8").splitlines() == ["variables,runtime", "30000,1.250000"]
    macros = (output / "phase1_results.tex").read_text(encoding="utf-8")
    assert "\\newcommand{\\ResultTimelyVzeroBone}{\\num{0.250}}" in macros
    assert "\\newcommand{\\ResultComputeMinutes}{\\num{1}}" in macros
    assert "\\newcommand{\\ResultMilpMethodAtOptimumCount}{1}" in macros
    assert "\\newcommand{\\ResultMilpPlanAtOptimumCount}{0}" in macros
    assert "\\newcommand{\\ResultHighsVersion}{1.12.0}" in macros
    assert "\\newcommand{\\ResultMilpStaticTotal}{1}" in macros
    assert "\\newcommand{\\ResultMilpEqualSplitTotal}{3}" in macros
    assert "\\newcommand{\\ResultBoneWinsVzero}" not in macros
    assert report_tables.macro_name("Gap", "v0-bh50", "B0") == "ResultGapVzeroBhfivezeroBzero"
    assert report_tables.macro_name("BoundTime", "v0", "B1") == "ResultBoundTimeVzeroBone"
    assert report_tables.macro_name("BoneWins", "v0-bh50") == "ResultBoneWinsVzeroBhfivezero"


def test_report_tables_refuse_results_from_other_source_code(tmp_path: Path, capsys):
    results = _write_results(tmp_path, "0" * 64)
    assert report_tables.main(["--results", str(results), "--output", str(tmp_path / "data")]) == 1
    assert "different source code" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run --extra test pytest tests/runner/test_report_tables.py -v`

Expected: collection fails with `FileNotFoundError` for `experiments/report_tables.py`.

- [ ] **Step 3: Implement the generator**

```python
# experiments/report_tables.py
"""Write report CSVs and result macros for docs/report from a completed Phase 1 run."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Mapping, Sequence

import yaml

from runner.provenance import source_hash

DIGIT_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
MILP_STATUS = {"optimal": "tối ưu", "time_limit": "hết giờ"}


def num(value: float, digits: int = 3) -> str:
    return f"\\num{{{float(value):.{digits}f}}}"


def macro_name(*parts: str) -> str:
    words = []
    for part in parts:
        for piece in part.replace("_", "-").split("-"):
            spelled = "".join(DIGIT_WORDS.get(char, char) for char in piece)
            words.append(spelled[:1].upper() + spelled[1:])
    return "Result" + "".join(words)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_semicolon(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    lines = [";".join(header)] + [";".join(row) for row in rows]
    for line in lines[1:]:
        if line.count(";") != len(header) - 1:
            raise ValueError(f"{path.name}: a cell contains a semicolon: {line}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def kbit(bits_per_second: float) -> str:
    return f"\\qty{{{bits_per_second / 1000:g}}}{{\\kilo\\bit\\per\\second}}"


def params_rows(scenario: Mapping, stressed: Mapping, experiment: Mapping, data_profile: Mapping) -> list[list[str]]:
    topology, workload, failures = scenario["topology"], scenario["workload"], scenario["failures"]
    time, uav, spectrum, channel = scenario["time"], scenario["uav"], scenario["spectrum"], scenario["channel"]
    span = experiment["realization_ids"]
    return [
        ["Kích thước vùng mô phỏng", "$L_{\\mathrm{box}}$", f"\\qty{{{data_profile['box_size_m']:g}}}{{\\metre}}"],
        ["Topology đánh giá / phát triển", "---", f"{topology['strata'] - topology['dev_count']} / {topology['dev_count']}"],
        ["Replicate mỗi topology (đánh giá / phát triển)", "---", f"{topology['eval_replicates']} / {topology['dev_replicates']}"],
        ["Số realization kênh", "$M$", str(span["stop"] - span["start"])],
        ["Độ dài slot / số slot", "$\\Delta$, $N$", f"\\qty{{{time['slot_s']:g}}}{{\\second}} / {time['num_slots']}"],
        ["Tỷ lệ nửa slot truy nhập", "$\\tau$", num(time["access_fraction"], 1)],
        ["Độ cao / tốc độ tối đa UAV", "$H$, $V_{\\max}$", f"\\qty{{{uav['altitude_m']:g}}}{{\\metre}} / \\qty{{{uav['v_max_mps']:g}}}{{\\metre\\per\\second}}"],
        ["Công suất UAV / số downlink đồng thời", "$P_U$", f"\\qty{{{uav['total_power_w']:g}}}{{\\watt}} / {uav['max_active_downlinks']}"],
        ["Công suất nguồn", "$p_s$", f"\\qty{{{scenario['source_power_w']:g}}}{{\\watt}}"],
        ["Tổng băng thông mỗi nửa slot", "$B_{\\mathrm{tot}}$", f"\\qty{{{spectrum['b_tot_hz'] / 1e6:g}}}{{\\mega\\hertz}}"],
        ["Mật độ phổ nhiễu", "$N_0$", f"\\qty{{{spectrum['noise_psd_dbm_per_hz']:g}}}{{\\dBm\\per\\hertz}}"],
        ["Tham số PrLoS", "$(a, b)$", f"({num(channel['uav']['plos_a'], 2)}, {num(channel['uav']['plos_b'], 2)})"],
        ["Suy hao NLoS / số mũ LoS, NLoS", "$\\mu$, $\\alpha_L$, $\\alpha_N$", f"{num(channel['uav']['nlos_attenuation'], 1)} / {num(channel['uav']['alpha_los'], 1)}, {num(channel['uav']['alpha_nlos'], 1)}"],
        ["Độ lợi tham chiếu", "$\\beta_0$", f"\\qty{{{channel['uav']['beta0_db']:g}}}{{\\dB}}"],
        ["Số mũ suy hao mặt đất", "$\\alpha_G$", num(channel["ground"]["alpha"], 1)],
        ["Ngưỡng tốc độ khả dụng", "$R_{\\min}$", kbit(channel["usable_rate_bps"])],
        ["Candidate path (tối đa đường mặt đất)", "$K$", f"{scenario['paths']['k']} ({scenario['paths']['max_ground']})"],
        ["Vùng hư hại: số / khoảng cách / bán kính", "---", f"{workload['zone_count']} / \\qty{{{workload['zone_min_separation_m']:g}}}{{\\metre}} / \\qty{{{workload['zone_radius_m']:g}}}{{\\metre}}"],
        ["Số nguồn / độ lệch chuẩn vị trí", "$|S|$, $\\sigma$", f"{workload['source_count']} / \\qty{{{workload['source_sigma_m']:g}}}{{\\metre}}"],
        ["Số cảnh báo / cửa sổ deadline", "$|A|$, $d_a-r_a$", f"{workload['alert_count']} / {workload['deadline_min_slots']}--{workload['deadline_max_slots']} slot"],
        ["Kích thước tham chiếu (tỷ lệ cắt, instance)", "$L_{\\mathrm{ref}}$", f"\\qty{{{workload['size_ref_bits'] / 1000:g}}}{{\\kilo\\bit}} ([{num(workload['size_ratio_min'], 2)}, {num(workload['size_ratio_max'], 0)}], \\texttt{{{workload['sndlib_instance']}}})"],
        ["Xác suất hỏng cạnh trong / ngoài vùng", "$p_{\\mathrm{in}}$, $p_{\\mathrm{out}}$", f"{num(failures['p_in'], 2)} / {num(failures['p_out'], 2)}"],
        ["Dung lượng backhaul (v0 / v0-bh50)", "$C_e$", f"{kbit(scenario['backhaul_capacity_bps'])} / {kbit(stressed['backhaul_capacity_bps'])}"],
        ["Số tiếp tuyến trong cận LP / lần lấy mẫu bootstrap", "---", f"{experiment['bounds']['tangents']} / {experiment['bootstrap']['resamples']}"],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="report-tables")
    parser.add_argument("--results", type=Path, default=Path("results/phase1"))
    parser.add_argument("--experiment", type=Path, default=Path("configs/experiments/phase1.yaml"))
    parser.add_argument("--scenario-config", type=Path, default=Path("configs/scenarios/v0.yaml"))
    parser.add_argument("--stressed-config", type=Path, default=Path("configs/scenarios/v0-bh50.yaml"))
    parser.add_argument("--data-profile", type=Path, default=Path("configs/data/paper.yaml"))
    parser.add_argument("--output", type=Path, default=Path("docs/report/data"))
    args = parser.parse_args(argv)

    manifest_path = args.results / "manifest.json"
    if not manifest_path.exists():
        print(f"missing {manifest_path}; run experiments/run_phase1.py first", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["source_hash"] != source_hash():
        print("results are incomplete or were produced by different source code; rerun the experiment", file=sys.stderr)
        return 1

    load = lambda path: yaml.safe_load(path.read_text(encoding="utf-8"))
    scenario, stressed, experiment = load(args.scenario_config), load(args.stressed_config), load(args.experiment)
    capacities = {scenario["set_id"]: scenario["backhaul_capacity_bps"], stressed["set_id"]: stressed["backhaul_capacity_bps"]}
    args.output.mkdir(parents=True, exist_ok=True)
    _write_semicolon(args.output / "params.csv", ["parameter", "symbol", "value"], params_rows(scenario, stressed, experiment, load(args.data_profile)))

    summary = _read_rows(args.results / "summary.csv")
    main_rows, macros = [], []
    for row in summary:
        set_id, method = row["set_id"], row["method"]
        main_rows.append(
            [
                set_id, kbit(capacities[set_id]), method, num(row["timely_ratio_mean"]),
                f"[{num(row['timely_ratio_ci_low'])}, {num(row['timely_ratio_ci_high'])}]",
                num(row["conn_ratio_mean"]), num(row["bound_ratio_mean"]), num(row["gap_mean"]),
                num(row["planning_runtime_s_mean"]), num(row["evaluation_runtime_s_per_realization_mean"]), num(row["bound_runtime_s_mean"]),
            ]
        )
        for metric, column in (
            ("Timely", "timely_ratio_mean"), ("TimelyLow", "timely_ratio_ci_low"), ("TimelyHigh", "timely_ratio_ci_high"),
            ("Conn", "conn_ratio_mean"), ("Bound", "bound_ratio_mean"), ("Gap", "gap_mean"),
            ("Planning", "planning_runtime_s_mean"), ("Evaluation", "evaluation_runtime_s_per_realization_mean"),
            ("BoundTime", "bound_runtime_s_mean"),
        ):
            macros.append((macro_name(metric, set_id, method), num(row[column])))
        macros.append((macro_name("GapUndefined", set_id, method), row["gap_undefined_count"]))
    _write_semicolon(
        args.output / "phase1_main.csv",
        ["set", "backhaul", "method", "timely", "ci", "conn", "bound", "gap", "planning", "evaluation", "boundtime"],
        main_rows,
    )

    milp_rows = []
    crosscheck = _read_rows(args.results / "milp_crosscheck.csv")
    for row in crosscheck:
        milp_rows.append(
            [
                row["scenario_id"], row["alert_count"], num(row["method_value"], 0), num(row["milp_plan_value"], 0), num(row["milp_incumbent"], 0),
                num(row["milp_dual_bound"], 2), num(row["lp_ub"], 2), MILP_STATUS.get(row["milp_status"], row["milp_status"]),
                num(row["milp_runtime_s"], 1), num(row["tangent_max_overestimate"], 4),
            ]
        )
    _write_semicolon(
        args.output / "phase1_milp.csv",
        ["scenario", "alerts", "method", "plan", "incumbent", "dual", "lp", "status", "runtime", "tangent"],
        milp_rows,
    )

    bounds = _read_rows(args.results / "bounds.csv")
    with (args.output / "lp_runtime.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variables", "runtime"])
        writer.writerows([row["variables"], f"{float(row['runtime_s']):.6f}"] for row in bounds)

    runtimes = [float(row["runtime_s"]) for row in bounds]
    variables = [int(row["variables"]) for row in bounds]
    method_rows = _read_rows(args.results / "method_realizations.csv")
    planning = {(row["set_id"], row["scenario_id"], row["method"]): float(row["planning_runtime_s"]) for row in method_rows}
    compute_s = (
        sum(runtimes)
        + sum(float(row["milp_runtime_s"]) for row in crosscheck)
        + sum(float(row["evaluation_runtime_s"]) for row in method_rows)
        + sum(planning.values())
    )
    macros += [
        ("ResultLpCount", str(len(bounds))),
        ("ResultLpRuntimeMedian", num(statistics.median(runtimes), 2)),
        ("ResultLpRuntimeMax", num(max(runtimes), 1)),
        ("ResultLpVariablesMedian", str(int(statistics.median(variables)))),
        ("ResultLpVariablesMax", str(max(variables))),
        ("ResultMilpCount", str(len(crosscheck))),
        ("ResultMilpOptimalCount", str(sum(1 for row in crosscheck if row["milp_status"] == "optimal"))),
        ("ResultTangentOverestimateMax", num(max(float(row["tangent_max_overestimate"]) for row in crosscheck), 4)),
        ("ResultComputeMinutes", num(compute_s / 60.0, 0)),
        ("ResultMilpMethodAtOptimumCount", str(sum(1 for row in crosscheck if float(row["method_value"]) >= float(row["milp_incumbent"]) - 1e-9 and row["milp_status"] == "optimal"))),
        ("ResultMilpStaticTotal", str(int(sum(float(row["milp_plan_static_value"]) for row in crosscheck)))),
        ("ResultMilpEqualSplitTotal", str(int(sum(float(row["milp_plan_equal_split_value"]) for row in crosscheck)))),
        ("ResultMilpPlanAtOptimumCount", str(sum(1 for row in crosscheck if float(row["milp_plan_value"]) >= float(row["milp_incumbent"]) - 1e-9 and row["milp_status"] == "optimal"))),
        ("ResultWorkers", str(manifest["workers"])),
        ("ResultCpuCount", str(manifest["environment"]["cpu_count"])),
        ("ResultScenarioCount", summary[0]["scenario_count"] if summary else "0"),
        ("ResultPythonVersion", manifest["environment"]["python"]),
        ("ResultScipyVersion", manifest["environment"]["scipy"]),
        ("ResultHighsVersion", manifest["environment"]["highs"]),
    ]
    scenario_means: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in method_rows:
        scenario_means.setdefault((row["set_id"], row["method"]), {}).setdefault(row["scenario_id"], []).append(float(row["timely_ratio"]))
    for set_id in sorted({row["set_id"] for row in method_rows}):
        first, second = scenario_means.get((set_id, "B0")), scenario_means.get((set_id, "B1"))
        if first and second:
            means = {scenario: (statistics.fmean(first[scenario]), statistics.fmean(second[scenario])) for scenario in first}
            macros.append((macro_name("BoneWins", set_id), str(sum(1 for b0, b1 in means.values() if b1 > b0))))
            macros.append((macro_name("BoneTies", set_id), str(sum(1 for b0, b1 in means.values() if b1 == b0))))
    lines = ["% Generated by experiments/report_tables.py from results/phase1; do not edit."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros]
    (args.output / "phase1_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"params": len(params_rows(scenario, stressed, experiment, load(args.data_profile))), "summary": len(main_rows), "milp": len(milp_rows), "lp": len(bounds)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `uv run --extra test pytest tests/runner/test_report_tables.py -v`

Expected: 2 passed.

Run: `uv run --extra test pytest -q`

Expected: 119 passed, 1 skipped.

- [ ] **Step 5: Generate the report data**

Run: `uv run python experiments/report_tables.py`

Expected: exit code 0 printing `{"params": 24, "summary": 4, "milp": 5, "lp": 3600}` and five files in `docs/report/data/`.

- [ ] **Step 6: Commit**

```bash
git add experiments/report_tables.py tests/runner/test_report_tables.py docs/report/data
git commit -m "feat: generate report data from phase 1 results"
```

### Task 10: Report model, methods, bound, hardness, and related work

**Files:**
- Modify: `docs/report/preamble.tex`, `docs/report/references.bib`
- Modify (full rewrite): `docs/report/tables/params.tex`, `docs/report/sections/method.tex`, `docs/report/sections/related-work.tex`
- Create: `docs/report/tables/phase1_main.tex`, `docs/report/tables/phase1_milp.tex`

**Interfaces:**
- Consumes: Task 9 CSV files through `\csvreader`.
- Produces: labels `sec:model`, `sec:problem`, `sec:baselines`, `sec:bound`, `sec:hardness`, `sec:complexity`, `sec:algorithm`, `def:timely`, `prop:ub`, `prop:hardness`, `eq:lp`, `eq:sandwich`, `tab:params`, `tab:phase1-main`, `tab:phase1-milp`, `tab:related`; bibliography keys `lawler1990preemptive`, `lenstra2020elements`.

- [ ] **Step 1: Load the table and plot packages and add cross-reference names**

In `docs/report/preamble.tex`:

1. Replace

```latex
\usepackage[table]{xcolor}
```

   with

```latex
\usepackage[table]{xcolor}
\usepackage{csvsimple}           % bảng đọc thẳng từ CSV (skill: Tables from CSV Data)
\usepackage{pgfplots}             % biểu đồ từ CSV (skill: charts-and-graphs)
\pgfplotsset{compat=1.18}
```

2. Replace

```latex
\sisetup{detect-all}
```

   with

```latex
\sisetup{detect-all, output-decimal-marker={,}, per-mode=symbol}  % dấu phẩy thập phân theo văn bản tiếng Việt; đơn vị dạng kbit/s
```

3. Replace

```latex
\crefname{theorem}{Định lý}{Định lý}
\Crefname{theorem}{Định lý}{Định lý}
```

   with

```latex
\crefname{theorem}{Định lý}{Định lý}
\Crefname{theorem}{Định lý}{Định lý}
\crefname{proposition}{Mệnh đề}{Mệnh đề}
\Crefname{proposition}{Mệnh đề}{Mệnh đề}
\crefname{definition}{Định nghĩa}{Định nghĩa}
\Crefname{definition}{Định nghĩa}{Định nghĩa}
\crefname{remark}{Nhận xét}{Nhận xét}
\Crefname{remark}{Nhận xét}{Nhận xét}
```

- [ ] **Step 2: Add the verified references**

Replace the whole content of `docs/report/references.bib` with the version below. It keeps the existing entries, records which entries were verified and against which source, and adds `lenstra2020elements` and `lawler1990preemptive`:

```bibtex
% docs/report/references.bib
% references.bib — tài liệu tham khảo (biblatex + biber, style=ieee)
% huang2025ddatsap: metadata PDF gốc. knight2011zoo, orlowski2010sndlib: đối chiếu Crossref.
% lawler1990preemptive, lenstra2020elements: đối chiếu Springer, arXiv và tổng quan arXiv:2405.18789.
% rahnemoonfar2023rescuenet chưa được đối chiếu; phải kiểm tra trước khi trích dẫn (Pha 2).

@article{huang2025ddatsap,
  author  = {Huang, Ping and Lin, Jie and Liu, Tong and Ning, Jin and Luo, Junsong and Duo, Bin},
  title   = {Dynamic Dual-Antenna Time-Slot Allocation Protocol for {UAV}-Aided Relaying System Under Probabilistic {LoS}-Channel},
  journal = {Sensors},
  year    = {2025},
  volume  = {25},
  number  = {24},
  pages   = {7443},
  doi     = {10.3390/s25247443},
}

@book{boyd2004convex,
  author    = {Boyd, Stephen and Vandenberghe, Lieven},
  title     = {Convex Optimization},
  publisher = {Cambridge University Press},
  year      = {2004},
}

@article{knight2011zoo,
  author  = {Knight, Simon and Nguyen, Hung X. and Falkner, Nickolas and Bowden, Rhys and Roughan, Matthew},
  title   = {The {Internet} {Topology} {Zoo}},
  journal = {IEEE Journal on Selected Areas in Communications},
  year    = {2011},
  volume  = {29},
  number  = {9},
  pages   = {1765--1775},
  doi     = {10.1109/JSAC.2011.111002},
}

@article{orlowski2010sndlib,
  author  = {Orlowski, Sebastian and Wess{\"a}ly, Roland and Pi{\'o}ro, Micha{\l} and Tomaszewski, Artur},
  title   = {{SNDlib} 1.0---{S}urvivable Network Design Library},
  journal = {Networks},
  year    = {2010},
  volume  = {55},
  number  = {3},
  pages   = {276--286},
  doi     = {10.1002/net.20371},
}

@article{rahnemoonfar2023rescuenet,
  author  = {Rahnemoonfar, Maryam and Chowdhury, Tashnim and Murphy, Robin},
  title   = {{RescueNet}: A High Resolution {UAV} Semantic Segmentation Dataset for Natural Disaster Damage Assessment},
  journal = {Scientific Data},
  year    = {2023},
  volume  = {10},
  pages   = {913},
  doi     = {10.1038/s41597-023-02799-4},
}

@article{lawler1990preemptive,
  author  = {Lawler, Eugene L.},
  title   = {A Dynamic Programming Algorithm for Preemptive Scheduling of a Single Machine to Minimize the Number of Late Jobs},
  journal = {Annals of Operations Research},
  year    = {1990},
  volume  = {26},
  pages   = {125--133},
  doi     = {10.1007/BF02248588},
}

@misc{lenstra2020elements,
  author        = {Lenstra, Jan Karel and Shmoys, David B.},
  title         = {Elements of Scheduling},
  year          = {2020},
  eprint        = {2001.06005},
  archiveprefix = {arXiv},
  primaryclass  = {cs.DS},
  doi           = {10.48550/arXiv.2001.06005},
}
```

- [ ] **Step 3: Replace the parameter table and add the result tables**

Replace the whole content of `docs/report/tables/params.tex`, and create the two result tables:

```latex
% docs/report/tables/params.tex
% Bảng tham số — hàng đọc từ data/params.csv (sinh bởi experiments/report_tables.py).
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Tham số của bộ scenario v0, họ v0-bh50 và thí nghiệm Pha 1.}
\label{tab:params}
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Tham số} & \textbf{Ký hiệu} & \textbf{Giá trị} \\
\midrule
\csvreader[separator=semicolon, late after line=\\]{data/params.csv}{1=\colparameter, 2=\colsymbol, 3=\colvalue}{\colparameter & \colsymbol & \colvalue}
\bottomrule
\end{tabular}
\end{table}
```

```latex
% docs/report/tables/phase1_main.tex
% Kết quả Pha 1 — hàng đọc từ data/phase1_main.csv (sinh bởi experiments/report_tables.py).
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Kết quả Pha 1 trên 30 scenario đánh giá, mỗi scenario 30 realization kênh. Khoảng tin cậy 95\% lấy bằng bootstrap theo topology; gap tính so với cận trên cùng thông tin (B0: cận chỉ mặt đất, B1: cận trên quỹ đạo B1); thời gian tính bằng giây.}
\label{tab:phase1-main}
\begin{tabular}{@{}llccccccc@{}}
\toprule
\textbf{Backhaul} & \textbf{PP} & \textbf{Đúng hạn} & \textbf{KTC 95\%} & \textbf{Conn} & \textbf{Cận trên} & \textbf{Gap} & \textbf{Lập KH} & \textbf{LP} \\
\midrule
\csvreader[separator=semicolon, late after line=\\]{data/phase1_main.csv}{2=\colbackhaul, 3=\colmethod, 4=\coltimely, 5=\colci, 6=\colconn, 7=\colbound, 8=\colgap, 9=\colplanning, 11=\colboundtime}{\colbackhaul & \colmethod & \coltimely & \colci & \colconn & \colbound & \colgap & \colplanning & \colboundtime}
\bottomrule
\end{tabular}
\end{table}
```

```latex
% docs/report/tables/phase1_milp.tex
% Đối chiếu MILP — hàng đọc từ data/phase1_milp.csv (sinh bởi experiments/report_tables.py).
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Đối chiếu trên instance thu nhỏ (10 cảnh báo, quỹ đạo B1, realization 0): số cảnh báo đúng hạn của B1 và của kế hoạch dựng từ nghiệm MILP (giá trị tốt hơn giữa băng thông của nghiệm và chia đều theo backlog), nghiệm MILP tốt nhất, cận đối ngẫu MILP, cận LP và thời gian giải MILP.}
\label{tab:phase1-milp}
\begin{tabular}{@{}lccccccc@{}}
\toprule
\textbf{Scenario} & \textbf{$|A|$} & \textbf{B1} & \textbf{KH MILP} & \textbf{MILP} & \textbf{Cận MILP} & \textbf{Cận LP} & \textbf{Thời gian (s)} \\
\midrule
\csvreader[separator=semicolon, late after line=\\]{data/phase1_milp.csv}{1=\colscenario, 2=\colalerts, 3=\colmethod, 4=\colplan, 5=\colincumbent, 6=\coldual, 7=\collp, 9=\colruntime}{\texttt{\colscenario} & \colalerts & \colmethod & \colplan & \colincumbent & \coldual & \collp & \colruntime}
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 4: Rewrite the method section**

Replace the whole content of `docs/report/sections/method.tex` with:

```latex
% docs/report/sections/method.tex
\section{Mô hình và phương pháp}
\label{sec:method}

\subsection{Mô hình hệ thống}
\label{sec:model}

Mạng relay mặt đất sau thiên tai được biểu diễn bằng đồ thị $G=(V,E)$ lấy từ Topology Zoo~\cite{knight2011zoo} và chuẩn hoá vào vùng vuông cạnh $L_{\mathrm{box}}$ với cùng một hệ số tỷ lệ trên hai trục. Nút $c\in V$ là trung tâm cứu hộ, mọi nút còn lại là relay. Cạnh $e\in E$ còn hoạt động có dung lượng backhaul $C_e$; cạnh hỏng có dung lượng bằng $0$. Các điểm phát cảnh báo tạo thành tập $S$; chúng không thuộc $G$ và chỉ truyền được bằng vô tuyến. Mỗi cảnh báo $a\in A$ có nguồn $s_a\in S$, slot phát sinh $r_a$, hạn $d_a$ và kích thước $L_a$ bit. Nhiệm vụ gồm $N$ slot độ dài $\Delta$; mỗi slot chia thành nửa \emph{truy nhập} dài $\tau\Delta$, trong đó nguồn gửi tới relay hoặc UAV, và nửa \emph{downlink} dài $(1-\tau)\Delta$, trong đó UAV gửi xuống một nút của $G$.

UAV bay ở độ cao cố định $H$ theo quỹ đạo $q[0],\dots,q[N]\in\R^2$ thoả
\begin{equation}
\label{eq:mobility}
q[0]=q_s,\qquad q[N]=q_t,\qquad \norm{q[n+1]-q[n]}\le V_{\max}\Delta,\quad n=0,\dots,N-1,
\end{equation}
và vị trí đại diện của slot $n$ là trung điểm $\bar q[n]=\bigl(q[n]+q[n+1]\bigr)/2$.

\paragraph{Kênh.}
Liên kết giữa UAV và điểm mặt đất $k$ ở toạ độ $w_k$ dùng mô hình xác suất tầm nhìn thẳng (PrLoS) của~\cite{huang2025ddatsap}:
\begin{equation}
\label{eq:geometry}
\begin{aligned}
d_k[n]&=\sqrt{\norm{\bar q[n]-w_k}^2+H^2},\qquad
\theta_k[n]=\frac{180}{\pi}\operatorname{atan2}\bigl(H,\norm{\bar q[n]-w_k}\bigr),\\
P_k^{L}[n]&=\frac{1}{1+a\exp\bigl[-b(\theta_k[n]-a)\bigr]},
\end{aligned}
\end{equation}
với độ lợi $g^{L}=\beta_0 d^{-\alpha_L}$ khi có tầm nhìn thẳng và $g^{N}=\mu\beta_0 d^{-\alpha_N}$ khi không. Với công suất phát $p$ và mật độ phổ nhiễu $N_0$, đặt $s=pg/N_0$ (đơn vị Hz); tốc độ khi cấp băng thông $b$ là
\begin{equation}
\label{eq:rate}
R(b,s)=b\log_2\!\Bigl(1+\frac{s}{b}\Bigr),\qquad R(0,s)=0.
\end{equation}
Trong realization $\omega$, trạng thái LoS/NLoS của liên kết tới điểm $k$ ở slot $n$ được lấy mẫu độc lập với xác suất $P_k^{L}[n]$ tính trên chính quỹ đạo đang xét; mọi kế hoạch được so sánh trên cùng các số ngẫu nhiên. Liên kết từ nguồn tới relay mặt đất có độ lợi tất định $\beta_0 d^{-\alpha_G}$ và chỉ tồn tại khi $R(B_{\mathrm{tot}},s)\ge R_{\min}$. Nguồn phát công suất $p_s$; UAV có tổng công suất $P_U$ chia đều cho tối đa hai downlink hoạt động trong một slot, nên ngân sách công suất không tăng theo quy mô.

\paragraph{Đường chuyển tiếp.}
Mỗi cảnh báo có tối đa $K$ candidate path thuộc hai loại: đường \emph{mặt đất} $s_a\to v\leadsto c$ qua một relay $v$ trong tầm, và đường \emph{UAV} $s_a\to\text{UAV}\to u\leadsto c$ với nút thả $u\in V$. Đoạn $v\leadsto c$ là đường backhaul ít chặng nhất trên các cạnh còn hoạt động. Tập candidate gồm tối đa hai đường mặt đất có chi phí ước lượng thấp nhất, bổ sung bằng các đường UAV có chi phí thấp nhất; tập này cố định trước khi lập kế hoạch và mọi phương pháp dùng chung.

\subsection{Phát biểu bài toán}
\label{sec:problem}

Một kế hoạch gồm quỹ đạo $q$ thoả \eqref{eq:mobility}, lựa chọn đường $x_{a,p}\in\set{0,1}$ với $\sum_p x_{a,p}\le 1$ (cho phép không phục vụ cảnh báo), và băng thông $b_{e,n}\ge 0$ trên các liên kết vô tuyến. Kế hoạch được lập trước nhiệm vụ chỉ từ tốc độ kỳ vọng và được đánh giá trên các realization kênh độc lập. Khi thực thi, trong mỗi slot $n$:
\begin{enumerate}
\item \emph{Phổ:} tổng băng thông của các liên kết truy nhập và của các downlink đều không vượt $B_{\mathrm{tot}}$; có tối đa hai downlink hoạt động.
\item \emph{Dung lượng:} liên kết vô tuyến $e$ mang tối đa $t_e\Delta\,R(b_{e,n},s_{e,n})$ bit, với $t_e=\tau$ cho truy nhập và $t_e=1-\tau$ cho downlink; cạnh backhaul mang tối đa $\Delta C_e$ bit, chia chung cho mọi cảnh báo đi qua.
\item \emph{Nhân quả:} bit của $a$ có mặt ở nguồn từ đầu slot $r_a$; truyền trong slot $n$ chỉ dùng dữ liệu có trong buffer ở đầu slot $n$, nên bit nhận trong slot $n$ chỉ được chuyển tiếp từ slot $n+1$; bit chỉ đi trên đường đã chọn và buffer không âm.
\item \emph{Hạn:} ở đầu slot $d_a$, mọi bit còn lại của $a$ bị huỷ.
\end{enumerate}
Mọi phương pháp dùng chung bộ điều phối phục vụ mỗi hàng đợi theo thứ tự hạn sớm nhất (EDF).

\begin{definition}[Cảnh báo đúng hạn]
\label[definition]{def:timely}
Cảnh báo $a$ \emph{đúng hạn} trong realization $\omega$, ký hiệu $z_{a,\omega}=1$, nếu ít nhất $L_a$ bit của nó tới trung tâm $c$ không muộn hơn cuối slot $d_a-1$; ngược lại $z_{a,\omega}=0$.
\end{definition}

Gọi $D_{a,\omega}$ là số bit của $a$ tới $c$ trước hạn. Các kế hoạch được so sánh theo thứ tự từ điển của
\begin{equation}
\label{eq:objective}
\Bigl(\hat P_{\mathrm{timely}},\ \mathrm{Conn},\ -\sum_{\omega}\sum_{a}\frac{L_a-D_{a,\omega}}{L_a}\Bigr),
\qquad
\hat P_{\mathrm{timely}}=\frac{1}{M\abs{A}}\sum_{\omega=1}^{M}\sum_{a\in A}z_{a,\omega},
\end{equation}
trong đó $\mathrm{Conn}$ là tỷ lệ nguồn có đường theo thời gian tới $c$ trên đồ thị mở rộng theo thời gian: mỗi bước truyền tốn một slot, được phép chờ trong buffer, và liên kết UAV dùng được ở slot $n$ khi $R(B_{\mathrm{tot}},s_{e,n})\ge R_{\min}$.

\begin{remark}
Thành phần thứ ba là tỷ lệ dữ liệu còn thiếu tại hạn (shortfall), thay cho tổng độ trễ: vì cảnh báo quá hạn bị huỷ, cảnh báo trễ không có thời điểm hoàn thành để tính độ trễ.
\end{remark}

\subsection{Phương pháp cơ sở}
\label{sec:baselines}

Hai phương pháp cơ sở dùng chung tập candidate, bộ điều phối EDF và quy tắc chia đều $B_{\mathrm{tot}}$ cho các liên kết đang có dữ liệu chờ trong mỗi nửa slot; chúng chỉ khác nhau ở vai trò của UAV. Cả hai lập kế hoạch từ tốc độ kỳ vọng và không dùng các realization đánh giá.

\paragraph{B0 --- không có UAV.}
UAV đứng yên tại điểm xuất phát và không được giao cảnh báo nào. Mỗi cảnh báo đi theo đường mặt đất có chi phí ước lượng thấp nhất; nguồn không có relay nào trong tầm thì cảnh báo không được phục vụ. Với B0, $\mathrm{Conn}$ được tính trên đồ thị không có liên kết UAV. Chi phí lập kế hoạch là $O(\abs{A}K)$.

\paragraph{B1 --- UAV bay theo quỹ đạo cố định.}
UAV xuất phát từ trung tâm, lần lượt bay tới tâm các vùng hư hại theo thứ tự láng giềng gần nhất với tốc độ $V_{\max}$, chia đều thời gian còn lại để lơ lửng tại các vùng, rồi trở về; vùng được thêm sau cùng bị bỏ nếu hành trình không vừa $N$ slot. Với mỗi cảnh báo, mọi candidate được mô phỏng riêng cho cảnh báo đó trên quỹ đạo này với tốc độ kỳ vọng và toàn bộ băng thông; B1 chọn candidate giao xong sớm nhất (hoà thì lấy chỉ số nhỏ hơn) và không phục vụ cảnh báo nếu không candidate nào kịp hạn. Chi phí lập kế hoạch là $O(\abs{A}KNH_{\max})$, với $H_{\max}$ là số chặng lớn nhất của một candidate.

\subsection{Cận trên theo từng realization}
\label{sec:bound}

Cố định quỹ đạo $q$, tập candidate và realization $\omega$; khi đó $s_{e,n}$ đã biết. Với chặng $h$ của candidate $p$ và slot $n\in[r_a+h,\,d_a-1]$, gọi $f_{a,p,h,n}$ là số bit đi qua chặng và $B_{a,p,h,n}$ là buffer ở đầu slot. Chọn các điểm tiếp tuyến $b_k=B_{\mathrm{tot}}2^{-k}$, $k=0,\dots,9$, và xét bài toán quy hoạch tuyến tính
\begin{subequations}
\label{eq:lp}
\begin{align}
\max\quad & \textstyle\sum_{a\in A} z_a \label{eq:lp-objective}\\
\text{s.t.}\quad
& \textstyle\sum_{p} x_{a,p}\le 1, \label{eq:lp-choice}\\
& B_{a,p,0,r_a}=L_a x_{a,p},\qquad B_{a,p,h,r_a+h}=f_{a,p,h-1,r_a+h-1}\ (h\ge 1), \label{eq:lp-init}\\
& B_{a,p,h,n+1}=B_{a,p,h,n}-f_{a,p,h,n}+f_{a,p,h-1,n},\qquad f_{a,p,h,n}\le B_{a,p,h,n}, \label{eq:lp-buffer}\\
& L_a z_a\le \textstyle\sum_{p}\sum_{n} f_{a,p,H_p-1,n}, \label{eq:lp-deadline}\\
& \textstyle\sum_{(a,p,h)\,\text{qua}\,e} f_{a,p,h,n}\le t_e\Delta\bigl(R(b_k,s_{e,n})+R'(b_k,s_{e,n})(b_{e,n}-b_k)\bigr)\quad \forall k, \label{eq:lp-radio}\\
& \textstyle\sum_{(a,p,h)\,\text{qua}\,e} f_{a,p,h,n}\le \Delta C_e, \label{eq:lp-backhaul}\\
& \textstyle\sum_{e\,\text{truy nhập}} b_{e,n}\le B_{\mathrm{tot}},\qquad \sum_{e\,\text{downlink}} b_{e,n}\le B_{\mathrm{tot}}, \label{eq:lp-spectrum}\\
& x,z\in[0,1],\qquad f,B,b\ge 0, \label{eq:lp-domain}
\end{align}
\end{subequations}
trong đó $f_{a,p,-1,n}\equiv 0$, $H_p$ là số chặng của candidate $p$, \eqref{eq:lp-buffer} áp dụng khi $n+1\le d_a-1$, \eqref{eq:lp-radio} áp dụng cho liên kết vô tuyến có $s_{e,n}>0$ (nếu $s_{e,n}=0$ thì luồng qua liên kết bằng $0$) và \eqref{eq:lp-backhaul} cho cạnh backhaul. Gọi $\mathrm{UB}_\omega(q)$ là giá trị tối ưu.

\begin{proposition}[Cận trên]
\label[proposition]{prop:ub}
Với quỹ đạo $q$, tập candidate và realization $\omega$ cố định, $\mathrm{UB}_\omega(q)$ không nhỏ hơn số cảnh báo đúng hạn của mọi kế hoạch có quỹ đạo $q$ khi thực thi trên $\omega$.
\end{proposition}

\begin{proof}
Với $s>0$ cố định, $R(b,s)=b\,\phi(1/b)$ với $\phi(t)=\log_2(1+st)$ lõm, nên $R(\cdot,s)$ là phép phối cảnh của một hàm lõm và do đó lõm trên $b>0$~\cite{boyd2004convex}. Một hàm lõm nằm dưới mọi tiếp tuyến, nên $R(b,s)\le R(b_k,s)+R'(b_k,s)(b-b_k)$ với mọi $k$.
Xét một kế hoạch bất kỳ có quỹ đạo $q$ và quá trình thực thi của nó trên $\omega$. Đặt $x_{a,p}=1$ cho đường được chọn, $z_a=z_{a,\omega}$, $f_{a,p,h,n}$ bằng số bit truyền qua chặng $h$ trong slot $n$, $B_{a,p,h,n}$ bằng buffer ở đầu slot và $b_{e,n}$ bằng băng thông được cấp. Ràng buộc nhân quả và buffer cho \eqref{eq:lp-init}--\eqref{eq:lp-buffer}; \cref{def:timely} cho \eqref{eq:lp-deadline}; ràng buộc phổ và dung lượng cho \eqref{eq:lp-spectrum} và \eqref{eq:lp-backhaul}; bất đẳng thức tiếp tuyến cho \eqref{eq:lp-radio}. Điểm này khả thi và có giá trị mục tiêu bằng số cảnh báo đúng hạn, nên không vượt $\mathrm{UB}_\omega(q)$.
\end{proof}

Bài toán \eqref{eq:lp} còn nới lỏng tính nguyên của $x,z$, thứ tự EDF, giới hạn hai downlink và biết trước kênh; mỗi nới lỏng chỉ làm miền khả thi rộng thêm. Cận chỉ hợp lệ với quỹ đạo và tập candidate đã cho, không phải cận toàn cục trên mọi quỹ đạo. Với mỗi scenario, cận được lấy trung bình trên các realization và chia cho $\abs{A}$; gap của một phương pháp là $(\overline{\mathrm{UB}}-\overline{P})/\overline{\mathrm{UB}}$. B1 được so với cận trên quỹ đạo B1; B0 được so với cận của cùng bài toán khi chỉ giữ các candidate mặt đất.

Để đo độ chặt, trên các instance thu nhỏ bài toán \eqref{eq:lp} được giải lại với $x,z$ nhị phân (MILP). Kế hoạch dựng từ nghiệm MILP --- đường đã chọn và băng thông của nghiệm, giữ hai downlink lớn nhất nếu vượt giới hạn --- được thực thi trên cùng $\omega$, cho chuỗi
\begin{equation}
\label{eq:sandwich}
P_\omega(\text{kế hoạch MILP})\ \le\ \mathrm{OPT}_\omega(q)\ \le\ \mathrm{UB}^{\mathrm{MILP}}_\omega(q)\ \le\ \mathrm{UB}_\omega(q).
\end{equation}
Kế hoạch MILP biết trước kênh nên chỉ dùng để đo độ chặt của cận, không dùng để so sánh phương pháp.

\subsection{Độ khó của bài toán}
\label{sec:hardness}

\begin{proposition}
\label[proposition]{prop:hardness}
Ngay cả khi quỹ đạo cố định, dữ liệu được chia nhỏ tuỳ ý giữa các slot, mạng chỉ có hai relay và mỗi cảnh báo có hai candidate path, bài toán tối đa số cảnh báo đúng hạn là NP-khó.
\end{proposition}

\begin{proof}
Quy dẫn từ PARTITION: cho các số nguyên dương $s_1,\dots,s_m$ có tổng $2Q$, hỏi có tập con có tổng $Q$ hay không. Dựng mạng gồm trung tâm $c$, hai relay $v_1,v_2$ và hai cạnh backhaul $(v_j,c)$ có $\Delta C_{(v_j,c)}=Q$ bit. Một nguồn nằm trong tầm của cả hai relay, với độ lợi đủ lớn để $\tau\Delta\,R(B_{\mathrm{tot}}/2,s)\ge 2Q$. Cảnh báo $i$ có $L_i=s_i$, $r_i=0$, $d_i=2$ và hai candidate: qua $v_1$ hoặc qua $v_2$. Theo quy ước nhân quả, bit của cảnh báo đúng hạn phải được gửi lên relay trong slot $0$ và đi qua backhaul trong slot $1$.
Nếu tồn tại tập con có tổng $Q$, cho các cảnh báo tương ứng đi qua $v_1$, phần còn lại qua $v_2$, và cấp $B_{\mathrm{tot}}/2$ cho mỗi liên kết truy nhập trong slot $0$: mỗi liên kết truy nhập mang được $2Q$ bit và mỗi cạnh backhaul mang đúng $Q$ bit trong slot $1$, nên cả $m$ cảnh báo đúng hạn.
Ngược lại, nếu cả $m$ cảnh báo đúng hạn thì tổng kích thước các cảnh báo qua $v_j$ không vượt $Q$ với $j=1,2$, trong khi tổng hai phần bằng $2Q$; vậy mỗi phần bằng $Q$ và PARTITION có lời giải. Phép dựng thực hiện trong thời gian đa thức, nên quyết định ``có ít nhất $m$ cảnh báo đúng hạn hay không'' là NP-khó.
\end{proof}

\begin{remark}
\Cref{prop:hardness} chỉ khẳng định NP-khó theo nghĩa yếu; báo cáo không khẳng định bài toán thuộc NP (băng thông là biến liên tục) và không xét trường hợp $K=1$. Độ khó đến từ ràng buộc mỗi cảnh báo đi theo đúng một đường, không đến từ việc lập lịch trên một liên kết: khi chỉ có một liên kết và dữ liệu được chia nhỏ theo slot, bài toán tương ứng với lập lịch một máy có ngắt quãng $1\,|\,r_j,\mathrm{pmtn}\,|\,\sum U_j$, vốn giải được trong thời gian đa thức~\cite{lawler1990preemptive}. Phiên bản không ngắt quãng $1\,|\,r_j\,|\,\sum U_j$ là NP-khó mạnh~\cite{lenstra2020elements}, nhưng mô hình ở đây cho phép chia bit theo slot nên kết quả đó không áp dụng trực tiếp.
\end{remark}

\subsection{Độ phức tạp tính toán}
\label{sec:complexity}

Sinh candidate path tốn $O(\abs{V}+\abs{E})$ cho một lần tìm kiếm theo chiều rộng từ $c$ và $O(\abs{A}\abs{V}\log\abs{V})$ cho việc xếp hạng. Đánh giá một kế hoạch trên $M$ realization tốn $O\bigl(MN(\abs{\mathcal L}+\abs{A}\log\abs{A})\bigr)$, với $\mathcal L$ là tập liên kết; bộ kiểm tra độc lập có cùng bậc, còn $\mathrm{Conn}$ tốn $O\bigl(MN(\abs{V}+\abs{E}+\abs{S}\abs{V})\bigr)$. B0 lập kế hoạch trong $O(\abs{A}K)$ và B1 trong $O(\abs{A}KNH_{\max})$. Bài toán \eqref{eq:lp} có $\sum_a\sum_p H_p(d_a-r_a)$ biến luồng và cùng số biến buffer, tối đa $\abs{\mathcal L_{\mathrm{vt}}}N$ biến băng thông và tối đa $10\abs{\mathcal L_{\mathrm{vt}}}N$ ràng buộc tiếp tuyến, với $\mathcal L_{\mathrm{vt}}$ là các liên kết vô tuyến của tập candidate. Cận được giải bằng HiGHS qua \texttt{scipy}; báo cáo không đưa ra cận thời gian đa thức cho bộ giải mà đo thời gian thực của mọi bài toán LP (\cref{sec:experiments}).

\subsection{Phương pháp đề xuất cho Pha 2}
\label{sec:algorithm}

\Cref{alg:bcd-repair} là phương pháp dự kiến cho Pha 2: phân rã theo ba khối như cấu trúc BCD của~\cite{huang2025ddatsap}, sau đó áp dụng hai thao tác sửa lịch cho các cảnh báo có độ dư thời gian thấp. Phương pháp này chưa được cài đặt và chưa được đánh giá trong Pha 1.

\begin{algorithm}[htbp]
\caption{Phân rã theo khối kết hợp sửa lịch theo độ dư thời gian (Pha 2, chưa đánh giá)}
\label{alg:bcd-repair}
\begin{algorithmic}[1]
\Require Scenario, tập candidate, số vòng tối đa $I_{\max}$
\Ensure Quỹ đạo $q$, lựa chọn đường $x$, băng thông $b$
\State Khởi tạo quỹ đạo khả thi $q^{(0)}$ và lựa chọn đường $x^{(0)}$
\For{$i = 1, \dots, I_{\max}$}
    \State Cập nhật $b$ với $(q, x)$ cố định
    \State Cập nhật $x$ với $(q, b)$ cố định \Comment{hoán đổi trong tập candidate}
    \State Cập nhật $q$ với $(x, b)$ cố định \Comment{tìm kiếm cục bộ}
    \State Tính độ dư thời gian của từng cảnh báo; chọn tập nguy cơ $A_{\mathrm{risk}}$
    \ForAll{$a \in A_{\mathrm{risk}}$}
        \State Thử chuyển băng thông từ luồng còn dư sang $a$
        \State Thử đổi đường của $a$ trong tập candidate
        \State Nhận thay đổi nếu mục tiêu từ điển không giảm
    \EndFor
    \If{không cải thiện trong $\kappa$ vòng liên tiếp} \textbf{break} \EndIf
\EndFor
\State \Return $(q, x, b)$
\end{algorithmic}
\end{algorithm}
```

- [ ] **Step 5: Rewrite the related-work section**

Replace the whole content of `docs/report/sections/related-work.tex` with:

```latex
% docs/report/sections/related-work.tex
\section{Nghiên cứu liên quan}
\label{sec:related}

\paragraph{UAV relay và tối ưu quỹ đạo.}
Huang và cộng sự~\cite{huang2025ddatsap} nghiên cứu hệ relay hai chiều dùng UAV cho nhiều cặp người dùng mặt đất dưới mô hình kênh PrLoS, đề xuất giao thức cấp phát khe thời gian hai ăng-ten động (DDATSAP) và tối đa tốc độ trung bình nhỏ nhất bằng cách tối ưu đồng thời hệ số lập lịch tài nguyên, công suất phát và quỹ đạo, với ràng buộc vùng cấm bay. Bài toán không lồi được giải bằng thuật toán lặp kết hợp phân rã theo khối (BCD) và xấp xỉ lồi liên tiếp (SCA). Đề tài này kế thừa mô hình kênh PrLoS và cấu trúc phân rã đó, nhưng thay mục tiêu bằng tỷ lệ cảnh báo giao đúng hạn, bổ sung lựa chọn relay trên mạng mặt đất có dung lượng backhaul hữu hạn, và dùng giao thức hai nửa slot bán song công thay cho DDATSAP.

\paragraph{Lập lịch với thời điểm phát sinh và hạn.}
Khi chỉ xét một liên kết, việc chọn tập cảnh báo giao đúng hạn gần với bài toán lập lịch một máy tối thiểu số công việc trễ. Phiên bản không ngắt quãng $1\,|\,r_j\,|\,\sum U_j$ là NP-khó mạnh~\cite{lenstra2020elements}, còn phiên bản có ngắt quãng $1\,|\,r_j,\mathrm{pmtn}\,|\,\sum U_j$ giải được bằng quy hoạch động trong thời gian đa thức~\cite{lawler1990preemptive}. Mô hình ở \cref{sec:method} cho phép chia dữ liệu theo slot, gần với trường hợp có ngắt quãng; độ khó của nó đến từ ràng buộc mỗi cảnh báo đi theo đúng một đường (\cref{sec:hardness}).

\paragraph{Tối ưu không lồi và cận trên.}
Cận trên ở \cref{sec:bound} dựa trên tính lõm của hàm tốc độ theo băng thông và việc thay hàm lõm bằng bao các tiếp tuyến, là các công cụ chuẩn của tối ưu lồi~\cite{boyd2004convex}.

\paragraph{Phạm vi của phần này.}
Tổng quan có hệ thống về UAV relay có hạn giao, lựa chọn relay với backhaul, luồng trên đồ thị mở rộng theo thời gian và phân bổ băng thông, cùng việc chọn một phương pháp mạnh từ tài liệu để so sánh, thuộc Pha 2 và chưa được thực hiện trong báo cáo Pha 1.

\begin{table}[htbp]
\centering
\small
\caption{So sánh với công trình tham chiếu.}
\label{tab:related}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Công trình} & \textbf{Mục tiêu} & \textbf{Chọn relay} & \textbf{Hạn giao} & \textbf{Kênh} & \textbf{Phương pháp} \\
\midrule
\cite{huang2025ddatsap} & Max--min tốc độ & Không & Không & PrLoS & BCD--SCA \\
Đề tài này & Tỷ lệ đúng hạn & Có & Có & PrLoS & B0, B1, cận LP (Pha 1) \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 6: Compile and check references**

Run: `cd docs/report && mkdir -p build && bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --preview --preview-dir build/preview 2> build/compile-stderr.txt; cd ../..`

Expected: `docs/report/main.pdf` and PNG previews in `docs/report/build/preview/` are written. The skill script deletes `main.log` after its own log analysis, whose findings go to `build/compile-stderr.txt`.

Run: `grep -c -E "^  ! (Undefined|Environment)" docs/report/build/compile-stderr.txt; pdftotext docs/report/main.pdf - | grep -c '??'`

Expected: `0` twice: the skill's analysis reports no undefined control sequence, citation, or environment, and the PDF text contains no `??` from a broken reference. Open the preview pages of the method section and confirm that the LP (`eq:lp`) and both propositions render without lines running into the margin.

- [ ] **Step 7: Commit**

```bash
git add docs/report/preamble.tex docs/report/tables/params.tex docs/report/tables/phase1_main.tex docs/report/tables/phase1_milp.tex docs/report/sections/method.tex docs/report/sections/related-work.tex docs/report/references.bib
git commit -m "docs: write phase 1 model, bound, and hardness sections"
```

### Task 11: Report introduction, experiments, discussion, conclusion, and README

**Files:**
- Modify: `docs/report/main.tex` (load the result macros)
- Modify (full rewrite): `docs/report/sections/abstract.tex`, `docs/report/sections/introduction.tex`, `docs/report/sections/experiments.tex`, `docs/report/sections/discussion.tex`, `docs/report/sections/conclusion.tex`
- Modify: `README.md` (append the Phase 1 section)

**Interfaces:**
- Consumes: Task 9 macros in `docs/report/data/phase1_results.tex` and CSV files; Task 10 labels and tables.
- Produces: the compiled Phase 1 report `docs/report/main.pdf` (ignored by Git) and its preview pages.

The text states only facts that the committed Task 8 results support; every number comes from a macro or a table.

- [ ] **Step 1: Load the result macros**

In `docs/report/main.tex`, replace

```latex
\input{preamble}
```

with

```latex
\input{preamble}
\input{data/phase1_results}  % macro số liệu Pha 1, sinh bởi experiments/report_tables.py
```

- [ ] **Step 2: Rewrite the abstract**

Replace the whole content of `docs/report/sections/abstract.tex` with:

```latex
% docs/report/sections/abstract.tex
\begin{abstract}
\noindent
Sau thiên tai, một phần mạng mặt đất bị hỏng và các cảnh báo phát sinh tại hiện trường cần được đưa về trung tâm cứu hộ trước hạn. Báo cáo này nghiên cứu bài toán đồng tối ưu quỹ đạo của một UAV làm relay di động, lựa chọn relay hoặc đường chuyển tiếp và phân bổ băng thông, nhằm tối đa tỷ lệ cảnh báo giao đúng hạn dưới kênh không--đất theo mô hình xác suất tầm nhìn thẳng~\cite{huang2025ddatsap}. Chúng tôi chứng minh bài toán là NP-khó ngay cả khi quỹ đạo cố định, xây dựng cận trên quy hoạch tuyến tính hợp lệ theo từng realization kênh, và đánh giá hai phương pháp cơ sở trên các kịch bản tổng hợp từ Topology Zoo~\cite{knight2011zoo} và SNDlib~\cite{orlowski2010sndlib}. Trên \ResultScenarioCount{} kịch bản đánh giá, UAV bay theo quỹ đạo cố định giao đúng hạn \ResultTimelyVzeroBone{} số cảnh báo so với \ResultTimelyVzeroBzero{} khi không có UAV và đưa mọi nguồn về trạng thái có kết nối, nhưng vẫn cách cận trên một gap \ResultGapVzeroBone, và gap này tăng khi backhaul băng hẹp. Kết quả Pha 1 là nền cho phương pháp phân rã kết hợp sửa lịch ở Pha 2.

\vspace{0.5em}
\noindent
\textbf{Từ khoá:} UAV relay, quỹ đạo, lựa chọn relay, phân bổ băng thông, hạn giao, cận trên quy hoạch tuyến tính.
\end{abstract}
```

- [ ] **Step 3: Rewrite the introduction**

Replace the whole content of `docs/report/sections/introduction.tex` with:

```latex
% docs/report/sections/introduction.tex
\section{Giới thiệu}
\label{sec:intro}

Sau thiên tai, một phần mạng mặt đất bị đứt hoặc suy giảm trong khi các đội cứu hộ, cảm biến và người dân tại hiện trường liên tục phát sinh cảnh báo cần đưa về trung tâm điều phối. Giá trị của một cảnh báo phụ thuộc vào việc nó tới nơi trước một hạn nhất định: một tin báo sập nhà tới muộn có thể không còn giúp được việc cứu người. Vì vậy thước đo phù hợp không phải là tốc độ truyền trung bình mà là tỷ lệ cảnh báo được giao đủ dữ liệu trước hạn.

UAV làm relay di động có thể nối các khu vực mất kết nối với phần mạng còn hoạt động nhờ khả năng triển khai nhanh và liên kết tầm nhìn thẳng với thiết bị mặt đất. Huang và cộng sự~\cite{huang2025ddatsap} tối ưu đồng thời lập lịch tài nguyên, công suất và quỹ đạo của một UAV relay dưới kênh PrLoS để tối đa tốc độ trung bình nhỏ nhất. Mục tiêu đó không phản ánh hạn giao của từng cảnh báo, và công trình đó không xét việc chọn relay trên một mạng mặt đất có dung lượng backhaul hữu hạn. Trong bối cảnh cứu hộ, một relay gần nguồn chưa chắc là lựa chọn tốt vì nó có thể mất backhaul hoặc quá tải, và vị trí có kênh tốt nhất cho UAV chưa chắc giúp giao được nhiều cảnh báo đúng hạn nhất.

Báo cáo này nghiên cứu bài toán đồng tối ưu quỹ đạo của một UAV, lựa chọn relay hoặc đường chuyển tiếp cho từng cảnh báo và phân bổ băng thông, nhằm tối đa tỷ lệ cảnh báo giao đúng hạn; khả năng kết nối theo thời gian tới trung tâm là mục tiêu phụ theo thứ tự từ điển. Nghiên cứu hoàn toàn bằng mô phỏng trên các kịch bản tổng hợp từ dữ liệu công khai Topology Zoo~\cite{knight2011zoo} và SNDlib~\cite{orlowski2010sndlib}.

\paragraph{Đóng góp của Pha 1.}
Thứ nhất, chúng tôi mô hình hoá bài toán với thời điểm phát sinh, hạn và kích thước của cảnh báo, mạng relay có backhaul hữu hạn, kênh PrLoS và quy ước nhân quả theo slot, cùng một bộ đánh giá độc lập với mọi phương pháp và bộ kịch bản tái lập được (\cref{sec:method}). Thứ hai, chúng tôi cài đặt hai phương pháp cơ sở B0 và B1, xây dựng cận trên quy hoạch tuyến tính theo từng realization kênh kèm chứng minh tính hợp lệ, và đối chiếu cận với MILP trên các instance thu nhỏ (\cref{sec:bound}). Thứ ba, chúng tôi chứng minh bài toán là NP-khó ngay cả khi quỹ đạo cố định và phân tích độ phức tạp tính toán (\cref{sec:hardness,sec:complexity}). Thứ tư, chúng tôi đánh giá hai phương pháp cơ sở và cận trên trên hai họ kịch bản, với backhaul thông thường và backhaul băng hẹp (\cref{sec:experiments}).

Phương pháp phân rã theo khối kết hợp sửa lịch theo độ dư thời gian, các phép ablation, kiểm định thống kê ghép cặp, so sánh với một phương pháp mạnh từ tài liệu và case study RescueNet thuộc Pha 2 và chưa được trình bày ở đây.

Phần còn lại của báo cáo được tổ chức như sau. \Cref{sec:related} điểm qua nghiên cứu liên quan. \Cref{sec:method} trình bày mô hình, phương pháp cơ sở, cận trên và phân tích độ khó. \Cref{sec:experiments} báo cáo kết quả Pha 1, \cref{sec:discussion} thảo luận và \cref{sec:conclusion} kết luận.
```

- [ ] **Step 4: Rewrite the experiments section**

Replace the whole content of `docs/report/sections/experiments.tex` with:

```latex
% docs/report/sections/experiments.tex
\section{Kết quả thực nghiệm (Pha 1)}
\label{sec:experiments}

\subsection{Thiết lập}
\label{sec:setup}

Bộ kịch bản v0 được sinh tất định từ 130 topology Topology Zoo có 15--40 nút, có toạ độ và liên thông. Các topology được chia tầng theo số nút; mười topology được dùng để đánh giá với ba replicate mỗi topology, ba topology khác dành cho phát triển với hai replicate và không được dùng trong Pha 1, vì các phương pháp cơ sở không có tham số cần hiệu chỉnh. Kích thước cảnh báo lấy tỷ lệ tương đối từ demand của instance SNDlib \texttt{abilene}~\cite{orlowski2010sndlib}; vùng hư hại, điểm nguồn, thời điểm phát sinh, hạn và cạnh hỏng được sinh bằng các luồng số ngẫu nhiên có seed riêng. Họ v0-bh50 giữ nguyên mọi seed và chỉ giảm dung lượng backhaul, nên hai họ cùng topology, cảnh báo và mẫu cạnh hỏng. \Cref{tab:params} liệt kê các tham số. Mọi kết hợp là kịch bản tổng hợp từ dữ liệu công khai, không phải mạng cứu hộ đo thực địa.

Mỗi phương pháp lập kế hoạch một lần cho mỗi scenario và được đánh giá trên 30 realization kênh dùng chung số ngẫu nhiên; cận trên được giải cho từng realization. Trên mỗi họ, kết quả được lấy trung bình theo realization rồi theo \ResultScenarioCount{} scenario đánh giá; khoảng tin cậy 95\% dùng bootstrap theo cụm topology, vì các replicate của cùng một topology không độc lập. Kiểm định ghép cặp giữa các phương pháp thuộc Pha 2. Thí nghiệm chạy bằng Python \ResultPythonVersion, \texttt{scipy} \ResultScipyVersion{} và HiGHS \ResultHighsVersion{} với \ResultWorkers{} tiến trình trên máy \ResultCpuCount{} CPU; tổng thời gian tính của các task là \ResultComputeMinutes{} phút.

\input{tables/params}

\subsection{Kết quả chính}
\label{sec:results-main}

\input{tables/phase1_main}

\Cref{tab:phase1-main} tóm tắt kết quả. Trên họ v0, B1 giao đúng hạn trung bình \ResultTimelyVzeroBone{} số cảnh báo, so với \ResultTimelyVzeroBzero{} của B0; B1 cao hơn B0 trên \ResultBoneWinsVzero{} trong \ResultScenarioCount{} scenario và bằng B0 trên \ResultBoneTiesVzero{} scenario. Khả năng kết nối cho thấy vai trò của UAV rõ hơn: không có UAV, chỉ \ResultConnVzeroBzero{} nguồn có đường theo thời gian tới trung tâm, còn với quỹ đạo B1 tỷ lệ này là \ResultConnVzeroBone. Khoảng tin cậy của hai phương pháp chồng lấn; kết luận thống kê về chênh lệch cần kiểm định ghép cặp ở Pha 2.

Gap của B0 so với cận chỉ dùng mặt đất là \ResultGapVzeroBzero, còn gap của B1 so với cận trên quỹ đạo B1 là \ResultGapVzeroBone. Hai cận dùng thông tin khác nhau, nên gap đo phần còn có thể cải thiện trong cùng điều kiện của từng phương pháp, không dùng để so trực tiếp hai phương pháp. Với quỹ đạo B1 cố định, cận trên cho thấy việc chọn đường và cấp băng thông tốt hơn có thể giao tới \ResultBoundVzeroBone{} số cảnh báo.

Khi backhaul chỉ còn \qty{50}{\kilo\bit\per\second}, tỷ lệ đúng hạn giảm còn \ResultTimelyVzeroBhfivezeroBzero{} với B0 và \ResultTimelyVzeroBhfivezeroBone{} với B1, trong khi gap tăng lên \ResultGapVzeroBhfivezeroBzero{} và \ResultGapVzeroBhfivezeroBone. B1 vẫn cao hơn B0 trên \ResultBoneWinsVzeroBhfivezero{} scenario. Gap tăng cho thấy quy tắc chia đều băng thông và chọn đường theo độ trễ ước tính của các phương pháp cơ sở không thích nghi với nghẽn backhaul, là tình huống mà cơ chế sửa lịch ở Pha 2 nhắm tới.

\subsection{Độ chặt của cận trên}
\label{sec:results-milp}

\input{tables/phase1_milp}

\Cref{tab:phase1-milp} đối chiếu cận LP với MILP trên \ResultMilpCount{} instance thu nhỏ. HiGHS giải tối ưu \ResultMilpOptimalCount{} trong \ResultMilpCount{} bài MILP. Trên các instance này, B1 đạt đúng giá trị tối ưu MILP ở \ResultMilpMethodAtOptimumCount{} instance và kế hoạch dựng từ nghiệm MILP đạt tối ưu ở \ResultMilpPlanAtOptimumCount{} instance, nên khoảng cách giữa cận LP và giá trị đạt được chủ yếu đến từ việc nới lỏng tính nguyên của lựa chọn đường. Khi dựng kế hoạch, băng thông lấy thẳng từ nghiệm MILP chỉ giao đúng hạn tổng cộng \ResultMilpStaticTotal{} cảnh báo trên năm instance, so với \ResultMilpEqualSplitTotal{} khi giữ các đường của nghiệm và chia đều băng thông theo backlog, vì lịch băng thông tĩnh lệch nhịp với cách bộ điều phối EDF phục vụ hàng đợi. Bao mười tiếp tuyến ước lượng tốc độ cao hơn thực tế tối đa \ResultTangentOverestimateMax{} (theo tỷ lệ tương đối) trên $[B_{\mathrm{tot}}/512, B_{\mathrm{tot}}]$. Vì vậy gap trên các scenario đầy đủ ở \cref{tab:phase1-main} nên được hiểu là cận trên của phần có thể cải thiện, có thể lớn hơn đáng kể so với phần thực sự đạt được.

\subsection{Thời gian tính toán}
\label{sec:results-runtime}

Lập kế hoạch B0 gần như tức thời, còn B1 mất trung bình \ResultPlanningVzeroBone{} giây cho mỗi scenario v0 vì phải mô phỏng từng candidate; đánh giá một kế hoạch trên một realization mất khoảng \ResultEvaluationVzeroBone{} giây. \Cref{fig:lp-runtime} cho thấy thời gian giải của \ResultLpCount{} bài LP: số biến có trung vị \ResultLpVariablesMedian{} và lớn nhất \ResultLpVariablesMax{}, thời gian giải có trung vị \ResultLpRuntimeMedian{} giây và lớn nhất \ResultLpRuntimeMax{} giây. Cận trên quỹ đạo B1 trong họ v0-bh50 mất trung bình \ResultBoundTimeVzeroBhfivezeroBone{} giây, lâu hơn \ResultBoundTimeVzeroBone{} giây của họ v0.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{loglogaxis}[
    width=0.8\textwidth, height=6cm,
    xlabel={Số biến của bài toán LP}, ylabel={Thời gian giải (s)},
    grid=major, /pgf/number format/use comma,
]
\addplot[only marks, mark=*, mark size=0.5pt, color=blue!60!black, opacity=0.4]
    table[col sep=comma, x=variables, y=runtime] {data/lp_runtime.csv};
\end{loglogaxis}
\end{tikzpicture}
\caption{Thời gian giải của \ResultLpCount{} bài toán LP cận trên (hai họ scenario, hai loại cận, 30 realization) theo số biến.}
\label{fig:lp-runtime}
\end{figure}
```

- [ ] **Step 5: Rewrite the discussion**

Replace the whole content of `docs/report/sections/discussion.tex` with:

```latex
% docs/report/sections/discussion.tex
\section{Thảo luận}
\label{sec:discussion}

\paragraph{Vai trò của UAV.}
Trên cả hai họ kịch bản, một quỹ đạo cố định đơn giản đã nâng khả năng kết nối lên mọi nguồn và tăng tỷ lệ cảnh báo đúng hạn so với mạng mặt đất đơn thuần. Tuy vậy, khoảng tin cậy chồng lấn và gap so với cận trên còn lớn, nhất là khi backhaul băng hẹp. Điều này cho thấy lợi ích của UAV phụ thuộc vào việc quỹ đạo, lựa chọn đường và băng thông được phối hợp, không chỉ vào việc có UAV hay không.

\paragraph{Độ chặt của cận.}
Cận LP hợp lệ cho mọi kế hoạch có cùng quỹ đạo và tập candidate, nhưng nó biết trước kênh và nới lỏng tính nguyên. Trên instance thu nhỏ, phần lớn khoảng cách giữa cận và giá trị đạt được đến từ nới lỏng tính nguyên, nên gap trên kịch bản đầy đủ có thể phóng đại phần còn cải thiện được. Việc siết cận, chẳng hạn bằng MILP trên instance lớn hơn hoặc thêm bất đẳng thức hợp lệ, là một hướng của Pha 2.

\paragraph{Giới hạn.}
Kết quả dựa trên mô phỏng và không chứng minh hiệu năng thực địa. Mạng backbone gốc được thu nhỏ về vùng \qty{2}{\kilo\metre}, nên chỉ còn mượn cấu trúc. Trạng thái kênh độc lập giữa các slot, kênh mặt đất tất định và công suất cố định. Khoảng tin cậy dựa trên mười cụm topology nên còn rộng. Cận chỉ hợp lệ với quỹ đạo và tập candidate đã cho, không phải cận toàn cục. Các phương pháp cơ sở là heuristic và không có bảo đảm tối ưu.

\paragraph{Hướng tiếp theo.}
Pha 2 sẽ cài đặt BCD kiểu Huang (B2), biến thể dùng mục tiêu max--min (B3) và phương pháp đề xuất với hai thao tác sửa lịch (P), thực hiện ablation, phân tích độ nhạy và kiểm định thống kê ghép cặp, so sánh với một phương pháp mạnh từ tài liệu, và bổ sung case study với mục tiêu lấy từ RescueNet.
```

- [ ] **Step 6: Rewrite the conclusion**

Replace the whole content of `docs/report/sections/conclusion.tex` with:

```latex
% docs/report/sections/conclusion.tex
\section{Kết luận}
\label{sec:conclusion}

Báo cáo Pha 1 đã mô hình hoá bài toán đồng tối ưu quỹ đạo UAV, lựa chọn relay và phân bổ băng thông cho cảnh báo có hạn giao, chứng minh bài toán NP-khó ngay cả khi quỹ đạo cố định, và xây dựng một cận trên quy hoạch tuyến tính hợp lệ theo từng realization kênh. Trên \ResultScenarioCount{} kịch bản đánh giá với backhaul thông thường, phương pháp cơ sở dùng UAV bay theo quỹ đạo cố định giao đúng hạn \ResultTimelyVzeroBone{} số cảnh báo, so với \ResultTimelyVzeroBzero{} khi không có UAV, và gap so với cận trên còn \ResultGapVzeroBone. Khi backhaul băng hẹp, cả hai phương pháp suy giảm và gap tăng. Pha 2 sẽ tận dụng khoảng cách này bằng phân rã theo khối và cơ chế sửa lịch theo độ dư thời gian.
```

- [ ] **Step 7: Document the Phase 1 commands**

Append to `README.md`:

````markdown
## Pha 1: baseline, cận trên và báo cáo

Sinh họ scenario backhaul yếu `v0-bh50` (giống v0, chỉ đổi dung lượng backhaul còn 50 kbit/s) và kiểm tra hiệu chỉnh:

```bash
uv run python -m data.cli scenarios --config configs/scenarios/v0-bh50.yaml
uv run python experiments/calibrate_v0.py --manifest data/manifests/scenarios_v0-bh50.csv --scenario-dir data/processed/scenarios/v0-bh50 --output results/calibration/v0-bh50.json
```

Chạy thí nghiệm Pha 1 (B0, B1, cận trên LP theo từng realization và đối chiếu MILP) trên cả hai họ scenario:

```bash
uv run python experiments/run_phase1.py --config configs/experiments/phase1.yaml
```

Kết quả nằm trong `results/phase1/`; `results/phase1/shards/` (bị Git ignore) lưu từng task nên chạy lại lệnh sẽ bỏ qua task đã xong. Lệnh trả mã khác 0 nếu có task lỗi, LP không tối ưu hoặc kiểm tra tính hợp lệ của cận thất bại.

Sinh dữ liệu cho báo cáo (bảng tham số, bảng kết quả, macro số liệu, dữ liệu hình runtime) rồi biên dịch báo cáo bằng script của skill LaTeX:

```bash
uv run python experiments/report_tables.py
cd docs/report && bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --preview --preview-dir build/preview
```
````

- [ ] **Step 8: Compile and check the report**

Run: `cd docs/report && mkdir -p build && bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --preview --preview-dir build/preview 2> build/compile-stderr.txt; cd ../..`

Expected: `docs/report/main.pdf` and the PNG previews are rewritten.

Run: `grep -c -E "^  ! (Undefined|Environment)" docs/report/build/compile-stderr.txt; pdftotext docs/report/main.pdf - | grep -c '??'`

Expected: `0` twice.

Run: `grep -rn -E "\\[[^]]*(minh ho|Thay bằng|Đoạn|Kết quả chính|Một câu|nhận xét về|Bảng ablation|Hình sensitivity|Giải thích|Liệt kê|Phác thảo|phân tích theo|Tổng quan các)" docs/report/sections docs/report/tables`

Expected: no output: no placeholder remains.

Open every preview page in `docs/report/build/preview/` and confirm that tables fit the page width, the LP runtime figure shows points on log-log axes, and no `??` appears for references or citations.

- [ ] **Step 9: Run the full suite**

Run: `uv run --extra test pytest -q`

Expected: 119 passed, 1 skipped.

- [ ] **Step 10: Commit**

```bash
git add docs/report/main.tex docs/report/sections/abstract.tex docs/report/sections/introduction.tex docs/report/sections/experiments.tex docs/report/sections/discussion.tex docs/report/sections/conclusion.tex README.md
git commit -m "docs: report phase 1 results"
```

---

## Spec Coverage

| Spec section | Task |
|---|---|
| 3.3 Environment (uv, skill tooling) | Global constraints, 9, 10, 11 |
| 4 Architecture and file layout | 1–7, 9 |
| 5 Baselines B0, B1 and `Method` | 1 |
| 6.1–6.3 LP bound, validity, variants | 2, 6, 8 |
| 6.4 MILP cross-check | 3, 6, 8 |
| 7 Backhaul-stressed family | 4 |
| 8 Runner: config, tasks, resume, outputs, aggregation | 5, 6, 7, 8 |
| 9.1 Complexity analysis | 10 |
| 9.2 Hardness proposition | 10 |
| 9.3 Literature verification | 10 (verified entries only) |
| 10 Report updates | 9, 10, 11 |
| 11 Error handling | 5, 6, 7, 9 |
| 12 Testing | 1–7, 9 |
| 13 Acceptance criteria 1–6 | 1–11 (1: every task; 2: Task 4; 3–4: Task 8; 5: Tasks 10–11; 6: Task 11) |
