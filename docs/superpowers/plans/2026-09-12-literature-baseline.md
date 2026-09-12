# Literature Baseline Implementation Plan (Sub-Project D, Plan D.1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Tran et al. (2022) half-duplex design as the planning method `TRAN`, choose its parameters on the development split, run it on the `v0` and `v0-bh50` evaluation scenarios, and compare it with the committed P and B2 results after a B1 anchor check.

**Architecture:** A new package `src/literature` builds a Tran instance from a scenario (`tran_model`), runs the inner-approximation algorithm with cvxpy and Clarabel (`tran_ia`), converts iterates into static-schedule plans (`tran_plan`), and exposes the method `TranHD` (`tran`) and the pilot logic (`pilot`). The runner accepts the base `TRAN`, computes its trajectory bounds inside method tasks, and gains `--output-dir`. `experiments/tran_pilot.py` chooses θ, μ and the block size on the development split and writes `configs/experiments/phase2_literature.yaml`; `experiments/phase2_literature_statistics.py` pairs TRAN with the main results of plan C.2.

**Tech Stack:** Python 3.12 in the uv-managed `.venv`; `numpy`, `scipy` 1.18 (HiGHS), `cvxpy` 1.9 with Clarabel 0.11 (new), `PyYAML`, `pytest`; systemd user scopes for memory caps.

**Spec:** `docs/superpowers/specs/2026-09-12-extensions-final-delivery-design.md`: Sections 5–7, 14, 15.1, acceptance criteria 16.1, and Section 17 "Plan D.1 adjustments" (committed with this plan). Uses the committed results of plan C.2 in `results/phase2/main/`.

## Global Constraints

- Run project code through uv: tests with `uv run --extra test pytest`, scripts with `uv run python ...`. The only new dependency is `cvxpy>=1.6,<2`, added with `uv add` in Task 2; Clarabel comes with it.
- Import boundaries: `src/models`, `src/baselines`, `src/bounds`, `src/optimization` and `src/analysis` do not change; `src/literature` imports only `models` and `baselines`; `src/runner` imports `literature`; nothing under `src` imports `experiments`.
- **Memory caps.** Every command that solves TRAN programs runs inside a systemd user scope with a hard memory limit and no swap: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0` in front of pytest, `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0` in front of the pilot and the comparison experiment. A parametrized cvxpy prototype grew to 17 GB and exhausted WSL memory; under the cap the kernel stops only that process (exit code 137). Never stop processes with `pkill -f`, whose pattern also matches the calling shell.
- TRAN defaults: `theta` 0.5, `mu` 1, `max_iterations` 30, `tolerance` 1e-3, `block_slots` 1. Pilot grid: θ ∈ {1/3, 1/2, 2/3} × μ ∈ {1, 10}; block rule: median planning time above 600 s at θ = 1/2, μ = 1 switches to 5-slot blocks; Samir rule: the chosen TRAN not better than its initial point on at least half of the development scenarios stops the plan.
- Each inner-approximation iteration builds program (58) with numeric constants (no cvxpy parameters), approximates the power constraint with cvxpy's second-order cones, and solves with Clarabel at `tol_gap_abs = tol_gap_rel = 1e-6`, `tol_feas = 1e-7`, SciPy canonicalization backend.
- Experiments use split `eval`, realization ids 0–29, 10 tangents, bootstrap 10,000 resamples with seed 20260911, and 12 workers; the pilot uses the six development scenarios of `v0` with 30 evaluation realizations.
- Statistics: P − TRAN and B2 − TRAN on `v0` and `v0-bh50`, Holm over these four, with the functions of `src/analysis/statistics.py` and the columns of `results/phase2/statistics.csv`; the B1 anchor compares `timely_count`, `connected_count` and `shortfall` per (set, scenario, realization).
- Commit `runs.csv`, `channel_fit.csv`, `choice.json`, `pilot_note.md` of `results/phase2/tran_pilot/`; `method_realizations.csv`, `bounds.csv`, `summary.csv`, `manifest.json` of `results/phase2/literature/`; `results/phase2/statistics_literature.csv`. `results/phase2/**/shards/` stays Git-ignored.
- Test module basenames are unique across `tests/`; every code task follows test-driven development and ends with its own commit.

---

### Task 1: TRAN instance model

**Files:**
- Create: `src/literature/__init__.py`, `src/literature/tran_model.py`
- Test: `tests/support/tran_builders.py`, `tests/literature/test_tran_model.py`

**Interfaces:**
- Consumes: models.channel (`BACKHAUL`, `db_to_linear`, `ground_snr_per_hz`, `los_probability`, `noise_psd_w_per_hz`), models.paths (`GROUND`, `CandidatePath`, `candidate_paths`), `models.scenario.Scenario`, test helper `scenario_builders`.
- Produces: `literature.tran_model`: frozen `TranParams(theta=0.5, mu=1.0, max_iterations=30, tolerance=1e-3, block_slots=1)` with `from_mapping(raw) -> TranParams` (unknown keys raise `ValueError("unknown TRAN parameters ...")`); `EffectiveChannel(beta, alpha, r_squared, max_log_error)`; `TranDevice(alert_id, candidate, ground, source, gateway, size_bits, uplink, downlink, ground_efficiency)` with windows `(start, stop)` in blocks and property `active`; `TranInstance(scenario, devices, channel, block_slots, num_blocks, points, point_snr, uplink_device, uplink_block, uplink_point, downlink_device, downlink_block, downlink_point)`; `expected_uav_gain(scenario, horizontal_m)`, `ground_point_diameter(scenario) -> float`, `fit_effective_channel(scenario) -> EffectiveChannel`, `backhaul_slots(scenario, path, size_bits) -> int`, `build_device(scenario, candidates, alert_index, params) -> TranDevice`, `build_instance(scenario, candidates, params) -> TranInstance` (raises when `block_slots` does not divide `num_slots`). Test helper `tran_builders.tran_raw(num_slots=40, uav_size=200000)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/support/tran_builders.py`:

```python
import copy

from scenario_builders import BASE_SCENARIO, make_alert


def tran_raw(num_slots=40, uav_size=200000):
    """Ground-served source s00 and UAV-only source s02; alert-0001 and alert-0002 travel through the UAV to n1."""
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["sources"].append({"id": "s02", "x_m": 1500.0, "y_m": 1400.0, "zone": 0})
    raw["damage_zones"].append({"x_m": 1500.0, "y_m": 1400.0, "radius_m": 400.0})
    raw["time"]["num_slots"] = num_slots
    raw["alerts"] = [
        make_alert("alert-0000", "s00", 0, 20, 200000),
        make_alert("alert-0001", "s02", 0, 40, uav_size),
        make_alert("alert-0002", "s02", 5, 40, uav_size // 2),
    ]
    return raw
```

Create `tests/literature/test_tran_model.py`:

```python
import math

import pytest

from literature.tran_model import TranParams, backhaul_slots, build_instance, fit_effective_channel
from models.channel import db_to_linear, ground_snr_per_hz, noise_psd_w_per_hz
from models.paths import candidate_paths
from models.scenario import scenario_from_dict
from scenario_builders import make_alert
from tran_builders import tran_raw


def _instance(raw, **params):
    scenario = scenario_from_dict(raw)
    candidates = candidate_paths(scenario)
    return scenario, candidates, build_instance(scenario, candidates, TranParams(**params))


def test_params_validate_and_reject_unknown_keys():
    assert TranParams() == TranParams(theta=0.5, mu=1.0, max_iterations=30, tolerance=1e-3, block_slots=1)
    assert TranParams.from_mapping({"theta": 0.25, "mu": 10}) == TranParams(theta=0.25, mu=10.0)
    with pytest.raises(ValueError, match="unknown TRAN parameters"):
        TranParams.from_mapping({"gamma": 1})
    for bad, message in ((dict(theta=1.0), "theta"), (dict(mu=0), "mu"), (dict(tolerance=-1.0), "tolerance"),
                         (dict(max_iterations=True), "max_iterations"), (dict(block_slots=0), "block_slots")):
        with pytest.raises(ValueError, match=message):
            TranParams(**bad)


def test_effective_channel_recovers_a_pure_los_power_law():
    raw = tran_raw()
    raw["channel"]["uav"]["plos_a"] = 1e-12
    channel = fit_effective_channel(scenario_from_dict(raw))
    assert channel.alpha == pytest.approx(2.2, rel=1e-6)
    assert channel.beta == pytest.approx(db_to_linear(-50.0), rel=1e-6)
    assert channel.r_squared == pytest.approx(1.0, abs=1e-9) and channel.max_log_error < 1e-6


def test_devices_follow_cost_choice_backhaul_and_theta_windows():
    scenario, candidates, instance = _instance(tran_raw())
    ground, first, second = instance.devices
    assert (ground.ground, ground.candidate, ground.uplink, ground.downlink) == (True, 0, (0, 18), (0, 0))
    assert backhaul_slots(scenario, candidates["alert-0000"][0], 200000) == 2
    distance = math.dist(scenario.source_by_id["s00"].xy, scenario.node_by_id["n1"].xy)
    assert ground.ground_efficiency == pytest.approx(math.log2(1.0 + ground_snr_per_hz(scenario, distance) / 1e6))
    assert (first.ground, first.gateway, first.uplink, first.downlink) == (False, "n1", (0, 19), (19, 38))
    assert (second.uplink, second.downlink) == ((5, 22), (22, 38))
    assert instance.points == ("s02", "n1")
    noise = noise_psd_w_per_hz(scenario) * 1e6
    assert instance.point_snr.tolist() == pytest.approx([0.1 * instance.channel.beta / noise, 0.05 * instance.channel.beta / noise])
    assert (len(instance.uplink_device), len(instance.downlink_device)) == (18 + 19 + 17, 19 + 16)
    assert set(instance.uplink_point[instance.uplink_device == 0]) == {-1}


def test_blocks_round_windows_inward_and_empty_windows_deactivate():
    raw = tran_raw()
    raw["alerts"].append(make_alert("alert-0003", "s02", 30, 33, 100000))
    _, _, instance = _instance(raw, block_slots=5)
    ground, first, second, tight = instance.devices
    assert (ground.uplink, first.uplink, first.downlink, second.uplink, second.downlink) == ((0, 3), (0, 3), (4, 7), (1, 4), (5, 7))
    assert instance.num_blocks == 8 and not tight.active and 3 not in set(instance.uplink_device)
    with pytest.raises(ValueError, match="must divide"):
        _instance(tran_raw(), block_slots=3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/literature/test_tran_model.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'literature'` (`1 error`).

- [ ] **Step 3: Implement**

Create `src/literature/__init__.py`:

```python
"""Literature baselines adapted to the project model: Tran et al. (2022), half-duplex (spec D Section 5)."""
```

Create `src/literature/tran_model.py`:

```python
"""Tran et al. (2022) half-duplex instance built from a scenario (spec D Section 5.1)."""

from dataclasses import dataclass, fields
import math
from typing import Mapping

import numpy as np

from models.channel import BACKHAUL, db_to_linear, ground_snr_per_hz, los_probability, noise_psd_w_per_hz
from models.paths import GROUND, CandidatePath
from models.scenario import Scenario

FIT_STEP_M = 10.0
ROUNDING_TOLERANCE = 1e-9


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class TranParams:
    theta: float = 0.5
    mu: float = 1.0
    max_iterations: int = 30
    tolerance: float = 1e-3
    block_slots: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < _number(self.theta, "theta") < 1.0:
            raise ValueError(f"theta must lie in (0, 1), got {self.theta!r}")
        if _number(self.mu, "mu") <= 0.0:
            raise ValueError(f"mu must be positive, got {self.mu!r}")
        if _number(self.tolerance, "tolerance") <= 0.0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance!r}")
        _positive_integer(self.max_iterations, "max_iterations")
        _positive_integer(self.block_slots, "block_slots")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "TranParams":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown TRAN parameters {unknown}; known parameters: {sorted(known)}")
        return cls(**raw)


@dataclass(frozen=True)
class EffectiveChannel:
    """Power law beta * d^-alpha fitted to the expected PrLoS gain; r_squared and max_log_error judge the fit."""

    beta: float
    alpha: float
    r_squared: float
    max_log_error: float


@dataclass(frozen=True)
class TranDevice:
    """One alert as a Tran device; windows are [start, stop) in blocks, empty when stop == start."""

    alert_id: str
    candidate: int
    ground: bool
    source: str
    gateway: str | None
    size_bits: int
    uplink: tuple[int, int]
    downlink: tuple[int, int]
    ground_efficiency: float

    @property
    def active(self) -> bool:
        uplink = self.uplink[1] > self.uplink[0]
        return uplink if self.ground else uplink and self.downlink[1] > self.downlink[0]


@dataclass(frozen=True, eq=False)
class TranInstance:
    """Devices, radio points and per-entry index arrays of the half-duplex program.

    An uplink entry is one (active device, block) pair of the device's uplink window; `uplink_point` indexes
    `points` for UAV devices and is -1 for ground devices. Downlink entries exist for active UAV devices only.
    """

    scenario: Scenario
    devices: tuple[TranDevice, ...]
    channel: EffectiveChannel
    block_slots: int
    num_blocks: int
    points: tuple[str, ...]
    point_snr: np.ndarray
    uplink_device: np.ndarray
    uplink_block: np.ndarray
    uplink_point: np.ndarray
    downlink_device: np.ndarray
    downlink_block: np.ndarray
    downlink_point: np.ndarray


def expected_uav_gain(scenario: Scenario, horizontal_m: np.ndarray) -> np.ndarray:
    channel = scenario.uav_channel
    altitude = scenario.uav.altitude_m
    distance = np.hypot(horizontal_m, altitude)
    probability = los_probability(horizontal_m, altitude, channel.plos_a, channel.plos_b)
    los = distance ** (-channel.alpha_los)
    nlos = channel.nlos_attenuation * distance ** (-channel.alpha_nlos)
    return db_to_linear(channel.beta0_db) * (probability * los + (1.0 - probability) * nlos)


def ground_point_diameter(scenario: Scenario) -> float:
    points = np.array([node.xy for node in scenario.nodes] + [source.xy for source in scenario.sources])
    return float(max(np.linalg.norm(points - point, axis=1).max() for point in points))


def fit_effective_channel(scenario: Scenario) -> EffectiveChannel:
    """Least squares of ln(expected gain) on ln(distance) over horizontal distances 0..diameter in 10 m steps."""
    horizontal = np.arange(0.0, ground_point_diameter(scenario) + FIT_STEP_M, FIT_STEP_M)
    log_distance = np.log(np.hypot(horizontal, scenario.uav.altitude_m))
    log_gain = np.log(expected_uav_gain(scenario, horizontal))
    slope, intercept = np.polyfit(log_distance, log_gain, 1)
    residual = log_gain - (intercept + slope * log_distance)
    total = float(np.sum((log_gain - log_gain.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else 1.0
    return EffectiveChannel(float(np.exp(intercept)), float(-slope), r_squared, float(np.abs(residual).max()))


def backhaul_slots(scenario: Scenario, path: CandidatePath, size_bits: int) -> int:
    """Slots the backhaul route needs, with the per-link delay of models.paths (size/capacity + one slot)."""
    delay_s = sum(
        size_bits / scenario.backhaul_capacity_bps(link[1], link[2]) + scenario.time.slot_s
        for link in path.links
        if link[0] == BACKHAUL
    )
    return math.ceil(delay_s / scenario.time.slot_s - ROUNDING_TOLERANCE)


def _blocks(start_slot: int, stop_slot: int, block_slots: int) -> tuple[int, int]:
    start = -(-start_slot // block_slots)
    stop = stop_slot // block_slots
    return (start, max(start, stop))


def build_device(
    scenario: Scenario, candidates: tuple[CandidatePath, ...], alert_index: int, params: TranParams
) -> TranDevice:
    alert = scenario.alerts[alert_index]
    candidate = min(range(len(candidates)), key=lambda index: (candidates[index].cost_s, index))
    path = candidates[candidate]
    end = alert.deadline_slot - backhaul_slots(scenario, path, alert.size_bits)
    G = params.block_slots
    if path.kind == GROUND:
        distance = math.dist(scenario.source_by_id[alert.source].xy, scenario.node_by_id[path.entry].xy)
        efficiency = math.log2(1.0 + ground_snr_per_hz(scenario, distance) / scenario.spectrum.b_tot_hz)
        return TranDevice(
            alert.id, candidate, True, alert.source, None, alert.size_bits,
            _blocks(alert.release_slot, end, G), (0, 0), efficiency,
        )
    length = end - alert.release_slot
    split = alert.release_slot + (math.ceil(params.theta * length - ROUNDING_TOLERANCE) if length >= 1 else 0)
    return TranDevice(
        alert.id, candidate, False, alert.source, path.entry, alert.size_bits,
        _blocks(alert.release_slot, split, G), _blocks(split, end, G), 0.0,
    )


def build_instance(
    scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]], params: TranParams
) -> TranInstance:
    num_slots = scenario.time.num_slots
    if num_slots % params.block_slots:
        raise ValueError(f"block_slots {params.block_slots} must divide num_slots {num_slots}")
    devices = tuple(
        build_device(scenario, candidates[alert.id], index, params) for index, alert in enumerate(scenario.alerts)
    )
    channel = fit_effective_channel(scenario)
    noise_w = noise_psd_w_per_hz(scenario) * scenario.spectrum.b_tot_hz
    active_uav = [device for device in devices if device.active and not device.ground]
    sources = sorted({device.source for device in active_uav})
    gateways = sorted({device.gateway for device in active_uav})
    points = tuple(sources + gateways)
    point_snr = np.array(
        [scenario.source_power_w * channel.beta / noise_w] * len(sources)
        + [scenario.uav.downlink_power_w * channel.beta / noise_w] * len(gateways)
    )
    point_index = {point: index for index, point in enumerate(points)}
    uplink, downlink = [], []
    for index, device in enumerate(devices):
        if not device.active:
            continue
        uplink_point = -1 if device.ground else point_index[device.source]
        uplink.extend((index, block, uplink_point) for block in range(*device.uplink))
        if not device.ground:
            downlink.extend((index, block, point_index[device.gateway]) for block in range(*device.downlink))
    up = np.array(uplink, dtype=int).reshape(-1, 3)
    down = np.array(downlink, dtype=int).reshape(-1, 3)
    return TranInstance(
        scenario, devices, channel, params.block_slots, num_slots // params.block_slots, points, point_snr,
        up[:, 0], up[:, 1], up[:, 2], down[:, 0], down[:, 1], down[:, 2],
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/literature/test_tran_model.py`

Expected: `4 passed`.

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `213 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/support/tran_builders.py tests/literature/test_tran_model.py src/literature/__init__.py src/literature/tran_model.py
git commit -m "feat: build Tran half-duplex instances from scenarios"
```

### Task 2: Inner-approximation solver

**Files:**
- Create: `src/literature/tran_ia.py`
- Modify: `pyproject.toml`, `uv.lock`
- Test: `tests/literature/test_tran_ia.py`

**Interfaces:**
- Consumes: Task 1; `baselines.trajectories.zone_tour_trajectory`.
- Produces: `literature.tran_ia`: `SPEED_MARGIN = 1e-6`, `CLARABEL_SETTINGS`, `CONVERGED = "converged"`, `MAX_ITERATIONS = "max_iterations"`; `Iterate(trajectory, uplink, downlink, lam)` (block-boundary trajectory in metres, share per uplink and downlink entry); `IaResult(iterate, objectives, status, iterations)`; `relaxed_objective(lam, mu) -> float`; `initial_iterate(instance) -> Iterate`; `efficiency_tangent(snr_scaled, z0) -> (value, slope)`; `slacks(instance, trajectory) -> ndarray`; `solve_linearized(instance, params, current) -> (status, Iterate | None)`; `run_inner_approximation(instance, initial, params) -> IaResult` with status `converged`, `max_iterations` or `solver_<cvxpy status>`.

The program is built with numeric constants in every iteration; do not replace them with cvxpy parameters: a parametrized prototype grew to 17 GB during canonicalization. The power constraint uses cvxpy's second-order-cone approximation and Clarabel runs with `CLARABEL_SETTINGS` (spec Section 17, Plan D.1 adjustments).

- [ ] **Step 1: Write the failing tests**

Create `tests/literature/test_tran_ia.py`:

```python
import numpy as np
import pytest

from baselines.trajectories import zone_tour_trajectory
import literature.tran_ia as tran_ia
from literature.tran_ia import (
    CONVERGED, MAX_ITERATIONS, Iterate, efficiency_tangent, initial_iterate, product_bound, relaxed_objective,
    run_inner_approximation,
)
from literature.tran_model import TranParams, build_instance
from models.paths import candidate_paths
from models.scenario import scenario_from_dict
from tran_builders import tran_raw


def _instance(**raw_changes):
    scenario = scenario_from_dict(tran_raw(**raw_changes))
    return build_instance(scenario, candidate_paths(scenario), TranParams())


def test_initial_iterate_is_the_zone_tour_with_equal_shares_and_zero_lambda():
    instance = _instance()
    initial = initial_iterate(instance)
    assert np.array_equal(initial.trajectory, zone_tour_trajectory(instance.scenario))
    totals = np.bincount(instance.uplink_block, weights=initial.uplink, minlength=instance.num_blocks)
    assert np.allclose(totals[np.bincount(instance.uplink_block, minlength=instance.num_blocks) > 0], 1.0)
    assert initial.lam.tolist() == [0.0, 0.0, 0.0] and relaxed_objective(initial.lam, 1.0) == 0.0


def test_efficiency_tangent_is_a_tight_global_lower_bound():
    snr, z0 = np.array([4.7]), np.array([3.0])
    value, slope = efficiency_tangent(snr, z0)
    z = np.linspace(1.0, 50.0, 200)
    assert value[0] == pytest.approx(np.log2(1.0 + 4.7 / 3.0))
    assert np.all(np.log2(1.0 + 4.7 / z) >= value[0] + slope[0] * (z - 3.0) - 1e-12)


def test_product_bound_is_tight_at_the_anchor_and_below_the_product():
    share, efficiency = np.array([0.3, 0.7]), np.array([2.0, 0.1])
    assert product_bound(share + efficiency, share, efficiency).value == pytest.approx(share * efficiency)
    rng = np.random.default_rng(3)
    shares, efficiencies = rng.uniform(0.0, 1.0, 50), rng.uniform(0.0, 3.0, 50)
    assert np.all(product_bound(np.full(50, 1.2), shares, efficiencies).value <= shares * efficiencies + 1e-12)


def test_inner_approximation_is_monotone_bounded_and_deterministic():
    instance = _instance(uav_size=12000000)
    params = TranParams(max_iterations=6)
    first = run_inner_approximation(instance, initial_iterate(instance), params)
    second = run_inner_approximation(instance, initial_iterate(instance), params)
    assert first.status in (CONVERGED, MAX_ITERATIONS) and first.iterations >= 1
    assert np.all(np.diff(first.objectives) >= -1e-5)
    assert np.all((first.iterate.lam >= 0.0) & (first.iterate.lam <= 1.0))
    assert np.array_equal(first.iterate.trajectory, second.iterate.trajectory)
    assert np.array_equal(first.iterate.uplink, second.iterate.uplink) and first.objectives == second.objectives


def _scripted(monkeypatch, steps):
    calls = iter(steps)
    monkeypatch.setattr(tran_ia, "solve_linearized", lambda instance, params, current: next(calls)(current))


def test_solver_failure_keeps_the_last_accepted_iterate(monkeypatch):
    instance = _instance()
    initial = initial_iterate(instance)
    accepted = Iterate(initial.trajectory, initial.uplink, initial.downlink, np.array([1.0, 0.0, 0.0]))
    _scripted(monkeypatch, [lambda current: ("optimal", accepted), lambda current: ("infeasible", None)])
    result = run_inner_approximation(instance, initial, TranParams())
    assert (result.status, result.iterations, result.objectives) == ("solver_infeasible", 1, (0.0, 1.0))
    assert result.iterate is accepted


def test_stops_on_small_change_or_at_the_iteration_cap(monkeypatch):
    instance = _instance()
    initial = initial_iterate(instance)
    _scripted(monkeypatch, [lambda current: ("optimal", current)])
    assert run_inner_approximation(instance, initial, TranParams()).status == CONVERGED
    growing = [lambda current, k=k: ("optimal", Iterate(current.trajectory, current.uplink, current.downlink, np.array([k / 3.0, 0.0, 0.0])))
               for k in (1, 2, 3)]
    _scripted(monkeypatch, growing)
    result = run_inner_approximation(instance, initial, TranParams(max_iterations=3, mu=0.001))
    assert (result.status, result.iterations, len(result.objectives)) == (MAX_ITERATIONS, 3, 4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/literature/test_tran_ia.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'literature.tran_ia'` (`1 error`).

- [ ] **Step 3: Implement**

Run: `uv add 'cvxpy>=1.6,<2'`

Expected: `pyproject.toml` gains `"cvxpy>=1.6,<2",` as the first entry of `[project].dependencies`, and `uv.lock` gains the resolved packages, among them cvxpy 1.9.2 and clarabel 0.11.1 (the versions resolved when this plan was written; a later resolution within the bound is acceptable).

Create `src/literature/tran_ia.py`:

```python
"""Inner approximation of Tran et al. (2022), Algorithm 3, on a TranInstance (spec D Sections 5.2–5.3).

Each iteration builds the convex program (58) with the linearization point as numeric constants (cvxpy parameters
of this size exhaust memory during canonicalization). Lengths are scaled by the UAV altitude and rates by B_tot.
The power constraint uses cvxpy's second-order-cone approximation and Clarabel runs with the tolerances below:
with exact power cones or default tolerances Clarabel stalls on development scenarios at slot resolution.
"""

from dataclasses import dataclass
import math
import warnings

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix

from baselines.trajectories import zone_tour_trajectory

from .tran_model import TranInstance, TranParams

SPEED_MARGIN = 1e-6
CLARABEL_SETTINGS = {"tol_gap_abs": 1e-6, "tol_gap_rel": 1e-6, "tol_feas": 1e-7}
CONVERGED = "converged"
MAX_ITERATIONS = "max_iterations"


@dataclass(frozen=True, eq=False)
class Iterate:
    """Trajectory at block boundaries (metres), bandwidth fraction per uplink and downlink entry, and lambda."""

    trajectory: np.ndarray
    uplink: np.ndarray
    downlink: np.ndarray
    lam: np.ndarray


@dataclass(frozen=True, eq=False)
class IaResult:
    iterate: Iterate
    objectives: tuple[float, ...]
    status: str
    iterations: int


def relaxed_objective(lam: np.ndarray, mu: float) -> float:
    """Tran eq. (26a): sum(lambda) + mu * sum(lambda * (lambda - 1))."""
    return float(np.sum(lam) + mu * np.sum(lam * (lam - 1.0)))


def _equal_split(blocks: np.ndarray, num_blocks: int) -> np.ndarray:
    counts = np.bincount(blocks, minlength=num_blocks)
    return 1.0 / counts[blocks] if len(blocks) else np.zeros(0)


def initial_iterate(instance: TranInstance) -> Iterate:
    """B1 zone tour at block boundaries, equal bandwidth split among entries of each block, lambda = 0."""
    trajectory = zone_tour_trajectory(instance.scenario)[:: instance.block_slots]
    return Iterate(
        np.asarray(trajectory, dtype=float),
        _equal_split(instance.uplink_block, instance.num_blocks),
        _equal_split(instance.downlink_block, instance.num_blocks),
        np.zeros(len(instance.devices)),
    )


def efficiency_tangent(snr_scaled: np.ndarray, z0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Value and slope of the convex efficiency log2(1 + c/z) at z0; the tangent is a global lower bound."""
    efficiency0 = np.log2(1.0 + snr_scaled / z0)
    slope = -snr_scaled / (math.log(2.0) * z0 * (z0 + snr_scaled))
    return efficiency0, slope


def product_bound(anchor, share, efficiency):
    """Concave lower bound of share * efficiency, tight where share + efficiency equals anchor (Tran eq. 57)."""
    return 0.25 * (2.0 * cp.multiply(anchor, share + efficiency) - anchor**2) - 0.25 * cp.square(share - efficiency)


def slacks(instance: TranInstance, trajectory: np.ndarray) -> np.ndarray:
    """Tight z = (1 + |m - w|^2)^(alpha/2) per point and block, lengths in altitude units."""
    unit = instance.scenario.uav.altitude_m
    midpoints = 0.5 * (trajectory[:-1] + trajectory[1:]) / unit
    xy = np.array([instance.scenario.position(point) for point in instance.points]).reshape(-1, 2) / unit
    squared = np.sum((midpoints[None, :, :] - xy[:, None, :]) ** 2, axis=2)
    return (1.0 + squared) ** (instance.channel.alpha / 2.0)


def _incidence(rows: np.ndarray, count: int) -> csr_matrix:
    return csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(count, len(rows)))


def _active_upper(instance: TranInstance) -> np.ndarray:
    return np.array([1.0 if device.active else 0.0 for device in instance.devices])


def solve_linearized(instance: TranInstance, params: TranParams, current: Iterate) -> tuple[str, Iterate | None]:
    """Build and solve program (58) linearized at `current`; return the solver status and the next iterate."""
    scenario = instance.scenario
    unit = scenario.uav.altitude_m
    G, P = instance.block_slots, instance.num_blocks
    K, point_count = len(instance.devices), len(instance.points)
    bandwidth = scenario.spectrum.b_tot_hz
    up_device, up_block, up_point = instance.uplink_device, instance.uplink_block, instance.uplink_point
    down_device, down_block, down_point = instance.downlink_device, instance.downlink_block, instance.downlink_point
    upper = _active_upper(instance)

    q = cp.Variable((P + 1, 2))
    a1 = cp.Variable(len(up_device), nonneg=True)
    a2 = cp.Variable(len(down_device), nonneg=True)
    r1 = cp.Variable(len(up_device))
    r2 = cp.Variable(len(down_device))
    lam = cp.Variable(K, nonneg=True)
    step = G * scenario.uav.v_max_mps * scenario.time.slot_s * (1.0 - SPEED_MARGIN) / unit
    constraints = [
        q[0] == np.asarray(scenario.uav.start_xy_m) / unit,
        q[P] == np.asarray(scenario.uav.end_xy_m) / unit,
        cp.norm(q[1:] - q[:-1], 2, axis=1) <= step,
        lam <= upper,
    ]
    delta_up = scenario.time.access_fraction * scenario.time.slot_s * G
    delta_down = (1.0 - scenario.time.access_fraction) * scenario.time.slot_s * G
    sizes = np.array([device.size_bits for device in instance.devices], dtype=float)
    ground = np.array([device.ground for device in instance.devices])
    up_need = upper * sizes / (delta_up * bandwidth)
    down_need = upper * np.where(ground, 0.0, sizes / (delta_down * bandwidth))

    if point_count:
        z = cp.Variable((point_count, P), nonneg=True)
        phi = cp.Variable((point_count, P))
        midpoints = 0.5 * (q[:-1] + q[1:])
        xy = np.array([scenario.position(point) for point in instance.points]) / unit
        for index in range(point_count):
            constraints.append(
                1.0 + cp.sum(cp.square(midpoints - xy[index]), axis=1)
                <= cp.power(z[index], 2.0 / instance.channel.alpha)
            )
        z0 = slacks(instance, current.trajectory)
        snr = (instance.point_snr / unit**instance.channel.alpha)[:, None]
        efficiency0, slope = efficiency_tangent(snr, z0)
        constraints.append(phi <= efficiency0 - slope * z0 + cp.multiply(slope, z))
        phi_flat = cp.reshape(phi, (point_count * P,), order="C")
        for a, r, blocks, point_of, selected, shares in (
            (a1, r1, up_block, up_point, np.flatnonzero(up_point >= 0), current.uplink),
            (a2, r2, down_block, down_point, np.arange(len(down_device)), current.downlink),
        ):
            if not len(selected):
                continue
            anchor = shares[selected] + efficiency0[point_of[selected], blocks[selected]]
            efficiency = phi_flat[point_of[selected] * P + blocks[selected]]
            share = a[selected]
            constraints.append(r[selected] <= product_bound(anchor, share, efficiency))
    ground_up = np.flatnonzero(up_point < 0)
    if len(ground_up):
        efficiency = np.array([instance.devices[device].ground_efficiency for device in up_device[ground_up]])
        constraints.append(r1[ground_up] <= cp.multiply(efficiency, a1[ground_up]))
    if len(up_device):
        constraints += [
            _incidence(up_device, K) @ r1 >= cp.multiply(up_need, lam),
            _incidence(up_block, P) @ a1 <= 1.0,
        ]
    if len(down_device):
        constraints += [
            _incidence(down_device, K) @ r2 >= cp.multiply(down_need, lam),
            _incidence(down_block, P) @ a2 <= 1.0,
        ]
    problem = cp.Problem(cp.Maximize(cp.sum(lam) + params.mu * 2.0 * (current.lam @ lam)), constraints)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Power atom", category=UserWarning)
        problem.solve(solver=cp.CLARABEL, ignore_dpp=True, canon_backend=cp.SCIPY_CANON_BACKEND, **CLARABEL_SETTINGS)
    if problem.status != cp.OPTIMAL:
        return problem.status, None
    trajectory = np.asarray(q.value, dtype=float) * unit
    trajectory[0] = scenario.uav.start_xy_m
    trajectory[-1] = scenario.uav.end_xy_m
    shares = []
    for variable, blocks in ((a1, up_block), (a2, down_block)):
        if not len(blocks):
            shares.append(np.zeros(0))
            continue
        values = np.clip(np.asarray(variable.value, dtype=float), 0.0, None)
        totals = np.bincount(blocks, weights=values, minlength=P)
        shares.append(values / np.maximum(totals, 1.0)[blocks])
    lam_value = np.clip(np.asarray(lam.value, dtype=float), 0.0, upper)
    return problem.status, Iterate(trajectory, shares[0], shares[1], lam_value)


def run_inner_approximation(instance: TranInstance, initial: Iterate, params: TranParams) -> IaResult:
    """Repeat: linearize at the accepted iterate, solve (58), accept; stop on small change, the cap, or a solver failure."""
    current = initial
    objectives = [relaxed_objective(initial.lam, params.mu)]
    for iteration in range(1, params.max_iterations + 1):
        status, following = solve_linearized(instance, params, current)
        if following is None:
            return IaResult(current, tuple(objectives), f"solver_{status}", iteration - 1)
        current = following
        objectives.append(relaxed_objective(current.lam, params.mu))
        if abs(objectives[-1] - objectives[-2]) <= params.tolerance * max(1.0, abs(objectives[-2])):
            return IaResult(current, tuple(objectives), CONVERGED, iteration)
    return IaResult(current, tuple(objectives), MAX_ITERATIONS, params.max_iterations)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/literature/test_tran_ia.py`

Expected: `6 passed`.

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `219 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/literature/test_tran_ia.py src/literature/tran_ia.py pyproject.toml uv.lock
git commit -m "feat: add the Tran inner-approximation solver"
```

### Task 3: Plan conversion and the TRAN method

**Files:**
- Create: `src/literature/tran_plan.py`, `src/literature/tran.py`
- Test: `tests/literature/test_tran_plan.py`

**Interfaces:**
- Consumes: Tasks 1–2; `models.plan` (`Plan`, `StaticSchedule`, `validate_plan`), `models.evaluate.EvaluationCounter`.
- Produces: `literature.tran_plan`: `expand_trajectory(block_trajectory, block_slots) -> ndarray`, `keep_top_downlinks(bandwidth, links, limit) -> None` (in place), `plan_from_iterate(instance, candidates, iterate) -> Plan` (raises `ValueError` for an invalid plan). `literature.tran`: `TranOutcome(plan, initial_plan, channel, objectives, status, iterations, penalty)`; frozen `TranHD(name="TRAN", params=TranParams(), uses_uav=True)` with `plan(scenario, candidates, counter) -> Plan`, `solve(scenario, candidates) -> TranOutcome`, and `trajectory(scenario)` raising `NotImplementedError`.

- [ ] **Step 1: Write the failing tests**

Create `tests/literature/test_tran_plan.py`:

```python
import numpy as np
import pytest

from literature.tran import TranHD
from literature.tran_ia import initial_iterate
from literature.tran_model import TranParams, build_instance
from literature.tran_plan import expand_trajectory, keep_top_downlinks, plan_from_iterate
from models.channel import ACCESS
from models.evaluate import EvaluationCounter
from models.paths import candidate_paths
from models.plan import validate_plan
from models.scenario import scenario_from_dict
from tran_builders import tran_raw


def test_expand_trajectory_interpolates_inside_blocks():
    expanded = expand_trajectory(np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]), 5)
    assert expanded.shape == (11, 2)
    assert expanded[:6, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] and expanded[-1].tolist() == [5.0, 5.0]


def test_keep_top_downlinks_breaks_ties_by_link_order():
    links = [("down", "UAV", "a"), ("down", "UAV", "b"), ("down", "UAV", "c")]
    bandwidth = dict(zip(links, (np.array([3.0, 1.0]), np.array([3.0, 2.0]), np.array([1.0, 5.0]))))
    keep_top_downlinks(bandwidth, links, 2)
    assert [bandwidth[link].tolist() for link in links] == [[3.0, 0.0], [3.0, 2.0], [0.0, 5.0]]


@pytest.mark.parametrize("block_slots", [1, 5])
def test_initial_iterate_converts_to_a_valid_static_plan(block_slots):
    scenario = scenario_from_dict(tran_raw())
    candidates = candidate_paths(scenario)
    instance = build_instance(scenario, candidates, TranParams(block_slots=block_slots))
    plan = plan_from_iterate(instance, candidates, initial_iterate(instance))
    assert validate_plan(scenario, candidates, plan) == []
    assert plan.path_choice == {"alert-0000": 0, "alert-0001": 0, "alert-0002": 0}
    access = sum(values for link, values in plan.bandwidth.bandwidth_hz.items() if link[0] == ACCESS)
    assert np.all(access <= 1e6 * (1.0 + 1e-9))


def test_tran_method_plans_validly_and_has_no_fixed_trajectory():
    scenario = scenario_from_dict(tran_raw(uav_size=12000000))
    candidates = candidate_paths(scenario)
    method = TranHD(params=TranParams(max_iterations=3))
    outcome = method.solve(scenario, candidates)
    assert validate_plan(scenario, candidates, outcome.plan) == [] and validate_plan(scenario, candidates, outcome.initial_plan) == []
    assert outcome.penalty <= 0.0 and len(outcome.objectives) == outcome.iterations + 1
    assert np.array_equal(method.plan(scenario, candidates, EvaluationCounter()).trajectory, outcome.plan.trajectory)
    with pytest.raises(NotImplementedError):
        method.trajectory(scenario)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/literature/test_tran_plan.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'literature.tran'` (`1 error`).

- [ ] **Step 3: Implement**

Create `src/literature/tran_plan.py`:

```python
"""Conversion of a Tran iterate into a project Plan with a static schedule (spec D Section 5.4)."""

from typing import Mapping

import numpy as np

from models.channel import LinkId
from models.paths import CandidatePath
from models.plan import Plan, StaticSchedule, validate_plan

from .tran_ia import Iterate
from .tran_model import TranInstance


def expand_trajectory(block_trajectory: np.ndarray, block_slots: int) -> np.ndarray:
    """Slot-boundary trajectory interpolated linearly inside each block."""
    starts, stops = block_trajectory[:-1], block_trajectory[1:]
    fractions = np.arange(block_slots) / block_slots
    inner = starts[:, None, :] + (stops - starts)[:, None, :] * fractions[None, :, None]
    return np.vstack([inner.reshape(-1, 2), block_trajectory[-1:]])


def keep_top_downlinks(bandwidth: dict[LinkId, np.ndarray], links: list[LinkId], limit: int) -> None:
    """Per slot keep the `limit` downlinks with the most bandwidth (ties: link order); zero the others in place."""
    if not links:
        return
    matrix = np.vstack([bandwidth[link] for link in links])
    order = np.argsort(-matrix, axis=0, kind="stable")
    keep = np.zeros(matrix.shape, dtype=bool)
    np.put_along_axis(keep, order[:limit], True, axis=0)
    for index, link in enumerate(links):
        bandwidth[link] = np.where(keep[index], matrix[index], 0.0)


def plan_from_iterate(
    instance: TranInstance, candidates: Mapping[str, tuple[CandidatePath, ...]], iterate: Iterate
) -> Plan:
    scenario = instance.scenario
    G = instance.block_slots
    total = scenario.spectrum.b_tot_hz
    bandwidth: dict[LinkId, np.ndarray] = {}
    downlinks: set[LinkId] = set()
    for devices, blocks, shares, hop in (
        (instance.uplink_device, instance.uplink_block, iterate.uplink, 0),
        (instance.downlink_device, instance.downlink_block, iterate.downlink, 1),
    ):
        for device_index, block, share in zip(devices, blocks, shares):
            device = instance.devices[device_index]
            link = candidates[device.alert_id][device.candidate].links[hop]
            values = bandwidth.setdefault(link, np.zeros(scenario.time.num_slots))
            values[block * G : (block + 1) * G] += share * total
            if hop == 1:
                downlinks.add(link)
    keep_top_downlinks(bandwidth, sorted(downlinks), scenario.uav.max_active_downlinks)
    path_choice = {device.alert_id: device.candidate for device in instance.devices}
    plan = Plan(expand_trajectory(iterate.trajectory, G), path_choice, StaticSchedule(bandwidth))
    violations = validate_plan(scenario, candidates, plan)
    if violations:
        raise ValueError(f"TRAN produced an invalid plan: {violations[:3]}")
    return plan
```

Create `src/literature/tran.py`:

```python
"""Planning method TRAN: Tran et al. (2022) half-duplex design adapted to the project model (spec D Section 5)."""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from models.evaluate import EvaluationCounter
from models.paths import CandidatePath
from models.plan import Plan
from models.scenario import Scenario

from .tran_ia import initial_iterate, run_inner_approximation
from .tran_model import EffectiveChannel, TranParams, build_instance
from .tran_plan import plan_from_iterate


@dataclass(frozen=True, eq=False)
class TranOutcome:
    plan: Plan
    initial_plan: Plan
    channel: EffectiveChannel
    objectives: tuple[float, ...]
    status: str
    iterations: int
    penalty: float


@dataclass(frozen=True)
class TranHD:
    name: str = "TRAN"
    params: TranParams = field(default_factory=TranParams)
    uses_uav: bool = True

    def trajectory(self, scenario: Scenario) -> np.ndarray:
        raise NotImplementedError(f"{self.name}: the trajectory of TRAN exists only after planning")

    def plan(
        self,
        scenario: Scenario,
        candidates: Mapping[str, tuple[CandidatePath, ...]],
        counter: EvaluationCounter,
    ) -> Plan:
        return self.solve(scenario, candidates).plan

    def solve(self, scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]]) -> TranOutcome:
        instance = build_instance(scenario, candidates, self.params)
        initial = initial_iterate(instance)
        result = run_inner_approximation(instance, initial, self.params)
        lam = result.iterate.lam
        return TranOutcome(
            plan=plan_from_iterate(instance, candidates, result.iterate),
            initial_plan=plan_from_iterate(instance, candidates, initial),
            channel=instance.channel,
            objectives=result.objectives,
            status=result.status,
            iterations=result.iterations,
            penalty=float(np.sum(lam * (lam - 1.0))),
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/literature/test_tran_plan.py`

Expected: `5 passed`.

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `224 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/literature/test_tran_plan.py src/literature/tran_plan.py src/literature/tran.py
git commit -m "feat: add the TRAN planning method"
```

### Task 4: Runner support for TRAN and --output-dir

**Files:**
- Modify: `src/runner/config.py`, `src/runner/methods.py`, `src/runner/experiment.py`, `src/runner/provenance.py`, `experiments/run_phase1.py`, `experiments/run_phase2.py`
- Test: `tests/runner/test_runner_config.py`, `tests/runner/test_runner_provenance.py`, `tests/runner/test_runner_experiment.py`, `tests/runner/test_runner_tran_tasks.py`, `tests/runner/test_runner_output_dir.py`

**Interfaces:**
- Consumes: Task 3; runner modules of plans C.1–C.2; `tests/runner/experiment_fixtures.write_tiny_experiment`.
- Produces: `runner.config.TRAN_BASE = "TRAN"`, `PLANNED_TRAJECTORY_BASES = ("search", "TRAN")`, `MethodSpec.tran_params() -> TranParams`; `runner.methods.build_method` returns `TranHD(spec.id, spec.tran_params())` for base `TRAN` and `bound_owner` gives `("trajectory", spec.id)` or `None` for it; `build_tasks` creates no bound tasks for TRAN; `runner.provenance.SOURCE_PACKAGES` includes `literature` and `environment()` adds `cvxpy` and `clarabel`; `experiments/run_phase1.py` and `run_phase2.py` accept `--output-dir PATH`.

- [ ] **Step 1: Write the failing tests**

In `tests/runner/test_runner_config.py` (2 edits, then an append):

Edit 1: replace

```python
from optimization.bcd import SearchParams
```

with

```python
from literature.tran import TranHD
from literature.tran_model import TranParams
from optimization.bcd import SearchParams
```

Edit 2: replace

```python
        (lambda raw: raw.update(methods=[{"id": "P", "base": "search", "bounds": "yes"}]), "bounds"),
```

with

```python
        (lambda raw: raw.update(methods=[{"id": "P", "base": "search", "bounds": "yes"}]), "bounds"),
        (lambda raw: raw.update(methods=[{"id": "TRAN", "base": "TRAN", "params": {"alpha": 2}}]), "unknown TRAN parameters"),
        (lambda raw: raw.update(methods=[{"id": "TRAN", "base": "TRAN", "params": {"theta": 1.5}}]), "theta must lie"),
```

Append to the end of the file:

```python
def test_tran_specs_parse_build_and_bound_inside_their_tasks(tmp_path: Path):
    methods = ["B1", {"id": "TRAN", "base": "TRAN", "params": {"theta": 0.5, "mu": 10, "block_slots": 1}, "bounds": True}]
    config = load_experiment_config(_write(tmp_path, lambda raw: raw.update(methods=methods)))
    spec = config.methods[1]
    assert spec == MethodSpec("TRAN", "TRAN", (("block_slots", 1), ("mu", 10), ("theta", 0.5)), True)
    assert spec.tran_params() == TranParams(theta=0.5, mu=10.0, block_slots=1)
    method = build_method(spec)
    assert isinstance(method, TranHD) and method.name == "TRAN" and method.params == spec.tran_params()
    assert bound_owner(spec) == ("trajectory", "TRAN")
```


In `tests/runner/test_runner_provenance.py` (2 edits, then an append):

Edit 1: replace

```python
from runner.provenance import environment, git_state, source_hash
```

with

```python
from runner.provenance import SOURCE_PACKAGES, environment, git_state, source_hash
```

Edit 2: replace

```python
    assert set(environment()) == {"python", "numpy", "scipy", "highs", "platform", "cpu_count"}
```

with

```python
    assert set(environment()) == {"python", "numpy", "scipy", "highs", "cvxpy", "clarabel", "platform", "cpu_count"}
```

Append to the end of the file:

```python
def test_source_hash_covers_the_literature_package():
    assert "literature" in SOURCE_PACKAGES
```


In `tests/runner/test_runner_experiment.py` (1 edit):

Edit 1: replace

```python
{"python", "numpy", "scipy", "highs", "platform", "cpu_count"}
```

with

```python
{"python", "numpy", "scipy", "highs", "cvxpy", "clarabel", "platform", "cpu_count"}
```


Create `tests/runner/test_runner_tran_tasks.py`:

```python
import csv
import json
from pathlib import Path

from experiment_fixtures import write_tiny_experiment
from runner.config import load_experiment_config
from runner.experiment import build_tasks, run_experiment, scenario_refs
from runner.tasks import BoundTask, MethodTask

TRAN_SPEC = {"id": "TRAN", "base": "TRAN", "params": {"max_iterations": 3}, "bounds": True}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_tran_gets_no_bound_tasks(tmp_path: Path):
    config = load_experiment_config(write_tiny_experiment(tmp_path, tmp_path / "out", 1, ("B1", TRAN_SPEC), crosscheck=False))
    tasks = build_tasks(config, scenario_refs(config))
    assert {(task.variant, task.trajectory_method) for task in tasks if isinstance(task, BoundTask)} == {("trajectory", "B1")}
    assert sum(isinstance(task, MethodTask) and task.spec.id == "TRAN" for task in tasks) == 2


def test_tran_runs_in_the_experiment_runner_with_in_task_bounds(tmp_path: Path):
    output = tmp_path / "out"
    config_path = write_tiny_experiment(tmp_path, output, 1, (TRAN_SPEC,), crosscheck=False)
    assert run_experiment(load_experiment_config(config_path), config_path) == 0
    rows, bounds = _rows(output / "method_realizations.csv"), _rows(output / "bounds.csv")
    assert {row["method"] for row in rows} == {"TRAN"} and len(rows) == 4
    assert {row["trajectory_method"] for row in bounds} == {"TRAN"} and len(bounds) == 4
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "complete"
```

Create `tests/runner/test_runner_output_dir.py`:

```python
import importlib.util
from pathlib import Path

import pytest

from experiment_fixtures import write_tiny_experiment

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["run_phase1", "run_phase2"])
def test_output_dir_overrides_the_configured_directory(tmp_path: Path, name: str):
    configured = tmp_path / "configured"
    config_path = write_tiny_experiment(tmp_path, configured, 1, ("B0",), crosscheck=False)
    override = tmp_path / "override"
    assert _script(name).main(["--config", str(config_path), "--output-dir", str(override), "--workers", "1"]) == 0
    assert (override / "summary.csv").exists() and not configured.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/runner`

Expected: `10 failed, 36 passed`; the failing tests are:

- `tests/runner/test_runner_config.py::test_invalid_configs_are_rejected[<lambda>-theta must lie]`
- `tests/runner/test_runner_config.py::test_invalid_configs_are_rejected[<lambda>-unknown TRAN parameters]`
- `tests/runner/test_runner_config.py::test_tran_specs_parse_build_and_bound_inside_their_tasks`
- `tests/runner/test_runner_experiment.py::test_results_do_not_depend_on_worker_count`
- `tests/runner/test_runner_output_dir.py::test_output_dir_overrides_the_configured_directory[run_phase1]`
- `tests/runner/test_runner_output_dir.py::test_output_dir_overrides_the_configured_directory[run_phase2]`
- `tests/runner/test_runner_provenance.py::test_environment_and_git_state_outside_a_repository`
- `tests/runner/test_runner_provenance.py::test_source_hash_covers_the_literature_package`
- `tests/runner/test_runner_tran_tasks.py::test_tran_gets_no_bound_tasks`
- `tests/runner/test_runner_tran_tasks.py::test_tran_runs_in_the_experiment_runner_with_in_task_bounds`

- [ ] **Step 3: Implement**

In `src/runner/config.py` (5 edits):

Edit 1: replace

```python
from baselines import METHODS
from optimization.bcd import SearchParams

SEARCH_BASE = "search"
```

with

```python
from baselines import METHODS
from literature.tran_model import TranParams
from optimization.bcd import SearchParams

SEARCH_BASE = "search"
TRAN_BASE = "TRAN"
PLANNED_TRAJECTORY_BASES = (SEARCH_BASE, TRAN_BASE)
```

Edit 2: replace

```python
    """B0 or B1 by name, or a search variant with sorted parameter pairs and in-task trajectory bounds."""
```

with

```python
    """B0 or B1 by name, or a search or TRAN variant with sorted parameter pairs and in-task trajectory bounds."""
```

Edit 3: replace

```python
    def search_params(self) -> SearchParams:
        return SearchParams.from_mapping(dict(self.params))
```

with

```python
    def search_params(self) -> SearchParams:
        return SearchParams.from_mapping(dict(self.params))

    def tran_params(self) -> TranParams:
        return TranParams.from_mapping(dict(self.params))
```

Edit 4: replace

```python
    if base != SEARCH_BASE:
        raise ValueError(f"{label}: unknown base {base!r}; expected one of {[*sorted(METHODS), SEARCH_BASE]}")
```

with

```python
    if base not in PLANNED_TRAJECTORY_BASES:
        raise ValueError(f"{label}: unknown base {base!r}; expected one of {[*sorted(METHODS), *PLANNED_TRAJECTORY_BASES]}")
```

Edit 5: replace

```python
    try:
        spec.search_params()
    except (TypeError, ValueError) as error:
```

with

```python
    try:
        spec.search_params() if base == SEARCH_BASE else spec.tran_params()
    except (TypeError, ValueError) as error:
```


In `src/runner/methods.py` (3 edits):

Edit 1: replace

```python
from baselines.method import Method
from optimization.bcd import SearchMethod

from .config import SEARCH_BASE, MethodSpec
```

with

```python
from baselines.method import Method
from literature.tran import TranHD
from optimization.bcd import SearchMethod

from .config import PLANNED_TRAJECTORY_BASES, SEARCH_BASE, TRAN_BASE, MethodSpec
```

Edit 2: replace

```python
    if spec.base == SEARCH_BASE:
        return SearchMethod(spec.id, spec.search_params())
    return get_method(spec.base)
```

with

```python
    if spec.base == SEARCH_BASE:
        return SearchMethod(spec.id, spec.search_params())
    if spec.base == TRAN_BASE:
        return TranHD(spec.id, spec.tran_params())
    return get_method(spec.base)
```

Edit 3: replace

```python
    """(variant, trajectory_method) of the bound rows for the method; None for a search method without bounds."""
    if spec.base == SEARCH_BASE:
```

with

```python
    """(variant, trajectory_method) of the bound rows for the method; None for a planned-trajectory method without bounds."""
    if spec.base in PLANNED_TRAJECTORY_BASES:
```


In `src/runner/experiment.py` (2 edits):

Edit 1: replace

```python
from .config import SEARCH_BASE, ExperimentConfig
```

with

```python
from .config import PLANNED_TRAJECTORY_BASES, ExperimentConfig
```

Edit 2: replace

```python
    """Method tasks for every spec; bound tasks for B0/B1 trajectories (search methods bound inside their tasks)."""
    tasks: list[Task] = []
    variants = sorted({bound_owner(spec) for spec in config.methods if spec.base != SEARCH_BASE})
```

with

```python
    """Method tasks for every spec; bound tasks for B0/B1 trajectories (search and TRAN methods bound inside their tasks)."""
    tasks: list[Task] = []
    variants = sorted({bound_owner(spec) for spec in config.methods if spec.base not in PLANNED_TRAJECTORY_BASES})
```


In `src/runner/provenance.py` (3 edits):

Edit 1: replace

```python
import numpy
import scipy
```

with

```python
import clarabel
import cvxpy
import numpy
import scipy
```

Edit 2: replace

```python
SOURCE_PACKAGES = ("models", "baselines", "bounds", "optimization", "runner")
```

with

```python
SOURCE_PACKAGES = ("models", "baselines", "bounds", "optimization", "literature", "runner")
```

Edit 3: replace

```python
        "highs": highs_version(),
```

with

```python
        "highs": highs_version(),
        "cvxpy": cvxpy.__version__,
        "clarabel": clarabel.__version__,
```


In `experiments/run_phase1.py` (3 edits):

Edit 1: replace

```python
import argparse
from pathlib import Path
```

with

```python
import argparse
from dataclasses import replace
from pathlib import Path
```

Edit 2: replace

```python
    parser.add_argument("--workers", type=int)
```

with

```python
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir", type=Path, help="write results here instead of the configured output_dir")
```

Edit 3: replace

```python
    config = load_experiment_config(args.config)
```

with

```python
    config = load_experiment_config(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
```


In `experiments/run_phase2.py` (3 edits):

Edit 1: replace

```python
import argparse
from pathlib import Path
```

with

```python
import argparse
from dataclasses import replace
from pathlib import Path
```

Edit 2: replace

```python
    parser.add_argument("--workers", type=int)
```

with

```python
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir", type=Path, help="write results here instead of the configured output_dir")
```

Edit 3: replace

```python
    config = load_experiment_config(args.config)
```

with

```python
    config = load_experiment_config(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
```


- [ ] **Step 4: Run the tests to verify they pass**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/runner`

Expected: `46 passed`.

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `232 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/runner/test_runner_config.py tests/runner/test_runner_provenance.py tests/runner/test_runner_experiment.py tests/runner/test_runner_tran_tasks.py tests/runner/test_runner_output_dir.py src/runner/config.py src/runner/methods.py src/runner/experiment.py src/runner/provenance.py experiments/run_phase1.py experiments/run_phase2.py
git commit -m "feat: run TRAN from experiment configurations"
```

### Task 5: Development-set pilot

**Files:**
- Create: `src/literature/pilot.py`, `experiments/tran_pilot.py`
- Test: `tests/literature/test_tran_pilot.py`

**Interfaces:**
- Consumes: Tasks 3–4; `models.evaluate` (`REALIZED`, `evaluate`); `runner.config.load_experiment_config` in the test.
- Produces: `literature.pilot`: `RUNTIME_LIMIT_S = 600.0`, `FALLBACK_BLOCK_SLOTS = 5`, `DEFAULT_THETA = 0.5`, `DEFAULT_MU = 1.0`, `THETAS = (1/3, 1/2, 2/3)`, `MUS = (1.0, 10.0)`, `RUN_COLUMNS`, `FIT_COLUMNS`; `run_job((step, scenario_path, theta, mu, block_slots, realizations)) -> dict`; `choose_block_slots(rows) -> int`; `choose_parameters(rows) -> (theta, mu)`; `samir_fallback_fires(rows) -> bool`; `literature_config(choice) -> dict`. `experiments/tran_pilot.py --manifest --scenario-dir --output --config-output --realizations --workers` returns 0 and writes the configuration, 1 without development scenarios, 2 when the Samir rule fires.

The job function lives in `src/literature/pilot.py` instead of the script: workers are spawned and must import it.

- [ ] **Step 1: Write the failing tests**

Create `tests/literature/test_tran_pilot.py`:

```python
import csv
import importlib.util
import json
from pathlib import Path

from literature import pilot
from models.scenario import scenario_from_dict
from runner.config import load_experiment_config
from tran_builders import tran_raw

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "tran_pilot.py"
SPEC = importlib.util.spec_from_file_location("tran_pilot", SCRIPT)
tran_pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tran_pilot)


def _row(theta, mu, timely, initial=0.1, runtime=1.0):
    return {"theta": theta, "mu": mu, "timely_ratio": timely, "initial_timely_ratio": initial, "planning_runtime_s": runtime}


def test_block_rule_switches_above_the_median_runtime_limit():
    assert pilot.choose_block_slots([_row(0.5, 1.0, 0.3, runtime=value) for value in (100.0, 700.0, 500.0)]) == 1
    assert pilot.choose_block_slots([_row(0.5, 1.0, 0.3, runtime=value) for value in (601.0, 700.0, 500.0)]) == 5


def test_parameter_choice_prefers_mean_then_theta_half_then_small_mu():
    rows = [_row(1 / 3, 1.0, 0.4), _row(1 / 3, 1.0, 0.2), _row(0.5, 10.0, 0.3), _row(0.5, 10.0, 0.3), _row(2 / 3, 1.0, 0.1)]
    assert pilot.choose_parameters(rows) == (0.5, 10.0)
    assert pilot.choose_parameters([_row(2 / 3, 1.0, 0.5), _row(1 / 3, 10.0, 0.6)]) == (1 / 3, 10.0)


def test_samir_rule_fires_when_half_the_scenarios_do_not_improve():
    assert not pilot.samir_fallback_fires([_row(0.5, 1.0, 0.3), _row(0.5, 1.0, 0.3), _row(0.5, 1.0, 0.1)])
    assert pilot.samir_fallback_fires([_row(0.5, 1.0, 0.3), _row(0.5, 1.0, 0.1)])


def test_pilot_writes_runs_choice_and_a_loadable_comparison_config(tmp_path: Path):
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    rows = []
    for index in range(2):
        raw = tran_raw()
        raw["scenario_id"], raw["split"] = f"Tiny{index}-r0", "dev"
        raw["network"]["network_id"] = f"Tiny{index}"
        raw["provenance"]["scenario_seed"] = 30 + index
        (scenario_dir / f"{raw['scenario_id']}.json").write_text(json.dumps(raw), encoding="utf-8")
        rows.append({"scenario_id": raw["scenario_id"], "split": "dev", "sha256": scenario_from_dict(raw).sha256})
    manifest = tmp_path / "scenarios.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario_id", "split", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    output, config_path = tmp_path / "pilot", tmp_path / "phase2_literature.yaml"
    code = tran_pilot.main(["--manifest", str(manifest), "--scenario-dir", str(scenario_dir), "--output", str(output),
                            "--config-output", str(config_path), "--realizations", "2", "--workers", "1"])
    choice = json.loads((output / "choice.json").read_text(encoding="utf-8"))
    with (output / "runs.csv").open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    assert [row["step"] for row in runs].count("runtime") == 2 and [row["step"] for row in runs].count("grid") == 12
    assert choice["block_slots"] == 1 and choice["scenarios"] == ["Tiny0-r0", "Tiny1-r0"]
    if choice["samir_fallback"]:
        assert code == 2 and not config_path.exists()
    else:
        assert code == 0
        spec = load_experiment_config(config_path).methods[1]
        assert spec.tran_params().theta == choice["theta"] and spec.tran_params().mu == choice["mu"] and spec.bounds
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/literature/test_tran_pilot.py`

Expected: collection fails with `ImportError: cannot import name 'pilot' from 'literature' (src/literature/__init__.py)` (`1 error`).

- [ ] **Step 3: Implement**

Create `src/literature/pilot.py`:

```python
"""TRAN development-set pilot logic (spec D Section 7.1): one pilot job, block and parameter rules, Samir rule."""

from collections import defaultdict
from pathlib import Path
import statistics
import time

from models.evaluate import REALIZED, evaluate
from models.paths import candidate_paths
from models.scenario import load_scenario

from .tran import TranHD
from .tran_model import TranParams

RUNTIME_LIMIT_S = 600.0
FALLBACK_BLOCK_SLOTS = 5
DEFAULT_THETA, DEFAULT_MU = 0.5, 1.0
THETAS = (1.0 / 3.0, 0.5, 2.0 / 3.0)
MUS = (1.0, 10.0)
TIE_TOLERANCE = 1e-12
RUN_COLUMNS = (
    "step", "scenario_id", "theta", "mu", "block_slots", "status", "iterations", "relaxed_objective", "penalty",
    "planning_runtime_s", "initial_timely_ratio", "timely_ratio",
)
FIT_COLUMNS = ("scenario_id", "beta", "alpha", "r_squared", "max_log_error")


def run_job(job: tuple[str, str, float, float, int, int]) -> dict[str, object]:
    step, scenario_path, theta, mu, block_slots, realizations = job
    scenario = load_scenario(Path(scenario_path))
    candidates = candidate_paths(scenario)
    started = time.perf_counter()
    outcome = TranHD(params=TranParams(theta=theta, mu=mu, block_slots=block_slots)).solve(scenario, candidates)
    planning_s = time.perf_counter() - started
    ids = tuple(range(realizations))
    initial = evaluate(scenario, outcome.initial_plan, REALIZED, ids, candidates=candidates)
    final = evaluate(scenario, outcome.plan, REALIZED, ids, candidates=candidates)
    if not (initial.feasible and final.feasible):
        raise ValueError(f"{scenario.scenario_id}: infeasible TRAN plan")
    return {
        "step": step, "scenario_id": scenario.scenario_id, "theta": theta, "mu": mu, "block_slots": block_slots,
        "status": outcome.status, "iterations": outcome.iterations, "relaxed_objective": outcome.objectives[-1],
        "penalty": outcome.penalty, "planning_runtime_s": planning_s, "initial_timely_ratio": initial.timely_ratio,
        "timely_ratio": final.timely_ratio, "beta": outcome.channel.beta, "alpha": outcome.channel.alpha,
        "r_squared": outcome.channel.r_squared, "max_log_error": outcome.channel.max_log_error,
    }


def choose_block_slots(rows: list[dict[str, object]]) -> int:
    median = statistics.median(float(row["planning_runtime_s"]) for row in rows)
    return FALLBACK_BLOCK_SLOTS if median > RUNTIME_LIMIT_S else 1


def choose_parameters(rows: list[dict[str, object]]) -> tuple[float, float]:
    """Highest mean timely ratio over scenarios; ties go to theta nearest 1/2, then the smaller mu."""
    groups: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in rows:
        groups[(float(row["theta"]), float(row["mu"]))].append(float(row["timely_ratio"]))
    means = {key: statistics.fmean(values) for key, values in groups.items()}
    best = max(means.values())
    return min((key for key, value in means.items() if value >= best - TIE_TOLERANCE), key=lambda key: (abs(key[0] - DEFAULT_THETA), key[1]))


def samir_fallback_fires(rows: list[dict[str, object]]) -> bool:
    """Section 5.8: the chosen TRAN is not better than its initial point on at least half of the scenarios."""
    not_better = sum(float(row["timely_ratio"]) <= float(row["initial_timely_ratio"]) for row in rows)
    return 2 * not_better >= len(rows)


def literature_config(choice: dict[str, object]) -> dict[str, object]:
    params = {"theta": choice["theta"], "mu": choice["mu"], "block_slots": choice["block_slots"]}
    return {
        "experiment_id": "phase2-literature",
        "scenario_sets": [
            {"set_id": "v0", "manifest": "data/manifests/scenarios_v0.csv", "scenario_dir": "data/processed/scenarios/v0"},
            {"set_id": "v0-bh50", "manifest": "data/manifests/scenarios_v0-bh50.csv", "scenario_dir": "data/processed/scenarios/v0-bh50"},
        ],
        "split": "eval",
        "realization_ids": {"start": 0, "stop": 30},
        "methods": ["B1", {"id": "TRAN", "base": "TRAN", "params": params, "bounds": True}],
        "bounds": {"tangents": 10},
        "bootstrap": {"resamples": 10000, "seed": 20260911, "confidence": 0.95},
        "workers": 12,
        "output_dir": "results/phase2/literature",
    }
```

Create `experiments/tran_pilot.py`:

```python
"""Development-set pilot of TRAN (spec D Section 7.1).

Step 1 runs theta = 1/2 and mu = 1 at slot resolution and switches to 5-slot blocks when the median planning time
exceeds 600 s (Section 5.6). Step 2 runs the theta x mu grid at the chosen block size and picks the highest mean
timely ratio. Writes runs.csv, channel_fit.csv and choice.json, and the comparison configuration of Section 7.2;
exits 2 without that configuration when the Samir fallback rule of Section 5.8 fires.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import statistics
import sys

import yaml

from literature.pilot import (
    DEFAULT_MU, DEFAULT_THETA, FIT_COLUMNS, MUS, RUN_COLUMNS, RUNTIME_LIMIT_S, THETAS, choose_block_slots, choose_parameters,
    literature_config, run_job, samir_fallback_fires,
)


def _run(jobs: list[tuple], workers: int) -> list[dict[str, object]]:
    if not jobs:
        return []
    # spawn, as in the runner: forked children can deadlock on solver thread pools of the parent
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(run_job, jobs))


def _write_rows(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tran-pilot")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/scenarios_v0.csv"))
    parser.add_argument("--scenario-dir", type=Path, default=Path("data/processed/scenarios/v0"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2/tran_pilot"))
    parser.add_argument("--config-output", type=Path, default=Path("configs/experiments/phase2_literature.yaml"))
    parser.add_argument("--realizations", type=int, default=30)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        scenario_ids = sorted(row["scenario_id"] for row in csv.DictReader(stream) if row["split"] == "dev")
    if not scenario_ids:
        print(f"{args.manifest} lists no development scenarios", file=sys.stderr)
        return 1
    paths = [str(args.scenario_dir / f"{scenario_id}.json") for scenario_id in scenario_ids]
    runtime_rows = _run([("runtime", path, DEFAULT_THETA, DEFAULT_MU, 1, args.realizations) for path in paths], args.workers)
    block_slots = choose_block_slots(runtime_rows)
    reused = [dict(row, step="grid") for row in runtime_rows] if block_slots == 1 else []
    jobs = [
        ("grid", path, theta, mu, block_slots, args.realizations)
        for theta in THETAS for mu in MUS for path in paths
        if not (reused and (theta, mu) == (DEFAULT_THETA, DEFAULT_MU))
    ]
    grid_rows = reused + _run(jobs, args.workers)
    theta, mu = choose_parameters(grid_rows)
    chosen = [row for row in grid_rows if (row["theta"], row["mu"]) == (theta, mu)]
    choice = {
        "theta": theta, "mu": mu, "block_slots": block_slots, "runtime_limit_s": RUNTIME_LIMIT_S,
        "median_runtime_step1_s": statistics.median(float(row["planning_runtime_s"]) for row in runtime_rows),
        "scenarios": scenario_ids, "mean_timely_ratio": statistics.fmean(float(row["timely_ratio"]) for row in chosen),
        "better_than_initial_count": sum(float(row["timely_ratio"]) > float(row["initial_timely_ratio"]) for row in chosen),
        "samir_fallback": samir_fallback_fires(chosen),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    order = lambda row: (row["step"] != "runtime", row["scenario_id"], row["theta"], row["mu"])
    _write_rows(args.output / "runs.csv", RUN_COLUMNS, sorted(runtime_rows + grid_rows, key=order))
    _write_rows(args.output / "channel_fit.csv", FIT_COLUMNS, sorted(runtime_rows, key=order))
    (args.output / "choice.json").write_text(json.dumps(choice, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(choice))
    if choice["samir_fallback"]:
        print("TRAN does not improve on its initial point on half of the development scenarios: stop and report (spec D 5.8)", file=sys.stderr)
        return 2
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    header = "# Generated by experiments/tran_pilot.py from results/phase2/tran_pilot/choice.json\n"
    args.config_output.write_text(header + yaml.safe_dump(literature_config(choice), sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/literature/test_tran_pilot.py`

Expected: `4 passed`.

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `236 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/literature/test_tran_pilot.py src/literature/pilot.py experiments/tran_pilot.py
git commit -m "feat: add the TRAN development pilot"
```

### Task 6: B1 anchor check and literature statistics

**Files:**
- Create: `experiments/phase2_literature_statistics.py`
- Test: `tests/analysis/test_phase2_literature_statistics.py`

**Interfaces:**
- Consumes: `analysis.statistics` (`BOOTSTRAP_RESAMPLES`, `BOOTSTRAP_SEED`, `STATISTICS_COLUMNS`, `Comparison`, `compare_methods`).
- Produces: `experiments/phase2_literature_statistics.py`: `ANCHOR_METHOD = "B1"`, `ANCHOR_COLUMNS`, `LITERATURE_COMPARISONS`, `anchor_mismatches(main_rows, literature_rows) -> list[str]`, `main(argv) -> int` (1 when an experiment is incomplete or the anchor check fails).

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_phase2_literature_statistics.py`:

```python
import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "phase2_literature_statistics.py"
SPEC = importlib.util.spec_from_file_location("phase2_literature_statistics", SCRIPT)
literature_statistics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(literature_statistics)
COLUMNS = ["set_id", "scenario_id", "topology_id", "method", "realization_id", "timely_ratio", "timely_count", "connected_count", "shortfall"]


def _experiment(root: Path, name: str, values: dict[str, float], b1_timely: int = 3) -> Path:
    results = root / name
    results.mkdir()
    (results / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    with (results / "method_realizations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for set_id in ("v0", "v0-bh50"):
            for topology in range(10):
                for method, value in {**values, "B1": 0.1}.items():
                    timely = b1_timely if method == "B1" else round(40 * value)
                    writer.writerow({"set_id": set_id, "scenario_id": f"T{topology}-r0", "topology_id": f"T{topology}", "method": method,
                                     "realization_id": "0", "timely_ratio": str(value), "timely_count": str(timely),
                                     "connected_count": "7", "shortfall": "12.5"})
    return results


def test_writes_the_four_literature_comparisons(tmp_path: Path):
    main = _experiment(tmp_path, "main", {"P": 0.5, "B2": 0.375})
    literature = _experiment(tmp_path, "literature", {"TRAN": 0.25})
    output = tmp_path / "statistics_literature.csv"
    assert literature_statistics.main(["--main", str(main), "--literature", str(literature), "--output", str(output)]) == 0
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["set_id"], row["comparison"]) for row in rows] == [
        ("v0", "P - TRAN"), ("v0", "B2 - TRAN"), ("v0-bh50", "P - TRAN"), ("v0-bh50", "B2 - TRAN"),
    ]
    assert float(rows[0]["mean_difference"]) == 0.25 and float(rows[1]["mean_difference"]) == 0.125
    assert float(rows[0]["p_holm"]) == 4 * 2 / 1024


def test_anchor_mismatch_stops_before_pairing(tmp_path: Path, capsys):
    main = _experiment(tmp_path, "main", {"P": 0.5, "B2": 0.375})
    literature = _experiment(tmp_path, "literature", {"TRAN": 0.25}, b1_timely=4)
    output = tmp_path / "statistics_literature.csv"
    assert literature_statistics.main(["--main", str(main), "--literature", str(literature), "--output", str(output)]) == 1
    assert "timely_count" in capsys.readouterr().err and not output.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/analysis/test_phase2_literature_statistics.py`

Expected: collection fails with `FileNotFoundError: [Errno 2] No such file or directory: 'experiments/phase2_literature_statistics.py'` (`1 error`).

- [ ] **Step 3: Implement**

Create `experiments/phase2_literature_statistics.py`:

```python
"""B1 anchor check and paired comparisons of TRAN with P and B2 (spec D Section 7.3)."""

import argparse
import csv
import json
from pathlib import Path
import sys

from analysis.statistics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, STATISTICS_COLUMNS, Comparison, compare_methods

ANCHOR_METHOD = "B1"
ANCHOR_COLUMNS = ("timely_count", "connected_count", "shortfall")
LITERATURE_COMPARISONS = tuple(
    Comparison(set_id, first, "TRAN") for set_id in ("v0", "v0-bh50") for first in ("P", "B2")
)


def _complete_rows(results: Path) -> list[dict[str, str]] | None:
    manifest = results / "manifest.json"
    if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8"))["status"] != "complete":
        return None
    with (results / "method_realizations.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def anchor_mismatches(main_rows: list[dict[str, str]], literature_rows: list[dict[str, str]]) -> list[str]:
    """Differences of B1 between the two experiments on (set, scenario, realization); empty when they agree."""
    def index(rows):
        return {(row["set_id"], row["scenario_id"], row["realization_id"]): row for row in rows if row["method"] == ANCHOR_METHOD}

    main, literature = index(main_rows), index(literature_rows)
    if not literature:
        return ["the literature results hold no B1 rows"]
    messages = [f"{key}: B1 row missing from the main results" for key in sorted(literature.keys() - main.keys())]
    for key in sorted(literature.keys() & main.keys()):
        for column in ANCHOR_COLUMNS:
            if float(main[key][column]) != float(literature[key][column]):
                messages.append(f"{key} {column}: main {main[key][column]}, literature {literature[key][column]}")
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase2-literature-statistics")
    parser.add_argument("--main", type=Path, default=Path("results/phase2/main"))
    parser.add_argument("--literature", type=Path, default=Path("results/phase2/literature"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2/statistics_literature.csv"))
    args = parser.parse_args(argv)
    main_rows, literature_rows = _complete_rows(args.main), _complete_rows(args.literature)
    if main_rows is None or literature_rows is None:
        print("both --main and --literature must hold complete experiments", file=sys.stderr)
        return 1
    mismatches = anchor_mismatches(main_rows, literature_rows)
    if mismatches:
        print("B1 anchor check failed; pairing with the main results is invalid:", file=sys.stderr)
        for message in mismatches[:5]:
            print(f"  {message}", file=sys.stderr)
        return 1
    rows = [row for row in main_rows if row["method"] in ("P", "B2")] + [row for row in literature_rows if row["method"] == "TRAN"]
    results = compare_methods(rows, LITERATURE_COMPARISONS, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/analysis/test_phase2_literature_statistics.py`

Expected: `2 passed`.

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `238 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/analysis/test_phase2_literature_statistics.py experiments/phase2_literature_statistics.py
git commit -m "feat: compare TRAN with P and B2 after a B1 anchor check"
```

### Task 7: Development pilot

**Files:**
- Create (generated): `results/phase2/tran_pilot/runs.csv`, `results/phase2/tran_pilot/channel_fit.csv`, `results/phase2/tran_pilot/choice.json`, `results/phase2/tran_pilot/pilot_note.md`, `configs/experiments/phase2_literature.yaml`

**Interfaces:**
- Consumes: Tasks 1–5; the six development scenarios of `data/manifests/scenarios_v0.csv` (generated scenario files under `data/processed/scenarios/v0/`).
- Produces: `theta`, `mu` and `block_slots` in `choice.json` and in `configs/experiments/phase2_literature.yaml` for Task 8 (spec Section 7.1, acceptance criterion 16.1.2).

- [ ] **Step 1: Run the pilot**

Run: `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/tran_pilot.py`

Expected: exit code 0 and one JSON line with `theta`, `mu`, `block_slots`, `runtime_limit_s` 600.0, `median_runtime_step1_s`, `scenarios` `["Bbnplanet-r0", "Bbnplanet-r1", "Canerie-r0", "Canerie-r1", "Fccn-r0", "Fccn-r1"]`, `mean_timely_ratio`, `better_than_initial_count` and `"samir_fallback": false`, after about 15 minutes with 12 workers. In the prototype, θ = 1/2 and μ = 1 at slot resolution planned in 12–157 s per scenario and improved on the initial point on all six scenarios, so `block_slots` 1 is expected.

If the command exits with 2, the Samir rule of spec Section 5.8 fired: stop, report `runs.csv` and `choice.json` to the user, and do not run Task 8. If it exits with 137, the memory cap stopped it: report the last output instead of raising the cap.

- [ ] **Step 2: Check the pilot files**

````bash
uv run python - <<'PY'
import csv
import json
from pathlib import Path

from runner.config import load_experiment_config

root = Path("results/phase2/tran_pilot")
choice = json.loads((root / "choice.json").read_text(encoding="utf-8"))
with (root / "runs.csv").open(newline="", encoding="utf-8") as stream:
    runs = list(csv.DictReader(stream))
with (root / "channel_fit.csv").open(newline="", encoding="utf-8") as stream:
    fits = list(csv.DictReader(stream))
steps = [row["step"] for row in runs]
print("runs", len(runs), "runtime", steps.count("runtime"), "grid", steps.count("grid"))
print("solver failures", sum(row["status"].startswith("solver_") for row in runs))
print("channel fits", len(fits), "min r_squared above 0.99:", min(float(row["r_squared"]) for row in fits) > 0.99)
spec = load_experiment_config(Path("configs/experiments/phase2_literature.yaml")).methods[1]
params = spec.tran_params()
print("config matches choice:", (params.theta, params.mu, params.block_slots) == (choice["theta"], choice["mu"], choice["block_slots"]) and spec.bounds)
PY
````

Expected:

```text
runs 42 runtime 6 grid 36
solver failures 0
channel fits 6 min r_squared above 0.99: True
config matches choice: True
```

With block size 1 the grid reuses the six step-1 runs for θ = 1/2, μ = 1 (as rows with step `grid`) and runs the other 30; with block size 5 it runs all 36. A solver failure is not a stop condition, because the method keeps its last accepted iterate; the note of Step 3 reports every status.

- [ ] **Step 3: Write the pilot note**

````bash
uv run python - <<'PY'
import csv
import json
from pathlib import Path
import statistics

root = Path("results/phase2/tran_pilot")
choice = json.loads((root / "choice.json").read_text(encoding="utf-8"))
with (root / "runs.csv").open(newline="", encoding="utf-8") as stream:
    runs = list(csv.DictReader(stream))
with (root / "channel_fit.csv").open(newline="", encoding="utf-8") as stream:
    fits = list(csv.DictReader(stream))
runtime = [row for row in runs if row["step"] == "runtime"]
groups = {}
for row in runs:
    if row["step"] == "grid":
        groups.setdefault((float(row["theta"]), float(row["mu"])), []).append(row)
means = {key: statistics.fmean(float(row["timely_ratio"]) for row in rows) for key, rows in groups.items()}
statuses = {}
for row in runs:
    statuses[row["status"]] = statuses.get(row["status"], 0) + 1
lines = [
    "# TRAN pilot on the development split",
    "",
    f"Command `experiments/tran_pilot.py` on the {len(choice['scenarios'])} development scenarios of `v0` "
    f"({', '.join(choice['scenarios'])}), 30 evaluation realizations per plan. "
    "Spec: `docs/superpowers/specs/2026-09-12-extensions-final-delivery-design.md` Sections 5.6, 5.8 and 7.1.",
    "",
    "## Step 1: theta = 1/2, mu = 1, slot resolution",
    "",
    "| Scenario | Status | Iterations | Planning time (s) | Initial timely ratio | TRAN timely ratio |",
    "|---|---|---|---|---|---|",
]
for row in runtime:
    lines.append(
        f"| {row['scenario_id']} | {row['status']} | {row['iterations']} | {float(row['planning_runtime_s']):.1f} "
        f"| {float(row['initial_timely_ratio']):.3f} | {float(row['timely_ratio']):.3f} |"
    )
lines += [
    "",
    f"Median planning time {choice['median_runtime_step1_s']:.1f} s against the limit of {choice['runtime_limit_s']:.0f} s, "
    f"so the grid runs with block size {choice['block_slots']}.",
    "",
    "## Grid",
    "",
    "| theta | mu | Mean timely ratio | Mean planning time (s) | Mean iterations |",
    "|---|---|---|---|---|",
]
for (theta, mu), rows in sorted(groups.items()):
    planning = statistics.fmean(float(row["planning_runtime_s"]) for row in rows)
    iterations = statistics.fmean(int(row["iterations"]) for row in rows)
    lines.append(f"| {theta:.3f} | {mu:g} | {means[(theta, mu)]:.3f} | {planning:.1f} | {iterations:.1f} |")
lines += ["", "## Effective channel fit", "", "| Scenario | beta | alpha | R² | Max log error |", "|---|---|---|---|---|"]
for row in fits:
    lines.append(
        f"| {row['scenario_id']} | {float(row['beta']):.3e} | {float(row['alpha']):.3f} | {float(row['r_squared']):.4f} "
        f"| {float(row['max_log_error']):.3f} |"
    )
spread = max(means.values()) - min(means.values())
lines += [
    "",
    "## Decision",
    "",
    f"- **Parameters for the comparison:** theta = {choice['theta']:.3f}, mu = {choice['mu']:g}, block size {choice['block_slots']} "
    f"(highest mean timely ratio {choice['mean_timely_ratio']:.3f}; the grid means span {spread:.3f}).",
    f"- **Samir rule:** the chosen TRAN beats its initial point on {choice['better_than_initial_count']} of {len(choice['scenarios'])} "
    f"scenarios, so the rule {'fires' if choice['samir_fallback'] else 'does not fire'}.",
    "- **Solver statuses over all runs:** " + ", ".join(f"{status} {count}" for status, count in sorted(statuses.items())) + ".",
    "- Evaluation-split scenarios were not used; `configs/experiments/phase2_literature.yaml` was written from `choice.json`.",
]
(root / "pilot_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(root / "pilot_note.md")
PY
````

Expected: `results/phase2/tran_pilot/pilot_note.md` is printed; the note has the sections "Step 1", "Grid", "Effective channel fit" and "Decision".

- [ ] **Step 4: Confirm that only the generated files are new, then commit**

Run: `git status --short --untracked-files=all results/phase2/tran_pilot configs/experiments`

Expected: exactly the five files listed under **Files**.

```bash
git add results/phase2/tran_pilot/runs.csv results/phase2/tran_pilot/channel_fit.csv results/phase2/tran_pilot/choice.json results/phase2/tran_pilot/pilot_note.md configs/experiments/phase2_literature.yaml
git commit -m "feat: record the TRAN development pilot"
```

### Task 8: Comparison with P and B2

**Files:**
- Create (generated): `results/phase2/literature/method_realizations.csv`, `results/phase2/literature/bounds.csv`, `results/phase2/literature/summary.csv`, `results/phase2/literature/manifest.json`, `results/phase2/statistics_literature.csv`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 4, 6 and 7; `results/phase2/main/` from plan C.2.
- Produces: TRAN results and the four paired comparisons for plan D.3 (acceptance criteria 16.1.3–16.1.4). The TRAN rows of `results/phase2/literature/summary.csv` (planning runtime, timely ratio) are the quality–runtime points of spec Section 7.4, drawn by plan D.3.

- [ ] **Step 1: Run the comparison**

Run: `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/run_phase2.py --config configs/experiments/phase2_literature.yaml`

Expected: `{"task_count": 1920, "executed_task_count": 1920, "skipped_task_count": 0, "status": "complete"}` and exit code 0 after roughly 30 minutes with 12 workers. Each of the 60 scenarios has two method tasks (B1, TRAN) and 30 B1 bound tasks; TRAN solves its 30 trajectory bounds inside its method task. If the status is `failed`, stop and report the failures printed by the runner.

- [ ] **Step 2: Check the results**

````bash
uv run python - results/phase2/literature <<'PY'
import csv
import json
import sys

root = sys.argv[1]


def rows(name):
    with open(f"{root}/{name}", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


manifest = json.load(open(f"{root}/manifest.json", encoding="utf-8"))
methods, bounds, summary = rows("method_realizations.csv"), rows("bounds.csv"), rows("summary.csv")
print(manifest["status"], manifest["task_count"], "failures", len(manifest["failures"]))
print(len(methods), "method rows;", len(bounds), "bound rows; statuses", sorted({row["status"] for row in bounds}))
print("bounded trajectories", sorted({row["trajectory_method"] for row in bounds}))
print(len(summary), "summary rows; evaluate calls", sorted({row["evaluate_calls"] for row in methods}))
PY
````

Expected:

```text
complete 1920 failures 0
3600 method rows; 3600 bound rows; statuses ['optimal']
bounded trajectories ['B1', 'TRAN']
4 summary rows; evaluate calls ['1']
```

TRAN makes no design evaluations; the single call per method row is the runner's final evaluation.

- [ ] **Step 3: Confirm that a rerun skips every task**

Run: `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/run_phase2.py --config configs/experiments/phase2_literature.yaml`

Expected: `{"task_count": 1920, "executed_task_count": 0, "skipped_task_count": 1920, "status": "complete"}`. The rerun rewrites `manifest.json` with its own timestamps; the result CSVs are unchanged.

- [ ] **Step 4: Check the anchor and compute the statistics**

Run: `uv run python experiments/phase2_literature_statistics.py`

Expected: `{"comparisons": 4, "output": "results/phase2/statistics_literature.csv"}`. If the script exits with 1 and prints "B1 anchor check failed", stop and report the listed differences: the TRAN rows cannot be paired with the main results.

````bash
uv run python - <<'PY'
import csv

with open("results/phase2/statistics_literature.csv", newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        print(row["set_id"], row["comparison"], row["topology_count"], f"{float(row['mean_difference']):+.3f}", f"p_holm={float(row['p_holm']):.4f}")
PY
````

Expected: four lines in the order `v0 P - TRAN`, `v0 B2 - TRAN`, `v0-bh50 P - TRAN`, `v0-bh50 B2 - TRAN`, each with topology count 10. The values are reported whatever their sign or significance.

- [ ] **Step 5: Document the commands in the README**

Append to the end of `README.md`:

````markdown

## Pha 2: baseline từ tài liệu (Tran et al. 2022, bán song công)

Phương pháp `TRAN` chuyển thể thuật toán xấp xỉ trong (inner approximation) của Tran et al. (2022) cho mô hình của dự án (`src/literature/`, spec D Mục 5). Mỗi vòng lặp giải một bài toán nón bậc hai bằng `cvxpy` với Clarabel. Các lệnh giải TRAN chạy trong `systemd-run` có giới hạn bộ nhớ, để một lỗi tràn bộ nhớ chỉ dừng đúng tiến trình đó.

Chạy pilot trên tập phát triển của `v0`; pilot chọn θ, μ, kích thước block và sinh `configs/experiments/phase2_literature.yaml`:

```bash
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/tran_pilot.py
```

Chạy thí nghiệm so sánh (B1 làm mốc và TRAN trên `v0`, `v0-bh50`), rồi kiểm tra mốc B1 và tính bốn so sánh ghép cặp với P và B2 của thí nghiệm chính:

```bash
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/run_phase2.py --config configs/experiments/phase2_literature.yaml
uv run python experiments/phase2_literature_statistics.py
```

Kết quả nằm trong `results/phase2/tran_pilot/`, `results/phase2/literature/` và `results/phase2/statistics_literature.csv`. `experiments/run_phase1.py` và `experiments/run_phase2.py` nhận thêm `--output-dir` để ghi kết quả vào thư mục khác với thư mục trong cấu hình.
````

- [ ] **Step 6: Confirm the changed files, then commit**

Run: `git status --short --untracked-files=all results/phase2 README.md`

Expected: the five result files listed under **Files** as new and `README.md` as modified; no shard appears.

```bash
git add results/phase2/literature/method_realizations.csv results/phase2/literature/bounds.csv results/phase2/literature/summary.csv results/phase2/literature/manifest.json results/phase2/statistics_literature.csv README.md
git commit -m "feat: record TRAN comparison results and statistics"
```

## Spec Coverage

| Spec section | Task |
|---|---|
| 5.1 Instance construction | 1 |
| 5.2 Convex program per iteration | 2 |
| 5.3 Initialization and stopping | 2 |
| 5.4 Plan conversion | 3 |
| 5.5 Parameters | 1 (validation), 4 (configuration) |
| 5.6 Time aggregation fallback | 1 (block windows), 2 (block program), 3 (expansion), 5 (rule), 7 (applied) |
| 5.7 Complexity | counts from `TranInstance` (Task 1) and iterations and runtime from the pilot (Task 7); written up by plan D.3 |
| 5.8 Fallback to Samir et al. | 5 (rule), 7 (stop) |
| 6 Runner changes | 4 |
| 7.1 Pilot | 5, 7 |
| 7.2 Comparison | 7 (configuration), 8 |
| 7.3 Statistics | 6, 8 |
| 7.4 Quality–runtime | data: 8; figure: plan D.3 |
| 14 Error handling: TRAN, statistics | 2, 3, 6 |
| 15.1 Testing | 1–6 |
| 16.1.1 Full test suite | every code task |
| 16.1.2 Pilot results and note | 7 |
| 16.1.3 Comparison complete, LPs optimal, rerun skips, results committed | 8 |
| 16.1.4 Four comparisons after the anchor check | 8 |
| 17 Plan D.1 adjustments | 1 (fit range, block divisibility), 2 (numeric constants, solver settings, tight slacks), 5 (pilot module, configuration), Global Constraints (memory caps) |
