# Model and Evaluator Implementation Plan (Sub-Project A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v0 scenario set (30 evaluation + 6 development scenarios) and an independent, tested plan evaluator that every later method uses.

**Architecture:** A new package `src/models/` holds the typed scenario loader, channel models, candidate paths, plans, a slot simulator with an EDF dispatcher, an independent ledger checker, metrics, and the single `evaluate(...)` entry point; it never imports optimization or experiment code. `src/data/scenarios.py` is rewritten as the schema v2 generator and exposed through a new `scenarios` stage of `data.cli`. `experiments/calibrate_v0.py` runs the calibration check on the generated set.

**Tech Stack:** Python 3.11+, `numpy`, `networkx`, `PyYAML`, `pytest` (all already declared in `pyproject.toml`); standard-library `dataclasses`, `hashlib`, `json`, `csv`, `math`, `argparse`.

**Spec:** `docs/superpowers/specs/2026-09-11-model-evaluator-design.md` (roadmap: `docs/superpowers/specs/2026-09-11-implementation-roadmap.md`)

## Global Constraints

- No new third-party dependency; `pyproject.toml` stays unchanged. The editable install already puts `src/` on `sys.path`, so `models` imports without reinstalling.
- `src/models/` imports nothing from `data`, `src/baselines/`, `src/optimization/`, or `experiments/`. `src/models/checker.py` must not import `src/models/simulator.py`.
- Scenario `schema_version` is 2, `generator_version` is 2, `paths.version` is 1.
- v0 defaults: 13 strata, 3 development topologies, 3 evaluation replicates, 2 development replicates; 3 damage zones at least 600 m apart with radius 400 m; 15 sources with sigma 150 m; 40 alerts; deadline window 20–60 slots; `L_ref = 250000` bits with size ratio clipped to [0.25, 4] from SNDlib `abilene`; `p_in = 0.6`, `p_out = 0.05`; backhaul 1,000,000 bit/s; H = 100 m; V_max = 50 m/s; slot 1 s; N = 150; access fraction 0.5; B_tot = 1,000,000 Hz; noise −140 dBm/Hz; PrLoS a = 9.61, b = 0.16, mu = 0.2, alpha_L = 2.2, alpha_N = 3.3, beta0 = −50 dB; ground alpha 2.8, beta0 −50 dB; usable rate 10,000 bit/s; source power 0.1 W; UAV total power 0.1 W with at most 2 active downlinks; K = 3 with at most 2 ground candidates.
- Evaluation realizations draw from `named_rng(scenario_seed, f"channel-eval:{realization_id}")`; the `channel-design:` namespace is reserved and unused here.
- Every transfer in slot n uses only the buffer present at the start of slot n; queues are served in `(deadline_slot, alert_id)` order.
- An alert is timely when at least `size_bits - 1e-6` bits reach the center by the end of slot `deadline_slot - 1`.
- Lexicographic key: `(timely_count, connected_source_count, -round(total_shortfall, 9))`; an infeasible plan has key `(-1, -1, -inf)`.
- Default tests run offline with `python -m pytest` from the repository root. Each task follows test-driven development and ends with its own commit.
- Raw and processed data stay Git-ignored; `data/manifests/scenarios_v0.csv` and `results/calibration/v0.json` are committed.

---
### Task 1: `models` package, deterministic RNG, and scenario schema v2 loader

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/rng.py`
- Create: `src/models/scenario.py`
- Create: `tests/models/scenario_builders.py`
- Create: `tests/models/conftest.py`
- Test: `tests/models/test_scenario.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `models.rng.named_rng(seed: int, concern: str) -> numpy.random.Generator` and `canonical_sha256(payload: Mapping[str, object]) -> str`. `models.scenario` exports `SCHEMA_VERSION = 2`, `UAV_ID = "UAV"`, `ScenarioValidationError` (attribute `field: str`), frozen dataclasses `Node(id, x_m, y_m)` with `.xy`, `Edge(source, target, failed, capacity_bps)` with `.alive`, `DamageZone(x_m, y_m, radius_m)`, `SourcePoint(id, x_m, y_m, zone)` with `.xy`, `Alert(id, source, release_slot, deadline_slot, size_bits, size_ratio, provenance)`, `UavConfig(altitude_m, v_max_mps, start_xy_m, end_xy_m, total_power_w, max_active_downlinks)` with `.downlink_power_w`, `TimeConfig(slot_s, num_slots, access_fraction)`, `SpectrumConfig(b_tot_hz, noise_psd_dbm_per_hz)`, `UavChannelConfig(plos_a, plos_b, nlos_attenuation, alpha_los, alpha_nlos, beta0_db)`, `GroundChannelConfig(alpha, beta0_db)`, `PathConfig(k, max_ground, version)`, and `Scenario` (fields `scenario_id, split, network_id, nodes, edges, center, relays, damage_zones, sources, alerts, uav, time, spectrum, uav_channel, ground_channel, usable_rate_bps, source_power_w, paths, scenario_seed, provenance, sha256`; helpers `node_by_id`, `source_by_id`, `alert_by_id`, `position(point_id) -> tuple[float, float]`, `backhaul_capacity_bps(first, second) -> float`). Functions `scenario_from_dict(raw: Mapping[str, object]) -> Scenario` and `load_scenario(path: Path) -> Scenario`. Test helpers: `scenario_builders.BASE_SCENARIO`, `scenario_builders.make_alert(alert_id, source, release, deadline, size) -> dict`, and the `raw_scenario` fixture (a deep copy of `BASE_SCENARIO`).

The tiny test scenario places center `n0` at (1000, 1000), relays `n1` (200 m east), `n2` (400 m north), and `n3` (behind `n1`), source `s00` 30 m from `n1`, and source `s01` far away at (100, 100). Later tasks rely on these distances.

- [ ] **Step 1: Write the test helpers and the failing loader tests**

```python
# tests/models/scenario_builders.py
BASE_SCENARIO = {
    "schema_version": 2,
    "scenario_id": "Tiny-r0",
    "split": "eval",
    "network": {
        "network_id": "Tiny",
        "nodes": [
            {"id": "n0", "x_m": 1000.0, "y_m": 1000.0},
            {"id": "n1", "x_m": 1200.0, "y_m": 1000.0},
            {"id": "n2", "x_m": 1000.0, "y_m": 1400.0},
            {"id": "n3", "x_m": 1700.0, "y_m": 1000.0},
        ],
        "edges": [
            {"source": "n0", "target": "n1", "failed": False, "capacity_bps": 1000000.0},
            {"source": "n0", "target": "n2", "failed": False, "capacity_bps": 1000000.0},
            {"source": "n1", "target": "n3", "failed": False, "capacity_bps": 1000000.0},
        ],
        "normalization": {"box_size_m": 2000.0, "scale_x": 1.0, "scale_y": 1.0},
    },
    "center": "n0",
    "relays": ["n1", "n2", "n3"],
    "damage_zones": [{"x_m": 1230.0, "y_m": 1000.0, "radius_m": 400.0}],
    "sources": [
        {"id": "s00", "x_m": 1230.0, "y_m": 1000.0, "zone": 0},
        {"id": "s01", "x_m": 100.0, "y_m": 100.0, "zone": 0},
    ],
    "alerts": [
        {
            "id": "alert-0000",
            "source": "s00",
            "release_slot": 0,
            "deadline_slot": 10,
            "size_bits": 1000000,
            "size_ratio": 1.0,
            "provenance": "derived-from-sndlib-demand",
        }
    ],
    "uav": {
        "altitude_m": 100.0,
        "v_max_mps": 50.0,
        "start_xy_m": [1000.0, 1000.0],
        "end_xy_m": [1000.0, 1000.0],
        "total_power_w": 0.1,
        "max_active_downlinks": 2,
    },
    "time": {"slot_s": 1.0, "num_slots": 20, "access_fraction": 0.5},
    "spectrum": {"b_tot_hz": 1000000.0, "noise_psd_dbm_per_hz": -140.0},
    "channel": {
        "uav": {
            "plos_a": 9.61,
            "plos_b": 0.16,
            "nlos_attenuation": 0.2,
            "alpha_los": 2.2,
            "alpha_nlos": 3.3,
            "beta0_db": -50.0,
        },
        "ground": {"alpha": 2.8, "beta0_db": -50.0},
        "usable_rate_bps": 10000.0,
    },
    "source_power_w": 0.1,
    "paths": {"k": 3, "max_ground": 2, "version": 1},
    "provenance": {"scenario_seed": 7, "generator_version": 2},
}


def make_alert(alert_id, source, release, deadline, size):
    return {
        "id": alert_id,
        "source": source,
        "release_slot": release,
        "deadline_slot": deadline,
        "size_bits": size,
        "size_ratio": 1.0,
        "provenance": "derived-from-sndlib-demand",
    }


def make_alert(alert_id, source, release, deadline, size):
    return {
        "id": alert_id,
        "source": source,
        "release_slot": release,
        "deadline_slot": deadline,
        "size_bits": size,
        "size_ratio": 1.0,
        "provenance": "derived-from-sndlib-demand",
    }
```

```python
# tests/models/conftest.py
import copy

import pytest

from scenario_builders import BASE_SCENARIO


@pytest.fixture
def raw_scenario():
    return copy.deepcopy(BASE_SCENARIO)
```

```python
# tests/models/test_scenario.py
import json

import pytest

from models.rng import canonical_sha256
from models.scenario import ScenarioValidationError, load_scenario, scenario_from_dict


def test_loads_valid_scenario_and_hashes_canonical_json(raw_scenario, tmp_path):
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(raw_scenario, indent=2), encoding="utf-8")
    scenario = load_scenario(path)
    assert scenario.center == "n0"
    assert scenario.scenario_seed == 7
    assert scenario.uav.downlink_power_w == pytest.approx(0.05)
    assert scenario.sha256 == canonical_sha256(raw_scenario)
    assert scenario.position("s00") == (1230.0, 1000.0)
    assert scenario.backhaul_capacity_bps("n1", "n0") == 1000000.0
    assert scenario.backhaul_capacity_bps("n2", "n1") == 0.0


def test_missing_field_names_its_path(raw_scenario):
    del raw_scenario["alerts"][0]["deadline_slot"]
    with pytest.raises(ScenarioValidationError) as error:
        scenario_from_dict(raw_scenario)
    assert error.value.field == "alerts[0].deadline_slot"


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda raw: raw.update(schema_version=1), "schema_version"),
        (lambda raw: raw["alerts"][0].update(deadline_slot=21), "alerts[0].deadline_slot"),
        (lambda raw: raw["alerts"][0].update(source="s99"), "alerts[0].source"),
        (lambda raw: raw.update(center="n9"), "center"),
        (lambda raw: raw["sources"][0].update(id="n1"), "sources[0].id"),
        (lambda raw: raw["network"]["edges"][0].update(target="n9"), "network.edges[0].target"),
        (lambda raw: raw["time"].update(access_fraction=1.0), "time.access_fraction"),
    ],
)
def test_rejects_invalid_references_and_ranges(raw_scenario, mutate, field):
    mutate(raw_scenario)
    with pytest.raises(ScenarioValidationError) as error:
        scenario_from_dict(raw_scenario)
    assert error.value.field == field
```

- [ ] **Step 2: Run the loader tests and verify they fail**

Run: `python -m pytest tests/models/test_scenario.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'models'`.

- [ ] **Step 3: Implement the package, RNG helpers, and loader**

```python
# src/models/__init__.py
"""System model, channel, candidate paths, and plan evaluator."""
```

```python
# src/models/rng.py
import hashlib
import json
from typing import Mapping

import numpy as np


def named_rng(seed: int, concern: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{concern}".encode()).digest()
    child_seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(child_seed)


def canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
```

```python
# src/models/scenario.py
from dataclasses import dataclass
from functools import cached_property
import json
import math
from pathlib import Path
from typing import Mapping

from .rng import canonical_sha256

SCHEMA_VERSION = 2
UAV_ID = "UAV"


class ScenarioValidationError(ValueError):
    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field


@dataclass(frozen=True)
class Node:
    id: str
    x_m: float
    y_m: float

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x_m, self.y_m)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    failed: bool
    capacity_bps: float

    @property
    def alive(self) -> bool:
        return not self.failed and self.capacity_bps > 0.0


@dataclass(frozen=True)
class DamageZone:
    x_m: float
    y_m: float
    radius_m: float


@dataclass(frozen=True)
class SourcePoint:
    id: str
    x_m: float
    y_m: float
    zone: int

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x_m, self.y_m)


@dataclass(frozen=True)
class Alert:
    id: str
    source: str
    release_slot: int
    deadline_slot: int
    size_bits: int
    size_ratio: float
    provenance: str


@dataclass(frozen=True)
class UavConfig:
    altitude_m: float
    v_max_mps: float
    start_xy_m: tuple[float, float]
    end_xy_m: tuple[float, float]
    total_power_w: float
    max_active_downlinks: int

    @property
    def downlink_power_w(self) -> float:
        return self.total_power_w / self.max_active_downlinks


@dataclass(frozen=True)
class TimeConfig:
    slot_s: float
    num_slots: int
    access_fraction: float


@dataclass(frozen=True)
class SpectrumConfig:
    b_tot_hz: float
    noise_psd_dbm_per_hz: float


@dataclass(frozen=True)
class UavChannelConfig:
    plos_a: float
    plos_b: float
    nlos_attenuation: float
    alpha_los: float
    alpha_nlos: float
    beta0_db: float


@dataclass(frozen=True)
class GroundChannelConfig:
    alpha: float
    beta0_db: float


@dataclass(frozen=True)
class PathConfig:
    k: int
    max_ground: int
    version: int


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str
    network_id: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    center: str
    relays: tuple[str, ...]
    damage_zones: tuple[DamageZone, ...]
    sources: tuple[SourcePoint, ...]
    alerts: tuple[Alert, ...]
    uav: UavConfig
    time: TimeConfig
    spectrum: SpectrumConfig
    uav_channel: UavChannelConfig
    ground_channel: GroundChannelConfig
    usable_rate_bps: float
    source_power_w: float
    paths: PathConfig
    scenario_seed: int
    provenance: Mapping[str, object]
    sha256: str

    @cached_property
    def node_by_id(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}

    @cached_property
    def source_by_id(self) -> dict[str, SourcePoint]:
        return {source.id: source for source in self.sources}

    @cached_property
    def alert_by_id(self) -> dict[str, Alert]:
        return {alert.id: alert for alert in self.alerts}

    @cached_property
    def _edge_capacity(self) -> dict[frozenset[str], float]:
        return {frozenset((edge.source, edge.target)): edge.capacity_bps for edge in self.edges if edge.alive}

    def position(self, point_id: str) -> tuple[float, float]:
        if point_id in self.node_by_id:
            return self.node_by_id[point_id].xy
        return self.source_by_id[point_id].xy

    def backhaul_capacity_bps(self, first: str, second: str) -> float:
        return self._edge_capacity.get(frozenset((first, second)), 0.0)


def _name(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _field(raw: object, key: str, path: str) -> object:
    if not isinstance(raw, Mapping):
        raise ScenarioValidationError(path or "scenario", "expected an object")
    if key not in raw:
        raise ScenarioValidationError(_name(path, key), "missing field")
    return raw[key]


def _number(raw: object, key: str, path: str) -> float:
    value = _field(raw, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ScenarioValidationError(_name(path, key), "expected a finite number")
    return float(value)


def _integer(raw: object, key: str, path: str) -> int:
    value = _field(raw, key, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenarioValidationError(_name(path, key), "expected an integer")
    return value


def _text(raw: object, key: str, path: str) -> str:
    value = _field(raw, key, path)
    if not isinstance(value, str) or not value:
        raise ScenarioValidationError(_name(path, key), "expected a non-empty string")
    return value


def _boolean(raw: object, key: str, path: str) -> bool:
    value = _field(raw, key, path)
    if not isinstance(value, bool):
        raise ScenarioValidationError(_name(path, key), "expected a boolean")
    return value


def _items(raw: object, key: str, path: str) -> list[object]:
    value = _field(raw, key, path)
    if not isinstance(value, list):
        raise ScenarioValidationError(_name(path, key), "expected a list")
    return value


def _pair(raw: object, key: str, path: str) -> tuple[float, float]:
    value = _field(raw, key, path)
    valid = (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )
    if not valid:
        raise ScenarioValidationError(_name(path, key), "expected two numbers")
    return (float(value[0]), float(value[1]))


def _require(condition: bool, field: str, message: str) -> None:
    if not condition:
        raise ScenarioValidationError(field, message)


def _unique(ids: list[str], path: str) -> None:
    seen: set[str] = set()
    for index, value in enumerate(ids):
        _require(value not in seen, f"{path}[{index}].id", f"duplicate id '{value}'")
        seen.add(value)


def scenario_from_dict(raw: Mapping[str, object]) -> Scenario:
    version = _integer(raw, "schema_version", "")
    _require(version == SCHEMA_VERSION, "schema_version", f"expected {SCHEMA_VERSION}, got {version}")
    split = _text(raw, "split", "")
    _require(split in ("eval", "dev"), "split", "expected 'eval' or 'dev'")

    network = _field(raw, "network", "")
    nodes = tuple(
        Node(
            id=_text(item, "id", f"network.nodes[{index}]"),
            x_m=_number(item, "x_m", f"network.nodes[{index}]"),
            y_m=_number(item, "y_m", f"network.nodes[{index}]"),
        )
        for index, item in enumerate(_items(network, "nodes", "network"))
    )
    _unique([node.id for node in nodes], "network.nodes")
    node_ids = {node.id for node in nodes}
    edges = []
    for index, item in enumerate(_items(network, "edges", "network")):
        path = f"network.edges[{index}]"
        edge = Edge(
            source=_text(item, "source", path),
            target=_text(item, "target", path),
            failed=_boolean(item, "failed", path),
            capacity_bps=_number(item, "capacity_bps", path),
        )
        _require(edge.source in node_ids, f"{path}.source", f"unknown node '{edge.source}'")
        _require(edge.target in node_ids, f"{path}.target", f"unknown node '{edge.target}'")
        _require(edge.capacity_bps >= 0.0, f"{path}.capacity_bps", "must be non-negative")
        edges.append(edge)

    center = _text(raw, "center", "")
    _require(center in node_ids, "center", f"unknown node '{center}'")
    relays = []
    for index, value in enumerate(_items(raw, "relays", "")):
        _require(isinstance(value, str) and value in node_ids, f"relays[{index}]", f"unknown node '{value}'")
        relays.append(value)

    zones = tuple(
        DamageZone(
            x_m=_number(item, "x_m", f"damage_zones[{index}]"),
            y_m=_number(item, "y_m", f"damage_zones[{index}]"),
            radius_m=_number(item, "radius_m", f"damage_zones[{index}]"),
        )
        for index, item in enumerate(_items(raw, "damage_zones", ""))
    )
    sources = tuple(
        SourcePoint(
            id=_text(item, "id", f"sources[{index}]"),
            x_m=_number(item, "x_m", f"sources[{index}]"),
            y_m=_number(item, "y_m", f"sources[{index}]"),
            zone=_integer(item, "zone", f"sources[{index}]"),
        )
        for index, item in enumerate(_items(raw, "sources", ""))
    )
    _require(bool(sources), "sources", "expected at least one source")
    _unique([source.id for source in sources], "sources")
    for index, source in enumerate(sources):
        _require(
            source.id not in node_ids and source.id != UAV_ID,
            f"sources[{index}].id",
            f"'{source.id}' collides with a node id or the UAV id",
        )
    source_ids = {source.id for source in sources}

    time_raw = _field(raw, "time", "")
    time = TimeConfig(
        slot_s=_number(time_raw, "slot_s", "time"),
        num_slots=_integer(time_raw, "num_slots", "time"),
        access_fraction=_number(time_raw, "access_fraction", "time"),
    )
    _require(time.slot_s > 0.0, "time.slot_s", "must be positive")
    _require(time.num_slots >= 1, "time.num_slots", "must be at least 1")
    _require(0.0 < time.access_fraction < 1.0, "time.access_fraction", "must lie in (0, 1)")

    alerts = []
    for index, item in enumerate(_items(raw, "alerts", "")):
        path = f"alerts[{index}]"
        alert = Alert(
            id=_text(item, "id", path),
            source=_text(item, "source", path),
            release_slot=_integer(item, "release_slot", path),
            deadline_slot=_integer(item, "deadline_slot", path),
            size_bits=_integer(item, "size_bits", path),
            size_ratio=_number(item, "size_ratio", path),
            provenance=_text(item, "provenance", path),
        )
        _require(alert.source in source_ids, f"{path}.source", f"unknown source '{alert.source}'")
        _require(alert.release_slot >= 0, f"{path}.release_slot", "must be non-negative")
        _require(alert.deadline_slot > alert.release_slot, f"{path}.deadline_slot", "must exceed release_slot")
        _require(alert.deadline_slot <= time.num_slots, f"{path}.deadline_slot", "must not exceed time.num_slots")
        _require(alert.size_bits >= 1, f"{path}.size_bits", "must be positive")
        alerts.append(alert)
    _unique([alert.id for alert in alerts], "alerts")

    uav_raw = _field(raw, "uav", "")
    uav = UavConfig(
        altitude_m=_number(uav_raw, "altitude_m", "uav"),
        v_max_mps=_number(uav_raw, "v_max_mps", "uav"),
        start_xy_m=_pair(uav_raw, "start_xy_m", "uav"),
        end_xy_m=_pair(uav_raw, "end_xy_m", "uav"),
        total_power_w=_number(uav_raw, "total_power_w", "uav"),
        max_active_downlinks=_integer(uav_raw, "max_active_downlinks", "uav"),
    )
    _require(uav.max_active_downlinks >= 1, "uav.max_active_downlinks", "must be at least 1")

    spectrum_raw = _field(raw, "spectrum", "")
    spectrum = SpectrumConfig(
        b_tot_hz=_number(spectrum_raw, "b_tot_hz", "spectrum"),
        noise_psd_dbm_per_hz=_number(spectrum_raw, "noise_psd_dbm_per_hz", "spectrum"),
    )
    _require(spectrum.b_tot_hz > 0.0, "spectrum.b_tot_hz", "must be positive")

    channel_raw = _field(raw, "channel", "")
    uav_channel_raw = _field(channel_raw, "uav", "channel")
    uav_channel = UavChannelConfig(
        **{
            key: _number(uav_channel_raw, key, "channel.uav")
            for key in ("plos_a", "plos_b", "nlos_attenuation", "alpha_los", "alpha_nlos", "beta0_db")
        }
    )
    ground_raw = _field(channel_raw, "ground", "channel")
    ground_channel = GroundChannelConfig(
        alpha=_number(ground_raw, "alpha", "channel.ground"),
        beta0_db=_number(ground_raw, "beta0_db", "channel.ground"),
    )
    usable_rate_bps = _number(channel_raw, "usable_rate_bps", "channel")
    _require(usable_rate_bps > 0.0, "channel.usable_rate_bps", "must be positive")

    paths_raw = _field(raw, "paths", "")
    paths = PathConfig(
        k=_integer(paths_raw, "k", "paths"),
        max_ground=_integer(paths_raw, "max_ground", "paths"),
        version=_integer(paths_raw, "version", "paths"),
    )
    _require(paths.k >= 1, "paths.k", "must be at least 1")
    _require(0 <= paths.max_ground <= paths.k, "paths.max_ground", "must lie in [0, k]")

    provenance = _field(raw, "provenance", "")
    _require(isinstance(provenance, Mapping), "provenance", "expected an object")
    return Scenario(
        scenario_id=_text(raw, "scenario_id", ""),
        split=split,
        network_id=_text(network, "network_id", "network"),
        nodes=nodes,
        edges=tuple(edges),
        center=center,
        relays=tuple(relays),
        damage_zones=zones,
        sources=sources,
        alerts=tuple(alerts),
        uav=uav,
        time=time,
        spectrum=spectrum,
        uav_channel=uav_channel,
        ground_channel=ground_channel,
        usable_rate_bps=usable_rate_bps,
        source_power_w=_number(raw, "source_power_w", ""),
        paths=paths,
        scenario_seed=_integer(provenance, "scenario_seed", "provenance"),
        provenance=dict(provenance),
        sha256=canonical_sha256(raw),
    )


def load_scenario(path: Path) -> Scenario:
    return scenario_from_dict(json.loads(path.read_text(encoding="utf-8")))
```

- [ ] **Step 4: Run the loader tests and the full suite**

Run: `python -m pytest tests/models/test_scenario.py -v`

Expected: 9 passed.

Run: `python -m pytest -q`

Expected: 38 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/models/__init__.py src/models/rng.py src/models/scenario.py tests/models/scenario_builders.py tests/models/conftest.py tests/models/test_scenario.py
git commit -m "feat: add scenario schema v2 loader"
```

### Task 2: PrLoS UAV channel, ground channel, and realization sampling

**Files:**
- Create: `src/models/channel.py`
- Test: `tests/models/test_channel.py`

**Interfaces:**
- Consumes: `Scenario`, `UAV_ID` from Task 1; `named_rng` from Task 1.
- Produces: `LinkId = tuple[str, str, str]` with kinds `ACCESS = "access"`, `DOWNLINK = "down"`, `BACKHAUL = "backhaul"` (links are `("access", source_id, relay_id | "UAV")`, `("down", "UAV", node_id)`, `("backhaul", tail_node, head_node)`); `db_to_linear(value_db)`, `noise_psd_w_per_hz(scenario)`, `rate_bps(bandwidth_hz: float, snr_per_hz: float) -> float`, `los_probability(horizontal_m, altitude_m, plos_a, plos_b)` (scalar or array), `uav_snr_per_hz(scenario, horizontal_m, power_w) -> (los, nlos)`, `ground_snr_per_hz(scenario, distance_m) -> float`, `ground_range_m(scenario) -> float`, `in_ground_range(scenario, distance_m) -> bool`, frozen dataclass `LinkChannel(los_weight, snr_los_per_hz, snr_nlos_per_hz)` with `LinkChannel.constant(num_slots, snr_per_hz)` and `.rate_bps(slot, bandwidth_hz) -> float`, `all_uav_links(scenario) -> tuple[LinkId, ...]`, `slot_midpoints(trajectory) -> np.ndarray`, `ground_point_ids(scenario) -> tuple[str, ...]`, `channel_uniforms(scenario, realization_id) -> dict[str, np.ndarray]`, and `build_link_channels(scenario, links, trajectory, uniforms: Mapping[str, np.ndarray] | None) -> dict[LinkId, LinkChannel]` (`uniforms=None` gives expected-rate channels).

`snr_per_hz` means `p * g / N0` in Hz, so `rate_bps(b, s) = b * log2(1 + s / b)`. A realized channel stores `los_weight` 1.0 or 0.0 per slot; an expected channel stores the LoS probability. Ground access links are deterministic, so their channel has weight 1 and equal LoS/NLoS SNR.

- [ ] **Step 1: Write the failing channel tests**

```python
# tests/models/test_channel.py
import math

import numpy as np
import pytest

from models.channel import (
    ACCESS,
    DOWNLINK,
    LinkChannel,
    build_link_channels,
    channel_uniforms,
    ground_range_m,
    ground_snr_per_hz,
    los_probability,
    rate_bps,
    uav_snr_per_hz,
)
from models.scenario import UAV_ID, scenario_from_dict


def test_rate_is_zero_without_bandwidth_or_signal():
    assert rate_bps(0.0, 1e6) == 0.0
    assert rate_bps(1e6, 0.0) == 0.0
    assert rate_bps(1e6, 1e6) == pytest.approx(1e6)


def test_los_probability_matches_hand_values():
    assert float(los_probability(0.0, 100.0, 9.61, 0.16)) == pytest.approx(0.999975, abs=1e-5)
    assert float(los_probability(100.0, 100.0, 9.61, 0.16)) == pytest.approx(0.9677, abs=1e-4)


def test_expected_rate_weights_rates_not_gains(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    los, nlos = uav_snr_per_hz(scenario, 300.0, 0.1)
    probability = float(los_probability(300.0, 100.0, 9.61, 0.16))
    channel = LinkChannel(np.array([probability]), np.array([los]), np.array([nlos]))
    expected = probability * rate_bps(1e6, los) + (1 - probability) * rate_bps(1e6, nlos)
    mean_gain_rate = rate_bps(1e6, probability * los + (1 - probability) * nlos)
    assert channel.rate_bps(0, 1e6) == pytest.approx(expected)
    assert channel.rate_bps(0, 1e6) < mean_gain_rate


def test_realized_state_is_los_exactly_when_uniform_below_probability(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    trajectory = np.tile([1000.0, 1000.0], (21, 1))
    link = (ACCESS, "s00", UAV_ID)
    probability = float(los_probability(230.0, 100.0, 9.61, 0.16))
    uniforms = {"s00": np.full(20, probability - 1e-9)}
    uniforms["s00"][3] = probability + 1e-9
    channel = build_link_channels(scenario, [link], trajectory, uniforms)[link]
    assert channel.los_weight[0] == 1.0
    assert channel.los_weight[3] == 0.0
    expected = build_link_channels(scenario, [link], trajectory, None)[link]
    assert expected.los_weight[0] == pytest.approx(probability)


def test_downlink_uses_per_link_uav_power(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    link = (DOWNLINK, UAV_ID, "n1")
    channel = build_link_channels(scenario, [link], np.tile([1000.0, 1000.0], (21, 1)), None)[link]
    los, _ = uav_snr_per_hz(scenario, 200.0, 0.05)
    assert channel.snr_los_per_hz[0] == pytest.approx(los)


def test_ground_range_matches_usable_rate_threshold(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    distance = ground_range_m(scenario)
    assert 340.0 < distance < 380.0
    assert rate_bps(1e6, ground_snr_per_hz(scenario, distance)) == pytest.approx(10000.0, rel=1e-9)


def test_channel_uniforms_are_reproducible_per_realization(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    first = channel_uniforms(scenario, 0)
    assert set(first) == {"s00", "s01", "n0", "n1", "n2", "n3"}
    assert first["s00"].shape == (20,)
    assert np.array_equal(first["n2"], channel_uniforms(scenario, 0)["n2"])
    assert not np.array_equal(first["n2"], channel_uniforms(scenario, 1)["n2"])
```

- [ ] **Step 2: Run the channel tests and verify they fail**

Run: `python -m pytest tests/models/test_channel.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'models.channel'`.

- [ ] **Step 3: Implement the channel module**

```python
# src/models/channel.py
from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np

from .rng import named_rng
from .scenario import Scenario, UAV_ID

LinkId = tuple[str, str, str]
ACCESS = "access"
DOWNLINK = "down"
BACKHAUL = "backhaul"


def db_to_linear(value_db: float) -> float:
    return 10.0 ** (value_db / 10.0)


def noise_psd_w_per_hz(scenario: Scenario) -> float:
    return 10.0 ** ((scenario.spectrum.noise_psd_dbm_per_hz - 30.0) / 10.0)


def rate_bps(bandwidth_hz: float, snr_per_hz: float) -> float:
    """Shannon rate b*log2(1 + s/b), where s = p*g/N0 in Hz; zero bandwidth gives zero rate."""
    if bandwidth_hz <= 0.0 or snr_per_hz <= 0.0:
        return 0.0
    return bandwidth_hz * math.log2(1.0 + snr_per_hz / bandwidth_hz)


def los_probability(horizontal_m, altitude_m: float, plos_a: float, plos_b: float):
    elevation_deg = np.degrees(np.arctan2(altitude_m, horizontal_m))
    return 1.0 / (1.0 + plos_a * np.exp(-plos_b * (elevation_deg - plos_a)))


def uav_snr_per_hz(scenario: Scenario, horizontal_m, power_w: float):
    channel = scenario.uav_channel
    distance = np.hypot(horizontal_m, scenario.uav.altitude_m)
    scale = power_w * db_to_linear(channel.beta0_db) / noise_psd_w_per_hz(scenario)
    los = scale * distance ** (-channel.alpha_los)
    nlos = scale * channel.nlos_attenuation * distance ** (-channel.alpha_nlos)
    return los, nlos


def ground_snr_per_hz(scenario: Scenario, distance_m: float) -> float:
    channel = scenario.ground_channel
    gain = db_to_linear(channel.beta0_db) * max(distance_m, 1.0) ** (-channel.alpha)
    return scenario.source_power_w * gain / noise_psd_w_per_hz(scenario)


def ground_range_m(scenario: Scenario) -> float:
    bandwidth = scenario.spectrum.b_tot_hz
    required_snr = bandwidth * (2.0 ** (scenario.usable_rate_bps / bandwidth) - 1.0)
    channel = scenario.ground_channel
    reference = scenario.source_power_w * db_to_linear(channel.beta0_db) / noise_psd_w_per_hz(scenario)
    return (reference / required_snr) ** (1.0 / channel.alpha)


def in_ground_range(scenario: Scenario, distance_m: float) -> bool:
    return distance_m <= ground_range_m(scenario) * (1.0 + 1e-12)


@dataclass(frozen=True)
class LinkChannel:
    """Per-slot two-state channel: rate = w*R(b, s_los) + (1 - w)*R(b, s_nlos)."""

    los_weight: np.ndarray
    snr_los_per_hz: np.ndarray
    snr_nlos_per_hz: np.ndarray

    @classmethod
    def constant(cls, num_slots: int, snr_per_hz: float) -> "LinkChannel":
        values = np.full(num_slots, float(snr_per_hz))
        return cls(np.ones(num_slots), values, values.copy())

    def rate_bps(self, slot: int, bandwidth_hz: float) -> float:
        weight = float(self.los_weight[slot])
        los = rate_bps(bandwidth_hz, float(self.snr_los_per_hz[slot]))
        nlos = rate_bps(bandwidth_hz, float(self.snr_nlos_per_hz[slot]))
        return weight * los + (1.0 - weight) * nlos


def all_uav_links(scenario: Scenario) -> tuple[LinkId, ...]:
    uplinks = tuple((ACCESS, source.id, UAV_ID) for source in scenario.sources)
    downlinks = tuple((DOWNLINK, UAV_ID, node.id) for node in scenario.nodes)
    return uplinks + downlinks


def slot_midpoints(trajectory: np.ndarray) -> np.ndarray:
    points = np.asarray(trajectory, dtype=float)
    return 0.5 * (points[:-1] + points[1:])


def ground_point_ids(scenario: Scenario) -> tuple[str, ...]:
    return tuple(source.id for source in scenario.sources) + tuple(node.id for node in scenario.nodes)


def channel_uniforms(scenario: Scenario, realization_id: int) -> dict[str, np.ndarray]:
    rng = named_rng(scenario.scenario_seed, f"channel-eval:{realization_id}")
    point_ids = ground_point_ids(scenario)
    draws = rng.random((len(point_ids), scenario.time.num_slots))
    return {point_id: draws[index] for index, point_id in enumerate(point_ids)}


def build_link_channels(
    scenario: Scenario,
    links: Iterable[LinkId],
    trajectory: np.ndarray,
    uniforms: Mapping[str, np.ndarray] | None,
) -> dict[LinkId, LinkChannel]:
    """Build radio channels; uniforms=None yields expected-rate channels."""
    midpoints = slot_midpoints(trajectory)
    num_slots = scenario.time.num_slots
    channel = scenario.uav_channel
    channels: dict[LinkId, LinkChannel] = {}
    for link in links:
        kind, transmitter, receiver = link
        if kind == ACCESS and receiver != UAV_ID:
            distance = math.dist(scenario.position(transmitter), scenario.position(receiver))
            channels[link] = LinkChannel.constant(num_slots, ground_snr_per_hz(scenario, distance))
            continue
        if kind == ACCESS:
            ground_id, power_w = transmitter, scenario.source_power_w
        elif kind == DOWNLINK:
            ground_id, power_w = receiver, scenario.uav.downlink_power_w
        else:
            raise ValueError(f"link {link} has no radio channel")
        horizontal = np.linalg.norm(midpoints - np.asarray(scenario.position(ground_id)), axis=1)
        probability = los_probability(horizontal, scenario.uav.altitude_m, channel.plos_a, channel.plos_b)
        snr_los, snr_nlos = uav_snr_per_hz(scenario, horizontal, power_w)
        weight = probability if uniforms is None else (uniforms[ground_id] < probability).astype(float)
        channels[link] = LinkChannel(np.asarray(weight, dtype=float), snr_los, snr_nlos)
    return channels
```

- [ ] **Step 4: Run the channel tests and the full suite**

Run: `python -m pytest tests/models/test_channel.py -v`

Expected: 7 passed.

Run: `python -m pytest -q`

Expected: 45 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/models/channel.py tests/models/test_channel.py
git commit -m "feat: add PrLoS and ground channel models"
```

### Task 3: Backhaul routes and candidate paths

**Files:**
- Create: `src/models/paths.py`
- Test: `tests/models/test_paths.py`

**Interfaces:**
- Consumes: `Scenario`, `UAV_ID` (Task 1); `ACCESS`, `DOWNLINK`, `BACKHAUL`, `LinkId`, `ground_snr_per_hz`, `in_ground_range`, `rate_bps` (Task 2).
- Produces: `GROUND = "ground"`, `VIA_UAV = "uav"`; frozen dataclass `CandidatePath(kind: str, entry: str, links: tuple[LinkId, ...], cost_s: float)` with `.holders -> tuple[str, ...]` (the transmitter of each link); frozen dataclass `BackhaulRoutes(center, next_hop: Mapping[str, str], hop_count: Mapping[str, int])` with `.route_links(node_id) -> tuple[LinkId, ...]`; `backhaul_routes(scenario) -> BackhaulRoutes`; `candidate_paths(scenario, routes=None) -> dict[str, tuple[CandidatePath, ...]]` keyed by alert ID; `radio_links(candidates) -> tuple[LinkId, ...]` (sorted access and downlink links of all candidates); `ground_connected_source_fraction(scenario, routes=None) -> float`.

Ground entry nodes include the center itself when a source is within ground range of it, matching the UAV rule that the center is a valid drop node. Candidate order is the ground paths by `(cost_s, entry)` followed by the UAV paths by `(cost_s, entry)`.

- [ ] **Step 1: Write the failing path tests**

```python
# tests/models/test_paths.py
from models.paths import (
    GROUND,
    VIA_UAV,
    backhaul_routes,
    candidate_paths,
    ground_connected_source_fraction,
    radio_links,
)
from models.scenario import UAV_ID, scenario_from_dict
from scenario_builders import make_alert


def test_routes_use_fewest_hops_then_length_then_node_id(raw_scenario):
    raw_scenario["network"]["nodes"].append({"id": "n4", "x_m": 1200.0, "y_m": 1400.0})
    raw_scenario["network"]["edges"] += [
        {"source": "n4", "target": "n1", "failed": False, "capacity_bps": 1e6},
        {"source": "n4", "target": "n2", "failed": False, "capacity_bps": 1e6},
    ]
    routes = backhaul_routes(scenario_from_dict(raw_scenario))
    assert routes.hop_count == {"n0": 0, "n1": 1, "n2": 1, "n3": 2, "n4": 2}
    assert routes.next_hop["n4"] == "n1"
    assert routes.route_links("n3") == (("backhaul", "n3", "n1"), ("backhaul", "n1", "n0"))


def test_candidates_mix_ground_and_uav_paths(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    paths = candidate_paths(scenario)["alert-0000"]
    assert [(path.kind, path.entry) for path in paths] == [(GROUND, "n1"), (GROUND, "n0"), (VIA_UAV, "n1")]
    assert paths[0].links == (("access", "s00", "n1"), ("backhaul", "n1", "n0"))
    assert paths[2].links == (("access", "s00", UAV_ID), ("down", UAV_ID, "n1"), ("backhaul", "n1", "n0"))
    assert paths[2].holders == ("s00", UAV_ID, "n1")


def test_far_source_gets_only_uav_paths_and_center_drop_exists(raw_scenario):
    raw_scenario["alerts"].append(make_alert("alert-0001", "s01", 0, 10, 1000))
    paths = candidate_paths(scenario_from_dict(raw_scenario))["alert-0001"]
    assert [path.kind for path in paths] == [VIA_UAV, VIA_UAV, VIA_UAV]
    assert paths[0].entry == "n0"


def test_failed_edges_remove_isolated_relays(raw_scenario):
    raw_scenario["network"]["edges"][0].update(failed=True, capacity_bps=0.0)
    scenario = scenario_from_dict(raw_scenario)
    assert set(backhaul_routes(scenario).hop_count) == {"n0", "n2"}
    paths = candidate_paths(scenario)["alert-0000"]
    assert all(path.entry in {"n0", "n2"} for path in paths)
    assert [(path.kind, path.entry) for path in paths] == [(GROUND, "n0"), (VIA_UAV, "n0"), (VIA_UAV, "n2")]


def test_radio_links_and_ground_connected_fraction(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    links = radio_links(candidate_paths(scenario))
    assert ("access", "s00", "n1") in links
    assert all(link[0] != "backhaul" for link in links)
    assert ground_connected_source_fraction(scenario) == 0.5
```

- [ ] **Step 2: Run the path tests and verify they fail**

Run: `python -m pytest tests/models/test_paths.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'models.paths'`.

- [ ] **Step 3: Implement routes and candidate generation**

```python
# src/models/paths.py
from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping

from .channel import ACCESS, BACKHAUL, DOWNLINK, LinkId, ground_snr_per_hz, in_ground_range, rate_bps
from .scenario import Scenario, UAV_ID

GROUND = "ground"
VIA_UAV = "uav"


@dataclass(frozen=True)
class CandidatePath:
    kind: str
    entry: str
    links: tuple[LinkId, ...]
    cost_s: float

    @property
    def holders(self) -> tuple[str, ...]:
        return tuple(link[1] for link in self.links)


@dataclass(frozen=True)
class BackhaulRoutes:
    center: str
    next_hop: Mapping[str, str]
    hop_count: Mapping[str, int]

    def route_links(self, node_id: str) -> tuple[LinkId, ...]:
        links = []
        current = node_id
        while current != self.center:
            following = self.next_hop[current]
            links.append((BACKHAUL, current, following))
            current = following
        return tuple(links)


def backhaul_routes(scenario: Scenario) -> BackhaulRoutes:
    neighbours: dict[str, set[str]] = {node.id: set() for node in scenario.nodes}
    for edge in scenario.edges:
        if edge.alive:
            neighbours[edge.source].add(edge.target)
            neighbours[edge.target].add(edge.source)
    hop_count = {scenario.center: 0}
    queue = deque([scenario.center])
    while queue:
        current = queue.popleft()
        for other in sorted(neighbours[current]):
            if other not in hop_count:
                hop_count[other] = hop_count[current] + 1
                queue.append(other)
    length = {scenario.center: 0.0}
    next_hop: dict[str, str] = {}
    for node_id in sorted(hop_count, key=lambda item: (hop_count[item], item)):
        if node_id == scenario.center:
            continue
        options = [
            (math.dist(scenario.position(node_id), scenario.position(other)) + length[other], other)
            for other in neighbours[node_id]
            if hop_count.get(other) == hop_count[node_id] - 1
        ]
        length[node_id], next_hop[node_id] = min(options)
    return BackhaulRoutes(center=scenario.center, next_hop=next_hop, hop_count=hop_count)


def _backhaul_delay_s(scenario: Scenario, route: tuple[LinkId, ...], size_bits: int) -> float:
    return sum(
        size_bits / scenario.backhaul_capacity_bps(link[1], link[2]) + scenario.time.slot_s for link in route
    )


def candidate_paths(
    scenario: Scenario, routes: BackhaulRoutes | None = None
) -> dict[str, tuple[CandidatePath, ...]]:
    routes = routes or backhaul_routes(scenario)
    entries = sorted(routes.hop_count)
    bandwidth = scenario.spectrum.b_tot_hz
    candidates: dict[str, tuple[CandidatePath, ...]] = {}
    for alert in scenario.alerts:
        source = scenario.source_by_id[alert.source]
        ground: list[CandidatePath] = []
        aerial: list[CandidatePath] = []
        for node_id in entries:
            route = routes.route_links(node_id)
            backhaul_s = _backhaul_delay_s(scenario, route, alert.size_bits)
            distance = math.dist(source.xy, scenario.node_by_id[node_id].xy)
            if in_ground_range(scenario, distance):
                access_rate = rate_bps(bandwidth, ground_snr_per_hz(scenario, distance))
                ground.append(
                    CandidatePath(
                        kind=GROUND,
                        entry=node_id,
                        links=((ACCESS, source.id, node_id), *route),
                        cost_s=alert.size_bits / access_rate + backhaul_s,
                    )
                )
            aerial.append(
                CandidatePath(
                    kind=VIA_UAV,
                    entry=node_id,
                    links=((ACCESS, source.id, UAV_ID), (DOWNLINK, UAV_ID, node_id), *route),
                    cost_s=distance / scenario.uav.v_max_mps + backhaul_s,
                )
            )
        ground.sort(key=lambda path: (path.cost_s, path.entry))
        aerial.sort(key=lambda path: (path.cost_s, path.entry))
        chosen = ground[: scenario.paths.max_ground]
        candidates[alert.id] = tuple(chosen + aerial[: scenario.paths.k - len(chosen)])
    return candidates


def radio_links(candidates: Mapping[str, tuple[CandidatePath, ...]]) -> tuple[LinkId, ...]:
    return tuple(
        sorted(
            {
                link
                for paths in candidates.values()
                for path in paths
                for link in path.links
                if link[0] != BACKHAUL
            }
        )
    )


def ground_connected_source_fraction(scenario: Scenario, routes: BackhaulRoutes | None = None) -> float:
    routes = routes or backhaul_routes(scenario)
    connected = sum(
        1
        for source in scenario.sources
        if any(
            in_ground_range(scenario, math.dist(source.xy, scenario.node_by_id[node_id].xy))
            for node_id in routes.hop_count
        )
    )
    return connected / len(scenario.sources)
```

- [ ] **Step 4: Run the path tests and the full suite**

Run: `python -m pytest tests/models/test_paths.py -v`

Expected: 5 passed.

Run: `python -m pytest -q`

Expected: 50 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/models/paths.py tests/models/test_paths.py
git commit -m "feat: generate candidate relay paths"
```

### Task 4: Plans, bandwidth policies, and input validation

**Files:**
- Create: `src/models/plan.py`
- Test: `tests/models/test_plan.py`

**Interfaces:**
- Consumes: `Scenario` (Task 1); `ACCESS`, `LinkId` (Task 2); `CandidatePath`, `radio_links` (Task 3).
- Produces: frozen dataclasses `EqualSplitBacklogged()`, `StaticSchedule(bandwidth_hz: Mapping[LinkId, np.ndarray])` with `.at(link, slot) -> float` (missing links give 0), and `Plan(trajectory: np.ndarray, path_choice: Mapping[str, int | None], bandwidth: EqualSplitBacklogged | StaticSchedule)`; `stationary_trajectory(scenario) -> np.ndarray` of shape `(N + 1, 2)` at the start point; `chosen_paths(scenario, candidates, plan) -> dict[str, CandidatePath | None]`; `validate_plan(scenario, candidates, plan) -> list[str]` (empty when valid). Tolerances: `POSITION_TOLERANCE_M = 1e-6`, `RELATIVE_TOLERANCE = 1e-9`.

- [ ] **Step 1: Write the failing plan tests**

```python
# tests/models/test_plan.py
import numpy as np

from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, StaticSchedule, stationary_trajectory, validate_plan
from models.scenario import UAV_ID, scenario_from_dict


def _setup(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    return scenario, candidate_paths(scenario)


def test_stationary_equal_split_plan_is_valid(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0}, EqualSplitBacklogged())
    assert validate_plan(scenario, candidates, plan) == []


def test_trajectory_endpoint_and_speed_violations(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    trajectory = stationary_trajectory(scenario)
    trajectory[0] = [1000.0, 1001.0]
    trajectory[5] = [1100.0, 1000.0]
    violations = validate_plan(scenario, candidates, Plan(trajectory, {"alert-0000": 0}, EqualSplitBacklogged()))
    assert any("q[0] is not the start point" in item for item in violations)
    assert any("slot 4 moves" in item for item in violations)
    assert any("slot 5 moves" in item for item in violations)


def test_path_choice_violations(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    trajectory = stationary_trajectory(scenario)
    assert "path_choice: missing alert alert-0000" in validate_plan(
        scenario, candidates, Plan(trajectory, {}, EqualSplitBacklogged())
    )
    assert any(
        "invalid candidate index 3" in item
        for item in validate_plan(scenario, candidates, Plan(trajectory, {"alert-0000": 3}, EqualSplitBacklogged()))
    )


def test_static_schedule_violations(raw_scenario):
    scenario, candidates = _setup(raw_scenario)
    trajectory = stationary_trajectory(scenario)
    schedule = StaticSchedule(
        {
            ("access", "s00", "n1"): np.full(20, 6e5),
            ("access", "s00", "n0"): np.full(20, 6e5),
            ("down", UAV_ID, "n1"): np.full(20, -1.0),
            ("down", UAV_ID, "n9"): np.zeros(20),
        }
    )
    violations = validate_plan(scenario, candidates, Plan(trajectory, {"alert-0000": 0}, schedule))
    assert any("slot 0 access total" in item for item in violations)
    assert any("finite non-negative" in item for item in violations)
    assert any("unknown radio link" in item for item in violations)


def test_static_schedule_downlink_cap(raw_scenario):
    raw_scenario["uav"]["max_active_downlinks"] = 1
    raw_scenario["paths"]["max_ground"] = 0
    scenario, candidates = _setup(raw_scenario)
    downlinks = [link for path in candidates["alert-0000"] for link in path.links if link[0] == "down"]
    schedule = StaticSchedule({link: np.full(20, 1e5) for link in downlinks[:2]})
    violations = validate_plan(
        scenario, candidates, Plan(stationary_trajectory(scenario), {"alert-0000": 0}, schedule)
    )
    assert any("has 2 active downlinks" in item for item in violations)
```

- [ ] **Step 2: Run the plan tests and verify they fail**

Run: `python -m pytest tests/models/test_plan.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'models.plan'`.

- [ ] **Step 3: Implement plans and validation**

```python
# src/models/plan.py
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .channel import ACCESS, LinkId
from .paths import CandidatePath, radio_links
from .scenario import Scenario

POSITION_TOLERANCE_M = 1e-6
RELATIVE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class EqualSplitBacklogged:
    """Split B_tot equally among backlogged radio links in each half-slot."""


@dataclass(frozen=True)
class StaticSchedule:
    """Offline bandwidth b_e[n] in Hz; links absent from the mapping get zero."""

    bandwidth_hz: Mapping[LinkId, np.ndarray]

    def at(self, link: LinkId, slot: int) -> float:
        values = self.bandwidth_hz.get(link)
        return 0.0 if values is None else float(values[slot])


@dataclass(frozen=True)
class Plan:
    trajectory: np.ndarray
    path_choice: Mapping[str, int | None]
    bandwidth: EqualSplitBacklogged | StaticSchedule


def stationary_trajectory(scenario: Scenario) -> np.ndarray:
    start = np.asarray(scenario.uav.start_xy_m, dtype=float)
    return np.tile(start, (scenario.time.num_slots + 1, 1))


def chosen_paths(
    scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]], plan: Plan
) -> dict[str, CandidatePath | None]:
    return {
        alert.id: None if plan.path_choice[alert.id] is None else candidates[alert.id][plan.path_choice[alert.id]]
        for alert in scenario.alerts
    }


def validate_plan(
    scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]], plan: Plan
) -> list[str]:
    violations = _trajectory_violations(scenario, plan.trajectory)
    alert_ids = {alert.id for alert in scenario.alerts}
    for alert_id in sorted(alert_ids - set(plan.path_choice)):
        violations.append(f"path_choice: missing alert {alert_id}")
    for alert_id in sorted(plan.path_choice):
        index = plan.path_choice[alert_id]
        if alert_id not in alert_ids:
            violations.append(f"path_choice: unknown alert {alert_id}")
            continue
        valid_index = (
            isinstance(index, (int, np.integer))
            and not isinstance(index, bool)
            and 0 <= int(index) < len(candidates[alert_id])
        )
        if index is not None and not valid_index:
            violations.append(f"path_choice: alert {alert_id} has invalid candidate index {index}")
    if isinstance(plan.bandwidth, StaticSchedule):
        violations.extend(_schedule_violations(scenario, candidates, plan.bandwidth))
    elif not isinstance(plan.bandwidth, EqualSplitBacklogged):
        violations.append("bandwidth: unknown policy")
    return violations


def _trajectory_violations(scenario: Scenario, trajectory: np.ndarray) -> list[str]:
    num_slots = scenario.time.num_slots
    points = np.asarray(trajectory, dtype=float)
    if points.shape != (num_slots + 1, 2) or not np.all(np.isfinite(points)):
        return [f"trajectory: expected finite shape {(num_slots + 1, 2)}, got {points.shape}"]
    violations = []
    for label, index, target in (("start", 0, scenario.uav.start_xy_m), ("end", num_slots, scenario.uav.end_xy_m)):
        if math.dist(points[index], target) > POSITION_TOLERANCE_M:
            violations.append(f"trajectory: q[{index}] is not the {label} point")
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    limit = scenario.uav.v_max_mps * scenario.time.slot_s * (1.0 + RELATIVE_TOLERANCE)
    for slot in np.flatnonzero(steps > limit):
        violations.append(f"trajectory: slot {slot} moves {steps[slot]:.3f} m, limit {limit:.3f} m")
    return violations


def _schedule_violations(
    scenario: Scenario, candidates: Mapping[str, tuple[CandidatePath, ...]], schedule: StaticSchedule
) -> list[str]:
    num_slots = scenario.time.num_slots
    known = set(radio_links(candidates))
    access_total = np.zeros(num_slots)
    downlink_total = np.zeros(num_slots)
    active_downlinks = np.zeros(num_slots, dtype=int)
    violations = []
    for link in sorted(schedule.bandwidth_hz):
        if link not in known:
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

- [ ] **Step 4: Run the plan tests and the full suite**

Run: `python -m pytest tests/models/test_plan.py -v`

Expected: 5 passed.

Run: `python -m pytest -q`

Expected: 55 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/models/plan.py tests/models/test_plan.py
git commit -m "feat: add plans and plan validation"
```

### Task 5: Slot simulator with EDF dispatcher and bit ledger

**Files:**
- Create: `src/models/ledger.py`
- Create: `src/models/simulator.py`
- Create: `tests/models/channel_builders.py`
- Test: `tests/models/test_simulator.py`

**Interfaces:**
- Consumes: `Scenario` (Task 1); `ACCESS`, `BACKHAUL`, `DOWNLINK`, `LinkChannel`, `LinkId`, `all_uav_links` (Task 2); `CandidatePath`, `radio_links` (Task 3); `Plan`, `StaticSchedule`, `chosen_paths` (Task 4).
- Produces: frozen dataclass `Ledger(realization_id: int | None, transfers: tuple[(slot, link, alert_id, bits), ...], bandwidth: tuple[(slot, link, hz), ...], capacity_bits: tuple[(slot, link, bits), ...], drops: tuple[(slot, alert_id, holder, bits), ...])` in `models.ledger`; frozen dataclass `SimulationOutcome(delivered_bits: dict[str, float], delivery_slot: dict[str, int | None], ledger: Ledger)`; `simulate(scenario, candidates, plan, channels: Mapping[LinkId, LinkChannel], realization_id=None) -> SimulationOutcome`. Test helper `channel_builders.constant_channels(scenario, candidates, snr_per_hz, overrides=None) -> dict[LinkId, LinkChannel]`.

With `snr_per_hz = 1e6` and the full 1 MHz band, a link carries exactly 1,000,000 bit/s, so an access or downlink half-slot carries 500,000 bits. The hand-computed expectations in the tests use this. Buffers are keyed by `(alert_id, hop_index)`; bits at hop `i` wait to cross `path.links[i]`, and bits crossing the last link are delivered.

- [ ] **Step 1: Write the channel helper and the failing simulator tests**

```python
# tests/models/channel_builders.py
from models.channel import LinkChannel, all_uav_links
from models.paths import radio_links


def constant_channels(scenario, candidates, snr_per_hz, overrides=None):
    links = set(radio_links(candidates)) | set(all_uav_links(scenario))
    channels = {link: LinkChannel.constant(scenario.time.num_slots, snr_per_hz) for link in links}
    channels.update(overrides or {})
    return channels
```

```python
# tests/models/test_simulator.py
import pytest

from models.channel import LinkChannel
from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, StaticSchedule, stationary_trajectory
from models.scenario import UAV_ID, scenario_from_dict
from models.simulator import simulate
from channel_builders import constant_channels
from scenario_builders import make_alert


def _run(raw_scenario, choice, snr_per_hz=1e6, bandwidth=None, overrides=None):
    scenario = scenario_from_dict(raw_scenario)
    candidates = candidate_paths(scenario)
    plan = Plan(stationary_trajectory(scenario), choice, bandwidth or EqualSplitBacklogged())
    channels = constant_channels(scenario, candidates, snr_per_hz, overrides)
    return scenario, candidates, simulate(scenario, candidates, plan, channels)


def _transfers(outcome, alert_id):
    return [(slot, link, bits) for slot, link, alert, bits in outcome.ledger.transfers if alert == alert_id]


def test_ground_path_store_and_forward_meets_deadline_boundary(raw_scenario):
    _, candidates, outcome = _run(raw_scenario, {"alert-0000": 0})
    assert candidates["alert-0000"][0].entry == "n1"
    assert _transfers(outcome, "alert-0000") == [
        (0, ("access", "s00", "n1"), 500000.0),
        (1, ("access", "s00", "n1"), 500000.0),
        (1, ("backhaul", "n1", "n0"), 500000.0),
        (2, ("backhaul", "n1", "n0"), 500000.0),
    ]
    assert outcome.delivered_bits["alert-0000"] == 1000000.0
    assert outcome.delivery_slot["alert-0000"] == 2


def test_arrival_in_deadline_slot_is_late_and_expired_bits_drop(raw_scenario):
    raw_scenario["alerts"][0]["deadline_slot"] = 2
    _, _, outcome = _run(raw_scenario, {"alert-0000": 0})
    assert outcome.delivered_bits["alert-0000"] == 500000.0
    assert outcome.delivery_slot["alert-0000"] is None
    assert outcome.ledger.drops == ((2, "alert-0000", "n1", 500000.0),)
    assert all(slot < 2 for slot, _, _ in _transfers(outcome, "alert-0000"))


def test_zero_bandwidth_and_zero_power_deliver_nothing(raw_scenario):
    _, _, no_band = _run(raw_scenario, {"alert-0000": 0}, bandwidth=StaticSchedule({}))
    assert no_band.ledger.transfers == ()
    assert no_band.delivered_bits["alert-0000"] == 0.0
    _, _, no_power = _run(raw_scenario, {"alert-0000": 0}, snr_per_hz=0.0)
    assert no_power.ledger.transfers == ()


def test_no_transmission_before_release(raw_scenario):
    raw_scenario["alerts"][0].update(release_slot=5, deadline_slot=12)
    _, _, outcome = _run(raw_scenario, {"alert-0000": 0})
    assert min(slot for slot, _, _ in _transfers(outcome, "alert-0000")) == 5
    assert outcome.delivery_slot["alert-0000"] == 7


def test_unserved_alert_never_enters_the_network(raw_scenario):
    _, _, outcome = _run(raw_scenario, {"alert-0000": None})
    assert outcome.ledger.transfers == ()
    assert outcome.ledger.drops == ()


def test_backhaul_bottleneck_serves_earliest_deadline_first(raw_scenario):
    raw_scenario["network"]["edges"][0]["capacity_bps"] = 250000.0
    raw_scenario["alerts"] = [
        make_alert("alert-0000", "s00", 0, 10, 500000),
        make_alert("alert-0001", "s00", 0, 5, 500000),
    ]
    _, candidates, outcome = _run(raw_scenario, {"alert-0000": 0, "alert-0001": 0})
    assert candidates["alert-0001"][0].entry == "n1"
    assert outcome.delivery_slot == {"alert-0000": 4, "alert-0001": 2}
    assert _transfers(outcome, "alert-0001")[0] == (0, ("access", "s00", "n1"), 500000.0)
    assert _transfers(outcome, "alert-0000")[0] == (1, ("access", "s00", "n1"), 500000.0)


def test_disconnected_source_uses_uav_store_carry_forward(raw_scenario):
    raw_scenario["alerts"] = [make_alert("alert-0001", "s01", 0, 10, 1000000)]
    _, candidates, outcome = _run(raw_scenario, {"alert-0001": 0})
    assert candidates["alert-0001"][0].links == (("access", "s01", UAV_ID), ("down", UAV_ID, "n0"))
    assert _transfers(outcome, "alert-0001") == [
        (0, ("access", "s01", UAV_ID), 500000.0),
        (1, ("access", "s01", UAV_ID), 500000.0),
        (1, ("down", UAV_ID, "n0"), 500000.0),
        (2, ("down", UAV_ID, "n0"), 500000.0),
    ]
    assert outcome.delivered_bits["alert-0001"] == 1000000.0


def test_equal_split_serves_at_most_two_downlinks_by_deadline(raw_scenario):
    raw_scenario["alerts"] = [
        make_alert("alert-0000", "s01", 0, 15, 500000),
        make_alert("alert-0001", "s01", 0, 5, 500000),
        make_alert("alert-0002", "s01", 0, 10, 500000),
    ]
    scenario = scenario_from_dict(raw_scenario)
    candidates = candidate_paths(scenario)
    entries = [path.entry for path in candidates["alert-0000"]]
    assert entries == ["n0", "n1", "n2"]
    uplink = ("access", "s01", UAV_ID)
    _, _, outcome = _run(
        raw_scenario,
        {"alert-0000": 0, "alert-0001": 1, "alert-0002": 2},
        overrides={uplink: LinkChannel.constant(20, 1e9)},
    )
    slot_one = [(link, hz) for slot, link, hz in outcome.ledger.bandwidth if slot == 1 and link[0] == "down"]
    assert slot_one == [(("down", UAV_ID, "n1"), 500000.0), (("down", UAV_ID, "n2"), 500000.0)]
```

- [ ] **Step 2: Run the simulator tests and verify they fail**

Run: `python -m pytest tests/models/test_simulator.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'models.simulator'`.

- [ ] **Step 3: Implement the ledger and simulator**

```python
# src/models/ledger.py
from dataclasses import dataclass

from .channel import LinkId


@dataclass(frozen=True)
class Ledger:
    """Everything the simulator did, in a form the independent checker can replay."""

    realization_id: int | None
    transfers: tuple[tuple[int, LinkId, str, float], ...]
    bandwidth: tuple[tuple[int, LinkId, float], ...]
    capacity_bits: tuple[tuple[int, LinkId, float], ...]
    drops: tuple[tuple[int, str, str, float], ...]
```

```python
# src/models/simulator.py
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

from .channel import ACCESS, BACKHAUL, DOWNLINK, LinkChannel, LinkId
from .ledger import Ledger
from .paths import CandidatePath
from .plan import Plan, StaticSchedule, chosen_paths
from .scenario import Scenario

DELIVERY_TOLERANCE_BITS = 1e-6


@dataclass(frozen=True)
class SimulationOutcome:
    delivered_bits: dict[str, float]
    delivery_slot: dict[str, int | None]
    ledger: Ledger


def simulate(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    plan: Plan,
    channels: Mapping[LinkId, LinkChannel],
    realization_id: int | None = None,
) -> SimulationOutcome:
    slot_s = scenario.time.slot_s
    access_s = scenario.time.access_fraction * slot_s
    downlink_s = slot_s - access_s
    paths = chosen_paths(scenario, candidates, plan)
    served = [alert for alert in scenario.alerts if paths[alert.id] is not None]
    buffers: dict[tuple[str, int], float] = {}
    delivered = {alert.id: 0.0 for alert in scenario.alerts}
    delivery_slot: dict[str, int | None] = {alert.id: None for alert in scenario.alerts}
    transfers: list[tuple[int, LinkId, str, float]] = []
    bandwidth_log: list[tuple[int, LinkId, float]] = []
    capacity_log: list[tuple[int, LinkId, float]] = []
    drops: list[tuple[int, str, str, float]] = []

    for slot in range(scenario.time.num_slots):
        for alert in served:
            links = paths[alert.id].links
            if slot >= alert.deadline_slot:
                for hop in range(len(links)):
                    bits = buffers.pop((alert.id, hop), 0.0)
                    if bits > 0.0:
                        drops.append((slot, alert.id, links[hop][1], bits))
            elif slot == alert.release_slot:
                buffers[(alert.id, 0)] = float(alert.size_bits)

        queues: dict[LinkId, list[tuple[int, str, int]]] = defaultdict(list)
        for (alert_id, hop), bits in buffers.items():
            if bits > 0.0:
                deadline = scenario.alert_by_id[alert_id].deadline_slot
                queues[paths[alert_id].links[hop]].append((deadline, alert_id, hop))

        allocation = _allocate_bandwidth(scenario, plan, queues, slot)
        for link in sorted(allocation):
            if allocation[link] > 0.0:
                bandwidth_log.append((slot, link, allocation[link]))

        arrivals: list[tuple[str, int, float]] = []
        for link in sorted(queues):
            if link[0] == BACKHAUL:
                capacity = slot_s * scenario.backhaul_capacity_bps(link[1], link[2])
            else:
                duration = access_s if link[0] == ACCESS else downlink_s
                capacity = duration * channels[link].rate_bps(slot, allocation.get(link, 0.0))
            capacity_log.append((slot, link, capacity))
            remaining = capacity
            for _, alert_id, hop in sorted(queues[link]):
                if remaining <= 0.0:
                    break
                sent = min(buffers[(alert_id, hop)], remaining)
                buffers[(alert_id, hop)] -= sent
                remaining -= sent
                transfers.append((slot, link, alert_id, sent))
                arrivals.append((alert_id, hop + 1, sent))

        for alert_id, hop, bits in arrivals:
            if hop == len(paths[alert_id].links):
                delivered[alert_id] += bits
                size = scenario.alert_by_id[alert_id].size_bits
                if delivery_slot[alert_id] is None and delivered[alert_id] >= size - DELIVERY_TOLERANCE_BITS:
                    delivery_slot[alert_id] = slot
            else:
                buffers[(alert_id, hop)] = buffers.get((alert_id, hop), 0.0) + bits
        buffers = {key: bits for key, bits in buffers.items() if bits > 0.0}

    ledger = Ledger(
        realization_id=realization_id,
        transfers=tuple(transfers),
        bandwidth=tuple(bandwidth_log),
        capacity_bits=tuple(capacity_log),
        drops=tuple(drops),
    )
    return SimulationOutcome(delivered_bits=delivered, delivery_slot=delivery_slot, ledger=ledger)


def _allocate_bandwidth(
    scenario: Scenario,
    plan: Plan,
    queues: Mapping[LinkId, list[tuple[int, str, int]]],
    slot: int,
) -> dict[LinkId, float]:
    if isinstance(plan.bandwidth, StaticSchedule):
        return {link: plan.bandwidth.at(link, slot) for link in plan.bandwidth.bandwidth_hz}
    total = scenario.spectrum.b_tot_hz
    access = sorted(link for link in queues if link[0] == ACCESS)
    ranked_downlinks = sorted(
        (min(entry[0] for entry in queues[link]), link[2], link) for link in queues if link[0] == DOWNLINK
    )
    active = [item[2] for item in ranked_downlinks[: scenario.uav.max_active_downlinks]]
    allocation = {link: total / len(access) for link in access}
    allocation.update({link: total / len(active) for link in active})
    return allocation
```

- [ ] **Step 4: Run the simulator tests and the full suite**

Run: `python -m pytest tests/models/test_simulator.py -v`

Expected: 8 passed.

Run: `python -m pytest -q`

Expected: 63 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/models/ledger.py src/models/simulator.py tests/models/channel_builders.py tests/models/test_simulator.py
git commit -m "feat: simulate slot flows with EDF dispatcher"
```

### Task 6: Independent ledger checker

**Files:**
- Create: `src/models/checker.py`
- Test: `tests/models/test_checker.py`

**Interfaces:**
- Consumes: `Scenario` (Task 1); `ACCESS`, `BACKHAUL`, `DOWNLINK`, `LinkChannel`, `LinkId` (Task 2); `CandidatePath` (Task 3); `Plan`, `StaticSchedule` (Task 4); `Ledger` (Task 5). The tests also use `simulate` to produce valid ledgers.
- Produces: `SimulatorInvariantError(violations: list[str])` with `.violations: tuple[str, ...]`; `check_ledger(scenario, candidates, plan, channels, ledger, delivered_bits: Mapping[str, float] | None = None) -> list[str]`.

The checker resolves chosen paths itself and replays per-alert buffers from the ledger alone. Replay with non-negative buffers is the per-node flow conservation check. It does not import `models.simulator`.

- [ ] **Step 1: Write the failing checker tests**

```python
# tests/models/test_checker.py
from dataclasses import replace

from models.checker import check_ledger
from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import scenario_from_dict
from models.simulator import simulate
from channel_builders import constant_channels


def _valid_run(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    candidates = candidate_paths(scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0}, EqualSplitBacklogged())
    channels = constant_channels(scenario, candidates, 1e6)
    outcome = simulate(scenario, candidates, plan, channels)
    return scenario, candidates, plan, channels, outcome


def _with_transfers(ledger, transfers):
    return replace(ledger, transfers=tuple(transfers))


def test_valid_ledger_has_no_violations(raw_scenario):
    scenario, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    assert check_ledger(scenario, candidates, plan, channels, outcome.ledger, outcome.delivered_bits) == []


def test_detects_over_capacity_transfer(raw_scenario):
    scenario, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    transfers = list(outcome.ledger.transfers)
    slot, link, alert, _ = transfers[0]
    transfers[0] = (slot, link, alert, 900000.0)
    violations = check_ledger(scenario, candidates, plan, channels, _with_transfers(outcome.ledger, transfers))
    assert any("over capacity" in item for item in violations)


def test_detects_drop_larger_than_buffer(raw_scenario):
    scenario, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    ledger = replace(outcome.ledger, drops=((12, "alert-0000", "n1", 1000000.0),))
    violations = check_ledger(scenario, candidates, plan, channels, ledger)
    assert any("drops 1000000.000 bits at hop 1 holding only 0.000" in item for item in violations)


def test_detects_off_path_flow(raw_scenario):
    scenario, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    transfers = list(outcome.ledger.transfers)
    slot, _, alert, bits = transfers[2]
    transfers[2] = (slot, ("backhaul", "n2", "n0"), alert, bits)
    violations = check_ledger(scenario, candidates, plan, channels, _with_transfers(outcome.ledger, transfers))
    assert any("outside its chosen path" in item for item in violations)


def test_detects_transmission_before_release(raw_scenario):
    _, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    raw_scenario["alerts"][0]["release_slot"] = 1
    shifted = scenario_from_dict(raw_scenario)
    violations = check_ledger(shifted, candidates, plan, channels, outcome.ledger)
    assert any("transmits before release" in item for item in violations)


def test_detects_same_slot_forwarding(raw_scenario):
    scenario, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    transfers = list(outcome.ledger.transfers)
    _, link, alert, bits = transfers[3]
    transfers[3] = (1, link, alert, bits)
    violations = check_ledger(scenario, candidates, plan, channels, _with_transfers(outcome.ledger, transfers))
    assert any("sends 1000000.000 bits at hop 1 holding only 500000.000" in item for item in violations)


def test_detects_reported_delivery_mismatch(raw_scenario):
    scenario, candidates, plan, channels, outcome = _valid_run(raw_scenario)
    violations = check_ledger(scenario, candidates, plan, channels, outcome.ledger, {"alert-0000": 0.0})
    assert any("reported 0.000 delivered bits" in item for item in violations)
```

- [ ] **Step 2: Run the checker tests and verify they fail**

Run: `python -m pytest tests/models/test_checker.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'models.checker'`.

- [ ] **Step 3: Implement the checker**

```python
# src/models/checker.py
from collections import defaultdict
import math
from typing import Mapping

from .channel import ACCESS, BACKHAUL, DOWNLINK, LinkChannel, LinkId
from .ledger import Ledger
from .paths import CandidatePath
from .plan import Plan, StaticSchedule
from .scenario import Scenario

ABSOLUTE_TOLERANCE_BITS = 1e-6
RELATIVE_TOLERANCE = 1e-9


class SimulatorInvariantError(RuntimeError):
    def __init__(self, violations: list[str]) -> None:
        super().__init__("simulator invariant violated: " + "; ".join(violations[:5]))
        self.violations = tuple(violations)


def check_ledger(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    plan: Plan,
    channels: Mapping[LinkId, LinkChannel],
    ledger: Ledger,
    delivered_bits: Mapping[str, float] | None = None,
) -> list[str]:
    """Replay the ledger without simulator code and list every broken invariant."""
    paths = {
        alert.id: None if plan.path_choice[alert.id] is None else candidates[alert.id][plan.path_choice[alert.id]]
        for alert in scenario.alerts
    }
    violations: list[str] = []
    bandwidth = _check_bandwidth(scenario, plan, ledger, violations)
    capacity = _check_capacity(scenario, channels, ledger, bandwidth, violations)
    flows = _check_transfers(ledger, paths, capacity, violations)
    drops: dict[tuple[str, int, int], float] = defaultdict(float)
    for slot, alert_id, holder, bits in ledger.drops:
        path = paths.get(alert_id)
        if path is None or holder not in path.holders:
            violations.append(f"slot {slot}: alert {alert_id} drops bits at {holder} outside its chosen path")
            continue
        drops[(alert_id, slot, path.holders.index(holder))] += bits
    replayed = _replay_buffers(scenario, paths, flows, drops, violations)
    if delivered_bits is not None:
        for alert in scenario.alerts:
            if not math.isclose(delivered_bits[alert.id], replayed[alert.id], abs_tol=ABSOLUTE_TOLERANCE_BITS):
                violations.append(
                    f"alert {alert.id}: reported {delivered_bits[alert.id]:.3f} delivered bits, "
                    f"ledger gives {replayed[alert.id]:.3f}"
                )
    return violations


def _check_bandwidth(
    scenario: Scenario, plan: Plan, ledger: Ledger, violations: list[str]
) -> dict[tuple[int, LinkId], float]:
    bandwidth: dict[tuple[int, LinkId], float] = {}
    totals: dict[tuple[int, str], float] = defaultdict(float)
    active_downlinks: dict[int, int] = defaultdict(int)
    for slot, link, hz in ledger.bandwidth:
        bandwidth[(slot, link)] = hz
        if link[0] not in (ACCESS, DOWNLINK):
            violations.append(f"slot {slot}: bandwidth assigned to non-radio link {link}")
            continue
        if hz < 0.0:
            violations.append(f"slot {slot}: negative bandwidth on {link}")
        totals[(slot, link[0])] += hz
        if link[0] == DOWNLINK and hz > 0.0:
            active_downlinks[slot] += 1
        if isinstance(plan.bandwidth, StaticSchedule) and not math.isclose(
            hz, plan.bandwidth.at(link, slot), rel_tol=RELATIVE_TOLERANCE, abs_tol=1e-9
        ):
            violations.append(f"slot {slot}: bandwidth on {link} differs from the static schedule")
    limit = scenario.spectrum.b_tot_hz * (1.0 + RELATIVE_TOLERANCE)
    for (slot, kind), total in sorted(totals.items()):
        if total > limit:
            violations.append(f"slot {slot}: {kind} bandwidth {total:.1f} Hz exceeds B_tot")
    for slot, count in sorted(active_downlinks.items()):
        if count > scenario.uav.max_active_downlinks:
            violations.append(f"slot {slot}: {count} active downlinks exceed the cap")
    return bandwidth


def _check_capacity(
    scenario: Scenario,
    channels: Mapping[LinkId, LinkChannel],
    ledger: Ledger,
    bandwidth: Mapping[tuple[int, LinkId], float],
    violations: list[str],
) -> dict[tuple[int, LinkId], float]:
    slot_s = scenario.time.slot_s
    access_s = scenario.time.access_fraction * slot_s
    capacity: dict[tuple[int, LinkId], float] = {}
    for slot, link, bits in ledger.capacity_bits:
        capacity[(slot, link)] = bits
        if link[0] == BACKHAUL:
            expected = slot_s * scenario.backhaul_capacity_bps(link[1], link[2])
        else:
            duration = access_s if link[0] == ACCESS else slot_s - access_s
            expected = duration * channels[link].rate_bps(slot, bandwidth.get((slot, link), 0.0))
        if not math.isclose(bits, expected, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE_BITS):
            violations.append(f"slot {slot}: recorded capacity {bits:.3f} of {link} differs from {expected:.3f}")
    return capacity


def _check_transfers(
    ledger: Ledger,
    paths: Mapping[str, CandidatePath | None],
    capacity: Mapping[tuple[int, LinkId], float],
    violations: list[str],
) -> dict[tuple[str, int, int], float]:
    load: dict[tuple[int, LinkId], float] = defaultdict(float)
    flows: dict[tuple[str, int, int], float] = defaultdict(float)
    for slot, link, alert_id, bits in ledger.transfers:
        path = paths.get(alert_id)
        if path is None or link not in path.links:
            violations.append(f"slot {slot}: alert {alert_id} uses {link} outside its chosen path")
            continue
        if bits <= 0.0:
            violations.append(f"slot {slot}: alert {alert_id} records non-positive transfer on {link}")
        load[(slot, link)] += bits
        flows[(alert_id, slot, path.links.index(link))] += bits
    for (slot, link), bits in sorted(load.items()):
        limit = capacity.get((slot, link))
        if limit is None:
            violations.append(f"slot {slot}: {link} carries bits without a recorded capacity")
        elif bits > limit * (1.0 + RELATIVE_TOLERANCE) + ABSOLUTE_TOLERANCE_BITS:
            violations.append(f"slot {slot}: {link} carries {bits:.3f} bits over capacity {limit:.3f}")
    return flows


def _replay_buffers(
    scenario: Scenario,
    paths: Mapping[str, CandidatePath | None],
    flows: Mapping[tuple[str, int, int], float],
    drops: Mapping[tuple[str, int, int], float],
    violations: list[str],
) -> dict[str, float]:
    delivered: dict[str, float] = {}
    for alert in scenario.alerts:
        delivered[alert.id] = 0.0
        path = paths[alert.id]
        if path is None:
            continue
        hops = len(path.links)
        buffer = [0.0] * hops
        for slot in range(scenario.time.num_slots):
            if slot == alert.release_slot:
                buffer[0] += float(alert.size_bits)
            for hop in range(hops):
                dropped = drops.get((alert.id, slot, hop), 0.0)
                if dropped > 0.0 and slot < alert.deadline_slot:
                    violations.append(f"slot {slot}: alert {alert.id} dropped before its deadline")
                if dropped > buffer[hop] + ABSOLUTE_TOLERANCE_BITS:
                    violations.append(
                        f"slot {slot}: alert {alert.id} drops {dropped:.3f} bits at hop {hop} "
                        f"holding only {buffer[hop]:.3f}"
                    )
                buffer[hop] -= dropped
                if slot >= alert.deadline_slot and buffer[hop] > ABSOLUTE_TOLERANCE_BITS:
                    violations.append(f"slot {slot}: alert {alert.id} keeps expired bits at hop {hop}")
            incoming = [0.0] * (hops + 1)
            for hop in range(hops):
                sent = flows.get((alert.id, slot, hop), 0.0)
                if sent > 0.0 and slot < alert.release_slot:
                    violations.append(f"slot {slot}: alert {alert.id} transmits before release")
                if sent > buffer[hop] + ABSOLUTE_TOLERANCE_BITS:
                    violations.append(
                        f"slot {slot}: alert {alert.id} sends {sent:.3f} bits at hop {hop} "
                        f"holding only {buffer[hop]:.3f}"
                    )
                buffer[hop] -= sent
                incoming[hop + 1] += sent
            for hop in range(1, hops):
                buffer[hop] += incoming[hop]
            delivered[alert.id] += incoming[hops]
        if delivered[alert.id] > alert.size_bits + ABSOLUTE_TOLERANCE_BITS:
            violations.append(f"alert {alert.id}: delivered {delivered[alert.id]:.3f} bits exceed its size")
    return delivered
```

- [ ] **Step 4: Run the checker tests and the full suite**

Run: `python -m pytest tests/models/test_checker.py -v`

Expected: 7 passed.

Run: `python -m pytest -q`

Expected: 70 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/models/checker.py tests/models/test_checker.py
git commit -m "feat: add independent ledger checker"
```

### Task 7: Timely flags, shortfall, and time-expanded connectivity

**Files:**
- Create: `src/models/metrics.py`
- Test: `tests/models/test_metrics.py`

**Interfaces:**
- Consumes: `Scenario`, `UAV_ID` (Task 1); `ACCESS`, `DOWNLINK`, `LinkChannel`, `LinkId`, `all_uav_links`, `in_ground_range` (Task 2).
- Produces: `timely_flags(scenario, delivered_bits) -> dict[str, bool]`; `shortfall(scenario, delivered_bits) -> float`; frozen dataclass `Connectivity(connected: dict[str, bool], disconnected_slots: dict[str, int])`; `connectivity(scenario, uav_channels: Mapping[LinkId, LinkChannel] | None) -> Connectivity` (`None` removes the UAV; otherwise the mapping must contain every link of `all_uav_links(scenario)`).

Connectivity runs backward over slots: a vertex at slot n reaches the center if it waits or takes one usable link into a vertex that reaches the center at slot n + 1. A UAV link is usable in slot n when its rate with the full `B_tot` reaches `usable_rate_bps`.

- [ ] **Step 1: Write the failing metric tests**

```python
# tests/models/test_metrics.py
import numpy as np
import pytest

from models.channel import LinkChannel, all_uav_links
from models.metrics import connectivity, shortfall, timely_flags
from models.scenario import UAV_ID, scenario_from_dict
from scenario_builders import make_alert


def _uav_channels(scenario, usable):
    channels = {link: LinkChannel.constant(scenario.time.num_slots, 0.0) for link in all_uav_links(scenario)}
    for link, slot in usable:
        snr = np.zeros(scenario.time.num_slots)
        snr[slot] = 1e6
        channels[link] = LinkChannel(np.ones(scenario.time.num_slots), snr, snr.copy())
    return channels


def test_timely_flags_and_shortfall(raw_scenario):
    raw_scenario["alerts"].append(make_alert("alert-0001", "s01", 0, 10, 500000))
    scenario = scenario_from_dict(raw_scenario)
    delivered = {"alert-0000": 1000000.0 - 1e-9, "alert-0001": 125000.0}
    assert timely_flags(scenario, delivered) == {"alert-0000": True, "alert-0001": False}
    assert shortfall(scenario, delivered) == pytest.approx(0.75)


def test_source_connects_only_through_uav_visit_then_backhaul(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    channels = _uav_channels(scenario, [(("access", "s01", UAV_ID), 5), (("down", UAV_ID, "n2"), 7)])
    reach = connectivity(scenario, channels)
    assert reach.connected == {"s00": True, "s01": True}
    assert reach.disconnected_slots == {"s00": 0, "s01": 14}
    ground_only = connectivity(scenario, None)
    assert ground_only.connected == {"s00": True, "s01": False}
    assert ground_only.disconnected_slots["s01"] == 20


def test_downlink_before_uplink_does_not_connect(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    channels = _uav_channels(scenario, [(("access", "s01", UAV_ID), 8), (("down", UAV_ID, "n2"), 7)])
    assert connectivity(scenario, channels).connected["s01"] is False
```

- [ ] **Step 2: Run the metric tests and verify they fail**

Run: `python -m pytest tests/models/test_metrics.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'models.metrics'`.

- [ ] **Step 3: Implement the metrics**

```python
# src/models/metrics.py
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .channel import ACCESS, DOWNLINK, LinkChannel, LinkId, all_uav_links, in_ground_range
from .scenario import Scenario, UAV_ID

DELIVERY_TOLERANCE_BITS = 1e-6


def timely_flags(scenario: Scenario, delivered_bits: Mapping[str, float]) -> dict[str, bool]:
    return {
        alert.id: delivered_bits[alert.id] >= alert.size_bits - DELIVERY_TOLERANCE_BITS
        for alert in scenario.alerts
    }


def shortfall(scenario: Scenario, delivered_bits: Mapping[str, float]) -> float:
    return sum(
        max(0.0, alert.size_bits - delivered_bits[alert.id]) / alert.size_bits for alert in scenario.alerts
    )


@dataclass(frozen=True)
class Connectivity:
    connected: dict[str, bool]
    disconnected_slots: dict[str, int]


def connectivity(scenario: Scenario, uav_channels: Mapping[LinkId, LinkChannel] | None) -> Connectivity:
    """Backward reachability to the center on the time-expanded graph; None removes the UAV."""
    num_slots = scenario.time.num_slots
    neighbours: dict[str, list[str]] = {node.id: [] for node in scenario.nodes}
    for edge in scenario.edges:
        if edge.alive:
            neighbours[edge.source].append(edge.target)
            neighbours[edge.target].append(edge.source)
    entries = {
        source.id: [
            node.id for node in scenario.nodes if in_ground_range(scenario, math.dist(source.xy, node.xy))
        ]
        for source in scenario.sources
    }
    usable: dict[LinkId, np.ndarray] = {}
    if uav_channels is not None:
        for link in all_uav_links(scenario):
            channel = uav_channels[link]
            usable[link] = np.array(
                [
                    channel.rate_bps(slot, scenario.spectrum.b_tot_hz) >= scenario.usable_rate_bps
                    for slot in range(num_slots)
                ]
            )

    node_reach = {node.id: node.id == scenario.center for node in scenario.nodes}
    uav_reach = False
    source_reach = {source.id: False for source in scenario.sources}
    disconnected = {source.id: 0 for source in scenario.sources}
    for slot in range(num_slots - 1, -1, -1):
        next_nodes, next_uav, next_sources = node_reach, uav_reach, source_reach
        node_reach = {
            node_id: next_nodes[node_id] or any(next_nodes[other] for other in neighbours[node_id])
            for node_id in next_nodes
        }
        uav_reach = next_uav or any(
            next_nodes[node.id] and usable[(DOWNLINK, UAV_ID, node.id)][slot]
            for node in scenario.nodes
            if usable
        )
        source_reach = {}
        for source in scenario.sources:
            reach = (
                next_sources[source.id]
                or any(next_nodes[node_id] for node_id in entries[source.id])
                or (bool(usable) and next_uav and bool(usable[(ACCESS, source.id, UAV_ID)][slot]))
            )
            source_reach[source.id] = reach
            if not reach:
                disconnected[source.id] += 1
    return Connectivity(connected=source_reach, disconnected_slots=disconnected)
```

- [ ] **Step 4: Run the metric tests and the full suite**

Run: `python -m pytest tests/models/test_metrics.py -v`

Expected: 3 passed.

Run: `python -m pytest -q`

Expected: 73 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/models/metrics.py tests/models/test_metrics.py
git commit -m "feat: add timely, shortfall, and connectivity metrics"
```

### Task 8: `evaluate(...)` entry point

**Files:**
- Create: `src/models/evaluate.py`
- Test: `tests/models/test_evaluate.py`

**Interfaces:**
- Consumes: `Scenario` (Task 1); `all_uav_links`, `build_link_channels`, `channel_uniforms` (Task 2); `CandidatePath`, `candidate_paths`, `radio_links` (Task 3); `Plan`, `validate_plan` (Task 4); `Ledger`, `simulate` (Task 5); `SimulatorInvariantError`, `check_ledger` (Task 6); `connectivity`, `shortfall`, `timely_flags` (Task 7).
- Produces: `EXPECTED = "expected"`, `REALIZED = "realized"`, `INFEASIBLE_KEY = (-1, -1, -math.inf)`; dataclass `EvaluationCounter(calls: int = 0)`; frozen dataclasses `RealizationResult(realization_id, timely, delivered_bits, delivery_slot, shortfall, connected, disconnected_time_s, ledger)` and `EvaluationResult(feasible, violations, mode, realizations, timely_ratio, connectivity_ratio, connectivity_ratio_without_uav, total_shortfall, key, paths_version, scenario_sha256, runtime_s)`; `evaluate(scenario, plan, mode, realization_ids=(), counter=None, candidates=None, keep_ledger=False) -> EvaluationResult`.

`evaluate` raises `ValueError` for an unknown mode, for realization IDs in expected mode, and for no realization IDs in realized mode. It counts one call per invocation, returns `feasible=False` without simulating when `validate_plan` reports violations, and raises `SimulatorInvariantError` when the checker finds a problem.

- [ ] **Step 1: Write the failing evaluator tests**

```python
# tests/models/test_evaluate.py
import math

import numpy as np
import pytest

from models.evaluate import EXPECTED, INFEASIBLE_KEY, REALIZED, EvaluationCounter, evaluate
from models.paths import candidate_paths, radio_links
from models.plan import EqualSplitBacklogged, Plan, StaticSchedule, stationary_trajectory
from models.scenario import scenario_from_dict
from scenario_builders import make_alert


def test_mode_arguments_are_checked(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0}, EqualSplitBacklogged())
    with pytest.raises(ValueError):
        evaluate(scenario, plan, EXPECTED, (0,))
    with pytest.raises(ValueError):
        evaluate(scenario, plan, REALIZED)


def test_invalid_plan_is_infeasible_and_counted(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    counter = EvaluationCounter()
    plan = Plan(stationary_trajectory(scenario), {}, EqualSplitBacklogged())
    result = evaluate(scenario, plan, REALIZED, (0, 1, 2), counter=counter)
    assert result.feasible is False
    assert result.key == INFEASIBLE_KEY
    assert result.realizations == ()
    assert counter.calls == 1


def test_realized_results_are_reproducible(raw_scenario):
    raw_scenario["alerts"].append(make_alert("alert-0001", "s01", 0, 20, 20000))
    scenario = scenario_from_dict(raw_scenario)
    plan = Plan(stationary_trajectory(scenario), {"alert-0000": 0, "alert-0001": 0}, EqualSplitBacklogged())
    counter = EvaluationCounter()
    first = evaluate(scenario, plan, REALIZED, (0, 1), counter=counter, keep_ledger=True)
    second = evaluate(scenario, plan, REALIZED, (0, 1), counter=counter)
    assert counter.calls == 2
    assert first.feasible and first.key == second.key
    assert [r.delivered_bits for r in first.realizations] == [r.delivered_bits for r in second.realizations]
    assert first.realizations[0].ledger is not None and second.realizations[0].ledger is None
    assert first.connectivity_ratio_without_uav == 0.5
    assert first.scenario_sha256 == scenario.sha256 and first.paths_version == 1


def _out_and_back(scenario, target, hover_until):
    start = np.asarray(scenario.uav.start_xy_m)
    goal = np.asarray(target)
    steps = math.ceil(np.linalg.norm(goal - start) / (scenario.uav.v_max_mps * scenario.time.slot_s))
    outward = [start + (goal - start) * j / steps for j in range(steps + 1)]
    hover = [goal] * (hover_until - steps)
    back = [goal + (start - goal) * j / steps for j in range(1, steps + 1)]
    return np.asarray(outward + hover + back)


def test_flying_to_far_source_beats_hovering_in_expected_mode(raw_scenario):
    raw_scenario["time"]["num_slots"] = 80
    raw_scenario["alerts"] = [make_alert("alert-0001", "s01", 0, 80, 200000)]
    scenario = scenario_from_dict(raw_scenario)
    trajectory = _out_and_back(scenario, (100.0, 100.0), 54)
    assert trajectory.shape == (81, 2)
    flying = evaluate(scenario, Plan(trajectory, {"alert-0001": 0}, EqualSplitBacklogged()), EXPECTED)
    hovering = evaluate(
        scenario, Plan(stationary_trajectory(scenario), {"alert-0001": 0}, EqualSplitBacklogged()), EXPECTED
    )
    assert flying.feasible and hovering.feasible
    assert flying.timely_ratio == 1.0
    assert hovering.timely_ratio == 0.0
    assert flying.key > hovering.key


def test_random_valid_plans_never_break_invariants(raw_scenario):
    raw_scenario["alerts"] = [
        make_alert("alert-0000", "s00", 0, 12, 400000),
        make_alert("alert-0001", "s00", 3, 20, 700000),
        make_alert("alert-0002", "s01", 1, 18, 50000),
        make_alert("alert-0003", "s01", 6, 20, 90000),
    ]
    scenario = scenario_from_dict(raw_scenario)
    candidates = candidate_paths(scenario)
    links = radio_links(candidates)
    rng = np.random.default_rng(20260911)
    for _ in range(15):
        steps = rng.uniform(-35.0, 35.0, size=(10, 2))
        outward = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
        offsets = np.vstack([outward, outward[-2::-1]])
        trajectory = np.asarray(scenario.uav.start_xy_m) + offsets
        choice = {
            alert.id: None if rng.random() < 0.2 else int(rng.integers(0, len(candidates[alert.id])))
            for alert in scenario.alerts
        }
        if rng.random() < 0.5:
            policy = EqualSplitBacklogged()
        else:
            values = {link: rng.random(20) for link in links}
            for kind in ("access", "down"):
                group = [link for link in links if link[0] == kind]
                total = sum(values[link] for link in group)
                for link in group:
                    values[link] = values[link] / total * scenario.spectrum.b_tot_hz
            downlinks = [link for link in links if link[0] == "down"]
            for slot in range(20):
                keep = set(rng.choice(len(downlinks), size=min(2, len(downlinks)), replace=False))
                for index, link in enumerate(downlinks):
                    if index not in keep:
                        values[link][slot] = 0.0
            policy = StaticSchedule(values)
        result = evaluate(scenario, Plan(trajectory, choice, policy), REALIZED, (0, 1), candidates=candidates)
        assert result.feasible, result.violations
        for realization in result.realizations:
            for alert in scenario.alerts:
                assert realization.delivered_bits[alert.id] <= alert.size_bits + 1e-6
```

- [ ] **Step 2: Run the evaluator tests and verify they fail**

Run: `python -m pytest tests/models/test_evaluate.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'models.evaluate'`.

- [ ] **Step 3: Implement the evaluator**

```python
# src/models/evaluate.py
from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

from .channel import all_uav_links, build_link_channels, channel_uniforms
from .checker import SimulatorInvariantError, check_ledger
from .ledger import Ledger
from .metrics import connectivity, shortfall, timely_flags
from .paths import CandidatePath, candidate_paths, radio_links
from .plan import Plan, validate_plan
from .scenario import Scenario
from .simulator import simulate

EXPECTED = "expected"
REALIZED = "realized"
INFEASIBLE_KEY = (-1, -1, -math.inf)


@dataclass
class EvaluationCounter:
    calls: int = 0


@dataclass(frozen=True)
class RealizationResult:
    realization_id: int | None
    timely: dict[str, bool]
    delivered_bits: dict[str, float]
    delivery_slot: dict[str, int | None]
    shortfall: float
    connected: dict[str, bool]
    disconnected_time_s: dict[str, float]
    ledger: Ledger | None


@dataclass(frozen=True)
class EvaluationResult:
    feasible: bool
    violations: tuple[str, ...]
    mode: str
    realizations: tuple[RealizationResult, ...]
    timely_ratio: float
    connectivity_ratio: float
    connectivity_ratio_without_uav: float
    total_shortfall: float
    key: tuple[int, int, float]
    paths_version: int
    scenario_sha256: str
    runtime_s: float


def evaluate(
    scenario: Scenario,
    plan: Plan,
    mode: str,
    realization_ids: Sequence[int] = (),
    counter: EvaluationCounter | None = None,
    candidates: Mapping[str, tuple[CandidatePath, ...]] | None = None,
    keep_ledger: bool = False,
) -> EvaluationResult:
    started = time.perf_counter()
    if mode not in (EXPECTED, REALIZED):
        raise ValueError(f"unknown evaluation mode {mode!r}")
    if mode == EXPECTED and realization_ids:
        raise ValueError("expected mode takes no realization ids")
    if mode == REALIZED and not realization_ids:
        raise ValueError("realized mode needs at least one realization id")
    if counter is not None:
        counter.calls += 1
    candidates = candidates if candidates is not None else candidate_paths(scenario)
    violations = validate_plan(scenario, candidates, plan)
    if violations:
        return EvaluationResult(
            feasible=False,
            violations=tuple(violations),
            mode=mode,
            realizations=(),
            timely_ratio=0.0,
            connectivity_ratio=0.0,
            connectivity_ratio_without_uav=0.0,
            total_shortfall=math.inf,
            key=INFEASIBLE_KEY,
            paths_version=scenario.paths.version,
            scenario_sha256=scenario.sha256,
            runtime_s=time.perf_counter() - started,
        )

    links = set(radio_links(candidates)) | set(all_uav_links(scenario))
    without_uav = connectivity(scenario, None)
    results = []
    for realization_id in realization_ids if mode == REALIZED else (None,):
        uniforms = None if realization_id is None else channel_uniforms(scenario, realization_id)
        channels = build_link_channels(scenario, links, plan.trajectory, uniforms)
        outcome = simulate(scenario, candidates, plan, channels, realization_id)
        problems = check_ledger(scenario, candidates, plan, channels, outcome.ledger, outcome.delivered_bits)
        if problems:
            raise SimulatorInvariantError(problems)
        reach = connectivity(scenario, channels)
        results.append(
            RealizationResult(
                realization_id=realization_id,
                timely=timely_flags(scenario, outcome.delivered_bits),
                delivered_bits=outcome.delivered_bits,
                delivery_slot=outcome.delivery_slot,
                shortfall=shortfall(scenario, outcome.delivered_bits),
                connected=reach.connected,
                disconnected_time_s={
                    source_id: slots * scenario.time.slot_s for source_id, slots in reach.disconnected_slots.items()
                },
                ledger=outcome.ledger if keep_ledger else None,
            )
        )

    timely_count = sum(sum(result.timely.values()) for result in results)
    connected_count = sum(sum(result.connected.values()) for result in results)
    total_shortfall = sum(result.shortfall for result in results)
    return EvaluationResult(
        feasible=True,
        violations=(),
        mode=mode,
        realizations=tuple(results),
        timely_ratio=timely_count / (len(results) * len(scenario.alerts)) if scenario.alerts else 0.0,
        connectivity_ratio=connected_count / (len(results) * len(scenario.sources)),
        connectivity_ratio_without_uav=sum(without_uav.connected.values()) / len(scenario.sources),
        total_shortfall=total_shortfall,
        key=(timely_count, connected_count, -round(total_shortfall, 9)),
        paths_version=scenario.paths.version,
        scenario_sha256=scenario.sha256,
        runtime_s=time.perf_counter() - started,
    )
```

- [ ] **Step 4: Run the evaluator tests and the full suite**

Run: `python -m pytest tests/models/test_evaluate.py -v`

Expected: 5 passed.

Run: `python -m pytest -q`

Expected: 78 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add src/models/evaluate.py tests/models/test_evaluate.py
git commit -m "feat: add plan evaluator entry point"
```

### Task 9: Scenario schema v2 generator and v0 configuration

**Files:**
- Modify (full rewrite): `src/data/scenarios.py`
- Create: `configs/scenarios/v0.yaml`
- Modify: `src/data/cli.py` (remove the schema v1 import and the placeholder scenario block in `run_preprocess`)
- Modify (full rewrite): `tests/data/test_scenarios.py`
- Test: `tests/data/test_preprocess.py`

**Interfaces:**
- Consumes: `models.rng.named_rng` (Task 1); `models.scenario.scenario_from_dict` in tests (Task 1).
- Produces: `data.scenarios.GENERATOR_VERSION = 2`; frozen dataclass `ScenarioSetConfig` (fields listed in the code, plus `physical: Mapping[str, object]` holding the `uav`, `time`, `spectrum`, `channel`, `source_power_w`, and `paths` sections); `load_scenario_set_config(path: Path) -> ScenarioSetConfig`; `select_topologies(eligible: Sequence[tuple[str, int]], strata: int, dev_count: int, seed: int) -> dict[str, str]` (topology ID to `"eval"` or `"dev"`); `scenario_seed(base_seed: int, topology_id: str, replicate: int) -> int`; `choose_center(network) -> str`; `sample_damage_zones(...)`, `sample_sources(...)`, `sample_alerts(...)`, `sample_failures(...)`; `generate_scenario(network, split, replicate, demand_values, config, network_sha256, raw_sha256, config_hash) -> dict[str, object]`; `build_scenario_set(networks, network_sha256, demand_values, config, raw_sha256, config_hash) -> list[dict[str, object]]`.

Schema v1 (`ScenarioInputs`, `canonical_hash`, `generate_scenario(inputs)`) is removed. Its only other user is `run_preprocess`, which must stop writing the placeholder scenario in this task; otherwise `data.cli` fails to import.

- [ ] **Step 1: Replace the scenario tests and add the preprocess regression test**

Replace the whole content of `tests/data/test_scenarios.py` with:

```python
# tests/data/test_scenarios.py
from dataclasses import replace
from pathlib import Path

import pytest

from data.scenarios import (
    build_scenario_set,
    choose_center,
    generate_scenario,
    load_scenario_set_config,
    scenario_seed,
    select_topologies,
)
from models.scenario import scenario_from_dict

CONFIG = load_scenario_set_config(Path("configs/scenarios/v0.yaml"))


def grid_network(network_id: str, rows: int, cols: int) -> dict[str, object]:
    nodes = [
        {"id": f"n{row * cols + col}", "x_m": 2000.0 * col / (cols - 1), "y_m": 2000.0 * row / (rows - 1)}
        for row in range(rows)
        for col in range(cols)
    ]
    edges = []
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            if col + 1 < cols:
                edges.append({"source": f"n{index}", "target": f"n{index + 1}"})
            if row + 1 < rows:
                edges.append({"source": f"n{index}", "target": f"n{index + cols}"})
    return {
        "network_id": network_id,
        "nodes": nodes,
        "edges": edges,
        "normalization": {"box_size_m": 2000.0, "scale_x": 1.0, "scale_y": 1.0},
    }


def _generate(config=CONFIG, network=None, replicate=0):
    return generate_scenario(
        network or grid_network("Grid", 4, 5),
        "eval",
        replicate,
        [1.0, 50.0, 100.0],
        config,
        "a" * 64,
        ["c" * 64, "b" * 64],
        "d" * 64,
    )


def test_v0_config_values():
    assert CONFIG.set_id == "v0"
    assert (CONFIG.strata, CONFIG.dev_count, CONFIG.eval_replicates, CONFIG.dev_replicates) == (13, 3, 3, 2)
    assert CONFIG.physical["time"]["num_slots"] == 150
    assert CONFIG.physical["channel"]["uav"]["beta0_db"] == -50.0


def test_topology_selection_is_stratified_deterministic_and_disjoint():
    eligible = [(f"T{index:02d}", 15 + index) for index in range(26)]
    first = select_topologies(eligible, 13, 3, 20260911)
    assert first == select_topologies(eligible, 13, 3, 20260911)
    assert len(first) == 13
    assert sorted(first.values()).count("dev") == 3
    chosen_sizes = sorted(15 + int(topology_id[1:]) for topology_id in first)
    assert all(15 + 2 * stratum <= size <= 16 + 2 * stratum for stratum, size in enumerate(chosen_sizes))
    with pytest.raises(ValueError):
        select_topologies(eligible[:5], 13, 3, 1)


def test_scenario_seed_and_center_rule():
    assert scenario_seed(1, "Grid", 0) == scenario_seed(1, "Grid", 0)
    assert scenario_seed(1, "Grid", 0) != scenario_seed(1, "Grid", 1)
    star = {"nodes": [{"id": "b"}, {"id": "a"}, {"id": "c"}], "edges": [{"source": "a", "target": "b"}, {"source": "a", "target": "c"}]}
    assert choose_center(star) == "a"
    pair = {"nodes": [{"id": "y"}, {"id": "x"}], "edges": [{"source": "x", "target": "y"}]}
    assert choose_center(pair) == "x"


def test_generated_scenario_is_valid_and_reproducible():
    raw = _generate()
    scenario = scenario_from_dict(raw)
    assert scenario_from_dict(_generate()).sha256 == scenario.sha256
    assert scenario_from_dict(_generate(replicate=1)).sha256 != scenario.sha256
    assert scenario.center not in scenario.relays and len(scenario.relays) == 19
    assert scenario.uav.start_xy_m == scenario.node_by_id[scenario.center].xy
    assert len(scenario.sources) == 15 and len(scenario.alerts) == 40
    assert all(0.0 <= source.x_m <= 2000.0 and 0.0 <= source.y_m <= 2000.0 for source in scenario.sources)
    assert all(0 <= alert.release_slot <= 89 for alert in scenario.alerts)
    assert all(20 <= alert.deadline_slot - alert.release_slot <= 60 for alert in scenario.alerts)
    assert {alert.size_ratio for alert in scenario.alerts} <= {0.25, 50.0 / 50.0, 100.0 / 50.0}
    assert raw["provenance"]["raw_sha256"] == ["b" * 64, "c" * 64]
    assert all(edge.capacity_bps == (0.0 if edge.failed else 1e6) for edge in scenario.edges)


def test_changing_alert_rule_leaves_other_streams_unchanged():
    base = _generate()
    fewer_alerts = _generate(config=replace(CONFIG, alert_count=10))
    assert fewer_alerts["network"]["edges"] == base["network"]["edges"]
    assert fewer_alerts["sources"] == base["sources"]
    assert fewer_alerts["damage_zones"] == base["damage_zones"]
    assert len(fewer_alerts["alerts"]) == 10


def test_generation_rule_failures_name_the_topology():
    with pytest.raises(ValueError, match="Grid: damage-zone rule placed"):
        _generate(config=replace(CONFIG, zone_min_separation_m=5000.0))
    slow = replace(CONFIG, physical={**CONFIG.physical, "uav": {**CONFIG.physical["uav"], "v_max_mps": 10.0}})
    with pytest.raises(ValueError, match="Grid: v_max"):
        _generate(config=slow)


def test_build_scenario_set_counts_replicates_per_split():
    networks = {f"G{index:02d}": grid_network(f"G{index:02d}", 3, 5 + index) for index in range(13)}
    scenarios = build_scenario_set(
        networks, {key: "e" * 64 for key in networks}, [1.0], CONFIG, [], "f" * 64
    )
    splits = [scenario["split"] for scenario in scenarios]
    assert splits.count("eval") == 30 and splits.count("dev") == 6
```

Create:

```python
# tests/data/test_preprocess.py
from pathlib import Path
import shutil

from data.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_preprocess_no_longer_writes_placeholder_scenarios(tmp_path: Path):
    root = tmp_path / "data"
    topology = root / "raw" / "topology-zoo"
    topology.mkdir(parents=True)
    shutil.copy(FIXTURES / "topology_zoo" / "eligible.graphml", topology / "Eligible.graphml")
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "profile: fixture\n"
        f"data_root: {root}\n"
        "rescuenet_selection_count: 30\n"
        "selection_seed: 20260910\n"
        "box_size_m: 2000.0\n"
        "artifacts: []\n",
        encoding="utf-8",
    )
    assert main(["preprocess", "--profile", str(profile)]) == 0
    assert (root / "processed" / "networks" / "Eligible.json").exists()
    assert not (root / "processed" / "scenarios").exists()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/data/test_scenarios.py tests/data/test_preprocess.py -v`

Expected: `tests/data/test_scenarios.py` fails to collect with `ImportError: cannot import name 'build_scenario_set' from 'data.scenarios'`; `test_preprocess_no_longer_writes_placeholder_scenarios` fails because `processed/scenarios` exists.

- [ ] **Step 3: Add the v0 configuration**

```yaml
# configs/scenarios/v0.yaml
set_id: v0
data_root: data
selection_seed: 20260911
scenario_base_seed: 20260911
topology:
  strata: 13
  dev_count: 3
  eval_replicates: 3
  dev_replicates: 2
workload:
  zone_count: 3
  zone_min_separation_m: 600.0
  zone_radius_m: 400.0
  zone_max_attempts: 1000
  source_count: 15
  source_sigma_m: 150.0
  alert_count: 40
  deadline_min_slots: 20
  deadline_max_slots: 60
  size_ref_bits: 250000
  size_ratio_min: 0.25
  size_ratio_max: 4.0
  sndlib_instance: abilene
failures:
  p_in: 0.6
  p_out: 0.05
backhaul_capacity_bps: 1000000.0
uav:
  altitude_m: 100.0
  v_max_mps: 50.0
  total_power_w: 0.1
  max_active_downlinks: 2
time:
  slot_s: 1.0
  num_slots: 150
  access_fraction: 0.5
spectrum:
  b_tot_hz: 1000000.0
  noise_psd_dbm_per_hz: -140.0
channel:
  uav:
    plos_a: 9.61
    plos_b: 0.16
    nlos_attenuation: 0.2
    alpha_los: 2.2
    alpha_nlos: 3.3
    beta0_db: -50.0
  ground:
    alpha: 2.8
    beta0_db: -50.0
  usable_rate_bps: 10000.0
source_power_w: 0.1
paths:
  k: 3
  max_ground: 2
  version: 1
```

- [ ] **Step 4: Rewrite the generator**

Replace the whole content of `src/data/scenarios.py` with:

```python
# src/data/scenarios.py
import copy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import yaml

from models.rng import named_rng

GENERATOR_VERSION = 2
DEMAND_PROVENANCE = "derived-from-sndlib-demand"
PHYSICAL_SECTIONS = ("uav", "time", "spectrum", "channel", "source_power_w", "paths")


@dataclass(frozen=True)
class ScenarioSetConfig:
    set_id: str
    data_root: Path
    selection_seed: int
    scenario_base_seed: int
    strata: int
    dev_count: int
    eval_replicates: int
    dev_replicates: int
    zone_count: int
    zone_min_separation_m: float
    zone_radius_m: float
    zone_max_attempts: int
    source_count: int
    source_sigma_m: float
    alert_count: int
    deadline_min_slots: int
    deadline_max_slots: int
    size_ref_bits: int
    size_ratio_min: float
    size_ratio_max: float
    sndlib_instance: str
    failure_p_in: float
    failure_p_out: float
    backhaul_capacity_bps: float
    physical: Mapping[str, object]


def load_scenario_set_config(path: Path) -> ScenarioSetConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    topology, workload, failures = raw["topology"], raw["workload"], raw["failures"]
    config = ScenarioSetConfig(
        set_id=str(raw["set_id"]),
        data_root=Path(raw.get("data_root", "data")),
        selection_seed=int(raw["selection_seed"]),
        scenario_base_seed=int(raw["scenario_base_seed"]),
        strata=int(topology["strata"]),
        dev_count=int(topology["dev_count"]),
        eval_replicates=int(topology["eval_replicates"]),
        dev_replicates=int(topology["dev_replicates"]),
        zone_count=int(workload["zone_count"]),
        zone_min_separation_m=float(workload["zone_min_separation_m"]),
        zone_radius_m=float(workload["zone_radius_m"]),
        zone_max_attempts=int(workload["zone_max_attempts"]),
        source_count=int(workload["source_count"]),
        source_sigma_m=float(workload["source_sigma_m"]),
        alert_count=int(workload["alert_count"]),
        deadline_min_slots=int(workload["deadline_min_slots"]),
        deadline_max_slots=int(workload["deadline_max_slots"]),
        size_ref_bits=int(workload["size_ref_bits"]),
        size_ratio_min=float(workload["size_ratio_min"]),
        size_ratio_max=float(workload["size_ratio_max"]),
        sndlib_instance=str(workload["sndlib_instance"]),
        failure_p_in=float(failures["p_in"]),
        failure_p_out=float(failures["p_out"]),
        backhaul_capacity_bps=float(raw["backhaul_capacity_bps"]),
        physical={section: copy.deepcopy(raw[section]) for section in PHYSICAL_SECTIONS},
    )
    num_slots = int(config.physical["time"]["num_slots"])
    if not 1 <= config.deadline_min_slots <= config.deadline_max_slots < num_slots:
        raise ValueError("workload: deadline window must satisfy 1 <= min <= max < time.num_slots")
    if not 0 < config.dev_count < config.strata:
        raise ValueError("topology: dev_count must lie in (0, strata)")
    if not (0.0 <= config.failure_p_in <= 1.0 and 0.0 <= config.failure_p_out <= 1.0):
        raise ValueError("failures: probabilities must lie in [0, 1]")
    return config


def select_topologies(
    eligible: Sequence[tuple[str, int]], strata: int, dev_count: int, seed: int
) -> dict[str, str]:
    ordered = sorted(eligible, key=lambda item: (item[1], item[0]))
    if len(ordered) < strata:
        raise ValueError(f"need at least {strata} eligible topologies, found {len(ordered)}")
    size, extra = divmod(len(ordered), strata)
    groups, start = [], 0
    for index in range(strata):
        end = start + size + (1 if index < extra else 0)
        groups.append(ordered[start:end])
        start = end
    pick = named_rng(seed, "topology-selection")
    chosen = [group[int(pick.integers(0, len(group)))][0] for group in groups]
    dev_indices = {int(index) for index in named_rng(seed, "dev-split").choice(strata, size=dev_count, replace=False)}
    return {topology_id: "dev" if index in dev_indices else "eval" for index, topology_id in enumerate(chosen)}


def scenario_seed(base_seed: int, topology_id: str, replicate: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{topology_id}:{replicate}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def choose_center(network: Mapping[str, object]) -> str:
    graph = nx.Graph()
    graph.add_nodes_from(str(node["id"]) for node in network["nodes"])
    graph.add_edges_from((str(edge["source"]), str(edge["target"])) for edge in network["edges"])
    closeness = nx.closeness_centrality(graph)
    return min(graph.nodes, key=lambda node: (-round(closeness[node], 12), node))


def sample_damage_zones(
    rng: np.random.Generator,
    box_m: float,
    count: int,
    min_separation_m: float,
    radius_m: float,
    max_attempts: int,
    label: str,
) -> list[dict[str, float]]:
    centers: list[np.ndarray] = []
    for _ in range(max_attempts):
        if len(centers) == count:
            break
        candidate = rng.uniform(0.0, box_m, size=2)
        if all(math.dist(candidate, other) >= min_separation_m for other in centers):
            centers.append(candidate)
    if len(centers) < count:
        raise ValueError(
            f"{label}: damage-zone rule placed {len(centers)} of {count} zones in {max_attempts} attempts"
        )
    return [{"x_m": float(center[0]), "y_m": float(center[1]), "radius_m": radius_m} for center in centers]


def sample_sources(
    rng: np.random.Generator, zones: Sequence[Mapping[str, float]], count: int, sigma_m: float, box_m: float
) -> list[dict[str, object]]:
    sources = []
    for index in range(count):
        zone = index % len(zones)
        offset = rng.normal(0.0, sigma_m, size=2)
        sources.append(
            {
                "id": f"s{index:02d}",
                "x_m": float(np.clip(zones[zone]["x_m"] + offset[0], 0.0, box_m)),
                "y_m": float(np.clip(zones[zone]["y_m"] + offset[1], 0.0, box_m)),
                "zone": zone,
            }
        )
    return sources


def sample_alerts(
    alert_rng: np.random.Generator,
    size_rng: np.random.Generator,
    source_ids: Sequence[str],
    config: ScenarioSetConfig,
    demand_values: Sequence[float],
) -> list[dict[str, object]]:
    median = float(np.median(demand_values))
    if median <= 0.0:
        raise ValueError(f"{config.sndlib_instance}: demand median must be positive")
    num_slots = int(config.physical["time"]["num_slots"])
    alerts = []
    for index in range(config.alert_count):
        source = source_ids[int(alert_rng.integers(0, len(source_ids)))]
        release = int(alert_rng.integers(0, num_slots - config.deadline_max_slots))
        deadline = release + int(alert_rng.integers(config.deadline_min_slots, config.deadline_max_slots + 1))
        value = float(demand_values[int(size_rng.integers(0, len(demand_values)))])
        ratio = float(np.clip(value / median, config.size_ratio_min, config.size_ratio_max))
        alerts.append(
            {
                "id": f"alert-{index:04d}",
                "source": source,
                "release_slot": release,
                "deadline_slot": deadline,
                "size_bits": int(round(config.size_ref_bits * ratio)),
                "size_ratio": ratio,
                "provenance": DEMAND_PROVENANCE,
            }
        )
    return alerts


def sample_failures(
    rng: np.random.Generator,
    network: Mapping[str, object],
    zones: Sequence[Mapping[str, float]],
    config: ScenarioSetConfig,
) -> list[dict[str, object]]:
    positions = {str(node["id"]): (float(node["x_m"]), float(node["y_m"])) for node in network["nodes"]}
    edges = []
    for edge in sorted(network["edges"], key=lambda item: (str(item["source"]), str(item["target"]))):
        first, second = positions[str(edge["source"])], positions[str(edge["target"])]
        midpoint = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
        inside = any(math.dist(midpoint, (zone["x_m"], zone["y_m"])) <= zone["radius_m"] for zone in zones)
        failed = bool(rng.random() < (config.failure_p_in if inside else config.failure_p_out))
        edges.append(
            {
                "source": str(edge["source"]),
                "target": str(edge["target"]),
                "failed": failed,
                "capacity_bps": 0.0 if failed else config.backhaul_capacity_bps,
            }
        )
    return edges


def generate_scenario(
    network: Mapping[str, object],
    split: str,
    replicate: int,
    demand_values: Sequence[float],
    config: ScenarioSetConfig,
    network_sha256: str,
    raw_sha256: Sequence[str],
    config_hash: str,
) -> dict[str, object]:
    network_id = str(network["network_id"])
    box_m = float(network["normalization"]["box_size_m"])
    physical = copy.deepcopy(dict(config.physical))
    reach_m = float(physical["uav"]["v_max_mps"]) * int(physical["time"]["num_slots"]) * float(physical["time"]["slot_s"])
    if reach_m < 2.0 * math.sqrt(2.0) * box_m:
        raise ValueError(f"{network_id}: v_max * num_slots * slot_s must cover two box diagonals")
    seed = scenario_seed(config.scenario_base_seed, network_id, replicate)
    center = choose_center(network)
    nodes = sorted(
        ({"id": str(node["id"]), "x_m": float(node["x_m"]), "y_m": float(node["y_m"])} for node in network["nodes"]),
        key=lambda node: node["id"],
    )
    zones = sample_damage_zones(
        named_rng(seed, "damage-zones"),
        box_m,
        config.zone_count,
        config.zone_min_separation_m,
        config.zone_radius_m,
        config.zone_max_attempts,
        network_id,
    )
    sources = sample_sources(named_rng(seed, "sources"), zones, config.source_count, config.source_sigma_m, box_m)
    alerts = sample_alerts(
        named_rng(seed, "alerts"), named_rng(seed, "sizes"), [source["id"] for source in sources], config, demand_values
    )
    edges = sample_failures(named_rng(seed, "failures"), network, zones, config)
    center_node = next(node for node in nodes if node["id"] == center)
    center_xy = [center_node["x_m"], center_node["y_m"]]
    return {
        "schema_version": 2,
        "scenario_id": f"{network_id}-r{replicate}",
        "split": split,
        "network": {
            "network_id": network_id,
            "nodes": nodes,
            "edges": edges,
            "normalization": dict(network["normalization"]),
        },
        "center": center,
        "relays": [node["id"] for node in nodes if node["id"] != center],
        "damage_zones": zones,
        "sources": sources,
        "alerts": alerts,
        "uav": {**physical["uav"], "start_xy_m": center_xy, "end_xy_m": list(center_xy)},
        "time": physical["time"],
        "spectrum": physical["spectrum"],
        "channel": physical["channel"],
        "source_power_w": physical["source_power_w"],
        "paths": physical["paths"],
        "provenance": {
            "network_sha256": network_sha256,
            "sndlib_instance": config.sndlib_instance,
            "raw_sha256": sorted(raw_sha256),
            "config_hash": config_hash,
            "scenario_seed": seed,
            "generator_version": GENERATOR_VERSION,
        },
    }


def build_scenario_set(
    networks: Mapping[str, Mapping[str, object]],
    network_sha256: Mapping[str, str],
    demand_values: Sequence[float],
    config: ScenarioSetConfig,
    raw_sha256: Sequence[str],
    config_hash: str,
) -> list[dict[str, object]]:
    eligible = [(topology_id, len(network["nodes"])) for topology_id, network in networks.items()]
    splits = select_topologies(eligible, config.strata, config.dev_count, config.selection_seed)
    scenarios = []
    for topology_id in sorted(splits):
        split = splits[topology_id]
        replicates = config.eval_replicates if split == "eval" else config.dev_replicates
        for replicate in range(replicates):
            scenarios.append(
                generate_scenario(
                    networks[topology_id],
                    split,
                    replicate,
                    demand_values,
                    config,
                    network_sha256[topology_id],
                    raw_sha256,
                    config_hash,
                )
            )
    return scenarios
```

- [ ] **Step 5: Stop writing the placeholder scenario in preprocess**

In `src/data/cli.py`, delete this import line:

```python
from .scenarios import ScenarioInputs, canonical_hash, generate_scenario
```

Then replace the whole `run_preprocess` function, which currently reads:

```python
def run_preprocess(context: PipelineContext) -> None:
    processed = context.data_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    network_dir = processed / "networks"
    network_dir.mkdir(parents=True, exist_ok=True)
    topology_root = context.data_root / "raw" / "topology-zoo"
    normalized = []
    for path in sorted(topology_root.rglob("*.graphml")) if topology_root.exists() else []:
        record = inventory_topology(path)
        if record.eligible:
            network = normalize_topology(path, context.config.box_size_m)
            output = network_dir / f"{record.topology_id}.json"
            output.write_text(json.dumps(network, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            normalized.append(network)
    if not normalized:
        return
    sndlib_root = context.data_root / "raw" / "sndlib-networks-xml" / "extracted"
    demand_records = []
    first_xml = next(iter(sorted(sndlib_root.rglob("*.xml"))), None) if sndlib_root.exists() else None
    if first_xml:
        demand_records = tuple(
            {"original_value": str(demand.value)}
            for demand in parse_sndlib_xml(first_xml).demands
        )
    if not demand_records:
        demand_records = ({"original_value": "1"},)
    config_hash = hashlib.sha256(
        json.dumps(asdict(context.config), default=str, sort_keys=True).encode()
    ).hexdigest()
    scenario = generate_scenario(
        ScenarioInputs(
            network=normalized[0],
            demands=demand_records,
            targets=(),
            raw_sha256=tuple(sorted(record.sha256 for record in context.records)),
            config_hash=config_hash,
            seed=context.config.selection_seed,
            alert_count=30,
            failure_probability=0.2,
        )
    )
    scenario_dir = processed / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / f"scenario-{canonical_hash(scenario)[:12]}.json").write_text(
        json.dumps(scenario, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
```

with:

```python
def run_preprocess(context: PipelineContext) -> None:
    processed = context.data_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    network_dir = processed / "networks"
    network_dir.mkdir(parents=True, exist_ok=True)
    topology_root = context.data_root / "raw" / "topology-zoo"
    for path in sorted(topology_root.rglob("*.graphml")) if topology_root.exists() else []:
        record = inventory_topology(path)
        if record.eligible:
            network = normalize_topology(path, context.config.box_size_m)
            output = network_dir / f"{record.topology_id}.json"
            output.write_text(json.dumps(network, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

Keep the `import hashlib` line; Task 10 uses it again.

Remove the stale schema v1 file left by the old preprocess stage (a Git-ignored local artifact):

```bash
rm -f data/processed/scenarios/scenario-*.json
```

- [ ] **Step 6: Run the tests and the full suite**

Run: `python -m pytest tests/data/test_scenarios.py tests/data/test_preprocess.py -v`

Expected: 8 passed.

Run: `python -m pytest -q`

Expected: 83 passed, 1 skipped.

- [ ] **Step 7: Commit**

```bash
git add src/data/scenarios.py src/data/cli.py configs/scenarios/v0.yaml tests/data/test_scenarios.py tests/data/test_preprocess.py
git commit -m "feat: generate scenario schema v2"
```

### Task 10: `scenarios` CLI stage and manifest

**Files:**
- Modify: `src/data/cli.py` (imports, `build_parser`, new `run_scenarios`, `main`)
- Create: `scripts/data/scenarios.py`
- Modify: `tests/data/test_cli.py` (append one test)
- Test: `tests/data/test_scenario_stage.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `build_scenario_set`, `load_scenario_set_config` (Task 9); `scenario_from_dict` (Task 1); `ground_connected_source_fraction` (Task 3); existing `parse_sndlib_xml` and `_write_csv` in `data.cli`.
- Produces: command `python -m data.cli scenarios --config PATH [--data-root PATH]`; `run_scenarios(config_path: Path, data_root: Path | None) -> int`. The command reads `manifests/topology_inventory.csv` (rows with `eligible == "True"` whose `processed/networks/<id>.json` exists), the first `raw/sndlib-networks-xml/extracted/**/<sndlib_instance>.xml`, and `manifests/sources.json` (SHA-256 of `topology-zoo` and `sndlib-networks-xml`). It clears and rewrites `processed/scenarios/<set_id>/*.json` and writes `manifests/scenarios_<set_id>.csv` with columns `alert_count, failed_edge_count, ground_connected_source_fraction, node_count, replicate, scenario_id, scenario_seed, sha256, split, topology_id` (alphabetical, as `_write_csv` sorts them). The config hash is the SHA-256 of the YAML file bytes.

- [ ] **Step 1: Write the failing CLI tests**

Append to `tests/data/test_cli.py`:

```python
def test_scenarios_command_is_separate_from_profile_stages():
    args = build_parser().parse_args(["scenarios", "--config", "configs/scenarios/v0.yaml"])
    assert args.command == "scenarios"
    assert args.config == Path("configs/scenarios/v0.yaml")
```

Create:

```python
# tests/data/test_scenario_stage.py
import csv
import json
from pathlib import Path
import shutil

import yaml

from data.cli import main
from models.scenario import load_scenario

FIXTURES = Path(__file__).parent / "fixtures"


def _write_inputs(root: Path) -> None:
    network_dir = root / "processed" / "networks"
    network_dir.mkdir(parents=True)
    rows = []
    for index in range(6):
        nodes = [{"id": f"n{i}", "x_m": 2000.0 * i / (15 + index), "y_m": 1000.0 + 50.0 * (i % 3)} for i in range(16 + index)]
        edges = [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(15 + index)]
        network = {
            "network_id": f"Line{index}",
            "nodes": nodes,
            "edges": edges,
            "normalization": {"box_size_m": 2000.0, "scale_x": 1.0, "scale_y": 1.0},
        }
        (network_dir / f"Line{index}.json").write_text(json.dumps(network), encoding="utf-8")
        rows.append({"topology_id": f"Line{index}", "eligible": "True"})
    rows.append({"topology_id": "Tiny", "eligible": "False"})
    manifests = root / "manifests"
    manifests.mkdir()
    with (manifests / "topology_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["eligible", "topology_id"])
        writer.writeheader()
        writer.writerows(rows)
    (manifests / "sources.json").write_text(
        json.dumps([{"source_id": "topology-zoo", "sha256": "1" * 64}, {"source_id": "rescuenet-validation", "sha256": "2" * 64}]),
        encoding="utf-8",
    )
    sndlib = root / "raw" / "sndlib-networks-xml" / "extracted" / "sndlib-networks-xml"
    sndlib.mkdir(parents=True)
    shutil.copy(FIXTURES / "sndlib" / "mini.xml", sndlib / "abilene.xml")


def _write_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/scenarios/v0.yaml").read_text(encoding="utf-8"))
    raw["topology"].update(strata=4, dev_count=1, eval_replicates=2, dev_replicates=1)
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_scenarios_stage_writes_valid_files_and_reproducible_manifest(tmp_path: Path):
    root = tmp_path / "data"
    _write_inputs(root)
    config = _write_config(tmp_path)
    assert main(["scenarios", "--config", str(config), "--data-root", str(root)]) == 0
    manifest = root / "manifests" / "scenarios_v0.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7
    assert sorted(row["split"] for row in rows).count("dev") == 1
    for row in rows:
        scenario = load_scenario(root / "processed" / "scenarios" / "v0" / f"{row['scenario_id']}.json")
        assert scenario.sha256 == row["sha256"]
        assert scenario.provenance["raw_sha256"] == ["1" * 64]
        assert 0.0 <= float(row["ground_connected_source_fraction"]) <= 1.0
    first = manifest.read_text(encoding="utf-8")
    assert main(["scenarios", "--config", str(config), "--data-root", str(root)]) == 0
    assert manifest.read_text(encoding="utf-8") == first
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/data/test_cli.py tests/data/test_scenario_stage.py -v`

Expected: both new tests fail with `SystemExit: 2` because argparse reports `invalid choice: 'scenarios'`.

- [ ] **Step 3: Add the scenarios stage**

In `src/data/cli.py`, add these imports directly below `import httpx` (keep the blank line before the relative imports):

```python
from models.paths import ground_connected_source_fraction
from models.scenario import scenario_from_dict
```

and add this relative import directly below the `.rescuenet` import:

```python
from .scenarios import build_scenario_set, load_scenario_set_config
```

Replace `build_parser`:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "verify", "inventory", "preprocess", "all"):
        command = subparsers.add_parser(name)
        command.add_argument("--profile", type=Path, required=True)
        command.add_argument("--data-root", type=Path)
        command.add_argument("--dry-run", action="store_true")
    return parser
```

with:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "verify", "inventory", "preprocess", "all"):
        command = subparsers.add_parser(name)
        command.add_argument("--profile", type=Path, required=True)
        command.add_argument("--data-root", type=Path)
        command.add_argument("--dry-run", action="store_true")
    scenarios = subparsers.add_parser("scenarios")
    scenarios.add_argument("--config", type=Path, required=True)
    scenarios.add_argument("--data-root", type=Path)
    return parser
```

Insert this function between `run_preprocess` and `STAGES = (...)`:

```python
def run_scenarios(config_path: Path, data_root: Path | None) -> int:
    config = load_scenario_set_config(config_path)
    root = data_root or config.data_root
    with (root / "manifests" / "topology_inventory.csv").open(newline="", encoding="utf-8") as stream:
        eligible_ids = sorted(row["topology_id"] for row in csv.DictReader(stream) if row["eligible"] == "True")
    networks, network_sha256 = {}, {}
    for topology_id in eligible_ids:
        path = root / "processed" / "networks" / f"{topology_id}.json"
        if path.exists():
            payload = path.read_bytes()
            networks[topology_id] = json.loads(payload)
            network_sha256[topology_id] = hashlib.sha256(payload).hexdigest()
    sndlib_matches = sorted((root / "raw" / "sndlib-networks-xml" / "extracted").rglob(f"{config.sndlib_instance}.xml"))
    if not sndlib_matches:
        raise FileNotFoundError(f"missing SNDlib instance {config.sndlib_instance}.xml under {root / 'raw'}")
    demand_values = [float(demand.value) for demand in parse_sndlib_xml(sndlib_matches[0]).demands]
    ledger = json.loads((root / "manifests" / "sources.json").read_text(encoding="utf-8"))
    raw_sha256 = [
        record["sha256"] for record in ledger if record["source_id"] in ("topology-zoo", "sndlib-networks-xml")
    ]
    scenarios = build_scenario_set(
        networks,
        network_sha256,
        demand_values,
        config,
        raw_sha256,
        hashlib.sha256(config_path.read_bytes()).hexdigest(),
    )
    output_dir = root / "processed" / "scenarios" / config.set_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.json"):
        stale.unlink()
    rows = []
    for raw in scenarios:
        scenario = scenario_from_dict(raw)
        (output_dir / f"{scenario.scenario_id}.json").write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "split": scenario.split,
                "topology_id": scenario.network_id,
                "replicate": int(scenario.scenario_id.rsplit("-r", 1)[1]),
                "scenario_seed": scenario.scenario_seed,
                "sha256": scenario.sha256,
                "node_count": len(scenario.nodes),
                "alert_count": len(scenario.alerts),
                "failed_edge_count": sum(1 for edge in scenario.edges if edge.failed),
                "ground_connected_source_fraction": round(ground_connected_source_fraction(scenario), 6),
            }
        )
    _write_csv(root / "manifests" / f"scenarios_{config.set_id}.csv", rows)
    return 0
```

Replace `main`:

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.profile)
    data_root = args.data_root or config.data_root
    return run_stage(args.command, config, data_root, args.dry_run)
```

with:

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scenarios":
        return run_scenarios(args.config, args.data_root)
    config = load_config(args.profile)
    data_root = args.data_root or config.data_root
    return run_stage(args.command, config, data_root, args.dry_run)
```

Create the thin script entry point, matching the other stage scripts:

```python
# scripts/data/scenarios.py
import sys

from data.cli import main


raise SystemExit(main(["scenarios", *sys.argv[1:]]))
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `python -m pytest tests/data/test_cli.py tests/data/test_scenario_stage.py -v`

Expected: 4 passed.

Run: `python -m pytest -q`

Expected: 85 passed, 1 skipped.

- [ ] **Step 5: Document the command**

Append this section to the end of `README.md`:

````markdown
## Bộ scenario v0

Sau khi chạy profile `paper`, sinh bộ scenario v0 gồm 30 scenario đánh giá và 6 scenario phát triển:

```bash
python -m data.cli scenarios --config configs/scenarios/v0.yaml
```

Lệnh đọc network đã chuẩn hóa trong `data/processed/networks/` và instance SNDlib `abilene`, ghi file scenario vào `data/processed/scenarios/v0/` (bị Git ignore) và manifest `data/manifests/scenarios_v0.csv` (được commit). Chạy lại với cùng cấu hình cho cùng cột `sha256`.
````

- [ ] **Step 6: Commit**

```bash
git add src/data/cli.py scripts/data/scenarios.py tests/data/test_cli.py tests/data/test_scenario_stage.py README.md
git commit -m "feat: add scenario set generation stage"
```

### Task 11: Calibration check, v0 scenario set, and acceptance run

**Files:**
- Create: `experiments/calibrate_v0.py`
- Test: `tests/models/test_calibration.py`
- Create (generated, committed): `data/manifests/scenarios_v0.csv`
- Create (generated, committed): `results/calibration/v0.json`
- Modify: `README.md`

**Interfaces:**
- Consumes: `load_scenario` (Task 1); `GROUND`, `VIA_UAV`, `CandidatePath`, `candidate_paths` (Task 3); `EqualSplitBacklogged`, `Plan`, `stationary_trajectory` (Task 4); `REALIZED`, `evaluate` (Task 8); the `scenarios` stage (Task 10).
- Produces: `experiments/calibrate_v0.py` with `LOWER_BOUND = 0.05`, `UPPER_BOUND = 0.95`, `first_index(paths, kind) -> int | None`, `zone_tour_trajectory(scenario) -> np.ndarray`, `ground_only_plan(scenario, candidates) -> Plan`, `zone_tour_plan(scenario, candidates) -> Plan`, `PLANS: dict[str, Callable]`, and `main(argv) -> int` (0 when the check passes, 1 otherwise). The report JSON has keys `scenario_count`, `realization_ids`, `pass_bounds`, `plans` (per plan: `feasible`, `mean_timely_ratio`, `per_scenario_timely_ratio`, `mean_runtime_s_per_evaluation`), `zone_tour_beats_ground_only`, and `passed`.

The two calibration plans are not the B0/B1 baselines of sub-project B. Steps 5–7 need the pinned data produced by the `paper` profile (`data/raw/`, `data/processed/networks/`, `data/manifests/topology_inventory.csv`, `data/manifests/sources.json`).

- [ ] **Step 1: Write the failing calibration tests**

```python
# tests/models/test_calibration.py
import importlib.util
from pathlib import Path

import numpy as np

from models.evaluate import REALIZED, evaluate
from models.paths import candidate_paths
from models.plan import validate_plan
from models.scenario import scenario_from_dict

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "calibrate_v0.py"
SPEC = importlib.util.spec_from_file_location("calibrate_v0", SCRIPT)
calibrate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibrate)


def test_zone_tour_visits_zones_and_is_feasible(raw_scenario):
    raw_scenario["time"]["num_slots"] = 120
    raw_scenario["damage_zones"] = [
        {"x_m": 1800.0, "y_m": 1000.0, "radius_m": 400.0},
        {"x_m": 200.0, "y_m": 200.0, "radius_m": 400.0},
    ]
    scenario = scenario_from_dict(raw_scenario)
    candidates = candidate_paths(scenario)
    plan = calibrate.zone_tour_plan(scenario, candidates)
    assert validate_plan(scenario, candidates, plan) == []
    for zone in scenario.damage_zones:
        assert np.min(np.linalg.norm(plan.trajectory - [zone.x_m, zone.y_m], axis=1)) < 1e-6


def test_zone_tour_drops_zones_that_do_not_fit(raw_scenario):
    raw_scenario["damage_zones"] = [
        {"x_m": 1200.0, "y_m": 1000.0, "radius_m": 400.0},
        {"x_m": 100.0, "y_m": 100.0, "radius_m": 400.0},
    ]
    scenario = scenario_from_dict(raw_scenario)
    trajectory = calibrate.zone_tour_trajectory(scenario)
    assert trajectory.shape == (21, 2)
    assert np.min(np.linalg.norm(trajectory - [1200.0, 1000.0], axis=1)) < 1e-6
    assert np.min(np.linalg.norm(trajectory - [100.0, 100.0], axis=1)) > 100.0


def test_calibration_plans_evaluate_feasibly(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    candidates = candidate_paths(scenario)
    for build in calibrate.PLANS.values():
        result = evaluate(scenario, build(scenario, candidates), REALIZED, (0,), candidates=candidates)
        assert result.feasible
```

- [ ] **Step 2: Run the calibration tests and verify they fail**

Run: `python -m pytest tests/models/test_calibration.py -v`

Expected: collection fails with `FileNotFoundError` for `experiments/calibrate_v0.py`.

- [ ] **Step 3: Implement the calibration script**

```python
# experiments/calibrate_v0.py
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Callable, Mapping

import numpy as np

from models.evaluate import REALIZED, evaluate
from models.paths import GROUND, VIA_UAV, CandidatePath, candidate_paths
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import Scenario, load_scenario

LOWER_BOUND = 0.05
UPPER_BOUND = 0.95


def first_index(paths: tuple[CandidatePath, ...], kind: str) -> int | None:
    return next((index for index, path in enumerate(paths) if path.kind == kind), None)


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

- [ ] **Step 4: Run the calibration tests and the full suite**

Run: `python -m pytest tests/models/test_calibration.py -v`

Expected: 3 passed.

Run: `python -m pytest -q`

Expected: 88 passed, 1 skipped.

- [ ] **Step 5: Generate the v0 scenario set twice and confirm reproducibility**

```bash
python -m data.cli scenarios --config configs/scenarios/v0.yaml
cp data/manifests/scenarios_v0.csv /tmp/scenarios_v0.first.csv
python -m data.cli scenarios --config configs/scenarios/v0.yaml
cmp /tmp/scenarios_v0.first.csv data/manifests/scenarios_v0.csv && echo REPRODUCIBLE
ls data/processed/scenarios/v0 | wc -l
```
Expected: `REPRODUCIBLE` and `36`. The manifest lists evaluation topologies `Agis`, `Belnet2008`, `BsonetEurope`, `BtNorthAmerica`, `Cesnet200603`, `Digex`, `Easynet`, `Garr200112`, `Garr200404`, `Marwan` (replicates 0–2) and development topologies `Bbnplanet`, `Canerie`, `Fccn` (replicates 0–1).

- [ ] **Step 6: Run the calibration check**

Run: `python experiments/calibrate_v0.py`

Expected: exit code 0 after roughly 30 seconds, printing `{"ground_only": 0.4133, "zone_tour": 0.1036, "passed": true}`. If `passed` is false, stop: follow spec Section 10.3 (adjust `size_ref_bits` first, then the deadline window), record the change in spec Section 12, and rerun Steps 5–6.

- [ ] **Step 7: Document the evaluator and calibration**

Append this section to the end of `README.md`, after the section added in Task 10:

````markdown
## Evaluator và kiểm tra hiệu chỉnh

Mọi phương pháp chấm điểm kế hoạch qua cùng một hàm:

```python
from pathlib import Path

from models.evaluate import REALIZED, evaluate
from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import load_scenario

scenario = load_scenario(Path("data/processed/scenarios/v0/Agis-r0.json"))
candidates = candidate_paths(scenario)
plan = Plan(stationary_trajectory(scenario), {alert.id: 0 for alert in scenario.alerts}, EqualSplitBacklogged())
result = evaluate(scenario, plan, REALIZED, tuple(range(30)), candidates=candidates)
print(result.timely_ratio, result.key)
```

Kiểm tra hiệu chỉnh chạy hai kế hoạch đơn giản trên 30 scenario đánh giá với 30 realization kênh và ghi `results/calibration/v0.json`:

```bash
python experiments/calibrate_v0.py
```

Lệnh trả mã 0 khi cả hai kế hoạch khả thi và tỷ lệ đúng hạn trung bình của mỗi kế hoạch nằm trong [0.05, 0.95].
````

Run: `python -m pytest -q`

Expected: 88 passed, 1 skipped.

- [ ] **Step 8: Commit**

```bash
git add experiments/calibrate_v0.py tests/models/test_calibration.py data/manifests/scenarios_v0.csv results/calibration/v0.json README.md
git commit -m "feat: calibrate v0 scenarios"
```

---

## Spec Coverage

| Spec section | Task |
|---|---|
| 4 Architecture, file layout, import boundary | 1–8, 9, 10, 11 |
| 5.1 Generation rules | 9 |
| 5.2 JSON layout and loader rejections | 1, 9 |
| 5.3 Command and manifest | 10, 11 |
| 6.1 UAV links, 6.2 realizations, 6.3 ground and backhaul, 6.4 spectrum and power | 2, 5 |
| 7 Candidate paths | 3 |
| 8.1 API, 8.2 input validation | 4, 8 |
| 8.3 Slot semantics | 5 |
| 8.4 Metrics and lexicographic key | 7, 8 |
| 8.5 Independent checker | 6, 8 |
| 8.6 Result and cost | 8, 11 |
| 9 Error handling | 1, 4, 8, 9, 10 |
| 10.1–10.2 Tests | 1–8, 9, 10 |
| 10.3 Calibration check | 11 |
| 11 Acceptance criteria | 9, 10, 11 |
