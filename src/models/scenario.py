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
