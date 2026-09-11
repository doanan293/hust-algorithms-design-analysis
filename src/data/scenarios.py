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
