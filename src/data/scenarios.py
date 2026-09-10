from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping

import numpy as np


def named_rng(seed: int, concern: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{concern}".encode()).digest()
    child_seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(child_seed)


def canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ScenarioInputs:
    network: Mapping[str, object]
    demands: tuple[Mapping[str, object], ...]
    targets: tuple[Mapping[str, object], ...]
    raw_sha256: tuple[str, ...]
    config_hash: str
    seed: int
    alert_count: int
    failure_probability: float

    def with_seed(self, seed: int) -> "ScenarioInputs":
        return replace(self, seed=seed)


def generate_scenario(inputs: ScenarioInputs) -> dict[str, object]:
    if not 0.0 <= inputs.failure_probability <= 1.0:
        raise ValueError("failure_probability must be in [0, 1]")
    node_ids = sorted(str(node["id"]) for node in inputs.network["nodes"])
    if len(node_ids) < 2 or inputs.alert_count <= 0 or not inputs.demands:
        raise ValueError("scenario requires nodes, positive alert_count, and demands")
    center_rng = named_rng(inputs.seed, "rescue-center")
    center = node_ids[int(center_rng.integers(0, len(node_ids)))]
    relay_rng = named_rng(inputs.seed, "relay-set")
    relays = [node_ids[index] for index in relay_rng.permutation(len(node_ids))[: max(1, len(node_ids) // 2)]]
    failure_rng = named_rng(inputs.seed, "failures")
    failures = [
        {**edge, "failed": bool(failure_rng.random() < inputs.failure_probability)}
        for edge in inputs.network["edges"]
    ]
    workload_rng = named_rng(inputs.seed, "workload")
    deadline_rng = named_rng(inputs.seed, "deadlines")
    alerts = []
    for index in range(inputs.alert_count):
        demand = inputs.demands[index % len(inputs.demands)]
        release = int(workload_rng.integers(0, 20))
        deadline = release + int(deadline_rng.integers(2, 11))
        weight = float(demand["original_value"])
        alerts.append(
            {
                "id": f"alert-{index:04d}",
                "source": node_ids[index % len(node_ids)],
                "release_slot": release,
                "deadline_slot": deadline,
                "size_bits": max(1, int(round(weight * 1_000_000))),
                "provenance": "derived-from-sndlib-demand",
            }
        )
    return {
        "schema_version": 1,
        "network": inputs.network,
        "rescue_center": center,
        "relays": sorted(relays),
        "alerts": alerts,
        "failures": failures,
        "targets": list(inputs.targets),
        "provenance": {
            "raw_sha256": sorted(inputs.raw_sha256),
            "config_hash": inputs.config_hash,
            "seed": inputs.seed,
            "generator_version": 1,
        },
    }
