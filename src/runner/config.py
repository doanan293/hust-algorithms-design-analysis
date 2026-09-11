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
