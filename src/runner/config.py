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
    """A generated scenario set; `replicates` keeps only the listed replicate numbers (all when None)."""

    set_id: str
    manifest: Path
    scenario_dir: Path
    replicates: tuple[int, ...] | None = None


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
        ScenarioSetRef(str(item["set_id"]), Path(item["manifest"]), Path(item["scenario_dir"]), _replicates(item, index))
        for index, item in enumerate(raw["scenario_sets"])
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
