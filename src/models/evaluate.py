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
    channel_namespace: str = "eval"


def evaluate(
    scenario: Scenario,
    plan: Plan,
    mode: str,
    realization_ids: Sequence[int] = (),
    counter: EvaluationCounter | None = None,
    candidates: Mapping[str, tuple[CandidatePath, ...]] | None = None,
    keep_ledger: bool = False,
    channel_namespace: str = "eval",
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
            channel_namespace=channel_namespace,
        )

    links = set(radio_links(candidates)) | set(all_uav_links(scenario))
    without_uav = connectivity(scenario, None)
    results = []
    for realization_id in realization_ids if mode == REALIZED else (None,):
        uniforms = None if realization_id is None else channel_uniforms(scenario, realization_id, channel_namespace)
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
        channel_namespace=channel_namespace,
    )
