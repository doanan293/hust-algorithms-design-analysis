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
