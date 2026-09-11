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
