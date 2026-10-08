"""Segment shifts of the incumbent trajectory toward congested UAV links (spec E Section 6)."""

from dataclasses import dataclass
import math

import numpy as np

from models.channel import ACCESS
from models.paths import radio_links_of_paths
from models.plan import Plan
from models.scenario import UAV_ID, Scenario

from .ledger_summary import LedgerSummary, alert_window

MIN_SHIFT_M = 1e-6
STEP_MARGIN = 1.0 - 1e-9


def max_shift(anchor: np.ndarray, point: np.ndarray, direction: np.ndarray, step_m: float) -> float:
    """Largest λ ≥ 0 with ‖point + λ·direction − anchor‖ ≤ step_m for a unit direction; 0 when none exists."""
    offset = np.asarray(point, dtype=float) - np.asarray(anchor, dtype=float)
    b = float(np.dot(offset, direction))
    c = float(np.dot(offset, offset)) - step_m * step_m
    discriminant = b * b - c
    if discriminant < 0.0:
        return 0.0
    return max(-b + math.sqrt(discriminant), 0.0)


def shift_segment(
    trajectory: np.ndarray, start: int, stop: int, direction: np.ndarray, fraction: float, step_m: float
) -> np.ndarray | None:
    """q[start..stop] moved by fraction·λ* along `direction`; None when λ* is (numerically) zero."""
    last = len(trajectory) - 1
    if not 1 <= start <= stop <= last - 1:
        raise ValueError(f"segment [{start}, {stop}] must lie inside [1, {last - 1}]")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("direction must be non-zero")
    unit = np.asarray(direction, dtype=float) / norm
    limit = step_m * STEP_MARGIN
    reach = min(max_shift(trajectory[start - 1], trajectory[start], unit, limit), max_shift(trajectory[stop + 1], trajectory[stop], unit, limit))
    if reach < MIN_SHIFT_M:
        return None
    shifted = np.array(trajectory, dtype=float, copy=True)
    shifted[start : stop + 1] += fraction * reach * unit
    return shifted


def feasible_segment(
    trajectory: np.ndarray, start: int, stop: int, direction: np.ndarray, step_m: float, max_widening: int
) -> tuple[int, int] | None:
    """Widen [start, stop] by one point per side, at most `max_widening` times, until a shift is possible."""
    last = len(trajectory) - 1
    for _ in range(max_widening + 1):
        if shift_segment(trajectory, start, stop, direction, 1.0, step_m) is not None:
            return start, stop
        widened = (max(start - 1, 1), min(stop + 1, last - 1))
        if widened == (start, stop):
            return None
        start, stop = widened
    return None


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int
    direction: np.ndarray
    priority: float


def sample_uav_segments(
    scenario: Scenario, plan: Plan, summary: LedgerSummary, rng: np.random.Generator, count: int
) -> list[Segment]:
    """Runs of congested slots on UAV links, drawn with probability ∝ priority, returned in decreasing priority."""
    num_slots = scenario.time.num_slots
    last = num_slots
    candidates: list[Segment] = []
    for link in radio_links_of_paths(plan.paths):
        if UAV_ID not in (link[1], link[2]):
            continue
        urgency = np.zeros(num_slots)
        for alert in scenario.alerts:
            path = plan.paths[alert.id]
            if path is not None and link in path.links:
                window = alert_window(scenario, alert)
                urgency[window.start : window.stop] += alert.size_bits / (alert.deadline_slot - alert.release_slot)
        congested = np.array([summary.congested_realizations(link, slot) for slot in range(num_slots)], dtype=float)
        priority = congested * urgency
        ground_id = link[1] if link[0] == ACCESS else link[2]
        ground = np.asarray(scenario.position(ground_id), dtype=float)
        slot = 0
        while slot < num_slots:
            if priority[slot] <= 0.0:
                slot += 1
                continue
            run_start = slot
            while slot < num_slots and priority[slot] > 0.0:
                slot += 1
            start, stop = max(run_start, 1), min(slot, last - 1)
            if start > stop:
                continue
            centroid = plan.trajectory[start : stop + 1].mean(axis=0)
            direction = ground - centroid
            norm = float(np.linalg.norm(direction))
            if norm > 0.0:
                candidates.append(Segment(start, stop, direction / norm, float(priority[run_start:slot].sum())))
    if not candidates:
        return []
    weights = np.array([segment.priority for segment in candidates])
    drawn = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False, p=weights / weights.sum())
    return sorted((candidates[index] for index in drawn), key=lambda segment: (-segment.priority, segment.start))
