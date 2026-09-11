"""Tour family of the trajectory block: feasible by construction (spec C Section 7.3)."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
import math

import numpy as np

from models.scenario import Scenario

HOVER_SPLITS = ("uniform", "load", "urgency")
HOVER_POINTS = ("zone_center", "source_centroid")


@dataclass(frozen=True)
class TourSpec:
    zones: tuple[int, ...]
    split: str
    hover: str


def zone_orders(zone_count: int) -> list[tuple[int, ...]]:
    """Every non-empty subset in every order: subset size descending, then lexicographic."""
    orders = [order for size in range(1, zone_count + 1) for subset in combinations(range(zone_count), size) for order in permutations(subset)]
    return sorted(orders, key=lambda order: (-len(order), order))


def hover_point(scenario: Scenario, zone: int, hover: str) -> np.ndarray:
    if hover == "source_centroid":
        points = [source.xy for source in scenario.sources if source.zone == zone]
        if points:
            return np.mean(np.asarray(points, dtype=float), axis=0)
    elif hover != "zone_center":
        raise ValueError(f"unknown hover point {hover!r}")
    return np.array([scenario.damage_zones[zone].x_m, scenario.damage_zones[zone].y_m])


def split_weights(scenario: Scenario, zones: tuple[int, ...], split: str) -> list[float]:
    """Per-stop hover weights; a zero weight sum falls back to uniform."""
    if split == "uniform":
        return [1.0] * len(zones)
    if split not in HOVER_SPLITS:
        raise ValueError(f"unknown hover split {split!r}")
    zone_of = {source.id: source.zone for source in scenario.sources}
    weights = []
    for zone in zones:
        alerts = [alert for alert in scenario.alerts if zone_of[alert.source] == zone]
        if split == "load":
            weights.append(float(len(alerts)))
        else:
            weights.append(sum(1.0 / (alert.deadline_slot - alert.release_slot) for alert in alerts))
    return weights if sum(weights) > 0.0 else [1.0] * len(zones)


def hover_slots(spare: int, weights: list[float]) -> list[int]:
    total = sum(Fraction(weight) for weight in weights)
    return [math.floor(spare * Fraction(weight) / total) for weight in weights]


def tour_trajectory(scenario: Scenario, spec: TourSpec) -> np.ndarray | None:
    """Straight segments at V_max, hover at each stop, remaining slots at the center; None if travel exceeds N."""
    num_slots = scenario.time.num_slots
    step_m = scenario.uav.v_max_mps * scenario.time.slot_s
    center = np.asarray(scenario.uav.start_xy_m, dtype=float)
    stops = [center, *(hover_point(scenario, zone, spec.hover) for zone in spec.zones), center]
    travel = [math.ceil(float(np.linalg.norm(stop - start)) / step_m) for start, stop in zip(stops, stops[1:])]
    if sum(travel) > num_slots:
        return None
    hovers = hover_slots(num_slots - sum(travel), split_weights(scenario, spec.zones, spec.split))
    points = [center]
    for index, (start, stop, steps) in enumerate(zip(stops, stops[1:], travel)):
        points.extend(start + (stop - start) * step / steps for step in range(1, steps + 1))
        if index < len(spec.zones):
            points.extend([stop] * hovers[index])
    points.extend([center] * (num_slots + 1 - len(points)))
    return np.asarray(points)


def tour_family(scenario: Scenario) -> list[tuple[TourSpec, np.ndarray]]:
    """Feasible tours in enumeration order; a tour identical to an earlier one is dropped."""
    family: list[tuple[TourSpec, np.ndarray]] = []
    for zones in zone_orders(len(scenario.damage_zones)):
        for split in HOVER_SPLITS:
            for hover in HOVER_POINTS:
                spec = TourSpec(zones, split, hover)
                trajectory = tour_trajectory(scenario, spec)
                if trajectory is not None and not any(np.array_equal(trajectory, other) for _, other in family):
                    family.append((spec, trajectory))
    return family
