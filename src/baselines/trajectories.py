import math

import numpy as np

from models.scenario import Scenario


def zone_tour_trajectory(scenario: Scenario) -> np.ndarray:
    num_slots = scenario.time.num_slots
    step_m = scenario.uav.v_max_mps * scenario.time.slot_s
    center = np.asarray(scenario.uav.start_xy_m, dtype=float)
    remaining = [np.array([zone.x_m, zone.y_m]) for zone in scenario.damage_zones]
    order: list[np.ndarray] = []
    current = center
    while remaining:
        index = min(range(len(remaining)), key=lambda item: (float(np.linalg.norm(remaining[item] - current)), item))
        current = remaining.pop(index)
        order.append(current)
    while True:
        stops = [center, *order, center]
        travel = [math.ceil(float(np.linalg.norm(stop - start)) / step_m) for start, stop in zip(stops, stops[1:])]
        if sum(travel) <= num_slots or not order:
            break
        order.pop()
    hover = (num_slots - sum(travel)) // len(order) if order else 0
    points = [center]
    for index, (start, stop, steps) in enumerate(zip(stops, stops[1:], travel)):
        points.extend(start + (stop - start) * step / steps for step in range(1, steps + 1))
        if index < len(order):
            points.extend([stop] * hover)
    points.extend([center] * (num_slots + 1 - len(points)))
    return np.asarray(points)
