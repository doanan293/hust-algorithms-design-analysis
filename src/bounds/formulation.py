from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from models.channel import ACCESS, BACKHAUL, LinkChannel, LinkId
from models.paths import GROUND, CandidatePath
from models.scenario import Scenario


def rate_tangent(bandwidth_hz: float, snr_per_hz: float) -> tuple[float, float]:
    """Value and slope of R(b) = b*log2(1 + s/b) at b > 0."""
    ratio = snr_per_hz / bandwidth_hz
    value = bandwidth_hz * math.log2(1.0 + ratio)
    slope = math.log2(1.0 + ratio) - ratio / ((1.0 + ratio) * math.log(2.0))
    return value, slope


def tangent_points(b_tot_hz: float, tangents: int) -> tuple[float, ...]:
    return tuple(b_tot_hz * 2.0 ** (-index) for index in range(tangents))


def restrict_to_ground(
    candidates: Mapping[str, tuple[CandidatePath, ...]],
) -> dict[str, tuple[CandidatePath, ...]]:
    return {alert_id: tuple(path for path in paths if path.kind == GROUND) for alert_id, paths in candidates.items()}


def realized_snr(channel: LinkChannel, slot: int) -> float:
    weight = float(channel.los_weight[slot])
    if weight not in (0.0, 1.0):
        raise ValueError("the bound needs realized channels with LoS weight 0 or 1")
    return float(channel.snr_los_per_hz[slot] if weight == 1.0 else channel.snr_nlos_per_hz[slot])


@dataclass(frozen=True)
class BoundModel:
    """minimize objective @ v  s.t.  a_ub v <= b_ub, a_eq v = b_eq, 0 <= v <= upper."""

    objective: np.ndarray
    a_ub: csr_matrix
    b_ub: np.ndarray
    a_eq: csr_matrix
    b_eq: np.ndarray
    upper: np.ndarray
    integrality: np.ndarray
    columns: Mapping[tuple, int]

    @property
    def variables(self) -> int:
        return len(self.objective)

    @property
    def rows(self) -> int:
        return self.a_ub.shape[0] + self.a_eq.shape[0]

    @property
    def nonzeros(self) -> int:
        return self.a_ub.nnz + self.a_eq.nnz


class _Rows:
    def __init__(self) -> None:
        self.row_index: list[int] = []
        self.column_index: list[int] = []
        self.values: list[float] = []
        self.rhs: list[float] = []

    def add(self, coefficients: Mapping[int, float], rhs: float) -> None:
        row = len(self.rhs)
        for column, value in coefficients.items():
            self.row_index.append(row)
            self.column_index.append(column)
            self.values.append(value)
        self.rhs.append(rhs)

    def matrix(self, width: int) -> tuple[csr_matrix, np.ndarray]:
        shape = (len(self.rhs), width)
        matrix = coo_matrix((self.values, (self.row_index, self.column_index)), shape=shape).tocsr()
        return matrix, np.asarray(self.rhs, dtype=float)


def build_bound_model(
    scenario: Scenario,
    candidates: Mapping[str, tuple[CandidatePath, ...]],
    channels: Mapping[LinkId, LinkChannel],
    tangents: int,
) -> BoundModel:
    slot_s = scenario.time.slot_s
    access_s = scenario.time.access_fraction * slot_s
    columns: dict[tuple, int] = {}

    def column(key: tuple) -> int:
        if key not in columns:
            columns[key] = len(columns)
        return columns[key]

    upper_rows, equal_rows = _Rows(), _Rows()
    link_flows: dict[tuple[LinkId, int], list[int]] = {}
    for alert in scenario.alerts:
        paths = candidates[alert.id]
        z = column(("z", alert.id))
        if paths:
            upper_rows.add({column(("x", alert.id, index)): 1.0 for index in range(len(paths))}, 1.0)
        delivered = {z: float(alert.size_bits)}
        for index, path in enumerate(paths):
            hops = len(path.links)
            for hop in range(hops):
                for slot in range(alert.release_slot + hop, alert.deadline_slot):
                    link_flows.setdefault((path.links[hop], slot), []).append(column(("f", alert.id, index, hop, slot)))
            for hop in range(hops):
                start = alert.release_slot + hop
                if start >= alert.deadline_slot:
                    break
                for slot in range(start, alert.deadline_slot):
                    flow = columns[("f", alert.id, index, hop, slot)]
                    buffer = column(("B", alert.id, index, hop, slot))
                    upper_rows.add({flow: 1.0, buffer: -1.0}, 0.0)
                    if slot + 1 < alert.deadline_slot:
                        balance = {column(("B", alert.id, index, hop, slot + 1)): 1.0, buffer: -1.0, flow: 1.0}
                        if hop > 0:
                            balance[columns[("f", alert.id, index, hop - 1, slot)]] = -1.0
                        equal_rows.add(balance, 0.0)
                first = columns[("B", alert.id, index, hop, start)]
                if hop == 0:
                    equal_rows.add({first: 1.0, columns[("x", alert.id, index)]: -float(alert.size_bits)}, 0.0)
                else:
                    equal_rows.add({first: 1.0, columns[("f", alert.id, index, hop - 1, start - 1)]: -1.0}, 0.0)
            last = hops - 1
            for slot in range(alert.release_slot + last, alert.deadline_slot):
                delivered[columns[("f", alert.id, index, last, slot)]] = -1.0
        upper_rows.add(delivered, 0.0)

    spectrum: dict[tuple[str, int], list[int]] = {}
    points = tangent_points(scenario.spectrum.b_tot_hz, tangents)
    for link, slot in sorted(link_flows):
        load = {flow: 1.0 for flow in link_flows[(link, slot)]}
        if link[0] == BACKHAUL:
            upper_rows.add(load, slot_s * scenario.backhaul_capacity_bps(link[1], link[2]))
            continue
        duration = access_s if link[0] == ACCESS else slot_s - access_s
        snr = realized_snr(channels[link], slot)
        if snr <= 0.0:
            upper_rows.add(load, 0.0)
            continue
        bandwidth = column(("b", link, slot))
        spectrum.setdefault((link[0], slot), []).append(bandwidth)
        for point in points:
            value, slope = rate_tangent(point, snr)
            upper_rows.add({**load, bandwidth: -duration * slope}, duration * (value - slope * point))
    for key in sorted(spectrum):
        upper_rows.add({index: 1.0 for index in spectrum[key]}, scenario.spectrum.b_tot_hz)

    width = len(columns)
    objective = np.zeros(width)
    upper = np.full(width, np.inf)
    integrality = np.zeros(width)
    for key, index in columns.items():
        if key[0] == "z":
            objective[index] = -1.0
        if key[0] in ("x", "z"):
            upper[index] = 1.0
            integrality[index] = 1
    a_ub, b_ub = upper_rows.matrix(width)
    a_eq, b_eq = equal_rows.matrix(width)
    return BoundModel(objective, a_ub, b_ub, a_eq, b_eq, upper, integrality, columns)
