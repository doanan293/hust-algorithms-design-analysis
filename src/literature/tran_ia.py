"""Inner approximation of Tran et al. (2022), Algorithm 3, on a TranInstance (spec D Sections 5.2–5.3).

Each iteration builds the convex program (58) with the linearization point as numeric constants (cvxpy parameters
of this size exhaust memory during canonicalization). Lengths are scaled by the UAV altitude and rates by B_tot.
The power constraint uses cvxpy's second-order-cone approximation and Clarabel runs with the tolerances below:
with exact power cones or default tolerances Clarabel stalls on development scenarios at slot resolution.
"""

from dataclasses import dataclass
import math
import warnings

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix

from baselines.trajectories import zone_tour_trajectory

from .tran_model import TranInstance, TranParams

SPEED_MARGIN = 1e-6
CLARABEL_SETTINGS = {"tol_gap_abs": 1e-6, "tol_gap_rel": 1e-6, "tol_feas": 1e-7}
CONVERGED = "converged"
MAX_ITERATIONS = "max_iterations"


@dataclass(frozen=True, eq=False)
class Iterate:
    """Trajectory at block boundaries (metres), bandwidth fraction per uplink and downlink entry, and lambda."""

    trajectory: np.ndarray
    uplink: np.ndarray
    downlink: np.ndarray
    lam: np.ndarray


@dataclass(frozen=True, eq=False)
class IaResult:
    iterate: Iterate
    objectives: tuple[float, ...]
    status: str
    iterations: int


def relaxed_objective(lam: np.ndarray, mu: float) -> float:
    """Tran eq. (26a): sum(lambda) + mu * sum(lambda * (lambda - 1))."""
    return float(np.sum(lam) + mu * np.sum(lam * (lam - 1.0)))


def _equal_split(blocks: np.ndarray, num_blocks: int) -> np.ndarray:
    counts = np.bincount(blocks, minlength=num_blocks)
    return 1.0 / counts[blocks] if len(blocks) else np.zeros(0)


def initial_iterate(instance: TranInstance) -> Iterate:
    """B1 zone tour at block boundaries, equal bandwidth split among entries of each block, lambda = 0."""
    trajectory = zone_tour_trajectory(instance.scenario)[:: instance.block_slots]
    return Iterate(
        np.asarray(trajectory, dtype=float),
        _equal_split(instance.uplink_block, instance.num_blocks),
        _equal_split(instance.downlink_block, instance.num_blocks),
        np.zeros(len(instance.devices)),
    )


def efficiency_tangent(snr_scaled: np.ndarray, z0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Value and slope of the convex efficiency log2(1 + c/z) at z0; the tangent is a global lower bound."""
    efficiency0 = np.log2(1.0 + snr_scaled / z0)
    slope = -snr_scaled / (math.log(2.0) * z0 * (z0 + snr_scaled))
    return efficiency0, slope


def product_bound(anchor, share, efficiency):
    """Concave lower bound of share * efficiency, tight where share + efficiency equals anchor (Tran eq. 57)."""
    return 0.25 * (2.0 * cp.multiply(anchor, share + efficiency) - anchor**2) - 0.25 * cp.square(share - efficiency)


def slacks(instance: TranInstance, trajectory: np.ndarray) -> np.ndarray:
    """Tight z = (1 + |m - w|^2)^(alpha/2) per point and block, lengths in altitude units."""
    unit = instance.scenario.uav.altitude_m
    midpoints = 0.5 * (trajectory[:-1] + trajectory[1:]) / unit
    xy = np.array([instance.scenario.position(point) for point in instance.points]).reshape(-1, 2) / unit
    squared = np.sum((midpoints[None, :, :] - xy[:, None, :]) ** 2, axis=2)
    return (1.0 + squared) ** (instance.channel.alpha / 2.0)


def _incidence(rows: np.ndarray, count: int) -> csr_matrix:
    return csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(count, len(rows)))


def _active_upper(instance: TranInstance) -> np.ndarray:
    return np.array([1.0 if device.active else 0.0 for device in instance.devices])


def solve_linearized(instance: TranInstance, params: TranParams, current: Iterate) -> tuple[str, Iterate | None]:
    """Build and solve program (58) linearized at `current`; return the solver status and the next iterate."""
    scenario = instance.scenario
    unit = scenario.uav.altitude_m
    G, P = instance.block_slots, instance.num_blocks
    K, point_count = len(instance.devices), len(instance.points)
    bandwidth = scenario.spectrum.b_tot_hz
    up_device, up_block, up_point = instance.uplink_device, instance.uplink_block, instance.uplink_point
    down_device, down_block, down_point = instance.downlink_device, instance.downlink_block, instance.downlink_point
    upper = _active_upper(instance)

    q = cp.Variable((P + 1, 2))
    a1 = cp.Variable(len(up_device), nonneg=True)
    a2 = cp.Variable(len(down_device), nonneg=True)
    r1 = cp.Variable(len(up_device))
    r2 = cp.Variable(len(down_device))
    lam = cp.Variable(K, nonneg=True)
    step = G * scenario.uav.v_max_mps * scenario.time.slot_s * (1.0 - SPEED_MARGIN) / unit
    constraints = [
        q[0] == np.asarray(scenario.uav.start_xy_m) / unit,
        q[P] == np.asarray(scenario.uav.end_xy_m) / unit,
        cp.norm(q[1:] - q[:-1], 2, axis=1) <= step,
        lam <= upper,
    ]
    delta_up = scenario.time.access_fraction * scenario.time.slot_s * G
    delta_down = (1.0 - scenario.time.access_fraction) * scenario.time.slot_s * G
    sizes = np.array([device.size_bits for device in instance.devices], dtype=float)
    ground = np.array([device.ground for device in instance.devices])
    up_need = upper * sizes / (delta_up * bandwidth)
    down_need = upper * np.where(ground, 0.0, sizes / (delta_down * bandwidth))

    if point_count:
        z = cp.Variable((point_count, P), nonneg=True)
        phi = cp.Variable((point_count, P))
        midpoints = 0.5 * (q[:-1] + q[1:])
        xy = np.array([scenario.position(point) for point in instance.points]) / unit
        for index in range(point_count):
            constraints.append(
                1.0 + cp.sum(cp.square(midpoints - xy[index]), axis=1)
                <= cp.power(z[index], 2.0 / instance.channel.alpha)
            )
        z0 = slacks(instance, current.trajectory)
        snr = (instance.point_snr / unit**instance.channel.alpha)[:, None]
        efficiency0, slope = efficiency_tangent(snr, z0)
        constraints.append(phi <= efficiency0 - slope * z0 + cp.multiply(slope, z))
        phi_flat = cp.reshape(phi, (point_count * P,), order="C")
        for a, r, blocks, point_of, selected, shares in (
            (a1, r1, up_block, up_point, np.flatnonzero(up_point >= 0), current.uplink),
            (a2, r2, down_block, down_point, np.arange(len(down_device)), current.downlink),
        ):
            if not len(selected):
                continue
            anchor = shares[selected] + efficiency0[point_of[selected], blocks[selected]]
            efficiency = phi_flat[point_of[selected] * P + blocks[selected]]
            share = a[selected]
            constraints.append(r[selected] <= product_bound(anchor, share, efficiency))
    ground_up = np.flatnonzero(up_point < 0)
    if len(ground_up):
        efficiency = np.array([instance.devices[device].ground_efficiency for device in up_device[ground_up]])
        constraints.append(r1[ground_up] <= cp.multiply(efficiency, a1[ground_up]))
    if len(up_device):
        constraints += [
            _incidence(up_device, K) @ r1 >= cp.multiply(up_need, lam),
            _incidence(up_block, P) @ a1 <= 1.0,
        ]
    if len(down_device):
        constraints += [
            _incidence(down_device, K) @ r2 >= cp.multiply(down_need, lam),
            _incidence(down_block, P) @ a2 <= 1.0,
        ]
    problem = cp.Problem(cp.Maximize(cp.sum(lam) + params.mu * 2.0 * (current.lam @ lam)), constraints)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Power atom", category=UserWarning)
        problem.solve(solver=cp.CLARABEL, ignore_dpp=True, canon_backend=cp.SCIPY_CANON_BACKEND, **CLARABEL_SETTINGS)
    if problem.status != cp.OPTIMAL:
        return problem.status, None
    trajectory = np.asarray(q.value, dtype=float) * unit
    trajectory[0] = scenario.uav.start_xy_m
    trajectory[-1] = scenario.uav.end_xy_m
    shares = []
    for variable, blocks in ((a1, up_block), (a2, down_block)):
        if not len(blocks):
            shares.append(np.zeros(0))
            continue
        values = np.clip(np.asarray(variable.value, dtype=float), 0.0, None)
        totals = np.bincount(blocks, weights=values, minlength=P)
        shares.append(values / np.maximum(totals, 1.0)[blocks])
    lam_value = np.clip(np.asarray(lam.value, dtype=float), 0.0, upper)
    return problem.status, Iterate(trajectory, shares[0], shares[1], lam_value)


def run_inner_approximation(instance: TranInstance, initial: Iterate, params: TranParams) -> IaResult:
    """Repeat: linearize at the accepted iterate, solve (58), accept; stop on small change, the cap, or a solver failure."""
    current = initial
    objectives = [relaxed_objective(initial.lam, params.mu)]
    for iteration in range(1, params.max_iterations + 1):
        status, following = solve_linearized(instance, params, current)
        if following is None:
            return IaResult(current, tuple(objectives), f"solver_{status}", iteration - 1)
        current = following
        objectives.append(relaxed_objective(current.lam, params.mu))
        if abs(objectives[-1] - objectives[-2]) <= params.tolerance * max(1.0, abs(objectives[-2])):
            return IaResult(current, tuple(objectives), CONVERGED, iteration)
    return IaResult(current, tuple(objectives), MAX_ITERATIONS, params.max_iterations)
