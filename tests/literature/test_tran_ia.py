import numpy as np
import pytest

from baselines.trajectories import zone_tour_trajectory
import literature.tran_ia as tran_ia
from literature.tran_ia import (
    CONVERGED, MAX_ITERATIONS, Iterate, efficiency_tangent, initial_iterate, product_bound, relaxed_objective,
    run_inner_approximation,
)
from literature.tran_model import TranParams, build_instance
from models.paths import candidate_paths
from models.scenario import scenario_from_dict
from tran_builders import tran_raw


def _instance(**raw_changes):
    scenario = scenario_from_dict(tran_raw(**raw_changes))
    return build_instance(scenario, candidate_paths(scenario), TranParams())


def test_initial_iterate_is_the_zone_tour_with_equal_shares_and_zero_lambda():
    instance = _instance()
    initial = initial_iterate(instance)
    assert np.array_equal(initial.trajectory, zone_tour_trajectory(instance.scenario))
    totals = np.bincount(instance.uplink_block, weights=initial.uplink, minlength=instance.num_blocks)
    assert np.allclose(totals[np.bincount(instance.uplink_block, minlength=instance.num_blocks) > 0], 1.0)
    assert initial.lam.tolist() == [0.0, 0.0, 0.0] and relaxed_objective(initial.lam, 1.0) == 0.0


def test_efficiency_tangent_is_a_tight_global_lower_bound():
    snr, z0 = np.array([4.7]), np.array([3.0])
    value, slope = efficiency_tangent(snr, z0)
    z = np.linspace(1.0, 50.0, 200)
    assert value[0] == pytest.approx(np.log2(1.0 + 4.7 / 3.0))
    assert np.all(np.log2(1.0 + 4.7 / z) >= value[0] + slope[0] * (z - 3.0) - 1e-12)


def test_product_bound_is_tight_at_the_anchor_and_below_the_product():
    share, efficiency = np.array([0.3, 0.7]), np.array([2.0, 0.1])
    assert product_bound(share + efficiency, share, efficiency).value == pytest.approx(share * efficiency)
    rng = np.random.default_rng(3)
    shares, efficiencies = rng.uniform(0.0, 1.0, 50), rng.uniform(0.0, 3.0, 50)
    assert np.all(product_bound(np.full(50, 1.2), shares, efficiencies).value <= shares * efficiencies + 1e-12)


def test_inner_approximation_is_monotone_bounded_and_deterministic():
    instance = _instance(uav_size=12000000)
    params = TranParams(max_iterations=6)
    first = run_inner_approximation(instance, initial_iterate(instance), params)
    second = run_inner_approximation(instance, initial_iterate(instance), params)
    assert first.status in (CONVERGED, MAX_ITERATIONS) and first.iterations >= 1
    assert np.all(np.diff(first.objectives) >= -1e-5)
    assert np.all((first.iterate.lam >= 0.0) & (first.iterate.lam <= 1.0))
    assert np.array_equal(first.iterate.trajectory, second.iterate.trajectory)
    assert np.array_equal(first.iterate.uplink, second.iterate.uplink) and first.objectives == second.objectives


def _scripted(monkeypatch, steps):
    calls = iter(steps)
    monkeypatch.setattr(tran_ia, "solve_linearized", lambda instance, params, current: next(calls)(current))


def test_solver_failure_keeps_the_last_accepted_iterate(monkeypatch):
    instance = _instance()
    initial = initial_iterate(instance)
    accepted = Iterate(initial.trajectory, initial.uplink, initial.downlink, np.array([1.0, 0.0, 0.0]))
    _scripted(monkeypatch, [lambda current: ("optimal", accepted), lambda current: ("infeasible", None)])
    result = run_inner_approximation(instance, initial, TranParams())
    assert (result.status, result.iterations, result.objectives) == ("solver_infeasible", 1, (0.0, 1.0))
    assert result.iterate is accepted


def test_stops_on_small_change_or_at_the_iteration_cap(monkeypatch):
    instance = _instance()
    initial = initial_iterate(instance)
    _scripted(monkeypatch, [lambda current: ("optimal", current)])
    assert run_inner_approximation(instance, initial, TranParams()).status == CONVERGED
    growing = [lambda current, k=k: ("optimal", Iterate(current.trajectory, current.uplink, current.downlink, np.array([k / 3.0, 0.0, 0.0])))
               for k in (1, 2, 3)]
    _scripted(monkeypatch, growing)
    result = run_inner_approximation(instance, initial, TranParams(max_iterations=3, mu=0.001))
    assert (result.status, result.iterations, len(result.objectives)) == (MAX_ITERATIONS, 3, 4)
