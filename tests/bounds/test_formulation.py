import copy

import numpy as np
import pytest

from bounds.formulation import build_bound_model, rate_tangent, restrict_to_ground, tangent_points
from bounds.solve import OPTIMAL, highs_version, solve_lp, solve_milp
from models.channel import LinkChannel, build_link_channels, channel_uniforms, rate_bps
from models.evaluate import REALIZED, evaluate
from models.paths import candidate_paths, radio_links
from models.plan import EqualSplitBacklogged, Plan, StaticSchedule, stationary_trajectory
from models.scenario import UAV_ID, scenario_from_dict
from channel_builders import constant_channels
from scenario_builders import BASE_SCENARIO, make_alert


def _scenario(alerts):
    raw = copy.deepcopy(BASE_SCENARIO)
    raw["alerts"] = alerts
    return scenario_from_dict(raw)


def test_tangent_envelope_lies_above_the_rate():
    for snr in (1e3, 1e6, 1e9):
        for point in tangent_points(1e6, 10):
            value, slope = rate_tangent(point, snr)
            assert value == pytest.approx(rate_bps(point, snr))
            for bandwidth in np.geomspace(1e2, 1e6, 60):
                assert value + slope * (bandwidth - point) >= rate_bps(bandwidth, snr) - 1e-6


@pytest.mark.parametrize(("deadline", "size", "expected"), [(2, 1000000, 0.5), (3, 1000000, 1.0), (2, 500000, 1.0)])
def test_single_ground_path_bound_matches_store_and_forward_capacity(deadline, size, expected):
    scenario = _scenario([make_alert("alert-0000", "s00", 0, deadline, size)])
    only_n1 = {"alert-0000": (candidate_paths(scenario)["alert-0000"][0],)}
    assert only_n1["alert-0000"][0].entry == "n1"
    model = build_bound_model(scenario, only_n1, constant_channels(scenario, only_n1, 1e6), tangents=10)
    result = solve_lp(model)
    assert result.status == OPTIMAL
    assert result.value == pytest.approx(expected, abs=1e-7)


def test_ground_only_bound_uses_no_uav_link_and_milp_stays_below_lp():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 6, 900000), make_alert("alert-0001", "s00", 0, 6, 900000)])
    candidates = restrict_to_ground(candidate_paths(scenario))
    model = build_bound_model(scenario, candidates, constant_channels(scenario, candidates, 1e6), tangents=10)
    assert not any(UAV_ID in key[1] for key in model.columns if key[0] == "b")
    lp = solve_lp(model)
    milp = solve_milp(model, time_limit_s=30)
    assert milp.status == OPTIMAL
    assert milp.value <= lp.value + 1e-6
    assert milp.dual_bound >= milp.value - 1e-6


def test_bound_rejects_expected_rate_channels():
    scenario = _scenario([make_alert("alert-0000", "s00", 0, 6, 100000)])
    candidates = candidate_paths(scenario)
    channels = build_link_channels(scenario, radio_links(candidates), stationary_trajectory(scenario), None)
    with pytest.raises(ValueError, match="realized channels"):
        build_bound_model(scenario, candidates, channels, tangents=10)


def test_bound_is_at_least_every_evaluated_random_plan():
    scenario = _scenario(
        [
            make_alert("alert-0000", "s00", 0, 12, 400000),
            make_alert("alert-0001", "s00", 3, 20, 700000),
            make_alert("alert-0002", "s01", 1, 18, 50000),
            make_alert("alert-0003", "s01", 6, 20, 90000),
        ]
    )
    candidates = candidate_paths(scenario)
    links = radio_links(candidates)
    rng = np.random.default_rng(7)
    for trial in range(8):
        steps = rng.uniform(-35.0, 35.0, size=(10, 2))
        outward = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
        trajectory = np.asarray(scenario.uav.start_xy_m) + np.vstack([outward, outward[-2::-1]])
        choice = {alert.id: int(rng.integers(0, len(candidates[alert.id]))) for alert in scenario.alerts}
        policy = EqualSplitBacklogged() if trial % 2 == 0 else StaticSchedule({link: np.full(20, 1e6 / len(links)) for link in links})
        if isinstance(policy, StaticSchedule):
            downlinks = [link for link in links if link[0] == "down"]
            for link in downlinks[2:]:
                policy.bandwidth_hz[link][:] = 0.0
        realization = int(rng.integers(0, 5))
        result = evaluate(scenario, Plan(trajectory, choice, policy), REALIZED, (realization,), candidates=candidates)
        assert result.feasible, result.violations
        channels = build_link_channels(scenario, links, trajectory, channel_uniforms(scenario, realization))
        bound = solve_lp(build_bound_model(scenario, candidates, channels, tangents=10))
        assert bound.value >= sum(result.realizations[0].timely.values()) - 1e-6


def test_highs_version_is_reported():
    major, minor, patch = highs_version().split(".")
    assert int(major) >= 1


def _one_variable_model():
    from scipy.sparse import csr_matrix

    from bounds.formulation import BoundModel

    return BoundModel(
        objective=np.array([-1.0]), a_ub=csr_matrix(np.array([[1.0]])), b_ub=np.array([0.5]), a_eq=csr_matrix((0, 1)),
        b_eq=np.array([]), upper=np.array([1.0]), integrality=np.array([0]), columns={},
    )


def test_solve_lp_resolves_without_presolve_when_highs_cannot_classify(monkeypatch):
    from types import SimpleNamespace

    from bounds import solve

    calls = []

    def flaky_linprog(*args, **kwargs):
        calls.append(kwargs.get("options"))
        if len(calls) == 1:
            return SimpleNamespace(status=4, message="model_status is Unknown", fun=None, x=None)
        return real_linprog(*args, **kwargs)

    real_linprog = solve.linprog
    monkeypatch.setattr(solve, "linprog", flaky_linprog)
    result = solve_lp(_one_variable_model())
    assert (result.status, result.value) == (OPTIMAL, pytest.approx(0.5))
    assert calls == [None, {"presolve": False}]


def test_solve_lp_keeps_an_optimal_first_solve(monkeypatch):
    from bounds import solve

    calls = []
    real_linprog = solve.linprog
    monkeypatch.setattr(solve, "linprog", lambda *args, **kwargs: calls.append(kwargs.get("options")) or real_linprog(*args, **kwargs))
    assert solve_lp(_one_variable_model()).status == OPTIMAL
    assert calls == [None]
