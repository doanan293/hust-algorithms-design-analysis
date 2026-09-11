import math

import pytest

from runner.aggregate import (
    cluster_bootstrap_ci,
    crosscheck_violations,
    relative_gap,
    summarize,
    union_bound_ratios,
    validity_violations,
)
from runner.config import BootstrapConfig

OWNERS = {"B0": ("ground_only", "")}


def _method_row(scenario, topology, realization, timely):
    return {
        "set_id": "s", "scenario_id": scenario, "topology_id": topology, "method": "B0", "realization_id": realization,
        "alert_count": 4, "timely_count": timely, "timely_ratio": timely / 4, "source_count": 2, "connected_count": 1,
        "conn_ratio": 0.5, "shortfall": 4 - timely, "planning_runtime_s": 0.1, "evaluation_runtime_s": 0.2, "evaluate_calls": 1,
    }


def _bound_row(scenario, realization, value):
    return {
        "set_id": "s", "scenario_id": scenario, "topology_id": "T", "variant": "ground_only", "trajectory_method": "",
        "realization_id": realization, "value": value, "ratio": value / 4, "status": "optimal", "runtime_s": 0.3,
        "variables": 1, "rows": 1, "nonzeros": 1,
    }


def test_bootstrap_interval_and_gap_helpers():
    assert cluster_bootstrap_ci({"a": [0.4, 0.4], "b": [0.4]}, 50, 1, 0.95) == pytest.approx((0.4, 0.4))
    low, high = cluster_bootstrap_ci({"a": [0.0], "b": [1.0]}, 400, 1, 0.9)
    assert (low, high) == cluster_bootstrap_ci({"a": [0.0], "b": [1.0]}, 400, 1, 0.9)
    assert 0.0 <= low <= 0.5 <= high <= 1.0
    assert relative_gap(0.5, 0.25) == 0.5
    assert relative_gap(0.0, 0.0) is None


def test_summary_means_gaps_and_undefined_gap_count():
    method_rows = [_method_row("A", "T1", 0, 2), _method_row("A", "T1", 1, 4), _method_row("B", "T2", 0, 0), _method_row("B", "T2", 1, 0)]
    bound_rows = [_bound_row("A", 0, 4.0), _bound_row("A", 1, 4.0), _bound_row("B", 0, 0.0), _bound_row("B", 1, 0.0)]
    [row] = summarize(method_rows, bound_rows, BootstrapConfig(100, 3, 0.95), OWNERS)
    assert row["timely_ratio_mean"] == pytest.approx((0.75 + 0.0) / 2)
    assert row["bound_ratio_mean"] == pytest.approx(0.5)
    assert row["gap_mean"] == pytest.approx(0.25)
    assert row["gap_undefined_count"] == 1
    assert row["shortfall_mean"] == pytest.approx((1.0 + 4.0) / 2)
    assert validity_violations(method_rows, bound_rows, OWNERS) == []


def test_validity_and_crosscheck_violations_are_reported():
    method_rows = [_method_row("A", "T1", 0, 3)]
    assert "exceed the bound" in validity_violations(method_rows, [_bound_row("A", 0, 2.0)], OWNERS)[0]
    assert "missing ground_only bound" in validity_violations(method_rows, [], OWNERS)[0]
    bad_status = dict(_bound_row("A", 0, 5.0), status="status 2: infeasible")
    assert any("infeasible" in item for item in validity_violations(method_rows, [bad_status], OWNERS))
    good = {"set_id": "s", "scenario_id": "A", "lp_ub": 5.0, "lp_status": "optimal", "milp_incumbent": 4.0,
            "milp_dual_bound": 4.5, "milp_status": "time_limit", "milp_plan_value": 3.0, "method_value": 2.0}
    assert crosscheck_violations([good]) == []
    assert "chain broken" in crosscheck_violations([dict(good, milp_plan_value=5.0)])[0]
    assert "chain broken" in crosscheck_violations([dict(good, method_value=4.6)])[0]
    assert "incomplete" in crosscheck_violations([dict(good, milp_plan_value=math.nan)])[0]


def _trajectory_row(scenario, realization, owner, value):
    return dict(_bound_row(scenario, realization, value), variant="trajectory", trajectory_method=owner)


def test_union_bound_is_the_per_realization_maximum_over_trajectories():
    bound_rows = [
        _trajectory_row("A", 0, "B1", 2.0), _trajectory_row("A", 0, "P", 3.0),
        _trajectory_row("A", 1, "B1", 4.0), _trajectory_row("A", 1, "P", 3.0),
        _bound_row("A", 0, 4.0),
    ]
    assert union_bound_ratios(bound_rows) == {("s", "A", 0): 0.75, ("s", "A", 1): 1.0}
    method_rows = [dict(_method_row("A", "T1", realization, 2), method="P") for realization in (0, 1)]
    owners = {"P": ("trajectory", "P")}
    [row] = summarize(method_rows, bound_rows, BootstrapConfig(50, 3, 0.95), owners)
    assert row["bound_ratio_mean"] == pytest.approx(0.75)
    assert row["union_bound_ratio_mean"] == pytest.approx(0.875)
    assert row["union_gap_mean"] == pytest.approx((0.875 - 0.5) / 0.875)
    assert row["evaluate_calls_mean"] == 1.0
    assert validity_violations(method_rows, bound_rows, owners) == []


def test_methods_without_bounds_are_summarized_without_gaps_or_validity_checks():
    method_rows = [dict(_method_row("A", "T1", 0, 4), method="P_op1")]
    [row] = summarize(method_rows, [], BootstrapConfig(50, 3, 0.95), {"P_op1": None})
    assert math.isnan(row["bound_ratio_mean"]) and math.isnan(row["gap_mean"]) and math.isnan(row["union_gap_mean"])
    assert row["gap_undefined_count"] == 1
    assert validity_violations(method_rows, [], {"P_op1": None}) == []
