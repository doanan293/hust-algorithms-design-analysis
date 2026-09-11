import math

import pytest

from runner.aggregate import (
    cluster_bootstrap_ci,
    crosscheck_violations,
    relative_gap,
    summarize,
    validity_violations,
)
from runner.config import BootstrapConfig


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
    [row] = summarize(method_rows, bound_rows, BootstrapConfig(100, 3, 0.95))
    assert row["timely_ratio_mean"] == pytest.approx((0.75 + 0.0) / 2)
    assert row["bound_ratio_mean"] == pytest.approx(0.5)
    assert row["gap_mean"] == pytest.approx(0.25)
    assert row["gap_undefined_count"] == 1
    assert row["shortfall_mean"] == pytest.approx((1.0 + 4.0) / 2)
    assert validity_violations(method_rows, bound_rows) == []


def test_validity_and_crosscheck_violations_are_reported():
    method_rows = [_method_row("A", "T1", 0, 3)]
    assert "exceed the bound" in validity_violations(method_rows, [_bound_row("A", 0, 2.0)])[0]
    assert "missing ground_only bound" in validity_violations(method_rows, [])[0]
    bad_status = dict(_bound_row("A", 0, 5.0), status="status 2: infeasible")
    assert any("infeasible" in item for item in validity_violations(method_rows, [bad_status]))
    good = {"set_id": "s", "scenario_id": "A", "lp_ub": 5.0, "lp_status": "optimal", "milp_incumbent": 4.0,
            "milp_dual_bound": 4.5, "milp_status": "time_limit", "milp_plan_value": 3.0, "method_value": 2.0}
    assert crosscheck_violations([good]) == []
    assert "chain broken" in crosscheck_violations([dict(good, milp_plan_value=5.0)])[0]
    assert "chain broken" in crosscheck_violations([dict(good, method_value=4.6)])[0]
    assert "incomplete" in crosscheck_violations([dict(good, milp_plan_value=math.nan)])[0]
