import math

import numpy as np
import pytest

from analysis.statistics import (
    PRIMARY_COMPARISONS,
    STATISTICS_COLUMNS,
    Comparison,
    bootstrap_mean_ci,
    compare_methods,
    holm_adjust,
    rank_biserial,
    sign_flip_p_value,
    topology_means,
)


def rows_for(set_id, method, values_by_scenario):
    return [
        {"set_id": set_id, "scenario_id": scenario, "topology_id": scenario.split("-r")[0], "method": method,
         "realization_id": str(index), "timely_ratio": str(value)}
        for scenario, values in values_by_scenario.items()
        for index, value in enumerate(values)
    ]


def test_sign_flip_test_is_exact():
    assert sign_flip_p_value([0.1] * 10) == 2 / 1024
    assert sign_flip_p_value([0.2, -0.2] * 5) == 1.0
    assert sign_flip_p_value([0.3, 0.1, 0.2]) == 0.25
    with pytest.raises(ValueError, match="between 1 and 20"):
        sign_flip_p_value([0.1] * 21)


def test_holm_adjustment_matches_hand_examples():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_rank_biserial_matches_hand_examples():
    assert rank_biserial([1.0, -2.0, 3.0]) == pytest.approx(1 / 3)
    assert rank_biserial([0.0, 2.0, -2.0, 1.0]) == pytest.approx(1 / 6)
    assert math.isnan(rank_biserial([0.0, 0.0]))


def test_bootstrap_interval_is_deterministic_for_its_seed():
    differences = np.array([0.1, 0.3, -0.05, 0.2, 0.0, 0.15, 0.4, -0.1, 0.05, 0.25])
    first = bootstrap_mean_ci(differences, 2000, 20260911)
    assert first == bootstrap_mean_ci(differences, 2000, 20260911)
    assert first != bootstrap_mean_ci(differences, 2000, 7)
    assert first[0] < differences.mean() < first[1]
    assert bootstrap_mean_ci(np.full(10, 0.2), 500, 1) == pytest.approx((0.2, 0.2))


def test_topology_means_average_scenarios_then_topologies():
    rows = rows_for("v0", "B1", {"A-r0": [0.2, 0.4], "A-r1": [0.6, 0.6], "B-r0": [0.1, 0.3]})
    rows += rows_for("v0", "B2", {"A-r0": [1.0]}) + rows_for("v0-bh50", "B1", {"A-r0": [1.0]})
    assert topology_means(rows, "v0", "B1") == pytest.approx({"A": 0.45, "B": 0.2})


def test_compare_methods_reports_every_column_with_holm():
    topologies = [f"T{index}" for index in range(10)]
    rows = rows_for("v0", "B2", {f"{name}-r0": [0.5] for name in topologies})
    rows += rows_for("v0", "P", {f"{name}-r0": [0.75] for name in topologies})
    rows += rows_for("v0", "B1", {f"{name}-r0": [0.375 if index % 2 == 0 else 0.625] for index, name in enumerate(topologies)})
    first, second = compare_methods(rows, (Comparison("v0", "P", "B2"), Comparison("v0", "B2", "B1")), 500, 3)
    assert tuple(first) == STATISTICS_COLUMNS
    assert (first["comparison"], first["mean_difference"], first["topology_count"]) == ("P - B2", 0.25, 10)
    assert (first["ci_low"], first["ci_high"], first["rank_biserial"]) == pytest.approx((0.25, 0.25, 1.0))
    assert (first["p_value"], first["p_holm"]) == pytest.approx((2 / 1024, 4 / 1024))
    assert (second["comparison"], second["mean_difference"], second["rank_biserial"]) == ("B2 - B1", 0.0, 0.0)
    assert (second["p_value"], second["p_holm"]) == (1.0, 1.0)
    with pytest.raises(ValueError, match="same non-empty topologies"):
        compare_methods(rows[:-1], (Comparison("v0", "B2", "B1"),), 10, 1)


def test_primary_comparisons_are_the_eight_of_the_spec():
    assert [(item.set_id, item.label) for item in PRIMARY_COMPARISONS] == [
        (set_id, label) for set_id in ("v0", "v0-bh50") for label in ("P - B2", "B2 - B1", "B2 - B3", "P - B1")
    ]
