import numpy as np

from baselines.trajectories import zone_tour_trajectory
from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, validate_plan
from models.scenario import scenario_from_dict
from optimization.tours import (
    TourSpec,
    hover_point,
    hover_slots,
    split_weights,
    tour_family,
    tour_trajectory,
    zone_orders,
)
from search_builders import three_zone_raw


def test_zone_orders_go_by_size_then_lexicographic():
    orders = zone_orders(3)
    assert len(orders) == 15
    assert orders[:7] == [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (0, 1)]
    assert orders[-3:] == [(0,), (1,), (2,)]


def test_family_skips_tours_that_exceed_the_mission_and_every_tour_is_valid():
    scenario = scenario_from_dict(three_zone_raw(num_slots=50))
    candidates = candidate_paths(scenario)
    family = tour_family(scenario)
    assert {spec.zones for spec, _ in family} == {(0, 1), (1, 0), (0,), (1,)}
    assert tour_trajectory(scenario, TourSpec((2,), "uniform", "zone_center")) is None
    for _, trajectory in family:
        plan = Plan(trajectory, {alert.id: None for alert in scenario.alerts}, EqualSplitBacklogged())
        assert validate_plan(scenario, candidates, plan) == []
    assert len({spec for spec, _ in family}) == len(family) <= 90


def test_hover_splits_follow_load_and_urgency():
    scenario = scenario_from_dict(three_zone_raw(num_slots=50))
    assert split_weights(scenario, (0, 1), "uniform") == [1.0, 1.0]
    assert split_weights(scenario, (0, 1), "load") == [1.0, 2.0]
    assert split_weights(scenario, (0, 1), "urgency") == [0.1, 0.05 + 0.025]
    assert split_weights(scenario, (2,), "load") == [1.0]
    assert hover_slots(27, [1.0, 1.0]) == [13, 13]
    assert hover_slots(27, [1.0, 2.0]) == [9, 18]
    assert hover_slots(27, [0.1, 0.075]) == [15, 11]
    trajectory = tour_trajectory(scenario, TourSpec((0, 1), "load", "zone_center"))
    at_first = np.all(np.isclose(trajectory, [1230.0, 1000.0]), axis=1).sum()
    at_second = np.all(np.isclose(trajectory, [1000.0, 1400.0]), axis=1).sum()
    assert (at_first, at_second) == (9 + 1, 18 + 1)


def test_hover_points_use_source_centroid_or_zone_center():
    raw = three_zone_raw()
    scenario = scenario_from_dict(raw)
    assert np.allclose(hover_point(scenario, 1, "source_centroid"), [1050.0, 1400.0])
    assert np.allclose(hover_point(scenario, 1, "zone_center"), [1000.0, 1400.0])
    for source in raw["sources"]:
        if source["zone"] == 1:
            source["zone"] = 0
    moved = scenario_from_dict(raw)
    assert np.allclose(hover_point(moved, 1, "source_centroid"), [1000.0, 1400.0])


def test_b1_tour_belongs_to_the_family():
    scenario = scenario_from_dict(three_zone_raw(num_slots=150))
    assert any(np.array_equal(trajectory, zone_tour_trajectory(scenario)) for _, trajectory in tour_family(scenario))
