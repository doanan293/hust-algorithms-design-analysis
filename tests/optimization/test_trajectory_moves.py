import numpy as np
import pytest

from baselines.trajectories import zone_tour_trajectory
from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, _trajectory_violations
from models.scenario import UAV_ID, scenario_from_dict
from optimization.ledger_summary import LedgerSummary
from optimization.trajectory_moves import Segment, feasible_segment, max_shift, sample_uav_segments, shift_segment
from scenario_builders import make_alert
from search_builders import three_zone_raw

STEP = 50.0
EAST = np.array([1.0, 0.0])


def _scenario20():
    raw = three_zone_raw(num_slots=20)
    raw["alerts"] = [make_alert("alert-0000", "s00", 0, 20, 1000)]
    return scenario_from_dict(raw)


def _hover(num_slots=20, point=(1000.0, 1000.0)):
    return np.tile(np.asarray(point, dtype=float), (num_slots + 1, 1))


def test_max_shift_solves_the_boundary_circle():
    anchor = np.array([0.0, 0.0])
    assert max_shift(anchor, np.array([0.0, 0.0]), EAST, STEP) == pytest.approx(STEP)
    assert max_shift(anchor, np.array([30.0, 0.0]), EAST, STEP) == pytest.approx(20.0)
    assert max_shift(anchor, np.array([0.0, 30.0]), EAST, STEP) == pytest.approx(40.0)
    assert max_shift(anchor, np.array([50.0, 0.0]), EAST, STEP) == pytest.approx(0.0)
    assert max_shift(anchor, np.array([60.0, 0.0]), EAST, STEP) == 0.0


def test_shift_segment_keeps_feasibility_and_endpoints():
    scenario = _scenario20()
    trajectory = _hover()
    shifted = shift_segment(trajectory, 1, 19, EAST, 1.0, STEP)
    assert np.allclose(shifted[1:20], [1050.0, 1000.0]) and np.array_equal(shifted[0], trajectory[0]) and np.array_equal(shifted[20], trajectory[20])
    assert _trajectory_violations(scenario, shifted) == []
    half = shift_segment(trajectory, 5, 7, EAST, 0.5, STEP)
    assert np.allclose(half[5:8], [1025.0, 1000.0]) and np.array_equal(half[4], trajectory[4]) and np.array_equal(half[8], trajectory[8])
    rng = np.random.default_rng(3)
    for _ in range(50):
        start = int(rng.integers(1, 19))
        stop = int(rng.integers(start, 20))
        direction = rng.normal(size=2)
        out = shift_segment(trajectory, start, stop, direction, float(rng.uniform(0.1, 1.0)), STEP)
        assert out is not None and _trajectory_violations(scenario, out) == []


def test_shift_segment_refuses_a_cruise_boundary_and_widening_finds_room():
    scenario = _scenario20()
    cruise = np.array(
        [[1000.0, 1000.0]] * 3                                              # q[0..2] hover at the start
        + [[1000.0 + STEP * index, 1000.0] for index in range(1, 6)]        # q[3..7] cruise east at V_max
        + [[1250.0, 1000.0]] * 8                                            # q[8..15] hover
        + [[1250.0 - STEP * index, 1000.0] for index in range(1, 6)]        # q[16..20] cruise back
    )
    assert cruise.shape == (21, 2) and _trajectory_violations(scenario, cruise) == []
    north = np.array([0.0, 1.0])
    assert shift_segment(cruise, 4, 5, north, 1.0, STEP) is None
    assert feasible_segment(cruise, 4, 5, north, STEP, 0) is None
    assert feasible_segment(cruise, 4, 5, north, STEP, 10) == (2, 7)
    shifted = shift_segment(cruise, 2, 7, north, 1.0, STEP)
    assert shifted is not None and _trajectory_violations(scenario, shifted) == []
    with pytest.raises(ValueError):
        shift_segment(cruise, 0, 3, north, 1.0, STEP)
    with pytest.raises(ValueError):
        shift_segment(cruise, 3, 3, np.zeros(2), 1.0, STEP)


def _uav_plan(scenario):
    candidates = candidate_paths(scenario)
    uav = {alert.id: next(path for path in candidates[alert.id] if path.kind == "uav") for alert in scenario.alerts}
    return Plan(zone_tour_trajectory(scenario), uav, EqualSplitBacklogged())


def test_sample_uav_segments_follows_priority_and_handles_no_congestion():
    raw = three_zone_raw(num_slots=20)
    raw["alerts"] = [make_alert("alert-0000", "s00", 0, 10, 1000000), make_alert("alert-0001", "s01", 2, 20, 1000000)]
    scenario = scenario_from_dict(raw)
    plan = _uav_plan(scenario)
    uplink_a, uplink_b = ("access", "s00", UAV_ID), ("access", "s01", UAV_ID)
    assert sample_uav_segments(scenario, plan, LedgerSummary(1, ({},), {}), np.random.default_rng(0), 5) == []
    summary = LedgerSummary(2, ({uplink_a: {1, 3, 4, 5, 9}, uplink_b: {12}}, {uplink_a: {4, 5}}), {})
    segments = sample_uav_segments(scenario, plan, summary, np.random.default_rng(0), 5)
    # slot 9 is dropped: the UAV hovers right above s00 there, so there is no direction to move in
    assert [(segment.start, segment.stop) for segment in segments] == [(3, 6), (1, 2), (12, 13)]
    assert segments[0].priority > segments[1].priority > segments[2].priority > 0.0
    source = np.asarray(scenario.source_by_id["s00"].xy)
    centroid = plan.trajectory[3:7].mean(axis=0)
    assert np.allclose(segments[0].direction, (source - centroid) / np.linalg.norm(source - centroid))
    assert len(sample_uav_segments(scenario, plan, summary, np.random.default_rng(0), 2)) == 2
    firsts = [sample_uav_segments(scenario, plan, summary, np.random.default_rng(seed), 1)[0].start for seed in range(100)]
    assert firsts.count(3) > firsts.count(12)
