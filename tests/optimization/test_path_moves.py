import copy

import numpy as np
import pytest

from models.paths import GROUND, VIA_UAV, CandidatePath, backhaul_routes, candidate_paths, path_violations
from models.plan import stationary_trajectory
from models.scenario import UAV_ID, scenario_from_dict
from optimization.ledger_summary import LedgerSummary
from optimization.path_moves import (
    backhaul_detour,
    entry_switches,
    estimated_completion_s,
    is_ground_connected,
    kind_flip,
    path_neighbours,
    urgency_order,
)
from scenario_builders import BASE_SCENARIO, make_alert

ACCESS_N1 = ("access", "s00", "n1")
BACKHAUL_N1 = ("backhaul", "n1", "n0")
GROUND_N1 = CandidatePath(GROUND, "n1", (ACCESS_N1, BACKHAUL_N1), 2.3)
UAV_N1 = CandidatePath(VIA_UAV, "n1", (("access", "s00", UAV_ID), ("down", UAV_ID, "n1"), BACKHAUL_N1), 2.6)
EMPTY = LedgerSummary(1, ({},), {})


def _scenario(extra_edge=False, alerts=None):
    raw = copy.deepcopy(BASE_SCENARIO)
    if extra_edge:
        raw["network"]["edges"].append({"source": "n3", "target": "n2", "failed": False, "capacity_bps": 1000000.0})
    if alerts is not None:
        raw["alerts"] = alerts
    return scenario_from_dict(raw)


def _congested(link, slots):
    return LedgerSummary(1, ({link: set(slots)},), {})


def test_ground_connected_follows_the_b0_range_test():
    scenario = _scenario()
    routes = backhaul_routes(scenario)
    assert is_ground_connected(scenario, scenario.source_by_id["s00"], routes)
    assert not is_ground_connected(scenario, scenario.source_by_id["s01"], routes)


def test_kind_flip_keeps_entry_and_backhaul_and_respects_range():
    scenario = _scenario()
    alert = scenario.alerts[0]
    flipped = kind_flip(scenario, alert, GROUND_N1)
    assert flipped.kind == VIA_UAV and flipped.entry == "n1" and flipped.links[2:] == (BACKHAUL_N1,)
    assert kind_flip(scenario, alert, flipped).links == GROUND_N1.links
    far = CandidatePath(VIA_UAV, "n3", (("access", "s00", UAV_ID), ("down", UAV_ID, "n3"), ("backhaul", "n3", "n1"), BACKHAUL_N1), 9.0)
    assert kind_flip(scenario, alert, far) is None
    direct = CandidatePath(GROUND, "n0", (("access", "s00", "n0"),), 1.0)
    assert kind_flip(scenario, alert, direct).links == (("access", "s00", UAV_ID), ("down", UAV_ID, "n0"))


def test_estimated_completion_grows_with_load_and_distance():
    scenario = _scenario()
    alert = scenario.alerts[0]
    trajectory = stationary_trajectory(scenario)
    idle = estimated_completion_s(scenario, alert, GROUND_N1, trajectory, EMPTY)
    capacity = scenario.backhaul_capacity_bps("n1", "n0")
    busy = LedgerSummary(1, ({},), {(BACKHAUL_N1, slot): 0.9 * capacity for slot in range(20)})
    assert estimated_completion_s(scenario, alert, GROUND_N1, trajectory, busy) > idle
    near = estimated_completion_s(scenario, alert, UAV_N1, trajectory, EMPTY)
    far_trajectory = np.tile([100.0, 100.0], (21, 1))
    far_trajectory[0] = far_trajectory[-1] = scenario.uav.start_xy_m
    assert estimated_completion_s(scenario, alert, UAV_N1, far_trajectory, EMPTY) > near


def test_entry_switches_draw_distinct_entries_favouring_lower_times():
    scenario = _scenario()
    alert = scenario.alerts[0]
    routes = backhaul_routes(scenario)
    trajectory = stationary_trajectory(scenario)
    rng = np.random.default_rng(0)
    drawn = entry_switches(scenario, alert, UAV_N1, routes, trajectory, EMPTY, rng, 2)
    assert len(drawn) == 2 and len({path.entry for path in drawn}) == 2 and all(path.entry != "n1" for path in drawn)
    assert all(path.kind == VIA_UAV and path_violations(scenario, "s00", path) == [] for path in drawn)
    ground = entry_switches(scenario, alert, GROUND_N1, routes, trajectory, EMPTY, rng, 5)
    assert [path.entry for path in ground] == ["n0"]
    firsts = [entry_switches(scenario, alert, UAV_N1, routes, trajectory, EMPTY, np.random.default_rng(seed), 1)[0].entry for seed in range(200)]
    assert firsts.count("n0") > firsts.count("n3")


def test_detour_avoids_the_congested_link_and_prefix_nodes():
    scenario = _scenario(extra_edge=True)
    alert = scenario.alerts[0]
    assert backhaul_detour(scenario, alert, GROUND_N1, EMPTY) is None
    detour = backhaul_detour(scenario, alert, GROUND_N1, _congested(BACKHAUL_N1, range(0, 10)))
    assert detour.links == (ACCESS_N1, ("backhaul", "n1", "n3"), ("backhaul", "n3", "n2"), ("backhaul", "n2", "n0"))
    assert detour.kind == GROUND and detour.entry == "n1" and path_violations(scenario, "s00", detour) == []
    long_path = CandidatePath(GROUND, "n1", (ACCESS_N1, ("backhaul", "n1", "n3"), ("backhaul", "n3", "n2"), ("backhaul", "n2", "n0")), 5.0)
    assert backhaul_detour(scenario, alert, long_path, _congested(("backhaul", "n2", "n0"), range(0, 10))) is None
    base = _scenario()
    assert backhaul_detour(base, base.alerts[0], GROUND_N1, _congested(BACKHAUL_N1, range(0, 10))) is None
    direct = CandidatePath(GROUND, "n0", (("access", "s00", "n0"),), 1.0)
    assert backhaul_detour(scenario, alert, direct, _congested(BACKHAUL_N1, range(0, 10))) is None


def test_neighbours_are_valid_distinct_and_only_uav_for_unconnected_sources():
    scenario = _scenario(alerts=[make_alert("alert-0000", "s00", 0, 10, 1000000), make_alert("alert-0001", "s01", 0, 20, 1000000)])
    candidates = candidate_paths(scenario)
    routes = backhaul_routes(scenario)
    trajectory = stationary_trajectory(scenario)
    rng = np.random.default_rng(1)
    for alert in scenario.alerts:
        current = candidates[alert.id][0]
        neighbours = path_neighbours(scenario, alert, current, candidates, routes, trajectory, EMPTY, rng, 3)
        assert neighbours[-1] is None and current not in neighbours
        paths = [path for path in neighbours if path is not None]
        assert len(paths) == len(set(paths))
        assert all(path_violations(scenario, alert.source, path) == [] for path in paths)
        if alert.source == "s01":
            assert all(path.kind == VIA_UAV for path in paths)
    assert path_neighbours(scenario, scenario.alerts[0], None, candidates, routes, trajectory, EMPTY, rng, 3) == list(candidates["alert-0000"])


def test_urgency_order_is_a_permutation_biased_to_short_windows():
    scenario = _scenario(alerts=[make_alert("alert-0000", "s00", 0, 2, 1), make_alert("alert-0001", "s00", 0, 20, 1), make_alert("alert-0002", "s00", 0, 20, 1)])
    alerts = list(scenario.alerts)
    firsts = []
    for seed in range(300):
        order = urgency_order(alerts, np.random.default_rng(seed))
        assert sorted(alert.id for alert in order) == [alert.id for alert in alerts]
        firsts.append(order[0].id)
    assert firsts.count("alert-0000") > firsts.count("alert-0001") + firsts.count("alert-0002")
