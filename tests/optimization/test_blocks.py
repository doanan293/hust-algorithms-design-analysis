import numpy as np

from models.paths import backhaul_routes, candidate_paths, uav_path
from models.plan import stationary_trajectory, validate_plan
from models.scenario import scenario_from_dict
from optimization.bcd import SearchParams
from optimization.blocks import (
    SearchState,
    bandwidth_block,
    catalog_trajectory_block,
    links_by_load,
    path_block,
    trajectory_block,
    unit_weights,
    with_unit_weights,
)
from optimization.scoring import DesignScorer
from optimization.tours import tour_family
from scenario_builders import make_alert
from search_builders import bandwidth_raw, hopeless_raw, three_zone_raw

FIRST_ACCESS = ("access", "s00", "n1")
SECOND_ACCESS = ("access", "s02", "n2")
PARAMS = SearchParams(design_ids=(), budget=500)


def _start(raw, choice, trajectory=None):
    scenario = scenario_from_dict(raw)
    candidates = candidate_paths(scenario)
    scorer = DesignScorer(scenario, design_ids=(), budget=500)
    paths = {alert_id: None if index is None else candidates[alert_id][index] for alert_id, index in choice.items()}
    state = SearchState(stationary_trajectory(scenario) if trajectory is None else trajectory, paths, unit_weights(scenario, candidates))
    state.key = scorer.score(state.plan())
    return scenario, candidates, scorer, state


def test_try_change_adopts_only_strict_improvements():
    scenario, candidates, scorer, state = _start(hopeless_raw(), {"alert-0000": 0, "alert-0001": 0})
    before = state.key
    assert not state.try_change(scorer, paths=dict(state.paths))
    assert state.key == before and scorer.calls == 2
    assert state.try_change(scorer, paths={"alert-0000": None, "alert-0001": candidates["alert-0001"][0]})
    assert state.paths == {"alert-0000": None, "alert-0001": candidates["alert-0001"][0]} and state.key > before


def test_with_unit_weights_adds_missing_radio_links_only():
    scenario, candidates, _, state = _start(hopeless_raw(), {"alert-0000": 0, "alert-0001": 0})
    new_path = uav_path(scenario, scenario.alert_by_id["alert-0001"], "n1", (("backhaul", "n1", "n0"),))
    completed = with_unit_weights(state.weights, {"alert-0000": None, "alert-0001": new_path}, 20)
    uplink, downlink = ("access", "s00", "UAV"), ("down", "UAV", "n1")
    assert set(completed) == set(state.weights) | {uplink, downlink}
    assert np.all(completed[uplink] == 1.0) and completed[uplink].shape == (20,)
    assert completed[FIRST_ACCESS] is state.weights[FIRST_ACCESS]
    assert with_unit_weights(state.weights, dict(state.paths), 20) == state.weights


def test_path_block_improves_strictly_and_frees_the_backhaul_for_the_feasible_alert():
    scenario, candidates, scorer, state = _start(hopeless_raw(), {"alert-0000": 0, "alert-0001": 0})
    before = state.key
    assert before[0] == 0
    routes = backhaul_routes(scenario)
    assert path_block(scenario, candidates, routes, scorer, state, PARAMS, np.random.default_rng(0))
    assert state.key > before and validate_plan(scenario, state.plan()) == []
    rng = np.random.default_rng(1)
    _, _, scorer, state = _start(hopeless_raw(), {"alert-0000": 0, "alert-0001": 0})
    for _ in range(3):
        path_block(scenario, candidates, routes, scorer, state, PARAMS, rng)
    assert state.key[0] == 1
    hopeless = state.paths["alert-0000"]
    assert hopeless is None or ("backhaul", "n1", "n0") not in hopeless.links


def test_path_block_only_scores_through_the_scorer_and_stays_valid():
    raw = three_zone_raw(num_slots=50)
    scenario, candidates, scorer, state = _start(raw, {"alert-0000": 0, "alert-0001": 0, "alert-0002": 0})
    rng = np.random.default_rng(1)
    before = scorer.calls
    path_block(scenario, candidates, backhaul_routes(scenario), scorer, state, PARAMS, rng)
    assert scorer.calls > before and validate_plan(scenario, state.plan()) == []
    assert all(link in state.weights for path in state.paths.values() if path is not None for link in path.links if link[0] != "backhaul")


def test_catalog_block_flies_to_a_source_only_the_uav_reaches():
    raw = three_zone_raw(num_slots=150)
    raw["alerts"] = [make_alert("alert-0000", "s01", 0, 150, 1000000)]
    scenario, candidates, scorer, state = _start(raw, {"alert-0000": 0})
    assert [path.kind for path in candidates["alert-0000"]] == ["uav", "uav", "uav"]
    assert state.key[0] == 0
    assert catalog_trajectory_block(scorer, state, tour_family(scenario))
    assert state.key[0] == 1
    assert np.min(np.linalg.norm(state.trajectory - [100.0, 100.0], axis=1)) < 1e-6
    assert validate_plan(scenario, state.plan()) == []


def test_local_trajectory_block_moves_only_by_feasible_shifts_within_its_budget():
    raw = three_zone_raw(num_slots=150)
    raw["alerts"] = [make_alert("alert-0000", "s01", 0, 150, 1000000), make_alert("alert-0001", "s00", 0, 150, 4000000)]
    scenario, candidates, scorer, state = _start(raw, {"alert-0000": 0, "alert-0001": 2})
    params = SearchParams(design_ids=(), budget=500, catalog_probability=0.0)
    rng = np.random.default_rng(0)
    before_calls, before_key = scorer.calls, state.key
    trajectory_block(scenario, scorer, state, params, rng, tour_family(scenario))
    assert scorer.calls - before_calls <= 1 + 2 * params.trajectory_segments
    assert state.key >= before_key and validate_plan(scenario, state.plan()) == []
    assert np.array_equal(state.trajectory[0], stationary_trajectory(scenario)[0])
    always_catalog = SearchParams(design_ids=(), budget=500, catalog_probability=1.0)
    calls = scorer.calls
    trajectory_block(scenario, scorer, state, always_catalog, np.random.default_rng(0), tour_family(scenario))
    assert scorer.calls - calls <= 2 + 2 * always_catalog.trajectory_segments


def test_bandwidth_block_raises_the_weight_of_the_congested_link():
    scenario, candidates, scorer, state = _start(bandwidth_raw(), {"alert-0000": 0, "alert-0001": 0})
    assert links_by_load(scenario, state.paths) == [FIRST_ACCESS, SECOND_ACCESS]
    assert state.key[0] == 1
    assert bandwidth_block(scenario, scorer, state)
    assert state.key[0] == 2
    assert np.all(state.weights[FIRST_ACCESS] == 2.0) and np.all(state.weights[SECOND_ACCESS] == 1.0)


def test_links_by_load_breaks_ties_by_link_and_skips_unserved_alerts():
    raw = bandwidth_raw()
    raw["alerts"][0]["size_bits"] = 4000000
    scenario = scenario_from_dict(raw)
    candidates = candidate_paths(scenario)
    paths = {"alert-0000": candidates["alert-0000"][0], "alert-0001": candidates["alert-0001"][0]}
    assert links_by_load(scenario, paths) == [FIRST_ACCESS, SECOND_ACCESS]
    assert links_by_load(scenario, {**paths, "alert-0000": None}) == [SECOND_ACCESS]
