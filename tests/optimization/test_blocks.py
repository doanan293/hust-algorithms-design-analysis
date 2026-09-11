import numpy as np

from models.paths import candidate_paths
from models.plan import stationary_trajectory, validate_plan
from models.scenario import scenario_from_dict
from optimization.blocks import SearchState, bandwidth_block, links_by_load, path_block, trajectory_block, unit_weights
from optimization.scoring import DesignScorer
from optimization.tours import tour_family
from scenario_builders import make_alert
from search_builders import bandwidth_raw, hopeless_raw, three_zone_raw

FIRST_ACCESS = ("access", "s00", "n1")
SECOND_ACCESS = ("access", "s02", "n2")


def _start(raw, choice):
    scenario = scenario_from_dict(raw)
    candidates = candidate_paths(scenario)
    scorer = DesignScorer(scenario, candidates, design_ids=(), budget=500)
    state = SearchState(stationary_trajectory(scenario), dict(choice), unit_weights(scenario, candidates))
    state.key = scorer.score(state.plan())
    return scenario, candidates, scorer, state


def test_try_change_adopts_only_strict_improvements():
    _, _, scorer, state = _start(hopeless_raw(), {"alert-0000": 0, "alert-0001": 0})
    before = state.key
    assert not state.try_change(scorer, path_choice={"alert-0000": 0, "alert-0001": 0})
    assert state.key == before and scorer.calls == 2
    assert state.try_change(scorer, path_choice={"alert-0000": None, "alert-0001": 0})
    assert state.path_choice == {"alert-0000": None, "alert-0001": 0} and state.key > before


def test_path_block_drops_a_hopeless_alert_that_blocks_another():
    scenario, candidates, scorer, state = _start(hopeless_raw(), {"alert-0000": 0, "alert-0001": 0})
    assert state.key[0] == 0
    assert path_block(scenario, candidates, scorer, state)
    assert state.path_choice == {"alert-0000": None, "alert-0001": 0}
    assert state.key[0] == 1
    assert scorer.calls == 3


def test_trajectory_block_flies_to_a_source_only_the_uav_reaches():
    raw = three_zone_raw(num_slots=150)
    raw["alerts"] = [make_alert("alert-0000", "s01", 0, 150, 1000000)]
    scenario, candidates, scorer, state = _start(raw, {"alert-0000": 0})
    assert [path.kind for path in candidates["alert-0000"]] == ["uav", "uav", "uav"]
    assert state.key[0] == 0
    assert trajectory_block(scorer, state, tour_family(scenario))
    assert state.key[0] == 1
    assert np.min(np.linalg.norm(state.trajectory - [100.0, 100.0], axis=1)) < 1e-6
    assert validate_plan(scenario, candidates, state.plan()) == []


def test_bandwidth_block_raises_the_weight_of_the_congested_link():
    scenario, candidates, scorer, state = _start(bandwidth_raw(), {"alert-0000": 0, "alert-0001": 0})
    assert links_by_load(scenario, candidates, state.path_choice) == [FIRST_ACCESS, SECOND_ACCESS]
    assert state.key[0] == 1
    assert bandwidth_block(scenario, candidates, scorer, state)
    assert state.key[0] == 2
    assert np.all(state.weights[FIRST_ACCESS] == 2.0) and np.all(state.weights[SECOND_ACCESS] == 1.0)


def test_links_by_load_breaks_ties_by_link_and_skips_unserved_alerts():
    raw = bandwidth_raw()
    raw["alerts"][0]["size_bits"] = 4000000
    scenario = scenario_from_dict(raw)
    candidates = candidate_paths(scenario)
    assert links_by_load(scenario, candidates, {"alert-0000": 0, "alert-0001": 0}) == [FIRST_ACCESS, SECOND_ACCESS]
    assert links_by_load(scenario, candidates, {"alert-0000": None, "alert-0001": 0}) == [SECOND_ACCESS]
