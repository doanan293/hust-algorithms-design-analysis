from models.paths import (
    GROUND,
    VIA_UAV,
    backhaul_routes,
    candidate_paths,
    ground_connected_source_fraction,
    radio_links,
)
from models.scenario import UAV_ID, scenario_from_dict
from scenario_builders import make_alert


def test_routes_use_fewest_hops_then_length_then_node_id(raw_scenario):
    raw_scenario["network"]["nodes"].append({"id": "n4", "x_m": 1200.0, "y_m": 1400.0})
    raw_scenario["network"]["edges"] += [
        {"source": "n4", "target": "n1", "failed": False, "capacity_bps": 1e6},
        {"source": "n4", "target": "n2", "failed": False, "capacity_bps": 1e6},
    ]
    routes = backhaul_routes(scenario_from_dict(raw_scenario))
    assert routes.hop_count == {"n0": 0, "n1": 1, "n2": 1, "n3": 2, "n4": 2}
    assert routes.next_hop["n4"] == "n1"
    assert routes.route_links("n3") == (("backhaul", "n3", "n1"), ("backhaul", "n1", "n0"))


def test_candidates_mix_ground_and_uav_paths(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    paths = candidate_paths(scenario)["alert-0000"]
    assert [(path.kind, path.entry) for path in paths] == [(GROUND, "n1"), (GROUND, "n0"), (VIA_UAV, "n1")]
    assert paths[0].links == (("access", "s00", "n1"), ("backhaul", "n1", "n0"))
    assert paths[2].links == (("access", "s00", UAV_ID), ("down", UAV_ID, "n1"), ("backhaul", "n1", "n0"))
    assert paths[2].holders == ("s00", UAV_ID, "n1")


def test_far_source_gets_only_uav_paths_and_center_drop_exists(raw_scenario):
    raw_scenario["alerts"].append(make_alert("alert-0001", "s01", 0, 10, 1000))
    paths = candidate_paths(scenario_from_dict(raw_scenario))["alert-0001"]
    assert [path.kind for path in paths] == [VIA_UAV, VIA_UAV, VIA_UAV]
    assert paths[0].entry == "n0"


def test_failed_edges_remove_isolated_relays(raw_scenario):
    raw_scenario["network"]["edges"][0].update(failed=True, capacity_bps=0.0)
    scenario = scenario_from_dict(raw_scenario)
    assert set(backhaul_routes(scenario).hop_count) == {"n0", "n2"}
    paths = candidate_paths(scenario)["alert-0000"]
    assert all(path.entry in {"n0", "n2"} for path in paths)
    assert [(path.kind, path.entry) for path in paths] == [(GROUND, "n0"), (VIA_UAV, "n0"), (VIA_UAV, "n2")]


def test_radio_links_and_ground_connected_fraction(raw_scenario):
    scenario = scenario_from_dict(raw_scenario)
    links = radio_links(candidate_paths(scenario))
    assert ("access", "s00", "n1") in links
    assert all(link[0] != "backhaul" for link in links)
    assert ground_connected_source_fraction(scenario) == 0.5
