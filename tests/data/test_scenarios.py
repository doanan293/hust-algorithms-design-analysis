from dataclasses import replace
from pathlib import Path

import pytest

from data.scenarios import (
    build_scenario_set,
    choose_center,
    generate_scenario,
    load_scenario_set_config,
    scenario_seed,
    select_topologies,
)
from models.scenario import scenario_from_dict

CONFIG = load_scenario_set_config(Path("configs/scenarios/v0.yaml"))


def grid_network(network_id: str, rows: int, cols: int) -> dict[str, object]:
    nodes = [
        {"id": f"n{row * cols + col}", "x_m": 2000.0 * col / (cols - 1), "y_m": 2000.0 * row / (rows - 1)}
        for row in range(rows)
        for col in range(cols)
    ]
    edges = []
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            if col + 1 < cols:
                edges.append({"source": f"n{index}", "target": f"n{index + 1}"})
            if row + 1 < rows:
                edges.append({"source": f"n{index}", "target": f"n{index + cols}"})
    return {
        "network_id": network_id,
        "nodes": nodes,
        "edges": edges,
        "normalization": {"box_size_m": 2000.0, "scale_x": 1.0, "scale_y": 1.0},
    }


def _generate(config=CONFIG, network=None, replicate=0):
    return generate_scenario(
        network or grid_network("Grid", 4, 5),
        "eval",
        replicate,
        [1.0, 50.0, 100.0],
        config,
        "a" * 64,
        ["c" * 64, "b" * 64],
        "d" * 64,
    )


def test_v0_config_values():
    assert CONFIG.set_id == "v0"
    assert (CONFIG.strata, CONFIG.dev_count, CONFIG.eval_replicates, CONFIG.dev_replicates) == (13, 3, 3, 2)
    assert CONFIG.physical["time"]["num_slots"] == 150
    assert CONFIG.physical["channel"]["uav"]["beta0_db"] == -50.0


def test_topology_selection_is_stratified_deterministic_and_disjoint():
    eligible = [(f"T{index:02d}", 15 + index) for index in range(26)]
    first = select_topologies(eligible, 13, 3, 20260911)
    assert first == select_topologies(eligible, 13, 3, 20260911)
    assert len(first) == 13
    assert sorted(first.values()).count("dev") == 3
    chosen_sizes = sorted(15 + int(topology_id[1:]) for topology_id in first)
    assert all(15 + 2 * stratum <= size <= 16 + 2 * stratum for stratum, size in enumerate(chosen_sizes))
    with pytest.raises(ValueError):
        select_topologies(eligible[:5], 13, 3, 1)


def test_scenario_seed_and_center_rule():
    assert scenario_seed(1, "Grid", 0) == scenario_seed(1, "Grid", 0)
    assert scenario_seed(1, "Grid", 0) != scenario_seed(1, "Grid", 1)
    star = {"nodes": [{"id": "b"}, {"id": "a"}, {"id": "c"}], "edges": [{"source": "a", "target": "b"}, {"source": "a", "target": "c"}]}
    assert choose_center(star) == "a"
    pair = {"nodes": [{"id": "y"}, {"id": "x"}], "edges": [{"source": "x", "target": "y"}]}
    assert choose_center(pair) == "x"


def test_generated_scenario_is_valid_and_reproducible():
    raw = _generate()
    scenario = scenario_from_dict(raw)
    assert scenario_from_dict(_generate()).sha256 == scenario.sha256
    assert scenario_from_dict(_generate(replicate=1)).sha256 != scenario.sha256
    assert scenario.center not in scenario.relays and len(scenario.relays) == 19
    assert scenario.uav.start_xy_m == scenario.node_by_id[scenario.center].xy
    assert len(scenario.sources) == 15 and len(scenario.alerts) == 40
    assert all(0.0 <= source.x_m <= 2000.0 and 0.0 <= source.y_m <= 2000.0 for source in scenario.sources)
    assert all(0 <= alert.release_slot <= 89 for alert in scenario.alerts)
    assert all(20 <= alert.deadline_slot - alert.release_slot <= 60 for alert in scenario.alerts)
    assert {alert.size_ratio for alert in scenario.alerts} <= {0.25, 50.0 / 50.0, 100.0 / 50.0}
    assert raw["provenance"]["raw_sha256"] == ["b" * 64, "c" * 64]
    assert all(edge.capacity_bps == (0.0 if edge.failed else 1e6) for edge in scenario.edges)


def test_changing_alert_rule_leaves_other_streams_unchanged():
    base = _generate()
    fewer_alerts = _generate(config=replace(CONFIG, alert_count=10))
    assert fewer_alerts["network"]["edges"] == base["network"]["edges"]
    assert fewer_alerts["sources"] == base["sources"]
    assert fewer_alerts["damage_zones"] == base["damage_zones"]
    assert len(fewer_alerts["alerts"]) == 10


def test_generation_rule_failures_name_the_topology():
    with pytest.raises(ValueError, match="Grid: damage-zone rule placed"):
        _generate(config=replace(CONFIG, zone_min_separation_m=5000.0))
    slow = replace(CONFIG, physical={**CONFIG.physical, "uav": {**CONFIG.physical["uav"], "v_max_mps": 10.0}})
    with pytest.raises(ValueError, match="Grid: v_max"):
        _generate(config=slow)


def test_build_scenario_set_counts_replicates_per_split():
    networks = {f"G{index:02d}": grid_network(f"G{index:02d}", 3, 5 + index) for index in range(13)}
    scenarios = build_scenario_set(
        networks, {key: "e" * 64 for key in networks}, [1.0], CONFIG, [], "f" * 64
    )
    splits = [scenario["split"] for scenario in scenarios]
    assert splits.count("eval") == 30 and splits.count("dev") == 6
