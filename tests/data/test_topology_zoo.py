from pathlib import Path

import pytest

from data.topology_zoo import inventory_topology, normalize_topology


FIXTURES = Path("tests/data/fixtures/topology_zoo")


def test_inventory_accepts_connected_graph_with_coordinates():
    record = inventory_topology(FIXTURES / "eligible.graphml")
    assert record.eligible is True
    assert record.exclusion_reason == ""
    assert record.coordinate_count == record.node_count
    assert record.has_parallel_edges is True


def test_inventory_rejects_missing_coordinates():
    record = inventory_topology(FIXTURES / "missing_coordinates.graphml")
    assert record.eligible is False
    assert record.exclusion_reason == "missing_coordinates"


def test_normalization_uses_one_scale_for_both_axes():
    network = normalize_topology(FIXTURES / "eligible.graphml", 2000.0)
    xs = [node["x_m"] for node in network["nodes"]]
    ys = [node["y_m"] for node in network["nodes"]]
    assert max(max(xs) - min(xs), max(ys) - min(ys)) == pytest.approx(2000.0)
    assert network["normalization"]["scale_x"] == network["normalization"]["scale_y"]
    assert network["edges"] == sorted(network["edges"], key=lambda edge: (edge["source"], edge["target"]))
