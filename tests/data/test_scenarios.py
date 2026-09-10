import pytest

from data.scenarios import ScenarioInputs, canonical_hash, generate_scenario


@pytest.fixture
def scenario_inputs():
    return ScenarioInputs(
        network={
            "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
            "edges": [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}],
        },
        demands=({"original_value": "12.5"}, {"original_value": "7.0"}),
        targets=({"id": "target-1", "x_m": 100.0, "y_m": 200.0},),
        raw_sha256=("b" * 64, "a" * 64),
        config_hash="c" * 64,
        seed=20260910,
        alert_count=10,
        failure_probability=0.2,
    )


def test_same_seed_produces_identical_scenario(scenario_inputs):
    first = generate_scenario(scenario_inputs)
    second = generate_scenario(scenario_inputs)
    assert first == second
    assert canonical_hash(first) == canonical_hash(second)


def test_different_seed_changes_stochastic_fields(scenario_inputs):
    first = generate_scenario(scenario_inputs)
    second = generate_scenario(scenario_inputs.with_seed(20260911))
    assert first["failures"] != second["failures"]
    assert first["provenance"]["raw_sha256"] == second["provenance"]["raw_sha256"]


def test_alerts_obey_release_deadline_and_positive_size(scenario_inputs):
    scenario = generate_scenario(scenario_inputs)
    assert all(0 <= alert["release_slot"] < alert["deadline_slot"] for alert in scenario["alerts"])
    assert all(alert["size_bits"] > 0 for alert in scenario["alerts"])
