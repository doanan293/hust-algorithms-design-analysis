import json

import pytest

from models.rng import canonical_sha256
from models.scenario import ScenarioValidationError, load_scenario, scenario_from_dict


def test_loads_valid_scenario_and_hashes_canonical_json(raw_scenario, tmp_path):
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(raw_scenario, indent=2), encoding="utf-8")
    scenario = load_scenario(path)
    assert scenario.center == "n0"
    assert scenario.scenario_seed == 7
    assert scenario.uav.downlink_power_w == pytest.approx(0.05)
    assert scenario.sha256 == canonical_sha256(raw_scenario)
    assert scenario.position("s00") == (1230.0, 1000.0)
    assert scenario.backhaul_capacity_bps("n1", "n0") == 1000000.0
    assert scenario.backhaul_capacity_bps("n2", "n1") == 0.0


def test_missing_field_names_its_path(raw_scenario):
    del raw_scenario["alerts"][0]["deadline_slot"]
    with pytest.raises(ScenarioValidationError) as error:
        scenario_from_dict(raw_scenario)
    assert error.value.field == "alerts[0].deadline_slot"


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda raw: raw.update(schema_version=1), "schema_version"),
        (lambda raw: raw["alerts"][0].update(deadline_slot=21), "alerts[0].deadline_slot"),
        (lambda raw: raw["alerts"][0].update(source="s99"), "alerts[0].source"),
        (lambda raw: raw.update(center="n9"), "center"),
        (lambda raw: raw["sources"][0].update(id="n1"), "sources[0].id"),
        (lambda raw: raw["network"]["edges"][0].update(target="n9"), "network.edges[0].target"),
        (lambda raw: raw["time"].update(access_fraction=1.0), "time.access_fraction"),
    ],
)
def test_rejects_invalid_references_and_ranges(raw_scenario, mutate, field):
    mutate(raw_scenario)
    with pytest.raises(ScenarioValidationError) as error:
        scenario_from_dict(raw_scenario)
    assert error.value.field == field
