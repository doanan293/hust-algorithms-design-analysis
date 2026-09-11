from pathlib import Path

import yaml


def test_backhaul_stressed_family_differs_only_in_set_id_and_capacity():
    base = yaml.safe_load(Path("configs/scenarios/v0.yaml").read_text(encoding="utf-8"))
    stressed = yaml.safe_load(Path("configs/scenarios/v0-bh50.yaml").read_text(encoding="utf-8"))
    changed = ("set_id", "backhaul_capacity_bps")
    assert (stressed["set_id"], stressed["backhaul_capacity_bps"]) == ("v0-bh50", 50000.0)
    assert {key: value for key, value in stressed.items() if key not in changed} == {
        key: value for key, value in base.items() if key not in changed
    }
