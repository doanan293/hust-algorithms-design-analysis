from pathlib import Path

import pytest
import yaml

FAMILIES = {
    "sens-lref125k": {"workload.size_ref_bits": 125000},
    "sens-lref500k": {"workload.size_ref_bits": 500000},
    "sens-lref1m": {"workload.size_ref_bits": 1000000},
    "sens-bh100": {"backhaul_capacity_bps": 100000.0},
    "sens-physical": {"channel.uav.beta0_db": -40.0, "channel.ground.beta0_db": -40.0, "spectrum.noise_psd_dbm_per_hz": -164.0},
    "sens-deadline-short": {"workload.deadline_min_slots": 10, "workload.deadline_max_slots": 30},
    "sens-deadline-long": {"workload.deadline_min_slots": 40, "workload.deadline_max_slots": 120},
    "sens-btot05": {"spectrum.b_tot_hz": 500000.0},
    "sens-btot2": {"spectrum.b_tot_hz": 2000000.0},
    "sens-alerts20": {"workload.alert_count": 20},
    "sens-alerts60": {"workload.alert_count": 60},
    "sens-k2": {"paths.k": 2, "paths.max_ground": 1},
    "sens-k5": {"paths.k": 5},
}


def _flatten(raw, prefix=""):
    flat = {}
    for key, value in raw.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, name + "."))
        else:
            flat[name] = value
    return flat


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_sensitivity_family_differs_from_v0_only_in_its_fields(family):
    base = _flatten(yaml.safe_load(Path("configs/scenarios/v0.yaml").read_text(encoding="utf-8")))
    changed = _flatten(yaml.safe_load(Path(f"configs/scenarios/{family}.yaml").read_text(encoding="utf-8")))
    expected = {"set_id": family, **FAMILIES[family]}
    assert set(changed) == set(base)
    assert {key for key in base if base[key] != changed[key]} == set(expected)
    assert {key: changed[key] for key in expected} == expected


def test_every_sensitivity_config_is_listed():
    assert {path.stem for path in Path("configs/scenarios").glob("sens-*.yaml")} == set(FAMILIES)
