from pathlib import Path

import pytest
import yaml

from runner.config import load_experiment_config


def test_phase1_config_values():
    config = load_experiment_config(Path("configs/experiments/phase1.yaml"))
    assert [item.set_id for item in config.scenario_sets] == ["v0", "v0-bh50"]
    assert config.realization_ids == tuple(range(30))
    assert config.methods == ("B0", "B1")
    assert (config.tangents, config.workers, config.crosscheck.alert_count, config.crosscheck.time_limit_s) == (10, 12, 10, 300.0)
    assert (config.bootstrap.resamples, config.bootstrap.seed) == (10000, 20260911)
    assert len(config.config_hash) == 64


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(methods=["B9"]), "methods"),
        (lambda raw: raw["scenario_sets"].append(dict(raw["scenario_sets"][0])), "unique"),
        (lambda raw: raw["milp_crosscheck"].update(set_id="missing"), "milp_crosscheck.set_id"),
        (lambda raw: raw.update(realization_ids={"start": 3, "stop": 3}), "realization_ids"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, change, message):
    raw = yaml.safe_load(Path("configs/experiments/phase1.yaml").read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)
