from pathlib import Path

import pytest
import yaml

from optimization.bcd import SearchParams
from runner.config import MethodSpec, load_experiment_config
from runner.methods import bound_owner, build_method


def test_phase1_config_values():
    config = load_experiment_config(Path("configs/experiments/phase1.yaml"))
    assert [item.set_id for item in config.scenario_sets] == ["v0", "v0-bh50"]
    assert config.realization_ids == tuple(range(30))
    assert config.methods == (MethodSpec("B0", "B0"), MethodSpec("B1", "B1"))
    assert (config.tangents, config.workers, config.crosscheck.alert_count, config.crosscheck.time_limit_s) == (10, 12, 10, 300.0)
    assert (config.bootstrap.resamples, config.bootstrap.seed) == (10000, 20260911)
    assert len(config.config_hash) == 64


def test_phase2_pilot_config_values():
    config = load_experiment_config(Path("configs/experiments/phase2_pilot.yaml"))
    assert (config.split, config.crosscheck, config.output_dir) == ("dev", None, Path("results/phase2/pilot"))
    assert [item.set_id for item in config.scenario_sets] == ["v0"]
    assert [(spec.id, spec.base, spec.bounds) for spec in config.methods] == [
        ("B1", "B1", False), ("B2", "search", True), ("B3", "search", True), ("P", "search", True),
    ]
    assert [spec.search_params() for spec in config.methods[1:]] == [
        SearchParams(), SearchParams(objective="leximin"), SearchParams(operation1=True, operation2=True),
    ]


def _write(tmp_path: Path, change) -> Path:
    raw = yaml.safe_load(Path("configs/experiments/phase1.yaml").read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_method_specs_accept_names_and_search_mappings(tmp_path: Path):
    methods = [
        "B0",
        {"id": "B1", "base": "B1"},
        {"id": "P_expected_design", "base": "search", "params": {"operation2": True, "operation1": True, "design_ids": "expected"}, "bounds": True},
        {"id": "P_design10", "base": "search", "params": {"design_ids": list(range(10))}},
    ]
    config = load_experiment_config(_write(tmp_path, lambda raw: raw.update(methods=methods)))
    assert config.methods[:2] == (MethodSpec("B0", "B0"), MethodSpec("B1", "B1"))
    expected = config.methods[2]
    assert expected == MethodSpec("P_expected_design", "search", (("design_ids", "expected"), ("operation1", True), ("operation2", True)), True)
    assert expected.search_params() == SearchParams(operation1=True, operation2=True, design_ids=())
    assert config.methods[3].params == (("design_ids", tuple(range(10))),) and config.methods[3].bounds is False
    assert build_method(expected).name == "P_expected_design"
    assert [bound_owner(spec) for spec in config.methods] == [
        ("ground_only", ""), ("trajectory", "B1"), ("trajectory", "P_expected_design"), None,
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(methods=["B9"]), "methods"),
        (lambda raw: raw.update(methods=["B1", "B1"]), "unique"),
        (lambda raw: raw.update(methods=[{"id": "X", "base": "B7"}]), "unknown base"),
        (lambda raw: raw.update(methods=[{"id": "B1", "base": "B1", "params": {"budget": 5}}]), "takes no"),
        (lambda raw: raw.update(methods=[{"id": "B1", "base": "search"}]), "not name a baseline"),
        (lambda raw: raw.update(methods=[{"id": "P__x", "base": "search"}]), "single '_'"),
        (lambda raw: raw.update(methods=[{"id": "P", "base": "search", "params": {"repair": True}}]), "unknown search parameters"),
        (lambda raw: raw.update(methods=[{"id": "P", "base": "search", "bounds": "yes"}]), "bounds"),
        (lambda raw: raw["scenario_sets"].append(dict(raw["scenario_sets"][0])), "unique"),
        (lambda raw: raw["milp_crosscheck"].update(set_id="missing"), "milp_crosscheck.set_id"),
        (lambda raw: raw.update(realization_ids={"start": 3, "stop": 3}), "realization_ids"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, change, message):
    with pytest.raises(ValueError, match=message):
        load_experiment_config(_write(tmp_path, change))
