from dataclasses import replace
import csv
import hashlib
from pathlib import Path

import pytest
import yaml

from data.cli import main
from data.scenarios import RESCUENET, generate_scenario, load_scenario_set_config, load_targets, target_zones_and_sources
from models.scenario import load_scenario, scenario_from_dict
from scenario_stage_inputs import write_scenario_inputs

V0 = load_scenario_set_config(Path("configs/scenarios/v0.yaml"))
SYNTHETIC_GRID_SHA256 = "a9b6da316a34a8820d562ddb3288688fbc77732a2588bf3994c591ad173948cf"
POINTS = ((100.0, 100.0), (1000.0, 1800.0), (140.0, 60.0), (1900.0, 200.0), (1040.0, 1760.0), (60.0, 140.0), (1860.0, 240.0))
SOURCE_KEYS = {"zone_min_separation_m", "zone_max_attempts", "source_count", "source_sigma_m", "source_generator", "targets_manifest", "targets_manifest_sha256"}


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
    return {"network_id": network_id, "nodes": nodes, "edges": edges, "normalization": {"box_size_m": 2000.0, "scale_x": 1.0, "scale_y": 1.0}}


def _generate(config, targets=()):
    return generate_scenario(grid_network("Grid", 4, 5), "eval", 0, [1.0, 50.0, 100.0], config, "a" * 64, ["c" * 64, "b" * 64], "d" * 64, targets)


def _rescuenet_config():
    return replace(
        V0, set_id="rescuenet", source_generator=RESCUENET, zone_min_separation_m=None, zone_max_attempts=None, source_count=None,
        source_sigma_m=None, targets_manifest=Path("manifests/targets.csv"), targets_manifest_sha256="e" * 64,
    )


def test_synthetic_generation_is_unchanged_by_the_source_interface():
    assert V0.source_generator == "synthetic" and V0.targets_manifest is None
    assert scenario_from_dict(_generate(V0)).sha256 == SYNTHETIC_GRID_SHA256


def test_target_zones_are_sorted_kmeans_clusters_with_one_source_per_target():
    zones, sources = target_zones_and_sources(POINTS, 3, 400.0, "Grid")
    rounded = [(round(zone["x_m"], 6), round(zone["y_m"], 6), zone["radius_m"]) for zone in zones]
    assert rounded == [(100.0, 100.0, 400.0), (1020.0, 1780.0, 400.0), (1880.0, 220.0, 400.0)]
    assert [source["id"] for source in sources] == [f"s{index:02d}" for index in range(7)]
    assert [(source["x_m"], source["y_m"]) for source in sources] == list(POINTS)
    assert [source["zone"] for source in sources] == [0, 1, 0, 2, 1, 0, 2]
    reversed_zones, _ = target_zones_and_sources(tuple(reversed(POINTS)), 3, 400.0, "Grid")
    assert [(round(zone["x_m"], 6), round(zone["y_m"], 6)) for zone in reversed_zones] == [item[:2] for item in rounded]


def test_too_few_targets_for_the_zones_are_rejected():
    with pytest.raises(ValueError, match="Grid: 2 targets cannot form 3 zones"):
        target_zones_and_sources(POINTS[:2], 3, 400.0, "Grid")


def test_rescuenet_scenario_uses_targets_and_records_the_manifest_hash():
    config = _rescuenet_config()
    scenario = scenario_from_dict(_generate(config, POINTS))
    assert [source.xy for source in scenario.sources] == list(POINTS)
    assert len(scenario.damage_zones) == 3 and scenario.provenance["targets_manifest_sha256"] == "e" * 64
    assert scenario_from_dict(_generate(config, POINTS)).sha256 == scenario.sha256
    assert len(scenario.alerts) == 40 and {alert.source for alert in scenario.alerts} <= {source.id for source in scenario.sources}


def test_load_targets_checks_the_manifest_hash(tmp_path: Path):
    path = tmp_path / "targets.csv"
    path.write_text("target_id,x_m,y_m\nt1,1.5,2.5\nt2,3.0,4.0\n", encoding="utf-8")
    assert load_targets(path, hashlib.sha256(path.read_bytes()).hexdigest()) == ((1.5, 2.5), (3.0, 4.0))
    with pytest.raises(ValueError, match="does not match workload.targets_manifest_sha256"):
        load_targets(path, "0" * 64)


def test_rescuenet_config_differs_from_v0_only_in_set_id_and_source_fields():
    base = yaml.safe_load(Path("configs/scenarios/v0.yaml").read_text(encoding="utf-8"))
    family = yaml.safe_load(Path("configs/scenarios/rescuenet.yaml").read_text(encoding="utf-8"))
    config = load_scenario_set_config(Path("configs/scenarios/rescuenet.yaml"))
    assert (config.set_id, config.source_generator, config.targets_manifest, config.source_count) == (
        "rescuenet", RESCUENET, Path("manifests/rescuenet_targets.csv"), None,
    )
    assert {key: value for key, value in family.items() if key not in ("set_id", "workload")} == {
        key: value for key, value in base.items() if key not in ("set_id", "workload")
    }
    assert {key: value for key, value in family["workload"].items() if key not in SOURCE_KEYS} == {
        key: value for key, value in base["workload"].items() if key not in SOURCE_KEYS
    }


def _write_config(tmp_path: Path, source: str, change) -> Path:
    raw = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    change(raw["workload"])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("source", "change", "message"),
    [
        ("configs/scenarios/rescuenet.yaml", lambda workload: workload.update(source_generator="lidar"), "unknown generator 'lidar'"),
        ("configs/scenarios/rescuenet.yaml", lambda workload: workload.update(source_count=15), "apply only to source_generator synthetic"),
        ("configs/scenarios/rescuenet.yaml", lambda workload: workload.pop("targets_manifest_sha256"), "needs \\['targets_manifest_sha256'\\]"),
        ("configs/scenarios/v0.yaml", lambda workload: workload.update(targets_manifest="manifests/x.csv"), "apply only to source_generator rescuenet"),
    ],
)
def test_invalid_source_generator_fields_are_rejected(tmp_path: Path, source, change, message):
    with pytest.raises(ValueError, match=message):
        load_scenario_set_config(_write_config(tmp_path, source, change))


def test_scenarios_command_builds_rescuenet_scenarios_from_the_manifest(tmp_path: Path):
    root = tmp_path / "data"
    write_scenario_inputs(root)
    targets = root / "manifests" / "rescuenet_targets.csv"
    targets.write_text("target_id,x_m,y_m\n" + "".join(f"t{index},{x},{y}\n" for index, (x, y) in enumerate(POINTS)), encoding="utf-8")
    raw = yaml.safe_load(Path("configs/scenarios/rescuenet.yaml").read_text(encoding="utf-8"))
    raw["topology"].update(strata=4, dev_count=1, eval_replicates=2, dev_replicates=1)
    raw["workload"]["targets_manifest_sha256"] = hashlib.sha256(targets.read_bytes()).hexdigest()
    config = tmp_path / "rescuenet.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert main(["scenarios", "--config", str(config), "--data-root", str(root)]) == 0
    with (root / "manifests" / "scenarios_rescuenet.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7
    scenario = load_scenario(root / "processed" / "scenarios" / "rescuenet" / f"{rows[0]['scenario_id']}.json")
    assert [source.xy for source in scenario.sources] == list(POINTS)
    raw["workload"]["targets_manifest_sha256"] = "0" * 64
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        main(["scenarios", "--config", str(config), "--data-root", str(root)])
