import csv
import json
from pathlib import Path

from data.cli import main
from models.scenario import load_scenario
from scenario_stage_inputs import write_scenario_inputs, write_small_config


def test_scenarios_stage_writes_valid_files_and_reproducible_manifest(tmp_path: Path):
    root = tmp_path / "data"
    write_scenario_inputs(root)
    config = write_small_config(tmp_path)
    assert main(["scenarios", "--config", str(config), "--data-root", str(root)]) == 0
    manifest = root / "manifests" / "scenarios_v0.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7
    assert sorted(row["split"] for row in rows).count("dev") == 1
    for row in rows:
        scenario = load_scenario(root / "processed" / "scenarios" / "v0" / f"{row['scenario_id']}.json")
        assert scenario.sha256 == row["sha256"]
        assert scenario.provenance["raw_sha256"] == ["1" * 64]
        assert 0.0 <= float(row["ground_connected_source_fraction"]) <= 1.0
    first = manifest.read_text(encoding="utf-8")
    assert main(["scenarios", "--config", str(config), "--data-root", str(root)]) == 0
    assert manifest.read_text(encoding="utf-8") == first

