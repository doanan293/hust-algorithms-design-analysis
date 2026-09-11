import csv
import json
from pathlib import Path
import shutil

import yaml

from data.cli import main
from models.scenario import load_scenario

FIXTURES = Path(__file__).parent / "fixtures"


def _write_inputs(root: Path) -> None:
    network_dir = root / "processed" / "networks"
    network_dir.mkdir(parents=True)
    rows = []
    for index in range(6):
        nodes = [{"id": f"n{i}", "x_m": 2000.0 * i / (15 + index), "y_m": 1000.0 + 50.0 * (i % 3)} for i in range(16 + index)]
        edges = [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(15 + index)]
        network = {
            "network_id": f"Line{index}",
            "nodes": nodes,
            "edges": edges,
            "normalization": {"box_size_m": 2000.0, "scale_x": 1.0, "scale_y": 1.0},
        }
        (network_dir / f"Line{index}.json").write_text(json.dumps(network), encoding="utf-8")
        rows.append({"topology_id": f"Line{index}", "eligible": "True"})
    rows.append({"topology_id": "Tiny", "eligible": "False"})
    manifests = root / "manifests"
    manifests.mkdir()
    with (manifests / "topology_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["eligible", "topology_id"])
        writer.writeheader()
        writer.writerows(rows)
    (manifests / "sources.json").write_text(
        json.dumps([{"source_id": "topology-zoo", "sha256": "1" * 64}, {"source_id": "rescuenet-validation", "sha256": "2" * 64}]),
        encoding="utf-8",
    )
    sndlib = root / "raw" / "sndlib-networks-xml" / "extracted" / "sndlib-networks-xml"
    sndlib.mkdir(parents=True)
    shutil.copy(FIXTURES / "sndlib" / "mini.xml", sndlib / "abilene.xml")


def _write_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/scenarios/v0.yaml").read_text(encoding="utf-8"))
    raw["topology"].update(strata=4, dev_count=1, eval_replicates=2, dev_replicates=1)
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_scenarios_stage_writes_valid_files_and_reproducible_manifest(tmp_path: Path):
    root = tmp_path / "data"
    _write_inputs(root)
    config = _write_config(tmp_path)
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

