import copy
import csv
import json
from pathlib import Path

import yaml

from models.scenario import load_scenario
from scenario_builders import BASE_SCENARIO, make_alert


def write_tiny_experiment(root: Path, output_dir: Path, workers: int = 2, methods=("B0", "B1"), crosscheck: bool = True) -> Path:
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, topology in enumerate(("TinyA", "TinyB")):
        raw = copy.deepcopy(BASE_SCENARIO)
        raw["scenario_id"] = f"{topology}-r0"
        raw["network"]["network_id"] = topology
        raw["provenance"]["scenario_seed"] = 11 + index
        raw["alerts"] = [
            make_alert("alert-0000", "s00", 0, 12, 300000 + 100000 * index),
            make_alert("alert-0001", "s01", 1, 20, 40000),
            make_alert("alert-0002", "s00", 2, 16, 250000),
        ]
        path = scenario_dir / f"{raw['scenario_id']}.json"
        path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append({"scenario_id": raw["scenario_id"], "split": "eval", "topology_id": topology, "sha256": load_scenario(path).sha256})
    manifest = root / "scenarios.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario_id", "split", "topology_id", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "experiment_id": "tiny",
        "scenario_sets": [{"set_id": "tiny", "manifest": str(manifest), "scenario_dir": str(scenario_dir)}],
        "split": "eval",
        "realization_ids": {"start": 0, "stop": 2},
        "methods": list(methods),
        "bounds": {"tangents": 10},
        "bootstrap": {"resamples": 200, "seed": 5, "confidence": 0.95},
        "workers": workers,
        "output_dir": str(output_dir),
    }
    if crosscheck:
        config["milp_crosscheck"] = {"set_id": "tiny", "scenario_count": 2, "alert_count": 2, "realization_id": 0, "time_limit_s": 30}
    path = root / f"experiment-{output_dir.name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
