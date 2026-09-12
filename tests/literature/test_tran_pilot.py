import csv
import importlib.util
import json
from pathlib import Path

from literature import pilot
from models.scenario import scenario_from_dict
from runner.config import load_experiment_config
from tran_builders import tran_raw

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "tran_pilot.py"
SPEC = importlib.util.spec_from_file_location("tran_pilot", SCRIPT)
tran_pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tran_pilot)


def _row(theta, mu, timely, initial=0.1, runtime=1.0):
    return {"theta": theta, "mu": mu, "timely_ratio": timely, "initial_timely_ratio": initial, "planning_runtime_s": runtime}


def test_block_rule_switches_above_the_median_runtime_limit():
    assert pilot.choose_block_slots([_row(0.5, 1.0, 0.3, runtime=value) for value in (100.0, 700.0, 500.0)]) == 1
    assert pilot.choose_block_slots([_row(0.5, 1.0, 0.3, runtime=value) for value in (601.0, 700.0, 500.0)]) == 5


def test_parameter_choice_prefers_mean_then_theta_half_then_small_mu():
    rows = [_row(1 / 3, 1.0, 0.4), _row(1 / 3, 1.0, 0.2), _row(0.5, 10.0, 0.3), _row(0.5, 10.0, 0.3), _row(2 / 3, 1.0, 0.1)]
    assert pilot.choose_parameters(rows) == (0.5, 10.0)
    assert pilot.choose_parameters([_row(2 / 3, 1.0, 0.5), _row(1 / 3, 10.0, 0.6)]) == (1 / 3, 10.0)


def test_samir_rule_fires_when_half_the_scenarios_do_not_improve():
    assert not pilot.samir_fallback_fires([_row(0.5, 1.0, 0.3), _row(0.5, 1.0, 0.3), _row(0.5, 1.0, 0.1)])
    assert pilot.samir_fallback_fires([_row(0.5, 1.0, 0.3), _row(0.5, 1.0, 0.1)])


def test_pilot_writes_runs_choice_and_a_loadable_comparison_config(tmp_path: Path):
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    rows = []
    for index in range(2):
        raw = tran_raw()
        raw["scenario_id"], raw["split"] = f"Tiny{index}-r0", "dev"
        raw["network"]["network_id"] = f"Tiny{index}"
        raw["provenance"]["scenario_seed"] = 30 + index
        (scenario_dir / f"{raw['scenario_id']}.json").write_text(json.dumps(raw), encoding="utf-8")
        rows.append({"scenario_id": raw["scenario_id"], "split": "dev", "sha256": scenario_from_dict(raw).sha256})
    manifest = tmp_path / "scenarios.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario_id", "split", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    output, config_path = tmp_path / "pilot", tmp_path / "phase2_literature.yaml"
    code = tran_pilot.main(["--manifest", str(manifest), "--scenario-dir", str(scenario_dir), "--output", str(output),
                            "--config-output", str(config_path), "--realizations", "2", "--workers", "1"])
    choice = json.loads((output / "choice.json").read_text(encoding="utf-8"))
    with (output / "runs.csv").open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    assert [row["step"] for row in runs].count("runtime") == 2 and [row["step"] for row in runs].count("grid") == 12
    assert choice["block_slots"] == 1 and choice["scenarios"] == ["Tiny0-r0", "Tiny1-r0"]
    if choice["samir_fallback"]:
        assert code == 2 and not config_path.exists()
    else:
        assert code == 0
        spec = load_experiment_config(config_path).methods[1]
        assert spec.tran_params().theta == choice["theta"] and spec.tran_params().mu == choice["mu"] and spec.bounds
