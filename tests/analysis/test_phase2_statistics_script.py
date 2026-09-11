import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "phase2_statistics.py"
SPEC = importlib.util.spec_from_file_location("phase2_statistics", SCRIPT)
phase2_statistics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase2_statistics)
VALUES = {"B1": 0.25, "B2": 0.5, "B3": 0.375, "P": 0.5}


def _results(root: Path, status: str) -> Path:
    results = root / "main"
    results.mkdir()
    (results / "manifest.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    with (results / "method_realizations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["set_id", "scenario_id", "topology_id", "method", "realization_id", "timely_ratio"])
        writer.writeheader()
        for set_id in ("v0", "v0-bh50"):
            for topology in range(10):
                for method, value in VALUES.items():
                    writer.writerow({"set_id": set_id, "scenario_id": f"T{topology}-r0", "topology_id": f"T{topology}",
                                     "method": method, "realization_id": "0", "timely_ratio": str(value)})
    return results


def test_statistics_script_writes_the_eight_primary_comparisons(tmp_path: Path):
    output = tmp_path / "statistics.csv"
    assert phase2_statistics.main(["--results", str(_results(tmp_path, "complete")), "--output", str(output)]) == 0
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["set_id"], row["comparison"]) for row in rows][:4] == [("v0", "P - B2"), ("v0", "B2 - B1"), ("v0", "B2 - B3"), ("v0", "P - B1")]
    assert len(rows) == 8 and float(rows[1]["mean_difference"]) == 0.25 and float(rows[0]["p_value"]) == 1.0
    assert float(rows[1]["p_holm"]) == min(1.0, 8 * 2 / 1024)


def test_statistics_script_refuses_incomplete_results(tmp_path: Path, capsys):
    output = tmp_path / "statistics.csv"
    assert phase2_statistics.main(["--results", str(_results(tmp_path, "failed")), "--output", str(output)]) == 1
    assert "no complete experiment" in capsys.readouterr().err and not output.exists()
