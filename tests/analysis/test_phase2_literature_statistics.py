import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "phase2_literature_statistics.py"
SPEC = importlib.util.spec_from_file_location("phase2_literature_statistics", SCRIPT)
literature_statistics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(literature_statistics)
COLUMNS = ["set_id", "scenario_id", "topology_id", "method", "realization_id", "timely_ratio", "timely_count", "connected_count", "shortfall"]


def _experiment(root: Path, name: str, values: dict[str, float], b1_timely: int = 3) -> Path:
    results = root / name
    results.mkdir()
    (results / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    with (results / "method_realizations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for set_id in ("v0", "v0-bh50"):
            for topology in range(10):
                for method, value in {**values, "B1": 0.1}.items():
                    timely = b1_timely if method == "B1" else round(40 * value)
                    writer.writerow({"set_id": set_id, "scenario_id": f"T{topology}-r0", "topology_id": f"T{topology}", "method": method,
                                     "realization_id": "0", "timely_ratio": str(value), "timely_count": str(timely),
                                     "connected_count": "7", "shortfall": "12.5"})
    return results


def test_writes_the_four_literature_comparisons(tmp_path: Path):
    main = _experiment(tmp_path, "main", {"P": 0.5, "B2": 0.375})
    literature = _experiment(tmp_path, "literature", {"TRAN": 0.25})
    output = tmp_path / "statistics_literature.csv"
    assert literature_statistics.main(["--main", str(main), "--literature", str(literature), "--output", str(output)]) == 0
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["set_id"], row["comparison"]) for row in rows] == [
        ("v0", "P - TRAN"), ("v0", "B2 - TRAN"), ("v0-bh50", "P - TRAN"), ("v0-bh50", "B2 - TRAN"),
    ]
    assert float(rows[0]["mean_difference"]) == 0.25 and float(rows[1]["mean_difference"]) == 0.125
    assert float(rows[0]["p_holm"]) == 4 * 2 / 1024


def test_anchor_mismatch_stops_before_pairing(tmp_path: Path, capsys):
    main = _experiment(tmp_path, "main", {"P": 0.5, "B2": 0.375})
    literature = _experiment(tmp_path, "literature", {"TRAN": 0.25}, b1_timely=4)
    output = tmp_path / "statistics_literature.csv"
    assert literature_statistics.main(["--main", str(main), "--literature", str(literature), "--output", str(output)]) == 1
    assert "timely_count" in capsys.readouterr().err and not output.exists()
