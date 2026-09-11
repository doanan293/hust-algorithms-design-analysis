import csv
from pathlib import Path
from types import SimpleNamespace

from runner.config import ScenarioSetRef
from runner.experiment import scenario_refs


def test_scenario_refs_keep_the_split_and_the_listed_replicates(tmp_path: Path):
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    rows = [("B-r0", "eval", 0), ("A-r1", "eval", 1), ("A-r0", "eval", 0), ("C-r0", "dev", 0)]
    for scenario_id, _, _ in rows:
        (scenario_dir / f"{scenario_id}.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "scenarios.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario_id", "split", "replicate", "sha256"])
        writer.writerows([scenario_id, split, replicate, "0" * 64] for scenario_id, split, replicate in rows)

    def refs(replicates):
        config = SimpleNamespace(scenario_sets=(ScenarioSetRef("s", manifest, scenario_dir, replicates),), split="eval")
        return [ref.scenario_id for ref in scenario_refs(config)]

    assert refs(None) == ["A-r0", "A-r1", "B-r0"]
    assert refs((0,)) == ["A-r0", "B-r0"]
