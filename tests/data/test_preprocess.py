from pathlib import Path
import shutil

from data.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_preprocess_no_longer_writes_placeholder_scenarios(tmp_path: Path):
    root = tmp_path / "data"
    topology = root / "raw" / "topology-zoo"
    topology.mkdir(parents=True)
    shutil.copy(FIXTURES / "topology_zoo" / "eligible.graphml", topology / "Eligible.graphml")
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "profile: fixture\n"
        f"data_root: {root}\n"
        "rescuenet_selection_count: 30\n"
        "selection_seed: 20260910\n"
        "box_size_m: 2000.0\n"
        "artifacts: []\n",
        encoding="utf-8",
    )
    assert main(["preprocess", "--profile", str(profile)]) == 0
    assert (root / "processed" / "networks" / "Eligible.json").exists()
    assert not (root / "processed" / "scenarios").exists()
