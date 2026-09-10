from pathlib import Path

from data.cli import main


def test_fixture_dry_run_does_not_write_data(tmp_path: Path, capsys):
    profile = tmp_path / "fixture.yaml"
    profile.write_text(
        "profile: fixture\n"
        "data_root: " + str(tmp_path / "data") + "\n"
        "rescuenet_selection_count: 30\n"
        "selection_seed: 20260910\n"
        "box_size_m: 2000.0\n"
        "artifacts: []\n",
        encoding="utf-8",
    )
    assert main(["all", "--profile", str(profile), "--dry-run"]) == 0
    assert not (tmp_path / "data").exists()
    assert capsys.readouterr().out.strip() == ""
