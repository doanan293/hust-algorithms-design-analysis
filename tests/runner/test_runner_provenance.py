from pathlib import Path

from runner.provenance import environment, git_state, source_hash


def test_source_hash_tracks_file_content(tmp_path: Path):
    package = tmp_path / "models"
    package.mkdir()
    module = package / "a.py"
    module.write_text("x = 1\n", encoding="utf-8")
    first = source_hash(tmp_path, ("models",))
    assert first == source_hash(tmp_path, ("models",))
    module.write_text("x = 2\n", encoding="utf-8")
    assert source_hash(tmp_path, ("models",)) != first


def test_environment_and_git_state_outside_a_repository(tmp_path: Path):
    assert set(environment()) == {"python", "numpy", "scipy", "highs", "platform", "cpu_count"}
    assert git_state(tmp_path) == (None, None)
