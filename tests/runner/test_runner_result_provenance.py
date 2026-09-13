import subprocess
from pathlib import Path

from runner.provenance import result_provenance_problem, source_hash


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    for package, content in (("models", "x = 1\n"), ("runner", "y = 1\n")):
        (root / "src" / package).mkdir(parents=True)
        (root / "src" / package / "module.py").write_text(content, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.com")
    _git(root, "config", "user.name", "tests")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "results")
    return root, _git(root, "rev-parse", "HEAD")


def _manifest(commit: str, dirty: bool = False, source: str = "0" * 64) -> dict[str, object]:
    return {"status": "complete", "source_hash": source, "git_commit": commit, "git_dirty": dirty}


def test_results_with_the_current_source_hash_are_current(tmp_path: Path):
    root, commit = _repository(tmp_path)
    assert result_provenance_problem(_manifest(commit, source=source_hash(root / "src")), root, root / "src") is None


def test_runner_changes_after_a_clean_ancestor_commit_keep_results_current(tmp_path: Path):
    root, commit = _repository(tmp_path)
    (root / "src" / "runner" / "module.py").write_text("y = 2\n", encoding="utf-8")
    _git(root, "commit", "-qam", "runner")
    assert result_provenance_problem(_manifest(commit), root, root / "src") is None


def test_files_added_to_result_packages_after_the_commit_keep_results_current(tmp_path: Path):
    root, commit = _repository(tmp_path)
    (root / "src" / "literature").mkdir()
    (root / "src" / "literature" / "method.py").write_text("z = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "new package")
    assert result_provenance_problem(_manifest(commit), root, root / "src") is None


def test_committed_or_uncommitted_result_code_changes_are_rejected(tmp_path: Path):
    root, commit = _repository(tmp_path)
    (root / "src" / "models" / "module.py").write_text("x = 2\n", encoding="utf-8")
    assert f"changed after {commit}" in result_provenance_problem(_manifest(commit), root, root / "src")
    _git(root, "commit", "-qam", "models")
    assert f"changed after {commit}" in result_provenance_problem(_manifest(commit), root, root / "src")


def test_incomplete_dirty_and_foreign_results_are_rejected(tmp_path: Path):
    root, commit = _repository(tmp_path)
    assert result_provenance_problem({**_manifest(commit), "status": "failed"}, root, root / "src") == "the experiment is not complete"
    assert "record no clean commit" in result_provenance_problem(_manifest(commit, dirty=True), root, root / "src")
    assert "not an ancestor of HEAD" in result_provenance_problem(_manifest("f" * 40), root, root / "src")
