import hashlib
import os
from pathlib import Path
import platform
import subprocess
from typing import Mapping

import clarabel
import cvxpy
import numpy
import scipy

from bounds.solve import highs_version

SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
SOURCE_PACKAGES = ("models", "baselines", "bounds", "optimization", "literature", "runner")
RESULT_PACKAGES = ("models", "baselines", "bounds", "optimization", "literature")


def source_hash(src_root: Path = SRC_ROOT, packages: tuple[str, ...] = SOURCE_PACKAGES) -> str:
    digest = hashlib.sha256()
    for package in packages:
        for path in sorted((src_root / package).rglob("*.py")):
            digest.update(path.relative_to(src_root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "configs", "experiments"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "highs": highs_version(),
        "cvxpy": cvxpy.__version__,
        "clarabel": clarabel.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }


def result_provenance_problem(
    manifest: Mapping[str, object], repo_root: Path = REPO_ROOT, src_root: Path = SRC_ROOT
) -> str | None:
    """Why a results manifest does not describe the current result code, or None when it does (spec D, Plan D.3 adjustments).

    Results are current when their source hash equals the current one, or when they come from a clean commit that is an
    ancestor of HEAD and no file of the packages that determine results (RESULT_PACKAGES) was modified, deleted or renamed
    since that commit, including uncommitted changes. Files added later are ignored, because existing code must change to use
    them. The runner is left out: it orchestrates tasks, and the experiments level of experiments/reproduce.py checks it end
    to end.
    """
    if manifest.get("status") != "complete":
        return "the experiment is not complete"
    if manifest.get("source_hash") == source_hash(src_root):
        return None
    commit = manifest.get("git_commit")
    if not commit or manifest.get("git_dirty") is not False:
        return "the results were produced by different source code and record no clean commit"
    paths = [f"src/{package}" for package in RESULT_PACKAGES]
    try:
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", str(commit), "HEAD"], cwd=repo_root, capture_output=True)
        changed = subprocess.run(["git", "diff", "--quiet", "--diff-filter=a", str(commit), "--", *paths], cwd=repo_root, capture_output=True)
    except OSError:
        return "the results were produced by different source code and Git is unavailable"
    if ancestor.returncode != 0:
        return f"the results were produced by different source code at {commit}, which is not an ancestor of HEAD"
    if changed.returncode != 0:
        return f"the result code in {', '.join(paths)} changed after {commit}"
    return None
