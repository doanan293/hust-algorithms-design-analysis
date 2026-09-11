import hashlib
import os
from pathlib import Path
import platform
import subprocess

import numpy
import scipy

from bounds.solve import highs_version

SRC_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGES = ("models", "baselines", "bounds", "optimization", "runner")


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
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
