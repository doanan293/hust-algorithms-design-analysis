from pathlib import Path
import subprocess
import sys


def test_verify_script_shows_dry_run_sources():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/data/verify.py",
            "--profile",
            "configs/data/pilot.yaml",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "topology-zoo" in result.stdout
