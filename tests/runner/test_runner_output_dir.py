import importlib.util
from pathlib import Path

import pytest

from experiment_fixtures import write_tiny_experiment

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["run_phase1", "run_phase2"])
def test_output_dir_overrides_the_configured_directory(tmp_path: Path, name: str):
    configured = tmp_path / "configured"
    config_path = write_tiny_experiment(tmp_path, configured, 1, ("B0",), crosscheck=False)
    override = tmp_path / "override"
    assert _script(name).main(["--config", str(config_path), "--output-dir", str(override), "--workers", "1"]) == 0
    assert (override / "summary.csv").exists() and not configured.exists()
