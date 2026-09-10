from pathlib import Path

from data.cli import build_parser, main


def test_cli_exposes_all_lifecycle_stages():
    parser = build_parser()
    for command in ("download", "verify", "inventory", "preprocess", "all"):
        args = parser.parse_args([command, "--profile", "configs/data/paper.yaml", "--dry-run"])
        assert args.command == command
        assert args.dry_run is True


def test_dry_run_reports_profile_sources(capsys):
    assert main(["all", "--profile", "configs/data/pilot.yaml", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "topology-zoo" in output
    assert "sndlib-networks-xml" in output
    assert "rescuenet-descriptor" in output
