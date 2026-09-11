import argparse
from pathlib import Path
import sys

from runner.config import load_experiment_config
from runner.experiment import run_experiment


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="run-phase2")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/phase2_pilot.yaml"))
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(arguments)
    config = load_experiment_config(args.config)
    return run_experiment(config, args.config, workers=args.workers, argv=["experiments/run_phase2.py", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
