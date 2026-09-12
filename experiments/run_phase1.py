import argparse
from dataclasses import replace
from pathlib import Path
import sys

from runner.config import load_experiment_config
from runner.experiment import run_experiment


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="run-phase1")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/phase1.yaml"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir", type=Path, help="write results here instead of the configured output_dir")
    args = parser.parse_args(arguments)
    config = load_experiment_config(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    return run_experiment(config, args.config, workers=args.workers, argv=["experiments/run_phase1.py", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
