import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

import httpx

from .acquisition import download_http, verify_file
from .config import load_config
from .models import ArtifactSpec, PipelineConfig, SourceRecord
from .topology_zoo import fetch_pinned_repo


@dataclass
class PipelineContext:
    config: PipelineConfig
    data_root: Path
    records: list[SourceRecord] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "verify", "inventory", "preprocess", "all"):
        command = subparsers.add_parser(name)
        command.add_argument("--profile", type=Path, required=True)
        command.add_argument("--data-root", type=Path)
        command.add_argument("--dry-run", action="store_true")
    return parser


def _target_for(spec: ArtifactSpec, data_root: Path) -> Path:
    if spec.kind == "git":
        return data_root / "raw" / spec.source_id
    if not spec.filename:
        raise ValueError(f"HTTP artifact {spec.source_id} needs filename")
    return data_root / "raw" / spec.source_id / spec.filename


def run_download(context: PipelineContext) -> None:
    raw_root = context.data_root / "raw"
    with httpx.Client(timeout=None) as client:
        for spec in context.config.artifacts:
            if spec.kind == "git":
                record = fetch_pinned_repo(
                    spec, raw_root / spec.source_id, context.config.profile
                )
            else:
                record = download_http(
                    spec, raw_root, context.config.profile, client
                )
            context.records.append(record)


def run_verify(context: PipelineContext) -> None:
    for spec in context.config.artifacts:
        target = _target_for(spec, context.data_root)
        if spec.kind == "http":
            verify_file(target, spec)
        elif not target.exists():
            raise FileNotFoundError(f"missing Git artifact: {target}")


def run_inventory(context: PipelineContext) -> None:
    manifest_root = context.data_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "inventory-run.json").write_text(
        json.dumps(
            {
                "profile": context.config.profile,
                "artifact_ids": [spec.source_id for spec in context.config.artifacts],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_preprocess(context: PipelineContext) -> None:
    processed = context.data_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "README.txt").write_text(
        "Source-specific preprocessing is enabled after inventory.\n",
        encoding="utf-8",
    )


STAGES = ("download", "verify", "inventory", "preprocess")


def run_stage(command: str, config: PipelineConfig, data_root: Path, dry_run: bool) -> int:
    if dry_run:
        for spec in config.artifacts:
            print(
                json.dumps(
                    {
                        "source_id": spec.source_id,
                        "destination": str(data_root / "raw" / spec.source_id),
                        "revision": spec.revision or spec.upstream_version,
                        "expected_size": spec.expected_size,
                        "license_review_required": spec.license_review_required,
                    },
                    sort_keys=True,
                )
            )
        return 0
    stages = STAGES if command == "all" else (command,)
    context = PipelineContext(config=config, data_root=data_root)
    handlers = {
        "download": run_download,
        "verify": run_verify,
        "inventory": run_inventory,
        "preprocess": run_preprocess,
    }
    for stage in stages:
        handlers[stage](context)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.profile)
    data_root = args.data_root or config.data_root
    return run_stage(args.command, config, data_root, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
