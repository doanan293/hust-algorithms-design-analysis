import argparse
from dataclasses import asdict
from dataclasses import dataclass, field
import csv
import hashlib
import json
from pathlib import Path
import sys

import httpx

from .acquisition import download_http, safe_extract_zip, verify_file
from .config import load_config
from .manifests import write_source_ledger
from .models import ArtifactSpec, PipelineConfig, SourceRecord
from .rescuenet import count_mask_classes, load_label_map, pair_images_and_masks, RescueRecord, select_pairs
from .scenarios import ScenarioInputs, canonical_hash, generate_scenario
from .sndlib import inventory_sndlib, parse_sndlib_xml
from .topology_zoo import fetch_pinned_repo, inventory_topology, normalize_topology


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
    write_source_ledger(context.data_root / "manifests" / "sources.json", context.records)


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
    topology_root = context.data_root / "raw" / "topology-zoo"
    topology_rows = []
    if topology_root.exists():
        for path in sorted(topology_root.rglob("*.graphml")):
            topology_rows.append(asdict(inventory_topology(path)))
    _write_csv(manifest_root / "topology_inventory.csv", topology_rows)

    sndlib_archive = context.data_root / "raw" / "sndlib-networks-xml" / "sndlib-networks-xml.zip"
    sndlib_root = context.data_root / "raw" / "sndlib-networks-xml" / "extracted"
    if sndlib_archive.exists() and not sndlib_root.exists():
        safe_extract_zip(sndlib_archive, sndlib_root)
    sndlib_rows = []
    for path in sorted(sndlib_root.rglob("*.xml")) if sndlib_root.exists() else []:
        sndlib_rows.append(asdict(inventory_sndlib(path)))
    _write_csv(manifest_root / "sndlib_inventory.csv", sndlib_rows)

    rescue_archive = context.data_root / "raw" / "rescuenet-validation" / "segmentation-validationset.zip"
    rescue_root = context.data_root / "raw" / "rescuenet-validation" / "extracted"
    if rescue_archive.exists() and not rescue_root.exists():
        safe_extract_zip(rescue_archive, rescue_root)
    descriptor = context.data_root / "raw" / "rescuenet-descriptor" / "RescueNet-Segmentation-Dataset-Note.txt"
    rescue_rows = []
    rescue_records = []
    if rescue_root.exists() and descriptor.exists():
        pair_root = _find_pair_root(rescue_root)
        labels = load_label_map(descriptor)
        for pair in pair_images_and_masks(pair_root):
            counts = count_mask_classes(pair, labels)
            presence = tuple(sorted(label for label, pixels in counts.items() if label != 0 and pixels > 0))
            rescue_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "image": str(pair.image.relative_to(rescue_root)),
                    "mask": str(pair.mask.relative_to(rescue_root)),
                    "class_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
                    "presence_vector": json.dumps(presence),
                }
            )
            rescue_records.append(
                RescueRecord(
                    pair_id=pair.pair_id,
                    image=pair.image,
                    mask=pair.mask,
                    class_counts=tuple(sorted((int(k), int(v)) for k, v in counts.items())),
                    presence_vector=presence,
                )
            )
    _write_csv(manifest_root / "rescuenet_inventory.csv", rescue_rows)
    if rescue_records:
        selected = select_pairs(
            rescue_records,
            count=min(context.config.rescuenet_selection_count, len(rescue_records)),
            seed=context.config.selection_seed,
        )
        _write_csv(
            manifest_root / "rescuenet_selection.csv",
            [
                {
                    "rank": rank,
                    "pair_id": record.pair_id,
                    "image": str(record.image.relative_to(rescue_root)),
                    "mask": str(record.mask.relative_to(rescue_root)),
                    "class_counts": json.dumps(dict(record.class_counts), sort_keys=True),
                    "presence_vector": json.dumps(record.presence_vector),
                    "selection_seed": context.config.selection_seed,
                }
                for rank, record in enumerate(selected, start=1)
            ],
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _find_pair_root(root: Path) -> Path:
    directories = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    for directory in directories:
        standard = (directory / "images").is_dir() and (directory / "masks").is_dir()
        validation = (directory / "val-org-img").is_dir() and (
            directory / "val-label-img"
        ).is_dir()
        if standard or validation:
            return directory
    raise ValueError(f"RescueNet extracted archive has no images/masks root: {root}")


def run_preprocess(context: PipelineContext) -> None:
    processed = context.data_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    network_dir = processed / "networks"
    network_dir.mkdir(parents=True, exist_ok=True)
    topology_root = context.data_root / "raw" / "topology-zoo"
    normalized = []
    for path in sorted(topology_root.rglob("*.graphml")) if topology_root.exists() else []:
        record = inventory_topology(path)
        if record.eligible:
            network = normalize_topology(path, context.config.box_size_m)
            output = network_dir / f"{record.topology_id}.json"
            output.write_text(json.dumps(network, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            normalized.append(network)
    if not normalized:
        return
    sndlib_root = context.data_root / "raw" / "sndlib-networks-xml" / "extracted"
    demand_records = []
    first_xml = next(iter(sorted(sndlib_root.rglob("*.xml"))), None) if sndlib_root.exists() else None
    if first_xml:
        demand_records = tuple(
            {"original_value": str(demand.value)}
            for demand in parse_sndlib_xml(first_xml).demands
        )
    if not demand_records:
        demand_records = ({"original_value": "1"},)
    config_hash = hashlib.sha256(
        json.dumps(asdict(context.config), default=str, sort_keys=True).encode()
    ).hexdigest()
    scenario = generate_scenario(
        ScenarioInputs(
            network=normalized[0],
            demands=demand_records,
            targets=(),
            raw_sha256=tuple(sorted(record.sha256 for record in context.records)),
            config_hash=config_hash,
            seed=context.config.selection_seed,
            alert_count=30,
            failure_probability=0.2,
        )
    )
    scenario_dir = processed / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / f"scenario-{canonical_hash(scenario)[:12]}.json").write_text(
        json.dumps(scenario, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
