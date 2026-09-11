import argparse
from dataclasses import asdict
from dataclasses import dataclass, field
import csv
import hashlib
import json
from pathlib import Path
import sys

import httpx
from models.paths import ground_connected_source_fraction
from models.scenario import scenario_from_dict

from .acquisition import download_http, safe_extract_zip, verify_file
from .config import load_config
from .manifests import write_source_ledger
from .models import ArtifactSpec, PipelineConfig, SourceRecord
from .rescuenet import count_mask_classes, load_label_map, pair_images_and_masks, RescueRecord, select_pairs
from .scenarios import build_scenario_set, load_scenario_set_config
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
    scenarios = subparsers.add_parser("scenarios")
    scenarios.add_argument("--config", type=Path, required=True)
    scenarios.add_argument("--data-root", type=Path)
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
    for path in sorted(topology_root.rglob("*.graphml")) if topology_root.exists() else []:
        record = inventory_topology(path)
        if record.eligible:
            network = normalize_topology(path, context.config.box_size_m)
            output = network_dir / f"{record.topology_id}.json"
            output.write_text(json.dumps(network, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_scenarios(config_path: Path, data_root: Path | None) -> int:
    config = load_scenario_set_config(config_path)
    root = data_root or config.data_root
    with (root / "manifests" / "topology_inventory.csv").open(newline="", encoding="utf-8") as stream:
        eligible_ids = sorted(row["topology_id"] for row in csv.DictReader(stream) if row["eligible"] == "True")
    networks, network_sha256 = {}, {}
    for topology_id in eligible_ids:
        path = root / "processed" / "networks" / f"{topology_id}.json"
        if path.exists():
            payload = path.read_bytes()
            networks[topology_id] = json.loads(payload)
            network_sha256[topology_id] = hashlib.sha256(payload).hexdigest()
    sndlib_matches = sorted((root / "raw" / "sndlib-networks-xml" / "extracted").rglob(f"{config.sndlib_instance}.xml"))
    if not sndlib_matches:
        raise FileNotFoundError(f"missing SNDlib instance {config.sndlib_instance}.xml under {root / 'raw'}")
    demand_values = [float(demand.value) for demand in parse_sndlib_xml(sndlib_matches[0]).demands]
    ledger = json.loads((root / "manifests" / "sources.json").read_text(encoding="utf-8"))
    raw_sha256 = [
        record["sha256"] for record in ledger if record["source_id"] in ("topology-zoo", "sndlib-networks-xml")
    ]
    scenarios = build_scenario_set(
        networks,
        network_sha256,
        demand_values,
        config,
        raw_sha256,
        hashlib.sha256(config_path.read_bytes()).hexdigest(),
    )
    output_dir = root / "processed" / "scenarios" / config.set_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.json"):
        stale.unlink()
    rows = []
    for raw in scenarios:
        scenario = scenario_from_dict(raw)
        (output_dir / f"{scenario.scenario_id}.json").write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "split": scenario.split,
                "topology_id": scenario.network_id,
                "replicate": int(scenario.scenario_id.rsplit("-r", 1)[1]),
                "scenario_seed": scenario.scenario_seed,
                "sha256": scenario.sha256,
                "node_count": len(scenario.nodes),
                "alert_count": len(scenario.alerts),
                "failed_edge_count": sum(1 for edge in scenario.edges if edge.failed),
                "ground_connected_source_fraction": round(ground_connected_source_fraction(scenario), 6),
            }
        )
    _write_csv(root / "manifests" / f"scenarios_{config.set_id}.csv", rows)
    return 0

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
    if args.command == "scenarios":
        return run_scenarios(args.config, args.data_root)
    config = load_config(args.profile)
    data_root = args.data_root or config.data_root
    return run_stage(args.command, config, data_root, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
