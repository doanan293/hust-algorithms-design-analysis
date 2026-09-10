from pathlib import Path

import yaml

from .models import ArtifactSpec, PipelineConfig


def load_config(path: Path) -> PipelineConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    artifacts = tuple(
        ArtifactSpec(
            **{
                **item,
                "license_evidence": tuple(item.get("license_evidence", ())),
            }
        )
        for item in raw["artifacts"]
    )
    ids = [item.source_id for item in artifacts]
    duplicate = next((value for value in ids if ids.count(value) > 1), None)
    if duplicate:
        raise ValueError(f"duplicate source_id: {duplicate}")
    count = int(raw["rescuenet_selection_count"])
    if count != 30:
        raise ValueError("paper-compatible profiles require 30 RescueNet pairs")
    return PipelineConfig(
        profile=str(raw["profile"]),
        artifacts=artifacts,
        rescuenet_selection_count=count,
        selection_seed=int(raw.get("selection_seed", 20260910)),
        box_size_m=float(raw.get("box_size_m", 2000.0)),
        data_root=Path(raw.get("data_root", "data")),
    )
