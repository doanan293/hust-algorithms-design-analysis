from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ArtifactSpec:
    source_id: str
    dataset_family: str
    kind: Literal["http", "git"]
    url: str
    filename: str | None = None
    revision: str | None = None
    upstream_version: str | None = None
    expected_size: int | None = None
    upstream_checksum: str | None = None
    landing_page: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    license_evidence: tuple[str, ...] = field(default_factory=tuple)
    license_review_required: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    profile: str
    artifacts: tuple[ArtifactSpec, ...]
    rescuenet_selection_count: int
    selection_seed: int
    box_size_m: float
    data_root: Path = Path("data")


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    dataset_family: str
    landing_page: str
    download_url: str
    retrieved_at_utc: str
    upstream_version: str | None
    filename: str
    byte_size: int
    upstream_checksum: str | None
    sha256: str
    license_name: str | None
    license_url: str | None
    profile: str
    status: str
    license_review_required: bool
    evidence: tuple[str, ...] = field(default_factory=tuple)
