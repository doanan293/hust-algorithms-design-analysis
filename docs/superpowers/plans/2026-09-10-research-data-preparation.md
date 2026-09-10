# Research Data Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible pipeline that acquires, verifies, inventories, selects, and preprocesses the SNDlib, Topology Zoo, and RescueNet data used by the paper.

**Architecture:** A Python package under `src/data/` owns typed configuration, acquisition, source-specific parsers, deterministic selection, and scenario generation. Thin commands under `scripts/data/` compose those units, while checked-in YAML profiles and JSON/CSV manifests form the reproducibility contract; large raw and processed artifacts remain ignored by Git.

**Tech Stack:** Python 3.11+, `httpx`, `PyYAML`, `networkx`, `numpy`, `Pillow`, `pyproj`, `pytest`, standard-library `argparse`, `dataclasses`, `hashlib`, `json`, `csv`, `zipfile`, and `xml.etree.ElementTree`.

**Spec:** `docs/superpowers/specs/2026-09-10-research-data-preparation-design.md`

## Global Constraints

- The default `paper` profile downloads Topology Zoo at commit `e278b1bdaafea5dac33883bf9c97401db4cd7347`, the official SNDlib XML network archive, and RescueNet validation file ID `40582916`.
- The RescueNet paper subset contains exactly 30 valid image-mask pairs selected deterministically.
- Raw third-party data and large processed artifacts must never be committed to Git.
- Every acquired artifact records its official source, upstream identifier, byte size, upstream checksum when available, local SHA-256, retrieval time, license evidence, profile, and status.
- Official-source failure must be explicit; the pipeline must not silently substitute a mirror or synthetic data.
- Downloads publish atomically, resume when supported, quarantine invalid existing files, and extract ZIP files without path traversal or symlink escape.
- Original SNDlib values and units remain intact; transformed alerts are labelled as derived data.
- Geographic coordinates use one scale factor for both axes when normalized to the configured square.
- Default tests run without internet access; live-source checks are opt-in.
- Implementation follows test-driven development and each task ends with its own commit.

---

### Task 1: Python project, configuration, and typed data contracts

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/data/__init__.py`
- Create: `src/data/models.py`
- Create: `src/data/config.py`
- Create: `configs/data/pilot.yaml`
- Create: `configs/data/paper.yaml`
- Create: `configs/data/full-rescuenet.yaml`
- Create: `tests/data/test_config.py`

**Interfaces:**
- Consumes: the approved design spec.
- Produces: `ArtifactSpec`, `SourceRecord`, `PipelineConfig`, and `load_config(path: Path) -> PipelineConfig` for every later task.

- [ ] **Step 1: Write the failing configuration tests**

```python
# tests/data/test_config.py
from pathlib import Path

import pytest

from data.config import load_config


def test_paper_profile_pins_required_sources():
    config = load_config(Path("configs/data/paper.yaml"))
    by_id = {artifact.source_id: artifact for artifact in config.artifacts}
    assert config.profile == "paper"
    assert config.rescuenet_selection_count == 30
    assert by_id["topology-zoo"].revision == "e278b1bdaafea5dac33883bf9c97401db4cd7347"
    assert by_id["sndlib-networks-xml"].url == (
        "https://sndlib.put.poznan.pl/download/sndlib-networks-xml.zip"
    )
    assert by_id["rescuenet-validation"].upstream_checksum == (
        "md5:51360be8ea3f73233605a767801c9d1c"
    )


def test_config_rejects_duplicate_source_ids(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "profile: bad\nrescuenet_selection_count: 30\nartifacts:\n"
        "  - {source_id: x, kind: http, url: https://example.test/a}\n"
        "  - {source_id: x, kind: http, url: https://example.test/b}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate source_id: x"):
        load_config(path)
```

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run: `python -m pytest tests/data/test_config.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'data'`.

- [ ] **Step 3: Add project metadata and dependencies**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "uav-relay-research-data"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "httpx>=0.27,<1",
  "networkx>=3.3,<4",
  "numpy>=2.0,<3",
  "Pillow>=10.4,<12",
  "pyproj>=3.6,<4",
  "PyYAML>=6.0,<7",
]

[project.optional-dependencies]
test = ["pytest>=8.3,<9"]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```gitignore
# .gitignore
.venv/
__pycache__/
.pytest_cache/
*.egg-info/
data/raw/
data/processed/
data/manifests/*.tmp
data/manifests/*.partial
```

- [ ] **Step 4: Implement the typed contracts and strict YAML loader**

```python
# src/data/models.py
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
```

```python
# src/data/config.py
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
```

Create all three YAML profiles with explicit artifact records. Begin `paper.yaml` with the verified immutable source values:

```yaml
profile: paper
data_root: data
rescuenet_selection_count: 30
selection_seed: 20260910
box_size_m: 2000.0
artifacts:
  - source_id: topology-zoo
    dataset_family: topology-zoo
    kind: git
    url: https://github.com/sk2/topologyzoo.git
    revision: e278b1bdaafea5dac33883bf9c97401db4cd7347
    landing_page: https://github.com/sk2/topologyzoo
  - source_id: sndlib-networks-xml
    dataset_family: sndlib
    kind: http
    url: https://sndlib.put.poznan.pl/download/sndlib-networks-xml.zip
    filename: sndlib-networks-xml.zip
    upstream_version: archive:2014-07-07
    upstream_checksum: sha256:1490c86061503f4160efaa6eed3bf4a0b314566cc5257bc37169c3d6c4e1e96f
    landing_page: https://sndlib.put.poznan.pl/download.action
  - source_id: rescuenet-descriptor
    dataset_family: rescuenet
    kind: http
    url: https://ndownloader.figshare.com/files/40583321
    filename: RescueNet-Segmentation-Dataset-Note.txt
    upstream_version: article:22826606
    expected_size: 330
    upstream_checksum: md5:82c158ff471b67f73f60c5213ec84251
    landing_page: https://springernature.figshare.com/articles/dataset/22826606
    license_review_required: true
    license_evidence:
      - Figshare article metadata reports CC0.
      - Author repository README reports CC BY-NC-ND for dataset content.
  - source_id: rescuenet-validation
    dataset_family: rescuenet
    kind: http
    url: https://ndownloader.figshare.com/files/40582916
    filename: segmentation-validationset.zip
    upstream_version: article:22826369
    expected_size: 2373908074
    upstream_checksum: md5:51360be8ea3f73233605a767801c9d1c
    landing_page: https://springernature.figshare.com/articles/dataset/22826369
    license_review_required: true
    license_evidence:
      - Figshare article metadata reports CC0.
      - Author repository README reports CC BY-NC-ND for dataset content.
```

`pilot.yaml` uses the first three records and omits `rescuenet-validation`. `full-rescuenet.yaml` uses all paper records and adds training file ID `40581458` with size `18699171789` and MD5 `7ab79503e940de6301098a01a035a62f`, plus test file ID `40583084` with size `2387196841` and MD5 `6dbd095cf14d9ddfdf1312d5f1949aeb`. Both retain the same Figshare evidence strings, selection seed, count, and box size.

- [ ] **Step 5: Install and run the tests**

Run: `python -m pip install -e '.[test]' && python -m pytest tests/data/test_config.py -v`

Expected: both tests pass.

- [ ] **Step 6: Commit the foundation**

```bash
git add pyproject.toml .gitignore src/data configs/data tests/data/test_config.py
git commit -m "build: add research data pipeline foundation"
```

---

### Task 2: Resumable acquisition, integrity verification, ledger, and safe extraction

**Files:**
- Create: `src/data/acquisition.py`
- Create: `src/data/manifests.py`
- Create: `tests/data/test_acquisition.py`
- Create: `tests/data/test_manifests.py`

**Interfaces:**
- Consumes: `ArtifactSpec`, `SourceRecord`, and `PipelineConfig` from Task 1.
- Produces: `sha256_file(path: Path) -> str`, `verify_file(path: Path, spec: ArtifactSpec) -> None`, `download_http(spec: ArtifactSpec, root: Path, profile: str, client: httpx.Client) -> SourceRecord`, `safe_extract_zip(archive: Path, destination: Path) -> None`, and `write_source_ledger(path: Path, records: Sequence[SourceRecord]) -> None`.

- [ ] **Step 1: Write failing tests for verification, atomic download, resume, and quarantine**

```python
# tests/data/test_acquisition.py
from pathlib import Path
import hashlib
import httpx
import pytest

from data.acquisition import ChecksumMismatch, download_http, safe_extract_zip
from data.models import ArtifactSpec


def spec_for(payload: bytes) -> ArtifactSpec:
    return ArtifactSpec(
        source_id="fixture",
        kind="http",
        url="https://example.test/file.zip",
        filename="file.zip",
        expected_size=len(payload),
        upstream_checksum="md5:" + hashlib.md5(payload).hexdigest(),
        landing_page="https://example.test",
    )


def test_download_publishes_verified_file(tmp_path: Path):
    payload = b"verified payload"
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload)))
    record = download_http(spec_for(payload), tmp_path, "paper", client)
    assert (tmp_path / "fixture" / "file.zip").read_bytes() == payload
    assert not (tmp_path / "fixture" / "file.zip.partial").exists()
    assert record.sha256 == hashlib.sha256(payload).hexdigest()


def test_invalid_existing_file_is_quarantined(tmp_path: Path):
    target = tmp_path / "fixture" / "file.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt")
    payload = b"verified payload"
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload)))
    download_http(spec_for(payload), tmp_path, "paper", client)
    assert list(target.parent.glob("file.zip.corrupt-*"))


def test_checksum_mismatch_keeps_partial_file(tmp_path: Path):
    expected = spec_for(b"expected")
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"wrongxx")))
    with pytest.raises(ChecksumMismatch):
        download_http(expected, tmp_path, "paper", client)
    assert (tmp_path / "fixture" / "file.zip.partial").exists()
```

Add a resume test whose mock returns `206` only when `Range: bytes=4-` is present, and ZIP tests that reject `../escape.txt`, `/absolute.txt`, and symlink entries while accepting a normal nested file.

- [ ] **Step 2: Run the acquisition tests and verify they fail**

Run: `python -m pytest tests/data/test_acquisition.py -v`

Expected: collection fails because `data.acquisition` does not exist.

- [ ] **Step 3: Implement checksum verification and HTTP acquisition**

```python
# src/data/acquisition.py
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import os
import zipfile

import httpx

from .models import ArtifactSpec, SourceRecord


class ChecksumMismatch(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, spec: ArtifactSpec) -> None:
    if spec.expected_size is not None and path.stat().st_size != spec.expected_size:
        raise ChecksumMismatch(f"size mismatch for {spec.source_id}")
    if spec.upstream_checksum:
        algorithm, expected = spec.upstream_checksum.split(":", 1)
        digest = hashlib.new(algorithm)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ChecksumMismatch(f"{algorithm} mismatch for {spec.source_id}")


def download_http(spec: ArtifactSpec, root: Path, profile: str, client: httpx.Client) -> SourceRecord:
    directory = root / spec.source_id
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / str(spec.filename)
    if target.exists():
        try:
            verify_file(target, spec)
        except ChecksumMismatch:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            target.replace(target.with_name(f"{target.name}.corrupt-{stamp}"))
        else:
            return _record(spec, target, profile, "reused")
    partial = target.with_name(target.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with client.stream("GET", spec.url, headers=headers, follow_redirects=True) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        with partial.open("ab" if append else "wb") as stream:
            for chunk in response.iter_bytes(1024 * 1024):
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    verify_file(partial, spec)
    partial.replace(target)
    return _record(spec, target, profile, "downloaded")


def _record(spec: ArtifactSpec, target: Path, profile: str, status: str) -> SourceRecord:
    return SourceRecord(
        source_id=spec.source_id,
        dataset_family=spec.dataset_family,
        landing_page=spec.landing_page or spec.url,
        download_url=spec.url,
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        upstream_version=spec.upstream_version or spec.revision,
        filename=target.name,
        byte_size=target.stat().st_size,
        upstream_checksum=spec.upstream_checksum,
        sha256=sha256_file(target),
        license_name=spec.license_name,
        license_url=spec.license_url,
        profile=profile,
        status=status,
        license_review_required=spec.license_review_required,
        evidence=spec.license_evidence,
    )
```

Implement `safe_extract_zip` by resolving every member below the resolved destination, rejecting symlinks from `ZipInfo.external_attr`, extracting into a temporary sibling directory, then atomically renaming it:

```python
def safe_extract_zip(archive: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".extracting")
    temporary.mkdir(parents=True, exist_ok=False)
    root = temporary.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            output = (temporary / member.filename).resolve()
            if root not in output.parents and output != root:
                raise ValueError(f"unsafe ZIP path: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"ZIP symlink is not allowed: {member.filename}")
        bundle.extractall(temporary)
    temporary.replace(destination)
```

- [ ] **Step 4: Implement deterministic source-ledger writes**

```python
# src/data/manifests.py
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from .models import SourceRecord


def write_source_ledger(path: Path, records: Sequence[SourceRecord]) -> None:
    payload = [asdict(record) for record in sorted(records, key=lambda item: item.source_id)]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
```

Test sorting, final newline, atomic replacement, and round-trip JSON fields in `tests/data/test_manifests.py`.

- [ ] **Step 5: Run focused and full tests**

Run: `python -m pytest tests/data/test_acquisition.py tests/data/test_manifests.py -v && python -m pytest -q`

Expected: all tests pass without network access.

- [ ] **Step 6: Commit acquisition infrastructure**

```bash
git add src/data/acquisition.py src/data/manifests.py tests/data/test_acquisition.py tests/data/test_manifests.py
git commit -m "feat: add verified resumable data acquisition"
```

---

### Task 3: Topology Zoo fetch, inventory, filtering, and coordinate normalization

**Files:**
- Create: `src/data/topology_zoo.py`
- Create: `tests/data/fixtures/topology_zoo/eligible.graphml`
- Create: `tests/data/fixtures/topology_zoo/missing_coordinates.graphml`
- Create: `tests/data/test_topology_zoo.py`

**Interfaces:**
- Consumes: the pinned Git `ArtifactSpec` and manifest conventions.
- Produces: `fetch_pinned_repo(spec: ArtifactSpec, target: Path) -> SourceRecord`, `inventory_topology(path: Path) -> TopologyRecord`, and `normalize_topology(path: Path, box_size_m: float) -> dict[str, object]`.

- [ ] **Step 1: Write failing graph eligibility tests**

```python
# tests/data/test_topology_zoo.py
from pathlib import Path
import pytest

from data.topology_zoo import inventory_topology, normalize_topology

FIXTURES = Path("tests/data/fixtures/topology_zoo")


def test_inventory_accepts_connected_graph_with_coordinates():
    record = inventory_topology(FIXTURES / "eligible.graphml")
    assert record.eligible is True
    assert record.exclusion_reason == ""
    assert record.coordinate_count == record.node_count


def test_inventory_rejects_missing_coordinates():
    record = inventory_topology(FIXTURES / "missing_coordinates.graphml")
    assert record.eligible is False
    assert record.exclusion_reason == "missing_coordinates"


def test_normalization_uses_one_scale_for_both_axes():
    network = normalize_topology(FIXTURES / "eligible.graphml", 2000.0)
    xs = [node["x_m"] for node in network["nodes"]]
    ys = [node["y_m"] for node in network["nodes"]]
    assert max(max(xs) - min(xs), max(ys) - min(ys)) == pytest.approx(2000.0)
    assert network["normalization"]["scale_x"] == network["normalization"]["scale_y"]
```

The eligible fixture contains 15 nodes, complete `Latitude`/`Longitude` attributes, a connected path, and one parallel edge. The missing-coordinate fixture differs only by removing one longitude.

- [ ] **Step 2: Run the tests and verify the module is absent**

Run: `python -m pytest tests/data/test_topology_zoo.py -v`

Expected: collection fails because `data.topology_zoo` does not exist.

- [ ] **Step 3: Implement inventory and normalization**

```python
# src/data/topology_zoo.py
from dataclasses import dataclass
from pathlib import Path
import subprocess

import networkx as nx
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class TopologyRecord:
    topology_id: str
    node_count: int
    edge_count: int
    coordinate_count: int
    connected: bool
    has_parallel_edges: bool
    eligible: bool
    exclusion_reason: str


def _simple_graph(graph: nx.Graph) -> nx.Graph:
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    for left, right in graph.edges():
        if left != right:
            simple.add_edge(left, right)
    return simple


def inventory_topology(path: Path) -> TopologyRecord:
    raw = nx.read_graphml(path)
    graph = _simple_graph(raw)
    complete = sum(
        "Latitude" in attrs and "Longitude" in attrs
        for _, attrs in graph.nodes(data=True)
    )
    connected = graph.number_of_nodes() > 0 and nx.is_connected(graph)
    reason = ""
    if not 15 <= graph.number_of_nodes() <= 40:
        reason = "node_count_out_of_range"
    elif complete != graph.number_of_nodes():
        reason = "missing_coordinates"
    elif not connected:
        reason = "disconnected"
    return TopologyRecord(
        topology_id=path.stem,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        coordinate_count=complete,
        connected=connected,
        has_parallel_edges=raw.is_multigraph(),
        eligible=reason == "",
        exclusion_reason=reason,
    )
```

Implement `normalize_topology` with a local azimuthal-equidistant CRS centered on the mean latitude/longitude, then translate and uniformly scale projected coordinates. Sort nodes and edges by stable string ID before JSON serialization:

```python
def normalize_topology(path: Path, box_size_m: float) -> dict[str, object]:
    graph = _simple_graph(nx.read_graphml(path))
    record = inventory_topology(path)
    if not record.eligible:
        raise ValueError(f"ineligible topology {record.topology_id}: {record.exclusion_reason}")
    ordered = sorted(graph.nodes(data=True), key=lambda item: str(item[0]))
    latitudes = [float(attrs["Latitude"]) for _, attrs in ordered]
    longitudes = [float(attrs["Longitude"]) for _, attrs in ordered]
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={sum(latitudes)/len(latitudes)} "
        f"+lon_0={sum(longitudes)/len(longitudes)} +datum=WGS84 +units=m"
    )
    transformer = Transformer.from_crs("EPSG:4326", local, always_xy=True)
    projected = [transformer.transform(lon, lat) for lon, lat in zip(longitudes, latitudes, strict=True)]
    min_x, min_y = min(x for x, _ in projected), min(y for _, y in projected)
    span = max(max(x for x, _ in projected) - min_x, max(y for _, y in projected) - min_y)
    if span <= 0:
        raise ValueError(f"zero coordinate span: {record.topology_id}")
    scale = box_size_m / span
    nodes = [
        {"id": str(node_id), "x_m": (x - min_x) * scale, "y_m": (y - min_y) * scale}
        for (node_id, _), (x, y) in zip(ordered, projected, strict=True)
    ]
    edges = sorted({tuple(sorted((str(left), str(right)))) for left, right in graph.edges()})
    return {
        "network_id": record.topology_id,
        "nodes": nodes,
        "edges": [{"source": left, "target": right} for left, right in edges],
        "normalization": {"box_size_m": box_size_m, "scale_x": scale, "scale_y": scale},
    }
```

- [ ] **Step 4: Implement pinned Git acquisition**

`fetch_pinned_repo` initializes an empty target, fetches only the exact revision, and rejects drift:

```python
def fetch_pinned_repo(spec: ArtifactSpec, target: Path) -> SourceRecord:
    if not spec.revision or len(spec.revision) != 40:
        raise ValueError("a full 40-character Git revision is required")
    subprocess.run(["git", "init", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "remote", "add", "origin", spec.url], check=True)
    subprocess.run(
        ["git", "-C", str(target), "fetch", "--depth", "1", "origin", spec.revision],
        check=True,
    )
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if actual != spec.revision:
        raise ValueError(f"Git revision mismatch: expected {spec.revision}, got {actual}")
    return git_source_record(spec, target, actual)
```

Implement `git_source_record` with a deterministic SHA-256 over each sorted non-`.git` relative path, a NUL separator, and its bytes:

```python
def git_tree_sha256(target: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in target.rglob("*") if path.is_file() and ".git" not in path.parts)
    for path in paths:
        digest.update(path.relative_to(target).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
```

`git_source_record` fills the same `SourceRecord` fields as `_record`, using `actual` for `upstream_version`, the directory name for `filename`, `git_tree_sha256(target)` for `sha256`, and status `downloaded`. Tests create a local Git fixture, compare the recorded revision/tree hash, and prove a wrong revision fails.

- [ ] **Step 5: Run tests and lint the generated fixtures**

Run: `python -m pytest tests/data/test_topology_zoo.py -v && python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the Topology Zoo component**

```bash
git add src/data/topology_zoo.py tests/data/fixtures/topology_zoo tests/data/test_topology_zoo.py
git commit -m "feat: prepare Topology Zoo networks"
```

---

### Task 4: SNDlib XML archive parser and demand-preserving inventory

**Files:**
- Create: `src/data/sndlib.py`
- Create: `tests/data/fixtures/sndlib/mini.xml`
- Create: `tests/data/test_sndlib.py`

**Interfaces:**
- Consumes: the verified and safely extracted `sndlib-networks-xml.zip` artifact.
- Produces: `parse_sndlib_xml(path: Path) -> SndlibNetwork`, `inventory_sndlib(path: Path) -> SndlibRecord`, and `normalize_sndlib(network: SndlibNetwork) -> dict[str, object]`.

- [ ] **Step 1: Write failing parser tests with preserved units and demands**

```python
# tests/data/test_sndlib.py
from decimal import Decimal
from pathlib import Path

from data.sndlib import parse_sndlib_xml


def test_parser_preserves_ids_values_and_units():
    network = parse_sndlib_xml(Path("tests/data/fixtures/sndlib/mini.xml"))
    assert [node.node_id for node in network.nodes] == ["A", "B"]
    assert network.links[0].source == "A"
    assert network.links[0].target == "B"
    assert network.demands[0].source == "A"
    assert network.demands[0].target == "B"
    assert network.demands[0].value == Decimal("12.5")
    assert network.demand_unit == "Mbit/s"


def test_normalized_demand_is_marked_derived():
    network = parse_sndlib_xml(Path("tests/data/fixtures/sndlib/mini.xml"))
    payload = network.to_normalized_dict()
    assert payload["demands"][0]["provenance"] == "derived-from-sndlib-demand"
    assert payload["demands"][0]["original_value"] == "12.5"
```

The fixture follows the SNDlib XML namespace and contains two nodes, one link, one demand, coordinates, and explicit capacity/demand units.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/data/test_sndlib.py -v`

Expected: collection fails because `data.sndlib` does not exist.

- [ ] **Step 3: Implement namespace-tolerant XML parsing**

```python
# src/data/sndlib.py
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import xml.etree.ElementTree as ET


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str):
    return [child for child in element.iter() if _local(child.tag) == name]


@dataclass(frozen=True)
class SndlibNode:
    node_id: str
    x: Decimal | None
    y: Decimal | None


@dataclass(frozen=True)
class SndlibLink:
    link_id: str
    source: str
    target: str


@dataclass(frozen=True)
class SndlibDemand:
    demand_id: str
    source: str
    target: str
    value: Decimal


@dataclass(frozen=True)
class SndlibNetwork:
    network_id: str
    nodes: tuple[SndlibNode, ...]
    links: tuple[SndlibLink, ...]
    demands: tuple[SndlibDemand, ...]
    demand_unit: str | None
    capacity_unit: str | None

    def to_normalized_dict(self) -> dict[str, object]:
        return {
            "network_id": self.network_id,
            "demand_unit": self.demand_unit,
            "capacity_unit": self.capacity_unit,
            "demands": [
                {
                    "id": demand.demand_id,
                    "source": demand.source,
                    "target": demand.target,
                    "original_value": str(demand.value),
                    "original_unit": self.demand_unit,
                    "provenance": "derived-from-sndlib-demand",
                }
                for demand in self.demands
            ],
        }
```

Complete `parse_sndlib_xml` using local tag names, `Decimal` for source values, reference validation for every link/demand endpoint, and stable ID sorting. The parser uses these helpers rather than positional XML assumptions:

```python
def _text(element: ET.Element, name: str, path: Path) -> str:
    match = next((item for item in element.iter() if _local(item.tag) == name), None)
    if match is None or match.text is None or not match.text.strip():
        raise ValueError(f"{path}: missing {name}")
    return match.text.strip()


def parse_sndlib_xml(path: Path) -> SndlibNetwork:
    root = ET.parse(path).getroot()
    node_elements = _children(root, "node")
    nodes = tuple(
        sorted(
            (
                SndlibNode(
                    node_id=str(item.attrib["id"]),
                    x=Decimal(_text(item, "x", path)) if _children(item, "x") else None,
                    y=Decimal(_text(item, "y", path)) if _children(item, "y") else None,
                )
                for item in node_elements
            ),
            key=lambda item: item.node_id,
        )
    )
    links = tuple(sorted((_parse_link(item, path) for item in _children(root, "link")), key=lambda item: item.link_id))
    demands = tuple(sorted((_parse_demand(item, path) for item in _children(root, "demand")), key=lambda item: item.demand_id))
    node_ids = {node.node_id for node in nodes}
    for item in (*links, *demands):
        if item.source not in node_ids or item.target not in node_ids:
            raise ValueError(f"{path}: dangling endpoint in {item}")
    return SndlibNetwork(
        network_id=path.stem,
        nodes=nodes,
        links=links,
        demands=demands,
        demand_unit=_optional_text(root, "demandValue", "unit"),
        capacity_unit=_optional_text(root, "capacity", "unit"),
    )
```

Implement `_parse_link`, `_parse_demand`, and `_optional_text` with `_text`:

```python
def _parse_link(element: ET.Element, path: Path) -> SndlibLink:
    return SndlibLink(
        link_id=str(element.attrib["id"]),
        source=_text(element, "source", path),
        target=_text(element, "target", path),
    )


def _parse_demand(element: ET.Element, path: Path) -> SndlibDemand:
    return SndlibDemand(
        demand_id=str(element.attrib["id"]),
        source=_text(element, "source", path),
        target=_text(element, "target", path),
        value=Decimal(_text(element, "demandValue", path)),
    )


def _optional_text(root: ET.Element, element_name: str, attribute: str) -> str | None:
    match = next((item for item in root.iter() if _local(item.tag) == element_name), None)
    return None if match is None else match.attrib.get(attribute)
```

Raise `ValueError` with the filename and missing element/reference. Do not interpret capacity modules as operational capacity.

- [ ] **Step 4: Add archive-level inventory behavior**

`inventory_sndlib` catches only `ET.ParseError`, `KeyError`, `InvalidOperation`, and `ValueError`, then returns network ID, node/link/demand counts, coordinate completeness, source units, parser status, and the exact exception string. Add the following assertions to the test file:

```python
def test_inventory_records_dangling_endpoint(tmp_path):
    path = write_sndlib_fixture(tmp_path, demand_target="missing")
    record = inventory_sndlib(path)
    assert record.parser_status == "invalid"
    assert "dangling endpoint" in record.error


def test_inventory_records_missing_demand_value(tmp_path):
    path = write_sndlib_fixture(tmp_path, include_demand_value=False)
    record = inventory_sndlib(path)
    assert record.parser_status == "invalid"
    assert "missing demandValue" in record.error
```

- [ ] **Step 5: Run focused and full tests**

Run: `python -m pytest tests/data/test_sndlib.py -v && python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the SNDlib component**

```bash
git add src/data/sndlib.py tests/data/fixtures/sndlib tests/data/test_sndlib.py
git commit -m "feat: parse and inventory SNDlib networks"
```

---

### Task 5: RescueNet mask inventory and deterministic 30-pair selection

**Files:**
- Create: `src/data/rescuenet.py`
- Create: `tests/data/fixtures/rescuenet/images/`
- Create: `tests/data/fixtures/rescuenet/masks/`
- Create: `tests/data/test_rescuenet.py`

**Interfaces:**
- Consumes: the verified RescueNet descriptor and safely extracted validation archive.
- Produces: `load_label_map(path: Path) -> dict[int, str]`, `pair_images_and_masks(root: Path) -> tuple[RescuePair, ...]`, `count_mask_classes(pair: RescuePair, label_map: Mapping[int, str]) -> Mapping[int, int]`, and `select_pairs(records: Sequence[RescueRecord], count: int, seed: int) -> tuple[RescueRecord, ...]`.

- [ ] **Step 1: Write failing pairing and selection tests**

```python
# tests/data/test_rescuenet.py
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from data.rescuenet import RescueRecord, pair_images_and_masks, select_pairs


def test_pairs_images_and_masks_by_canonical_stem():
    pairs = pair_images_and_masks(Path("tests/data/fixtures/rescuenet"))
    assert [(pair.pair_id, pair.image.name, pair.mask.name) for pair in pairs] == [
        ("scene-001", "scene-001.jpg", "scene-001.png"),
        ("scene-002", "scene-002.jpg", "scene-002.png"),
    ]


def test_selection_is_deterministic_and_group_balanced(rescue_records):
    first = select_pairs(rescue_records, count=30, seed=20260910)
    second = select_pairs(tuple(reversed(rescue_records)), count=30, seed=20260910)
    assert [item.pair_id for item in first] == [item.pair_id for item in second]
    assert len({item.presence_vector for item in first[:4]}) == 4


def test_selection_refuses_to_duplicate_pairs(rescue_records):
    with pytest.raises(ValueError, match="need 30 valid pairs, found 29"):
        select_pairs(rescue_records[:29], count=30, seed=20260910)
```

Create small RGB images and indexed PNG masks as committed binary fixtures with this test helper; the `rescue_records` fixture contains 32 records across four damage-presence vectors:

```python
@pytest.fixture
def rescue_records(tmp_path):
    records = []
    vectors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0))
    for index in range(32):
        vector = vectors[index % len(vectors)]
        image = tmp_path / f"scene-{index:03d}.jpg"
        mask = tmp_path / f"scene-{index:03d}.png"
        Image.new("RGB", (2, 2), "white").save(image)
        Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8)).save(mask)
        records.append(
            RescueRecord(
                pair_id=f"scene-{index:03d}",
                image=image,
                mask=mask,
                class_counts=((0, 1), (1, 1), (2, 1), (3, 1)),
                presence_vector=vector,
            )
        )
    return tuple(records)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/data/test_rescuenet.py -v`

Expected: collection fails because `data.rescuenet` does not exist.

- [ ] **Step 3: Implement strict pairing and class counting**

```python
# src/data/rescuenet.py
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import hashlib
import random

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RescuePair:
    pair_id: str
    image: Path
    mask: Path


@dataclass(frozen=True)
class RescueRecord:
    pair_id: str
    image: Path
    mask: Path
    class_counts: tuple[tuple[int, int], ...]
    presence_vector: tuple[int, ...]


def count_mask_classes(pair: RescuePair, label_map: Mapping[int, str]) -> Mapping[int, int]:
    values, counts = np.unique(np.asarray(Image.open(pair.mask)), return_counts=True)
    unknown = sorted(set(map(int, values)) - set(label_map))
    if unknown:
        raise ValueError(f"unknown mask labels for {pair.pair_id}: {unknown}")
    return {int(value): int(count) for value, count in zip(values, counts, strict=True)}
```

Implement canonical pairing with case-insensitive image suffixes, duplicate-stem rejection, missing-partner rejection, sorted output, and descriptor parsing:

```python
def pair_images_and_masks(root: Path) -> tuple[RescuePair, ...]:
    images = _index_by_stem(root / "images", {".jpg", ".jpeg", ".png"})
    masks = _index_by_stem(root / "masks", {".png"})
    if images.keys() != masks.keys():
        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())
        raise ValueError(f"image-mask mismatch: masks={missing_masks}, images={missing_images}")
    return tuple(RescuePair(key, images[key], masks[key]) for key in sorted(images))


def _index_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            key = path.stem.casefold()
            if key in result:
                raise ValueError(f"duplicate canonical stem: {key}")
            result[key] = path
    return result
```

`load_label_map` parses each nonempty descriptor line as `<integer-id><whitespace><label>`, rejects duplicate IDs, and requires at least one damage label. Treat only official damage classes as presence-vector dimensions; background and unlabeled pixels remain in counts but do not define strata.

- [ ] **Step 4: Implement deterministic group-balanced selection**

Sort input records by `pair_id`, group by `presence_vector`, shuffle each group using a stable derived seed, and select round-robin across sorted group keys:

```python
def select_pairs(records: Sequence[RescueRecord], count: int, seed: int) -> tuple[RescueRecord, ...]:
    if len(records) < count:
        raise ValueError(f"need {count} valid pairs, found {len(records)}")
    groups: dict[tuple[int, ...], list[RescueRecord]] = {}
    for record in sorted(records, key=lambda item: item.pair_id):
        groups.setdefault(record.presence_vector, []).append(record)
    for key, values in groups.items():
        digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(values)
    selected: list[RescueRecord] = []
    while len(selected) < count:
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
    return tuple(selected)
```

The CSV writer enumerates this tuple from rank 1 and stores pair ID, relative image/mask paths, class counts, presence vector, seed, rank, archive file ID, and raw archive SHA-256.

- [ ] **Step 5: Run focused and full tests**

Run: `python -m pytest tests/data/test_rescuenet.py -v && python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the RescueNet component**

```bash
git add src/data/rescuenet.py tests/data/fixtures/rescuenet tests/data/test_rescuenet.py
git commit -m "feat: select reproducible RescueNet case studies"
```

---

### Task 6: Deterministic scenario generation and provenance hashes

**Files:**
- Create: `src/data/scenarios.py`
- Create: `tests/data/test_scenarios.py`

**Interfaces:**
- Consumes: normalized network dictionaries, SNDlib demand records, RescueNet target points, `PipelineConfig`, raw SHA-256 values, and a scenario seed.
- Produces: `named_rng(seed: int, concern: str) -> numpy.random.Generator`, `generate_scenario(inputs: ScenarioInputs) -> dict[str, object]`, and `canonical_hash(payload: Mapping[str, object]) -> str`.

- [ ] **Step 1: Write failing reproducibility and invariant tests**

```python
# tests/data/test_scenarios.py
from data.scenarios import ScenarioInputs, canonical_hash, generate_scenario


def test_same_seed_produces_identical_scenario(scenario_inputs):
    first = generate_scenario(scenario_inputs)
    second = generate_scenario(scenario_inputs)
    assert first == second
    assert canonical_hash(first) == canonical_hash(second)


def test_different_seed_changes_stochastic_fields(scenario_inputs):
    first = generate_scenario(scenario_inputs)
    second = generate_scenario(scenario_inputs.with_seed(20260911))
    assert first["failures"] != second["failures"]
    assert first["provenance"]["raw_sha256"] == second["provenance"]["raw_sha256"]


def test_alerts_obey_release_deadline_and_positive_size(scenario_inputs):
    scenario = generate_scenario(scenario_inputs)
    assert all(0 <= alert["release_slot"] < alert["deadline_slot"] for alert in scenario["alerts"])
    assert all(alert["size_bits"] > 0 for alert in scenario["alerts"])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/data/test_scenarios.py -v`

Expected: collection fails because `data.scenarios` does not exist.

- [ ] **Step 3: Implement named random streams and canonical hashing**

```python
# src/data/scenarios.py
from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping

import numpy as np


def named_rng(seed: int, concern: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{concern}".encode()).digest()
    child_seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(child_seed)


def canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ScenarioInputs:
    network: Mapping[str, object]
    demands: tuple[Mapping[str, object], ...]
    targets: tuple[Mapping[str, object], ...]
    raw_sha256: tuple[str, ...]
    config_hash: str
    seed: int
    alert_count: int
    failure_probability: float

    def with_seed(self, seed: int) -> "ScenarioInputs":
        return replace(self, seed=seed)
```

- [ ] **Step 4: Implement scenario fields and validation**

Use separate streams named `rescue-center`, `relay-set`, `workload`, `deadlines`, and `failures`. The core implementation is:

```python
def generate_scenario(inputs: ScenarioInputs) -> dict[str, object]:
    if not 0.0 <= inputs.failure_probability <= 1.0:
        raise ValueError("failure_probability must be in [0, 1]")
    node_ids = sorted(str(node["id"]) for node in inputs.network["nodes"])
    if len(node_ids) < 2 or inputs.alert_count <= 0 or not inputs.demands:
        raise ValueError("scenario requires nodes, positive alert_count, and demands")
    center_rng = named_rng(inputs.seed, "rescue-center")
    center = node_ids[int(center_rng.integers(0, len(node_ids)))]
    failure_rng = named_rng(inputs.seed, "failures")
    failures = [
        {**edge, "failed": bool(failure_rng.random() < inputs.failure_probability)}
        for edge in inputs.network["edges"]
    ]
    workload_rng = named_rng(inputs.seed, "workload")
    deadline_rng = named_rng(inputs.seed, "deadlines")
    alerts = []
    for index in range(inputs.alert_count):
        demand = inputs.demands[index % len(inputs.demands)]
        release = int(workload_rng.integers(0, 20))
        deadline = release + int(deadline_rng.integers(2, 11))
        weight = float(demand["original_value"])
        alerts.append({
            "id": f"alert-{index:04d}",
            "source": node_ids[index % len(node_ids)],
            "release_slot": release,
            "deadline_slot": deadline,
            "size_bits": max(1, int(round(weight * 1_000_000))),
            "provenance": "derived-from-sndlib-demand",
        })
    return {
        "schema_version": 1,
        "network": inputs.network,
        "rescue_center": center,
        "alerts": alerts,
        "failures": failures,
        "targets": list(inputs.targets),
        "provenance": {
            "raw_sha256": sorted(inputs.raw_sha256),
            "config_hash": inputs.config_hash,
            "seed": inputs.seed,
            "generator_version": 1,
        },
    }
```

Add relay selection using the separate `relay-set` stream and validate every returned node reference. Preserve the normalized topology and provenance fields exactly as shown.

- [ ] **Step 5: Run focused and full tests**

Run: `python -m pytest tests/data/test_scenarios.py -v && python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit scenario generation**

```bash
git add src/data/scenarios.py tests/data/test_scenarios.py
git commit -m "feat: generate reproducible UAV relay scenarios"
```

---

### Task 7: Command-line stages, aggregate workflow, and offline integration test

**Files:**
- Create: `src/data/cli.py`
- Create: `scripts/data/download.py`
- Create: `scripts/data/verify.py`
- Create: `scripts/data/inventory.py`
- Create: `scripts/data/preprocess.py`
- Create: `tests/data/test_cli.py`
- Create: `tests/data/test_pipeline_integration.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: every public interface from Tasks 1–6.
- Produces: `python -m data.cli {download,verify,inventory,preprocess,all} --profile PATH [--data-root PATH] [--dry-run]` plus matching thin scripts.

- [ ] **Step 1: Write failing CLI and offline workflow tests**

```python
# tests/data/test_cli.py
from data.cli import build_parser


def test_cli_exposes_all_lifecycle_stages():
    parser = build_parser()
    for command in ("download", "verify", "inventory", "preprocess", "all"):
        args = parser.parse_args([command, "--profile", "configs/data/paper.yaml", "--dry-run"])
        assert args.command == command
        assert args.dry_run is True
```

```python
# tests/data/test_pipeline_integration.py
def test_fixture_pipeline_writes_complete_provenance(fixture_pipeline):
    result = fixture_pipeline.run_all()
    assert result.source_ledger.exists()
    assert result.topology_inventory.exists()
    assert result.sndlib_inventory.exists()
    assert result.rescuenet_selection.exists()
    assert len(result.scenarios) >= 1
    assert all(item["provenance"]["raw_sha256"] for item in result.scenarios)
```

The fixture pipeline injects `httpx.MockTransport`, a local Git repository, and small source fixtures; it never opens an internet connection.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/data/test_cli.py tests/data/test_pipeline_integration.py -v`

Expected: collection fails because `data.cli` does not exist.

- [ ] **Step 3: Implement the command parser and orchestration**

```python
# src/data/cli.py
import argparse
from pathlib import Path

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "verify", "inventory", "preprocess", "all"):
        command = subparsers.add_parser(name)
        command.add_argument("--profile", type=Path, required=True)
        command.add_argument("--data-root", type=Path)
        command.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.profile)
    data_root = args.data_root or config.data_root
    return run_stage(args.command, config, data_root, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
```

Implement `run_stage` so `all` calls download, verify, inventory, and preprocess in order and stops on the first error:

```python
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
```

Define `PipelineContext` as a frozen dataclass and each handler as `Callable[[PipelineContext], None]`. `run_download` dispatches `http` specs to `download_http` and `git` specs to `fetch_pinned_repo`; the other handlers call the source-specific functions and write deterministic CSV/JSON. Use Python logging with concise console output and a JSON-lines log under `data/manifests/` for real runs.

- [ ] **Step 4: Add thin script entry points**

Each file under `scripts/data/` imports `data.cli.main` and supplies its fixed stage, for example:

```python
# scripts/data/verify.py
import sys
from data.cli import main

raise SystemExit(main(["verify", *sys.argv[1:]]))
```

- [ ] **Step 5: Document exact workflows**

Add a README section with these commands and their expected outputs:

```bash
python -m pip install -e '.[test]'
python -m data.cli all --profile configs/data/pilot.yaml --dry-run
python -m data.cli all --profile configs/data/pilot.yaml
python -m data.cli all --profile configs/data/paper.yaml
python -m data.cli verify --profile configs/data/paper.yaml
python -m pytest -q
```

Document that the paper command transfers the 2.37 GB RescueNet validation archive, raw data are ignored, failures never trigger mirror fallback, and `full-rescuenet.yaml` adds roughly 21 GB not required by the paper.

- [ ] **Step 6: Run integration and full tests**

Run: `python -m pytest tests/data/test_cli.py tests/data/test_pipeline_integration.py -v && python -m pytest -q`

Expected: all tests pass offline.

- [ ] **Step 7: Commit the runnable workflow**

```bash
git add src/data/cli.py scripts/data tests/data/test_cli.py tests/data/test_pipeline_integration.py README.md
git commit -m "feat: add reproducible data preparation workflow"
```

---

### Task 8: Live-source acquisition, paper manifests, and final verification

**Files:**
- Create: `src/data/live_sources.py`
- Create: `data/manifests/sources.json`
- Create: `data/manifests/topology_inventory.csv`
- Create: `data/manifests/sndlib_inventory.csv`
- Create: `data/manifests/rescuenet_inventory.csv`
- Create: `data/manifests/rescuenet_selection.csv`
- Create: `tests/data/test_live_sources.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: the completed pipeline and official live sources.
- Produces: the actual ignored raw dataset, checked-in paper manifests/inventories, normalized ignored artifacts, and evidence that the clean workflow works against current primary sources.

- [ ] **Step 1: Add opt-in metadata drift tests**

```python
# tests/data/test_live_sources.py
import os
import pytest

from data.config import load_config
from data.live_sources import inspect_remote_metadata


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_DATA_TESTS") != "1",
    reason="set RUN_LIVE_DATA_TESTS=1 to contact official dataset sources",
)


def test_official_source_metadata_matches_paper_profile():
    config = load_config(Path("configs/data/paper.yaml"))
    remote = inspect_remote_metadata(config)
    assert remote["rescuenet-validation"].file_id == 40582916
    assert remote["rescuenet-validation"].expected_size == 2373908074
    assert remote["topology-zoo"].revision == "e278b1bdaafea5dac33883bf9c97401db4cd7347"
```

Add `from pathlib import Path` to the test imports. Implement the metadata type and read-only probes as follows:

```python
# src/data/live_sources.py
from dataclasses import dataclass
import subprocess

import httpx

from .models import PipelineConfig


@dataclass(frozen=True)
class RemoteMetadata:
    source_id: str
    file_id: int | None
    expected_size: int | None
    revision: str | None


def inspect_remote_metadata(config: PipelineConfig) -> dict[str, RemoteMetadata]:
    result: dict[str, RemoteMetadata] = {}
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        for spec in config.artifacts:
            if spec.kind == "git":
                output = subprocess.check_output(["git", "ls-remote", spec.url, "HEAD"], text=True)
                result[spec.source_id] = RemoteMetadata(spec.source_id, None, None, output.split()[0])
            elif "figshare" in spec.url or "ndownloader" in spec.url:
                article_id = int(spec.upstream_version.split(":", 1)[1])
                article = client.get(f"https://api.figshare.com/v2/articles/{article_id}").raise_for_status().json()
                file_info = next(item for item in article["files"] if str(item["id"]) in spec.url)
                result[spec.source_id] = RemoteMetadata(
                    spec.source_id, int(file_info["id"]), int(file_info["size"]), None
                )
            else:
                response = client.head(spec.url)
                response.raise_for_status()
                size = int(response.headers["content-length"]) if "content-length" in response.headers else None
                result[spec.source_id] = RemoteMetadata(spec.source_id, None, size, None)
    return result
```

- [ ] **Step 2: Confirm dry-run scope and free space**

Run: `python -m data.cli all --profile configs/data/paper.yaml --dry-run && df -h data`

Expected: the dry-run lists Topology Zoo, SNDlib XML networks, RescueNet descriptor and validation; available space exceeds the declared transfer plus twice the largest archive for safe extraction.

- [ ] **Step 3: Run live metadata tests**

Run: `RUN_LIVE_DATA_TESTS=1 python -m pytest tests/data/test_live_sources.py -v`

Expected: all official identifiers, revisions, sizes, and checksums match the pinned paper profile. If upstream drift is reported, inspect the authoritative metadata, update the profile and spec with the new immutable identifier, rerun offline tests, and commit that evidence before downloading.

- [ ] **Step 4: Acquire and process the paper profile**

Run: `python -m data.cli all --profile configs/data/paper.yaml`

Expected: Topology Zoo checks out the pinned commit; SNDlib and RescueNet pass checksums; inventories are written; exactly 30 RescueNet pairs are selected; processed networks and scenarios validate successfully.

- [ ] **Step 5: Verify idempotence and integrity**

Run: `python -m data.cli all --profile configs/data/paper.yaml && python -m data.cli verify --profile configs/data/paper.yaml`

Expected: the second run reports verified reuse rather than retransferring artifacts, and verification reports no size, checksum, pairing, schema, or provenance errors.

- [ ] **Step 6: Run the full offline suite and inspect Git scope**

Run: `python -m pytest -q && git status --short && git check-ignore data/raw/rescuenet/segmentation-validationset.zip`

Expected: all tests pass; only source code, configuration, documentation, and manifest/inventory files appear as changes; `git check-ignore` prints the RescueNet raw path.

- [ ] **Step 7: Validate manifests programmatically**

Run:

```bash
python -m data.cli verify --profile configs/data/paper.yaml
python -c 'import csv; rows=list(csv.DictReader(open("data/manifests/rescuenet_selection.csv"))); assert len(rows)==30; assert len({r["pair_id"] for r in rows})==30'
python -c 'import json; rows=json.load(open("data/manifests/sources.json")); assert all(r["sha256"] and r["status"] in {"downloaded", "reused"} for r in rows)'
```

Expected: every command exits with status 0.

- [ ] **Step 8: Commit reproducibility evidence without raw data**

```bash
git add src/data/live_sources.py tests/data/test_live_sources.py data/manifests README.md configs/data
git commit -m "data: record reproducible paper dataset manifests"
```

- [ ] **Step 9: Record final verification evidence**

Run: `git status --short --branch && git log -8 --oneline && python -m pytest -q`

Expected: the branch is clean, the eight task commits are visible after the design/plan commits, and the entire offline test suite passes.
