from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import subprocess

import networkx as nx
from pyproj import CRS, Transformer

from .models import ArtifactSpec, SourceRecord


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


def _coordinate(attrs: dict[str, object], name: str) -> float | None:
    value = attrs.get(name)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def inventory_topology(path: Path) -> TopologyRecord:
    raw = nx.read_graphml(path)
    graph = _simple_graph(raw)
    coordinate_count = sum(
        ((_coordinate(attrs, "Latitude") is not None and _coordinate(attrs, "Longitude") is not None)
         or (_coordinate(attrs, "x") is not None and _coordinate(attrs, "y") is not None))
        for _, attrs in graph.nodes(data=True)
    )
    connected = graph.number_of_nodes() > 0 and nx.is_connected(graph)
    reason = ""
    if not 15 <= graph.number_of_nodes() <= 40:
        reason = "node_count_out_of_range"
    elif coordinate_count != graph.number_of_nodes():
        reason = "missing_coordinates"
    elif not connected:
        reason = "disconnected"
    return TopologyRecord(
        topology_id=path.stem,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        coordinate_count=coordinate_count,
        connected=connected,
        has_parallel_edges=raw.is_multigraph(),
        eligible=reason == "",
        exclusion_reason=reason,
    )


def normalize_topology(path: Path, box_size_m: float) -> dict[str, object]:
    record = inventory_topology(path)
    if not record.eligible:
        raise ValueError(f"ineligible topology {record.topology_id}: {record.exclusion_reason}")
    graph = _simple_graph(nx.read_graphml(path))
    ordered = sorted(graph.nodes(data=True), key=lambda item: str(item[0]))
    geographic = all(_coordinate(attrs, "Latitude") is not None and _coordinate(attrs, "Longitude") is not None for _, attrs in ordered)
    if geographic:
        latitudes = [float(attrs["Latitude"]) for _, attrs in ordered]
        longitudes = [float(attrs["Longitude"]) for _, attrs in ordered]
        local = CRS.from_proj4(f"+proj=aeqd +lat_0={sum(latitudes)/len(latitudes)} +lon_0={sum(longitudes)/len(longitudes)} +datum=WGS84 +units=m")
        transformer = Transformer.from_crs("EPSG:4326", local, always_xy=True)
        projected = [transformer.transform(lon, lat) for lon, lat in zip(longitudes, latitudes, strict=True)]
    else:
        projected = [(float(attrs["x"]), float(attrs["y"])) for _, attrs in ordered]
    min_x = min(x for x, _ in projected)
    min_y = min(y for _, y in projected)
    span = max(
        max(x for x, _ in projected) - min_x,
        max(y for _, y in projected) - min_y,
    )
    if span <= 0:
        raise ValueError(f"zero coordinate span: {record.topology_id}")
    scale = box_size_m / span
    nodes = [
        {
            "id": str(node_id),
            "x_m": (x - min_x) * scale,
            "y_m": (y - min_y) * scale,
        }
        for (node_id, _), (x, y) in zip(ordered, projected, strict=True)
    ]
    edges = sorted(
        {tuple(sorted((str(left), str(right)))) for left, right in graph.edges()}
    )
    return {
        "network_id": record.topology_id,
        "nodes": nodes,
        "edges": [{"source": left, "target": right} for left, right in edges],
        "normalization": {
            "box_size_m": box_size_m,
            "scale_x": scale,
            "scale_y": scale,
        },
    }


def git_tree_sha256(target: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in target.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    for path in paths:
        digest.update(path.relative_to(target).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fetch_pinned_repo(
    spec: ArtifactSpec, target: Path, profile: str = "paper"
) -> SourceRecord:
    if not spec.revision or len(spec.revision) != 40:
        raise ValueError("a full 40-character Git revision is required")
    current = None
    if (target / ".git").exists():
        try:
            current = subprocess.check_output(
                ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            current = None
    if current != spec.revision:
        if target.exists():
            shutil.rmtree(target)
        subprocess.run(["git", "init", str(target)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(target), "remote", "add", "origin", spec.url],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--depth", "1", "origin", spec.revision],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
            capture_output=True,
        )
    actual = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != spec.revision:
        raise ValueError(f"Git revision mismatch: expected {spec.revision}, got {actual}")
    return SourceRecord(
        source_id=spec.source_id,
        dataset_family=spec.dataset_family,
        landing_page=spec.landing_page or spec.url,
        download_url=spec.url,
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        upstream_version=actual,
        filename=target.name,
        byte_size=sum(path.stat().st_size for path in target.rglob("*") if path.is_file()),
        upstream_checksum=None,
        sha256=git_tree_sha256(target),
        license_name=spec.license_name,
        license_url=spec.license_url,
        profile=profile,
        status="downloaded",
        license_review_required=spec.license_review_required,
        evidence=spec.license_evidence,
    )
