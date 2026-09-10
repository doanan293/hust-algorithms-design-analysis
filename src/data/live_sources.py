from dataclasses import dataclass
import json
import re
import subprocess
from urllib.request import Request, urlopen

from .models import PipelineConfig


@dataclass(frozen=True)
class RemoteMetadata:
    source_id: str
    file_id: int | None
    expected_size: int | None
    revision: str | None


def _file_id(url: str) -> int | None:
    match = re.search(r"/files/(\d+)(?:$|[/?#])", url)
    return int(match.group(1)) if match else None


def _article_id(version: str | None) -> int | None:
    if not version or not version.startswith("article:"):
        return None
    return int(version.split(":", 1)[1])


def inspect_remote_metadata(config: PipelineConfig) -> dict[str, RemoteMetadata]:
    result: dict[str, RemoteMetadata] = {}
    for spec in config.artifacts:
        if spec.kind == "git":
            output = subprocess.check_output(
                ["git", "ls-remote", spec.url, "HEAD"], text=True
            )
            result[spec.source_id] = RemoteMetadata(
                source_id=spec.source_id,
                file_id=None,
                expected_size=None,
                revision=output.split()[0],
            )
            continue
        file_id = _file_id(spec.url)
        article_id = _article_id(spec.upstream_version)
        if article_id is not None:
            request = Request(
                f"https://api.figshare.com/v2/articles/{article_id}",
                headers={"Accept": "application/json", "User-Agent": "curl/8.0"},
            )
            with urlopen(request, timeout=30) as response:
                files = json.load(response).get("files", [])
            file_info = next((item for item in files if item["id"] == file_id), None)
            if file_info is None:
                raise ValueError(f"file {file_id} missing from Figshare article {article_id}")
            result[spec.source_id] = RemoteMetadata(
                source_id=spec.source_id,
                file_id=file_id,
                expected_size=int(file_info["size"]),
                revision=None,
            )
        else:
            request = Request(spec.url, method="HEAD")
            with urlopen(request, timeout=30) as response:
                size = response.headers.get("content-length")
            result[spec.source_id] = RemoteMetadata(
                source_id=spec.source_id,
                file_id=None,
                expected_size=int(size) if size else None,
                revision=None,
            )
    return result
