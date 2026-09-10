from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import zipfile

import httpx

from .models import ArtifactSpec, SourceRecord


class ChecksumMismatch(ValueError):
    """Raised when an artifact does not match its declared integrity data."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, spec: ArtifactSpec) -> None:
    if spec.expected_size is not None and path.stat().st_size != spec.expected_size:
        raise ChecksumMismatch(f"size mismatch for {spec.source_id}")
    if spec.upstream_checksum:
        algorithm, expected = spec.upstream_checksum.split(":", 1)
        actual = _digest_file(path, algorithm)
        if actual != expected:
            raise ChecksumMismatch(f"{algorithm} mismatch for {spec.source_id}")


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


def download_http(
    spec: ArtifactSpec, root: Path, profile: str, client: httpx.Client
) -> SourceRecord:
    if not spec.filename:
        raise ValueError(f"HTTP artifact {spec.source_id} needs filename")
    directory = root / spec.source_id
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec.filename
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


def safe_extract_zip(archive: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".extracting")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    root = temporary.resolve()
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                output = (temporary / member.filename).resolve()
                if root not in output.parents and output != root:
                    raise ValueError(f"unsafe ZIP path: {member.filename}")
                mode = (member.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ValueError(f"ZIP symlink is not allowed: {member.filename}")
            bundle.extractall(temporary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
