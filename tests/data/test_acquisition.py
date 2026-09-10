from pathlib import Path
import hashlib
import io
import zipfile

import httpx
import pytest

from data.acquisition import ChecksumMismatch, download_http, safe_extract_zip
from data.models import ArtifactSpec


def spec_for(payload: bytes) -> ArtifactSpec:
    return ArtifactSpec(
        source_id="fixture",
        dataset_family="fixture",
        kind="http",
        url="https://example.test/file.zip",
        filename="file.zip",
        expected_size=len(payload),
        upstream_checksum="md5:" + hashlib.md5(payload).hexdigest(),
        landing_page="https://example.test",
    )


def test_download_publishes_verified_file(tmp_path: Path):
    payload = b"verified payload"
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
    )
    record = download_http(spec_for(payload), tmp_path, "paper", client)
    assert (tmp_path / "fixture" / "file.zip").read_bytes() == payload
    assert not (tmp_path / "fixture" / "file.zip.partial").exists()
    assert record.sha256 == hashlib.sha256(payload).hexdigest()


def test_invalid_existing_file_is_quarantined(tmp_path: Path):
    target = tmp_path / "fixture" / "file.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt")
    payload = b"verified payload"
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
    )
    download_http(spec_for(payload), tmp_path, "paper", client)
    assert list(target.parent.glob("file.zip.corrupt-*"))


def test_checksum_mismatch_keeps_partial_file(tmp_path: Path):
    expected = spec_for(b"expected")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"wrongxx"))
    )
    with pytest.raises(ChecksumMismatch):
        download_http(expected, tmp_path, "paper", client)
    assert (tmp_path / "fixture" / "file.zip.partial").exists()


def test_download_resumes_partial_response(tmp_path: Path):
    payload = b"0123456789"
    partial = tmp_path / "fixture" / "file.zip.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:4])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=4-"
        return httpx.Response(206, headers={"Content-Range": "bytes 4-9/10"}, content=payload[4:])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    record = download_http(spec_for(payload), tmp_path, "paper", client)
    assert (tmp_path / "fixture" / "file.zip").read_bytes() == payload
    assert record.byte_size == len(payload)


def test_safe_extract_rejects_parent_traversal(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="unsafe ZIP path"):
        safe_extract_zip(archive, tmp_path / "out")


def test_safe_extract_accepts_nested_file(tmp_path: Path):
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/file.txt", "yes")
    destination = tmp_path / "out"
    safe_extract_zip(archive, destination)
    assert (destination / "nested/file.txt").read_text(encoding="utf-8") == "yes"
