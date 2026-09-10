import json
from pathlib import Path

from data.manifests import write_source_ledger
from data.models import SourceRecord


def record(source_id: str) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        dataset_family="fixture",
        landing_page="https://example.test",
        download_url="https://example.test/file",
        retrieved_at_utc="2026-09-10T00:00:00+00:00",
        upstream_version=None,
        filename="file",
        byte_size=1,
        upstream_checksum=None,
        sha256="a" * 64,
        license_name=None,
        license_url=None,
        profile="paper",
        status="downloaded",
        license_review_required=False,
    )


def test_source_ledger_is_sorted_and_round_trips(tmp_path: Path):
    path = tmp_path / "manifests" / "sources.json"
    write_source_ledger(path, [record("z"), record("a")])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [item["source_id"] for item in payload] == ["a", "z"]
    assert path.read_text(encoding="utf-8").endswith("\n")
