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
        "  - {source_id: x, dataset_family: x, kind: http, url: https://example.test/a}\n"
        "  - {source_id: x, dataset_family: x, kind: http, url: https://example.test/b}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate source_id: x"):
        load_config(path)
