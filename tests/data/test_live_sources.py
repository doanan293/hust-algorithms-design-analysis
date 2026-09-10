import os
from pathlib import Path

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
