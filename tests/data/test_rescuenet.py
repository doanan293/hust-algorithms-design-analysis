from pathlib import Path
import hashlib

import numpy as np
from PIL import Image
import pytest

from data.rescuenet import RescuePair, RescueRecord, count_mask_classes, load_label_map, pair_images_and_masks, select_pairs


FIXTURE = Path("tests/data/fixtures/rescuenet")


def test_descriptor_parses_label_ids():
    labels = load_label_map(FIXTURE / "RescueNet-Segmentation-Dataset-Note.txt")
    assert labels[0] == "Background"
    assert labels[10] == "Pool"


def test_pairs_images_and_masks_by_canonical_stem(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    Image.new("RGB", (2, 2), "white").save(tmp_path / "images/scene-001.jpg")
    Image.new("L", (2, 2), 1).save(tmp_path / "masks/scene-001.png")
    pairs = pair_images_and_masks(tmp_path)
    assert [(pair.pair_id, pair.image.name, pair.mask.name) for pair in pairs] == [
        ("scene-001", "scene-001.jpg", "scene-001.png")
    ]


def test_count_mask_classes_returns_pixel_counts(tmp_path: Path):
    mask = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 1], [1, 2]], dtype=np.uint8)).save(mask)
    pair = RescuePair("scene", tmp_path / "scene.jpg", mask)
    assert count_mask_classes(pair, {0: "Background", 1: "Water", 2: "Building"}) == {
        0: 1,
        1: 2,
        2: 1,
    }


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


def test_selection_is_deterministic_and_group_balanced(rescue_records):
    first = select_pairs(rescue_records, count=30, seed=20260910)
    second = select_pairs(tuple(reversed(rescue_records)), count=30, seed=20260910)
    assert [item.pair_id for item in first] == [item.pair_id for item in second]
    assert len({item.presence_vector for item in first[:4]}) == 4


def test_selection_refuses_to_duplicate_pairs(rescue_records):
    with pytest.raises(ValueError, match="need 30 valid pairs, found 29"):
        select_pairs(rescue_records[:29], count=30, seed=20260910)
