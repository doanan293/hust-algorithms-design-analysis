import csv
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import yaml

from data.cli import main
from data.rescuenet_targets import Target, damage_regions, merge_targets, normalize_points
from rescuenet_images import write_dji_image

DESCRIPTOR = Path("tests/data/fixtures/rescuenet/RescueNet-Segmentation-Dataset-Note.txt")
LATITUDE = 29.95
EAST_20_M_DEG = 20.0 / (111320.0 * math.cos(math.radians(LATITUDE)))
FIELDS = {"RelativeAltitude": "+60.0", "GimbalYawDegree": "0.0", "GimbalPitchDegree": "-90.0", "CalibratedFocalLength": "120.0"}


def _dms(value):
    value = abs(value)
    degrees = int(value)
    minutes = int((value - degrees) * 60.0)
    return (float(degrees), float(minutes), (value - degrees - minutes / 60.0) * 3600.0)


def _dataset(tmp_path: Path, second_value=4, second_width=300):
    """Two 300x200 images with GSD 0.5 m, the second 20 m east of the first; image A holds a speck, a western block
    and an eastern block that is the same building as the central block of image B."""
    root = tmp_path / "data"
    images = root / "raw" / "images"
    images.mkdir(parents=True)
    shutil.copy(DESCRIPTOR, root / "raw" / "descriptor.txt")
    first = np.zeros((200, 300), dtype=np.uint8)
    first[10, 10] = 3
    first[95:105, 40:50] = 5
    first[95:105, 185:195] = 4
    second = np.zeros((200, second_width), dtype=np.uint8)
    second[94:106, 144:156] = second_value
    rows = []
    for rank, (pair, longitude, mask) in enumerate((("A", -85.4, first), ("B", -85.4 + EAST_20_M_DEG, second)), start=1):
        write_dji_image(images / f"{pair}.jpg", size=(300, 200), latitude=_dms(LATITUDE), longitude=_dms(longitude), fields=FIELDS)
        Image.fromarray(mask).save(images / f"{pair}_lab.png")
        rows.append({"rank": rank, "pair_id": pair, "image": f"images/{pair}.jpg", "mask": f"images/{pair}_lab.png"})
    (root / "manifests").mkdir()
    with (root / "manifests" / "selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rank", "pair_id", "image", "mask"])
        writer.writeheader()
        writer.writerows(rows)
    raw = yaml.safe_load(Path("configs/data/rescuenet_targets.yaml").read_text(encoding="utf-8"))
    raw.update(selection_manifest="manifests/selection.csv", image_root="raw", label_descriptor="raw/descriptor.txt")
    config = tmp_path / "targets.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return root, config


def test_damage_regions_are_eight_connected_with_inner_representatives():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[0, 29] = 3
    mask[2:8, 3:12] = 4
    mask[10:18, 15:17] = 3
    mask[10:12, 17:25] = 5
    mask[15, 2] = 2
    regions = damage_regions(mask, (3, 4, 5))
    assert [(region.label, region.area_px, region.damage_class) for region in regions] == [(1, 1, 3), (2, 54, 4), (3, 32, 5)]
    assert all(mask[region.pixel_v, region.pixel_u] in (3, 4, 5) for region in regions)
    assert (regions[2].pixel_u, regions[2].pixel_v) != (18, 12) and 15 <= regions[2].pixel_u <= 24


def _target(identifier, east, area):
    return Target(identifier, identifier.split("-")[0], 1, 0, 0, area, 4, east, 0.0)


def test_merging_is_single_linkage_order_independent_and_keeps_the_largest():
    targets = [_target("a-r01", 0.0, 50.0), _target("b-r01", 6.0, 80.0), _target("c-r01", 14.0, 20.0), _target("d-r01", 100.0, 10.0)]
    merged = merge_targets(targets, 10.0)
    assert [(kept.target_id, members) for kept, members in merged] == [("b-r01", ("a-r01", "b-r01", "c-r01")), ("d-r01", ("d-r01",))]
    assert merge_targets(list(reversed(targets)), 10.0) == merged


def test_normalization_keeps_the_aspect_ratio_and_centres_the_short_axis():
    xs, ys, info = normalize_points([500.0, 3500.0, 2000.0], [100.0, 100.0, 1600.0], 2000.0)
    assert info["scale"] == pytest.approx(2000.0 / 3000.0) and info["offset_y_m"] == pytest.approx(500.0)
    assert xs.tolist() == pytest.approx([0.0, 2000.0, 1000.0]) and ys.tolist() == pytest.approx([500.0, 500.0, 1500.0])


def test_targets_command_filters_merges_and_normalizes(tmp_path: Path):
    root, config = _dataset(tmp_path)
    assert main(["rescuenet-targets", "--config", str(config), "--data-root", str(root)]) == 0
    with (root / "manifests" / "rescuenet_targets.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((root / "manifests" / "rescuenet_targets.json").read_text(encoding="utf-8"))
    assert [(row["target_id"], row["damage_class"], row["merged_from"]) for row in rows] == [("A-r02", "5", "A-r02"), ("B-r01", "4", "A-r03;B-r01")]
    assert (float(rows[0]["area_m2"]), float(rows[1]["area_m2"])) == (25.0, 36.0)
    assert [float(row["x_m"]) for row in rows] == [0.0, 2000.0] and all(abs(float(row["y_m"]) - 1000.0) < 20.0 for row in rows)
    assert (summary["region_count"], summary["target_count"]) == (4, 2)
    assert summary["dropped_regions"] == [{"area_m2": 0.25, "pair_id": "A", "reason": "below min_area_m2", "region_label": 1}]
    assert [(pair["kept"], pair["merged"]) for pair in summary["merged_pairs"]] == [("B-r01", "A-r03")]
    assert summary["merged_pairs"][0]["distance_m"] < 1.0
    assert [image["gsd_m_per_px"] for image in summary["images"]] == [0.5, 0.5]


@pytest.mark.parametrize(
    ("options", "message"),
    [({"second_value": 11}, "B: unknown mask classes \\[11\\]"), ({"second_width": 299}, "B: mask size \\(299, 200\\) differs")],
)
def test_targets_command_stops_on_bad_masks(tmp_path: Path, options, message):
    root, config = _dataset(tmp_path, **options)
    with pytest.raises(ValueError, match=message):
        main(["rescuenet-targets", "--config", str(config), "--data-root", str(root)])
