import csv
from pathlib import Path

import pytest

from data.rescuenet_targets import ImageMetadata, parse_dji_xmp, pixel_offset_m, read_metadata
from rescuenet_images import write_dji_image

FALLBACK = {"FC2103": (4.5, 6.17)}
CALIBRATED = {"RelativeAltitude": "+60.00", "GimbalYawDegree": "+40.00", "GimbalPitchDegree": "-89.90", "CalibratedFocalLength": "3000.0"}
SELECTION = Path("data/manifests/rescuenet_selection.csv")
IMAGE_ROOT = Path("data/raw/rescuenet-validation/extracted")


def _metadata(yaw_deg):
    return ImageMetadata("FC220", 4000, 3000, 29.9, -85.4, 60.0, yaw_deg, -90.0, 3000.0, "calibrated", (2000.0, 1500.0))


def test_gsd_is_relative_altitude_over_focal_length():
    assert _metadata(0.0).gsd_m_per_px == pytest.approx(0.02)


def test_yaw_zero_maps_image_up_to_north_and_right_to_east():
    assert pixel_offset_m(_metadata(0.0), 2000, 1400) == pytest.approx((0.0, 2.0))
    assert pixel_offset_m(_metadata(0.0), 2100, 1500) == pytest.approx((2.0, 0.0))


def test_yaw_ninety_maps_image_up_to_east_and_right_to_south():
    assert pixel_offset_m(_metadata(90.0), 2000, 1400) == pytest.approx((2.0, 0.0), abs=1e-12)
    assert pixel_offset_m(_metadata(90.0), 2100, 1500) == pytest.approx((0.0, -2.0), abs=1e-12)


def test_parse_dji_xmp_reads_attributes():
    text = '<rdf:Description drone-dji:RelativeAltitude="+60.80" drone-dji:GimbalYawDegree="-86.40"/>'
    assert parse_dji_xmp(text) == {"RelativeAltitude": "+60.80", "GimbalYawDegree": "-86.40"}


def test_read_metadata_combines_exif_gps_and_dji_xmp(tmp_path: Path):
    fields = {**CALIBRATED, "CalibratedOpticalCenterX": "20.5", "CalibratedOpticalCenterY": "14.5"}
    metadata = read_metadata(write_dji_image(tmp_path / "a.jpg", fields=fields), FALLBACK, 2.0, "a")
    assert (metadata.camera, metadata.width_px, metadata.height_px, metadata.focal_source) == ("FC220", 40, 30, "calibrated")
    assert metadata.latitude == pytest.approx(29.0 + 56.0 / 60.0 + 36.0 / 3600.0)
    assert metadata.longitude == pytest.approx(-(85.0 + 24.0 / 60.0))
    assert (metadata.relative_altitude_m, metadata.yaw_deg, metadata.pitch_deg) == (60.0, 40.0, -89.9)
    assert (metadata.focal_px, metadata.center_px) == (3000.0, (20.5, 14.5))


def test_fallback_focal_length_scales_with_image_width(tmp_path: Path):
    fields = {"RelativeAltitude": "+75.00", "GimbalYawDegree": "0", "GimbalPitchDegree": "-90.0"}
    metadata = read_metadata(write_dji_image(tmp_path / "b.jpg", size=(406, 304), camera="FC2103", fields=fields), FALLBACK, 2.0, "b")
    assert metadata.focal_source == "fallback" and metadata.focal_px == pytest.approx(4.5 * 406 / 6.17)
    assert metadata.center_px == (203.0, 152.0)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"latitude": None, "fields": CALIBRATED}, "c: image has no GPS position"),
        ({"fields": {key: value for key, value in CALIBRATED.items() if key != "GimbalYawDegree"}}, "c: XMP field GimbalYawDegree is missing"),
        ({"fields": {**CALIBRATED, "GimbalPitchDegree": "-80.0"}}, "c: gimbal pitch -80.0 deviates from nadir"),
        ({"camera": "XYZ", "fields": {key: value for key, value in CALIBRATED.items() if key != "CalibratedFocalLength"}}, "c: camera 'XYZ' has no calibrated focal length"),
    ],
)
def test_metadata_errors_name_the_pair(tmp_path: Path, options, message):
    path = write_dji_image(tmp_path / "c.jpg", **options)
    with pytest.raises(ValueError, match=message):
        read_metadata(path, FALLBACK, 2.0, "c")


@pytest.mark.skipif(not IMAGE_ROOT.exists(), reason="RescueNet validation images are not downloaded")
def test_selected_images_are_nadir_with_centimetre_ground_sampling():
    with SELECTION.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    metadata = [read_metadata(IMAGE_ROOT / row["image"], FALLBACK, 2.0, row["pair_id"]) for row in rows]
    assert len(metadata) == 30 and all(0.019 <= item.gsd_m_per_px <= 0.025 for item in metadata)
    assert [item.focal_source for item in metadata].count("fallback") == 1
