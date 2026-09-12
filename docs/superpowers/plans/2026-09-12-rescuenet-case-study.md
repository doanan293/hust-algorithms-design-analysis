# RescueNet Case Study Implementation Plan (Sub-Project D, Plan D.2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract georeferenced damage targets from the 30 selected RescueNet pairs, build the `rescuenet` scenario set on the `v0` topologies through a replaceable source-placement interface, run B0, B1, B2, P and TRAN on it, and write the map and delivery-progress trace of a representative scenario.

**Architecture:** `src/data/rescuenet_targets.py` reads DJI EXIF and XMP metadata, finds damage regions in the masks, projects them to UTM zone 16N, merges duplicates and normalizes the targets to the 2 km box; `python -m data.cli rescuenet-targets` writes the target manifests. `src/data/scenarios.py` places sources through `place_sources`, sampled for `synthetic` sets and taken from the target manifest for `rescuenet` sets. `configs/experiments/phase2_case_study.yaml` runs the five methods with the TRAN parameters chosen by plan D.1, and `experiments/case_study_trace.py` replans a representative scenario and writes the CSV files that plan D.3 draws.

**Tech Stack:** Python 3.12 in the uv-managed `.venv`; `numpy`, `scipy` (`ndimage`, HiGHS), Pillow 11 (EXIF and XMP), `pyproj`, `cvxpy` with Clarabel (TRAN), `PyYAML`, `pytest`; systemd user scopes for memory caps. No new dependency.

**Spec:** `docs/superpowers/specs/2026-09-12-extensions-final-delivery-design.md`: Sections 8–10, 14, 15.2, acceptance criteria 16.2, and Section 17 "Plan D.2 adjustments" (committed with this plan). Builds on plan D.1 (`docs/superpowers/plans/2026-09-12-literature-baseline.md`) and its `results/phase2/tran_pilot/choice.json`.

## Global Constraints

- Run project code through uv: tests with `uv run --extra test pytest`, scripts with `uv run python ...`. No dependency is added.
- Import boundaries: `src/data` imports `models` and nothing from `baselines`, `bounds`, `optimization`, `literature`, `runner`; `src/models`, `src/baselines`, `src/bounds`, `src/optimization`, `src/literature`, `src/runner` and `src/analysis` do not change; nothing under `src` imports `experiments`.
- **Memory caps.** pytest runs as `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest ...`; the case-study experiment runs under `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0`, the trace under `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0`, because both solve TRAN programs. Never stop processes with `pkill -f`.
- Targets: damage classes 3, 4, 5; eight-connected regions; minimum area 10 m²; representative pixel nearest the region centroid; GSD = relative altitude / focal length in pixels; UTM zone 16N (EPSG:32616); maximum pitch deviation from nadir 2°; camera fallback FC2103: focal length 4.5 mm, sensor width 6.17 mm; single-linkage merge within 10 m keeping the largest area; same scale on both axes into the 2000 m box with the shorter axis centred.
- Scenario sets: `configs/scenarios/rescuenet.yaml` equals `configs/scenarios/v0.yaml` except `set_id` and the source fields; the synthetic generator is unchanged (`data/manifests/scenarios_v0.csv` keeps SHA-256 `7f2325b4e955f3c6265e01c10203a7d90244490446ed22015321e707e818ce17`); RescueNet zones are three k-means clusters (farthest-point start, at most 100 Lloyd steps) sorted by (x, y), radius 400 m, one source per target in manifest order.
- Case-study experiment: set `rescuenet`, split `eval`, realization ids 0–29, 10 tangents, bootstrap 10,000 resamples with seed 20260911, 12 workers; methods B0, B1, B2 (bounds), P with both repair operations (bounds), TRAN with θ = 1/3, μ = 10, block size 1 (bounds). Results are descriptive; no hypothesis tests.
- RescueNet images and masks are not redistributed: commit only `data/manifests/rescuenet_targets.csv`, `data/manifests/rescuenet_targets.json` and `data/manifests/scenarios_rescuenet.csv`; `data/processed/` stays Git-ignored. Also commit `method_realizations.csv`, `bounds.csv`, `summary.csv`, `manifest.json` of `results/phase2/case_study/` and the six files of `results/phase2/case_study/trace/`.
- Test module basenames are unique across `tests/`; every code task follows test-driven development and ends with its own commit.

---

### Task 1: RescueNet target extraction

**Files:**
- Create: `src/data/rescuenet_targets.py`, `configs/data/rescuenet_targets.yaml`
- Modify: `src/data/cli.py`
- Test: `tests/support/rescuenet_images.py`, `tests/data/test_rescuenet_georeference.py`, `tests/data/test_rescuenet_targets.py`

**Interfaces:**
- Consumes: `data.rescuenet.load_label_map`; Pillow EXIF and XMP reading, `pyproj.Transformer`, `scipy.ndimage`; `data.cli.main`; the committed `data/manifests/rescuenet_selection.csv` and the test fixture `tests/data/fixtures/rescuenet/RescueNet-Segmentation-Dataset-Note.txt`.
- Produces: `data.rescuenet_targets`: `TARGET_COLUMNS`; frozen `TargetConfig(data_root, selection_manifest, image_root, label_descriptor, output_csv, output_json, damage_classes, min_area_m2, merge_radius_m, box_size_m, utm_epsg, max_pitch_deviation_deg, camera_fallback, config_sha256)` with `path(relative) -> Path`; `load_target_config(path, data_root=None) -> TargetConfig`; `ImageMetadata(camera, width_px, height_px, latitude, longitude, relative_altitude_m, yaw_deg, pitch_deg, focal_px, focal_source, center_px)` with `gsd_m_per_px`; `parse_dji_xmp(text) -> dict[str, str]`; `read_metadata(path, camera_fallback, max_pitch_deviation_deg, label) -> ImageMetadata`; `pixel_offset_m(metadata, pixel_u, pixel_v) -> (east_m, north_m)`; `DamageRegion(label, area_px, pixel_u, pixel_v, damage_class)`; `damage_regions(mask, damage_classes) -> list[DamageRegion]`; `Target(target_id, pair_id, region_label, pixel_u, pixel_v, area_m2, damage_class, utm_e, utm_n)`; `merge_targets(targets, radius_m) -> list[(Target, tuple[str, ...])]`; `normalize_points(eastings, northings, box_size_m) -> (xs, ys, info)`; `extract_targets(config) -> (rows, summary)`; `write_targets(config, rows, summary)`. Command `python -m data.cli rescuenet-targets --config PATH [--data-root PATH]` prints `{"regions": ..., "targets": ..., "output": ...}`. Test helper `rescuenet_images.write_dji_image(path, size, camera, latitude, longitude, fields)`.

The test helper writes real EXIF GPS and DJI XMP blocks with Pillow, so metadata reading is tested on JPEG files, not on mocks.

- [ ] **Step 1: Write the failing tests**

Create `tests/support/rescuenet_images.py`:

```python
from pathlib import Path

from PIL import Image


def write_dji_image(path: Path, size=(40, 30), camera="FC220", latitude=(29.0, 56.0, 36.0), longitude=(85.0, 24.0, 0.0), fields=None):
    """JPEG with EXIF GPS (north, west) and DJI XMP attributes; latitude=None leaves out the GPS block."""
    exif = Image.Exif()
    exif[272] = camera
    if latitude is not None:
        exif[0x8825] = {1: "N", 2: latitude, 3: "W", 4: longitude}
    attributes = " ".join(f'drone-dji:{key}="{value}"' for key, value in (fields or {}).items())
    xmp = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" {attributes}/></rdf:RDF></x:xmpmeta>'
    ).encode("utf-8")
    Image.new("RGB", size, "white").save(path, exif=exif, xmp=xmp)
    return path
```

Create `tests/data/test_rescuenet_georeference.py`:

```python
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
```

Create `tests/data/test_rescuenet_targets.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/data/test_rescuenet_georeference.py tests/data/test_rescuenet_targets.py`

Expected: collection fails (`2 errors`) with `ModuleNotFoundError: No module named 'data.rescuenet_targets'`.

- [ ] **Step 3: Implement**

Create `src/data/rescuenet_targets.py`:

```python
"""RescueNet damage targets georeferenced from DJI image metadata (spec D Section 8)."""

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy import ndimage
import yaml

from .rescuenet import load_label_map

GPS_IFD = 0x8825
MODEL_TAG = 272
XMP_FIELD = re.compile(r'drone-dji:(\w+)="([^"]*)"')
TARGET_COLUMNS = (
    "target_id", "pair_id", "region_label", "pixel_u", "pixel_v", "area_m2", "damage_class", "lat", "lon", "utm_e",
    "utm_n", "x_m", "y_m", "merged_from",
)


@dataclass(frozen=True)
class TargetConfig:
    """Paths are relative to `data_root`; `camera_fallback` maps a camera model to (focal_mm, sensor_width_mm)."""

    data_root: Path
    selection_manifest: Path
    image_root: Path
    label_descriptor: Path
    output_csv: Path
    output_json: Path
    damage_classes: tuple[int, ...]
    min_area_m2: float
    merge_radius_m: float
    box_size_m: float
    utm_epsg: int
    max_pitch_deviation_deg: float
    camera_fallback: Mapping[str, tuple[float, float]]
    config_sha256: str

    def path(self, relative: Path) -> Path:
        return self.data_root / relative


def load_target_config(path: Path, data_root: Path | None = None) -> TargetConfig:
    payload = path.read_bytes()
    raw = yaml.safe_load(payload)
    fallback = {
        str(model): (float(item["focal_mm"]), float(item["sensor_width_mm"]))
        for model, item in (raw.get("camera_fallback") or {}).items()
    }
    config = TargetConfig(
        data_root=data_root or Path(raw.get("data_root", "data")),
        selection_manifest=Path(raw["selection_manifest"]),
        image_root=Path(raw["image_root"]),
        label_descriptor=Path(raw["label_descriptor"]),
        output_csv=Path(raw["output_csv"]),
        output_json=Path(raw["output_json"]),
        damage_classes=tuple(int(value) for value in raw["damage_classes"]),
        min_area_m2=float(raw["min_area_m2"]),
        merge_radius_m=float(raw["merge_radius_m"]),
        box_size_m=float(raw["box_size_m"]),
        utm_epsg=int(raw["utm_epsg"]),
        max_pitch_deviation_deg=float(raw["max_pitch_deviation_deg"]),
        camera_fallback=fallback,
        config_sha256=hashlib.sha256(payload).hexdigest(),
    )
    if not config.damage_classes or config.min_area_m2 < 0.0 or config.merge_radius_m < 0.0 or config.box_size_m <= 0.0:
        raise ValueError("rescuenet targets: damage_classes must be non-empty, areas and radius non-negative, box size positive")
    return config


@dataclass(frozen=True)
class ImageMetadata:
    camera: str
    width_px: int
    height_px: int
    latitude: float
    longitude: float
    relative_altitude_m: float
    yaw_deg: float
    pitch_deg: float
    focal_px: float
    focal_source: str
    center_px: tuple[float, float]

    @property
    def gsd_m_per_px(self) -> float:
        return self.relative_altitude_m / self.focal_px


def parse_dji_xmp(text: str) -> dict[str, str]:
    return dict(XMP_FIELD.findall(text))


def _degrees(values: Sequence[object], reference: str) -> float:
    degrees, minutes, seconds = (float(value) for value in values)
    return (-1.0 if reference in ("S", "W") else 1.0) * (degrees + minutes / 60.0 + seconds / 3600.0)


def read_metadata(
    path: Path, camera_fallback: Mapping[str, tuple[float, float]], max_pitch_deviation_deg: float, label: str
) -> ImageMetadata:
    """GPS from EXIF; altitude, gimbal angles and calibration from DJI XMP; focal length from the fallback table if needed."""
    with Image.open(path) as image:
        exif = image.getexif()
        gps = dict(exif.get_ifd(GPS_IFD))
        camera = str(exif.get(MODEL_TAG, "")).strip("\x00").strip()
        xmp = image.info.get("xmp", b"")
        width, height = image.size
    fields = parse_dji_xmp(xmp.decode("utf-8", "ignore") if isinstance(xmp, bytes) else str(xmp))
    if not all(key in gps for key in (1, 2, 3, 4)):
        raise ValueError(f"{label}: image has no GPS position")
    for key in ("RelativeAltitude", "GimbalYawDegree", "GimbalPitchDegree"):
        if key not in fields:
            raise ValueError(f"{label}: XMP field {key} is missing")
    pitch = float(fields["GimbalPitchDegree"])
    if abs(pitch + 90.0) > max_pitch_deviation_deg:
        raise ValueError(f"{label}: gimbal pitch {pitch} deviates from nadir by more than {max_pitch_deviation_deg} degrees")
    if "CalibratedFocalLength" in fields:
        focal_px, source = float(fields["CalibratedFocalLength"]), "calibrated"
    elif camera in camera_fallback:
        focal_mm, sensor_width_mm = camera_fallback[camera]
        focal_px, source = focal_mm * width / sensor_width_mm, "fallback"
    else:
        raise ValueError(f"{label}: camera {camera!r} has no calibrated focal length and no fallback entry")
    center = (
        float(fields.get("CalibratedOpticalCenterX", width / 2.0)),
        float(fields.get("CalibratedOpticalCenterY", height / 2.0)),
    )
    return ImageMetadata(
        camera, width, height, _degrees(gps[2], gps[1]), _degrees(gps[4], gps[3]), float(fields["RelativeAltitude"]),
        float(fields["GimbalYawDegree"]), pitch, focal_px, source, center,
    )


def pixel_offset_m(metadata: ImageMetadata, pixel_u: float, pixel_v: float) -> tuple[float, float]:
    """East and north offset of a pixel from the optical centre of a nadir image whose top points to the yaw (clockwise from north)."""
    gsd = metadata.gsd_m_per_px
    right = (pixel_u - metadata.center_px[0]) * gsd
    up = (metadata.center_px[1] - pixel_v) * gsd
    yaw = math.radians(metadata.yaw_deg)
    return (right * math.cos(yaw) + up * math.sin(yaw), -right * math.sin(yaw) + up * math.cos(yaw))


@dataclass(frozen=True)
class DamageRegion:
    label: int
    area_px: int
    pixel_u: int
    pixel_v: int
    damage_class: int


def damage_regions(mask: np.ndarray, damage_classes: Sequence[int]) -> list[DamageRegion]:
    """Eight-connected regions of damage pixels; the representative is the region pixel nearest the region centroid."""
    labels, _ = ndimage.label(np.isin(mask, damage_classes), structure=np.ones((3, 3), dtype=int))
    regions = []
    for index, bounds in enumerate(ndimage.find_objects(labels), start=1):
        inside = labels[bounds] == index
        rows, cols = np.nonzero(inside)
        rows, cols = rows + bounds[0].start, cols + bounds[1].start
        nearest = int(np.argmin((cols - cols.mean()) ** 2 + (rows - rows.mean()) ** 2))
        regions.append(DamageRegion(index, len(rows), int(cols[nearest]), int(rows[nearest]), int(mask[bounds][inside].max())))
    return regions


@dataclass(frozen=True)
class Target:
    target_id: str
    pair_id: str
    region_label: int
    pixel_u: int
    pixel_v: int
    area_m2: float
    damage_class: int
    utm_e: float
    utm_n: float


def merge_targets(targets: Sequence[Target], radius_m: float) -> list[tuple[Target, tuple[str, ...]]]:
    """Single-linkage groups of targets within radius_m; each group keeps its largest member (ties: smallest id)."""
    ordered = sorted(targets, key=lambda target: target.target_id)
    parent = list(range(len(ordered)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(len(ordered)):
        for second in range(first + 1, len(ordered)):
            gap = math.hypot(ordered[first].utm_e - ordered[second].utm_e, ordered[first].utm_n - ordered[second].utm_n)
            if gap <= radius_m:
                parent[root(second)] = root(first)
    groups: dict[int, list[Target]] = {}
    for index, target in enumerate(ordered):
        groups.setdefault(root(index), []).append(target)
    merged = [
        (min(members, key=lambda target: (-target.area_m2, target.target_id)), tuple(sorted(target.target_id for target in members)))
        for members in groups.values()
    ]
    return sorted(merged, key=lambda item: item[0].target_id)


def normalize_points(
    eastings: Sequence[float], northings: Sequence[float], box_size_m: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Same scale on both axes so the longer extent fills the box; the shorter axis is centred."""
    east, north = np.asarray(eastings, dtype=float), np.asarray(northings, dtype=float)
    extent_e, extent_n = float(east.max() - east.min()), float(north.max() - north.min())
    longest = max(extent_e, extent_n)
    scale = box_size_m / longest if longest > 0.0 else 1.0
    offset_x, offset_y = (box_size_m - extent_e * scale) / 2.0, (box_size_m - extent_n * scale) / 2.0
    info = {
        "box_size_m": box_size_m, "scale": scale, "e_min": float(east.min()), "n_min": float(north.min()),
        "extent_e_m": extent_e, "extent_n_m": extent_n, "offset_x_m": offset_x, "offset_y_m": offset_y,
    }
    return (east - east.min()) * scale + offset_x, (north - north.min()) * scale + offset_y, info


def extract_targets(config: TargetConfig) -> tuple[list[dict[str, object]], dict[str, object]]:
    labels = load_label_map(config.path(config.label_descriptor))
    selection_path = config.path(config.selection_manifest)
    with selection_path.open(newline="", encoding="utf-8") as stream:
        selection = sorted(csv.DictReader(stream), key=lambda row: int(row["rank"]))
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{config.utm_epsg}", always_xy=True)
    targets: list[Target] = []
    images: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    for row in selection:
        pair = row["pair_id"]
        metadata = read_metadata(config.path(config.image_root) / row["image"], config.camera_fallback, config.max_pitch_deviation_deg, pair)
        with Image.open(config.path(config.image_root) / row["mask"]) as mask_image:
            mask = np.asarray(mask_image)
        if mask.shape != (metadata.height_px, metadata.width_px):
            raise ValueError(f"{pair}: mask size {mask.shape[::-1]} differs from image size {(metadata.width_px, metadata.height_px)}")
        unknown = sorted(set(np.unique(mask).tolist()) - set(labels))
        if unknown:
            raise ValueError(f"{pair}: unknown mask classes {unknown}")
        east0, north0 = to_utm.transform(metadata.longitude, metadata.latitude)
        pixel_area = metadata.gsd_m_per_px**2
        kept = 0
        for region in damage_regions(mask, config.damage_classes):
            area = region.area_px * pixel_area
            if area < config.min_area_m2:
                dropped.append({"pair_id": pair, "region_label": region.label, "area_m2": round(area, 3), "reason": "below min_area_m2"})
                continue
            offset_e, offset_n = pixel_offset_m(metadata, region.pixel_u, region.pixel_v)
            targets.append(
                Target(f"{pair}-r{region.label:02d}", pair, region.label, region.pixel_u, region.pixel_v, area,
                       region.damage_class, east0 + offset_e, north0 + offset_n)
            )
            kept += 1
        images.append({
            "pair_id": pair, "camera": metadata.camera, "gsd_m_per_px": round(metadata.gsd_m_per_px, 8),
            "yaw_deg": metadata.yaw_deg, "pitch_deg": metadata.pitch_deg, "focal_px": round(metadata.focal_px, 6),
            "focal_source": metadata.focal_source, "targets": kept,
        })
    if not targets:
        raise ValueError("rescuenet targets: no damage region passed the filters")
    merged = merge_targets(targets, config.merge_radius_m)
    xs, ys, normalization = normalize_points([item[0].utm_e for item in merged], [item[0].utm_n for item in merged], config.box_size_m)
    to_geo = Transformer.from_crs(f"EPSG:{config.utm_epsg}", "EPSG:4326", always_xy=True)
    by_id = {target.target_id: target for target in targets}
    rows, merged_pairs = [], []
    for (target, members), x, y in zip(merged, xs, ys):
        lon, lat = to_geo.transform(target.utm_e, target.utm_n)
        rows.append({
            "target_id": target.target_id, "pair_id": target.pair_id, "region_label": target.region_label,
            "pixel_u": target.pixel_u, "pixel_v": target.pixel_v, "area_m2": round(target.area_m2, 3),
            "damage_class": target.damage_class, "lat": round(lat, 8), "lon": round(lon, 8), "utm_e": round(target.utm_e, 3),
            "utm_n": round(target.utm_n, 3), "x_m": round(float(x), 3), "y_m": round(float(y), 3), "merged_from": ";".join(members),
        })
        for member in members:
            if member != target.target_id:
                other = by_id[member]
                distance = math.hypot(other.utm_e - target.utm_e, other.utm_n - target.utm_n)
                merged_pairs.append({"kept": target.target_id, "merged": member, "distance_m": round(distance, 3)})
    summary = {
        "config_sha256": config.config_sha256,
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "region_count": len(targets) + len(dropped),
        "target_count": len(rows),
        "images": images,
        "dropped_regions": dropped,
        "merged_pairs": merged_pairs,
        "normalization": {key: round(value, 6) for key, value in normalization.items()},
    }
    return rows, summary


def write_targets(config: TargetConfig, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    output = config.path(config.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(TARGET_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    config.path(config.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

In `src/data/cli.py` (4 edits):

Edit 1: replace

```python
from .rescuenet import count_mask_classes, load_label_map, pair_images_and_masks, RescueRecord, select_pairs
```

with

```python
from .rescuenet import count_mask_classes, load_label_map, pair_images_and_masks, RescueRecord, select_pairs
from .rescuenet_targets import extract_targets, load_target_config, write_targets
```

Edit 2: replace

```python
    scenarios.add_argument("--data-root", type=Path)
    return parser
```

with

```python
    scenarios.add_argument("--data-root", type=Path)
    targets = subparsers.add_parser("rescuenet-targets")
    targets.add_argument("--config", type=Path, required=True)
    targets.add_argument("--data-root", type=Path)
    return parser
```

Edit 3: replace

```python
STAGES = ("download", "verify", "inventory", "preprocess")
```

with

```python
def run_rescuenet_targets(config_path: Path, data_root: Path | None) -> int:
    config = load_target_config(config_path, data_root)
    rows, summary = extract_targets(config)
    write_targets(config, rows, summary)
    print(json.dumps({"regions": summary["region_count"], "targets": len(rows), "output": str(config.path(config.output_csv))}))
    return 0


STAGES = ("download", "verify", "inventory", "preprocess")
```

Edit 4: replace

```python
    if args.command == "scenarios":
        return run_scenarios(args.config, args.data_root)
```

with

```python
    if args.command == "scenarios":
        return run_scenarios(args.config, args.data_root)
    if args.command == "rescuenet-targets":
        return run_rescuenet_targets(args.config, args.data_root)
```


Create `configs/data/rescuenet_targets.yaml`:

```yaml
data_root: data
selection_manifest: manifests/rescuenet_selection.csv
image_root: raw/rescuenet-validation/extracted
label_descriptor: raw/rescuenet-descriptor/RescueNet-Segmentation-Dataset-Note.txt
output_csv: manifests/rescuenet_targets.csv
output_json: manifests/rescuenet_targets.json
damage_classes: [3, 4, 5]
min_area_m2: 10.0
merge_radius_m: 10.0
box_size_m: 2000.0
utm_epsg: 32616
max_pitch_deviation_deg: 2.0
camera_fallback:
  FC2103: {focal_mm: 4.5, sensor_width_mm: 6.17}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/data/test_rescuenet_georeference.py tests/data/test_rescuenet_targets.py`

Expected: `17 passed`.

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `255 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/support/rescuenet_images.py tests/data/test_rescuenet_georeference.py tests/data/test_rescuenet_targets.py src/data/rescuenet_targets.py src/data/cli.py configs/data/rescuenet_targets.yaml
git commit -m "feat: extract georeferenced RescueNet damage targets"
```

### Task 2: Replaceable source placement and the rescuenet scenario set

**Files:**
- Create: `configs/scenarios/rescuenet.yaml`
- Modify: `src/data/scenarios.py`, `src/data/cli.py`
- Test: `tests/support/scenario_stage_inputs.py`, `tests/data/test_scenario_stage.py` (replaced), `tests/data/test_scenario_sources.py`

**Interfaces:**
- Consumes: Task 1 (manifest columns `x_m`, `y_m`); `data.scenarios` of sub-project A; `data.cli.run_scenarios`.
- Produces: `data.scenarios`: `SYNTHETIC = "synthetic"`, `RESCUENET = "rescuenet"`, `SYNTHETIC_SOURCE_FIELDS`, `TARGET_SOURCE_FIELDS`, `KMEANS_MAX_ITERATIONS = 100`; `ScenarioSetConfig` gains `source_generator`, `targets_manifest`, `targets_manifest_sha256` (the four synthetic-only fields are `None` for `rescuenet`); `load_targets(path, expected_sha256) -> tuple[(x_m, y_m), ...]`; `target_zones_and_sources(targets, zone_count, radius_m, label) -> (zones, sources)`; `place_sources(config, seed, box_m, label, targets=()) -> (zones, sources)`; `generate_scenario(..., targets=())` and `build_scenario_set(..., targets=())`. `data.cli scenarios` loads the target manifest for `rescuenet` sets. `configs/scenarios/rescuenet.yaml`. Test helpers `scenario_stage_inputs.write_scenario_inputs(root)` and `write_small_config(tmp_path)`.

`test_synthetic_generation_is_unchanged_by_the_source_interface` pins the SHA-256 of a synthetic scenario computed with the code before this task; the synthetic branch keeps the RNG stream names `damage-zones` and `sources`. `configs/scenarios/rescuenet.yaml` records the SHA-256 of `data/manifests/rescuenet_targets.csv` that Task 4 regenerates.

- [ ] **Step 1: Write the failing tests**

Create `tests/support/scenario_stage_inputs.py`:

```python
import csv
import json
from pathlib import Path
import shutil

import yaml

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def write_scenario_inputs(root: Path) -> None:
    network_dir = root / "processed" / "networks"
    network_dir.mkdir(parents=True)
    rows = []
    for index in range(6):
        nodes = [{"id": f"n{i}", "x_m": 2000.0 * i / (15 + index), "y_m": 1000.0 + 50.0 * (i % 3)} for i in range(16 + index)]
        edges = [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(15 + index)]
        network = {
            "network_id": f"Line{index}",
            "nodes": nodes,
            "edges": edges,
            "normalization": {"box_size_m": 2000.0, "scale_x": 1.0, "scale_y": 1.0},
        }
        (network_dir / f"Line{index}.json").write_text(json.dumps(network), encoding="utf-8")
        rows.append({"topology_id": f"Line{index}", "eligible": "True"})
    rows.append({"topology_id": "Tiny", "eligible": "False"})
    manifests = root / "manifests"
    manifests.mkdir()
    with (manifests / "topology_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["eligible", "topology_id"])
        writer.writeheader()
        writer.writerows(rows)
    (manifests / "sources.json").write_text(
        json.dumps([{"source_id": "topology-zoo", "sha256": "1" * 64}, {"source_id": "rescuenet-validation", "sha256": "2" * 64}]),
        encoding="utf-8",
    )
    sndlib = root / "raw" / "sndlib-networks-xml" / "extracted" / "sndlib-networks-xml"
    sndlib.mkdir(parents=True)
    shutil.copy(FIXTURES / "sndlib" / "mini.xml", sndlib / "abilene.xml")


def write_small_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(Path("configs/scenarios/v0.yaml").read_text(encoding="utf-8"))
    raw["topology"].update(strata=4, dev_count=1, eval_replicates=2, dev_replicates=1)
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path
```

Replace the contents of `tests/data/test_scenario_stage.py` with:

```python
import csv
import json
from pathlib import Path

from data.cli import main
from models.scenario import load_scenario
from scenario_stage_inputs import write_scenario_inputs, write_small_config


def test_scenarios_stage_writes_valid_files_and_reproducible_manifest(tmp_path: Path):
    root = tmp_path / "data"
    write_scenario_inputs(root)
    config = write_small_config(tmp_path)
    assert main(["scenarios", "--config", str(config), "--data-root", str(root)]) == 0
    manifest = root / "manifests" / "scenarios_v0.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7
    assert sorted(row["split"] for row in rows).count("dev") == 1
    for row in rows:
        scenario = load_scenario(root / "processed" / "scenarios" / "v0" / f"{row['scenario_id']}.json")
        assert scenario.sha256 == row["sha256"]
        assert scenario.provenance["raw_sha256"] == ["1" * 64]
        assert 0.0 <= float(row["ground_connected_source_fraction"]) <= 1.0
    first = manifest.read_text(encoding="utf-8")
    assert main(["scenarios", "--config", str(config), "--data-root", str(root)]) == 0
    assert manifest.read_text(encoding="utf-8") == first
```

Create `tests/data/test_scenario_sources.py`:

```python
from dataclasses import replace
import csv
import hashlib
from pathlib import Path

import pytest
import yaml

from data.cli import main
from data.scenarios import RESCUENET, generate_scenario, load_scenario_set_config, load_targets, target_zones_and_sources
from models.scenario import load_scenario, scenario_from_dict
from scenario_stage_inputs import write_scenario_inputs

V0 = load_scenario_set_config(Path("configs/scenarios/v0.yaml"))
SYNTHETIC_GRID_SHA256 = "a9b6da316a34a8820d562ddb3288688fbc77732a2588bf3994c591ad173948cf"
POINTS = ((100.0, 100.0), (1000.0, 1800.0), (140.0, 60.0), (1900.0, 200.0), (1040.0, 1760.0), (60.0, 140.0), (1860.0, 240.0))
SOURCE_KEYS = {"zone_min_separation_m", "zone_max_attempts", "source_count", "source_sigma_m", "source_generator", "targets_manifest", "targets_manifest_sha256"}


def grid_network(network_id: str, rows: int, cols: int) -> dict[str, object]:
    nodes = [
        {"id": f"n{row * cols + col}", "x_m": 2000.0 * col / (cols - 1), "y_m": 2000.0 * row / (rows - 1)}
        for row in range(rows)
        for col in range(cols)
    ]
    edges = []
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            if col + 1 < cols:
                edges.append({"source": f"n{index}", "target": f"n{index + 1}"})
            if row + 1 < rows:
                edges.append({"source": f"n{index}", "target": f"n{index + cols}"})
    return {"network_id": network_id, "nodes": nodes, "edges": edges, "normalization": {"box_size_m": 2000.0, "scale_x": 1.0, "scale_y": 1.0}}


def _generate(config, targets=()):
    return generate_scenario(grid_network("Grid", 4, 5), "eval", 0, [1.0, 50.0, 100.0], config, "a" * 64, ["c" * 64, "b" * 64], "d" * 64, targets)


def _rescuenet_config():
    return replace(
        V0, set_id="rescuenet", source_generator=RESCUENET, zone_min_separation_m=None, zone_max_attempts=None, source_count=None,
        source_sigma_m=None, targets_manifest=Path("manifests/targets.csv"), targets_manifest_sha256="e" * 64,
    )


def test_synthetic_generation_is_unchanged_by_the_source_interface():
    assert V0.source_generator == "synthetic" and V0.targets_manifest is None
    assert scenario_from_dict(_generate(V0)).sha256 == SYNTHETIC_GRID_SHA256


def test_target_zones_are_sorted_kmeans_clusters_with_one_source_per_target():
    zones, sources = target_zones_and_sources(POINTS, 3, 400.0, "Grid")
    rounded = [(round(zone["x_m"], 6), round(zone["y_m"], 6), zone["radius_m"]) for zone in zones]
    assert rounded == [(100.0, 100.0, 400.0), (1020.0, 1780.0, 400.0), (1880.0, 220.0, 400.0)]
    assert [source["id"] for source in sources] == [f"s{index:02d}" for index in range(7)]
    assert [(source["x_m"], source["y_m"]) for source in sources] == list(POINTS)
    assert [source["zone"] for source in sources] == [0, 1, 0, 2, 1, 0, 2]
    reversed_zones, _ = target_zones_and_sources(tuple(reversed(POINTS)), 3, 400.0, "Grid")
    assert [(round(zone["x_m"], 6), round(zone["y_m"], 6)) for zone in reversed_zones] == [item[:2] for item in rounded]


def test_too_few_targets_for_the_zones_are_rejected():
    with pytest.raises(ValueError, match="Grid: 2 targets cannot form 3 zones"):
        target_zones_and_sources(POINTS[:2], 3, 400.0, "Grid")


def test_rescuenet_scenario_uses_targets_and_records_the_manifest_hash():
    config = _rescuenet_config()
    scenario = scenario_from_dict(_generate(config, POINTS))
    assert [source.xy for source in scenario.sources] == list(POINTS)
    assert len(scenario.damage_zones) == 3 and scenario.provenance["targets_manifest_sha256"] == "e" * 64
    assert scenario_from_dict(_generate(config, POINTS)).sha256 == scenario.sha256
    assert len(scenario.alerts) == 40 and {alert.source for alert in scenario.alerts} <= {source.id for source in scenario.sources}


def test_load_targets_checks_the_manifest_hash(tmp_path: Path):
    path = tmp_path / "targets.csv"
    path.write_text("target_id,x_m,y_m\nt1,1.5,2.5\nt2,3.0,4.0\n", encoding="utf-8")
    assert load_targets(path, hashlib.sha256(path.read_bytes()).hexdigest()) == ((1.5, 2.5), (3.0, 4.0))
    with pytest.raises(ValueError, match="does not match workload.targets_manifest_sha256"):
        load_targets(path, "0" * 64)


def test_rescuenet_config_differs_from_v0_only_in_set_id_and_source_fields():
    base = yaml.safe_load(Path("configs/scenarios/v0.yaml").read_text(encoding="utf-8"))
    family = yaml.safe_load(Path("configs/scenarios/rescuenet.yaml").read_text(encoding="utf-8"))
    config = load_scenario_set_config(Path("configs/scenarios/rescuenet.yaml"))
    assert (config.set_id, config.source_generator, config.targets_manifest, config.source_count) == (
        "rescuenet", RESCUENET, Path("manifests/rescuenet_targets.csv"), None,
    )
    assert {key: value for key, value in family.items() if key not in ("set_id", "workload")} == {
        key: value for key, value in base.items() if key not in ("set_id", "workload")
    }
    assert {key: value for key, value in family["workload"].items() if key not in SOURCE_KEYS} == {
        key: value for key, value in base["workload"].items() if key not in SOURCE_KEYS
    }


def _write_config(tmp_path: Path, source: str, change) -> Path:
    raw = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    change(raw["workload"])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("source", "change", "message"),
    [
        ("configs/scenarios/rescuenet.yaml", lambda workload: workload.update(source_generator="lidar"), "unknown generator 'lidar'"),
        ("configs/scenarios/rescuenet.yaml", lambda workload: workload.update(source_count=15), "apply only to source_generator synthetic"),
        ("configs/scenarios/rescuenet.yaml", lambda workload: workload.pop("targets_manifest_sha256"), "needs \\['targets_manifest_sha256'\\]"),
        ("configs/scenarios/v0.yaml", lambda workload: workload.update(targets_manifest="manifests/x.csv"), "apply only to source_generator rescuenet"),
    ],
)
def test_invalid_source_generator_fields_are_rejected(tmp_path: Path, source, change, message):
    with pytest.raises(ValueError, match=message):
        load_scenario_set_config(_write_config(tmp_path, source, change))


def test_scenarios_command_builds_rescuenet_scenarios_from_the_manifest(tmp_path: Path):
    root = tmp_path / "data"
    write_scenario_inputs(root)
    targets = root / "manifests" / "rescuenet_targets.csv"
    targets.write_text("target_id,x_m,y_m\n" + "".join(f"t{index},{x},{y}\n" for index, (x, y) in enumerate(POINTS)), encoding="utf-8")
    raw = yaml.safe_load(Path("configs/scenarios/rescuenet.yaml").read_text(encoding="utf-8"))
    raw["topology"].update(strata=4, dev_count=1, eval_replicates=2, dev_replicates=1)
    raw["workload"]["targets_manifest_sha256"] = hashlib.sha256(targets.read_bytes()).hexdigest()
    config = tmp_path / "rescuenet.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert main(["scenarios", "--config", str(config), "--data-root", str(root)]) == 0
    with (root / "manifests" / "scenarios_rescuenet.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7
    scenario = load_scenario(root / "processed" / "scenarios" / "rescuenet" / f"{rows[0]['scenario_id']}.json")
    assert [source.xy for source in scenario.sources] == list(POINTS)
    raw["workload"]["targets_manifest_sha256"] = "0" * 64
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        main(["scenarios", "--config", str(config), "--data-root", str(root)])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/data/test_scenario_sources.py tests/data/test_scenario_stage.py`

Expected: collection fails (`1 error`) with `ImportError: cannot import name 'RESCUENET' from 'data.scenarios' (src/data/scenarios.py)`.

- [ ] **Step 3: Implement**

In `src/data/scenarios.py` (12 edits):

Edit 1: replace

```python
import copy
from dataclasses import dataclass
import hashlib
import math
```

with

```python
import copy
import csv
from dataclasses import dataclass
import hashlib
import io
import math
```

Edit 2: replace

```python
PHYSICAL_SECTIONS = ("uav", "time", "spectrum", "channel", "source_power_w", "paths")
```

with

```python
PHYSICAL_SECTIONS = ("uav", "time", "spectrum", "channel", "source_power_w", "paths")
SYNTHETIC = "synthetic"
RESCUENET = "rescuenet"
SYNTHETIC_SOURCE_FIELDS = ("zone_min_separation_m", "zone_max_attempts", "source_count", "source_sigma_m")
TARGET_SOURCE_FIELDS = ("targets_manifest", "targets_manifest_sha256")
KMEANS_MAX_ITERATIONS = 100
```

Edit 3: replace

```python
    zone_count: int
    zone_min_separation_m: float
    zone_radius_m: float
    zone_max_attempts: int
    source_count: int
    source_sigma_m: float
```

with

```python
    zone_count: int
    zone_min_separation_m: float | None
    zone_radius_m: float
    zone_max_attempts: int | None
    source_count: int | None
    source_sigma_m: float | None
```

Edit 4: replace

```python
    backhaul_capacity_bps: float
    physical: Mapping[str, object]
```

with

```python
    backhaul_capacity_bps: float
    physical: Mapping[str, object]
    source_generator: str = SYNTHETIC
    targets_manifest: Path | None = None
    targets_manifest_sha256: str | None = None
```

Edit 5: replace

```python
def load_scenario_set_config(path: Path) -> ScenarioSetConfig:
```

with

```python
def _source_fields(workload: Mapping[str, object]) -> dict[str, object]:
    """Fields of the source generator: sampled zones and sources, or a RescueNet target manifest (spec D Section 9)."""
    generator = str(workload.get("source_generator", SYNTHETIC))
    if generator == SYNTHETIC:
        extra = sorted(key for key in TARGET_SOURCE_FIELDS if key in workload)
        if extra:
            raise ValueError(f"workload: {extra} apply only to source_generator {RESCUENET}")
        return {
            "source_generator": SYNTHETIC,
            "zone_min_separation_m": float(workload["zone_min_separation_m"]),
            "zone_max_attempts": int(workload["zone_max_attempts"]),
            "source_count": int(workload["source_count"]),
            "source_sigma_m": float(workload["source_sigma_m"]),
        }
    if generator != RESCUENET:
        raise ValueError(f"workload.source_generator: unknown generator {generator!r}; expected {SYNTHETIC} or {RESCUENET}")
    extra = sorted(key for key in SYNTHETIC_SOURCE_FIELDS if key in workload)
    if extra:
        raise ValueError(f"workload: {extra} apply only to source_generator {SYNTHETIC}")
    missing = sorted(key for key in TARGET_SOURCE_FIELDS if key not in workload)
    if missing:
        raise ValueError(f"workload: source_generator {RESCUENET} needs {missing}")
    return {
        "source_generator": RESCUENET,
        **dict.fromkeys(SYNTHETIC_SOURCE_FIELDS),
        "targets_manifest": Path(str(workload["targets_manifest"])),
        "targets_manifest_sha256": str(workload["targets_manifest_sha256"]),
    }


def load_scenario_set_config(path: Path) -> ScenarioSetConfig:
```

Edit 6: replace

```python
        zone_count=int(workload["zone_count"]),
        zone_min_separation_m=float(workload["zone_min_separation_m"]),
        zone_radius_m=float(workload["zone_radius_m"]),
        zone_max_attempts=int(workload["zone_max_attempts"]),
        source_count=int(workload["source_count"]),
        source_sigma_m=float(workload["source_sigma_m"]),
```

with

```python
        zone_count=int(workload["zone_count"]),
        zone_radius_m=float(workload["zone_radius_m"]),
        **_source_fields(workload),
```

Edit 7: replace

```python
def sample_alerts(
```

with

```python
def load_targets(path: Path, expected_sha256: str) -> tuple[tuple[float, float], ...]:
    """Normalized (x_m, y_m) of the RescueNet target manifest, in file order, after checking its SHA-256."""
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {actual} does not match workload.targets_manifest_sha256 {expected_sha256}")
    rows = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    return tuple((float(row["x_m"]), float(row["y_m"])) for row in rows)


def target_zones_and_sources(
    targets: Sequence[tuple[float, float]], zone_count: int, radius_m: float, label: str
) -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    """One source per target; zones are k-means clusters (farthest-point start, Lloyd steps) sorted by (x, y)."""
    points = np.asarray(targets, dtype=float).reshape(-1, 2)
    if len(points) < zone_count:
        raise ValueError(f"{label}: {len(points)} targets cannot form {zone_count} zones")
    chosen = [min(range(len(points)), key=lambda index: (points[index, 0], points[index, 1]))]
    while len(chosen) < zone_count:
        nearest = np.min(np.linalg.norm(points[:, None, :] - points[chosen][None, :, :], axis=2), axis=1)
        chosen.append(int(np.argmax(nearest)))
    centers = points[chosen].copy()
    assignment = np.full(len(points), -1)
    for _ in range(KMEANS_MAX_ITERATIONS):
        updated = np.argmin(np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2), axis=1)
        if np.array_equal(updated, assignment):
            break
        assignment = updated
        for zone in range(zone_count):
            members = points[assignment == zone]
            if len(members):
                centers[zone] = members.mean(axis=0)
    order = sorted(range(zone_count), key=lambda zone: (centers[zone, 0], centers[zone, 1]))
    rank = {zone: position for position, zone in enumerate(order)}
    zones = [{"x_m": float(centers[zone, 0]), "y_m": float(centers[zone, 1]), "radius_m": radius_m} for zone in order]
    width = max(2, len(str(len(points) - 1)))
    sources = [
        {"id": f"s{index:0{width}d}", "x_m": float(x), "y_m": float(y), "zone": rank[int(assignment[index])]}
        for index, (x, y) in enumerate(points)
    ]
    return zones, sources


def place_sources(
    config: ScenarioSetConfig, seed: int, box_m: float, label: str, targets: Sequence[tuple[float, float]] = ()
) -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    """Damage zones and source points of one scenario, sampled or taken from RescueNet targets (spec D Section 9)."""
    if config.source_generator == RESCUENET:
        return target_zones_and_sources(targets, config.zone_count, config.zone_radius_m, label)
    zones = sample_damage_zones(
        named_rng(seed, "damage-zones"),
        box_m,
        config.zone_count,
        config.zone_min_separation_m,
        config.zone_radius_m,
        config.zone_max_attempts,
        label,
    )
    return zones, sample_sources(named_rng(seed, "sources"), zones, config.source_count, config.source_sigma_m, box_m)


def sample_alerts(
```

Edit 8: replace

```python
    config_hash: str,
) -> dict[str, object]:
```

with

```python
    config_hash: str,
    targets: Sequence[tuple[float, float]] = (),
) -> dict[str, object]:
```

Edit 9: replace

```python
    zones = sample_damage_zones(
        named_rng(seed, "damage-zones"),
        box_m,
        config.zone_count,
        config.zone_min_separation_m,
        config.zone_radius_m,
        config.zone_max_attempts,
        network_id,
    )
    sources = sample_sources(named_rng(seed, "sources"), zones, config.source_count, config.source_sigma_m, box_m)
```

with

```python
    zones, sources = place_sources(config, seed, box_m, network_id, targets)
```

Edit 10: replace

```python
            "generator_version": GENERATOR_VERSION,
        },
```

with

```python
            "generator_version": GENERATOR_VERSION,
            **({"targets_manifest_sha256": config.targets_manifest_sha256} if config.source_generator == RESCUENET else {}),
        },
```

Edit 11: replace

```python
    config_hash: str,
) -> list[dict[str, object]]:
```

with

```python
    config_hash: str,
    targets: Sequence[tuple[float, float]] = (),
) -> list[dict[str, object]]:
```

Edit 12: replace

```python
                    raw_sha256,
                    config_hash,
                )
```

with

```python
                    raw_sha256,
                    config_hash,
                    targets,
                )
```


In `src/data/cli.py` (2 edits):

Edit 1: replace

```python
from .scenarios import build_scenario_set, load_scenario_set_config
```

with

```python
from .scenarios import RESCUENET, build_scenario_set, load_scenario_set_config, load_targets
```

Edit 2: replace

```python
    scenarios = build_scenario_set(
        networks,
        network_sha256,
        demand_values,
        config,
        raw_sha256,
        hashlib.sha256(config_path.read_bytes()).hexdigest(),
    )
```

with

```python
    targets = load_targets(root / config.targets_manifest, config.targets_manifest_sha256) if config.source_generator == RESCUENET else ()
    scenarios = build_scenario_set(
        networks,
        network_sha256,
        demand_values,
        config,
        raw_sha256,
        hashlib.sha256(config_path.read_bytes()).hexdigest(),
        targets,
    )
```


Create `configs/scenarios/rescuenet.yaml`:

```yaml
set_id: rescuenet
data_root: data
selection_seed: 20260911
scenario_base_seed: 20260911
topology:
  strata: 13
  dev_count: 3
  eval_replicates: 3
  dev_replicates: 2
workload:
  source_generator: rescuenet
  targets_manifest: manifests/rescuenet_targets.csv
  targets_manifest_sha256: f1e155c311661a40700a9ee0a9833b58c68f34b116700b3f5f1ba34c81124a0b
  zone_count: 3
  zone_radius_m: 400.0
  alert_count: 40
  deadline_min_slots: 20
  deadline_max_slots: 60
  size_ref_bits: 250000
  size_ratio_min: 0.25
  size_ratio_max: 4.0
  sndlib_instance: abilene
failures:
  p_in: 0.6
  p_out: 0.05
backhaul_capacity_bps: 1000000.0
uav:
  altitude_m: 100.0
  v_max_mps: 50.0
  total_power_w: 0.1
  max_active_downlinks: 2
time:
  slot_s: 1.0
  num_slots: 150
  access_fraction: 0.5
spectrum:
  b_tot_hz: 1000000.0
  noise_psd_dbm_per_hz: -140.0
channel:
  uav:
    plos_a: 9.61
    plos_b: 0.16
    nlos_attenuation: 0.2
    alpha_los: 2.2
    alpha_nlos: 3.3
    beta0_db: -50.0
  ground:
    alpha: 2.8
    beta0_db: -50.0
  usable_rate_bps: 10000.0
source_power_w: 0.1
paths:
  k: 3
  max_ground: 2
  version: 1
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/data/test_scenario_sources.py tests/data/test_scenario_stage.py`

Expected: `12 passed`.

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `266 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/support/scenario_stage_inputs.py tests/data/test_scenario_stage.py tests/data/test_scenario_sources.py src/data/scenarios.py src/data/cli.py configs/scenarios/rescuenet.yaml
git commit -m "feat: place scenario sources on RescueNet targets"
```

### Task 3: Case-study configuration and trace

**Files:**
- Create: `configs/experiments/phase2_case_study.yaml`, `experiments/case_study_trace.py`
- Test: `tests/runner/test_case_study_trace.py`

**Interfaces:**
- Consumes: `runner.config.load_experiment_config`, `runner.methods.build_method`, `models.evaluate` (`REALIZED`, `EvaluationCounter`, `evaluate`); `results/phase2/tran_pilot/choice.json` from plan D.1; `tests/runner/experiment_fixtures.write_tiny_experiment`.
- Produces: `configs/experiments/phase2_case_study.yaml`; `experiments/case_study_trace.py` with `RATIO_TOLERANCE`, `scenario_means(rows, method) -> dict`, `representative_scenario(rows, method) -> str`, `progress_counts(realizations, num_slots) -> list[float]`, and `main(argv) -> int` (0 after writing `targets.csv`, `nodes.csv`, `edges.csv`, `trajectories.csv`, `progress.csv`, `trace.json`; 1 when a replanned timely ratio differs from the runner).

- [ ] **Step 1: Write the failing tests**

Create `tests/runner/test_case_study_trace.py`:

```python
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from experiment_fixtures import write_tiny_experiment
from optimization.bcd import SearchParams
from runner.config import load_experiment_config
from runner.experiment import run_experiment

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "case_study_trace.py"
SPEC = importlib.util.spec_from_file_location("case_study_trace", SCRIPT)
case_study_trace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(case_study_trace)


def _rows(values):
    return [{"scenario_id": scenario_id, "method": method, "timely_ratio": str(value)} for scenario_id, method, value in values]


def test_case_study_configuration_uses_the_pilot_choice():
    config = load_experiment_config(Path("configs/experiments/phase2_case_study.yaml"))
    choice = json.loads(Path("results/phase2/tran_pilot/choice.json").read_text(encoding="utf-8"))
    assert [item.set_id for item in config.scenario_sets] == ["rescuenet"] and config.split == "eval"
    assert [(spec.id, spec.base, spec.bounds) for spec in config.methods] == [
        ("B0", "B0", False), ("B1", "B1", False), ("B2", "search", True), ("P", "search", True), ("TRAN", "TRAN", True),
    ]
    assert config.methods[3].search_params() == SearchParams(operation1=True, operation2=True)
    tran = config.methods[4].tran_params()
    assert (tran.theta, tran.mu, tran.block_slots) == (choice["theta"], choice["mu"], choice["block_slots"])
    assert (config.realization_ids, config.workers, config.output_dir) == (tuple(range(30)), 12, Path("results/phase2/case_study"))


def test_representative_scenario_is_closest_to_the_median_of_scenario_means():
    rows = _rows([("A-r0", "P", 0.2), ("A-r0", "P", 0.4), ("B-r0", "P", 0.5), ("C-r0", "P", 0.9), ("B-r0", "B1", 0.0)])
    assert case_study_trace.representative_scenario(rows, "P") == "B-r0"
    assert case_study_trace.representative_scenario(_rows([("B-r0", "P", 0.75), ("A-r0", "P", 0.25)]), "P") == "A-r0"


def test_progress_counts_timely_deliveries_by_slot():
    realizations = [
        SimpleNamespace(delivery_slot={"a": 1, "b": None, "c": 3}, timely={"a": True, "b": False, "c": True}),
        SimpleNamespace(delivery_slot={"a": 0, "b": 2, "c": None}, timely={"a": True, "b": True, "c": False}),
    ]
    assert case_study_trace.progress_counts(realizations, 5) == [0.5, 1.0, 1.5, 2.0, 2.0]


def test_trace_matches_the_runner_writes_map_data_and_stops_on_a_mismatch(tmp_path: Path, capsys):
    output = tmp_path / "out"
    methods = ("B1", {"id": "TRAN", "base": "TRAN", "params": {"max_iterations": 2}})
    config_path = write_tiny_experiment(tmp_path, output, 1, methods, crosscheck=False)
    assert run_experiment(load_experiment_config(config_path), config_path) == 0
    targets = tmp_path / "targets.csv"
    targets.write_text("target_id,x_m,y_m,damage_class\nt0,1230.0,1000.0,4\nt1,100.0,100.0,5\n", encoding="utf-8")
    trace = tmp_path / "trace"
    arguments = ["--config", str(config_path), "--results", str(output), "--targets", str(targets), "--output", str(trace),
                 "--methods", "B1", "TRAN", "--anchor", "B1"]
    assert case_study_trace.main(arguments) == 0
    summary = json.loads((trace / "trace.json").read_text(encoding="utf-8"))
    assert sorted(summary["methods"]) == ["B1", "TRAN"] and summary["realizations"] == 2
    counts = {}
    for name in ("targets", "nodes", "edges", "trajectories", "progress"):
        with (trace / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
            counts[name] = len(list(csv.DictReader(stream)))
    assert counts == {"targets": 2, "nodes": 4, "edges": 3, "trajectories": 42, "progress": 40}
    with (output / "method_realizations.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["timely_ratio"] = "0.123" if row["method"] == "B1" else row["timely_ratio"]
    with (output / "method_realizations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert case_study_trace.main(arguments) == 1
    assert "trace differs from the runner for ['B1']" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/runner/test_case_study_trace.py`

Expected: collection fails (`1 error`) with `FileNotFoundError: [Errno 2] No such file or directory: 'experiments/case_study_trace.py'`.

- [ ] **Step 3: Implement**

Create `configs/experiments/phase2_case_study.yaml`:

```yaml
experiment_id: phase2-case-study
scenario_sets:
  - set_id: rescuenet
    manifest: data/manifests/scenarios_rescuenet.csv
    scenario_dir: data/processed/scenarios/rescuenet
split: eval
realization_ids: {start: 0, stop: 30}
methods:
  - B0
  - B1
  - {id: B2, base: search, bounds: true}
  - {id: P, base: search, params: {operation1: true, operation2: true}, bounds: true}
  - {id: TRAN, base: TRAN, params: {theta: 0.3333333333333333, mu: 10.0, block_slots: 1}, bounds: true}
bounds: {tangents: 10}
bootstrap: {resamples: 10000, seed: 20260911, confidence: 0.95}
workers: 12
output_dir: results/phase2/case_study
```

Create `experiments/case_study_trace.py`:

```python
"""Trace of the representative case-study scenario (spec D Section 10.2): map and delivery-progress data.

The representative scenario has the anchor method's mean timely ratio closest to the median over scenarios (ties:
smallest scenario id). Each traced method is planned again and evaluated on the experiment's realizations; the
script stops without writing when a timely ratio differs from the runner's results.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Mapping, Sequence

from models.evaluate import REALIZED, EvaluationCounter, evaluate
from models.paths import candidate_paths
from models.scenario import load_scenario
from runner.config import load_experiment_config
from runner.methods import build_method

RATIO_TOLERANCE = 1e-12
RULE = "anchor mean timely ratio closest to the median over scenarios; ties go to the smallest scenario_id"


def scenario_means(rows: Sequence[Mapping[str, str]], method: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            values[row["scenario_id"]].append(float(row["timely_ratio"]))
    return {scenario_id: statistics.fmean(items) for scenario_id, items in sorted(values.items())}


def representative_scenario(rows: Sequence[Mapping[str, str]], method: str) -> str:
    means = scenario_means(rows, method)
    if not means:
        raise ValueError(f"no result rows for method {method}")
    median = statistics.median(means.values())
    return min(means, key=lambda scenario_id: (abs(means[scenario_id] - median), scenario_id))


def progress_counts(realizations: Sequence[object], num_slots: int) -> list[float]:
    """Mean over realizations of the alerts delivered on time by the end of each slot."""
    totals = [0.0] * num_slots
    for realization in realizations:
        for alert_id, slot in realization.delivery_slot.items():
            if slot is not None and realization.timely[alert_id]:
                for index in range(slot, num_slots):
                    totals[index] += 1.0
    return [total / len(realizations) for total in totals]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="case-study-trace")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/phase2_case_study.yaml"))
    parser.add_argument("--results", type=Path, default=Path("results/phase2/case_study"))
    parser.add_argument("--targets", type=Path, default=Path("data/manifests/rescuenet_targets.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2/case_study/trace"))
    parser.add_argument("--methods", nargs="+", default=["B1", "P", "TRAN"])
    parser.add_argument("--anchor", default="P")
    args = parser.parse_args(argv)
    config = load_experiment_config(args.config)
    with (args.results / "method_realizations.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    scenario_id = representative_scenario(rows, args.anchor)
    scenario = load_scenario(config.scenario_sets[0].scenario_dir / f"{scenario_id}.json")
    with args.targets.open(newline="", encoding="utf-8") as stream:
        targets = list(csv.DictReader(stream))
    if len(targets) != len(scenario.sources):
        raise ValueError(f"{args.targets}: {len(targets)} targets for {len(scenario.sources)} sources of {scenario_id}")
    candidates = candidate_paths(scenario)
    specs = {spec.id: spec for spec in config.methods}
    scenario_rows = [row for row in rows if row["scenario_id"] == scenario_id]
    checks, trajectories, progress = {}, [], []
    for method_id in args.methods:
        plan = build_method(specs[method_id]).plan(scenario, candidates, EvaluationCounter())
        result = evaluate(scenario, plan, REALIZED, config.realization_ids, candidates=candidates)
        checks[method_id] = {"timely_ratio_trace": result.timely_ratio, "timely_ratio_runner": scenario_means(scenario_rows, method_id)[scenario_id]}
        trajectories += [{"method": method_id, "slot": slot, "x_m": float(x), "y_m": float(y)} for slot, (x, y) in enumerate(plan.trajectory)]
        counts = progress_counts(result.realizations, scenario.time.num_slots)
        progress += [{"method": method_id, "slot": slot, "mean_timely_alerts": value} for slot, value in enumerate(counts)]
    differing = sorted(method for method, check in checks.items() if abs(check["timely_ratio_trace"] - check["timely_ratio_runner"]) > RATIO_TOLERANCE)
    if differing:
        print(f"trace differs from the runner for {differing}: {checks}", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    zone_of = {source.id: source.zone for source in scenario.sources}
    _write(args.output / "targets.csv", [
        {"source_id": source.id, "target_id": target["target_id"], "x_m": source.x_m, "y_m": source.y_m,
         "damage_class": target["damage_class"], "zone": zone_of[source.id]}
        for source, target in zip(scenario.sources, targets)
    ])
    _write(args.output / "nodes.csv", [
        {"node_id": node.id, "x_m": node.x_m, "y_m": node.y_m, "role": "center" if node.id == scenario.center else "relay"}
        for node in scenario.nodes
    ])
    _write(args.output / "edges.csv", [
        {"source": edge.source, "target": edge.target, "source_x_m": scenario.node_by_id[edge.source].x_m,
         "source_y_m": scenario.node_by_id[edge.source].y_m, "target_x_m": scenario.node_by_id[edge.target].x_m,
         "target_y_m": scenario.node_by_id[edge.target].y_m, "failed": int(edge.failed)}
        for edge in scenario.edges
    ])
    _write(args.output / "trajectories.csv", trajectories)
    _write(args.output / "progress.csv", progress)
    summary = {"scenario_id": scenario_id, "anchor": args.anchor, "rule": RULE, "realizations": len(config.realization_ids), "methods": checks}
    (args.output / "trace.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"scenario_id": scenario_id, "methods": sorted(checks), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q tests/runner/test_case_study_trace.py`

Expected: `4 passed`.

- [ ] **Step 5: Run the full suite**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run --extra test pytest -q`

Expected: `270 passed, 1 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/runner/test_case_study_trace.py configs/experiments/phase2_case_study.yaml experiments/case_study_trace.py
git commit -m "feat: add the case-study experiment and trace"
```

### Task 4: Targets and case-study scenarios

**Files:**
- Create (generated): `data/manifests/rescuenet_targets.csv`, `data/manifests/rescuenet_targets.json`, `data/manifests/scenarios_rescuenet.csv`
- Regenerate (ignored): `data/processed/scenarios/v0/`, `data/processed/scenarios/rescuenet/`

**Interfaces:**
- Consumes: Tasks 1–2; `data/raw/rescuenet-validation/extracted/`, `data/raw/rescuenet-descriptor/`, the Topology Zoo networks and the SNDlib archive under `data/`.
- Produces: the target manifests and the `rescuenet` scenario set for Task 5 (acceptance criterion 16.2.2).

- [ ] **Step 1: Extract the targets**

Run: `uv run python -m data.cli rescuenet-targets --config configs/data/rescuenet_targets.yaml`

Expected: `{"regions": 68, "targets": 63, "output": "data/manifests/rescuenet_targets.csv"}` after about a minute.

- [ ] **Step 2: Check the target manifests**

````bash
uv run python - <<'PY'
import hashlib
import json
from pathlib import Path

import yaml

csv_path, json_path = Path("data/manifests/rescuenet_targets.csv"), Path("data/manifests/rescuenet_targets.json")
expected = yaml.safe_load(Path("configs/scenarios/rescuenet.yaml").read_text(encoding="utf-8"))["workload"]["targets_manifest_sha256"]
summary = json.loads(json_path.read_text(encoding="utf-8"))
print("csv sha256 matches rescuenet.yaml:", hashlib.sha256(csv_path.read_bytes()).hexdigest() == expected)
print("json sha256", hashlib.sha256(json_path.read_bytes()).hexdigest())
print("regions", summary["region_count"], "targets", summary["target_count"], "dropped", len(summary["dropped_regions"]))
print("merged", len(summary["merged_pairs"]), "max distance", max(pair["distance_m"] for pair in summary["merged_pairs"]))
print("fallback images", [image["pair_id"] for image in summary["images"] if image["focal_source"] == "fallback"])
print("scale", summary["normalization"]["scale"], "offset_y_m", summary["normalization"]["offset_y_m"])
PY
````

Expected:

```text
csv sha256 matches rescuenet.yaml: True
json sha256 ccaef0affcb8e9f03a0db29022692f31daf2313e5f5778dba924974b6410aff4
regions 68 targets 63 dropped 0
merged 5 max distance 8.209
fallback images ['15142']
scale 0.693323 offset_y_m 488.196973
```

If the CSV hash does not match the value recorded in `configs/scenarios/rescuenet.yaml`, stop and report the first differing rows against the prototype values of spec Section 17 (Plan D.2 adjustments): the scenarios would not be reproducible.

- [ ] **Step 3: Regenerate `v0` and generate `rescuenet`**

Run:

```bash
uv run python -m data.cli scenarios --config configs/scenarios/v0.yaml
uv run python -m data.cli scenarios --config configs/scenarios/rescuenet.yaml
git status --short data/manifests/scenarios_v0.csv
```

Expected: both commands exit with 0 and `git status` prints nothing, because the synthetic manifest is unchanged.

- [ ] **Step 4: Check the scenario sets**

````bash
uv run python - <<'PY'
import csv
import hashlib
import json
from pathlib import Path
import statistics

manifests = Path("data/manifests")
print("v0 manifest sha256", hashlib.sha256((manifests / "scenarios_v0.csv").read_bytes()).hexdigest())
print("rescuenet manifest sha256", hashlib.sha256((manifests / "scenarios_rescuenet.csv").read_bytes()).hexdigest())
with (manifests / "scenarios_rescuenet.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
with (manifests / "scenarios_v0.csv").open(newline="", encoding="utf-8") as stream:
    v0_ids = sorted(row["scenario_id"] for row in csv.DictReader(stream))
print("rows", len(rows), "eval", sum(row["split"] == "eval" for row in rows), "same scenario ids as v0:", sorted(row["scenario_id"] for row in rows) == v0_ids)
shapes = set()
for row in rows:
    scenario = json.loads(Path(f"data/processed/scenarios/rescuenet/{row['scenario_id']}.json").read_text(encoding="utf-8"))
    shapes.add((len(scenario["sources"]), len(scenario["damage_zones"]), "targets_manifest_sha256" in scenario["provenance"]))
print("sources, zones, target provenance per scenario:", sorted(shapes))
fractions = [float(row["ground_connected_source_fraction"]) for row in rows if row["split"] == "eval"]
print("eval ground-connected fraction", round(statistics.fmean(fractions), 3))
PY
````

Expected:

```text
v0 manifest sha256 7f2325b4e955f3c6265e01c10203a7d90244490446ed22015321e707e818ce17
rescuenet manifest sha256 bb828acbae13a70290ab6b204a6b811efde9f7d3d5380d0a7bc3869937840ead
rows 36 eval 30 same scenario ids as v0: True
sources, zones, target provenance per scenario: [(63, 3, True)]
eval ground-connected fraction 0.376
```

- [ ] **Step 5: Confirm the new files, then commit**

Run: `git status --short --untracked-files=all data/manifests`

Expected: exactly the three manifests listed under **Files**.

```bash
git add data/manifests/rescuenet_targets.csv data/manifests/rescuenet_targets.json data/manifests/scenarios_rescuenet.csv
git commit -m "feat: record RescueNet targets and case-study scenarios"
```

### Task 5: Case-study experiment

**Files:**
- Create (generated): `results/phase2/case_study/method_realizations.csv`, `results/phase2/case_study/bounds.csv`, `results/phase2/case_study/summary.csv`, `results/phase2/case_study/manifest.json`

**Interfaces:**
- Consumes: Tasks 3–4; the TRAN method of plan D.1.
- Produces: case-study results for Task 6 and plan D.3 (acceptance criterion 16.2.3).

- [ ] **Step 1: Run the experiment**

Run: `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/run_phase2.py --config configs/experiments/phase2_case_study.yaml`

Expected: `{"task_count": 1950, "executed_task_count": 1950, "skipped_task_count": 0, "status": "complete"}` and exit code 0, after roughly an hour with 12 workers. Each of the 30 scenarios has five method tasks and 60 bound tasks (B0 ground-only and B1, 30 realizations each); B2, P and TRAN solve their 30 bounds inside their method tasks. Measured alone on `Agis-r0`, P planned in 60 s and TRAN in 179 s with peaks of 118 MB and 373 MB. If the status is `failed`, stop and report the failures printed by the runner.

- [ ] **Step 2: Check the results**

````bash
uv run python - results/phase2/case_study <<'PY'
import csv
import json
import sys

root = sys.argv[1]


def rows(name):
    with open(f"{root}/{name}", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


manifest = json.load(open(f"{root}/manifest.json", encoding="utf-8"))
methods, bounds, summary = rows("method_realizations.csv"), rows("bounds.csv"), rows("summary.csv")
limits = {"B0": 1, "B1": 1, "TRAN": 1, "B2": 2001, "P": 2001}
print(manifest["status"], manifest["task_count"], "failures", len(manifest["failures"]))
print(len(methods), "method rows;", len(bounds), "bound rows; statuses", sorted({row["status"] for row in bounds}))
print("bounded trajectories", sorted({row["trajectory_method"] or "ground" for row in bounds}))
print(len(summary), "summary rows; evaluate calls within limits:", all(int(row["evaluate_calls"]) <= limits[row["method"]] for row in methods))
PY
````

Expected:

```text
complete 1950 failures 0
4500 method rows; 4500 bound rows; statuses ['optimal']
bounded trajectories ['B1', 'B2', 'P', 'TRAN', 'ground']
5 summary rows; evaluate calls within limits: True
```

- [ ] **Step 3: Confirm that a rerun skips every task**

Run: `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/run_phase2.py --config configs/experiments/phase2_case_study.yaml`

Expected: `{"task_count": 1950, "executed_task_count": 0, "skipped_task_count": 1950, "status": "complete"}`.

- [ ] **Step 4: Confirm the new files, then commit**

Run: `git status --short --untracked-files=all results/phase2/case_study`

Expected: exactly the four files listed under **Files**; no shard appears.

```bash
git add results/phase2/case_study/method_realizations.csv results/phase2/case_study/bounds.csv results/phase2/case_study/summary.csv results/phase2/case_study/manifest.json
git commit -m "feat: record case-study results"
```

### Task 6: Trace of the representative scenario

**Files:**
- Create (generated): `results/phase2/case_study/trace/targets.csv`, `nodes.csv`, `edges.csv`, `trajectories.csv`, `progress.csv`, `trace.json`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 3–5.
- Produces: map and delivery-progress data for the case-study figures of plan D.3 (spec Section 10.2, acceptance criterion 16.2.3).

- [ ] **Step 1: Run the trace**

Run: `systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run python experiments/case_study_trace.py`

Expected: exit code 0 and one JSON line with the representative `scenario_id`, `"methods": ["B1", "P", "TRAN"]` and `"output": "results/phase2/case_study/trace"`, after about ten minutes (P and TRAN are planned again). If the script exits with 1 and prints "trace differs from the runner", stop and report: replanning did not reproduce the committed results.

- [ ] **Step 2: Check the trace**

````bash
uv run python - <<'PY'
import csv
import json
from pathlib import Path

trace = Path("results/phase2/case_study/trace")
summary = json.loads((trace / "trace.json").read_text(encoding="utf-8"))
counts = {}
for name in ("targets", "nodes", "edges", "trajectories", "progress"):
    with (trace / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
        counts[name] = len(list(csv.DictReader(stream)))
scenario = json.loads(Path(f"data/processed/scenarios/rescuenet/{summary['scenario_id']}.json").read_text(encoding="utf-8"))
print("scenario", summary["scenario_id"], "realizations", summary["realizations"])
print("targets", counts["targets"], "trajectories", counts["trajectories"], "progress", counts["progress"])
print("nodes and edges match the scenario:", (counts["nodes"], counts["edges"]) == (len(scenario["network"]["nodes"]), len(scenario["network"]["edges"])))
for method, check in sorted(summary["methods"].items()):
    print(method, f"timely {check['timely_ratio_trace']:.3f}", "matches runner:", abs(check["timely_ratio_trace"] - check["timely_ratio_runner"]) <= 1e-12)
PY
````

Expected: the first line names the representative scenario with `realizations 30`, then `targets 63 trajectories 453 progress 450`, `nodes and edges match the scenario: True`, and one line each for B1, P and TRAN ending in `matches runner: True`. The scenario id and the three timely ratios follow from the Task 5 results.

- [ ] **Step 3: Document the commands in the README**

Append to the end of `README.md`:

````markdown

## Pha 2: case study RescueNet

Target được lấy từ mask công trình hư hại (lớp 3–5) của 30 cặp ảnh RescueNet đã chọn. Vị trí target được chiếu sang UTM 16N bằng GPS, độ cao tương đối, hướng gimbal và tiêu cự trong metadata DJI của ảnh; các target trùng trong 10 m được gộp rồi chuẩn hóa về hộp 2 km (`src/data/rescuenet_targets.py`, spec D Mục 8). Ảnh và mask không được phân phối lại; kho chỉ lưu manifest target.

```bash
uv run python -m data.cli rescuenet-targets --config configs/data/rescuenet_targets.yaml
uv run python -m data.cli scenarios --config configs/scenarios/rescuenet.yaml
```

Họ scenario `rescuenet` giống `v0` nhưng đặt mỗi nguồn tại một target, với ba vùng hư hại là ba cụm k-means. Chạy B0, B1, B2, P và TRAN trên 30 scenario đánh giá, rồi lập lại kế hoạch cho scenario đại diện để có bản đồ target–relay–quỹ đạo và tiến trình giao cảnh báo:

```bash
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/run_phase2.py --config configs/experiments/phase2_case_study.yaml
systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run python experiments/case_study_trace.py
```

Kết quả nằm trong `results/phase2/case_study/` và `results/phase2/case_study/trace/`.
````

- [ ] **Step 4: Confirm the changed files, then commit**

Run: `git status --short --untracked-files=all results/phase2/case_study README.md`

Expected: the six trace files as new and `README.md` as modified.

```bash
git add results/phase2/case_study/trace/targets.csv results/phase2/case_study/trace/nodes.csv results/phase2/case_study/trace/edges.csv results/phase2/case_study/trace/trajectories.csv results/phase2/case_study/trace/progress.csv results/phase2/case_study/trace/trace.json README.md
git commit -m "feat: record the case-study trace"
```

## Spec Coverage

| Spec section | Task |
|---|---|
| 8.1 Configuration | 1 |
| 8.2 Extraction and georeferencing | 1 |
| 8.3 Outputs | 1 (command), 4 (manifests) |
| 9 Source placement interface, `rescuenet.yaml`, validation errors | 2, 4 |
| 10.1 Experiment | 3 (configuration), 5 |
| 10.2 Trace | 3, 6 |
| 14 Error handling: targets, scenarios, trace | 1, 2, 3 |
| 15.2 Testing | 1–3 |
| 16.2.1 Full test suite | every code task |
| 16.2.2 Target and scenario manifests committed; `scenarios_v0.csv` unchanged | 2 (pinned synthetic scenario), 4 |
| 16.2.3 Experiment complete, LPs optimal, results and trace committed | 5, 6 |
| 17 Plan D.2 adjustments | 1 (camera fallback width), 2 (stage-test helpers), 3 (trace options), 4 (measured values) |
