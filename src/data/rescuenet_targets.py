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
