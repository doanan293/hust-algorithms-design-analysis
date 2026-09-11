from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import re
from typing import Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RescuePair:
    pair_id: str
    image: Path
    mask: Path


@dataclass(frozen=True)
class RescueRecord:
    pair_id: str
    image: Path
    mask: Path
    class_counts: tuple[tuple[int, int], ...]
    presence_vector: tuple[int, ...]


def load_label_map(path: Path) -> dict[int, str]:
    text = path.read_text(encoding="utf-8")
    labels: dict[int, str] = {}
    for name, value in re.findall(r"'([^']+)'\s*:\s*(\d+)", text):
        numeric = int(value)
        if numeric in labels and labels[numeric] != name:
            raise ValueError(f"duplicate label ID: {numeric}")
        labels[numeric] = name
    if not labels:
        raise ValueError(f"no labels found in {path}")
    return labels


def _index_by_stem(
    directory: Path, suffixes: set[str], stem_suffix: str = ""
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            stem = path.stem
            if stem_suffix and stem.endswith(stem_suffix):
                stem = stem[: -len(stem_suffix)]
            key = stem.casefold()
            if key in result:
                raise ValueError(f"duplicate canonical stem: {key}")
            result[key] = path
    return result


def pair_images_and_masks(root: Path) -> tuple[RescuePair, ...]:
    if (root / "val-org-img").is_dir() and (root / "val-label-img").is_dir():
        image_dir, mask_dir, mask_suffix = (
            root / "val-org-img",
            root / "val-label-img",
            "_lab",
        )
    else:
        image_dir, mask_dir, mask_suffix = root / "images", root / "masks", ""
    images = _index_by_stem(image_dir, {".jpg", ".jpeg", ".png"})
    masks = _index_by_stem(mask_dir, {".png"}, mask_suffix)
    if images.keys() != masks.keys():
        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())
        raise ValueError(f"image-mask mismatch: masks={missing_masks}, images={missing_images}")
    return tuple(RescuePair(key, images[key], masks[key]) for key in sorted(images))


def count_mask_classes(pair: RescuePair, label_map: Mapping[int, str]) -> Mapping[int, int]:
    values, counts = np.unique(np.asarray(Image.open(pair.mask)), return_counts=True)
    unknown = sorted(set(map(int, values)) - set(label_map))
    if unknown:
        raise ValueError(f"unknown mask labels for {pair.pair_id}: {unknown}")
    return {int(value): int(count) for value, count in zip(values, counts, strict=True)}


def select_pairs(
    records: Sequence[RescueRecord], count: int, seed: int
) -> tuple[RescueRecord, ...]:
    if len(records) < count:
        raise ValueError(f"need {count} valid pairs, found {len(records)}")
    groups: dict[tuple[int, ...], list[RescueRecord]] = {}
    for record in sorted(records, key=lambda item: item.pair_id):
        groups.setdefault(record.presence_vector, []).append(record)
    for key, values in groups.items():
        digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(values)
    selected: list[RescueRecord] = []
    while len(selected) < count:
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
    return tuple(selected)
