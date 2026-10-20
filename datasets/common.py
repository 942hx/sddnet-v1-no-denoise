"""Strict image pairing, metadata validation, and synchronized RGB transforms."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FIELDS = ["filename", "split", "scene_id", "illum_level", "input_path", "gt_path"]
MILL_PATTERN = re.compile(r"^(?P<scene>.+)-(?P<level>10|[1-9])$")
# Official corrupt MILL-small sample; never delete it if present in source data.
MILL_EXCLUSIONS = {("test", "test13-9.png"): "Official MILL exclusion: corrupt Test13-9.png"}


@dataclass(frozen=True)
class MILLRecord:
    """Paths are absolute at runtime; illumination is the raw metadata value."""

    filename: str
    split: str
    scene_id: str
    illum_level: float
    input_path: Path
    gt_path: Path


def image_files(directory: Path) -> list[Path]:
    """List supported images deterministically and reject empty/missing folders."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files = sorted((p for p in directory.iterdir() if p.is_file() and not p.name.startswith((".", "_"))
                    and p.suffix.lower() in IMAGE_EXTENSIONS),
                   key=lambda p: p.name.casefold())
    if not files:
        raise ValueError(f"No supported images in {directory}")
    return files


def stem_index(directory: Path) -> dict[str, Path]:
    """Never silently pair duplicate stems such as image.png and image.jpg."""
    index = {}
    for path in image_files(directory):
        key = path.stem.casefold()
        if key in index:
            raise ValueError(f"Ambiguous image stem: {index[key]} and {path}")
        index[key] = path
    return index


def parse_mill_filename(filename: str) -> tuple[str, float]:
    """Scene1_a-10.png -> ('Scene1_a', 10); keep scene suffixes intact."""
    match = MILL_PATTERN.fullmatch(Path(filename).stem)
    if match is None:
        raise ValueError(f"Cannot parse scene/illumination from {filename!r}; use an edited metadata CSV/JSON.")
    return match.group("scene"), float(match.group("level"))


def scan_mill(root: Path, split: str) -> list[MILLRecord]:
    """Match each low image to its exact scene GT; GT may be shared by levels."""
    gt_index = stem_index(root / split / "gt")
    records = []
    for low in image_files(root / split / "input"):
        if (split, low.name.casefold()) in MILL_EXCLUSIONS:
            continue
        scene, level = parse_mill_filename(low.name)
        gt = gt_index.get(scene.casefold())
        if gt is None:
            raise FileNotFoundError(f"No GT for {low.name}; expected scene stem {scene!r}. Use metadata if needed.")
        records.append(MILLRecord(low.name, split, scene, level, low, gt))
    return records


def read_mill_metadata(path: Path, root: Path, split: str) -> list[MILLRecord]:
    """Relative input_path/gt_path entries are relative to the MILL dataset root."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        if Path(path).suffix.lower() == ".json":
            rows = json.load(stream)
        elif Path(path).suffix.lower() == ".csv":
            rows = list(csv.DictReader(stream))
        else:
            raise ValueError("Metadata must be .csv or .json.")
    if not isinstance(rows, list):
        raise ValueError("JSON metadata must be a list of row objects.")
    records, seen = [], set()
    for line, row in enumerate(rows, start=2):
        if not isinstance(row, dict) or any(field not in row for field in FIELDS):
            raise ValueError(f"Metadata row {line}: required fields are {FIELDS}")
        if str(row["split"]).strip() != split:
            continue
        if (split, str(row["filename"]).casefold()) in MILL_EXCLUSIONS:
            continue
        scene = str(row["scene_id"]).strip() if row["scene_id"] is not None else ""
        try:
            level = float(row["illum_level"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Metadata row {line}: set a numeric illum_level.") from error
        if not scene or not math.isfinite(level):
            raise ValueError(f"Metadata row {line}: scene_id must be non-empty and illum_level finite.")
        paths = []
        for field in ("input_path", "gt_path"):
            value = str(row[field]).strip() if row[field] is not None else ""
            if not value:
                raise ValueError(f"Metadata row {line}: missing {field}")
            candidate = Path(value.replace("\\", "/"))
            candidate = candidate if candidate.is_absolute() else root / candidate
            if not candidate.is_file():
                raise FileNotFoundError(f"Metadata row {line}: {candidate}")
            paths.append(candidate)
        key = str(paths[0].absolute()).casefold()
        if key in seen:
            raise ValueError(f"Duplicate input in metadata: {paths[0]}")
        seen.add(key)
        filename = str(row["filename"]).strip()
        if filename != paths[0].name:
            raise ValueError(f"Metadata row {line}: filename must equal input_path's basename.")
        records.append(MILLRecord(filename, split, scene, level, *paths))
    if not records:
        raise ValueError(f"No metadata rows for split={split!r} in {path}")
    return records


def mill_records(root: Path, split: str, metadata=None) -> list[MILLRecord]:
    """Use explicit labels when supplied; otherwise parse supported filenames."""
    root = Path(root).absolute()
    return read_mill_metadata(Path(metadata), root, split) if metadata else scan_mill(root, split)


def write_mill_metadata(root: Path, output: Path, splits, allow_unparsed=False) -> int:
    """Generate labels or an editable template without guessing unknown scenes/levels."""
    root, output = Path(root).absolute(), Path(output)
    rows = []
    for split in splits:
        gt_index = stem_index(root / split / "gt")
        for low in image_files(root / split / "input"):
            if (split, low.name.casefold()) in MILL_EXCLUSIONS:
                continue
            try:
                scene, level = parse_mill_filename(low.name)
            except ValueError:
                if not allow_unparsed:
                    raise
                scene, level = "", ""
            gt = gt_index.get(scene.casefold()) if scene else gt_index.get(low.stem.casefold())
            if gt is None and not allow_unparsed:
                raise FileNotFoundError(f"Missing scene GT for {low}; use --allow-unparsed to make a template.")
            rows.append(dict(filename=low.name, split=split, scene_id=scene, illum_level=level,
                             input_path=low.relative_to(root).as_posix(),
                             gt_path=gt.relative_to(root).as_posix() if gt else ""))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        if output.suffix.lower() == ".csv":
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        elif output.suffix.lower() == ".json":
            json.dump(rows, stream, ensure_ascii=False, indent=2)
        else:
            raise ValueError("Output metadata extension must be .csv or .json.")
    return len(rows)


def load_rgb(path: Path) -> Image.Image:
    """Decode and close file handles immediately, including on Windows workers."""
    with Path(path).open("rb") as stream, Image.open(stream) as image:
        return image.convert("RGB")


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert 8-bit RGB to contiguous CHW float32 [0, 1]."""
    return torch.from_numpy(np.array(image, dtype=np.float32, copy=True).transpose(2, 0, 1).copy()).div_(255.0)


def pad_for_crop(image: Image.Image, size: int) -> Image.Image:
    """Edge padding avoids rejecting training images smaller than the patch."""
    w, h = image.size
    if w >= size and h >= size:
        return image
    array = np.pad(np.asarray(image), ((0, max(0, size - h)), (0, max(0, size - w)), (0, 0)), mode="edge")
    return Image.fromarray(array)


def transform_pair(low, gt, patch_size=None, crop=None, augment=(False, False, False)):
    """Apply one crop and geometric augmentation to both members of a pair."""
    if low.size != gt.size:
        raise ValueError(f"Low/GT dimensions differ: {low.size} versus {gt.size}; alignment is required.")
    if patch_size:
        low, gt = pad_for_crop(low, patch_size), pad_for_crop(gt, patch_size)
        w, h = low.size
        top, left = crop if crop is not None else (random.randint(0, h - patch_size), random.randint(0, w - patch_size))
        box = (left, top, left + patch_size, top + patch_size)
        low, gt = low.crop(box), gt.crop(box)
    low, gt = to_tensor(low), to_tensor(gt)
    horizontal, vertical, transpose = augment
    if horizontal:
        low, gt = low.flip(-1), gt.flip(-1)
    if vertical:
        low, gt = low.flip(-2), gt.flip(-2)
    if transpose:
        low, gt = low.transpose(-1, -2), gt.transpose(-1, -2)
    return low.contiguous(), gt.contiguous()
