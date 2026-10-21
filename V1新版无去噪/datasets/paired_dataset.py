"""Native-resolution paired evaluation for MILL, LOLv1, and both LOLv2 subsets."""

from __future__ import annotations

from pathlib import Path
import csv

from torch.utils.data import Dataset

from .common import image_files, load_rgb, mill_records, stem_index, transform_pair


class PairedDataset(Dataset):
    """Common output: low [3,H,W], gt [3,H,W], and source filename."""

    def __init__(self, pairs):
        self.pairs = pairs
        if not pairs:
            raise ValueError("Paired dataset is empty.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        low_path, gt_path = self.pairs[index]
        low, gt = transform_pair(load_rgb(low_path), load_rgb(gt_path))
        return {"low": low, "gt": gt, "filename": low_path.name}


class ManifestPairedDataset(PairedDataset):
    """Use audited rows, including explicit scene/level fields for evaluation."""

    def __init__(self, root, metadata, split):
        with Path(metadata).open(encoding="utf-8-sig", newline="") as stream:
            self.rows = [r for r in csv.DictReader(stream) if r["split"] == split
                         and str(r.get("valid", "true")).lower() in ("true", "1")]
        root = Path(root)
        super().__init__([(root / r["input_path"], root / r["gt_path"]) for r in self.rows])


def pair_directories(low_dir: Path, gt_dir: Path):
    """Pair by exact, case-insensitive stems, never by sorted list positions."""
    gt_index = stem_index(gt_dir)
    pairs = []
    for low in image_files(low_dir):
        gt = gt_index.get(low.stem.casefold())
        if gt is None:
            raise FileNotFoundError(f"Missing GT with stem {low.stem!r} in {gt_dir}")
        pairs.append((low, gt))
    return pairs


class MILLPairedDataset(PairedDataset):
    """Default MILL test split; all available levels are evaluated."""

    def __init__(self, root, split="test", metadata=None):
        records = mill_records(root, split, metadata)
        super().__init__([(r.input_path, r.gt_path) for r in records])


class LOLv1PairedDataset(PairedDataset):
    """LOL-v1 our485 training pairs or eval15 test pairs."""

    def __init__(self, root, split="test"):
        split_map = {"train": "our485", "test": "eval15", "our485": "our485", "eval15": "eval15"}
        if split not in split_map:
            raise ValueError(f"Unknown LOLv1 split: {split}")
        folder = Path(root) / split_map[split]
        super().__init__(pair_directories(folder / "low", folder / "high"))


class LOLv2PairedDataset(PairedDataset):
    """LOL-v2 Real_captured or Synthetic with explicit Train/Test selection."""

    def __init__(self, root, subset="real", split="test"):
        subsets = {"real": "Real_captured", "synthetic": "Synthetic"}
        if subset.lower() not in subsets or split.lower() not in ("train", "test"):
            raise ValueError("LOLv2 requires subset=real/synthetic and split=train/test.")
        folder = Path(root) / subsets[subset.lower()] / split.title()
        super().__init__(pair_directories(folder / "Low", folder / "Normal"))
