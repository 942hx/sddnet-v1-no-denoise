"""MILL triplets: different-level positives and same-level/different-scene negatives."""

from __future__ import annotations

import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from .common import load_rgb, mill_records, transform_pair


class MILLTripletDataset(Dataset):
    """Normalize raw levels using fixed configured bounds, never per split.

    Anchor and positive share crop coordinates so their scene content matches.
    Each negative uses an independent crop of a different scene. No photometric
    augmentation is used because that would invalidate illumination labels.
    """

    def __init__(self, root, split="train", patch_size=256, metadata=None,
                 illum_min=1.0, illum_max=10.0, augment=True):
        if patch_size < 1 or illum_max <= illum_min:
            raise ValueError("patch_size must be positive and illum_max > illum_min.")
        self.records = mill_records(root, split, metadata)
        self.patch_size = int(patch_size)
        self.illum_min, self.illum_max = float(illum_min), float(illum_max)
        self.augment = augment
        self.by_scene = defaultdict(list)
        for index, record in enumerate(self.records):
            if not illum_min <= record.illum_level <= illum_max:
                raise ValueError(f"{record.filename}: level {record.illum_level} outside [{illum_min}, {illum_max}].")
            self.by_scene[record.scene_id].append(index)
        self.scenes = sorted(self.by_scene)
        self.scene_index = {name: index for index, name in enumerate(self.scenes)}
        if len(self.scenes) < 2:
            raise ValueError("Triplets need at least two distinct scenes.")
        self.positives = {}
        self.negatives = {}
        for index, record in enumerate(self.records):
            candidates = [i for i in self.by_scene[record.scene_id]
                          if self.records[i].illum_level != record.illum_level]
            if not candidates:
                raise ValueError(f"Scene {record.scene_id!r} needs at least two different illumination levels.")
            self.positives[index] = candidates
            negatives = [i for i, other in enumerate(self.records)
                         if other.scene_id != record.scene_id and other.illum_level == record.illum_level]
            if not negatives:
                raise ValueError(f"No different-scene negative at level {record.illum_level} for {record.filename}.")
            self.negatives[index] = negatives

    def __len__(self):
        return len(self.records)

    def sample_indices(self, index):
        """Sample only explicitly valid positive and same-level negative records."""
        positive = random.choice(self.positives[index])
        negative = random.choice(self.negatives[index])
        return index, positive, negative

    def __getitem__(self, index):
        indices = self.sample_indices(index)
        records = [self.records[i] for i in indices]
        images = [(load_rgb(r.input_path), load_rgb(r.gt_path)) for r in records]
        if images[0][0].size != images[1][0].size:
            raise ValueError("Anchor/positive must be registered images with matching dimensions.")
        w, h = images[0][0].size
        crop = (random.randint(0, max(0, h - self.patch_size)),
                random.randint(0, max(0, w - self.patch_size)))
        augment = tuple(random.random() < 0.5 for _ in range(3)) if self.augment else (False,) * 3
        result = {}
        for position, (name, record, pair) in enumerate(zip(("anchor", "positive", "negative"), records, images)):
            pair_augment = (tuple(random.random() < 0.5 for _ in range(3))
                            if self.augment and position == 2 else augment)
            low, gt = transform_pair(*pair, patch_size=self.patch_size,
                                     crop=crop if position < 2 else None, augment=pair_augment)
            result[f"{name}_low"] = low
            result[f"{name}_gt"] = gt
            level = (record.illum_level - self.illum_min) / (self.illum_max - self.illum_min)
            result[f"{name}_level"] = torch.tensor([level], dtype=torch.float32)
        return result
