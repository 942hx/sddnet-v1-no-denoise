"""Verify gradients, exact loss weighting, pairing, crop alignment, and checkpoint RNG."""

from __future__ import annotations

import csv
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from datasets import MILLPairedDataset, MILLTripletDataset, LOLv1PairedDataset, LOLv2PairedDataset
from datasets.common import parse_mill_filename, read_mill_metadata, write_mill_metadata
from losses import SDDLoss, ssim
from models import SDDNetV1
from scripts.train_mill import concatenate_triplets
from utils.inference import enhance_image
from utils.runtime import (PROJECT_ROOT, capture_rng, load_checkpoint, restore_rng,
                           save_checkpoint, seed_everything)


class ModelTests(unittest.TestCase):
    """A real backward pass catches disconnected branches and inplace errors."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_forward_losses_and_all_parameter_gradients(self):
        torch.manual_seed(7)
        model = SDDNetV1()
        shapes = {}
        handles = []
        for name in ("stem", "encoder1", "encoder2", "bottleneck"):
            def hook(module, args, output, name=name):
                shapes[name] = tuple(output.shape)
            handles.append(getattr(model, name).register_forward_hook(hook))
        output = model(torch.rand(3, 3, 32, 40))
        for handle in handles:
            handle.remove()
        self.assertEqual(shapes, {"stem": (3, 32, 32, 40), "encoder1": (3, 64, 16, 20),
                                  "encoder2": (3, 128, 8, 10), "bottleneck": (3, 256, 4, 5)})
        self.assertEqual(set(output), {"enhanced", "Fs", "Fd", "c_hat"})
        self.assertEqual(output["Fs"].shape, (3, 128, 4, 5))
        self.assertEqual(output["Fd"].shape, output["Fs"].shape)
        self.assertEqual(output["c_hat"].shape, (3, 1))
        self.assertTrue(((output["enhanced"] >= 0) & (output["enhanced"] <= 1)).all())
        losses = SDDLoss()(output, torch.rand(3, 3, 32, 40), torch.rand(3, 1))
        expected = losses["l1"] + 0.2 * losses["ssim_loss"] + 0.1 * losses["scene"] + 0.05 * losses["illumination"]
        torch.testing.assert_close(losses["total"], expected)
        losses["total"].backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_arbitrary_input_size_and_tiny_input(self):
        model = SDDNetV1().eval()
        with torch.no_grad():
            output = model(torch.rand(1, 3, 31, 37))
            self.assertEqual(output["enhanced"].shape, (1, 3, 31, 37))
            self.assertEqual(output["Fs"].shape, (1, 128, 4, 5))
            self.assertEqual(model(torch.rand(1, 3, 1, 1))["enhanced"].shape, (1, 3, 1, 1))

    def test_ssim_identity_reference_and_gradient(self):
        image = torch.rand(2, 3, 20, 21, requires_grad=True)
        torch.testing.assert_close(ssim(image, image), torch.tensor(1.0))
        black = torch.zeros(1, 3, 13, 13)
        white = torch.ones_like(black)
        self.assertAlmostEqual(ssim(black, white).item(), 0.01 ** 2 / (1 + 0.01 ** 2), places=5)
        (1 - ssim(image, torch.rand_like(image))).backward()
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_triplet_batch_order(self):
        batch = {}
        for value, name in enumerate(("anchor", "positive", "negative")):
            batch[f"{name}_low"] = torch.full((2, 3, 8, 8), float(value))
            batch[f"{name}_gt"] = batch[f"{name}_low"] + 10
            batch[f"{name}_level"] = torch.full((2, 1), float(value))
        low, gt, levels = concatenate_triplets(batch, torch.device("cpu"))
        self.assertEqual(low[:, 0, 0, 0].tolist(), [0, 0, 1, 1, 2, 2])
        torch.testing.assert_close(gt, low + 10)
        self.assertEqual(levels[:, 0].tolist(), [0, 0, 1, 1, 2, 2])

    def test_tiled_coverage_and_weight_normalization(self):
        class Identity(nn.Module):
            def forward(self, x):
                return {"enhanced": x}
        for shape in ((1, 3, 47, 71), (1, 3, 1, 71)):
            image = torch.rand(shape)
            actual = enhance_image(Identity(), image, torch.device("cpu"), tile_size=32, tile_overlap=8)
            torch.testing.assert_close(actual, image)


class DataAndCheckpointTests(unittest.TestCase):
    """Create a tiny registered fixture instead of modifying user dataset files."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.root = Path(self.temp.name)
        for kind in ("input", "gt"):
            (self.root / "train" / kind).mkdir(parents=True)
        grid = np.arange(40 * 48 * 3).reshape(40, 48, 3)
        for offset, scene in enumerate(("Scene1_a", "Scene2_b")):
            image = Image.fromarray(((grid + offset * 37) % 256).astype(np.uint8))
            image.save(self.root / "train" / "gt" / f"{scene}.png")
            for level in (1, 5, 10):
                image.save(self.root / "train" / "input" / f"{scene}-{level}.png")

    def tearDown(self):
        self.temp.cleanup()

    def test_sampling_normalization_and_aligned_crops(self):
        dataset = MILLTripletDataset(self.root, patch_size=24)
        self.assertEqual(len(dataset), 6)
        for index in range(len(dataset)):
            a, p, n = [dataset.records[i] for i in dataset.sample_indices(index)]
            self.assertEqual(a.scene_id, p.scene_id)
            self.assertNotEqual(a.illum_level, p.illum_level)
            self.assertNotEqual(a.scene_id, n.scene_id)
            self.assertEqual(a.illum_level, n.illum_level)
            sample = dataset[index]
            self.assertEqual(len(sample), 9)
            torch.testing.assert_close(sample["anchor_low"], sample["positive_low"])
            for name in ("anchor", "positive", "negative"):
                torch.testing.assert_close(sample[f"{name}_low"], sample[f"{name}_gt"])
                self.assertEqual(sample[f"{name}_low"].shape, (3, 24, 24))
            level = (dataset.records[index].illum_level - 1) / 9
            self.assertAlmostEqual(sample["anchor_level"].item(), level, places=6)
        padded = MILLTripletDataset(self.root, patch_size=64)[0]
        self.assertEqual(padded["anchor_low"].shape, (3, 64, 64))

    def test_csv_json_roundtrip_and_explicit_filename_override(self):
        for extension in ("csv", "json"):
            path = self.root / f"metadata.{extension}"
            self.assertEqual(write_mill_metadata(self.root, path, ["train"]), 6)
            loaded = MILLTripletDataset(self.root, metadata=path)
            self.assertEqual(len(loaded), 6)
        path = self.root / "metadata.csv"
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        original = self.root / rows[0]["input_path"]
        renamed = original.with_name("arbitrary_name.png")
        original.rename(renamed)
        rows[0]["filename"] = renamed.name
        rows[0]["input_path"] = renamed.relative_to(self.root).as_posix()
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.assertEqual(len(read_mill_metadata(path, self.root, "train")), 6)
        self.assertEqual(len(MILLTripletDataset(self.root, metadata=path)), 6)
        with self.assertRaises(ValueError):
            MILLTripletDataset(self.root)

    def test_filename_and_missing_pair_fail_fast(self):
        self.assertEqual(parse_mill_filename("Scene1_a-10.png"), ("Scene1_a", 10.0))
        self.assertEqual(parse_mill_filename("Test13-2.png"), ("Test13", 2.0))
        with self.assertRaises(ValueError):
            parse_mill_filename("ambiguous.png")
        (self.root / "train/gt/Scene1_a.png").unlink()
        with self.assertRaises(FileNotFoundError):
            MILLPairedDataset(self.root, split="train")

    def test_paired_lol_layouts_and_native_resolution(self):
        for base, low_dir, gt_dir in (("eval15", "low", "high"),
                                      ("Real_captured/Test", "Low", "Normal"),
                                      ("Synthetic/Test", "Low", "Normal")):
            for kind in (low_dir, gt_dir):
                directory = self.root / base / kind
                directory.mkdir(parents=True)
                Image.new("RGB", (39, 27), (42, 30, 20)).save(directory / "image.png")
        for dataset in (LOLv1PairedDataset(self.root), LOLv2PairedDataset(self.root, "real"),
                        LOLv2PairedDataset(self.root, "synthetic")):
            self.assertEqual(dataset[0]["low"].shape, (3, 27, 39))
            torch.testing.assert_close(dataset[0]["low"], dataset[0]["gt"])

    def test_checkpoint_roundtrip_rng(self):
        seed_everything(19)
        model = nn.Linear(2, 1)
        state = {"model": model.state_dict(), "rng": capture_rng(), "epoch": 3}
        path = self.root / "latest.pt"
        save_checkpoint(path, state)
        expected = (random.random(), np.random.rand(), torch.rand(1))
        loaded = load_checkpoint(path)
        model.load_state_dict(loaded["model"])
        restore_rng(loaded["rng"])
        self.assertEqual(loaded["epoch"], 3)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(1), expected[2])


if __name__ == "__main__":
    unittest.main()
