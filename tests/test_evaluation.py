"""Exercise four-set export and NA handling on disposable synthetic fixtures, never real test data."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from models import SDDNetV1
from scripts.evaluate import evaluate_all
from utils.evaluation import write_json
from utils.experiment import DATASETS, MODEL_REVISION, file_hash
from utils.runtime import PROJECT_ROOT, save_checkpoint


class EvaluationTests(unittest.TestCase):
    def test_exports_same_checkpoint_and_explicit_optional_na_without_optimizer(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as name:
            directory = Path(name)
            run = directory / "output"
            (run / "metadata").mkdir(parents=True)
            roots = {"mill": directory / "mill", "lolv1": directory / "lol1",
                     "lolv2_real": directory / "lol2" / "Real_captured",
                     "lolv2_synthetic": directory / "lol2" / "Synthetic"}
            hashes = {}
            for dataset, root in roots.items():
                root.mkdir(parents=True)
                # Identical input/GT provides an exact baseline reference.
                Image.new("RGB", (40, 32), (100, 120, 150)).save(root / "input.png")
                Image.new("RGB", (40, 32), (100, 120, 150)).save(root / "gt.png")
                row = dict(dataset=dataset, filename="input.png", split="test", scene_id="fixture",
                           illum_level="1" if dataset == "mill" else "", input_path="input.png", gt_path="gt.png")
                path = run / "metadata" / f"{dataset}.csv"
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                hashes[dataset] = file_hash(path)
            write_json(run / "data_audit.json", {"passed": True, "manifest_hashes": hashes})
            cfg = {"require_environment": "testmodel", "run_dir": str(run), "device": "cpu", "cpu_threads": 2,
                   "data": {"mill_root": str(roots["mill"]), "lolv1_root": str(roots["lolv1"]),
                            "lolv2_root": str(directory / "lol2")}, "evaluation": {"save_images": True}}
            checkpoint = directory / "untrained_fixture.pth"
            save_checkpoint(checkpoint, {"model": SDDNetV1().state_dict(), "epoch": 0,
                                        "model_revision": MODEL_REVISION, "manifest_hashes": hashes})
            with patch("scripts.evaluate.load_lpips", return_value=(None, {"available": False, "reason": "unit fixture: optional LPIPS deliberately unavailable"})), \
                 patch("torch.optim.AdamW", side_effect=AssertionError("Evaluation must not create an optimizer")):
                evaluate_all(cfg, checkpoint)
            with (run / "metrics" / "summary.csv").open(encoding="utf-8-sig") as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(len(summary), 8)
            self.assertEqual({row["dataset"] for row in summary}, set(DATASETS))
            self.assertEqual(len({row["checkpoint_sha256"] for row in summary}), 1)
            self.assertTrue(all(row["LPIPS_Alex"] == "NA" and row["LPIPS_reason"] for row in summary))
            for row in summary:
                if row["kind"] == "input":
                    self.assertEqual(float(row["SSIM_RGB"]), 1.0)
                    self.assertEqual(row["PSNR_RGB"], "inf")
                if row["dataset"] != "mill":
                    self.assertEqual(row["illumination_MAE"], "NA")
            self.assertTrue((run / "metrics" / "per_image.csv").exists())
            self.assertTrue((run / "metrics" / "mill_per_level.csv").exists())


if __name__ == "__main__":
    unittest.main()
