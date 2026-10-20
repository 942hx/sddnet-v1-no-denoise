"""Evaluate the same validation-selected checkpoint on all four test sets, in FP32."""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from datasets.paired_dataset import ManifestPairedDataset
from models import SDDNetV1
from utils.evaluation import write_json
from utils.experiment import (DATASETS, MODEL_REVISION, assert_environment, experiment_config,
                              file_hash, manifest_identity, run_path)
from utils.metrics import average, image_metrics, load_lpips
from utils.runtime import load_checkpoint, project_path


def write_csv(path, rows):
    """Persist real measurements as UTF-8 CSV, including explicit NA/reason fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset_root(cfg, dataset):
    if dataset == "mill":
        return project_path(cfg["data"]["mill_root"])
    if dataset == "lolv1":
        return project_path(cfg["data"]["lolv1_root"])
    return project_path(cfg["data"]["lolv2_root"]) / ("Real_captured" if dataset.endswith("real") else "Synthetic")


def png(tensor):
    """Quantization is only for saved previews, after all metric computation."""
    pixels = (tensor.detach().float().cpu()[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def comparison(low, output, gt, path, caption):
    images = [png(value) for value in (low, output, gt)]
    w, h = images[0].size
    canvas = Image.new("RGB", (3 * w, h + 30), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, title) in enumerate(zip(images, ("input", "output", "GT"))):
        canvas.paste(image, (index * w, 30))
        draw.text((index * w + 5, 5), f"{title} - {caption}", fill="black")
    canvas.save(path)


@torch.inference_mode()
def evaluate_all(cfg, checkpoint_path):
    assert_environment(cfg)
    hashes = manifest_identity(cfg)
    directory = run_path(cfg)
    metrics_dir = directory / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("model_revision") != MODEL_REVISION or checkpoint.get("manifest_hashes") != hashes:
        raise ValueError("Checkpoint architecture or audited data manifests are incompatible.")
    device = torch.device(cfg["device"])
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    model = SDDNetV1().to(device).float().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    lpips_model, lpips_status = load_lpips(device)
    write_json(metrics_dir / "lpips_status.json", lpips_status)
    checkpoint_sha = file_hash(project_path(checkpoint_path))
    per_image, summary, failures = [], [], {}
    for dataset in DATASETS:
        rows = []
        try:
            data = ManifestPairedDataset(dataset_root(cfg, dataset), directory / "metadata" / f"{dataset}.csv", "test")
            image_dir = directory / "enhanced" / dataset
            image_dir.mkdir(parents=True, exist_ok=True)
            for index in tqdm(range(len(data)), desc=f"Test {dataset}"):
                sample, metadata = data[index], data.rows[index]
                low = sample["low"].unsqueeze(0).to(device)
                gt = sample["gt"].unsqueeze(0).to(device)
                # Full original resolution, FP32, one input, no GT/labels in forward.
                outputs = model(low)
                enhanced = outputs["enhanced"]
                if not all(torch.isfinite(value).all().item() for value in outputs.values()):
                    raise FloatingPointError(f"Non-finite output for {metadata['filename']}")
                before = image_metrics(low, gt, lpips_model, lpips_status.get("reason", ""))
                after = image_metrics(enhanced, gt, lpips_model, lpips_status.get("reason", ""))
                row = {"dataset": dataset, "split": "test", "filename": metadata["filename"],
                       "scene_id": metadata["scene_id"], "illum_level": metadata["illum_level"] or "NA",
                       "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": checkpoint["epoch"],
                       "c_hat": outputs["c_hat"].item(), "c_gt": "NA", "illumination_abs_error": "NA"}
                if dataset == "mill":
                    row["c_gt"] = (float(metadata["illum_level"]) - 1) / 9
                    row["illumination_abs_error"] = abs(row["c_hat"] - row["c_gt"])
                for prefix, result in (("input", before), ("output", after)):
                    row.update({f"{prefix}_{key}": value for key, value in result.items()})
                row["delta_PSNR_RGB"] = after["PSNR_RGB"] - before["PSNR_RGB"]
                row["delta_SSIM_RGB"] = after["SSIM_RGB"] - before["SSIM_RGB"]
                row["delta_LPIPS_Alex"] = (after["LPIPS_Alex"] - before["LPIPS_Alex"]
                                            if isinstance(after["LPIPS_Alex"], float) and isinstance(before["LPIPS_Alex"], float) else "NA")
                rows.append(row)
                if cfg["evaluation"].get("save_images", True):
                    png(enhanced).save(image_dir / metadata["filename"])
                if index < 3:
                    comparison(low, enhanced, gt, image_dir / f"comparison_{index:02d}.png", metadata["filename"])
                del low, gt, outputs, enhanced
            for kind in ("input", "output"):
                result = {"dataset": dataset, "kind": kind, "count": len(rows),
                          "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": checkpoint["epoch"]}
                for metric in ("PSNR_RGB", "SSIM_RGB", "LPIPS_Alex"):
                    result[metric] = average([r[f"{kind}_{metric}"] for r in rows])
                result["LPIPS_reason"] = "; ".join(sorted({r[f"{kind}_LPIPS_reason"] for r in rows if r[f"{kind}_LPIPS_reason"]}))
                result["illumination_MAE"] = average([r["illumination_abs_error"] for r in rows]) if dataset == "mill" and kind == "output" else "NA"
                result["illumination_reason"] = "" if dataset == "mill" and kind == "output" else "No predicted input baseline label, or no comparable LOL level labels"
                result["delta_PSNR_RGB"] = average([r["delta_PSNR_RGB"] for r in rows]) if kind == "output" else "NA"
                result["PSNR_decline_fraction"] = sum(r["delta_PSNR_RGB"] < 0 for r in rows) / len(rows) if kind == "output" else "NA"
                summary.append(result)
            per_image.extend(rows)
        except Exception as error:
            failures[dataset] = f"{type(error).__name__}: {error}"
            print(f"Evaluation failed for {dataset}: {failures[dataset]}", flush=True)
            for kind in ("input", "output"):
                summary.append({"dataset": dataset, "kind": kind, "count": len(rows), "PSNR_RGB": "NA", "SSIM_RGB": "NA",
                                "LPIPS_Alex": "NA", "reason": failures[dataset], "checkpoint_sha256": checkpoint_sha})
        write_csv(metrics_dir / "summary.csv", summary)
        write_csv(metrics_dir / "per_image.csv", per_image)
    levels = []
    mill = [row for row in per_image if row["dataset"] == "mill"]
    for level in range(1, 11):
        subset = [row for row in mill if float(row["illum_level"]) == level]
        row = {"level": level, "count": len(subset)}
        for kind in ("input", "output"):
            for metric in ("PSNR_RGB", "SSIM_RGB", "LPIPS_Alex"):
                row[f"{kind}_{metric}"] = average([r[f"{kind}_{metric}"] for r in subset])
        row["delta_PSNR_RGB"] = average([r["delta_PSNR_RGB"] for r in subset])
        row["illumination_MAE"] = average([r["illumination_abs_error"] for r in subset])
        row["PSNR_decline_fraction"] = sum(r["delta_PSNR_RGB"] < 0 for r in subset) / len(subset) if subset else "NA"
        row["LPIPS_reason"] = "; ".join(sorted({r[f"{kind}_LPIPS_reason"] for r in subset
                                                 for kind in ("input", "output") if r[f"{kind}_LPIPS_reason"]}))
        if not subset:
            row["LPIPS_reason"] = "No evaluated images for this level; see evaluation_status.json"
        levels.append(row)
    write_csv(metrics_dir / "mill_per_level.csv", levels)
    write_json(metrics_dir / "evaluation_status.json", {"failures": failures, "checkpoint": str(project_path(checkpoint_path)),
               "checkpoint_sha256": checkpoint_sha, "epoch": checkpoint["epoch"], "complete": not failures,
               "protocol": "FP32, batch1, native RGB, no tiles/TTA/GT adjustment; arithmetic mean per image",
               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
               "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "ssim": "pytorch-msssim: data_range=1, win11, sigma1.5, K=(.01,.03), nonnegative=False"})
    if failures:
        raise RuntimeError("One or more test datasets failed; NA and reasons are saved.")
    print("Four-dataset evaluation completed with the same checkpoint.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sddnet_v1_100e.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate_all(experiment_config(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
