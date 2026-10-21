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
from models.sddnet_v1 import model_options
from utils.io import write_json, write_csv
from utils.experiment import (DATASETS, assert_environment, file_hash, manifest_identity, run_path, validate_formal_checkpoint)
from utils.metrics import average, image_metrics, load_lpips
from utils.runtime import load_checkpoint, project_path, choose_device


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
def evaluate_all(cfg, checkpoint_path, *, datasets=None, max_images=None, metadata_dir=None):
    assert_environment(cfg)
    checkpoint = load_checkpoint(checkpoint_path)
    validate_formal_checkpoint(checkpoint,cfg)
    hashes = manifest_identity(cfg)
    validate_formal_checkpoint(checkpoint,cfg,hashes)
    directory = run_path(cfg)
    supplied_metadata = metadata_dir is not None
    metadata_dir=metadata_dir if supplied_metadata else directory/'metadata'
    if cfg.get('ablation_version') and not supplied_metadata:
        # Test images are audited only after training/selection is frozen. Do not
        # replace the train/validation fingerprint stored in the checkpoint.
        from scripts.prepare_data import prepare_data
        prepare_data(cfg,scope='final')
        metadata_dir=directory/'final_data'/'metadata'
        from utils.io import read_csv
        identity=lambda row:tuple(row[k] for k in ('split','filename','input_sha256','gt_sha256'))
        trained={identity(r) for r in read_csv(directory/'metadata/mill.csv')}
        current={identity(r) for r in read_csv(metadata_dir/'mill.csv') if r['split'] in ('train','validation')}
        if trained!=current:raise ValueError('Raw train/validation files changed since the checkpoint audit.')
    metrics_dir = directory / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg['device'])
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    model = SDDNetV1(**model_options(cfg)).to(device).float().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    lpips_model, lpips_status = load_lpips(device)
    write_json(metrics_dir / "lpips_status.json", lpips_status)
    checkpoint_sha = file_hash(project_path(checkpoint_path))
    per_image, summary, failures = [], [], {}
    selected=DATASETS if datasets is None else tuple(datasets)
    if not selected or set(selected)-set(DATASETS):raise ValueError('Unknown evaluation dataset.')
    if max_images is not None and max_images<1:raise ValueError('max_images must be positive.')
    for dataset in selected:
        rows = []
        try:
            data = ManifestPairedDataset(dataset_root(cfg, dataset), metadata_dir / f"{dataset}.csv", "test")
            image_dir = directory / "enhanced" / dataset
            image_dir.mkdir(parents=True, exist_ok=True)
            count=len(data) if max_images is None else min(len(data),max_images)
            for index in tqdm(range(count), desc=f"Test {dataset}"):
                if supplied_metadata:
                    from .release import check_pair_files
                    check_pair_files(dataset_root(cfg,dataset),data.rows[index])
                sample, metadata = data[index], data.rows[index]
                low = sample["low"].unsqueeze(0).to(device)
                gt = sample["gt"].unsqueeze(0).to(device)
                # Full original resolution, FP32, one input, no GT/labels in forward.
                outputs = model(low,return_latents=True)
                enhanced = outputs["enhanced"].clamp(0,1)
                if not all(torch.isfinite(value).all().item() for value in outputs.values()):
                    raise FloatingPointError(f"Non-finite output for {metadata['filename']}")
                before = image_metrics(low, gt, lpips_model, lpips_status.get("reason", ""))
                after = image_metrics(enhanced, gt, lpips_model, lpips_status.get("reason", ""))
                row = {"dataset": dataset, "split": "test", "filename": metadata["filename"],
                       "scene_id": metadata["scene_id"], "illum_level": metadata["illum_level"] or "NA",
                       'content_id':metadata.get('content_id',metadata['scene_id']),'group_id':metadata.get('group_id',''),
                       "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": checkpoint["completed_epoch"],
                       "c_hat": outputs["c_hat"].item(), "c_gt": "NA", "illumination_abs_error": "NA"}
                if dataset == "mill":
                    row["c_gt"] = (float(metadata["illum_level"]) - 1) / 9
                    row["illumination_abs_error"] = abs(row["c_hat"] - row["c_gt"])
                for prefix, result in (("input", before), ("output", after)):
                    row.update({f"{prefix}_{key}": value for key, value in result.items()})
                row["delta_PSNR_RGB"] = after["PSNR_RGB"] - before["PSNR_RGB"]
                row['delta_psnr_vs_input']=row['delta_PSNR_RGB']
                row['delta_psnr_vs_control']='NA'  # requires a separately matched control run
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
                          "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": checkpoint["completed_epoch"],
                          "experiment": "MILL train; LOL is cross-dataset evaluation", "arch_revision": "v1_clean"}
                for metric in ("PSNR_RGB", "SSIM_RGB", "LPIPS_Alex"):
                    result[metric] = average([r[f"{kind}_{metric}"] for r in rows])
                result["LPIPS_reason"] = "; ".join(sorted({r[f"{kind}_LPIPS_reason"] for r in rows if r[f"{kind}_LPIPS_reason"]}))
                result["illumination_MAE"] = average([r["illumination_abs_error"] for r in rows]) if dataset == "mill" and kind == "output" else "NA"
                result["illumination_reason"] = "" if dataset == "mill" and kind == "output" else "No predicted input baseline label, or no comparable LOL level labels"
                result["delta_PSNR_RGB"] = average([r["delta_PSNR_RGB"] for r in rows]) if kind == "output" else "NA"
                if kind=='output':
                    from utils.ablation_metrics import paired_summary
                    result.update(paired_summary(rows))
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
               "checkpoint_sha256": checkpoint_sha, "epoch": checkpoint["completed_epoch"], "complete": not failures,
               "datasets": selected, "limited_smoke": max_images is not None, "max_images": max_images,
               "protocol": "FP32, batch1, native RGB, no tiles/TTA/GT adjustment; arithmetic mean per image",
               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
               "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "ssim": "pytorch-msssim: data_range=1, win11, sigma1.5, K=(.01,.03), nonnegative=False"})
    if failures:
        raise RuntimeError("One or more test datasets failed; NA and reasons are saved.")
    print("Explicitly selected evaluation completed with the same checkpoint.", flush=True)


