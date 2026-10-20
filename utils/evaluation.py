"""Per-image RGB PSNR/SSIM evaluation and optional PNG/CSV/JSON export."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from losses.ssim import ssim
from .inference import enhance_image


def psnr(pred, target):
    """Full RGB MSE with data_range=1; exact equality has infinite PSNR."""
    mse = (pred.float() - target.float()).square().mean().item()
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def write_json(path, value):
    """Write strict JSON, representing exact-match infinite PSNR as a string."""
    def sanitize(item):
        if isinstance(item, np.generic):
            return sanitize(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return "Infinity" if item > 0 else "-Infinity" if item < 0 else "NaN"
        if isinstance(item, dict):
            return {key: sanitize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(val) for val in item]
        return item
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(sanitize(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


@torch.inference_mode()
def evaluate(model, loader, device, amp=True, tile_size=512, tile_overlap=32,
             output_dir=None, save_images=False, max_samples=None, description="Evaluate"):
    """Average per-image metrics; do not resize, shave borders, or GT-correct exposure."""
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive or null.")
    was_training = model.training
    model.eval()
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if save_images:
            (output_dir / "images").mkdir(exist_ok=True)
    rows = []
    try:
        for index, batch in enumerate(tqdm(loader, desc=description, leave=False)):
            if max_samples is not None and index >= max_samples:
                break
            pred = enhance_image(model, batch["low"], device, amp, tile_size, tile_overlap)
            gt = batch["gt"].float().cpu()
            row = {"filename": batch["filename"][0], "psnr": psnr(pred, gt),
                   "ssim": ssim(pred, gt).item()}
            rows.append(row)
            if output_dir and save_images:
                pixels = (pred[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
                Image.fromarray(pixels).save(output_dir / "images" / (row["filename"] + ".png"))
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("Evaluation produced no images.")
    metrics = {"count": len(rows), "psnr": sum(r["psnr"] for r in rows) / len(rows),
               "ssim": sum(r["ssim"] for r in rows) / len(rows),
               "tile_size": tile_size, "tile_overlap": tile_overlap,
               "protocol": "RGB [0,1], native resolution, no border crop; mean per-image metrics"}
    if output_dir:
        write_json(output_dir / "metrics.json", metrics)
        with (output_dir / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["filename", "psnr", "ssim"])
            writer.writeheader()
            writer.writerows(rows)
    return metrics
