"""Fixed-parameter pytorch-msssim RGB SSIM, always evaluated in float32."""

from __future__ import annotations

import torch
from pytorch_msssim import ssim as library_ssim


def ssim(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Fixed 11x11 Gaussian; tiny tests must use a sufficiently large image."""
    if pred.shape != target.shape or pred.ndim != 4 or min(pred.shape[-2:]) < 11:
        raise ValueError("SSIM requires matching, non-empty [B,C,H,W] tensors.")
    if reduction not in ("mean", "none"):
        raise ValueError("reduction must be mean or none")
    with torch.autocast(device_type=pred.device.type, enabled=False):
        return library_ssim(pred.float(), target.float(), data_range=1.0,
                            size_average=reduction == "mean", win_size=11,
                            win_sigma=1.5, K=(0.01, 0.03), nonnegative_ssim=False)
