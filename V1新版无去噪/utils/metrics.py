"""Fixed RGB metrics and pretrained LPIPS_Alex evaluation (never a training loss)."""

from __future__ import annotations

import math
import warnings

import torch

from losses.ssim import ssim
from .runtime import PROJECT_ROOT


def psnr_rgb(pred, target):
    mse = (pred.float() - target.float()).square().mean().item()
    return math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)


def load_lpips(device):
    """Download official AlexNet backbone if absent; never substitute random weights."""
    torch.hub.set_dir(str(PROJECT_ROOT / "cache" / "torch"))
    try:
        import lpips
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            metric = lpips.LPIPS(net="alex", version="0.1", pretrained=True,
                                 pnet_rand=False, eval_mode=True, verbose=False).to(device).eval()
        metric.requires_grad_(False)
        # Record exact files so a later evaluation can establish weight provenance.
        from .experiment import file_hash
        from pathlib import Path
        files = list((PROJECT_ROOT / "cache" / "torch" / "checkpoints").glob("alexnet-*.pth"))
        files += list((Path(lpips.__file__).parent / "weights" / "v0.1").glob("alex.pth"))
        provenance = {str(path): file_hash(path) for path in files}
        if len(files) < 2:
            raise RuntimeError("Could not establish both official AlexNet and LPIPS linear weight files.")
        return metric, {"available": True, "net": "alex", "version": "0.1", "weight_sha256": provenance}
    except Exception as error:
        return None, {"available": False, "reason": f"{type(error).__name__}: {error}"}


@torch.inference_mode()
def image_metrics(pred, target, lpips_model=None, unavailable_reason="LPIPS weights unavailable"):
    """Evaluate unquantized RGB [0,1]. Failed optional LPIPS is explicitly NA."""
    metrics = {"PSNR_RGB": psnr_rgb(pred, target), "SSIM_RGB": ssim(pred, target).item(),
               "LPIPS_Alex": "NA", "LPIPS_reason": unavailable_reason}
    if lpips_model is not None:
        try:
            value = lpips_model(2 * pred.float() - 1, 2 * target.float() - 1).mean().item()
            if not math.isfinite(value):
                raise FloatingPointError("non-finite LPIPS")
            metrics.update(LPIPS_Alex=value, LPIPS_reason="")
        except Exception as error:
            metrics["LPIPS_reason"] = f"{type(error).__name__}: {error}"
    return metrics


def average(values):
    """Do not silently average a subset of failed measurements."""
    return sum(values) / len(values) if values and all(isinstance(v, (int, float)) for v in values) else "NA"
