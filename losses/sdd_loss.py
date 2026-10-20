"""Reconstruction + scene triplet + illumination supervision for SDDNet-v1."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ssim import ssim


class SDDLoss(nn.Module):
    """Expect model outputs ordered [all anchors, all positives, all negatives]."""

    def __init__(self, ssim_weight=0.2, scene_weight=0.1, illumination_weight=0.05, margin=0.2):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.scene_weight = scene_weight
        self.illumination_weight = illumination_weight
        self.margin = margin

    def forward(self, outputs, targets, levels):
        count = targets.shape[0]
        if count == 0 or count % 3:
            raise ValueError("A concatenated triplet batch must have 3*B samples.")
        if outputs["c_hat"].shape != levels.shape or levels.shape != (count, 1):
            raise ValueError("Illumination labels and predictions must both have shape [3*B, 1].")
        # Float32 avoids unstable SSIM/normalization under CUDA autocast.
        with torch.autocast(device_type=targets.device.type, enabled=False):
            enhanced = outputs["enhanced"].float()
            l1 = F.l1_loss(enhanced, targets.float())
            ssim_loss = 1.0 - ssim(enhanced, targets)
            rec = l1 + self.ssim_weight * ssim_loss
            embedding = F.normalize(outputs["Fs"].float().mean(dim=(2, 3)), p=2, dim=1, eps=1e-8)
            anchor, positive, negative = embedding.chunk(3, dim=0)
            distance_ap = torch.linalg.vector_norm(anchor - positive, ord=2, dim=1)
            distance_an = torch.linalg.vector_norm(anchor - negative, ord=2, dim=1)
            scene = (distance_ap - distance_an + self.margin).clamp_min(0).mean()
            illumination = F.l1_loss(outputs["c_hat"].float(), levels.float())
            total = rec + self.scene_weight * scene + self.illumination_weight * illumination
        return {"total": total, "rec": rec, "l1": l1, "ssim_loss": ssim_loss,
                "scene": scene, "illumination": illumination}
