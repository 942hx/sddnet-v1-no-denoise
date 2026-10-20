"""The single v1_clean objective: reconstruction, scene triplet, illumination, consistency."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .ssim import ssim


def consistency_loss(anchor, positive):
    """Per-position channel-normalized cosine distance, FP32 and same aligned scene only."""
    u = F.normalize(anchor.float(), p=2, dim=1, eps=1e-6)
    v = F.normalize(positive.float(), p=2, dim=1, eps=1e-6)
    return (1 - (u * v).sum(dim=1)).clamp_min(0).mean()


class SDDLoss(nn.Module):
    """progress is completed epochs plus fractional epoch; warmup never restarts on resume."""
    def __init__(self, ssim_weight=0.2, scene_weight=0.1, illumination_weight=0.05,
                 margin=0.2, lambda_cons=0.02, consistency_warmup_epochs=5):
        super().__init__()
        self.ssim_weight, self.scene_weight = float(ssim_weight), float(scene_weight)
        self.illumination_weight, self.lambda_cons = float(illumination_weight), float(lambda_cons)
        self.warmup = float(consistency_warmup_epochs)
        self.triplet = nn.TripletMarginLoss(margin=margin, p=2, eps=1e-6, swap=False, reduction="mean")
        if self.lambda_cons < 0 or self.warmup <= 0:
            raise ValueError("lambda_cons must be nonnegative and warmup epochs positive.")

    def consistency_weight(self, progress):
        return self.lambda_cons * min(max(float(progress) / self.warmup, 0.0), 1.0)

    def reconstruction(self, y_raw, target):
        # Never overwrite raw RGB: L1 must keep restoring gradients outside [0,1].
        l1 = F.l1_loss(y_raw.float(), target.float())
        ssim_loss = 1 - ssim(y_raw.float().clamp(0, 1), target.float())
        return l1 + self.ssim_weight * ssim_loss, l1, ssim_loss

    def forward(self, outputs, targets, levels, progress=0.0):
        count = targets.shape[0]
        if count == 0 or count % 3:
            raise ValueError("Expected [all anchors, all positives, all negatives], N=3*B.")
        if outputs["enhanced"].shape != targets.shape:
            raise ValueError("RGB prediction and GT must have identical unpadded shapes.")
        if levels.shape != (count, 1) or outputs["c_hat"].shape != levels.shape:
            raise ValueError("Prediction and label shapes must be exactly [N,1], without broadcasting.")
        if levels.dtype != torch.float32 or levels.device != outputs["c_hat"].device:
            raise ValueError("Illumination labels must be float32 [N,1] on the prediction device.")
        with torch.autocast(device_type=targets.device.type, enabled=False):
            rec, l1, ssim_loss = self.reconstruction(outputs["enhanced"], targets)
            fs = outputs["Fs"].float()
            embedding = F.normalize(F.adaptive_avg_pool2d(fs, 1).flatten(1), p=2, dim=1, eps=1e-6)
            a, p, n = embedding.chunk(3)
            scene = self.triplet(a, p, n)
            illumination = F.l1_loss(outputs["c_hat"].float(), levels)
            fs_a, fs_p, _ = fs.chunk(3)
            cons = consistency_loss(fs_a, fs_p)
            weight = self.consistency_weight(progress)
            weighted_scene = self.scene_weight * scene
            weighted_illum = self.illumination_weight * illumination
            weighted_cons = weight * cons
            total = rec + weighted_scene + weighted_illum + weighted_cons
        return dict(total=total, rec=rec, l1=l1, ssim_loss=ssim_loss, scene=scene,
                    illumination=illumination, cons=cons,
                    weighted_ssim=self.ssim_weight * ssim_loss,
                    weighted_scene=weighted_scene, weighted_illum=weighted_illum,
                    weighted_cons=weighted_cons, lambda_cons=total.new_tensor(weight))
