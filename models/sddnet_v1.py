"""Structure-Degradation Decoupling Network v1; no wavelets or denoising head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    """Two convolutions with an identity skip; normalization is intentionally absent."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


def blocks(channels: int, count: int) -> nn.Sequential:
    """Create independent residual blocks."""
    return nn.Sequential(*(ResBlock(channels) for _ in range(count)))


def stage(in_channels: int, out_channels: int, count: int, stride: int = 1):
    """Shared encoder stage or branch projection."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
        blocks(out_channels, count),
    )


class Fusion(nn.Module):
    """Spatial degradation and global illumination modulate structure features."""

    def __init__(self):
        super().__init__()
        self.gamma = nn.Conv2d(128, 128, 3, padding=1)
        self.beta = nn.Conv2d(128, 128, 3, padding=1)
        self.illumination_affine = nn.Sequential(
            nn.Linear(1, 64), nn.LeakyReLU(0.1, inplace=True), nn.Linear(64, 256)
        )
        self.fuse = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                  nn.LeakyReLU(0.1, inplace=True), blocks(256, 2))

    def forward(self, fs, fd, c_hat):
        channel_gamma, channel_beta = self.illumination_affine(c_hat).chunk(2, dim=1)
        gamma = self.gamma(fd) + channel_gamma[:, :, None, None]
        beta = self.beta(fd) + channel_beta[:, :, None, None]
        fs_mod = fs * (1.0 + 0.1 * gamma.tanh()) + 0.1 * beta.tanh()
        return self.fuse(torch.cat((fs, fs_mod), dim=1))


class UpBlock(nn.Module):
    """Bilinear upsampling followed by convolution and encoder skip fusion."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                     nn.LeakyReLU(0.1, inplace=True))
        self.refine = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            blocks(out_channels, 2),
        )

    def forward(self, x, skip):
        x = self.project(F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False))
        return self.refine(torch.cat((x, skip), dim=1))


class SDDNetV1(nn.Module):
    """Input RGB in [0, 1]; return enhanced, Fs, Fd, and c_hat.

    Non-multiples of eight are padded on the right/bottom. Only the RGB output
    is cropped back; latent spatial sizes are ceil(H/8), ceil(W/8).
    """

    def __init__(self):
        super().__init__()
        self.stem = stage(3, 32, 2)
        self.encoder1 = stage(32, 64, 2, stride=2)
        self.encoder2 = stage(64, 128, 2, stride=2)
        self.bottleneck = stage(128, 256, 4, stride=2)
        self.structure_branch = stage(256, 128, 2)
        self.degradation_branch = stage(257, 128, 2)
        self.illumination_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(128, 64), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.fusion = Fusion()
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.head = nn.Sequential(nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("Expected a non-empty RGB tensor [B, 3, H, W].")
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % 8, (-w) % 8
        mode = "reflect" if pad_h < h and pad_w < w else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        prior = F.avg_pool2d(x.amax(dim=1, keepdim=True), 5, stride=1, padding=2, count_include_pad=False)
        e0 = self.stem(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        z = self.bottleneck(e2)
        fs = self.structure_branch(z)
        prior = F.interpolate(prior, size=z.shape[-2:], mode="area")
        fd = self.degradation_branch(torch.cat((z, prior), dim=1))
        c_hat = self.illumination_head(fd)
        fr = self.fusion(fs, fd, c_hat)
        enhanced = self.head(self.up3(self.up2(self.up1(fr, e2), e1), e0))
        return {"enhanced": enhanced[:, :, :h, :w], "Fs": fs, "Fd": fd, "c_hat": c_hat}