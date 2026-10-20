"""SDDNetV1, arch_revision=v1_clean: residual RGB and multiscale conditional affine."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

ARCH_REVISION = "v1_clean"


class ResidualBlock(nn.Module):
    """Two convolutions; scaled residual, no normalization or post-add activation."""
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.LeakyReLU(0.1, inplace=False),
                                  nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + 0.1 * self.body(x)


def stage(cin, cout, count, stride=1):
    """Independent residual block instances in every encoder/branch stage."""
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                         nn.LeakyReLU(0.1, inplace=False),
                         *(ResidualBlock(cout) for _ in range(count)))


def brightness_prior(x):
    """Two input hints on the original image; RGB mean is not physical luminance."""
    return torch.cat([F.avg_pool2d(value, 5, 1, 2, count_include_pad=False)
                      for value in (x.amax(1, keepdim=True), x.mean(1, keepdim=True))], dim=1)


class ConditionalAffine(nn.Module):
    """Low-resolution local parameters plus global c; independent learnable bounded gain."""
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.local = nn.Conv2d(128, 2 * channels, kernel_size, padding=kernel_size // 2)
        self.global_mlp = nn.Sequential(nn.Linear(1, 64), nn.LeakyReLU(0.1, inplace=False),
                                        nn.Linear(64, 2 * channels))
        self.rho = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

    def affine_parameters(self, feature, fd, c_hat):
        # Convolve Fd at bottleneck resolution, then resize the parameter maps.
        local = self.local(fd)
        local = F.interpolate(local, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        raw = local + self.global_mlp(c_hat)[:, :, None, None]
        gamma_raw, beta_raw = raw.chunk(2, dim=1)
        gain = self.rho.sigmoid()
        return gain * gamma_raw.tanh(), gain * beta_raw.tanh(), gamma_raw, beta_raw

    def forward(self, feature, fd, c_hat):
        gamma, beta, _, _ = self.affine_parameters(feature, fd, c_hat)
        return feature * (1 + gamma) + beta


class UpBlock(nn.Module):
    """Resize to the actual skip size; fuse only the corresponding modulated skip."""
    def __init__(self, cin, cout):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                     nn.LeakyReLU(0.1, inplace=False))
        self.refine = nn.Sequential(nn.Conv2d(2 * cout, cout, 3, padding=1),
                                    nn.LeakyReLU(0.1, inplace=False),
                                    ResidualBlock(cout), ResidualBlock(cout))

    def forward(self, x, skip):
        x = self.project(F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False))
        return self.refine(torch.cat([x, skip], dim=1))


class SDDNetV1(nn.Module):
    """Input float RGB only. Enhanced is the raw, unclamped residual reconstruction."""
    arch_revision = ARCH_REVISION

    def __init__(self):
        super().__init__()
        self.stem = stage(3, 32, 2)
        self.encoder1 = stage(32, 64, 2, 2)
        self.encoder2 = stage(64, 128, 2, 2)
        self.bottleneck = stage(128, 256, 4, 2)
        self.structure_branch = stage(256, 128, 2)
        self.degradation_branch = stage(258, 128, 2)
        self.illumination_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(128, 64), nn.LeakyReLU(0.1, inplace=False), nn.Linear(64, 1), nn.Sigmoid())
        self.bottleneck_affine = ConditionalAffine(128, 3)
        self.fusion = stage(128, 256, 2)
        self.skip_affine2 = ConditionalAffine(128, 1)
        self.skip_affine1 = ConditionalAffine(64, 1)
        self.skip_affine0 = ConditionalAffine(32, 1)
        self.up1, self.up2, self.up3 = UpBlock(256, 128), UpBlock(128, 64), UpBlock(64, 32)
        self.head = nn.Conv2d(32, 3, 3, padding=1)
        self._initialize_new_model()

    def _initialize_new_model(self):
        """Explicit initialization at construction only; never rerun after loading weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        special = [self.head, self.illumination_head[-2]]
        for module in self.modules():
            if isinstance(module, ConditionalAffine):
                special += [module.local, module.global_mlp[-1]]
        for module in special:
            nn.init.normal_(module.weight, mean=0, std=1e-3)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_latents: bool = False) -> dict[str, torch.Tensor]:
        if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x):
            raise TypeError("Input must be a floating-point RGB Tensor [B,3,H,W].")
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[0], *x.shape[-2:]) < 1:
            raise ValueError("Expected non-empty [B,3,H,W], with B,H,W >= 1 and exactly 3 channels.")
        if not isinstance(return_latents, bool):
            raise TypeError("return_latents must be bool.")
        h, w = x.shape[-2:]
        prior = brightness_prior(x)  # Compute before introducing any padding.
        ph, pw = (-h) % 8, (-w) % 8
        mode = "reflect" if ph < h and pw < w else "replicate"
        padding = (0, pw, 0, ph)
        x = F.pad(x, padding, mode=mode)
        prior = F.pad(prior, padding, mode=mode)
        e0 = self.stem(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        z = self.bottleneck(e2)
        fs = self.structure_branch(z)
        fd = self.degradation_branch(torch.cat([z, F.interpolate(prior, size=z.shape[-2:], mode="area")], dim=1))
        c_hat = self.illumination_head(fd)
        fr = self.fusion(self.bottleneck_affine(fs, fd, c_hat))
        s2 = self.skip_affine2(e2, fd, c_hat)
        s1 = self.skip_affine1(e1, fd, c_hat)
        s0 = self.skip_affine0(e0, fd, c_hat)
        delta = self.head(self.up3(self.up2(self.up1(fr, s2), s1), s0))
        result = {"enhanced": (x + delta)[:, :, :h, :w]}
        if return_latents:
            result.update(Fs=fs, Fd=fd, c_hat=c_hat)
        return result
