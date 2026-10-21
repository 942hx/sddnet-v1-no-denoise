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
    def __init__(self, channels, kernel_size, affine_mode='bounded'):
        super().__init__()
        if affine_mode != 'bounded':
            raise ValueError('A0 requires affine_mode=bounded.')
        self.affine_mode = affine_mode
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

    def __init__(self, affine_mode='bounded', local_init_std=1e-3,
                 residual_gate=False, g_min=0.25):
        super().__init__()
        if affine_mode != 'bounded' or residual_gate is not False:
            raise ValueError('Only the A0 bounded model with residual_gate=false is supported.')
        if local_init_std != 1e-3 or g_min != 0.25:
            raise ValueError('A0 requires local_init_std=0.001 and g_min=0.25.')
        self.affine_mode, self.local_init_std = affine_mode, float(local_init_std)
        self.residual_gate, self.g_min = residual_gate, float(g_min)
        self.stem = stage(3, 32, 2)
        self.encoder1 = stage(32, 64, 2, 2)
        self.encoder2 = stage(64, 128, 2, 2)
        self.bottleneck = stage(128, 256, 4, 2)
        self.structure_branch = stage(256, 128, 2)
        self.degradation_branch = stage(258, 128, 2)
        self.illumination_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(128, 64), nn.LeakyReLU(0.1, inplace=False), nn.Linear(64, 1), nn.Sigmoid())
        self.bottleneck_affine = ConditionalAffine(128, 3, affine_mode)
        self.fusion = stage(128, 256, 2)
        self.skip_affine2 = ConditionalAffine(128, 1, affine_mode)
        self.skip_affine1 = ConditionalAffine(64, 1, affine_mode)
        self.skip_affine0 = ConditionalAffine(32, 1, affine_mode)
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
            local = any(isinstance(m, ConditionalAffine) and module is m.local for m in self.modules())
            nn.init.normal_(module.weight, mean=0, std=self.local_init_std if local else 1e-3)
            nn.init.zeros_(module.bias)

    def _encode(self, x):
        """Private cache for fixed-content diagnostics; image is the only input."""
        if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x):
            raise TypeError("Input must be a floating-point RGB Tensor [B,3,H,W].")
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[0], *x.shape[-2:]) < 1:
            raise ValueError("Expected non-empty [B,3,H,W], with B,H,W >= 1 and exactly 3 channels.")
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
        return dict(x=x, Fs=fs, Fd=fd, c_hat=c_hat, e0=e0, e1=e1, e2=e2, size=(h,w))

    def _decode(self, cache, fd=None, c_hat=None, affine_enabled=True, gate_enabled=None):
        """Private diagnostic intervention. Never changes input, Fs or raw skips.

        Neutral affine explicitly bypasses modulation; zero Fd is NOT neutral.
        gate_enabled=False is accepted only for historical A0 diagnostic compatibility.
        """
        fd = cache['Fd'] if fd is None else fd
        c_hat = cache['c_hat'] if c_hat is None else c_hat
        if fd.shape != cache['Fd'].shape or c_hat.shape != cache['c_hat'].shape:
            raise ValueError('Replacement conditions must match the encoded condition shapes.')
        def mod(module, feature):
            return module(feature,fd,c_hat) if affine_enabled else feature
        fr = self.fusion(mod(self.bottleneck_affine,cache['Fs']))
        s2 = mod(self.skip_affine2,cache['e2'])
        s1 = mod(self.skip_affine1,cache['e1'])
        s0 = mod(self.skip_affine0,cache['e0'])
        delta = self.head(self.up3(self.up2(self.up1(fr, s2), s1), s0))
        if gate_enabled not in (None, False):
            raise ValueError('A0 has no residual gate.')
        gate = torch.ones_like(c_hat)
        h,w = cache['size']
        # Keep the original arithmetic path EXACTLY when gating is disabled.
        raw = cache['x'] + delta
        return raw[:,:,:h,:w], delta[:,:,:h,:w], gate

    def forward(self, x: torch.Tensor, return_latents: bool = False) -> dict[str, torch.Tensor]:
        """Same predicted conditions in train/eval; no GT/level/debug input."""
        if not isinstance(return_latents, bool):
            raise TypeError("return_latents must be bool.")
        cache = self._encode(x)
        raw,_,_ = self._decode(cache)
        result = {"enhanced": raw}
        if return_latents:
            result.update(Fs=cache['Fs'], Fd=cache['Fd'], c_hat=cache['c_hat'])
        return result


def model_options(cfg):
    """Read the frozen A0 options without accepting another architecture."""
    if cfg.get("variant", "A0") != "A0":
        raise ValueError("Only variant A0 is supported.")
    options = dict(cfg.get("model", {}))
    if options.get("affine_mode", "bounded") != "bounded" or options.get("residual_gate", False) is not False:
        raise ValueError("Only bounded A0 without a residual gate is supported.")
    return options
