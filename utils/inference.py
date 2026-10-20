"""Full-image or overlapped tiled inference with bounded GPU memory use."""

from __future__ import annotations

import torch


def tile_positions(length: int, size: int, overlap: int) -> list[int]:
    """Cover the far edge even when image dimensions are not stride multiples."""
    if length <= size:
        return [0]
    positions = list(range(0, length - size + 1, size - overlap))
    if positions[-1] != length - size:
        positions.append(length - size)
    return positions


@torch.inference_mode()
def enhance_image(model, image, device, amp=True, tile_size=512, tile_overlap=32):
    """Input [1,3,H,W] on CPU; return float32 output on CPU at original size.

    Tiles have their own illumination prediction. Overlap blending reduces seams
    but is not mathematically identical to a full-image forward pass.
    """
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("Evaluation expects a single RGB image [1, 3, H, W].")
    if tile_size < 0 or (tile_size > 0 and not 0 <= tile_overlap < tile_size):
        raise ValueError("tile_size must be >= 0; overlap must be in [0, tile_size).")
    image = image.cpu()
    h, w = image.shape[-2:]
    use_amp = bool(amp and device.type == "cuda")

    def predict(patch):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(patch.to(device, non_blocking=True))["enhanced"]
        return output.float().cpu().clamp_(0, 1)

    if tile_size == 0 or max(h, w) <= tile_size:
        return predict(image)
    enhanced = torch.zeros((1, 3, h, w), dtype=torch.float32)
    weights = torch.zeros((1, 1, h, w), dtype=torch.float32)
    for top in tile_positions(h, tile_size, tile_overlap):
        for left in tile_positions(w, tile_size, tile_overlap):
            patch = image[:, :, top:top + tile_size, left:left + tile_size]
            output = predict(patch)
            ph, pw = output.shape[-2:]
            wy = torch.hann_window(ph, periodic=False).clamp_min_(0.05)
            wx = torch.hann_window(pw, periodic=False).clamp_min_(0.05)
            weight = (wy[:, None] * wx[None, :])[None, None]
            enhanced[:, :, top:top + ph, left:left + pw] += output * weight
            weights[:, :, top:top + ph, left:left + pw] += weight
    return enhanced.div_(weights).clamp_(0, 1)
