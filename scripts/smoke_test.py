"""Independent CUDA/gradient/sampling/metric/resume tests before formal training."""

from __future__ import annotations

import argparse
import json
import math
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from datasets import MILLTripletDataset
from losses import SDDLoss, ssim
from models import SDDNetV1
from utils.evaluation import write_json
from utils.experiment import (assert_environment, experiment_config, manifest_identity, run_path,
                              training_contract)
from utils.metrics import load_lpips, psnr_rgb
from utils.runtime import (capture_rng, load_checkpoint, make_loader, make_scaler, project_path,
                           restore_rng, save_checkpoint, seed_everything)
from utils.training import accumulation_samples, concatenate_triplets


def sampling_figure(dataset, destination):
    """Save actual synchronized triplet patches and raw record IDs for inspection."""
    random.seed(42)
    indices = dataset.sample_indices(0)
    # Reset so __getitem__ selects those same records before crop/augmentation.
    random.seed(42)
    sample = dataset[0]
    size = dataset.patch_size
    canvas = Image.new("RGB", (3 * size, 2 * size + 64), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (name, index) in enumerate(zip(("anchor", "positive", "negative"), indices)):
        record = dataset.records[index]
        draw.text((column * size + 4, 3), f"{name}: {record.scene_id}, L{int(record.illum_level)}", fill="black")
        for row, key in enumerate(("low", "gt")):
            array = (sample[f"{name}_{key}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(array), (column * size, 30 + row * (size + 30)))
    canvas.save(destination)


def smoke_test(cfg):
    assert_environment(cfg)
    seed_everything(cfg["seed"] + 1000)  # Formal initialization occurs in another process.
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    directory = run_path(cfg) / "smoke"
    directory.mkdir(parents=True, exist_ok=True)
    result = {"passed": False, "manifest_hashes": manifest_identity(cfg), "contract": training_contract(cfg), "checks": {}}
    device = torch.device(cfg["device"])
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke test requires the real CUDA GPU.")
    dataset = MILLTripletDataset(project_path(cfg["data"]["mill_root"]), patch_size=cfg["patch_size"],
                                 metadata=run_path(cfg) / "metadata" / "mill.csv", illum_min=1, illum_max=10)
    for anchor_index in range(len(dataset)):
        for _ in range(4):
            a, p, n = (dataset.records[i] for i in dataset.sample_indices(anchor_index))
            assert a.scene_id == p.scene_id and a.illum_level != p.illum_level
            assert a.scene_id != n.scene_id and a.illum_level == n.illum_level
    result["checks"]["triplet_semantics"] = {"anchors": len(dataset), "samples_checked": len(dataset) * 4}
    sampling_figure(dataset, directory / "triplet_sampling.png")
    loader = make_loader(dataset, batch_size=cfg["triplet_batch_size"], workers=0, seed=123)
    low, gt, levels = concatenate_triplets(next(iter(loader)), device)
    model = SDDNetV1().to(device)
    criterion = SDDLoss(**cfg["loss"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=cfg["min_lr"])
    scaler = make_scaler(cfg["amp"])
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=cfg["amp"]):
        outputs = model(low)
        losses = criterion(outputs, gt, levels)
    assert low.shape == (3 * cfg["triplet_batch_size"], 3, cfg["patch_size"], cfg["patch_size"])
    assert all(torch.isfinite(value).all().item() for value in outputs.values())
    assert all(torch.isfinite(value).item() for value in losses.values())
    scaler.scale(losses["total"]).backward()
    scaler.unscale_(optimizer)
    grad_norms = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all().item(), name
        if name.startswith(("structure_branch", "degradation_branch")):
            grad_norms[name] = parameter.grad.norm().item()
    assert any(value > 0 for name, value in grad_norms.items() if name.startswith("structure_branch"))
    assert any(value > 0 for name, value in grad_norms.items() if name.startswith("degradation_branch"))
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    torch.cuda.synchronize(device)
    result["checks"]["cuda_training_step"] = {"input_shape": list(low.shape), "losses": {k: v.item() for k, v in losses.items()},
        "all_gradients_finite": True, "both_branches_have_nonzero_gradients": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    optimizer.zero_grad(set_to_none=True)
    del outputs, losses, low, gt, levels
    model.eval()
    with torch.inference_mode():
        odd = model(torch.rand(1, 3, 257, 321, device=device))
    assert odd["enhanced"].shape == (1, 3, 257, 321)
    result["checks"]["odd_shape"] = list(odd["enhanced"].shape)
    del odd
    # Test all training state components, then compare the next optimizer step.
    sample = torch.rand(3, 3, 32, 40, device=device)
    targets, labels = torch.rand_like(sample), torch.rand(3, 1, device=device)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "rng": capture_rng(), "loader_rng": loader.generator.get_state(), "epoch": 3}
    checkpoint_path = directory / "resume_test.pth"
    save_checkpoint(checkpoint_path, state)
    saved = load_checkpoint(checkpoint_path)
    expected_random = (random.random(), np.random.rand(), torch.rand(3))

    def next_step(network, opt, amp_scaler):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg["amp"]):
            value = criterion(network(sample), targets, labels)["total"]
        amp_scaler.scale(value).backward()
        amp_scaler.step(opt)
        amp_scaler.update()

    next_step(model, optimizer, scaler)
    reference = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    scaler.load_state_dict(saved["scaler"])
    loader.generator.set_state(saved["loader_rng"])
    restore_rng(saved["rng"])
    assert random.random() == expected_random[0] and np.random.rand() == expected_random[1]
    torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
    next_step(model, optimizer, scaler)
    max_difference = max((value.detach().cpu() - reference[name]).abs().max().item() for name, value in model.state_dict().items())
    assert max_difference < 1e-5, max_difference
    assert scheduler.last_epoch == saved["scheduler"]["last_epoch"]
    result["checks"]["checkpoint_resume"] = {"passed": True, "next_step_max_parameter_difference": max_difference,
        "components": ["model", "optimizer", "scheduler", "scaler", "python/numpy/torch/cuda RNG", "loader_rng"]}
    del saved, reference, state, optimizer, scheduler, scaler, model, sample, targets, labels
    torch.cuda.empty_cache()
    # Validate the unequal tail accumulation denominator independently.
    assert [accumulation_samples(5, 1, 2, i) for i in range(5)] == [2, 2, 2, 2, 1]
    assert [accumulation_samples(9, 2, 2, i) for i in range(5)] == [4, 4, 4, 4, 1]
    result["checks"]["partial_accumulation_groups"] = True
    rng = np.random.default_rng(42)
    a = rng.random((37, 43, 3), dtype=np.float32)
    b = (0.8 * a + 0.2 * rng.random(a.shape, dtype=np.float32)).astype(np.float32)
    tensor_a, tensor_b = [torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(device) for x in (a, b)]
    actual = ssim(tensor_a, tensor_b).item()
    expected = structural_similarity(a, b, data_range=1, channel_axis=-1, gaussian_weights=True,
                                    sigma=1.5, use_sample_covariance=False, K1=0.01, K2=0.03, win_size=11)
    assert abs(actual - expected) < 2e-5, (actual, expected)
    same_ssim = ssim(tensor_a, tensor_a).item()
    assert abs(same_ssim - 1) < 1e-5 and math.isinf(psnr_rgb(tensor_a, tensor_a))
    lpips_model, status = load_lpips(device)
    if lpips_model is not None:
        with torch.inference_mode():
            value = lpips_model(2 * tensor_a - 1, 2 * tensor_a - 1).item()
        assert abs(value) < 1e-6
        status["same_image_value"] = value
    result["checks"]["metrics"] = {"SSIM_identical": same_ssim, "SSIM_pytorch_msssim": actual,
        "SSIM_scikit_image": float(expected), "SSIM_absolute_difference": float(abs(actual - expected)), "PSNR_identical": "Infinity", "LPIPS": status}
    result["passed"] = True
    write_json(directory / "smoke_results.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sddnet_v1_100e.yaml")
    args = parser.parse_args()
    cfg = experiment_config(args.config)
    try:
        smoke_test(cfg)
    except Exception as error:
        directory = run_path(cfg) / "smoke"
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "smoke_results.json", {"passed": False, "reason": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
