"""Train SDDNet-v1 on MILL triplets with AMP, TensorBoard, and epoch checkpoints."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

import torch
import yaml
from tqdm import tqdm

from datasets import MILLPairedDataset, MILLTripletDataset
from losses import SDDLoss
from models import SDDNetV1
from utils.evaluation import evaluate
from utils.runtime import (capture_rng, choose_device, load_checkpoint, load_config,
                           make_loader, make_scaler, project_path, restore_rng,
                           save_checkpoint, seed_everything)


def concatenate_triplets(batch, device):
    """One forward pass for [B anchors, B positives, B negatives], in that order."""
    names = ("anchor", "positive", "negative")
    low = torch.cat([batch[f"{name}_low"] for name in names], dim=0).to(device, non_blocking=True)
    gt = torch.cat([batch[f"{name}_gt"] for name in names], dim=0).to(device, non_blocking=True)
    level = torch.cat([batch[f"{name}_level"] for name in names], dim=0).to(device, non_blocking=True)
    return low, gt, level


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, settings,
                    epoch, global_step, writer=None):
    """Compute sample-weighted mean losses, including a possibly smaller final batch."""
    model.train()
    use_amp = bool(settings["amp"] and device.type == "cuda")
    sums, triplets = {}, 0
    max_steps = settings.get("max_steps")
    if max_steps is not None and max_steps < 1:
        raise ValueError("train.max_steps must be positive or null.")
    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        if max_steps is not None and step > max_steps:
            break
        low, gt, levels = concatenate_triplets(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(low)
            losses = criterion(outputs, gt, levels)
        if not torch.isfinite(losses["total"]).item():
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, step {step}.")
        scaler.scale(losses["total"]).backward()
        if settings.get("grad_clip"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        count = batch["anchor_low"].shape[0]
        values = {name: value.detach().item() for name, value in losses.items()}
        for name, value in values.items():
            sums[name] = sums.get(name, 0.0) + value * count
        triplets += count
        global_step += 1
        if step % max(1, int(settings.get("log_every", 10))) == 0 or step == 1:
            progress.set_postfix(loss=f"{values['total']:.4f}")
            if writer:
                for name, value in values.items():
                    writer.add_scalar(f"step/{name}", value, global_step)
        # Do not retain the previous iteration's six-image activation graph.
        del low, gt, levels, outputs, losses
    if not triplets:
        raise ValueError("No training batches were processed.")
    return {name: value / triplets for name, value in sums.items()}, global_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sddnet_v1_mill.yaml")
    parser.add_argument("--resume", help="Project checkpoint, typically runs/sddnet_v1_mill/latest.pt")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--epochs", type=int, help="Total target epochs, including completed epochs")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-steps", type=int, help="Smoke-test limit per epoch")
    parser.add_argument("--val-max-samples", type=int, help="Smoke-test validation limit")
    parser.add_argument("--output-dir")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if "run_dir" in config:
        if any(value is not None for value in (args.epochs, args.max_steps, args.val_max_samples, args.num_workers, args.output_dir)):
            raise ValueError("The fixed experiment takes its full training configuration from YAML; no smoke truncation flags.")
        from utils.experiment import experiment_config
        from utils.training import train
        train(experiment_config(args.config), args.resume)
        return
    settings, data, eval_config = config["train"], config["data"], config["eval"]
    for key, value in (("epochs", args.epochs), ("num_workers", args.num_workers), ("max_steps", args.max_steps)):
        if value is not None:
            settings[key] = value
    if args.val_max_samples is not None:
        eval_config["max_samples"] = args.val_max_samples
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.device:
        config["device"] = args.device
    if args.no_tensorboard:
        settings["tensorboard"] = False
    if settings["optimizer"].lower() != "adamw":
        raise ValueError("SDDNet-v1 currently supports optimizer: AdamW.")
    if settings["epochs"] < 1 or settings["learning_rate"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("epochs/lr must be positive, and weight_decay non-negative.")
    seed = int(config["seed"])
    seed_everything(seed)
    torch.set_num_threads(int(config.get("cpu_threads", 8)))
    device = choose_device(config["device"])
    output_dir = project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if settings.get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError("Install tensorboard via requirements.txt, or pass --no-tensorboard.") from error
        writer = SummaryWriter(str(output_dir / "tensorboard"))
    root = project_path(data["mill_root"])
    train_metadata = project_path(data["train_metadata"]) if data.get("train_metadata") else None
    val_metadata = project_path(data["val_metadata"]) if data.get("val_metadata") else None
    train_set = MILLTripletDataset(root, patch_size=settings["patch_size"], metadata=train_metadata,
                                  illum_min=data["illum_min"], illum_max=data["illum_max"],
                                  augment=settings.get("augment", True))
    val_set = MILLPairedDataset(root, split="validation", metadata=val_metadata)
    train_loader = make_loader(train_set, settings["batch_size"], settings["num_workers"],
                               shuffle=True, seed=seed, pin_memory=device.type == "cuda")
    val_loader = make_loader(val_set, workers=eval_config.get("num_workers", 0), seed=seed + 1)
    model = SDDNetV1().to(device)
    criterion = SDDLoss(**config["loss"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scaler = make_scaler(bool(settings["amp"] and device.type == "cuda"))
    start_epoch, global_step, best_psnr, best_epoch = 1, 0, -math.inf, 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_psnr, best_epoch = checkpoint["best_psnr"], checkpoint.get("best_epoch", 0)
        train_loader.generator.set_state(checkpoint["loader_rng"])
        restore_rng(checkpoint["rng"])
        print(f"Resumed {args.resume}; next epoch={start_epoch}, optimizer lr={optimizer.param_groups[0]['lr']:g}")
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    print(f"Device={device}; parameters={sum(p.numel() for p in model.parameters()):,}; "
          f"train={len(train_set)} triplets/epoch; validation={len(val_set)} images; "
          f"AMP={scaler.is_enabled()}")
    if start_epoch > settings["epochs"]:
        print("Checkpoint already reached the requested total epochs; nothing to train.")
        if writer:
            writer.close()
        return
    history_path = output_dir / "history.csv"
    columns = ["epoch", "train_loss", "rec", "scene", "illumination", "val_psnr", "val_ssim", "seconds"]
    try:
        for epoch in range(start_epoch, settings["epochs"] + 1):
            started = time.perf_counter()
            losses, global_step = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                                  device, settings, epoch, global_step, writer)
            metrics = evaluate(model, val_loader, device, amp=eval_config["amp"],
                               tile_size=eval_config["tile_size"], tile_overlap=eval_config["tile_overlap"],
                               max_samples=eval_config.get("max_samples"), description="Validation")
            improved = metrics["psnr"] > best_psnr
            if improved:
                best_psnr, best_epoch = metrics["psnr"], epoch
            state = {"model_name": "SDDNet-v1", "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "global_step": global_step, "best_psnr": best_psnr,
                     "best_epoch": best_epoch, "config": config,
                     "rng": capture_rng(), "loader_rng": train_loader.generator.get_state()}
            save_checkpoint(output_dir / "latest.pt", state)
            if improved:
                save_checkpoint(output_dir / "best_psnr.pt", state)
            row = dict(epoch=epoch, train_loss=losses["total"], rec=losses["rec"], scene=losses["scene"],
                       illumination=losses["illumination"], val_psnr=metrics["psnr"], val_ssim=metrics["ssim"],
                       seconds=time.perf_counter() - started)
            write_header = not history_path.exists()
            with history_path.open("a", encoding="utf-8", newline="") as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=columns)
                if write_header:
                    csv_writer.writeheader()
                csv_writer.writerow(row)
            if writer:
                for name, value in losses.items():
                    writer.add_scalar(f"epoch/train_{name}", value, epoch)
                writer.add_scalar("epoch/val_psnr", metrics["psnr"], epoch)
                writer.add_scalar("epoch/val_ssim", metrics["ssim"], epoch)
                writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)
                writer.flush()
            print(f"Epoch {epoch:03d}/{settings['epochs']:03d} | train loss {losses['total']:.6f} | "
                  f"val PSNR {metrics['psnr']:.4f} dB | val SSIM {metrics['ssim']:.6f} | "
                  f"best {best_psnr:.4f} (epoch {best_epoch}) | {row['seconds']:.1f}s", flush=True)
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    # Required for safe Windows DataLoader multiprocessing.
    torch.multiprocessing.freeze_support()
    main()
