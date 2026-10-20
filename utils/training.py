"""Complete-epoch MILL training, exact gradient accumulation, and resumable state."""

from __future__ import annotations

import csv
import json
import math
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import MILLTripletDataset
from datasets.paired_dataset import ManifestPairedDataset
from losses import SDDLoss
from models import SDDNetV1
from .experiment import (MODEL_REVISION, RunLock, assert_environment, manifest_identity,
                         read_json, run_path, save_resolved, training_contract)
from .metrics import psnr_rgb
from losses.ssim import ssim
from .runtime import (capture_rng, load_checkpoint, make_loader, make_scaler, project_path,
                      restore_rng, save_checkpoint, seed_everything)


def concatenate_triplets(batch, device):
    """Keep the a/p/n order consistent with the structure loss."""
    names = ("anchor", "positive", "negative")
    return tuple(torch.cat([batch[f"{name}_{key}"] for name in names], dim=0).to(device, non_blocking=True)
                 for key in ("low", "gt", "level"))


def accumulation_samples(total_samples, batch_size, steps, step_index):
    """Number of actual triplets in this accumulation group, including a short tail."""
    group_start = (step_index // steps) * steps
    return min(total_samples, (group_start + steps) * batch_size) - group_start * batch_size


def train_epoch(model, loader, criterion, optimizer, scaler, cfg, device, counters, writer=None, epoch=1):
    """Every dataset item is an anchor once; no debug truncation is available here."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums, count, skipped = {}, 0, 0
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        low, gt, levels = concatenate_triplets(batch, device)
        triplets = batch["anchor_low"].shape[0]
        group_samples = accumulation_samples(len(loader.dataset), cfg["triplet_batch_size"], cfg["accumulation_steps"], step)
        with torch.amp.autocast("cuda", enabled=cfg["amp"]):
            outputs = model(low)
            losses = criterion(outputs, gt, levels)
        if not all(torch.isfinite(value).item() for value in losses.values()):
            raise FloatingPointError(f"Non-finite loss: epoch {epoch}, batch {step+1}")
        scaler.scale(losses["total"] * (triplets / group_samples)).backward()
        do_update = (step + 1) % cfg["accumulation_steps"] == 0 or step + 1 == len(loader)
        if do_update:
            scaler.unscale_(optimizer)
            # AMP overflow is allowed to skip an update, but is always counted.
            if cfg["grad_clip_norm"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            overflow = scaler.get_scale() < scale_before
            skipped += int(overflow)
            counters["optimizer_updates"] += int(not overflow)
            optimizer.zero_grad(set_to_none=True)
        values = {key: val.detach().item() for key, val in losses.items()}
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + triplets * value
        count += triplets
        counters["global_step"] += 1
        if step % 25 == 0 or step + 1 == len(loader):
            print(f"epoch={epoch:03d} batch={step+1}/{len(loader)} anchors={count}/{len(loader.dataset)} "
                  f"loss={values['total']:.6f} updates={counters['optimizer_updates']} amp_overflows_epoch={skipped}", flush=True)
            if writer:
                for key, value in values.items():
                    writer.add_scalar(f"step/{key}", value, counters["global_step"])
        del low, gt, levels, outputs, losses
    if count != len(loader.dataset):
        raise RuntimeError(f"Incomplete epoch: {count}/{len(loader.dataset)} anchors.")
    counters["amp_overflows"] += skipped
    return {key: value / count for key, value in sums.items()}, count, skipped, time.perf_counter() - started


@torch.inference_mode()
def validate(model, dataset, device):
    """Full validation split, native size, batch1, FP32 and equal image weighting."""
    model.eval()
    total_psnr, total_ssim = 0.0, 0.0
    for index in range(len(dataset)):
        sample = dataset[index]
        low = sample["low"].unsqueeze(0).to(device)
        gt = sample["gt"].unsqueeze(0).to(device)
        prediction = model(low)["enhanced"]
        if not torch.isfinite(prediction).all().item():
            raise FloatingPointError(f"Non-finite validation prediction: {sample['filename']}")
        total_psnr += psnr_rgb(prediction, gt)
        total_ssim += ssim(prediction, gt).item()
    return {"PSNR_RGB": total_psnr / len(dataset), "SSIM_RGB": total_ssim / len(dataset), "count": len(dataset)}


def _history_after_resume(path, epoch):
    """Discard only rows newer than the restored reliable checkpoint (crash recovery)."""
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    kept = [row for row in rows if int(row["epoch"]) <= epoch]
    if len(kept) != len(rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(kept)


def train(cfg, resume=None):
    """Resume means finish through epoch 100, including restored scheduler/RNG state."""
    assert_environment(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for the formal experiment.")
    destination = run_path(cfg)
    with RunLock(destination):
        hashes = manifest_identity(cfg)
        smoke = read_json(destination / "smoke" / "smoke_results.json")
        if not smoke["passed"] or smoke["manifest_hashes"] != hashes or smoke["contract"] != training_contract(cfg):
            raise RuntimeError("Run the compatible smoke checks before formal training.")
        checkpoints = destination / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        history_path = destination / "history.csv"
        if not resume and (history_path.exists() or any(checkpoints.glob("*.pth"))):
            raise RuntimeError("Run contains training results. Specify --resume or a new run_dir; refusing to overwrite.")
        save_resolved(cfg)
        seed_everything(cfg["seed"])
        torch.set_num_threads(cfg.get("cpu_threads", 4))
        device = torch.device(cfg["device"])
        train_set = MILLTripletDataset(project_path(cfg["data"]["mill_root"]), patch_size=cfg["patch_size"],
                                      metadata=destination / "metadata" / "mill.csv", illum_min=1, illum_max=10)
        val_set = ManifestPairedDataset(project_path(cfg["data"]["mill_root"]), destination / "metadata" / "mill.csv", "validation")
        loader = make_loader(train_set, batch_size=cfg["triplet_batch_size"], workers=cfg["num_workers"],
                             shuffle=True, seed=cfg["seed"], pin_memory=cfg["pin_memory"])
        model = SDDNetV1().to(device)
        criterion = SDDLoss(**cfg["loss"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["min_lr"])
        scaler = make_scaler(cfg["amp"])
        start_epoch, best_epoch, best_val = 1, 0, -math.inf
        counters = {"global_step": 0, "optimizer_updates": 0, "amp_overflows": 0}
        elapsed_prior, peak_train_prior = 0.0, 0
        if resume:
            state = load_checkpoint(resume)
            if state.get("manifest_hashes") != hashes or state.get("contract") != training_contract(cfg):
                raise RuntimeError("Resume rejected: incompatible configuration or data manifests.")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            loader.generator.set_state(state["loader_rng"])
            restore_rng(state["rng"])
            start_epoch, best_epoch, best_val = state["epoch"] + 1, state["best_epoch"], state["best_val"]
            counters = state["counters"]
            elapsed_prior, peak_train_prior = state["elapsed_seconds"], state.get("peak_train_allocated_bytes", 0)
            _history_after_resume(history_path, state["epoch"])
            print(f"RESUME completed_epoch={state['epoch']}, next_epoch={start_epoch}, target=100", flush=True)
        writer = SummaryWriter(str(destination / "tensorboard"), purge_step=start_epoch if resume else None)
        session_start = time.perf_counter()
        try:
            for epoch in range(start_epoch, cfg["epochs"] + 1):
                epoch_start = time.perf_counter()
                lr = optimizer.param_groups[0]["lr"]
                torch.cuda.reset_peak_memory_stats(device)
                loss, anchors, overflows, train_seconds = train_epoch(model, loader, criterion, optimizer, scaler,
                                                                     cfg, device, counters, writer, epoch)
                torch.cuda.synchronize(device)
                peak_train = torch.cuda.max_memory_allocated(device)
                peak_train_prior = max(peak_train_prior, peak_train)
                metrics = validate(model, val_set, device)
                scheduler.step()  # Once per completed epoch, saved in post-epoch state.
                improved = metrics["PSNR_RGB"] > best_val
                if improved:
                    best_val, best_epoch = metrics["PSNR_RGB"], epoch
                elapsed = elapsed_prior + time.perf_counter() - session_start
                state = {"model_revision": MODEL_REVISION, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                         "global_step": counters["global_step"], "counters": dict(counters), "best_val": best_val,
                         "best_epoch": best_epoch, "config": cfg, "contract": training_contract(cfg),
                         "manifest_hashes": hashes, "rng": capture_rng(), "loader_rng": loader.generator.get_state(),
                         "elapsed_seconds": elapsed, "peak_train_allocated_bytes": peak_train_prior,
                         "validation": metrics}
                if improved:
                    save_checkpoint(checkpoints / "best.pth", state)
                save_checkpoint(checkpoints / "last.pth", state)
                if epoch % cfg["save_every"] == 0:
                    save_checkpoint(checkpoints / f"epoch_{epoch:03d}.pth", state)
                row = {"epoch": epoch, "anchors": anchors, "batches": len(loader), "train_loss": loss["total"],
                       "L_rec": loss["rec"], "L1": loss["l1"], "SSIM_loss": loss["ssim_loss"], "L_scene": loss["scene"],
                       "L_illum": loss["illumination"], "val_PSNR_RGB": metrics["PSNR_RGB"], "val_SSIM_RGB": metrics["SSIM_RGB"],
                       "val_count": metrics["count"], "lr": lr, "next_lr": scheduler.get_last_lr()[0],
                       "optimizer_updates_total": counters["optimizer_updates"], "AMP_overflows": overflows,
                       "train_seconds": train_seconds, "epoch_seconds": time.perf_counter() - epoch_start,
                       "elapsed_seconds": elapsed, "peak_train_allocated_bytes": peak_train,
                       "best_epoch": best_epoch, "best_val_PSNR_RGB": best_val}
                existed = history_path.exists()
                with history_path.open("a", encoding="utf-8", newline="") as stream:
                    csv_writer = csv.DictWriter(stream, fieldnames=list(row))
                    if not existed:
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                for name, value in row.items():
                    if isinstance(value, (float, int)):
                        writer.add_scalar(f"epoch/{name}", value, epoch)
                writer.flush()
                status = {"state": "training" if epoch < 100 else "training_complete", "completed_epoch": epoch,
                          "target_epochs": 100, "best_epoch": best_epoch, "best_val_PSNR_RGB": best_val,
                          "training_pid": os.getpid(), "elapsed_seconds": elapsed,
                          "best_checkpoint": str(checkpoints / "best.pth"), "last_checkpoint": str(checkpoints / "last.pth")}
                (destination / "training_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                print(f"EPOCH {epoch:03d}/100 COMPLETE anchors={anchors} loss={loss['total']:.6f} "
                      f"val_PSNR_RGB={metrics['PSNR_RGB']:.4f} val_SSIM_RGB={metrics['SSIM_RGB']:.6f} "
                      f"best_epoch={best_epoch} best={best_val:.4f} seconds={row['epoch_seconds']:.1f} "
                      f"peak_train_MiB={peak_train/2**20:.1f} amp_overflows={overflows}", flush=True)
                # A truthful report and curves stay current even before the long run finishes.
                from .reporting import generate_report
                generate_report(cfg)
        finally:
            writer.close()
        return checkpoints / "best.pth"
