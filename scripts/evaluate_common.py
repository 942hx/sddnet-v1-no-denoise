"""Shared CLI behind the three dataset-specific test entry points."""

from __future__ import annotations

import argparse

import torch

from datasets import MILLPairedDataset, LOLv1PairedDataset, LOLv2PairedDataset
from models import SDDNetV1
from utils.evaluation import evaluate, write_json
from utils.runtime import choose_device, load_checkpoint, load_config, make_loader, project_path


def run_test(dataset_name):
    """Keep checkpoint loading, metric protocol, and export identical across datasets."""
    parser = argparse.ArgumentParser(description=f"Evaluate SDDNet-v1 on {dataset_name}")
    parser.add_argument("--config", default="configs/sddnet_v1_mill.yaml")
    parser.add_argument("--checkpoint", default="runs/sddnet_v1_mill/best_psnr.pt")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", default=f"results/{dataset_name}")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--tile-size", type=int, help="0 disables tiling")
    parser.add_argument("--tile-overlap", type=int)
    parser.add_argument("--max-samples", type=int, help="Optional smoke-test limit")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--split", choices=["train", "validation", "test"] if dataset_name == "mill" else ["train", "test"], default="test")
    if dataset_name == "mill":
        parser.add_argument("--metadata", help="Optional MILL CSV/JSON")
    if dataset_name == "lolv2":
        parser.add_argument("--subset", choices=["real", "synthetic", "both"], default="both")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["eval"]
    for key, value in (("num_workers", args.num_workers), ("tile_size", args.tile_size),
                       ("tile_overlap", args.tile_overlap), ("max_samples", args.max_samples)):
        if value is not None:
            settings[key] = value
    torch.set_num_threads(int(config.get("cpu_threads", 8)))
    device = choose_device(args.device or config["device"])
    checkpoint = load_checkpoint(args.checkpoint)
    model = SDDNetV1().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    root_key = {"mill": "mill_root", "lolv1": "lolv1_root", "lolv2": "lolv2_root"}[dataset_name]
    root = project_path(args.data_root or config["data"][root_key])
    variants = ["real", "synthetic"] if dataset_name == "lolv2" and args.subset == "both" else [getattr(args, "subset", "")]
    for variant in variants:
        if dataset_name == "mill":
            meta_key = {"train": "train_metadata", "validation": "val_metadata", "test": "test_metadata"}[args.split]
            metadata = args.metadata or config["data"].get(meta_key)
            dataset = MILLPairedDataset(root, split=args.split, metadata=project_path(metadata) if metadata else None)
        elif dataset_name == "lolv1":
            dataset = LOLv1PairedDataset(root, split=args.split)
        else:
            dataset = LOLv2PairedDataset(root, subset=variant, split=args.split)
        loader = make_loader(dataset, workers=settings.get("num_workers", 0), seed=config["seed"])
        output_dir = project_path(args.output_dir) / args.split
        if variant:
            output_dir = output_dir / variant
        metrics = evaluate(model, loader, device, amp=settings["amp"] and not args.no_amp,
                           tile_size=settings["tile_size"], tile_overlap=settings["tile_overlap"],
                           max_samples=settings.get("max_samples"), output_dir=output_dir,
                           save_images=settings.get("save_images", True) and not args.no_save_images)
        metrics.update(dataset=dataset_name, subset=variant or None, split=args.split,
                       checkpoint=str(project_path(args.checkpoint)), epoch=checkpoint.get("epoch"))
        write_json(output_dir / "metrics.json", metrics)
        print(f"{dataset_name} {variant} {args.split}: n={metrics['count']}, "
              f"PSNR={metrics['psnr']:.4f} dB, SSIM={metrics['ssim']:.6f}; output={output_dir}")
