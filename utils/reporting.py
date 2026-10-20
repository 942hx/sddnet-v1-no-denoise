"""Generate truthful Markdown and plots from real checkpoints/history/metric CSVs."""

from __future__ import annotations

import csv
import json
import math

from .experiment import DATASETS, read_json, run_path
from .runtime import load_checkpoint


def rows_if_exists(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def format_value(value, digits=5):
    try:
        number = float(value)
        return f"{number:.{digits}f}" if math.isfinite(number) else str(number)
    except (TypeError, ValueError):
        return str(value)


def generate_report(cfg):
    """An interim report labels unrun tests as NA; only epoch=100 is train complete."""
    directory = run_path(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    history = rows_if_exists(directory / "history.csv")
    summary = rows_if_exists(directory / "metrics" / "summary.csv")
    levels = rows_if_exists(directory / "metrics" / "mill_per_level.csv")
    environment = read_json(directory / "environment.json") if (directory / "environment.json").exists() else {}
    profile = read_json(directory / "metrics" / "model_profile.json") if (directory / "metrics" / "model_profile.json").exists() else {}
    last = directory / "checkpoints" / "last.pth"
    checkpoint = load_checkpoint(last) if last.exists() else {}
    epoch = checkpoint.get("epoch", 0)
    best_epoch, best_psnr = checkpoint.get("best_epoch", "NA"), checkpoint.get("best_val", "NA")
    elapsed = checkpoint.get("elapsed_seconds", "NA")
    evaluation = read_json(directory / "metrics" / "evaluation_status.json") if (directory / "metrics" / "evaluation_status.json").exists() else {}
    pipeline = read_json(directory / "pipeline_status.json") if (directory / "pipeline_status.json").exists() else {}
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    if history:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs = [int(row["epoch"]) for row in history]
        fig, ax = plt.subplots(figsize=(8, 4))
        for key in ("train_loss", "L_rec", "L_scene", "L_illum"):
            ax.plot(epochs, [float(row[key]) for row in history], label=key)
        ax.set(xlabel="Completed epoch", ylabel="Unweighted loss terms / total", title="MILL training")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "loss.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key in zip(axes, ("val_PSNR_RGB", "val_SSIM_RGB")):
            ax.plot(epochs, [float(row[key]) for row in history])
            ax.set(xlabel="Completed epoch", ylabel=key, title="Full MILL validation")
        fig.tight_layout()
        fig.savefig(figures / "validation.png", dpi=160)
        plt.close(fig)
    if levels and all(row.get("input_PSNR_RGB") != "NA" and row.get("output_PSNR_RGB") != "NA" for row in levels):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        for kind in ("input", "output"):
            ax.plot([int(row["level"]) for row in levels], [float(row[f"{kind}_PSNR_RGB"]) for row in levels], marker="o", label=kind)
        ax.set(xlabel="MILL illumination index", ylabel="PSNR_RGB (dB)", xticks=list(range(1, 11)))
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "mill_per_level.png", dpi=160)
        plt.close(fig)
    lines = ["# SDDNet-v1 MILL 100-epoch experiment", "",
             f"Actual completed epochs (last checkpoint): **{epoch}/100**.",
             f"Training complete: **{epoch == 100}**. Four-set evaluation complete: **{evaluation.get('complete', False)}**.",
             f"Pipeline state: `{pipeline.get('state', 'preflight/training')}`; PID: `{pipeline.get('pid', 'NA')}`.",
             f"Best validation epoch: **{best_epoch}**; mean PSNR_RGB: **{format_value(best_psnr)} dB**.",
             f"Recorded training/validation elapsed seconds: **{format_value(elapsed, 2)}** (excludes environment setup and final testing).", "",
             "Best weights: `checkpoints/best.pth`; reliable epoch checkpoint: `checkpoints/last.pth`.",
             "A partially processed epoch is never counted as completed. No early stopping or test-based model selection.", "",
             "## Environment", "", f"Interpreter: `{environment.get('executable', 'NA')}`.",
             f"Torch: `{environment.get('torch', 'NA')}`, torchvision: `{environment.get('torchvision', 'NA')}`, CUDA wheel runtime: `{environment.get('torch_cuda', 'NA')}`.",
             f"GPU: `{environment.get('gpu', 'NA')}`; actual VRAM bytes: `{environment.get('vram_total_bytes', 'NA')}`.",
             f"System RAM bytes: `{environment.get('ram_total_bytes', 'NA')}`.", "",
             "Exact versions and commands: `environment_report.txt`, `requirements-lock.txt`.", "",
             "## Data and supervision", "",
             "Only MILL train is used for training, and the full MILL validation split chooses the best model. LOL training splits are audited but never trained on.",
             "Pairs are matched by explicit IDs, not sorted zip. Full MILL scene IDs are preserved. Positive: same scene/different level; negative: different scene/same level.",
             "Official exclusion: corrupt MILL test `Test13-9.png`; see `data_audit.json` and content-hashed CSVs in `metadata/`.",
             "c_gt=(level-1)/9. This is an independent-branch auxiliary-supervision variant, not a verbatim reproduction of MILL's method. It estimates normalized level, not physical Lux; it does not prove noise-map estimation or strict disentanglement.", "",
             "## Metric protocol", "",
             "FP32, eval+inference_mode, batch1, native-resolution RGB [0,1], no tiles/TTA/extra crop/GT exposure adjustment. Metrics are computed before PNG quantization and averaged equally per image.",
             "PSNR_RGB=10*log10(1/MSE_RGB), not the paper's PSNR_L. SSIM: pytorch-msssim, Gaussian win11 sigma1.5, K=(0.01,0.03), data_range=1, nonnegative=False. LPIPS_Alex: official pretrained v0.1, frozen/eval, input 2*x-1, evaluation only.", "",
             "## Measured test results", "", "| Dataset | Kind | N | PSNR_RGB dB | SSIM_RGB | LPIPS_Alex | Level MAE |",
             "|---|---|---:|---:|---:|---:|---:|"]
    if summary:
        for row in summary:
            lines.append(f"| {row['dataset']} | {row['kind']} | {row['count']} | {format_value(row.get('PSNR_RGB','NA'))} | {format_value(row.get('SSIM_RGB','NA'))} | {format_value(row.get('LPIPS_Alex','NA'))} | {format_value(row.get('illumination_MAE','NA'))} |")
    else:
        for dataset in DATASETS:
            lines.append(f"| {dataset} | input/output | NA | NA | NA | NA | NA |")
        lines += ["", "NA reason: final four-set evaluation has not run yet; training/checkpoint preparation is still in progress."]
    lines += ["", "Per-image deltas, decline fraction and MILL levels: `metrics/per_image.csv`, `metrics/summary.csv`, `metrics/mill_per_level.csv`.",
              "LOL illumination MAE is NA because it has no comparable MILL level labels.", "", "## Model and performance", ""]
    if profile:
        for key in ("parameters", "parameters_M", "input_shape", "precision", "warmup", "iterations", "latency_ms_mean", "FPS",
                    "peak_train_allocated_bytes", "peak_inference_allocated_bytes", "MACs", "FLOPs"):
            lines.append(f"- {key}: {profile.get(key, 'NA')}")
        lines += ["", "CUDA-synchronized timing excludes IO, transfers, metrics and PNG writing. Allocated peaks include model tensors, not total driver/desktop VRAM use."]
    else:
        lines += ["NA: " + pipeline.get("profile_failure", "the fixed 20-warmup/100-iteration FP32 profile has not run yet.")]
    reasons = []
    if (directory / "smoke" / "smoke_results.json").exists():
        smoke = read_json(directory / "smoke" / "smoke_results.json")
        lpips = smoke.get("checks", {}).get("metrics", {}).get("LPIPS", {})
        if not lpips.get("available", False):
            reasons.append("LPIPS: " + lpips.get("reason", smoke.get("reason", "preflight pending")))
    for row in summary:
        for key in ("reason", "LPIPS_reason"):
            if row.get(key):
                reasons.append(f"{row['dataset']}/{row['kind']}: {row[key]}")
    lines += ["", "## Failures / unavailable measurements", "", *[f"- {reason}" for reason in sorted(set(reasons))]]
    if not reasons:
        lines += ["No recorded failures in completed stages. Unrun stages remain NA above."]
    lines += ["", "## Run / resume / retest", "", "```bat", "conda activate testmodel",
              "python -m scripts.run_pipeline --config configs/sddnet_v1_100e.yaml",
              f"python -m scripts.run_pipeline --config configs/sddnet_v1_100e.yaml --resume {cfg['run_dir']}/checkpoints/last.pth",
              f"python -m scripts.run_pipeline --config configs/sddnet_v1_100e.yaml --eval-only --checkpoint {cfg['run_dir']}/checkpoints/best.pth",
              "```", "", "Run commands from the project root. A live run.lock rejects a duplicate process. Inspect PID before --clear-stale-lock.",
              "", "## V2 fair comparison", "", "V1 has no independent terminal denoiser or wavelets. V2 may add an independent denoising block in a separate experiment; preserve splits, seeds, total epochs, validation selection, precision and metric protocol, and report the added parameters/time/memory."]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import argparse
    from .experiment import experiment_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sddnet_v1_100e.yaml")
    args = parser.parse_args()
    generate_report(experiment_config(args.config))


if __name__ == "__main__":
    main()
