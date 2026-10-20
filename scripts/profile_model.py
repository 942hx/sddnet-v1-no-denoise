"""Measure real FP32 CUDA inference at [1,3,400,600], excluding IO and metrics."""

from __future__ import annotations

import argparse
import statistics

import torch

from models import SDDNetV1
from utils.evaluation import write_json
from utils.experiment import (MODEL_REVISION, assert_environment, experiment_config, file_hash,
                              manifest_identity, run_path)
from utils.runtime import load_checkpoint, project_path


@torch.inference_mode()
def profile_model(cfg, checkpoint_path):
    assert_environment(cfg)
    state = load_checkpoint(checkpoint_path)
    if state.get("model_revision") != MODEL_REVISION or state.get("manifest_hashes") != manifest_identity(cfg):
        raise ValueError("Profile checkpoint incompatible with this experiment.")
    device = torch.device(cfg["device"])
    model = SDDNetV1().to(device).float().eval()
    model.load_state_dict(state["model"])
    shape = cfg["profile"]["input_shape"]
    if shape != [1, 3, 400, 600] or cfg["profile"]["warmup"] != 20 or cfg["profile"]["iterations"] != 100:
        raise ValueError("Fixed profile requires [1,3,400,600], 20 warmups and 100 timed iterations.")
    sample = torch.rand(*shape, device=device, dtype=torch.float32)
    for _ in range(20):
        output = model(sample)
        del output
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(100):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        output = model(sample)
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))
        del output
    count = sum(p.numel() for p in model.parameters())
    last_path = run_path(cfg) / "checkpoints" / "last.pth"
    training_state = load_checkpoint(last_path) if last_path.exists() else state
    result = {"parameters": count, "parameters_M": count / 1e6, "gpu": torch.cuda.get_device_name(device),
              "vram_total_bytes": torch.cuda.get_device_properties(device).total_memory,
              "peak_train_allocated_bytes": training_state.get("peak_train_allocated_bytes", "NA"),
              "peak_inference_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_inference_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "input_shape": shape, "precision": "float32", "warmup": 20, "iterations": 100,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "latency_ms_mean": statistics.mean(times), "latency_ms_median": statistics.median(times),
              "latency_ms_p95": sorted(times)[94], "FPS": 1000.0 / statistics.mean(times),
              "timings_ms": times, "includes": "model forward, all four outputs and internal padding",
              "excludes": "disk IO, CPU/GPU transfer, metrics, image saving",
              "timing": "CUDA events with synchronization each iteration; eval+inference_mode",
              "checkpoint_sha256": file_hash(project_path(checkpoint_path)), "checkpoint_epoch": state["epoch"],
              "MACs": "NA", "FLOPs": "NA", "complexity_reason": "No operator-counting tool used; parameters counted exactly"}
    (run_path(cfg) / "metrics").mkdir(exist_ok=True)
    write_json(run_path(cfg) / "metrics" / "model_profile.json", result)
    print(f"Parameters={count/1e6:.6f} M; FP32 latency={result['latency_ms_mean']:.3f} ms; FPS={result['FPS']:.3f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sddnet_v1_100e.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    profile_model(experiment_config(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
