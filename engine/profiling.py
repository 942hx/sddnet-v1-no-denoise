"""Measure real FP32 CUDA inference at [1,3,400,600], excluding IO and metrics."""

from __future__ import annotations

import argparse
import statistics

import torch

from models import SDDNetV1
from utils.io import write_json
from utils.experiment import (assert_environment, file_hash, manifest_identity, run_path, validate_checkpoint)
from utils.runtime import load_checkpoint, project_path, choose_device


@torch.inference_mode()
def profile_model(cfg, checkpoint_path=None, output_path=None, scope='architecture diagnostic'):
    assert_environment(cfg)
    state = load_checkpoint(checkpoint_path) if checkpoint_path else None
    if state:
        validate_checkpoint(state,cfg,manifest_identity(cfg))
    device = choose_device(cfg['device'])
    model = SDDNetV1().to(device).float().eval()
    if state:
        model.load_state_dict(state["model"],strict=True)
    shape = cfg["profile"]["shape"]
    if shape != [1, 3, 400, 600] or cfg["profile"]["warmup"] != 20 or cfg["profile"]["iterations"] != 100:
        raise ValueError("Fixed profile requires [1,3,400,600], 20 warmups and 100 timed iterations.")
    sample = torch.rand(*shape, device=device, dtype=torch.float32)
    if device.type != 'cuda':
        result = dict(parameters=sum(p.numel() for p in model.parameters()),arch_revision='v1_clean',scope=scope,
                      latency_ms_mean='NA',FPS='NA',peak_inference_allocated_bytes='NA',reason='SKIP: CUDA profiling not available or not selected')
        write_json(output_path or run_path(cfg)/'profile.json',result)
        return result
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
    result = {"parameters": count, "parameters_M": count / 1e6, "gpu": torch.cuda.get_device_name(device),
              "vram_total_bytes": torch.cuda.get_device_properties(device).total_memory,
              "arch_revision": "v1_clean", "scope": scope,
              "peak_train_allocated_bytes": state.get("peak_memory", "NA") if state else "NA",
              "peak_inference_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_inference_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "input_shape": shape, "precision": "float32", "warmup": 20, "iterations": 100,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "latency_ms_mean": statistics.mean(times), "latency_ms_median": statistics.median(times),
              "latency_ms_p95": sorted(times)[94], "FPS": 1000.0 / statistics.mean(times),
              "timings_ms": times, "includes": "full internal model forward and padding; default enhanced-only return",
              "excludes": "disk IO, CPU/GPU transfer, metrics, image saving",
              "timing": "CUDA events with synchronization each iteration; eval+inference_mode",
              "checkpoint_sha256": file_hash(project_path(checkpoint_path)) if state else 'NA: untrained architecture',
              "checkpoint_epoch": state['completed_epoch'] if state else 0,
              "MACs": "NA", "FLOPs": "NA", "complexity_reason": "No operator-counting tool used; parameters counted exactly"}
    write_json(output_path or run_path(cfg)/'profile.json',result)
    print(f"Parameters={count/1e6:.6f} M; FP32 latency={result['latency_ms_mean']:.3f} ms; FPS={result['FPS']:.3f}", flush=True)
    return result


