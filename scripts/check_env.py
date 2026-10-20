"""Verify testmodel, the actual CUDA device, tensor arithmetic, and convolution backward."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import torch
import torchvision

from utils.experiment import assert_environment, experiment_config, run_path
from utils.runtime import PROJECT_ROOT


def check_environment(cfg):
    assert_environment(cfg)
    directory = run_path(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    conda = shutil.which("conda.exe") or shutil.which("conda")
    commands = {}
    for name, command in (("conda_env_list", [conda, "env", "list"] if conda else None),
                          ("nvidia_smi", ["nvidia-smi"])):
        if command:
            process = subprocess.run(command, capture_output=True, text=True, errors="replace")
            commands[name] = {"exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
        else:
            commands[name] = {"exit_code": None, "reason": "conda launcher not on PATH; explicit interpreter prefix checked"}
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")}
    report = {"executable": sys.executable, "prefix": sys.prefix, "python": sys.version,
              "platform": platform.platform(), "torch": torch.__version__, "torchvision": torchvision.__version__,
              "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "ram_total_bytes": psutil.virtual_memory().total, "ram_available_bytes": psutil.virtual_memory().available,
              "commands": commands, "versions": versions, "cuda_tests_passed": False}
    if not torch.cuda.is_available():
        (directory / "environment_report.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError("CUDA unavailable: formal training is prohibited.")
    torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    device = torch.device(cfg.get("device", "cuda"))
    props = torch.cuda.get_device_properties(device)
    report.update(gpu=props.name, vram_total_bytes=props.total_memory,
                  vram_free_bytes=torch.cuda.mem_get_info(device)[0], compute_capability=[props.major, props.minor])
    matrix = torch.rand(32, 32, device=device)
    assert torch.isfinite(matrix @ matrix).all().item()
    conv = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
    x = torch.rand(2, 3, 32, 32, device=device, requires_grad=True)
    conv(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in conv.parameters())
    torch.cuda.synchronize(device)
    report["cuda_tests_passed"] = True
    setup_log = PROJECT_ROOT / "setup_commands.log"
    report["installation_commands"] = setup_log.read_text(encoding="utf-8") if setup_log.exists() else "NA: no setup log"
    (directory / "environment_report.txt").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "environment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lock = "# Actual installed versions; install torch/torchvision separately from the official CUDA index.\n"
    lock += "\n".join(f"{name}=={version}" for name, version in sorted(versions.items(), key=lambda p: p[0].lower())) + "\n"
    (PROJECT_ROOT / "requirements-lock.txt").write_text(lock, encoding="utf-8")
    (directory / "requirements-lock.txt").write_text(lock, encoding="utf-8")
    print(f"Environment verified: {sys.executable}\nTorch {torch.__version__}, torchvision {torchvision.__version__}, "
          f"CUDA {torch.version.cuda}; {props.name}; VRAM {props.total_memory / 2**20:.0f} MiB", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sddnet_v1_100e.yaml")
    args = parser.parse_args()
    check_environment(experiment_config(args.config))


if __name__ == "__main__":
    main()
