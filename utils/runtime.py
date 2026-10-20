"""Portable configuration, reproducibility, AMP, and resumable checkpoints."""

from __future__ import annotations

import inspect
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).absolute().parents[1]


def project_path(value) -> Path:
    """Resolve relative paths against the project, independently of shell cwd."""
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path) -> dict:
    """Load safe YAML; reject a missing/invalid config before starting CUDA."""
    with project_path(path).open("r", encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config


def seed_everything(seed: int):
    """Seed Python, NumPy, CPU and CUDA (bitwise CUDA determinism is not forced)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Top-level callable is picklable by Windows multiprocessing spawn."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def choose_device(requested="auto") -> torch.device:
    """Auto-select CUDA; explicit unavailable CUDA is an actionable error."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This project supports cpu, cuda, or cuda:N.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install a CUDA PyTorch wheel or use --device cpu.")
    return device


def make_loader(dataset, batch_size=1, workers=0, shuffle=False, seed=42, pin_memory=False):
    """Bound prefetching to keep Windows worker memory manageable."""
    if workers < 0 or batch_size < 1:
        raise ValueError("workers must be non-negative; batch_size must be positive.")
    generator = torch.Generator().manual_seed(seed)
    options = {"prefetch_factor": 1} if workers > 0 else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=pin_memory, worker_init_fn=seed_worker, generator=generator,
                      drop_last=False, persistent_workers=False, **options)


def make_scaler(enabled):
    """Use modern AMP when available, with a fallback for older installed Torch."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def load_checkpoint(path):
    """Training checkpoints include RNG/optimizer objects, so load trusted files."""
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    # Python file handles also avoid old Torch C++ Unicode-path bugs on Windows.
    with project_path(path).open("rb") as stream:
        checkpoint = torch.load(stream, **kwargs)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a project checkpoint containing a 'model' state_dict.")
    return checkpoint


def capture_rng():
    """Record all random streams needed at an epoch boundary."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    """Resume stochastic sampling; CUDA state is restored only when available."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, state):
    """Atomic replacement protects latest.pt from partially written files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
    os.replace(temporary, path)
