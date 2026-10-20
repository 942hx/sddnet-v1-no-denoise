"""Shared 100-epoch experiment contracts, manifest identity, and process locks."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import psutil
import yaml

from .runtime import PROJECT_ROOT, load_config, project_path

MODEL_REVISION = "sddnet_v1_fixed_100e_1"
DATASETS = ("mill", "lolv1", "lolv2_real", "lolv2_synthetic")


def experiment_config(path):
    """Validate experimental invariants before any optimizer or output is created."""
    cfg = load_config(path)
    if cfg.get("model_revision") != MODEL_REVISION:
        raise ValueError("Model revision does not match the fixed SDDNet-v1 implementation.")
    if cfg["epochs"] != 100 or cfg["validate_every"] != 1 or cfg["early_stopping"]:
        raise ValueError("This experiment requires 100 complete epochs, validation every epoch, no early stopping.")
    if cfg["optimizer"] != "AdamW" or cfg["scheduler"] != "CosineAnnealingLR":
        raise ValueError("Expected AdamW + CosineAnnealingLR.")
    if cfg["evaluation"]["precision"] != "float32" or cfg["evaluation"]["tile_size"] != 0:
        raise ValueError("Official evaluation requires float32 full-image inference (no tiling).")
    for key in ("patch_size", "triplet_batch_size", "accumulation_steps"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if cfg["patch_size"] < 11 or cfg["num_workers"] < 0:
        raise ValueError("SSIM needs patch_size>=11; num_workers must be non-negative.")
    if cfg["data"]["illum_min"] != 1 or cfg["data"]["illum_max"] != 10:
        raise ValueError("Fixed experiment c_gt is (level-1)/9.")
    return cfg


def assert_environment(cfg):
    """The interpreter prefix, rather than the shell prompt, identifies conda env."""
    expected = cfg.get("require_environment", "testmodel")
    if Path(sys.prefix).name.casefold() != expected.casefold():
        raise RuntimeError(f"Expected conda env {expected!r}; actual sys.executable={sys.executable}")


def run_path(cfg):
    """Return a project-relative or user-specified absolute run directory."""
    return project_path(cfg["run_dir"])


def save_resolved(cfg):
    """Record the actual configuration, preserving the original source YAML."""
    destination = run_path(cfg)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "config_resolved.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def file_hash(path):
    """Stream files; never keep the decoded dataset in RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_identity(cfg):
    audit = read_json(run_path(cfg) / "data_audit.json")
    if not audit["passed"]:
        raise RuntimeError("Dataset audit failed; see data_audit.json")
    for dataset, expected in audit["manifest_hashes"].items():
        if file_hash(run_path(cfg) / "metadata" / f"{dataset}.csv") != expected:
            raise RuntimeError(f"Metadata changed since audit: {dataset}")
    return audit["manifest_hashes"]


def training_contract(cfg):
    """Resume rejects a changed objective, patch, batch, schedule, or architecture."""
    keys = ("model_revision", "seed", "epochs", "patch_size", "triplet_batch_size",
            "accumulation_steps", "amp", "optimizer", "lr", "weight_decay", "scheduler",
            "min_lr", "grad_clip_norm", "loss")
    return {key: cfg[key] for key in keys}


class RunLock:
    """Exclusive PID+creation-time lock; refuse live owners and require stale cleanup."""

    def __init__(self, directory, clear_stale=False):
        self.path = Path(directory) / "run.lock"
        self.clear_stale = clear_stale
        self.owned = False
        self.borrowed = False

    @staticmethod
    def owner_alive(owner):
        try:
            process = psutil.Process(int(owner["pid"]))
            return process.is_running() and abs(process.create_time() - owner["created"]) < 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, ValueError):
            return False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            owner = read_json(self.path)
            if self.owner_alive(owner):
                # Only a direct pipeline child can share its parent's lock.
                if (str(owner["pid"]) == os.environ.get("SDD_PIPELINE_PID")
                        and psutil.Process().ppid() == owner["pid"]):
                    self.borrowed = True
                    return self
                raise RuntimeError(f"Run already active: PID={owner['pid']}; lock={self.path}")
            if not self.clear_stale:
                raise RuntimeError(f"Stale lock at {self.path}; inspect PID then pass --clear-stale-lock.")
            self.path.unlink()
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created": psutil.Process().create_time(),
                       "command": sys.argv, "started_unix": time.time()}, stream, ensure_ascii=False)
        self.owned = True
        return self

    def __exit__(self, *args):
        if self.owned:
            self.path.unlink(missing_ok=True)
