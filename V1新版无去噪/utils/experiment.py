"""v1_clean configuration, data identity, strict checkpoint compatibility, and run locks."""
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

DATASETS = ("mill", "lolv1", "lolv2_real", "lolv2_synthetic")
ARCH_REVISION = "v1_clean"


def experiment_config(path):
    """One formal config; CPU is permitted for checks, not silent 100-epoch training."""
    cfg = load_config(path)
    if cfg.get("arch_revision") != ARCH_REVISION:
        raise ValueError("Expected arch_revision=v1_clean; legacy architectures/configurations are incompatible.")
    if cfg["epochs"] != 100 or cfg["early_stopping"]:
        raise ValueError("Formal configuration requires 100 complete epochs with no early stopping.")
    if cfg["optimizer"] != "AdamW" or cfg["scheduler"] != "cosine":
        raise ValueError("Expected AdamW and cosine scheduler.")
    if cfg["evaluation"]["precision"] != "float32" or cfg["evaluation"]["batch_size"] != 1:
        raise ValueError("Evaluation is float32, batch1, full original resolution.")
    for key in ("patch_size", "triplet_batch_size", "accumulation_steps"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if cfg["patch_size"] < 11 or cfg["num_workers"] < 0:
        raise ValueError("SSIM requires patch_size>=11 and nonnegative workers.")
    interval = cfg.setdefault('diagnostics', {}).setdefault('grad_norm_every', 100)
    if type(interval) is not int or interval < 0:
        raise ValueError('diagnostics.grad_norm_every must be a nonnegative integer.')
    if cfg.get('variant') != 'A0' or cfg.get('ablation_version') != 1:
        raise ValueError('Only the frozen A0 configuration is supported.')
    from models.sddnet_v1 import model_options
    options=model_options(cfg)
    if options != dict(affine_mode='bounded',local_init_std=0.001,residual_gate=False,g_min=0.25):
        raise ValueError('A0 model options must match the frozen configuration.')
    if cfg.get('sampling') != dict(mode='triplet') or cfg['loss'].get('mining_mode') != 'fixed':
        raise ValueError('A0 requires fixed triplet sampling; no alternative sampler is supported.')
    if cfg['data'].get('scope') != 'mill_development':
        raise ValueError('Training must use MILL train/validation only.')
    return cfg


def training_contract(cfg):
    """Resume compares architecture and every objective/optimization schedule setting."""
    keys = ("arch_revision", "seed", "epochs", "patch_size", "triplet_batch_size",
            "accumulation_steps", "amp", "optimizer", "learning_rate", "weight_decay",
            "scheduler", "min_lr", "grad_clip_norm", "loss")
    result={key: cfg[key] for key in keys}
    if cfg.get('ablation_version'):
        result.update(ablation_version=cfg['ablation_version'],variant=cfg['variant'],
                      model=cfg.get('model',{}),sampling=cfg.get('sampling',{}),
                      data_scope=cfg['data'].get('scope'))
    return result


def object_hash(value):
    """Stable JSON fingerprint, independent of dict insertion order."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def architecture_hash(cfg):
    """Forward architecture fingerprint; initialization is a separate config factor."""
    model=dict(affine_mode='bounded',residual_gate=False,g_min=.25)
    model.update({k:v for k,v in cfg.get('model',{}).items() if k!='local_init_std'})
    return object_hash(dict(arch_revision=ARCH_REVISION,model=model,channels=[32,64,128,256],depth=[2,2,2,4]))


def validate_checkpoint(state, cfg, hashes=None):
    """Common checks, also used by explicitly labelled smoke diagnostics.

    Formal evaluation/resume must additionally use validate_formal_checkpoint.
    Diagnostic frequency is intentionally outside the numerical training contract.
    """
    if state.get("arch_revision") != ARCH_REVISION:
        raise ValueError(f"Checkpoint arch_revision={state.get('arch_revision')!r}; expected v1_clean. Legacy resume rejected.")
    if state.get("contract") != training_contract(cfg):
        raise ValueError("Checkpoint training contract does not match this configuration.")
    if hashes is not None and state.get("manifest_hashes") != hashes:
        raise ValueError("Checkpoint data fingerprint differs from the audited manifests.")
    if cfg.get('ablation_version'):
        if state.get('variant')!=cfg['variant'] or state.get('architecture_hash')!=architecture_hash(cfg):
            raise ValueError('Checkpoint variant/architecture hash mismatch. Old weights are baseline-only, not new-variant resumes.')
        if state.get('config_hash')!=object_hash(training_contract(cfg)):
            raise ValueError('Checkpoint numerical configuration hash mismatch.')


def validate_formal_checkpoint(state, cfg, hashes=None, *, for_resume=False):
    """Fail closed on smoke/missing/unknown provenance, without guessing by epoch."""
    validate_checkpoint(state, cfg, hashes)
    kind = state.get('run_kind')
    purpose = 'formal resume' if for_resume else 'formal evaluation'
    if kind != 'formal':
        description = 'a smoke checkpoint' if kind == 'smoke' else f'a checkpoint with missing/unknown run_kind={kind!r}'
        raise ValueError(f'This is {description} and cannot be used for {purpose}. '
                         "Use a checkpoint saved by formal training (run_kind='formal').")
    if for_resume:
        sampler = state.get('sampler', {})
        if 'size' not in sampler or sampler.get('position') != sampler['size']:
            raise ValueError('Formal resume requires a completed formal epoch, not a partial checkpoint.')

def assert_environment(cfg):
    if Path(sys.prefix).name.casefold() != cfg.get('require_environment', 'testmodel').casefold():
        raise RuntimeError(f'Expected testmodel; actual interpreter: {sys.executable}')


def run_path(cfg):
    from .paths import output_path
    return output_path(cfg['run_dir'])


def save_resolved(cfg):
    destination = run_path(cfg)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'config_resolved.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_identity(cfg):
    audit = read_json(run_path(cfg) / 'data_audit.json')
    if not audit['passed']:
        raise RuntimeError('Data audit failed; no training is permitted.')
    for name, expected in audit['manifest_hashes'].items():
        if file_hash(run_path(cfg) / 'metadata' / f'{name}.csv') != expected:
            raise ValueError(f'Metadata changed after audit: {name}')
    return audit['manifest_hashes']


class RunLock:
    """Exclusive PID+creation-time lock. Stale removal must be explicitly requested."""
    def __init__(self, directory, clear_stale=False):
        self.path = Path(directory) / 'run.lock'
        self.clear_stale = clear_stale
        self.owned = False

    @staticmethod
    def owner_alive(owner):
        try:
            process = psutil.Process(int(owner['pid']))
            return process.is_running() and abs(process.create_time() - owner['created']) < 1
        except psutil.AccessDenied:
            return True  # Unknown ownership is not permission to launch a duplicate.
        except (psutil.NoSuchProcess, KeyError, ValueError):
            return False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            owner = read_json(self.path)
            if self.owner_alive(owner):
                raise RuntimeError(f"Run already active: PID={owner['pid']}; {self.path}")
            if not self.clear_stale:
                raise RuntimeError(f'Stale lock: {self.path}; inspect PID then pass --clear-stale-lock.')
            self.path.unlink()
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time(),
                           command=sys.argv, started_unix=time.time()), stream)
        self.owned = True
        return self

    def __exit__(self, *args):
        if self.owned:
            self.path.unlink(missing_ok=True)


def active_project_runs():
    """Check every project output lock, so an older revision is also protected."""
    result = []
    for path in (PROJECT_ROOT / 'outputs').glob('**/run.lock'):
        owner = read_json(path)
        if owner.get('pid') != os.getpid() and RunLock.owner_alive(owner):
            result.append(dict(lock=str(path), **owner))
    return result
