"""Atomic JSON and consistent CSV writers; unavailable values stay explicit NA."""
from __future__ import annotations
import csv
import json
import math
import warnings
from pathlib import Path
import numpy as np
from .paths import output_path


def write_json(path, value):
    def sanitize(item):
        if isinstance(item, np.generic):
            return sanitize(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return 'Infinity' if item > 0 else '-Infinity' if item < 0 else 'NaN'
        if isinstance(item, dict):
            return {key: sanitize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(val) for val in item]
        return item
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(sanitize(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows, allow_locked=False):
    """Write a complete atomic snapshot, retaining a sidecar if a viewer locks it.

    The opt-in fallback is for live training logs. The pending snapshot is read
    by read_csv and committed on the next successful write; metadata/evaluation
    writers still fail explicitly if their destination cannot be replaced.
    """
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    pending = path.with_name(path.name + '.pending.csv')
    temporary = pending.with_suffix(pending.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(pending)
    try:
        pending.replace(path)
    except PermissionError:
        if not allow_locked:
            raise
        warnings.warn(f'CSV destination is locked; latest complete log retained at {pending}', RuntimeWarning)
    return pending if pending.exists() else path


def read_csv(path):
    path = Path(path)
    pending = path.with_name(path.name + '.pending.csv')
    if pending.exists():
        path = pending
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))
