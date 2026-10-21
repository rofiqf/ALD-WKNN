"""Shared utilities: run ids, deterministic seeds, config and JSON helpers."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dump_config(config: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, default_flow_style=False, sort_keys=False)


def make_run_id(seed: int, run_name: str, timestamp: dt.datetime | None = None) -> str:
    ts = (timestamp or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_seed{seed}_{run_name}"


def derive_seed(global_seed: int, *parts: int) -> int:
    """Deterministic per-fold seed derived from the global seed."""
    seq = np.random.SeedSequence([int(global_seed), *[int(p) for p in parts]])
    return int(seq.generate_state(1)[0])


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def params_to_jsonable(params: dict) -> dict:
    return json.loads(json.dumps(params, default=json_default))
