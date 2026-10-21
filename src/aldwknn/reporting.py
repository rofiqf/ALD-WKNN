"""Artifact writers: manifests, CSV tables, Markdown tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aldwknn.utils import json_default


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_manifest(run_dir: str | Path, manifest: dict) -> None:
    run_dir = ensure_dir(run_dir)
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=json_default)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def df_to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def write_markdown_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(df_to_markdown(df) + "\n", encoding="utf-8")
