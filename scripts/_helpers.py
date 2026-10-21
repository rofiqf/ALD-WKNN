"""Shared helpers for pipeline scripts: import bootstrap and run-directory resolution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def runs_root(cfg: dict) -> Path:
    root = Path(cfg.get("project", {}).get("output_root", "artifacts"))
    return root / "runs"


def iter_run_dirs(cfg: dict) -> list[Path]:
    root = runs_root(cfg)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name != "latest")


def run_kind(run_dir: Path) -> str:
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            return str(data.get("kind", "main"))
        except json.JSONDecodeError:
            pass
    return "main"


def resolve_run_by_name(cfg: dict, run_name: str) -> Path | None:
    candidates = [p for p in iter_run_dirs(cfg) if p.name.endswith(f"_{run_name}")]
    return candidates[-1] if candidates else None


def resolve_main_run(cfg: dict) -> Path | None:
    """The run that holds the primary benchmark results (non-ablation)."""
    root = runs_root(cfg)
    latest = root / "latest"
    if latest.exists():
        target = latest.resolve()
        if (target / "metrics_outer.csv").exists() and run_kind(target) == "main":
            return target
    mains = [p for p in iter_run_dirs(cfg) if run_kind(p) == "main"
             and (p / "metrics_outer.csv").exists()]
    return mains[-1] if mains else None


def resolve_ablation_run(cfg: dict) -> Path | None:
    ablations = [p for p in iter_run_dirs(cfg) if run_kind(p) == "ablation"
                 and (p / "metrics_outer.csv").exists()]
    return ablations[-1] if ablations else None


def update_latest_link(cfg: dict, run_dir: Path) -> Path:
    root = runs_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    link = root / "latest"
    try:
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            else:
                import shutil

                shutil.rmtree(link)
        link.symlink_to(run_dir.name, target_is_directory=True)
        return link
    except OSError:
        pointer = root / "latest.txt"
        pointer.write_text(run_dir.name + "\n", encoding="utf-8")
        return pointer
