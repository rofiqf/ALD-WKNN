"""Re-run the first dataset with an identical configuration and compare artifacts.

The re-run executes the same nested-CV pipeline with the same global seed and
the same run-name (suffix ``__reprocheck``), then compares metrics and
predictions against the stored run with tight numerical tolerance
(``np.allclose`` with ``atol=1e-12``). Exit code 0 means reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _helpers as helpers

import numpy as np
import pandas as pd

import run_experiments
from aldwknn.utils import load_config

ATOL = 1e-12
METRIC_COLUMNS = [
    "f1_macro", "balanced_accuracy", "accuracy", "precision_macro", "recall_macro",
]


def _sorted(df: pd.DataFrame, cols) -> pd.DataFrame:
    return df.sort_values(cols).reset_index(drop=True)


def compare(stored_dir: Path, new_dir: Path, dataset: str) -> list[str]:
    problems: list[str] = []

    stored_m = pd.read_csv(stored_dir / "metrics_outer.csv")
    stored_m = stored_m[stored_m["dataset"] == dataset]
    new_m = pd.read_csv(new_dir / "metrics_outer.csv")
    key = ["method", "repeat", "outer_fold"]

    if stored_m.shape[0] != new_m.shape[0]:
        problems.append(
            f"metrics row count differs: stored={stored_m.shape[0]} new={new_m.shape[0]}"
        )
        return problems

    a, b = _sorted(stored_m, key), _sorted(new_m, key)
    if not (a["method"].tolist() == b["method"].tolist()):
        problems.append("method sets differ between runs")
    for col in METRIC_COLUMNS:
        if not np.allclose(a[col].to_numpy(), b[col].to_numpy(), atol=ATOL, rtol=0.0):
            problems.append(f"metric column {col!r} differs beyond atol={ATOL}")
    if a["best_params"].tolist() != b["best_params"].tolist():
        problems.append("selected best_params differ between runs")

    stored_p = pd.read_csv(stored_dir / "predictions_outer.csv.gz")
    stored_p = stored_p[stored_p["dataset"] == dataset]
    new_p = pd.read_csv(new_dir / "predictions_outer.csv.gz")
    if stored_p.shape[0] != new_p.shape[0]:
        problems.append(
            f"prediction row count differs: stored={stored_p.shape[0]} new={new_p.shape[0]}"
        )
        return problems
    pa, pb = _sorted(stored_p, key + ["row_id"]), _sorted(new_p, key + ["row_id"])
    if not (pa["y_pred"].tolist() == pb["y_pred"].tolist()):
        problems.append("predicted labels differ between runs")
    score_cols = [c for c in pa.columns if c.startswith("score_class_")]
    if score_cols != [c for c in pb.columns if c.startswith("score_class_")]:
        problems.append("score column sets differ between runs")
    else:
        for col in score_cols:
            if not np.allclose(pa[col].to_numpy(dtype=float), pb[col].to_numpy(dtype=float),
                               atol=ATOL, rtol=0.0):
                problems.append(f"score column {col!r} differs beyond atol={ATOL}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reproducibility check for a stored run.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    stored_dir = helpers.resolve_run_by_name(cfg, args.run_name)
    if stored_dir is None:
        print(f"[repro] no stored run named {args.run_name!r}", file=sys.stderr)
        return 1
    dataset = next(iter(cfg["datasets"]))
    print(f"[repro] stored run: {stored_dir.name}; re-running dataset: {dataset}", flush=True)

    check_name = f"{args.run_name}__reprocheck"
    new_dir = run_experiments.run_pipeline(
        cfg, check_name, dataset_names=[dataset], data_root=args.data_root, kind="reprocheck"
    )

    problems = compare(stored_dir, new_dir, dataset)
    log_path = Path(cfg.get("project", {}).get("output_root", "artifacts")) / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "Reproducibility check",
        f"config: {args.config}",
        f"stored run: {stored_dir}",
        f"re-run: {new_dir}",
        f"dataset: {dataset}",
        f"tolerance: np.allclose atol={ATOL} rtol=0",
        f"result: {'PASS' if not problems else 'FAIL'}",
    ]
    lines += [f"problem: {p}" for p in problems]
    (log_path / "reproducibility_check.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
