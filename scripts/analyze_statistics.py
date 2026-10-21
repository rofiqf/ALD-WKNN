"""Cross-dataset statistical analysis from a run's metrics_outer.csv.

Writes dataset-level aggregates, mean ranks, the Friedman omnibus test, and —
only when the omnibus test justifies post-hoc analysis—pairwise Wilcoxon
signed-rank tests with Holm correction, plus rank-biserial effect sizes.
Fully reproducible from the CSV artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _helpers as helpers

import numpy as np
import pandas as pd

from aldwknn.reporting import ensure_dir, write_csv
from aldwknn.statistics import (
    dataset_aggregates,
    friedman_test,
    mean_ranks,
    rank_biserial,
    score_matrix,
    wilcoxon_pairwise,
)
from aldwknn.utils import load_config


def analyze(cfg: dict) -> Path:
    metric = cfg.get("evaluation", {}).get("primary_metric", "f1_macro")
    out_dir = ensure_dir(Path(cfg.get("project", {}).get("output_root", "artifacts")) / "statistics")

    run_dir = helpers.resolve_main_run(cfg)
    if run_dir is None:
        raise FileNotFoundError(
            "No main run with metrics_outer.csv found under artifacts/runs"
        )
    metrics = pd.read_csv(run_dir / "metrics_outer.csv")
    print(f"[stats] using run {run_dir.name} ({len(metrics)} fold rows)", flush=True)

    aggregates = dataset_aggregates(metrics, metric)
    write_csv(aggregates, out_dir / "dataset_aggregates.csv")

    mat = score_matrix(aggregates, metric)
    ranks = mean_ranks(mat).rename("mean_rank").reset_index().rename(columns={"index": "method"})
    write_csv(ranks, out_dir / "mean_ranks.csv")

    focus = "ald_wknn" if "ald_wknn" in mat.columns else mat.columns[-1]
    others = [c for c in mat.columns if c != focus]

    try:
        fried = friedman_test(mat)
        friedman_df = pd.DataFrame([{**fried, "metric": metric, "focus": focus}])
        friedman_note = ""
    except ValueError as exc:
        friedman_df = pd.DataFrame(
            [{ "n_datasets": int(mat.shape[0]), "n_methods": int(mat.shape[1]),
               "statistic": np.nan, "df": int(mat.shape[1] - 1), "p_value": np.nan,
               "kendall_w": np.nan, "metric": metric, "focus": focus }]
        )
        friedman_note = f"Friedman test skipped: {exc}"
        print(f"[stats] {friedman_note}", flush=True)
    write_csv(friedman_df, out_dir / "friedman.csv")

    effect_rows = []
    for other in others:
        d = (mat[focus] - mat[other]).to_numpy(dtype=np.float64)
        effect_rows.append(
            {
                "comparison": f"{focus} vs {other}",
                "method_a": focus,
                "method_b": other,
                "n_datasets": int(mat.shape[0]),
                "median_difference": float(np.median(d)),
                "mean_difference": float(np.mean(d)),
                "rank_biserial": rank_biserial(d),
            }
        )
    write_csv(pd.DataFrame(effect_rows), out_dir / "effect_sizes.csv")

    p_value = friedman_df["p_value"].iloc[0]
    if pd.notna(p_value) and p_value < 0.05:
        pairwise = wilcoxon_pairwise(mat, focus=focus)
        write_csv(pairwise, out_dir / "wilcoxon_pairwise.csv")
        holm = pairwise[
            ["comparison", "method_a", "method_b", "n_datasets",
             "median_difference", "p_raw", "p_holm", "reject_holm", "rank_biserial"]
        ]
        write_csv(holm, out_dir / "holm_pairwise.csv")
        print(f"[stats] omnibus p={p_value:.4g} < 0.05: pairwise Wilcoxon + Holm written", flush=True)
    else:
        print(f"[stats] omnibus p={p_value}: post-hoc tests skipped (exploratory only)", flush=True)
    return out_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cross-dataset statistical analysis.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    analyze(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
