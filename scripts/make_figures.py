"""Generate the five publication figures (PNG + PDF) under artifacts/figures/."""

from __future__ import annotations

import argparse
from pathlib import Path

import _helpers as helpers

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from aldwknn.reporting import ensure_dir
from aldwknn.statistics import METHOD_LABELS, dataset_aggregates, mean_ranks, score_matrix
from aldwknn.utils import load_config

FIGURES = (
    "fig_dataset_performance",
    "fig_mean_rank",
    "fig_ald_delta",
    "fig_ablation",
    "fig_runtime",
)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def fig_dataset_performance(aggregates: pd.DataFrame, out_dir: Path) -> None:
    pivot = aggregates.pivot(index="dataset", columns="method", values="f1_macro_mean")
    pivot = pivot[[c for c in ("knn_uniform", "knn_distance", "power_wknn", "ald_wknn") if c in pivot.columns]]
    pivot.columns = [_label(c) for c in pivot.columns]
    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 by Dataset and Method")
    ax.legend(title="Method")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save(ax.get_figure(), out_dir, "fig_dataset_performance")


def fig_mean_rank(aggregates: pd.DataFrame, out_dir: Path) -> None:
    ranks = mean_ranks(score_matrix(aggregates, "f1_macro")).sort_values(ascending=True)
    labels = [_label(m) for m in ranks.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, ranks.values)
    ax.invert_yaxis()
    ax.set_xlabel("Mean rank (lower is better)")
    ax.set_title("Mean Rank across Datasets (Macro F1)")
    plt.tight_layout()
    _save(fig, out_dir, "fig_mean_rank")


def fig_ald_delta(aggregates: pd.DataFrame, out_dir: Path) -> None:
    pivot = aggregates.pivot(index="dataset", columns="method", values="f1_macro_mean")
    baseline = next((b for b in ("power_wknn", "knn_distance", "knn_uniform") if b in pivot.columns), None)
    if baseline is None or "ald_wknn" not in pivot.columns:
        print("[figures] fig_ald_delta skipped: ALD or baseline column missing", flush=True)
        return
    delta = (pivot["ald_wknn"] - pivot[baseline]).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    delta.plot(kind="barh", ax=ax)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Δ Macro F1: ALD-WKNN − {_label(baseline)}")
    ax.set_ylabel("Dataset")
    ax.set_title("Per-Dataset Effect of Density Compatibility")
    plt.tight_layout()
    _save(fig, out_dir, "fig_ald_delta")


def fig_ablation(cfg: dict, out_dir: Path) -> None:
    run_dir = helpers.resolve_ablation_run(cfg)
    if run_dir is None:
        print("[figures] fig_ablation skipped: no ablation run found", flush=True)
        return
    metrics = pd.read_csv(run_dir / "metrics_outer.csv")
    agg = dataset_aggregates(metrics, "f1_macro")
    order = [v for v in ("A0_distance_only", "A1_density_only", "A2_ald_unclipped", "A3_ald_full")
             if v in set(agg["method"])]
    sub = agg.set_index("method").loc[order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([_label(m) for m in order], sub["f1_macro_mean"],
           yerr=sub["f1_macro_std"], capsize=4)
    ax.set_ylabel("Macro F1 (mean ± std across datasets)")
    ax.set_title("Ablation: Mean Macro F1 per Variant")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    _save(fig, out_dir, "fig_ablation")


def fig_runtime(metrics: pd.DataFrame, out_dir: Path) -> None:
    timing = metrics.groupby("method", as_index=False)[["fit_seconds", "predict_seconds"]].mean()
    order = [m for m in ("knn_uniform", "knn_distance", "power_wknn", "ald_wknn")
             if m in set(timing["method"])]
    order += [m for m in timing["method"] if m not in order]
    timing = timing.set_index("method").loc[order]
    timing.index = [ _label(m) for m in timing.index ]
    fig, ax = plt.subplots(figsize=(9, 5))
    timing.plot(kind="bar", ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("Mean seconds per outer fold (log scale)")
    ax.set_title("Computational Cost by Method")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    _save(fig, out_dir, "fig_runtime")


def fig_outer_cv_descriptive(metrics: pd.DataFrame, out_dir: Path) -> None:
    """Optional descriptive boxplot; folds are NOT independent inferential samples."""
    import seaborn as sns

    show = metrics.copy()
    show["method"] = show["method"].map(_label)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=show, x="method", y="f1_macro", ax=ax)
    ax.set_xlabel("Method")
    ax.set_ylabel("Outer-fold Macro F1")
    ax.set_title("Distribution of Outer-CV Scores (descriptive only; folds are not independent)")
    plt.tight_layout()
    _save(fig, out_dir, "fig_outer_cv_descriptive")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate figures from artifacts.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    artifacts = Path(cfg.get("project", {}).get("output_root", "artifacts"))
    out_dir = ensure_dir(artifacts / "figures")
    run_dir = helpers.resolve_main_run(cfg)
    if run_dir is None:
        raise FileNotFoundError("No main run with metrics_outer.csv found under artifacts/runs")
    metrics = pd.read_csv(run_dir / "metrics_outer.csv")
    aggregates = dataset_aggregates(metrics, "f1_macro")

    fig_dataset_performance(aggregates, out_dir)
    fig_mean_rank(aggregates, out_dir)
    fig_ald_delta(aggregates, out_dir)
    fig_ablation(cfg, out_dir)
    fig_runtime(metrics, out_dir)
    try:
        fig_outer_cv_descriptive(metrics, out_dir)
    except Exception as exc:  # optional figure: never fail the pipeline
        print(f"[figures] descriptive boxplot skipped: {exc}", flush=True)
    print(f"[figures] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
