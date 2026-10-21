"""Generate publication tables under artifacts/tables/ from run and statistics artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _helpers as helpers

import numpy as np
import pandas as pd

from aldwknn.data import EXPECTED_SCHEMA
from aldwknn.reporting import ensure_dir, write_csv, write_markdown_table
from aldwknn.statistics import METHOD_LABELS, dataset_aggregates, mean_ranks, score_matrix
from aldwknn.utils import load_config
from run_ablation import VARIANT_FLAGS, VARIANTS


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _read_df(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def dataset_summary(cfg: dict, metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    present = list(dict.fromkeys(metrics["dataset"].tolist()))
    rows = []
    for name in present:
        exp = EXPECTED_SCHEMA[name]
        rows.append(
            {
                "dataset": name,
                "n_samples": exp["n_samples"],
                "n_features": exp["n_features"],
                "n_classes": exp["n_classes"],
            }
        )
    df = pd.DataFrame(rows)
    write_csv(df, out_dir / "dataset_summary.csv")
    write_markdown_table(df, out_dir / "dataset_summary.md")
    return df


def performance_mean_std(cfg: dict, metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    metric = cfg.get("evaluation", {}).get("primary_metric", "f1_macro")
    agg = dataset_aggregates(metrics, metric)
    write_csv(agg, out_dir / "performance_mean_std.csv")

    wide = agg.pivot(index="dataset", columns="method", values=f"{metric}_mean")
    stds = agg.pivot(index="dataset", columns="method", values=f"{metric}_std")
    methods = [m for m in ("knn_uniform", "knn_distance", "power_wknn", "ald_wknn") if m in wide.columns]
    rows = []
    for ds in wide.index:
        row = {"Dataset": ds}
        for m in methods:
            mean = wide.loc[ds, m]
            std = stds.loc[ds, m] if pd.notna(stds.loc[ds, m]) else 0.0
            row[_label(m)] = f"{mean:.3f} ± {std:.3f}"
        if "ald_wknn" in wide.columns and "power_wknn" in wide.columns:
            row["Δ ALD−PWKNN"] = f"{wide.loc[ds, 'ald_wknn'] - wide.loc[ds, 'power_wknn']:+.3f}"
        rows.append(row)
    if "ald_wknn" in wide.columns or methods:
        ranks = mean_ranks(score_matrix(agg, metric))
        rank_row = {"Dataset": "Mean rank"}
        for m in methods:
            rank_row[_label(m)] = f"{ranks[m]:.2f}" if m in ranks.index else "—"
        rank_row["Δ ALD−PWKNN"] = "—"
        rows.append(rank_row)
    df = pd.DataFrame(rows)
    write_markdown_table(df, out_dir / "performance_mean_std.md")
    return df


def ablation_summary(cfg: dict, out_dir: Path) -> None:
    path = out_dir / "ablation_summary.md"
    run_dir = helpers.resolve_ablation_run(cfg)
    if run_dir is None:
        path.write_text(
            "# Ablation summary\n\nTBD — no ablation run found under artifacts/runs.\n",
            encoding="utf-8",
        )
        return
    metrics = pd.read_csv(run_dir / "metrics_outer.csv")
    agg = dataset_aggregates(metrics, "f1_macro")
    wide = agg.pivot(index="dataset", columns="method", values="f1_macro_mean")
    ranks = mean_ranks(score_matrix(agg, "f1_macro"))
    rows = []
    for variant in VARIANTS:
        if variant not in wide.columns:
            continue
        dist, dens, clip = VARIANT_FLAGS[variant]
        rows.append(
            {
                "Variant": _label(variant),
                "Distance": dist,
                "Density compatibility": dens,
                "Clipping": clip,
                "Mean Macro F1": f"{wide[variant].mean():.3f}",
                "Mean Rank": f"{ranks[variant]:.2f}" if variant in ranks.index else "—",
            }
        )
    df = pd.DataFrame(rows)
    lines = [
        "# Ablation summary",
        "",
        f"Source run: `{run_dir.name}` (aggregate Macro F1 across datasets).",
        "",
        df.to_markdown(index=False),
        "",
        "Reading the ablation: A3 beating A0 is evidence for the density term; A3 on par with A0",
        "means no clear practical gain; A3 above A2 indicates that clipping helps; a weak A1",
        "alongside a strong A3 would show that density works as a modifier, not as a",
        "standalone signal.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def statistics_friedman(cfg: dict, out_dir: Path) -> None:
    path = out_dir / "statistics_friedman.md"
    stats_dir = Path(cfg.get("project", {}).get("output_root", "artifacts")) / "statistics"
    fried = _read_df(stats_dir / "friedman.csv")
    if fried is None:
        path.write_text("# Friedman test\n\nTBD — artifacts/statistics/friedman.csv not found.\n", encoding="utf-8")
        return
    r = fried.iloc[0]
    df_show = pd.DataFrame(
        [{
            "Metric": r["metric"],
            "Datasets": int(r["n_datasets"]),
            "Methods": int(r["n_methods"]),
            "Statistic": f"{r['statistic']:.4f}" if pd.notna(r["statistic"]) else "TBD",
            "df": int(r["df"]),
            "p-value": f"{r['p_value']:.4g}" if pd.notna(r["p_value"]) else "TBD",
            "Kendall W": f"{r['kendall_w']:.4f}" if pd.notna(r["kendall_w"]) else "TBD",
            "Decision": "reject H0 (post-hoc justified)" if pd.notna(r["p_value"]) and r["p_value"] < 0.05 else "insufficient evidence to reject H0",
        }]
    )
    if pd.notna(r["p_value"]) and r["p_value"] < 0.05:
        text = (
            f"A Friedman test detected evidence of differences among the evaluated "
            f"classifiers across the datasets, χ²F({int(r['df'])})={r['statistic']:.4f}, "
            f"p={r['p_value']:.4g}, with Kendall's W={r['kendall_w']:.4f}. Pairwise post-hoc "
            f"comparisons were therefore examined using Wilcoxon signed-rank tests with "
            f"Holm correction."
        )
    else:
        text = (
            f"The Friedman test did not provide sufficient evidence to reject the null "
            f"hypothesis of equal method ranks across the evaluated datasets"
            + (f", χ²F({int(r['df'])})={r['statistic']:.4f}, p={r['p_value']:.4g}" if pd.notna(r["p_value"]) else "")
            + ". Consequently, pairwise tests are reported, if shown at all, as "
            f"exploratory rather than as confirmatory evidence."
        )
    lines = ["# Friedman test", "", df_show.to_markdown(index=False), "", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def statistics_pairwise_holm(cfg: dict, out_dir: Path) -> None:
    path = out_dir / "statistics_pairwise_holm.md"
    stats_dir = Path(cfg.get("project", {}).get("output_root", "artifacts")) / "statistics"
    holm = _read_df(stats_dir / "holm_pairwise.csv")
    if holm is None:
        path.write_text(
            "# Pairwise comparisons (Wilcoxon + Holm)\n\n"
            "TBD — post-hoc tests were not executed "
            "(omnibus p >= 0.05, or statistics artifacts missing).\n",
            encoding="utf-8",
        )
        return
    show = holm.copy()
    show["Comparison"] = show["comparison"].map(lambda s: s.replace("_", " "))
    show["N datasets"] = show["n_datasets"]
    show["Median Δ F1"] = show["median_difference"].map("{:.4f}".format)
    show["Raw p"] = show["p_raw"].map("{:.4g}".format)
    show["Holm p"] = show["p_holm"].map("{:.4g}".format)
    show["Rank-biserial r_rb"] = show["rank_biserial"].map("{:.3f}".format)
    show["Reject H0?"] = show["reject_holm"].map({True: "yes", False: "no"})
    df_show = show[["Comparison", "N datasets", "Median Δ F1", "Raw p", "Holm p",
                    "Rank-biserial r_rb", "Reject H0?"]]
    lines = ["# Pairwise comparisons (Wilcoxon signed-rank + Holm)", "", df_show.to_markdown(index=False), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def hyperparameters(cfg: dict, run_dir: Path, out_dir: Path) -> None:
    path = out_dir / "hyperparameters.md"
    best = _read_df(run_dir / "best_params.csv")
    if best is None:
        path.write_text("# Hyperparameter selection\n\nTBD — best_params.csv not found.\n", encoding="utf-8")
        return
    lines = [
        "# Hyperparameter selection frequency",
        "",
        f"Source run: `{run_dir.name}`. Frequencies count outer-fold selections; "
        "nested CV may select different values per fold, so no single global "
        "'best parameter' is reported.",
        "",
    ]
    for method in dict.fromkeys(best["method"].tolist()):
        subset = best[best["method"] == method]
        counters: dict[str, Counter] = {}
        for raw in subset["best_params"]:
            params = json.loads(raw)
            for pname, value in params.items():
                counters.setdefault(pname, Counter())[json.dumps(value)] += 1
        n = len(subset)
        rows = []
        for pname in sorted(counters):
            for value_json, count in counters[pname].most_common():
                rows.append(
                    {
                        "Parameter": pname,
                        "Value": json.loads(value_json),
                        "Selections": count,
                        "Frequency (%)": f"{100.0 * count / n:.1f}",
                    }
                )
        lines.append(f"## {_label(method)}")
        lines.append("")
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate tables from artifacts.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = ensure_dir(Path(cfg.get("project", {}).get("output_root", "artifacts")) / "tables")
    run_dir = helpers.resolve_main_run(cfg)
    if run_dir is None:
        raise FileNotFoundError("No main run with metrics_outer.csv found under artifacts/runs")
    metrics = pd.read_csv(run_dir / "metrics_outer.csv")
    print(f"[tables] using run {run_dir.name}", flush=True)

    dataset_summary(cfg, metrics, out_dir)
    performance_mean_std(cfg, metrics, out_dir)
    ablation_summary(cfg, out_dir)
    statistics_friedman(cfg, out_dir)
    statistics_pairwise_holm(cfg, out_dir)
    hyperparameters(cfg, run_dir, out_dir)
    print(f"[tables] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
