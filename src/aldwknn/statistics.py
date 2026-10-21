"""Cross-dataset statistical analysis.

Unit of inference is the DATASET, not the CV fold: fold scores are first
aggregated to one value per dataset per method, then compared with the
Friedman omnibus test, followed (only when justified) by pairwise
Wilcoxon signed-rank tests with Holm correction and rank-biserial
effect sizes.

Sign convention: differences are defined as (ALD - baseline), so
r_rb > 0 means the direction favors ALD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

METHOD_ORDER = ("knn_uniform", "knn_distance", "power_wknn", "ald_wknn")
METHOD_LABELS = {
    "knn_uniform": "Uniform KNN",
    "knn_distance": "Distance KNN",
    "power_wknn": "Power-WKNN",
    "ald_wknn": "ALD-WKNN",
    "A0_distance_only": "A0 Distance-only",
    "A1_density_only": "A1 Density-only",
    "A2_ald_unclipped": "A2 ALD-Unclipped",
    "A3_ald_full": "A3 ALD-Full",
}


def dataset_aggregates(metrics_df: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    """One aggregate score per (dataset, method) from per-fold rows."""
    agg = (
        metrics_df.groupby(["dataset", "method"], as_index=False)[metric]
        .agg(mean="mean", std="std")
        .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
    )
    return agg


def score_matrix(aggregates: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    """Datasets x methods matrix of aggregate scores."""
    return aggregates.pivot(index="dataset", columns="method", values=f"{metric}_mean")


def mean_ranks(score_mat: pd.DataFrame) -> pd.Series:
    """Average rank per method; rank 1 = best (highest score)."""
    # rank along each row, highest score gets rank 1
    ranks = score_mat.rank(axis=1, ascending=False, method="average")
    return ranks.mean(axis=0).sort_values()


def friedman_test(score_mat: pd.DataFrame) -> dict:
    """Omnibus Friedman test across datasets; also reports Kendall's W."""
    columns = [score_mat[c].to_numpy() for c in score_mat.columns]
    statistic, p_value = friedmanchisquare(*columns)
    n_datasets, n_methods = score_mat.shape
    kendall_w = statistic / (n_datasets * (n_methods - 1))
    return {
        "n_datasets": int(n_datasets),
        "n_methods": int(n_methods),
        "statistic": float(statistic),
        "df": int(n_methods - 1),
        "p_value": float(p_value),
        "kendall_w": float(kendall_w),
    }


def rank_biserial(differences: np.ndarray) -> float:
    """Rank-biserial effect size for paired differences (zeros excluded).

    r_rb = (W+ - W-) / (W+ + W-); positive values favor the first-listed
    group, i.e. (ALD - baseline) > 0 means ALD is favored.
    """
    d = np.asarray(differences, dtype=np.float64)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    return float((w_plus - w_minus) / (w_plus + w_minus))


def wilcoxon_pairwise(
    score_mat: pd.DataFrame,
    focus: str = "ald_wknn",
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests: focus method vs every other.

    Returns one row per comparison with raw p, Holm-adjusted p, median
    paired difference and rank-biserial effect size. The Holm family is
    the set of comparisons returned here.
    """
    others = [c for c in score_mat.columns if c != focus]
    rows = []
    p_values = []
    for other in others:
        d = (score_mat[focus] - score_mat[other]).to_numpy(dtype=np.float64)
        stat, p = wilcoxon(
            score_mat[focus].to_numpy(dtype=np.float64),
            score_mat[other].to_numpy(dtype=np.float64),
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        rows.append(
            {
                "comparison": f"{focus} vs {other}",
                "method_a": focus,
                "method_b": other,
                "n_datasets": int(score_mat.shape[0]),
                "median_difference": float(np.median(d)),
                "statistic": float(stat),
                "p_raw": float(p),
                "rank_biserial": rank_biserial(d),
            }
        )
        p_values.append(float(p))

    reject, p_holm, _, _ = multipletests(p_values, alpha=0.05, method="holm")
    for row, ph, rj in zip(rows, p_holm, reject):
        row["p_holm"] = float(ph)
        row["reject_holm"] = bool(rj)
    return pd.DataFrame(rows)
