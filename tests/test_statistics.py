"""Sign conventions and sanity checks for the cross-dataset statistics module."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from aldwknn.statistics import (
    friedman_test,
    mean_ranks,
    rank_biserial,
    score_matrix,
    dataset_aggregates,
    wilcoxon_pairwise,
)
from aldwknn.utils import params_to_jsonable


def test_rank_biserial_all_positive_is_plus_one():
    assert rank_biserial(np.array([1.0, 2.0, 3.0, 4.0])) == 1.0


def test_rank_biserial_all_negative_is_minus_one():
    assert rank_biserial(np.array([-1.0, -2.0, -3.0, -4.0])) == -1.0


def test_rank_biserial_mixed_known_magnitudes():
    d = np.array([3.0, 1.0, -2.0])
    # |d| ranks: 3->3, 1->1, -2->2 ; W+ = 4, W- = 2 ; r = 2/6
    assert rank_biserial(d) == pytest.approx(1.0 / 3.0)
    d2 = np.array([1.0, -2.0, 3.0, -4.0])
    # W+ = 1+3 = 4, W- = 2+4 = 6 ; r = -0.2
    assert rank_biserial(d2) == pytest.approx(-0.2)


def test_rank_biserial_ignores_zero_differences():
    with_zeros = rank_biserial(np.array([2.0, 0.0, 2.0, -2.0]))
    without = rank_biserial(np.array([2.0, 2.0, -2.0]))
    assert with_zeros == pytest.approx(without)


def _synthetic_score_matrix(n_datasets: int = 8):
    rng = np.random.default_rng(5)
    base = 0.70 + 0.05 * rng.random(n_datasets)
    data = {
        "knn_uniform": base,
        "knn_distance": base + 0.03,
        "power_wknn": base + 0.06,
        "ald_wknn": base + 0.06 + np.where(rng.random(n_datasets) < 0.8, 0.04, -0.02),
    }
    rows = []
    for i in range(n_datasets):
        for method, scores in data.items():
            rows.append({"dataset": f"ds{i}", "method": method, "f1_macro": scores[i]})
    metrics = pd.DataFrame(rows)
    return score_matrix(dataset_aggregates(metrics))


def test_holm_adjustment_monotonic_and_dominates_raw_p():
    mat = _synthetic_score_matrix()
    result = wilcoxon_pairwise(mat, focus="ald_wknn")
    assert (result["p_holm"] >= result["p_raw"] - 1e-12).all()
    ordered = result.sort_values("p_raw").reset_index(drop=True)
    assert (ordered["p_holm"].diff().dropna() >= -1e-12).all()
    assert ordered["reject_holm"].astype(int).diff().fillna(0).max() <= 0  # monotone reject


def test_holm_sorted_input_adjusted_never_decreases():
    p_sorted = np.array([0.001, 0.01, 0.04, 0.2, 0.9])
    _, adjusted, _, _ = multipletests(p_sorted, alpha=0.05, method="holm")
    assert (adjusted >= p_sorted).all()
    assert (np.diff(adjusted) >= 0).all()


def test_friedman_rejects_matrix_with_fewer_than_two_methods():
    mat = _synthetic_score_matrix()
    with pytest.raises(ValueError):
        friedman_test(mat[["ald_wknn"]])


def test_friedman_rejects_matrix_with_two_methods():
    # scipy's friedmanchisquare requires >= 3 sample sets; document the guard.
    mat = _synthetic_score_matrix()[["power_wknn", "ald_wknn"]]
    with pytest.raises(ValueError):
        friedman_test(mat)


def test_friedman_runs_with_three_methods():
    mat = _synthetic_score_matrix()[["knn_distance", "power_wknn", "ald_wknn"]]
    out = friedman_test(mat)
    assert out["n_methods"] == 3
    assert out["df"] == 2
    assert 0.0 <= out["p_value"] <= 1.0


def test_mean_ranks_orders_methods_by_score():
    mat = _synthetic_score_matrix()
    ranks = mean_ranks(mat)
    assert list(ranks.index) == ["ald_wknn", "power_wknn", "knn_distance", "knn_uniform"]
    assert ranks.iloc[0] < ranks.iloc[-1]


def test_params_round_trip_jsonable():
    params = {"classifier__n_neighbors": np.int64(5), "classifier__distance_power": np.float64(1.5)}
    clean = params_to_jsonable({k.split("__")[-1]: v for k, v in params.items()})
    assert clean == {"n_neighbors": 5, "distance_power": 1.5}
