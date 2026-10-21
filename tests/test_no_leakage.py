"""Outer-test values must not influence fitted scaler statistics, density values, or tuning."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aldwknn.data import EXPECTED_SCHEMA
from aldwknn.model import ALDWKNNClassifier
from aldwknn.utils import derive_seed

SEED = 20261021


def _split(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(cv.split(X, y))
    return train_idx, test_idx


def _make_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", ALDWKNNClassifier(n_neighbors=5, density_neighbors=3)),
        ]
    )


def test_outer_test_does_not_alter_fitted_state():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    train_idx, test_idx = _split(X, y)
    X_train, X_test = X[train_idx], X[test_idx].copy()

    pipe_a = _make_pipeline().fit(X_train, y[train_idx])
    stats_a = (
        pipe_a.named_steps["scaler"].mean_.copy(),
        pipe_a.named_steps["scaler"].var_.copy(),
        pipe_a.named_steps["classifier"].rho_train_.copy(),
        pipe_a.named_steps["classifier"].r_train_.copy(),
    )

    X_test[:] = X_test * 1e6 + 12345.0  # total manipulation of the outer test
    np.random.default_rng(0).shuffle(X_test)
    pipe_b = _make_pipeline().fit(X_train, y[train_idx])
    stats_b = (
        pipe_b.named_steps["scaler"].mean_,
        pipe_b.named_steps["scaler"].var_,
        pipe_b.named_steps["classifier"].rho_train_,
        pipe_b.named_steps["classifier"].r_train_,
    )
    for a, b in zip(stats_a, stats_b):
        np.testing.assert_array_equal(a, b)


def test_outer_test_does_not_alter_selected_hyperparameters():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(150, 4))
    y = (X[:, 0] > 0).astype(int)
    train_idx, test_idx = _split(X, y)
    X_train, X_test = X[train_idx], X[test_idx].copy()

    grid = {"classifier__n_neighbors": [3, 5, 9]}
    inner_seed = derive_seed(SEED, 0, 0, 0)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=inner_seed)

    def _search():
        search = GridSearchCV(
            _make_pipeline(), grid, scoring="f1_macro", cv=inner_cv, n_jobs=1
        )
        search.fit(X_train, y[train_idx])
        return search

    search_a = _search()
    X_test[:] = X_test * -1e4 + 7.0
    search_b = _search()
    assert search_a.best_params_ == search_b.best_params_
    assert search_a.best_score_ == search_b.best_score_
    np.testing.assert_array_equal(
        search_a.best_estimator_.named_steps["classifier"].rho_train_,
        search_b.best_estimator_.named_steps["classifier"].rho_train_,
    )


def test_expected_schema_present_for_benchmark():
    assert len(EXPECTED_SCHEMA) == 8
    assert EXPECTED_SCHEMA["iris"] == {"n_samples": 150, "n_features": 4, "n_classes": 3}
