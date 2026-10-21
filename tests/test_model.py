"""alpha=0 disables the density term: ALD must match Power-WKNN exactly."""

import numpy as np

from aldwknn.model import ALDWKNNClassifier, PowerWeightedKNNClassifier


def _synthetic_data(seed: int = 7, n: int = 120, m: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def test_alpha_zero_matches_power_wknn_predictions():
    X, y = _synthetic_data()
    common = dict(n_neighbors=7, distance_power=2.0, epsilon=1e-12)
    ald = ALDWKNNClassifier(density_alpha=0.0, density_neighbors=5, **common)
    power = PowerWeightedKNNClassifier(**common)
    ald.fit(X, y)
    power.fit(X, y)
    Xq = X[:25]
    np.testing.assert_array_equal(ald.predict(Xq), power.predict(Xq))


def test_alpha_zero_matches_power_wknn_proba():
    X, y = _synthetic_data()
    common = dict(n_neighbors=5, distance_power=1.0, epsilon=1e-12)
    ald = ALDWKNNClassifier(density_alpha=0.0, density_neighbors=3, **common)
    power = PowerWeightedKNNClassifier(**common)
    ald.fit(X, y)
    power.fit(X, y)
    Xq = X[:40]
    np.testing.assert_array_equal(ald.predict_proba(Xq), power.predict_proba(Xq))
