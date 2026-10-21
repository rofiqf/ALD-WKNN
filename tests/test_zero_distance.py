"""Exact duplicates must trigger the deterministic zero-distance rule, never inf/nan."""

import numpy as np

from aldwknn.model import ALDWKNNClassifier, PowerWeightedKNNClassifier


def _duplicated_data():
    X = np.array([[0.0], [0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    return X, y


def test_exact_duplicate_prediction_is_finite_and_deterministic():
    X, y = _duplicated_data()
    for cls in (ALDWKNNClassifier, PowerWeightedKNNClassifier):
        model = cls(n_neighbors=5).fit(X, y)
        pred1 = model.predict([[0.0]])
        pred2 = model.predict([[0.0]])
        assert np.array_equal(pred1, pred2)
        assert pred1[0] in (0, 1)
        proba = model.predict_proba([[0.0]])
        assert np.isfinite(proba).all()
        assert (proba >= 0.0).all() and (proba <= 1.0).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_zero_distance_rule_ignores_non_identical_neighbors():
    X = np.array([[0.0], [0.0], [0.0], [5.0], [6.0], [7.0]])
    y = np.array([1, 0, 0, 1, 1, 1])
    for cls in (ALDWKNNClassifier, PowerWeightedKNNClassifier):
        model = cls(n_neighbors=5).fit(X, y)
        # Duplicates vote 2 vs 1 for class 0; far-away neighbors must be ignored.
        assert model.predict([[0.0]])[0] == 0


def test_near_duplicate_uses_weights_not_zero_rule():
    X, y = _duplicated_data()
    for cls in (ALDWKNNClassifier, PowerWeightedKNNClassifier):
        model = cls(n_neighbors=3).fit(X, y)
        proba = model.predict_proba([[0.0 + 1e-9]])
        assert np.isfinite(proba).all()
        assert (proba > 0.0).all()  # not the duplicate rule: both classes get weight
