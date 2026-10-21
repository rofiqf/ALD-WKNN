"""Same data + same configuration + same seeds => bit-identical predictions."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aldwknn.model import ALDWKNNClassifier, PowerWeightedKNNClassifier


def _data(seed: int = 123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(160, 6))
    y = (X[:, 0] - X[:, 1] + 0.25 * X[:, 2] > 0).astype(int)
    return X, y


def _ald_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                ALDWKNNClassifier(
                    n_neighbors=7, density_neighbors=5, distance_power=2.0,
                    density_alpha=1.0, density_clip=3.0,
                ),
            ),
        ]
    )


def test_pipeline_predictions_bit_identical_across_runs():
    X, y = _data()
    Xq = X[:50]
    pipe1 = _ald_pipeline().fit(X, y)
    pipe2 = _ald_pipeline().fit(X, y)
    np.testing.assert_array_equal(pipe1.predict(Xq), pipe2.predict(Xq))
    np.testing.assert_array_equal(pipe1.predict_proba(Xq), pipe2.predict_proba(Xq))


def test_estimator_predictions_bit_identical_across_runs():
    X, y = _data()
    Xq = X[50:120]
    for cls in (PowerWeightedKNNClassifier, ALDWKNNClassifier):
        model1 = cls(n_neighbors=5).fit(X, y)
        model2 = cls(n_neighbors=5).fit(X, y)
        np.testing.assert_array_equal(model1.predict(Xq), model2.predict(Xq))
        np.testing.assert_array_equal(model1.predict_proba(Xq), model2.predict_proba(Xq))
