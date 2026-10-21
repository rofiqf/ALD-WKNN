"""Core estimators for ALD-WKNN experiments.

Implements the weighting formulations specified in PROJECT_SPEC.md:

    w_i(x) = (d(x, x_i) + eps)^(-p)                       (distance)
    w_i(x) = exp(-alpha * min(c, |log(rho_q(x)/rho_q(x_i))|))   (density)
    w_i(x) = distance * density                            (ALD-WKNN full)

Density proxy: rho_q(z) = 1 / (r_q(z) + eps), where r_q(z) is the
distance from z to its q-th nearest neighbor (self excluded for
training points).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class _BaseWeightedKNN(BaseEstimator, ClassifierMixin):
    """Shared machinery: k-NN search, deterministic voting, zero-distance rule."""

    def __init__(self, n_neighbors: int = 5, epsilon: float = 1e-12):
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon

    # ---- subclass hook -------------------------------------------------
    def _neighbor_weights(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (distances, indices, weights) for the k classification neighbors."""
        raise NotImplementedError

    # ---- sklearn API ---------------------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64, ensure_2d=True)
        if type_of_target(y) not in ("binary", "multiclass"):
            raise ValueError(f"Unsupported target type: {type_of_target(y)}")
        self.classes_ = np.unique(y)
        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
        self._label_dtype = np.asarray(y).dtype
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y)
        self._fit_search()
        self._fit_density()
        return self

    def _fit_search(self) -> None:
        self.nn_index_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.nn_index_.fit(self.X_train_)

    def _fit_density(self) -> None:
        pass

    def predict(self, X):
        check_is_fitted(self, "X_train_")
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        scores = self._class_scores(X)
        # argmax over columns ordered by sorted classes -> deterministic
        # tie-break toward the smallest sorted encoded label.
        winner = np.argmax(scores, axis=1)
        return self.classes_[winner]

    def predict_proba(self, X):
        check_is_fitted(self, "X_train_")
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        scores = self._class_scores(X)
        totals = scores.sum(axis=1, keepdims=True)
        return scores / np.maximum(totals, self.epsilon)

    # ---- internals -----------------------------------------------------
    def _kneighbors(self, X: np.ndarray):
        n_neighbors = min(self.n_neighbors, self.X_train_.shape[0])
        distances, indices = self.nn_index_.kneighbors(X, n_neighbors=n_neighbors)
        return distances, indices

    def _class_scores(self, X: np.ndarray) -> np.ndarray:
        distances, indices, weights = self._neighbor_weights(X)
        scores = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=np.float64)
        encoded = np.searchsorted(self.classes_, self.y_train_)
        for row in range(X.shape[0]):
            dup_mask = distances[row] <= self.epsilon
            if dup_mask.any():
                # Zero-distance rule: vote only among exact duplicates.
                w = dup_mask.astype(np.float64)
            else:
                w = weights[row]
            np.add.at(
                scores[row],
                encoded[indices[row]],
                w,
            )
        if not np.isfinite(scores).all():
            raise FloatingPointError("Non-finite class scores encountered.")
        return scores


class PowerWeightedKNNClassifier(_BaseWeightedKNN):
    """w_i = (d_i + eps)^(-p). With p=0 this reduces to uniform KNN."""

    def __init__(
        self,
        n_neighbors: int = 5,
        distance_power: float = 1.0,
        epsilon: float = 1e-12,
    ):
        super().__init__(n_neighbors=n_neighbors, epsilon=epsilon)
        self.distance_power = distance_power

    def _neighbor_weights(self, X):
        distances, indices = self._kneighbors(X)
        weights = np.power(distances + self.epsilon, -self.distance_power)
        return distances, indices, weights


class ALDWKNNClassifier(_BaseWeightedKNN):
    """ALD-WKNN: distance weight x bounded density-compatibility factor.

    g_rho(x, x_i) = exp(-alpha * min(c, |log((rho_q(x)+eps)/(rho_q(x_i)+eps))|))
    w_i = (d_i + eps)^(-p) * g_rho
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        density_neighbors: int = 5,
        distance_power: float = 1.0,
        density_alpha: float = 1.0,
        density_clip: float = 3.0,
        epsilon: float = 1e-12,
    ):
        super().__init__(n_neighbors=n_neighbors, epsilon=epsilon)
        self.density_neighbors = density_neighbors
        self.distance_power = distance_power
        self.density_alpha = density_alpha
        self.density_clip = density_clip

    # ---- density fitting ----------------------------------------------
    def _fit_density(self) -> None:
        q = int(self.density_neighbors)
        n = self.X_train_.shape[0]
        # search q+1 neighbors so the q-th neighbor remains after self-removal
        n_search = min(q + 1, n)
        nn = NearestNeighbors(n_neighbors=n_search)
        nn.fit(self.X_train_)
        distances, indices = nn.kneighbors(self.X_train_)
        r_q = np.empty(n, dtype=np.float64)
        for i in range(n):
            keep = indices[i] != i
            r_q[i] = distances[i][keep][q - 1]
        self.r_train_ = r_q
        self.rho_train_ = 1.0 / (r_q + self.epsilon)

    # ---- prediction -----------------------------------------------------
    def _neighbor_weights(self, X):
        distances, indices = self._kneighbors(X)

        # query density proxy from the q-th training neighbor
        q = int(self.density_neighbors)
        nn_q = NearestNeighbors(n_neighbors=min(q, self.X_train_.shape[0]))
        nn_q.fit(self.X_train_)
        d_q, _ = nn_q.kneighbors(X)
        rho_query = 1.0 / (d_q[:, -1] + self.epsilon)

        rho_neighbors = self.rho_train_[indices]
        ratio = (rho_query[:, None] + self.epsilon) / (rho_neighbors + self.epsilon)
        log_ratio = np.abs(np.log(ratio))
        mismatch = np.minimum(float(self.density_clip), log_ratio)
        density_weight = np.exp(-self.density_alpha * mismatch)

        distance_weight = np.power(distances + self.epsilon, -self.distance_power)
        weights = distance_weight * density_weight
        return distances, indices, weights
