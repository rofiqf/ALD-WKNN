"""Density-compatibility factor behavior: identical radii -> 1, mismatch -> < 1, range (0, 1]."""

import numpy as np

from aldwknn.model import ALDWKNNClassifier, PowerWeightedKNNClassifier


def _density_weights(model: ALDWKNNClassifier, Xq: np.ndarray) -> np.ndarray:
    """Recover the compatibility factor g_rho from the fitted model's weights."""
    distances, _, weights = model._neighbor_weights(Xq)
    distance_weight = np.power(distances + model.epsilon, -model.distance_power)
    return weights / distance_weight


def test_identical_q_radius_gives_compatibility_one():
    X = np.array([[-4.0], [-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    model = ALDWKNNClassifier(
        n_neighbors=4, density_neighbors=1, distance_power=1.0,
        density_alpha=1.0, density_clip=3.0,
    ).fit(X, y)

    Xq = np.array([[0.0]])  # q-radius of the query equals every training r_q
    compat = _density_weights(model, Xq)
    np.testing.assert_array_equal(compat, np.ones_like(compat))

    power = PowerWeightedKNNClassifier(
        n_neighbors=4, distance_power=1.0, epsilon=model.epsilon
    ).fit(X, y)
    d_p, _, w_p = power._neighbor_weights(Xq)
    d_a, _, w_a = model._neighbor_weights(Xq)
    np.testing.assert_allclose(d_a, d_p)
    np.testing.assert_allclose(w_a, w_p)


def test_large_density_mismatch_gives_compatibility_below_one():
    dense = (0.1 * np.arange(10)).reshape(-1, 1)          # spacing 0.1, rho ~ 10
    sparse = (5.0 + 1.0 * np.arange(5)).reshape(-1, 1)    # spacing 1.0, rho ~ 1
    X = np.vstack([dense, sparse])
    y = np.array([0] * 10 + [1] * 5)
    alpha = 1.0
    model = ALDWKNNClassifier(
        n_neighbors=5, density_neighbors=1, distance_power=1.0,
        density_alpha=alpha, density_clip=3.0,
    ).fit(X, y)

    Xq = np.array([[7.5]])  # sparse-region query, rho_query ~= 2 vs neighbor rho ~= 1
    compat = _density_weights(model, Xq)
    expected = np.exp(-alpha * abs(np.log(2.0)))
    assert (compat < 1.0).all()
    np.testing.assert_allclose(compat, np.full_like(compat, expected), rtol=1e-6)

    Xq_mixed = np.array([[2.5]])  # neighbors from both density regimes
    compat_mixed = _density_weights(model, Xq_mixed)
    assert (compat_mixed < 1.0).all()


def test_compatibility_bounded_in_open_closed_interval():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 2))
    y = (X[:, 0] > 0).astype(int)
    model = ALDWKNNClassifier(
        n_neighbors=5, density_neighbors=3, distance_power=2.0,
        density_alpha=2.0, density_clip=3.0,
    ).fit(X, y)

    Xq = rng.normal(size=(30, 2))
    compat = _density_weights(model, Xq)
    assert np.isfinite(compat).all()
    assert (compat > 0.0).all()
    assert (compat <= 1.0).all()
    floor = np.exp(-2.0 * 3.0)  # clipping bounds the penalty away from 0
    assert compat.min() >= floor
