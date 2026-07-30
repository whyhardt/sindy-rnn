"""Tests for the examples/_common/estimators.py wrappers: fit(xs, ys),
predict(xs), simulate(x0, n_steps), save(path)/load(path).
"""
import os

import numpy as np
import pytest
import torch

from examples._common.estimators import (
    PolynomialRNNEstimator, RolloutSINDyRNNEstimator, StlsqEstimator,
)


def test_polynomial_rnn_estimator_fit_predict_simulate(tmp_path):
    torch.manual_seed(0)
    T = 300
    x = np.zeros((T + 1, 2), dtype=np.float32)
    for t in range(T):
        x[t + 1, 0] = 0.9 * x[t, 0] + 0.05
        x[t + 1, 1] = 0.8 * x[t, 1]

    xs, ys = x[:T][None], x[1:T + 1][None]

    est = PolynomialRNNEstimator(
        model_kwargs=dict(n_states=2, n_controls=0, ensemble_size=2,
                          polynomial_degree=1, dt=1.0, compiled_forward=False),
        fit_kwargs=dict(epochs=10, warmup_steps=3, learning_rate=1e-2, lambda_s=1e-4,
                        centered_diff=False, verbose=False),
    )
    est.fit(xs, ys)

    pred = est.predict(xs)
    assert pred.shape == (1, T, 2)

    sim = est.simulate(x[0:1], 10)
    assert sim.shape == (1, 10, 2)

    path = str(tmp_path / 'poly_rnn.pt')
    est.save(path)
    loaded = PolynomialRNNEstimator.load(path)
    np.testing.assert_allclose(pred, loaded.predict(xs), atol=1e-5)


def test_rollout_sindy_rnn_estimator_identity_fit_predict_simulate(tmp_path):
    torch.manual_seed(0)
    T = 200
    x = (np.random.default_rng(0).standard_normal((T + 1, 3)) * 0.1).astype(np.float32)

    est = RolloutSINDyRNNEstimator(
        n_sensors=3, n_latent=3, n_full=3, ensemble_size=2,
        polynomial_degree=2, dt=0.01, decomposed=True, identity=True,
        T_w=1, T_max=4, T_start=1, delta_T=1, epochs=5,
        batch_size=4, batches_per_epoch=1, verbose=False,
    )
    est.fit(x[:T], x[:T])

    pred = est.predict(x[:T])
    assert pred.shape == (T, 3)
    assert not np.isnan(pred).any()  # identity mode: no warmup gap

    sim = est.simulate(x[0:1], 10)
    assert sim.shape == (10, 3)

    path = str(tmp_path / 'rollout.pt')
    est.save(path)
    loaded = RolloutSINDyRNNEstimator.load(path, T_w=1)
    np.testing.assert_allclose(pred[1:], loaded.predict(x[:T])[1:], atol=1e-4)


def test_rollout_sindy_rnn_estimator_sparse_sensors(tmp_path):
    """Regression: non-identity mode (sparse sensors != full state) still works."""
    torch.manual_seed(0)
    T = 200
    rng = np.random.default_rng(1)
    x_full = (rng.standard_normal((T, 4)) * 0.1).astype(np.float32)
    x_sparse = x_full[:, :2]

    est = RolloutSINDyRNNEstimator(
        n_sensors=2, n_latent=3, n_full=4, ensemble_size=2,
        polynomial_degree=2, dt=0.1, gru_layers=1, decomposed=True,
        T_w=5, T_max=3, T_start=1, delta_T=1, epochs=3,
        batch_size=4, batches_per_epoch=1, verbose=False,
    )
    est.fit(x_sparse, x_full)
    pred = est.predict(x_sparse)
    assert pred.shape == (T, 4)


def test_stlsq_estimator_fit_predict_simulate(tmp_path):
    rng = np.random.default_rng(0)
    T = 300
    x = np.zeros((T + 1, 2))
    for t in range(T):
        x[t + 1, 0] = 0.9 * x[t, 0]
        x[t + 1, 1] = 0.8 * x[t, 1]
    x_noisy = x[:T] + 0.001 * rng.standard_normal((T, 2))

    est = StlsqEstimator(threshold=0.01, n_models=3, degree=1, dt=1.0)
    est.fit(x_noisy)
    assert est.coef_matrix.shape == (2, 3)  # [1, x, y] library, degree=1

    pred = est.predict(x_noisy)
    assert pred.shape == (T, 2)

    sim = est.simulate(x[0], 10)
    assert sim.shape[1] == 2

    path = str(tmp_path / 'stlsq.npz')
    est.save(path)
    loaded = StlsqEstimator.load(path)
    np.testing.assert_allclose(est.coef_matrix, loaded.coef_matrix)


def test_sindy_shred_estimator_available():
    """SindyShredEstimator needs the external sindy-shred/ submodule; skip
    if it isn't present rather than failing the suite for an optional dep.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(os.path.join(repo_root, 'sindy-shred')):
        pytest.skip("sindy-shred/ reference implementation not present")

    from examples._common.estimators import SindyShredEstimator
    assert SindyShredEstimator is not None
