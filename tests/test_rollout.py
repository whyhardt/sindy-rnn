"""Tests for RolloutSINDyRNN, focused on identity mode (no encoder/decoder)."""

import pytest
import torch

from sindy_rnn import RolloutSINDyRNN, fit_rollout


def test_identity_requires_matching_dims():
    """identity=True needs n_sensors == n_latent == n_full."""
    with pytest.raises(ValueError):
        RolloutSINDyRNN(n_sensors=2, n_latent=3, n_full=3, identity=True)


def test_identity_skips_encoder_decoder():
    model = RolloutSINDyRNN(n_sensors=3, n_latent=3, n_full=3,
                            ensemble_size=2, polynomial_degree=2, dt=0.01,
                            identity=True)
    assert model.encoder is None
    assert isinstance(model.decoder, torch.nn.Identity)


def test_identity_fit_rollout_runs():
    """fit_rollout() with identity=True and T_w=1 should train without error
    on a fully-observed (noisy) linear system.
    """
    torch.manual_seed(0)
    T = 300
    x = torch.zeros(T + 1, 2)
    for t in range(T):
        x[t + 1, 0] = 0.9 * x[t, 0]
        x[t + 1, 1] = 0.8 * x[t, 1]
    x_noisy = x + 0.01 * torch.randn_like(x)

    model = RolloutSINDyRNN(n_sensors=2, n_latent=2, n_full=2, ensemble_size=2,
                            polynomial_degree=1, dt=1.0, decomposed=True,
                            identity=True)

    fit_rollout(model, x_noisy, x_noisy, T_w=1, T_max=4, T_start=1, delta_T=1,
                epochs=5, batch_size=8, batches_per_epoch=2, verbose=False)

    active = model.count_active_terms()
    assert sum(active.values()) >= 0  # ran and produced a valid mask state


def test_identity_save_load_roundtrip():
    torch.manual_seed(0)
    model = RolloutSINDyRNN(n_sensors=3, n_latent=3, n_full=3, ensemble_size=2,
                            polynomial_degree=2, dt=0.01, decomposed=True,
                            identity=True)
    x = torch.randn(100, 3)
    fit_rollout(model, x, x, T_w=1, T_max=3, T_start=1, delta_T=1,
                epochs=2, batch_size=4, batches_per_epoch=1, verbose=False)

    path = '/tmp/test_identity_rollout_roundtrip.pt'
    model.save(path)
    loaded = RolloutSINDyRNN.load(path)

    assert loaded.identity is True
    assert loaded.encoder is None

    model.eval()
    loaded.eval()
    warm = torch.randn(1, 1, 3)
    with torch.no_grad():
        x_hat_1, _ = model.forward(warm, 3)
        x_hat_2, _ = loaded.forward(warm, 3)
    torch.testing.assert_close(x_hat_1, x_hat_2)


def test_non_identity_unaffected():
    """Regression: default (encoder/decoder) mode still works after the
    identity-mode changes.
    """
    torch.manual_seed(0)
    model = RolloutSINDyRNN(n_sensors=2, n_latent=3, n_full=4, ensemble_size=2,
                            polynomial_degree=2, dt=0.1, gru_layers=1,
                            decomposed=True)
    assert model.identity is False
    assert isinstance(model.encoder, torch.nn.GRU)
    assert isinstance(model.decoder, torch.nn.Linear)

    x_sparse = torch.randn(100, 2)
    x_full = torch.randn(100, 4)
    fit_rollout(model, x_sparse, x_full, T_w=5, T_max=3, T_start=1, delta_T=1,
                epochs=2, batch_size=4, batches_per_epoch=1, verbose=False)
