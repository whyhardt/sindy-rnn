"""Test pruning: CI test, patience, mask updates."""

import torch
import pytest
from sindy_rnn import PolynomialRNN, minimum_effect_ci_test
from sindy_rnn.pruning import ensemble_prune, threshold_patience_update, threshold_prune


def test_ci_test_consistent_nonzero_survives():
    """A term with consistent nonzero mean across ensemble members passes."""
    E, n_states, n_terms = 10, 2, 5
    coefficients = torch.ones(E, n_states, n_terms) * 2.0
    presence = torch.ones(E, n_states, n_terms, dtype=torch.bool)

    result = minimum_effect_ci_test(coefficients, presence, alpha=0.05, delta=0.0)
    assert result.all()


def test_ci_test_zero_mean_fails():
    """A term with zero mean fails the CI test."""
    E, n_states, n_terms = 10, 2, 5
    coefficients = torch.zeros(E, n_states, n_terms)
    presence = torch.ones(E, n_states, n_terms, dtype=torch.bool)

    result = minimum_effect_ci_test(coefficients, presence, alpha=0.05, delta=0.0)
    assert not result.any()


def test_ci_test_mixed_sign_fails():
    """A term with inconsistent sign across members should tend to fail."""
    E, n_states, n_terms = 10, 1, 1
    # Half positive, half negative -> mean near zero
    coefficients = torch.ones(E, n_states, n_terms)
    coefficients[:5] = -1.0
    presence = torch.ones(E, n_states, n_terms, dtype=torch.bool)

    result = minimum_effect_ci_test(coefficients, presence, alpha=0.05, delta=0.0)
    assert not result.any()


def test_ci_test_minimum_effect_size():
    """Terms below the minimum effect size delta should fail."""
    E, n_states, n_terms = 10, 1, 1
    # Small but consistent coefficient
    coefficients = torch.ones(E, n_states, n_terms) * 0.001
    presence = torch.ones(E, n_states, n_terms, dtype=torch.bool)

    # With delta=0, should survive
    result_no_delta = minimum_effect_ci_test(coefficients, presence, alpha=0.05, delta=0.0)
    assert result_no_delta.all()

    # With delta=0.01, should fail
    result_with_delta = minimum_effect_ci_test(coefficients, presence, alpha=0.05, delta=0.01)
    assert not result_with_delta.any()


def test_patience_increments_and_resets():
    """Patience counter increments on failure and resets on success."""
    torch.manual_seed(42)

    model = PolynomialRNN(
        n_states=2, n_controls=0, ensemble_size=5,
        polynomial_degree=2, compiled_forward=False,
    )

    # Initial patience should be all zeros
    assert (model.pruning_patience == 0).all()

    # Run pruning - some terms might fail
    with torch.no_grad():
        ensemble_prune(model, alpha=0.05, delta=0.5)

    # After one step, patience should be 0 or 1 (no prunes yet since max is 1)
    assert (model.pruning_patience <= 1).all()


def test_pruning_fires_at_patience_2():
    """Terms should be permanently pruned after 2 consecutive failures."""
    torch.manual_seed(0)

    # Use decomposed=False so we can directly access .weights/.biases
    model = PolynomialRNN(
        n_states=1, n_controls=0, ensemble_size=5,
        polynomial_degree=2, compiled_forward=False,
        decomposed=False,
    )

    # Make all polynomial weights very small so all terms fail CI test
    with torch.no_grad():
        for w in model.rnn.projection.weights:
            w.fill_(0.0)
        for b in model.rnn.projection.biases:
            b.fill_(0.0)

    # All masks should start as True
    assert model.coefficient_masks.all()

    # First pruning step - patience goes to 1
    with torch.no_grad():
        ensemble_prune(model, alpha=0.05, delta=0.01)

    # Not pruned yet (patience = 1)
    assert model.coefficient_masks.all()

    # Second pruning step - patience goes to 2, pruning fires
    with torch.no_grad():
        ensemble_prune(model, alpha=0.05, delta=0.01)

    # With Euler (no gate), zeroed weights give theta=0 everywhere.
    # ALL terms should be pruned (including self-term — no (1-alpha) boost).
    assert not model.coefficient_masks.any()


def test_pruning_fires_at_patience_2_decomposed():
    """Terms should be permanently pruned after 2 consecutive failures (decomposed)."""
    torch.manual_seed(0)

    model = PolynomialRNN(
        n_states=1, n_controls=0, ensemble_size=5,
        polynomial_degree=2, compiled_forward=False,
        decomposed=True,
    )

    # Make all polynomial weights very small so all terms fail CI test
    proj = model.rnn.projection
    with torch.no_grad():
        proj.constant_bias.fill_(0.0)
        proj.linear_weight.fill_(0.0)
        for weights in proj.higher_degree_weights.values():
            for w in weights:
                w.fill_(0.0)

    # All masks should start as True
    assert model.coefficient_masks.all()

    # First pruning step - patience goes to 1
    with torch.no_grad():
        ensemble_prune(model, alpha=0.05, delta=0.01)

    # Second pruning step - patience goes to 2, pruning fires
    with torch.no_grad():
        ensemble_prune(model, alpha=0.05, delta=0.01)

    # With Euler (no gate), zeroed weights give theta=0 everywhere.
    # ALL terms should be pruned.
    assert not model.coefficient_masks.any()


def test_threshold_pruning_single_ensemble():
    """Threshold pruning for single ensemble member."""
    torch.manual_seed(0)

    # Use decomposed=False so we can directly access .weights/.biases
    model = PolynomialRNN(
        n_states=2, n_controls=0, ensemble_size=1,
        polynomial_degree=2, compiled_forward=False,
        decomposed=False,
    )

    # Make weights tiny
    with torch.no_grad():
        for w in model.rnn.projection.weights:
            w.fill_(0.001)
        for b in model.rnn.projection.biases:
            b.fill_(0.001)

    # Two rounds of threshold patience update + prune
    with torch.no_grad():
        threshold_patience_update(model, threshold=0.5)
        threshold_patience_update(model, threshold=0.5)
        threshold_prune(model, patience_limit=2)

    # With Euler (no gate), tiny weights give tiny ODE coefficients.
    # All terms should be pruned (no (1-alpha) boost on self-term).
    assert not model.coefficient_masks.any()


def test_pruning_preserves_significant_terms():
    """Terms with large coefficients should survive pruning."""
    torch.manual_seed(42)

    model = PolynomialRNN(
        n_states=1, n_controls=0, ensemble_size=5,
        polynomial_degree=2, compiled_forward=False,
        decomposed=True,
    )

    # Set a large linear coefficient that should survive
    proj = model.rnn.projection
    with torch.no_grad():
        proj.constant_bias.fill_(0.0)
        proj.linear_weight.fill_(5.0)  # Large coefficient
        for weights in proj.higher_degree_weights.values():
            for w in weights:
                w.fill_(0.0)

    # Run two rounds of pruning
    with torch.no_grad():
        ensemble_prune(model, alpha=0.05, delta=0.5)
        ensemble_prune(model, alpha=0.05, delta=0.5)

    # Linear term (self-term) should survive
    self_idx = model.rnn._linear_indices[0].item()
    assert model.coefficient_masks[:, 0, self_idx].all()
