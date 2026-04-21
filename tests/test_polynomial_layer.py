"""Test the critical invariant: forward == forward_polynomial(mask=ones)."""

import torch
import pytest
from sindy_rnn import PolynomialRNN


@pytest.mark.parametrize("n_states,n_controls,degree,ensemble_size,decomposed", [
    # Decomposed (new default)
    (2, 0, 2, 1, True),
    (3, 0, 2, 5, True),
    (2, 1, 2, 3, True),
    (3, 2, 2, 4, True),
    (2, 0, 3, 2, True),
    (3, 1, 3, 3, True),
    (1, 0, 2, 1, True),
    (4, 0, 2, 2, True),
    # Coupled (original)
    (2, 0, 2, 1, False),
    (3, 0, 2, 5, False),
    (2, 1, 2, 3, False),
    (3, 2, 2, 4, False),
    (2, 0, 3, 2, False),
    (3, 1, 3, 3, False),
    (1, 0, 2, 1, False),
    (4, 0, 2, 2, False),
])
def test_forward_equals_forward_polynomial(n_states, n_controls, degree, ensemble_size, decomposed):
    """CRITICAL INVARIANT: standard forward == polynomial forward with mask=ones.

    This is the single most important test in the codebase.
    """
    torch.manual_seed(42)

    model = PolynomialRNN(
        n_states=n_states,
        n_controls=n_controls,
        ensemble_size=ensemble_size,
        polynomial_degree=degree,
        compiled_forward=False,
        decomposed=decomposed,
    )

    E = ensemble_size
    B = 4
    n_features = n_states + n_controls

    # Random hidden state and controls
    h = torch.randn(E, B, n_states)
    u = torch.randn(E, B, n_controls) if n_controls > 0 else None

    # Standard forward (through polynomial layer directly)
    h_standard = model.rnn._forward_impl(h, u)

    # Polynomial forward with mask=ones (via unfolding + library eval)
    mask = torch.ones(E, n_states, model.rnn._n_library_terms, dtype=torch.bool)
    h_poly = model.rnn.forward_polynomial(h, u, mask=mask)

    torch.testing.assert_close(h_standard, h_poly, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("n_states,n_controls,degree,decomposed", [
    (2, 0, 2, True),
    (3, 1, 2, True),
    (2, 0, 3, True),
    (2, 0, 2, False),
    (3, 1, 2, False),
    (2, 0, 3, False),
])
def test_forward_polynomial_no_mask_equals_mask_ones(n_states, n_controls, degree, decomposed):
    """forward_polynomial with mask=None should equal mask=all_ones."""
    torch.manual_seed(123)

    model = PolynomialRNN(
        n_states=n_states,
        n_controls=n_controls,
        ensemble_size=3,
        polynomial_degree=degree,
        compiled_forward=False,
        decomposed=decomposed,
    )

    E = 3
    B = 2
    h = torch.randn(E, B, n_states)
    u = torch.randn(E, B, n_controls) if n_controls > 0 else None

    h_no_mask = model.rnn.forward_polynomial(h, u, mask=None)
    mask = torch.ones(E, n_states, model.rnn._n_library_terms, dtype=torch.bool)
    h_with_mask = model.rnn.forward_polynomial(h, u, mask=mask)

    torch.testing.assert_close(h_no_mask, h_with_mask, rtol=1e-5, atol=1e-6)


def test_masking_removes_learned_contribution_keeps_decay():
    """Masking h_i coefficient to zero removes learned contribution but not (1-alpha) decay."""
    torch.manual_seed(99)

    model = PolynomialRNN(
        n_states=2, n_controls=0, ensemble_size=1,
        polynomial_degree=2, compiled_forward=False,
    )

    E, B = 1, 3
    h = torch.randn(E, B, 2)

    # Mask ALL terms for dimension 0
    mask = model.coefficient_masks.clone()
    mask[:, 0, :] = False

    h_next = model.rnn.forward_polynomial(h, None, mask=mask)

    # For dimension 0, the result should be (1-alpha) * h[:, :, 0]
    alpha = torch.sigmoid(model.rnn.damping_coefficient).item()
    expected_dim0 = (1 - alpha) * h[:, :, 0]

    torch.testing.assert_close(h_next[:, :, 0], expected_dim0, rtol=1e-5, atol=1e-6)


def test_sequence_forward_shape():
    """Test that full sequence forward produces correct shapes."""
    model = PolynomialRNN(
        n_states=3, n_controls=1, ensemble_size=5,
        polynomial_degree=2, compiled_forward=False,
    )

    B, T = 8, 20
    x = torch.randn(B, T, 4)  # 3 states + 1 control

    preds, final_h = model(x)

    assert preds.shape == (5, B, T, 3)
    assert final_h.shape == (5, B, 3)
