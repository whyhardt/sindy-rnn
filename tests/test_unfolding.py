"""Test polynomial coefficient unfolding correctness."""

import torch
import pytest
from sindy_rnn import PolynomialRNN
from sindy_rnn.polynomial_library import build_library_structure, get_library_feature_names


def test_library_structure_2features_degree2():
    """Verify mult_table for (n=2, d=2) against manual enumeration."""
    lib = build_library_structure(2, 2)

    # Expected terms: (), (0,), (1,), (0,0), (0,1), (1,1)
    assert lib['terms'] == [(), (0,), (1,), (0, 0), (0, 1), (1, 1)]
    assert lib['n_terms'] == 6
    assert lib['bias_index'] == 0
    assert lib['linear_indices'].tolist() == [1, 2]

    # Check mult_table:
    # () * x0 = (0,) -> idx 1;  () * x1 = (1,) -> idx 2
    # (0,) * x0 = (0,0) -> idx 3;  (0,) * x1 = (0,1) -> idx 4
    # (1,) * x0 = (0,1) -> idx 4;  (1,) * x1 = (1,1) -> idx 5
    # (0,0) * any -> degree 3 -> -1
    # (0,1) * any -> degree 3 -> -1
    # (1,1) * any -> degree 3 -> -1
    mt = lib['mult_table']
    assert mt[0, 0].item() == 1  # () * x0 = (0,)
    assert mt[0, 1].item() == 2  # () * x1 = (1,)
    assert mt[1, 0].item() == 3  # (0,) * x0 = (0,0)
    assert mt[1, 1].item() == 4  # (0,) * x1 = (0,1)
    assert mt[2, 0].item() == 4  # (1,) * x0 = (0,1)
    assert mt[2, 1].item() == 5  # (1,) * x1 = (1,1)
    assert mt[3, 0].item() == -1
    assert mt[3, 1].item() == -1
    assert mt[4, 0].item() == -1
    assert mt[4, 1].item() == -1
    assert mt[5, 0].item() == -1
    assert mt[5, 1].item() == -1


def test_library_feature_names():
    """Verify name generation for small cases."""
    names = get_library_feature_names(['h', 'u'], 2)
    assert names == ['1', 'h', 'u', 'h^2', 'h*u', 'u^2']

    names3 = get_library_feature_names(['x', 'y', 'z'], 2)
    assert names3 == ['1', 'x', 'y', 'z', 'x^2', 'x*y', 'x*z', 'y^2', 'y*z', 'z^2']


def test_unfolding_degree1():
    """For degree-1, unfolded coefficients should match W and b directly."""
    torch.manual_seed(0)

    model = PolynomialRNN(
        n_states=2, n_controls=1, ensemble_size=1,
        polynomial_degree=1, compiled_forward=False,
    )

    # Get the single weight matrix and bias
    W = model.rnn.projection.weights[0].data  # (1, 2, 3)
    b = model.rnn.projection.biases[0].data   # (1, 2)

    theta = model.rnn.unfold_polynomial_coefficients()  # (1, 2, 4) = (E, n_states, n_terms)

    # For degree 1: terms are [(), (0,), (1,), (2,)]
    # theta[e, i, 0] = b[e, i]  (bias term)
    # theta[e, i, 1:] = W[e, i, :]  (linear terms)
    assert theta.shape == (1, 2, 4)
    torch.testing.assert_close(theta[0, :, 0], b[0, :])
    torch.testing.assert_close(theta[0, :, 1:], W[0, :, :])


def test_unfolding_known_degree2():
    """Construct a degree-2 polynomial with known weights and verify unfolding.

    For 1 feature (n_states=1, n_controls=0), degree 2:
    Factor 0: w0*x + b0
    Factor 1: w1*x + b1
    Product: (w0*x + b0)(w1*x + b1) / sqrt(2) = (w0*w1*x^2 + (w0*b1 + w1*b0)*x + b0*b1) / sqrt(2)

    Terms: [(), (0,), (0,0)] = ['1', 'h_0', 'h_0^2']
    Expected coefficients:
        constant: b0*b1 / sqrt(2)
        linear: (w0*b1 + w1*b0) / sqrt(2)
        quadratic: w0*w1 / sqrt(2)
    """
    import math
    torch.manual_seed(42)

    model = PolynomialRNN(
        n_states=1, n_controls=0, ensemble_size=1,
        polynomial_degree=2, compiled_forward=False,
    )

    # Set known weights
    w0, b0 = 2.0, 3.0
    w1, b1 = -1.0, 0.5

    with torch.no_grad():
        model.rnn.projection.weights[0].fill_(0)
        model.rnn.projection.weights[0][0, 0, 0] = w0
        model.rnn.projection.biases[0][0, 0] = b0
        model.rnn.projection.weights[1].fill_(0)
        model.rnn.projection.weights[1][0, 0, 0] = w1
        model.rnn.projection.biases[1][0, 0] = b1

    theta = model.rnn.unfold_polynomial_coefficients()  # (1, 1, 3)

    sqrt2 = math.sqrt(2)
    expected_const = (b0 * b1) / sqrt2
    expected_linear = (w0 * b1 + w1 * b0) / sqrt2
    expected_quad = (w0 * w1) / sqrt2

    assert theta.shape == (1, 1, 3)
    torch.testing.assert_close(theta[0, 0, 0], torch.tensor(expected_const), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(theta[0, 0, 1], torch.tensor(expected_linear), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(theta[0, 0, 2], torch.tensor(expected_quad), rtol=1e-5, atol=1e-6)


def test_unfolding_known_degree2_multivariate():
    """Test unfolding for 2 features, degree 2 with known weights.

    For 2 features, degree 2:
    Factor 0: w00*x0 + w01*x1 + b0
    Factor 1: w10*x0 + w11*x1 + b1
    Product / sqrt(2):
      constant: b0*b1
      x0: w00*b1 + w10*b0
      x1: w01*b1 + w11*b0
      x0^2: w00*w10
      x0*x1: w00*w11 + w01*w10
      x1^2: w01*w11
    All divided by sqrt(2).
    """
    import math
    torch.manual_seed(0)

    model = PolynomialRNN(
        n_states=2, n_controls=0, ensemble_size=1,
        polynomial_degree=2, compiled_forward=False,
    )

    # Set known weights for output dimension 0 only
    w00, w01 = 1.0, 2.0   # Factor 0 weights for features [h0, h1]
    b0 = 0.5
    w10, w11 = -1.0, 3.0  # Factor 1 weights
    b1 = -0.5

    with torch.no_grad():
        model.rnn.projection.weights[0][0, 0, :] = torch.tensor([w00, w01])
        model.rnn.projection.biases[0][0, 0] = b0
        model.rnn.projection.weights[1][0, 0, :] = torch.tensor([w10, w11])
        model.rnn.projection.biases[1][0, 0] = b1

    theta = model.rnn.unfold_polynomial_coefficients()  # (1, 2, 6)
    # Only check dimension 0
    t = theta[0, 0, :]  # (6,)

    sqrt2 = math.sqrt(2)
    # Terms: [(), (0,), (1,), (0,0), (0,1), (1,1)]
    expected = torch.tensor([
        b0 * b1,
        w00 * b1 + w10 * b0,
        w01 * b1 + w11 * b0,
        w00 * w10,
        w00 * w11 + w01 * w10,
        w01 * w11,
    ]) / sqrt2

    torch.testing.assert_close(t, expected, rtol=1e-5, atol=1e-6)


def test_unfolding_differentiable():
    """Verify that unfold_polynomial_coefficients is differentiable."""
    model = PolynomialRNN(
        n_states=2, n_controls=0, ensemble_size=1,
        polynomial_degree=2, compiled_forward=False,
    )

    theta = model.rnn.unfold_polynomial_coefficients()
    loss = theta.sum()
    loss.backward()

    for w in model.rnn.projection.weights:
        assert w.grad is not None
        assert not torch.all(w.grad == 0)
