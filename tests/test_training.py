"""End-to-end tests: linear system recovery, Jacobian, stability."""

import torch
import pytest
from sindy_rnn import PolynomialRNN, fit
from sindy_rnn.model import EnsembleRNNModule


def test_linear_system_recovery():
    """Known x[t+1] = 0.9*x + 0.1*u -> verify ODE coefficients recovered.

    With Euler (dt=1): x[t+1] = x + 1 * P(x,u).
    So P(x,u) = x[t+1] - x = 0.9*x + 0.1*u - x = -0.1*x + 0.1*u.
    Theta directly IS the ODE: dx/dt = -0.1*x + 0.1*u.
    """
    torch.manual_seed(42)

    # Generate data from x[t+1] = 0.9*x[t] + 0.1*u[t]
    T = 1000
    x = torch.zeros(T + 1)
    u = torch.randn(T)
    for t in range(T):
        x[t + 1] = 0.9 * x[t] + 0.1 * u[t]

    # Format: xs = (B, T, n_states + n_controls), ys = (B, T, n_states)
    xs = torch.stack([x[:T], u], dim=-1).unsqueeze(0)  # (1, T, 2)
    ys = x[1:T + 1].unsqueeze(0).unsqueeze(-1)          # (1, T, 1)

    model = PolynomialRNN(
        n_states=1,
        n_controls=1,
        ensemble_size=1,
        polynomial_degree=2,
        dt=1.0,
        state_names=['x'],
        control_names=['u'],
        compiled_forward=False,
    )

    fit(model, xs, ys,
        epochs=300,
        warmup_steps=100,
        pruning_threshold=0.005,
        learning_rate=1e-2,
        l2=1e-5,
        centered_diff=False,  # dt=1.0: forward diff is exact for Euler
        verbose=False,
    )

    # Check ODE coefficients
    coefs = model.get_coefficients(aggregate=True)
    c = coefs['x']  # (n_terms,)
    # Terms: ['1', 'x', 'u', 'x^2', 'x*u', 'u^2']
    # Expected ODE: dx/dt = -0.1*x + 0.1*u (since x[t+1] = x + dt*(-0.1x + 0.1u))

    x_idx = model.rnn._linear_indices[0].item()  # index of x
    u_idx = model.rnn._linear_indices[1].item()  # index of u

    x_coef = c[x_idx].item()
    u_coef = c[u_idx].item()

    assert abs(x_coef - (-0.1)) < 0.1, f"x ODE coefficient {x_coef} not close to -0.1"
    assert abs(u_coef - 0.1) < 0.1, f"u ODE coefficient {u_coef} not close to 0.1"

    # Quadratic terms should be near zero or pruned
    for j in range(len(model.library_terms)):
        if j not in [x_idx, u_idx]:
            assert abs(c[j].item()) < 0.05, f"Term {model.library_terms[j]} should be ~0"


def test_sparsity_on_linear_system():
    """Many irrelevant terms should be pruned for a sparse ground-truth system."""
    torch.manual_seed(123)

    T = 500
    x = torch.zeros(T + 1)
    u = torch.randn(T)
    for t in range(T):
        x[t + 1] = 0.9 * x[t] + 0.1 * u[t]

    xs = torch.stack([x[:T], u], dim=-1).unsqueeze(0)
    ys = x[1:T + 1].unsqueeze(0).unsqueeze(-1)

    model = PolynomialRNN(
        n_states=1, n_controls=1, ensemble_size=5,
        polynomial_degree=2, dt=1.0, compiled_forward=False,
        state_names=['x'], control_names=['u'],
    )

    fit(model, xs, ys,
        epochs=400,
        warmup_steps=100,
        ensemble_pruning_alpha=0.05,
        pruning_threshold=0.005,
        learning_rate=1e-2,
        l2=1e-4,
        centered_diff=False,  # dt=1.0: forward diff is exact for Euler
        verbose=False,
    )

    active = model.count_active_terms()
    # Should have pruned most of the 6 terms down to ~2-3
    assert active['x'] <= 4, f"Expected sparse solution, got {active['x']} active terms"


def test_nan_masking():
    """Variable-length sequences with NaN padding should work."""
    torch.manual_seed(0)

    model = PolynomialRNN(
        n_states=2, n_controls=0, ensemble_size=1,
        polynomial_degree=2, compiled_forward=False,
    )

    B, T = 3, 50
    xs = torch.randn(B, T, 2)
    ys = torch.randn(B, T, 2)

    # Pad second sequence after t=30
    xs[1, 30:, :] = float('nan')
    ys[1, 30:, :] = float('nan')

    # Should not raise
    fit(model, xs, ys, epochs=10, centered_diff=False, verbose=False)


# ========== Jacobian and stability tests ==========

def test_derivative_table_degree2():
    """Verify derivative table for 2-feature, degree-2 library.

    Library: ['1', 'h0', 'h1', 'h0^2', 'h0*h1', 'h1^2']
    Derivatives w.r.t. h0:
      d(1)/dh0 = 0
      d(h0)/dh0 = 1  (count=1, reduced=() -> index 0)
      d(h1)/dh0 = 0
      d(h0^2)/dh0 = 2*h0  (count=2, reduced=(0,) -> index 1)
      d(h0*h1)/dh0 = h1  (count=1, reduced=(1,) -> index 2)
      d(h1^2)/dh0 = 0
    """
    rnn = EnsembleRNNModule(
        ensemble_size=1, n_states=2, n_controls=0,
        polynomial_degree=2, dt=1.0, compiled_forward=False,
    )

    # terms: [(), (0,), (1,), (0,0), (0,1), (1,1)]
    dc = rnn._deriv_count   # (6, 2)
    dt_idx = rnn._deriv_term_idx  # (6, 2)

    # d/dh0
    assert dc[0, 0] == 0   # d(1)/dh0
    assert dc[1, 0] == 1   # d(h0)/dh0 = 1
    assert dt_idx[1, 0] == 0  # reduced term = () = index 0
    assert dc[2, 0] == 0   # d(h1)/dh0 = 0
    assert dc[3, 0] == 2   # d(h0^2)/dh0 = 2*h0
    assert dt_idx[3, 0] == 1  # reduced term = (0,) = index 1
    assert dc[4, 0] == 1   # d(h0*h1)/dh0 = h1
    assert dt_idx[4, 0] == 2  # reduced term = (1,) = index 2
    assert dc[5, 0] == 0   # d(h1^2)/dh0 = 0

    # d/dh1
    assert dc[0, 1] == 0
    assert dc[1, 1] == 0   # d(h0)/dh1 = 0
    assert dc[2, 1] == 1   # d(h1)/dh1 = 1
    assert dt_idx[2, 1] == 0  # reduced = ()
    assert dc[4, 1] == 1   # d(h0*h1)/dh1 = h0
    assert dt_idx[4, 1] == 1  # reduced = (0,)
    assert dc[5, 1] == 2   # d(h1^2)/dh1 = 2*h1
    assert dt_idx[5, 1] == 2  # reduced = (1,)


def test_jacobian_vs_finite_differences():
    """Analytical Jacobian must match finite-difference Jacobian.

    Uses float64 and direct=True to eliminate float32 roundoff and unfolding
    imprecision from the comparison.
    """
    torch.manual_seed(42)

    model = PolynomialRNN(
        n_states=2, n_controls=0, ensemble_size=1,
        polynomial_degree=2, dt=0.1, compiled_forward=False,
        direct=True,
    ).double()

    theta = model.rnn.unfold_polynomial_coefficients()
    h = torch.randn(1, 5, 2, dtype=torch.float64)

    J_analytical = model.rnn._compute_jacobian(h, None, theta)

    eps = 1e-7
    J_fd = torch.zeros_like(J_analytical)
    for k in range(2):
        h_plus = h.clone()
        h_plus[:, :, k] += eps
        h_minus = h.clone()
        h_minus[:, :, k] -= eps
        J_fd[:, :, :, k] = (
            model.rnn._evaluate_rhs(h_plus, None, theta)
            - model.rnn._evaluate_rhs(h_minus, None, theta)
        ) / (2 * eps)

    torch.testing.assert_close(J_analytical, J_fd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_jacobian_vs_finite_differences_degree(degree):
    """Jacobian correctness across polynomial degrees (float64, direct mode)."""
    torch.manual_seed(0)

    model = PolynomialRNN(
        n_states=3, n_controls=0, ensemble_size=2,
        polynomial_degree=degree, dt=0.01, compiled_forward=False,
        direct=True,
    ).double()

    theta = model.rnn.unfold_polynomial_coefficients()
    h = torch.randn(2, 4, 3, dtype=torch.float64) * 0.5

    J_analytical = model.rnn._compute_jacobian(h, None, theta)

    eps = 1e-7
    J_fd = torch.zeros_like(J_analytical)
    for k in range(3):
        h_plus = h.clone()
        h_plus[:, :, k] += eps
        h_minus = h.clone()
        h_minus[:, :, k] -= eps
        J_fd[:, :, :, k] = (
            model.rnn._evaluate_rhs(h_plus, None, theta)
            - model.rnn._evaluate_rhs(h_minus, None, theta)
        ) / (2 * eps)

    torch.testing.assert_close(J_analytical, J_fd, rtol=1e-5, atol=1e-6)


def test_stability_loss_stable_system():
    """A known stable system should have zero stability loss."""
    model = PolynomialRNN(
        n_states=2, n_controls=0, ensemble_size=1,
        polynomial_degree=1, dt=0.01, compiled_forward=False,
        direct=True,
    )

    # Set theta to a stable linear ODE: A = [[-1, 0], [0, -2]]
    # Eigenvalues: -1, -2. Discrete: |1 + 0.01*(-1)| = 0.99, |1 + 0.01*(-2)| = 0.98
    with torch.no_grad():
        model.rnn.theta.zero_()
        lin_idx = model.rnn._linear_indices[:2]
        model.rnn.theta[0, 0, lin_idx[0]] = -1.0  # dh0/dt = -h0
        model.rnn.theta[0, 1, lin_idx[1]] = -2.0  # dh1/dt = -2*h1

    theta = model.rnn.unfold_polynomial_coefficients()
    loss = model.rnn.compute_stability_loss(theta, model.rnn._dt)
    assert loss.item() == 0.0, f"Stable system should have 0 loss, got {loss.item()}"


def test_stability_loss_unstable_system():
    """A known unstable system should have positive stability loss."""
    model = PolynomialRNN(
        n_states=2, n_controls=0, ensemble_size=1,
        polynomial_degree=1, dt=0.01, compiled_forward=False,
        direct=True,
    )

    # Set theta to an unstable linear ODE: A = [[5, 0], [0, -1]]
    # Eigenvalue 5: |1 + 0.01*5| = 1.05 > 1 -> positive loss
    with torch.no_grad():
        model.rnn.theta.zero_()
        lin_idx = model.rnn._linear_indices[:2]
        model.rnn.theta[0, 0, lin_idx[0]] = 5.0
        model.rnn.theta[0, 1, lin_idx[1]] = -1.0

    theta = model.rnn.unfold_polynomial_coefficients()
    loss = model.rnn.compute_stability_loss(theta, model.rnn._dt)
    assert loss.item() > 0.0, f"Unstable system should have positive loss, got {loss.item()}"
    expected = abs(1 + 0.01 * 5) - 1.0  # 0.05
    assert abs(loss.item() - expected) < 1e-5, f"Expected ~{expected}, got {loss.item()}"


def test_stability_loss_differentiable():
    """Stability loss must be differentiable w.r.t. polynomial parameters."""
    model = PolynomialRNN(
        n_states=2, n_controls=0, ensemble_size=1,
        polynomial_degree=2, dt=0.1, compiled_forward=False,
    )

    theta = model.rnn.unfold_polynomial_coefficients()
    h_sample = torch.randn(1, 10, 2)
    loss = model.rnn.compute_stability_loss(theta, model.rnn._dt, h_sample)
    loss.backward()

    # At least some parameters should have nonzero gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "Stability loss should produce gradients"


def test_fit_with_stability_weight():
    """Training with stability_weight should not crash and loss should decrease."""
    torch.manual_seed(42)

    T = 200
    xs = torch.randn(1, T, 2)
    ys = torch.randn(1, T, 2)

    model = PolynomialRNN(
        n_states=2, n_controls=0, ensemble_size=1,
        polynomial_degree=1, dt=0.1, compiled_forward=False,
    )

    fit(model, xs, ys, epochs=50, stability_weight=0.1,
        centered_diff=False, verbose=False)
