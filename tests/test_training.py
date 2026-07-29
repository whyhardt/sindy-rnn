"""End-to-end tests: linear system recovery."""

import torch
from sindy_rnn import PolynomialRNN, fit


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

