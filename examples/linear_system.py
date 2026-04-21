"""Linear system sanity check: x[t+1] = 0.9x + 0.1u."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sindy_rnn import PolynomialRNN, fit


def main():
    torch.manual_seed(42)

    # Generate data from x[t+1] = 0.9*x[t] + 0.1*u[t]
    T = 2000
    x = torch.zeros(T + 1)
    u = torch.randn(T)
    for t in range(T):
        x[t + 1] = 0.9 * x[t] + 0.1 * u[t]

    xs = torch.stack([x[:T], u], dim=-1).unsqueeze(0)  # (1, T, 2)
    ys = x[1:T + 1].unsqueeze(0).unsqueeze(-1)          # (1, T, 1)

    model = PolynomialRNN(
        n_states=1,
        n_controls=1,
        ensemble_size=5,
        polynomial_degree=2,
        state_names=['x'],
        control_names=['u'],
        compiled_forward=False,
    )

    fit(model, xs, ys,
        epochs=300,
        warmup_steps=75,
        ensemble_pruning_alpha=0.05,
        pruning_threshold=0.01,
        learning_rate=1e-2,
        l2=1e-4,
        verbose=True,
    )

    print("\nDiscovered equations:")
    model.print_equations()
    print(f"\nActive terms: {model.count_active_terms()}")
    print(f"\nExpected: x[t+1] = 0.900*x[t] + 0.100*u")


if __name__ == '__main__':
    main()
