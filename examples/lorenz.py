"""Lorenz system discovery example.

Trains a PolynomialRNN ensemble to discover the sparse polynomial equations
governing the Lorenz attractor from trajectory data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sindy_rnn import PolynomialRNN, fit


def lorenz_rk4(x, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """4th-order Runge-Kutta integrator for the Lorenz system."""
    def f(x):
        return np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        ])
    k1 = f(x)
    k2 = f(x + dt/2 * k1)
    k3 = f(x + dt/2 * k2)
    k4 = f(x + dt * k3)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def main():
    # Generate Lorenz trajectory
    x = np.array([1., 1., 1.])
    trajectory = [x]
    for _ in range(5000):
        x = lorenz_rk4(x)
        trajectory.append(x)
    trajectory = np.array(trajectory)

    xs = torch.tensor(trajectory[:-1], dtype=torch.float32).unsqueeze(0)  # (1, 5000, 3)
    ys = torch.tensor(trajectory[1:], dtype=torch.float32).unsqueeze(0)   # (1, 5000, 3)

    model = PolynomialRNN(
        n_states=3,
        n_controls=0,
        polynomial_degree=2,
        ensemble_size=10,
        state_names=['x', 'y', 'z'],
        compiled_forward=False,
    )

    fit(model, xs, ys,
        epochs=500,
        warmup_steps=100,
        ensemble_pruning_alpha=0.05,
        pruning_threshold=0.01,
        learning_rate=1e-2,
        l2=1e-4,
        verbose=True,
    )

    print("\nDiscovered equations:")
    model.print_equations()
    print(f"\nActive terms: {model.count_active_terms()}")


if __name__ == '__main__':
    main()
