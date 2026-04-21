"""Lotka-Volterra (predator-prey) system discovery.

dx/dt = alpha*x - beta*x*y
dy/dt = delta*x*y - gamma*y
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sindy_rnn import PolynomialRNN, fit


def lotka_volterra_rk4(state, dt=0.01, alpha=1.0, beta=0.5, delta=0.25, gamma=0.5):
    """4th-order Runge-Kutta for Lotka-Volterra."""
    def f(s):
        x, y = s
        return np.array([
            alpha * x - beta * x * y,
            delta * x * y - gamma * y
        ])
    k1 = f(state)
    k2 = f(state + dt/2 * k1)
    k3 = f(state + dt/2 * k2)
    k4 = f(state + dt * k3)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def main():
    # Generate trajectory
    state = np.array([2.0, 1.0])
    trajectory = [state.copy()]
    for _ in range(5000):
        state = lotka_volterra_rk4(state, dt=0.01)
        trajectory.append(state.copy())
    trajectory = np.array(trajectory)

    xs = torch.tensor(trajectory[:-1], dtype=torch.float32).unsqueeze(0)
    ys = torch.tensor(trajectory[1:], dtype=torch.float32).unsqueeze(0)

    model = PolynomialRNN(
        n_states=2,
        n_controls=0,
        polynomial_degree=2,
        ensemble_size=10,
        state_names=['prey', 'predator'],
        compiled_forward=False,
    )

    fit(model, xs, ys,
        epochs=500,
        warmup_steps=125,
        ensemble_pruning_alpha=0.05,
        pruning_threshold=0.005,
        learning_rate=1e-2,
        l2=1e-4,
        verbose=True,
    )

    print("\nDiscovered equations:")
    model.print_equations()
    print(f"\nActive terms: {model.count_active_terms()}")
    print("\nExpected structure:")
    print("  prey[t+1] ~ a*prey[t] - b*prey*predator")
    print("  predator[t+1] ~ c*prey*predator + d*predator[t]")


if __name__ == '__main__':
    main()
