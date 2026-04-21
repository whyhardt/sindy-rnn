"""Duffing oscillator discovery.

The Duffing equation:
    dx/dt = v
    dv/dt = -delta*v - alpha*x - beta*x^3 + gamma*cos(omega*t)

Discretized as a 2D state system [x, v] with forcing u = cos(omega*t).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sindy_rnn import PolynomialRNN, fit


def duffing_rk4(state, t, dt=0.01, delta=0.3, alpha=-1.0, beta=1.0,
                gamma=0.5, omega=1.2):
    """4th-order Runge-Kutta for forced Duffing oscillator."""
    def f(s, t_):
        x, v = s
        return np.array([
            v,
            -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t_)
        ])
    k1 = f(state, t)
    k2 = f(state + dt/2 * k1, t + dt/2)
    k3 = f(state + dt/2 * k2, t + dt/2)
    k4 = f(state + dt * k3, t + dt)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def main():
    dt = 0.01
    n_steps = 5000
    omega = 1.2

    # Generate trajectory
    state = np.array([1.0, 0.0])
    trajectory = [state.copy()]
    times = [0.0]
    for i in range(n_steps):
        t = i * dt
        state = duffing_rk4(state, t, dt=dt)
        trajectory.append(state.copy())
        times.append((i + 1) * dt)
    trajectory = np.array(trajectory)
    times = np.array(times)

    # Control input: cos(omega*t) for each timestep
    u = np.cos(omega * times[:-1]).reshape(-1, 1)  # (T, 1)

    # xs: (B, T, n_states + n_controls) = (1, T, 3)
    features = np.concatenate([trajectory[:-1], u], axis=-1)
    xs = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    ys = torch.tensor(trajectory[1:], dtype=torch.float32).unsqueeze(0)

    model = PolynomialRNN(
        n_states=2,
        n_controls=1,
        polynomial_degree=3,  # Need degree 3 for x^3 term
        ensemble_size=10,
        state_names=['x', 'v'],
        control_names=['F'],
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
    print("  x[t+1] ~ x[t] + dt*v")
    print("  v[t+1] ~ -delta*v[t] - alpha*x - beta*x^3 + gamma*F")


if __name__ == '__main__':
    main()
