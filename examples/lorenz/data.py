"""Lorenz data generation + ground truth. Values come from config.yaml so
train_sindy_rnn.py, train_stlsq.py, and analyze.py stay in sync.

Generation is deterministic (fixed seeds from config.yaml), so every script
that calls generate_or_load_data() with the same config gets byte-identical
data without needing to share a cache file.
"""
import os
from itertools import combinations_with_replacement

import numpy as np
import yaml

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXAMPLE_DIR, 'data')
PARAMS_DIR = os.path.join(EXAMPLE_DIR, 'params')
RESULTS_DIR = os.path.join(EXAMPLE_DIR, 'results')

# True continuous-time coefficients for a degree-2 polynomial library.
# Library order: [1, x, y, z, x^2, x*y, x*z, y^2, y*z, z^2]
SIGMA_DEFAULT, RHO_DEFAULT, BETA_DEFAULT = 10.0, 28.0, 8.0 / 3.0
TRUE_COEFS = np.zeros((3, 10))
TRUE_COEFS[0, 1] = -SIGMA_DEFAULT     # dx/dt: x
TRUE_COEFS[0, 2] = SIGMA_DEFAULT      # dx/dt: y
TRUE_COEFS[1, 1] = RHO_DEFAULT        # dy/dt: x
TRUE_COEFS[1, 2] = -1.0               # dy/dt: y
TRUE_COEFS[1, 6] = -1.0               # dy/dt: x*z
TRUE_COEFS[2, 3] = -BETA_DEFAULT      # dz/dt: z
TRUE_COEFS[2, 5] = 1.0                # dz/dt: x*y
TRUE_ACTIVE = TRUE_COEFS != 0


def load_config():
    with open(os.path.join(EXAMPLE_DIR, 'config.yaml')) as f:
        return yaml.safe_load(f)


def lorenz_rk4(x, dt, sigma, rho, beta):
    def f(x):
        return np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ])
    k1 = f(x)
    k2 = f(x + dt / 2 * k1)
    k3 = f(x + dt / 2 * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_lorenz(n_steps, dt, sigma, rho, beta, seed):
    rng = np.random.default_rng(seed)
    x = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)
    traj = [x.copy()]
    for _ in range(n_steps):
        x = lorenz_rk4(x, dt, sigma, rho, beta)
        traj.append(x.copy())
    return np.array(traj)


def add_noise(trajectory, noise_frac, seed):
    if noise_frac == 0:
        return trajectory.copy()
    rng = np.random.default_rng(seed)
    scale = np.std(trajectory, axis=0)
    noise = rng.normal(0, 1, trajectory.shape) * scale * noise_frac
    return trajectory + noise


def chunk_trajectory(trajectory, window_size):
    """(N+1, n_states) -> xs, ys of shape (n_windows, window_size, n_states)."""
    N = len(trajectory) - 1
    n_windows = N // window_size
    xs = trajectory[:n_windows * window_size].reshape(n_windows, window_size, -1)
    ys = trajectory[1:n_windows * window_size + 1].reshape(n_windows, window_size, -1)
    return xs, ys


def simulate_polynomial_ode(coef_matrix, h0, n_steps, dt, degree=2):
    """Euler integration of a polynomial ODE: h[t+1] = h[t] + dt * Theta(h) @ coef^T.

    Uses the same monomial ordering as both sindy-rnn and pysindy.
    """
    n_states = len(h0)
    h = h0.copy().astype(np.float64)
    trajectory = [h.copy()]

    for _ in range(n_steps):
        terms = [1.0]
        for d in range(1, degree + 1):
            for combo in combinations_with_replacement(range(n_states), d):
                val = 1.0
                for idx in combo:
                    val *= h[idx]
                terms.append(val)
        library = np.array(terms)

        dh = library @ coef_matrix.T
        h = h + dt * dh
        trajectory.append(h.copy())

        if np.any(np.abs(h) > 1e6):
            break

    return np.array(trajectory)


def compute_forecast_mse(true_traj, sim_traj):
    """Autonomous forecast MSE, truncated at divergence.

    Returns (mse, n_valid) over the non-diverged portion.
    """
    max_val = np.abs(true_traj).max() * 3
    n_common = min(len(true_traj), len(sim_traj))

    diverged_idx = np.where(np.abs(sim_traj[:n_common]).max(axis=1) > max_val)[0]
    n_valid = diverged_idx[0] if len(diverged_idx) > 0 else n_common

    if n_valid < 2:
        return float('inf'), 0

    mse = np.mean((true_traj[:n_valid] - sim_traj[:n_valid]) ** 2)
    return mse, n_valid


def generate_or_load_data(cfg):
    """Generate a fresh noisy training trajectory + a clean held-out
    trajectory for forecast evaluation.

    Returns dict with 'clean_train', 'noisy_train', 'clean_test' arrays,
    each (n_steps+1, 3).
    """
    lcfg = cfg['lorenz']

    clean_train = generate_lorenz(
        lcfg['n_steps'], lcfg['dt'], lcfg['sigma'], lcfg['rho'], lcfg['beta'],
        seed=lcfg['seed'])
    noisy_train = add_noise(clean_train, lcfg['noise_frac'], seed=lcfg['noise_seed'])
    clean_test = generate_lorenz(
        lcfg['forecast_steps'], lcfg['dt'], lcfg['sigma'], lcfg['rho'], lcfg['beta'],
        seed=lcfg['seed'] + 1)

    return {'clean_train': clean_train, 'noisy_train': noisy_train, 'clean_test': clean_test}
