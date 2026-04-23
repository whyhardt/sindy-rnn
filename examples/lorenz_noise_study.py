"""Lorenz system discovery with ensemble pruning across noise levels.

This script trains ensemble PolynomialRNNs on Lorenz attractor data with
increasing observation noise to study the robustness of sparse dynamics
discovery.

The Lorenz system (continuous):
    dx/dt = sigma * (y - x)        = -10*x + 10*y
    dy/dt = x * (rho - z) - y      = 28*x - y - x*z
    dz/dt = x * y - beta * z       = x*y - 2.667*z

With ensemble_size > 1, the training uses bootstrap resampling and
ensemble CI pruning to identify statistically significant terms.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sindy_rnn import PolynomialRNN, fit


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Hyperparameters ----
CONFIG = {
    'dt': 0.01,
    'n_steps': 10000,
    'ensemble_size': 11,
    'degree': 2,
    'epochs': 1000,
    'warmup_steps': 200,
    'window_size': 100,
    'learning_rate': 1e-2,
    'l1': 1e-2,
    'pruning_frequency': 100,
    'pruning_threshold': 0.1,
    'ensemble_pruning_alpha': 0.05,
    'feature_dropout': 0.,
    'dropout': 0.1,
    'direct': False,
    'refit_epochs': 100,
    'noise_fractions': [0.],#, 0.01, 0.02, 0.05, 0.1],
}


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


def generate_lorenz_data(n_steps=5000, dt=0.01, seed=42):
    """Generate clean Lorenz trajectory."""
    rng = np.random.default_rng(seed)
    x = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        x = lorenz_rk4(x, dt=dt)
        trajectory.append(x.copy())
    return np.array(trajectory)


def add_noise(trajectory, noise_std, seed=0):
    """Add Gaussian observation noise to trajectory."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, trajectory.shape)
    return trajectory + noise


def chunk_trajectory(trajectory, window_size=100):
    """Chunk a long trajectory into batched non-overlapping windows.

    Args:
        trajectory: (N+1, n_states) — full trajectory
        window_size: number of timesteps per window

    Returns:
        xs: (B, T, n_states) — input windows
        ys: (B, T, n_states) — target windows (one step ahead)
    """
    N = len(trajectory) - 1
    n_windows = N // window_size
    xs_all = trajectory[:n_windows * window_size]
    ys_all = trajectory[1:n_windows * window_size + 1]

    xs = xs_all.reshape(n_windows, window_size, -1)
    ys = ys_all.reshape(n_windows, window_size, -1)
    return xs, ys


def prepare_data(trajectory, window_size=100):
    """Convert trajectory to batched (xs, ys) tensors on device."""
    xs, ys = chunk_trajectory(trajectory, window_size=window_size)
    xs = torch.tensor(xs, dtype=torch.float32).to(DEVICE)
    ys = torch.tensor(ys, dtype=torch.float32).to(DEVICE)
    return xs, ys


def run_experiment(noise_std, trajectory_clean, config, seed=42):
    """Train PolynomialRNN with ensemble pruning on noisy Lorenz data."""
    torch.manual_seed(seed)

    if noise_std > 0:
        trajectory_noisy = add_noise(trajectory_clean, noise_std, seed=seed)
    else:
        trajectory_noisy = trajectory_clean.copy()

    xs, ys = prepare_data(trajectory_noisy, window_size=config['window_size'])

    # Test data from a different initial condition (clean, to measure true performance)
    test_traj = generate_lorenz_data(n_steps=1000, dt=config['dt'], seed=seed + 100)
    xs_test, ys_test = prepare_data(test_traj, window_size=config['window_size'])

    model = PolynomialRNN(
        n_states=3,
        n_controls=0,
        polynomial_degree=config['degree'],
        ensemble_size=config['ensemble_size'],
        dt=config['dt'],
        state_names=['x', 'y', 'z'],
        dropout=config['dropout'],
        feature_dropout=config['feature_dropout'],
        compiled_forward=False,
        direct=config.get('direct', False),
        decomposed=True,
    ).to(DEVICE)

    fit(model, xs, ys,
        xs_test=xs_test,
        ys_test=ys_test,
        epochs=config['epochs'],
        warmup_steps=config['warmup_steps'],
        ensemble_pruning_alpha=config['ensemble_pruning_alpha'],
        pruning_threshold=config['pruning_threshold'],
        pruning_frequency=config['pruning_frequency'],
        pruning_method='median',
        learning_rate=config['learning_rate'],
        l2=config['l1'],
        dt=config['dt'],
        refit_epochs=config.get('refit_epochs', 0),
        verbose=True,
    )

    # Compute final test MSE
    with torch.no_grad():
        model.eval()
        E = model.ensemble_size
        x_te = xs_test.unsqueeze(0).expand(E, -1, -1, -1)
        yp_te, _ = model(x_te)
        y_te_exp = ys_test.unsqueeze(0).expand(E, -1, -1, -1)
        test_mse = torch.nn.functional.mse_loss(yp_te, y_te_exp).item()

    return model, test_mse


def print_results(model, noise_std, test_mse, dt=0.01):
    """Print discovered equations and active term counts."""
    active = model.count_active_terms()
    total_active = sum(active.values())

    print(f"\n{'='*60}")
    print(f"Noise std = {noise_std:.4f} | Test MSE = {test_mse:.6f} | Active terms = {total_active}")
    print(f"{'='*60}")
    print("\nDiscovered ODE:")
    model.print_equations()
    print(f"\nActive terms per state: {active}")


def main():
    print("Lorenz System Discovery - Ensemble + Noise Robustness Study")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    print(f"\nSettings: {CONFIG}")
    print("\nGenerating clean Lorenz trajectory...")
    trajectory_clean = generate_lorenz_data(n_steps=CONFIG['n_steps'], dt=CONFIG['dt'], seed=42)

    scale = np.std(trajectory_clean, axis=0).mean()
    print(f"Average state std: {scale:.2f}")

    print("\nExpected continuous-time Lorenz ODE:")
    print("  dx/dt = -10.000*x + 10.000*y")
    print("  dy/dt = 28.000*x - 1.000*y - 1.000*x*z")
    print("  dz/dt = -2.667*z + 1.000*x*y")

    results = []
    
    # Step 1: First recover clean system to validate approach
    if 0 in CONFIG['noise_fractions']:
        print(f"\n\n{'#'*60}")
        print("# Step 1: Clean data recovery (noise = 0)")
        print(f"{'#'*60}")

        model_clean, mse_clean = run_experiment(
            noise_std=0.0,
            trajectory_clean=trajectory_clean,
            config=CONFIG,
            seed=42,
        )
        print_results(model_clean, 0.0, mse_clean, dt=CONFIG['dt'])

        active_clean = model_clean.count_active_terms()
        total_clean = sum(active_clean.values())
        if total_clean <= 8:
            print(f"\nClean recovery successful ({total_clean} active terms). Proceeding to noise sweep...")
        else:
            print(f"\nWARNING: Clean recovery has {total_clean} active terms (expected <=8). Check hyperparameters.")

        results.append({'noise_frac': 0.0, 'noise_std': 0.0, 'test_mse': mse_clean,
                'total_active': total_clean, 'active_per_state': active_clean})
        
    # Step 2: Sweep noise levels
    if 0 in CONFIG['noise_fractions']:
        noise_fractions = CONFIG['noise_fractions'][1:]
    else:
        noise_fractions = CONFIG['noise_fractions']
    noise_levels = [f * scale for f in noise_fractions]
    
    print(f"\n\nNoise levels (fraction of state std): {noise_fractions}")
    print(f"Noise levels (absolute): [{', '.join(f'{n:.3f}' for n in noise_levels)}]")

    for i, (frac, noise_std) in enumerate(zip(noise_fractions, noise_levels)):
        print(f"\n\n{'#'*60}")
        print(f"# Experiment {i+2 if 0 in CONFIG['noise_fractions'] else i+1}/{len(noise_levels)}: noise = {frac:.0%} of state std ({noise_std:.3f})")
        print(f"{'#'*60}")

        model, test_mse = run_experiment(
            noise_std=noise_std,
            trajectory_clean=trajectory_clean,
            config=CONFIG,
            seed=42,
        )

        print_results(model, noise_std, test_mse, dt=CONFIG['dt'])
        active = model.count_active_terms()
        results.append({
            'noise_frac': frac,
            'noise_std': noise_std,
            'test_mse': test_mse,
            'total_active': sum(active.values()),
            'active_per_state': active,
        })

    # Summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Noise %':<10} {'Noise std':<12} {'Test MSE':<12} {'Active terms':<12}")
    print("-" * 46)
    for r in results:
        print(f"{r['noise_frac']:<10.0%} {r['noise_std']:<12.4f} {r['test_mse']:<12.6f} {r['total_active']:<12}")

    # Expected: 8 active terms for clean Lorenz (3 self-terms + 2 linear cross + 2 bilinear + 1 more linear)
    # Specifically: x: x[t], y; y: x, y[t], x*z; z: z[t], x*y
    print("\nExpected: 8 active terms for clean recovery")
    print("  x: x[t], y")
    print("  y: x, y[t], x*z")
    print("  z: z[t], x*y")


if __name__ == '__main__':
    main()
