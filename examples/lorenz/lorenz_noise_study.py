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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sindy_rnn import PolynomialRNN, fit


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

# ---- Hyperparameters ----
CONFIG = {
    'dt': 0.01,
    'n_steps': 1000,
    'ensemble_size': 11,
    'degree': 2,
    'epochs': 3000,
    'warmup_steps': 1000,
    'window_size': 100,
    'learning_rate': 5e-2,
    'lambda': 5e-2,
    'pruning_frequency': 100,
    'pruning_threshold': 0.2,
    'ensemble_pruning_alpha': 0.05,
    'feature_dropout': 0.,
    'dropout': 0.1,
    'direct': False,
    'refit_epochs': 500,
    'num_euler_steps': 1,
    'dynamics_weight': 0,
    'noise_fractions': [0., 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
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


def generate_lorenz_data(n_steps=5000, dt=0.01, seed=SEED):
    """Generate clean Lorenz trajectory."""
    rng = np.random.default_rng(seed)
    x = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        x = lorenz_rk4(x, dt=dt)
        trajectory.append(x.copy())
    return np.array(trajectory)


def add_noise(trajectory, noise_std, seed=SEED):
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


def run_experiment(noise_std, trajectory_clean, config, seed=SEED):
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
        num_euler_steps=config.get('num_euler_steps', 1),
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
        l2=config['lambda'],
        dt=config['dt'],
        refit_epochs=config.get('refit_epochs', 0),
        dynamics_weight=config.get('dynamics_weight', 0),
        verbose=True,
    )

    return model


def simulate_ode(model, h0, n_steps, dt):
    """Autonomous forward simulation using discovered ODE coefficients.

    Rolls out h[t+1] = h[t] + dt * P(h[t]) using the aggregated (ensemble-mean)
    polynomial coefficients and the current sparsity mask.

    Args:
        model: trained PolynomialRNN
        h0: (n_states,) numpy array — initial condition
        n_steps: number of integration steps
        dt: timestep

    Returns:
        trajectory: (n_steps+1, n_states) numpy array
    """
    model.eval()
    device = next(model.parameters()).device

    # Get aggregated coefficients: (n_states, n_terms)
    coefs = model.get_coefficients(aggregate=True)
    state_names = model.state_names or [f's{i}' for i in range(model.n_states)]
    theta = torch.stack([coefs[s] for s in state_names], dim=0).to(device)  # (n_states, n_terms)

    h = torch.tensor(h0, dtype=torch.float32, device=device).unsqueeze(0)  # (1, n_states)
    trajectory = [h0.copy()]

    with torch.no_grad():
        for _ in range(n_steps):
            # Compute library: (1, n_terms)
            library = model.rnn._compute_library(h.unsqueeze(0))  # (1, 1, n_terms)
            library = library.squeeze(0)  # (1, n_terms)
            # P(h) = library @ theta^T: (1, n_states)
            P_h = library @ theta.T
            h = h + dt * P_h
            trajectory.append(h.squeeze(0).cpu().numpy())

    return np.array(trajectory)


def compute_forecast_mse(true_traj, sim_traj):
    """Autonomous forecast MSE, truncated at divergence.

    Returns:
        mse: mean squared error over valid (non-diverged) steps
        n_valid: number of valid steps before divergence (or full length)
    """
    max_val = np.abs(true_traj).max() * 3
    diverged_idx = np.where(np.abs(sim_traj).max(axis=1) > max_val)[0]
    if len(diverged_idx) > 0:
        n_valid = diverged_idx[0]
    else:
        n_valid = min(len(true_traj), len(sim_traj))

    if n_valid < 2:
        return float('inf'), 0

    mse = np.mean((true_traj[:n_valid] - sim_traj[:n_valid]) ** 2)
    return mse, n_valid


def plot_all_forecasts(true_traj, sim_results, save_path):
    """Combined figure: ground truth + one panel per noise level.

    Args:
        true_traj: (N, 3) ground truth trajectory
        sim_results: list of dicts with keys 'sim_traj', 'noise_frac', 'forecast_mse', 'n_valid'
        save_path: path to save the figure
    """
    n_panels = 1 + len(sim_results)  # ground truth + each noise level
    n_cols = min(n_panels, 3)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))

    # Axis limits from ground truth
    xlim = (true_traj[:, 0].min() - 2, true_traj[:, 0].max() + 2)
    ylim = (true_traj[:, 1].min() - 2, true_traj[:, 1].max() + 2)
    zlim = (true_traj[:, 2].min() - 2, true_traj[:, 2].max() + 2)
    max_val = np.abs(true_traj).max() * 3

    # Panel 1: ground truth
    ax = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
    ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2],
            lw=0.5, color='C0')
    ax.set_title('Ground Truth', fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)

    # Subsequent panels: discovered ODEs
    for i, r in enumerate(sim_results):
        ax = fig.add_subplot(n_rows, n_cols, i + 2, projection='3d')

        sim = r['sim_traj'].copy()
        diverged_idx = np.where(np.abs(sim).max(axis=1) > max_val)[0]
        if len(diverged_idx) > 0:
            sim = sim[:diverged_idx[0]]

        if len(sim) > 1:
            ax.plot(sim[:, 0], sim[:, 1], sim[:, 2], lw=0.5, color='C1')

        noise_label = f"{r['noise_frac']:.0%}" if r['noise_frac'] > 0 else 'clean'
        subtitle = f"noise: {noise_label}\nMSE: {r['forecast_mse']:.2e}"
        if r['n_valid'] < len(true_traj):
            subtitle += f" (div @ {r['n_valid']})"
        ax.set_title(subtitle, fontsize=9)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)

    fig.suptitle('Lorenz: Autonomous Forecast from Discovered ODE', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved combined forecast plot: {save_path}")


def print_results(model: PolynomialRNN, noise_std, forecast_mse, n_valid):
    """Print discovered equations and active term counts."""
    active = model.count_active_terms()
    total_active = sum(active.values())

    print(f"\n{'='*60}")
    print(f"Noise std = {noise_std:.4f} | Forecast MSE = {forecast_mse:.2e} | "
          f"Active terms = {total_active} | Valid steps = {n_valid}")
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
    trajectory_clean = generate_lorenz_data(n_steps=CONFIG['n_steps'], dt=CONFIG['dt'], seed=SEED)

    scale = np.std(trajectory_clean, axis=0).mean()
    print(f"Average state std: {scale:.2f}")

    print("\nExpected continuous-time Lorenz ODE:")
    print("  dx/dt = -10.000*x + 10.000*y")
    print("  dy/dt = 28.000*x - 1.000*y - 1.000*x*z")
    print("  dz/dt = -2.667*z + 1.000*x*y")

    results = []
    sim_results = []  # for combined plot

    # Build list of (noise_frac, noise_std) pairs
    experiments = []
    for frac in CONFIG['noise_fractions']:
        noise_std = frac * scale if frac > 0 else 0.0
        experiments.append((frac, noise_std))

    for i, (frac, noise_std) in enumerate(experiments):
        noise_label = f'{frac:.0%}' if frac > 0 else 'clean'
        print(f"\n\n{'#'*60}")
        print(f"# Experiment {i+1}/{len(experiments)}: noise = {noise_label} (std={noise_std:.3f})")
        print(f"{'#'*60}")

        model = run_experiment(
            noise_std=noise_std,
            trajectory_clean=trajectory_clean,
            config=CONFIG,
            seed=SEED,
        )

        # Autonomous forecast
        n_steps = 5000
        trajectory_clean = generate_lorenz_data(n_steps=n_steps, dt=CONFIG['dt'], seed=SEED)
        h0 = trajectory_clean[0]
        sim_traj = simulate_ode(model, h0, n_steps, CONFIG['dt'])
        forecast_mse, n_valid = compute_forecast_mse(trajectory_clean, sim_traj)

        print_results(model, noise_std, forecast_mse, n_valid)

        active = model.count_active_terms()
        results.append({
            'noise_frac': frac,
            'noise_std': noise_std,
            'forecast_mse': forecast_mse,
            'n_valid': n_valid,
            'total_active': sum(active.values()),
            'active_per_state': active,
        })
        sim_results.append({
            'sim_traj': sim_traj,
            'noise_frac': frac,
            'forecast_mse': forecast_mse,
            'n_valid': n_valid,
        })

    # Combined forecast plot
    plot_all_forecasts(trajectory_clean, sim_results, save_path='lorenz_forecasts.png')

    # Summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Noise %':<10} {'Noise std':<12} {'Forecast MSE':<14} {'Valid steps':<14} {'Active terms':<12}")
    print("-" * 62)
    for r in results:
        print(f"{r['noise_frac']:<10.0%} {r['noise_std']:<12.4f} {r['forecast_mse']:<14.2e} "
              f"{r['n_valid']:<14} {r['total_active']:<12}")


if __name__ == '__main__':
    main()
