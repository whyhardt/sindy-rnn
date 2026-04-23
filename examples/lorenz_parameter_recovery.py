"""Lorenz parameter recovery study: noise × data quantity × method.

Compares three methods for recovering Lorenz ODE coefficients:
  1. sindy-rnn (factored) — implicit regularization from multilinear factorization
  2. sindy-rnn (direct)   — direct learnable polynomial coefficients
  3. E-SINDy              — ensemble SINDy with bagging + median aggregation

Ground truth Lorenz system (continuous-time):
    dx/dt = -10*x + 10*y           (sigma=10)
    dy/dt = 28*x - y - x*z         (rho=28)
    dz/dt = -8/3*z + x*y           (beta=8/3)

True active terms: 7 (x: x,y; y: x,y,x*z; z: z,x*y)

Metrics:
  - Coefficient error: ||c_discovered - c_true||_2 / ||c_true||_2  (relative)
  - Structure recovery: fraction of correctly identified active/inactive terms
  - Autonomous forecast MSE on clean test trajectory
  - Number of active terms discovered
"""

import sys
import os
import json
import time
import math
from itertools import combinations_with_replacement

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.math = math  # pysindy compat

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sindy_rnn import PolynomialRNN, fit
import pysindy as ps

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# Configuration
# ============================================================

# Grid dimensions
NOISE_FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
DATA_LENGTHS = [5000]#[500, 1000, 2000, 5000, 10000]
NUM_SEEDS = 5
# METHODS = ['factored', 'direct', 'esindy']
METHODS = ['esindy']

# Lorenz parameters
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
DT = 0.01

# Forecast evaluation
FORECAST_STEPS = 5000  # 50 time units

# sindy-rnn shared hyperparameters
RNN_CONFIG = {
    'degree': 2,
    'ensemble_size': 11,
    'epochs': 2000,
    'warmup_steps': 1000,
    'window_size': 100,
    'learning_rate': 5e-2,
    'l2': 5e-2,
    'pruning_frequency': 20,
    'pruning_threshold': 0.5,
    'ensemble_pruning_alpha': 0.05,
    'dropout': 0.1,
    'feature_dropout': 0.,
    'refit_epochs': 200,
}

# E-SINDy hyperparameters
ESINDY_CONFIG = {
    'threshold': 0.2,   # STLSQ sparsity threshold
    # 'alpha': 0.05,      # STLSQ L2 regularization
    'n_models': 11,     # number of bagging models (match sindy-rnn E)
}


# ============================================================
# Ground truth
# ============================================================

# True continuous-time coefficients for degree-2 polynomial library
# Library order: [1, x, y, z, x^2, x*y, x*z, y^2, y*z, z^2]
TRUE_COEFS = np.zeros((3, 10))
# dx/dt = -10*x + 10*y
TRUE_COEFS[0, 1] = -SIGMA     # x
TRUE_COEFS[0, 2] = SIGMA      # y
# dy/dt = 28*x - y - x*z
TRUE_COEFS[1, 1] = RHO        # x
TRUE_COEFS[1, 2] = -1.0       # y
TRUE_COEFS[1, 6] = -1.0       # x*z
# dz/dt = -8/3*z + x*y
TRUE_COEFS[2, 3] = -BETA      # z
TRUE_COEFS[2, 5] = 1.0        # x*y

TRUE_ACTIVE = TRUE_COEFS != 0  # (3, 10) boolean


# ============================================================
# Data generation
# ============================================================

def lorenz_rk4(x, dt=0.01, sigma=SIGMA, rho=RHO, beta=BETA):
    def f(x):
        return np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        ])
    k1 = f(x)
    k2 = f(x + dt / 2 * k1)
    k3 = f(x + dt / 2 * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_lorenz(n_steps, dt=DT, seed=42):
    rng = np.random.default_rng(seed)
    x = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)
    traj = [x.copy()]
    for _ in range(n_steps):
        x = lorenz_rk4(x, dt=dt)
        traj.append(x.copy())
    return np.array(traj)


def add_noise(trajectory, noise_frac, seed=0):
    if noise_frac == 0:
        return trajectory.copy()
    rng = np.random.default_rng(seed)
    scale = np.std(trajectory, axis=0)
    noise = rng.normal(0, 1, trajectory.shape) * scale * noise_frac
    return trajectory + noise


def chunk_trajectory(trajectory, window_size=100):
    N = len(trajectory) - 1
    n_windows = N // window_size
    xs = trajectory[:n_windows * window_size].reshape(n_windows, window_size, -1)
    ys = trajectory[1:n_windows * window_size + 1].reshape(n_windows, window_size, -1)
    return xs, ys


# ============================================================
# Autonomous forecast (method-agnostic)
# ============================================================

def simulate_polynomial_ode(coef_matrix, h0, n_steps, dt, degree=2):
    """Euler integration of polynomial ODE: h[t+1] = h[t] + dt * Θ(h) · ξ.

    Uses the same monomial ordering as both sindy-rnn and pysindy:
    [1, x, y, z, x², xy, xz, y², yz, z²] for degree=2, n_states=3.

    Args:
        coef_matrix: (n_states, n_terms) — ODE coefficients per state
        h0: (n_states,) — initial condition
        n_steps: number of Euler steps
        dt: timestep
        degree: polynomial degree

    Returns:
        trajectory: (n_steps+1, n_states) numpy array
    """
    n_states = len(h0)
    h = h0.copy().astype(np.float64)
    trajectory = [h.copy()]

    for _ in range(n_steps):
        # Build polynomial library
        terms = [1.0]
        for d in range(1, degree + 1):
            for combo in combinations_with_replacement(range(n_states), d):
                val = 1.0
                for idx in combo:
                    val *= h[idx]
                terms.append(val)
        library = np.array(terms)

        # Euler step: h' = h + dt * library @ coefs^T
        dh = library @ coef_matrix.T
        h = h + dt * dh
        trajectory.append(h.copy())

        # Early termination on divergence
        if np.any(np.abs(h) > 1e6):
            break

    return np.array(trajectory)


def compute_forecast_mse(true_traj, sim_traj):
    """Autonomous forecast MSE, truncated at divergence.

    Returns:
        mse: mean squared error over valid (non-diverged) portion
        n_valid: number of valid steps before divergence
    """
    max_val = np.abs(true_traj).max() * 3
    n_common = min(len(true_traj), len(sim_traj))

    diverged_idx = np.where(np.abs(sim_traj[:n_common]).max(axis=1) > max_val)[0]
    if len(diverged_idx) > 0:
        n_valid = diverged_idx[0]
    else:
        n_valid = n_common

    if n_valid < 2:
        return float('inf'), 0

    mse = np.mean((true_traj[:n_valid] - sim_traj[:n_valid]) ** 2)
    return mse, n_valid


# ============================================================
# Method: sindy-rnn (factored or direct)
# ============================================================

def run_sindy_rnn(trajectory_noisy, config, direct=False, seed=42):
    """Train sindy-rnn and return coefficient matrix.

    Returns:
        coef_matrix: (3, 10) — ODE coefficients
        n_active: number of nonzero terms
    """
    torch.manual_seed(seed)

    xs, ys = chunk_trajectory(trajectory_noisy, config['window_size'])
    xs = torch.tensor(xs, dtype=torch.float32).to(DEVICE)
    ys = torch.tensor(ys, dtype=torch.float32).to(DEVICE)

    model = PolynomialRNN(
        n_states=3, n_controls=0,
        polynomial_degree=config['degree'],
        ensemble_size=config['ensemble_size'],
        dt=DT,
        state_names=['x', 'y', 'z'],
        dropout=config['dropout'],
        feature_dropout=config['feature_dropout'],
        compiled_forward=False,
        direct=direct,
        decomposed=not direct,
    ).to(DEVICE)

    fit(model, xs, ys,
        epochs=config['epochs'],
        warmup_steps=config['warmup_steps'],
        ensemble_pruning_alpha=config['ensemble_pruning_alpha'],
        pruning_threshold=config['pruning_threshold'],
        pruning_frequency=config['pruning_frequency'],
        pruning_method='median',
        learning_rate=config['learning_rate'],
        l2=config['l2'],
        dt=DT,
        refit_epochs=config.get('refit_epochs', 0),
        verbose=False,
    )

    # Extract ODE coefficients (theta IS the ODE)
    coefs = model.get_coefficients(aggregate=True)
    coef_matrix = np.zeros((3, model.rnn._n_library_terms))
    for i, name in enumerate(model.state_names):
        coef_matrix[i] = coefs[name].cpu().numpy()

    active = model.count_active_terms()
    n_active = sum(active.values())

    return coef_matrix, n_active


# ============================================================
# Method: E-SINDy (ensemble + bagging + median aggregation)
# ============================================================

def run_esindy(trajectory_noisy, config=ESINDY_CONFIG, seed=42):
    """Run E-SINDy with bagging and median coefficient aggregation.

    Returns:
        coef_matrix: (3, 10) — median ODE coefficients
        n_active: number of nonzero terms
    """
    z = trajectory_noisy
    z_dot = np.gradient(z, DT, axis=0)

    ensemble_optimizer = ps.EnsembleOptimizer(
        opt=ps.STLSQ(threshold=config['threshold']),
        bagging=True,
        n_models=config['n_models'],
    )

    sindy_model = ps.SINDy(
        optimizer=ensemble_optimizer,
        feature_library=ps.PolynomialLibrary(degree=2),
    )
    sindy_model.fit(z, t=DT, x_dot=z_dot)

    # Median aggregation (bragging) across ensemble members
    coef_stack = np.array(ensemble_optimizer.coef_list)  # (n_models, 3, 10)
    coef_matrix = np.median(coef_stack, axis=0)

    # Threshold small median values
    coef_matrix[np.abs(coef_matrix) < config['threshold']] = 0

    n_active = np.count_nonzero(coef_matrix)

    return coef_matrix, n_active


# ============================================================
# Evaluation metrics
# ============================================================

def compute_metrics(coef_matrix, n_active, forecast_mse, n_valid):
    """Compute coefficient error, structure recovery, and forecast metrics."""
    # Relative coefficient error
    true_norm = np.linalg.norm(TRUE_COEFS)
    coef_error = np.linalg.norm(coef_matrix - TRUE_COEFS) / true_norm

    # Structure recovery (F1 score)
    discovered_active = np.abs(coef_matrix) > 1e-6
    tp = np.sum(discovered_active & TRUE_ACTIVE)
    fp = np.sum(discovered_active & ~TRUE_ACTIVE)
    fn = np.sum(~discovered_active & TRUE_ACTIVE)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    # Exact structure match
    exact_match = np.array_equal(discovered_active, TRUE_ACTIVE)

    return {
        'coef_error': coef_error,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match': exact_match,
        'forecast_mse': forecast_mse,
        'n_valid': n_valid,
        'n_active': n_active,
    }


# ============================================================
# Plotting
# ============================================================

def plot_forecast_grid(all_results, forecast_traj, save_path):
    """Grid of forecast plots: rows=data sizes, cols=noise levels.

    Each subplot overlays all seeds per method with ground truth in black.
    Plots x(t) time series to show divergence and attractor behavior.

    Args:
        all_results: list of result dicts with 'method', 'noise_frac', 'n_steps', 'seed'
        forecast_traj: ground truth forecast trajectory (n_steps+1, 3)
        save_path: path to save the figure
    """
    n_rows = len(DATA_LENGTHS)
    n_cols = len(NOISE_FRACTIONS)

    colors = {'factored': '#2196F3', 'direct': '#FF9800', 'esindy': '#4CAF50'}
    labels = {'factored': 'factored', 'direct': 'direct', 'esindy': 'E-SINDy'}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows),
                             squeeze=False)

    # Time axis (subsample for plotting)
    t_full = np.arange(len(forecast_traj)) * DT
    step = max(1, len(t_full) // 500)  # subsample for speed

    for row, n_steps in enumerate(DATA_LENGTHS):
        for col, nf in enumerate(NOISE_FRACTIONS):
            ax = axes[row, col]

            # Ground truth
            ax.plot(t_full[::step], forecast_traj[::step, 0],
                    color='black', lw=0.8, alpha=0.6, zorder=10)

            # Overlay all seeds × methods
            for method in METHODS:
                for r in all_results:
                    if (r['method'] == method and r['noise_frac'] == nf
                            and r['n_steps'] == n_steps and 'sim_traj' in r):
                        sim = r['sim_traj']
                        t_sim = np.arange(len(sim)) * DT
                        step_sim = max(1, len(t_sim) // 500)
                        ax.plot(t_sim[::step_sim], sim[::step_sim, 0],
                                color=colors[method], lw=0.4, alpha=0.5)

            # Axis limits from ground truth
            x_range = forecast_traj[:, 0]
            margin = (x_range.max() - x_range.min()) * 0.15
            ax.set_ylim(x_range.min() - margin, x_range.max() + margin)
            ax.set_xlim(0, t_full[-1])

            # Labels
            if row == 0:
                noise_label = f'{nf:.0%}' if nf > 0 else 'clean'
                ax.set_title(f'noise: {noise_label}', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'N={n_steps}', fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel('t', fontsize=8)

            ax.tick_params(labelsize=6)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='black', lw=1, label='ground truth')]
    for method in METHODS:
        legend_elements.append(
            Line2D([0], [0], color=colors[method], lw=1, label=labels[method]))
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=len(METHODS) + 1, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle('Autonomous Forecast: x(t) from Discovered ODE', fontsize=13, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nForecast grid saved to {save_path}")


def plot_results(results):
    """Coefficient error plots and F1 heatmaps."""
    colors = {'factored': '#2196F3', 'direct': '#FF9800', 'esindy': '#4CAF50'}
    markers = {'factored': 'o', 'direct': 's', 'esindy': '^'}
    labels = {'factored': 'sindy-rnn (factored)', 'direct': 'sindy-rnn (direct)',
              'esindy': 'E-SINDy'}

    # --- Coefficient error vs noise (columns = different data sizes) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    data_sizes_to_plot = [1000, 5000, 10000]
    for col, n_steps in enumerate(data_sizes_to_plot):
        ax = axes[0, col]
        for method in METHODS:
            means, stds, nfs = [], [], []
            for nf in NOISE_FRACTIONS:
                vals = [r['coef_error'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == n_steps and not np.isnan(r.get('coef_error', float('nan')))]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    nfs.append(nf * 100)
            if means:
                ax.errorbar(nfs, means, yerr=stds, marker=markers[method],
                           color=colors[method], label=labels[method],
                           capsize=3, linewidth=1.5, markersize=5)
        ax.set_xlabel('Noise (%)')
        ax.set_ylabel('Relative coefficient error')
        ax.set_title(f'N = {n_steps}')
        ax.set_yscale('log')
        if col == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Coefficient error vs data size (columns = different noise levels) ---
    noise_to_plot = [0.0, 0.05, 0.10]
    for col, nf in enumerate(noise_to_plot):
        ax = axes[1, col]
        for method in METHODS:
            means, stds, ns = [], [], []
            for n_steps in DATA_LENGTHS:
                vals = [r['coef_error'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == n_steps and not np.isnan(r.get('coef_error', float('nan')))]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    ns.append(n_steps)
            if means:
                ax.errorbar(ns, means, yerr=stds, marker=markers[method],
                           color=colors[method], label=labels[method],
                           capsize=3, linewidth=1.5, markersize=5)
        ax.set_xlabel('Training data length')
        ax.set_ylabel('Relative coefficient error')
        ax.set_title(f'Noise = {nf:.0%}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        if col == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = 'lorenz_recovery_plot.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.close()

    # --- F1 heatmap per method ---
    fig, axes = plt.subplots(1, len(METHODS), figsize=(5 * len(METHODS), 4.5))
    if len(METHODS) == 1:
        axes = [axes]
    for idx, method in enumerate(METHODS):
        ax = axes[idx]
        f1_matrix = np.full((len(NOISE_FRACTIONS), len(DATA_LENGTHS)), np.nan)
        for i, nf in enumerate(NOISE_FRACTIONS):
            for j, ns in enumerate(DATA_LENGTHS):
                vals = [r['f1'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == ns]
                if vals:
                    f1_matrix[i, j] = np.mean(vals)

        im = ax.imshow(f1_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                       origin='lower')
        ax.set_xticks(range(len(DATA_LENGTHS)))
        ax.set_xticklabels(DATA_LENGTHS, fontsize=8)
        ax.set_yticks(range(len(NOISE_FRACTIONS)))
        ax.set_yticklabels([f'{nf:.0%}' for nf in NOISE_FRACTIONS], fontsize=8)
        ax.set_xlabel('Data length')
        ax.set_ylabel('Noise fraction')
        ax.set_title(labels[method])
        plt.colorbar(im, ax=ax, label='F1 Score')

        for i in range(len(NOISE_FRACTIONS)):
            for j in range(len(DATA_LENGTHS)):
                val = f1_matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if val < 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=7, color=color)

    plt.tight_layout()
    save_path = 'lorenz_recovery_f1_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"F1 heatmap saved to {save_path}")
    plt.close()


def print_summary_tables(results):
    """Print aggregated summary tables."""
    labels = {'factored': 'factored', 'direct': 'direct', 'esindy': 'E-SINDy'}

    print(f"\n\n{'='*80}")
    print("SUMMARY: Coefficient Error (mean ± std across seeds)")
    print(f"{'='*80}")

    for method in METHODS:
        print(f"\n--- {labels[method]} ---")
        header = f"{'Noise %':<10}"
        for n in DATA_LENGTHS:
            header += f"{'N='+str(n):>16}"
        print(header)
        print("-" * (10 + 16 * len(DATA_LENGTHS)))

        for nf in NOISE_FRACTIONS:
            row = f"{nf:<10.0%}"
            for n in DATA_LENGTHS:
                vals = [r['coef_error'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == n and not np.isnan(r.get('coef_error', float('nan')))]
                if vals:
                    row += f"{np.mean(vals):>8.4f}±{np.std(vals):>5.4f}"
                else:
                    row += f"{'N/A':>16}"
            print(row)

    print(f"\n\n{'='*80}")
    print("SUMMARY: Exact Structure Match Rate (across seeds)")
    print(f"{'='*80}")

    for method in METHODS:
        print(f"\n--- {labels[method]} ---")
        header = f"{'Noise %':<10}"
        for n in DATA_LENGTHS:
            header += f"{'N='+str(n):>12}"
        print(header)
        print("-" * (10 + 12 * len(DATA_LENGTHS)))

        for nf in NOISE_FRACTIONS:
            row = f"{nf:<10.0%}"
            for n in DATA_LENGTHS:
                vals = [r['exact_match'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == n]
                if vals:
                    row += f"{np.mean(vals):>10.0%}  "
                else:
                    row += f"{'N/A':>12}"
            print(row)

    print(f"\n\n{'='*80}")
    print("SUMMARY: Forecast MSE (mean across seeds)")
    print(f"{'='*80}")

    for method in METHODS:
        print(f"\n--- {labels[method]} ---")
        header = f"{'Noise %':<10}"
        for n in DATA_LENGTHS:
            header += f"{'N='+str(n):>14}"
        print(header)
        print("-" * (10 + 14 * len(DATA_LENGTHS)))

        for nf in NOISE_FRACTIONS:
            row = f"{nf:<10.0%}"
            for n in DATA_LENGTHS:
                vals = [r['forecast_mse'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == n and np.isfinite(r.get('forecast_mse', float('nan')))]
                if vals:
                    row += f"{np.mean(vals):>14.2e}"
                else:
                    row += f"{'div':>14}"
            print(row)


# ============================================================
# Main study
# ============================================================

def main():
    print("Lorenz Parameter Recovery Study")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Noise fractions: {NOISE_FRACTIONS}")
    print(f"Data lengths: {DATA_LENGTHS}")
    print(f"Seeds: {NUM_SEEDS}")
    print(f"Methods: {METHODS}")
    print(f"\nTrue ODE:")
    print(f"  dx/dt = -{SIGMA:.0f}*x + {SIGMA:.0f}*y")
    print(f"  dy/dt = {RHO:.0f}*x - y - x*z")
    print(f"  dz/dt = -{BETA:.4f}*z + x*y")
    print(f"  True active terms: {np.sum(TRUE_ACTIVE)}")

    # Pre-generate all trajectories (deterministic)
    print("\nPre-generating trajectories...")
    trajectories = {}
    for seed in range(NUM_SEEDS):
        for n_steps in DATA_LENGTHS:
            trajectories[(n_steps, seed)] = generate_lorenz(n_steps, dt=DT, seed=seed * 1000)
    print(f"  Generated {len(trajectories)} training trajectories")

    # Reference trajectory for forecast evaluation (same for all)
    forecast_traj = generate_lorenz(FORECAST_STEPS, dt=DT, seed=99999)
    h0_forecast = forecast_traj[0]
    print(f"  Forecast trajectory: {FORECAST_STEPS} steps from h0={h0_forecast}")

    # Compute average state std (for noise calibration)
    ref_traj = generate_lorenz(10000, dt=DT, seed=42)
    state_std = np.std(ref_traj, axis=0).mean()
    print(f"  Reference state std: {state_std:.2f}")

    all_results = []
    total_runs = len(NOISE_FRACTIONS) * len(DATA_LENGTHS) * NUM_SEEDS * len(METHODS)
    run_idx = 0

    for noise_frac in NOISE_FRACTIONS:
        for n_steps in DATA_LENGTHS:
            for seed in range(NUM_SEEDS):
                traj_clean = trajectories[(n_steps, seed)]
                traj_noisy = add_noise(traj_clean, noise_frac, seed=seed * 100 + 1)

                for method in METHODS:
                    run_idx += 1
                    tag = f"[{run_idx}/{total_runs}] noise={noise_frac:.0%}, N={n_steps}, seed={seed}, {method}"
                    print(f"\n{tag}")

                    t0 = time.time()
                    try:
                        if method == 'factored':
                            coefs, n_active = run_sindy_rnn(
                                traj_noisy, RNN_CONFIG, direct=False, seed=seed)
                        elif method == 'direct':
                            coefs, n_active = run_sindy_rnn(
                                traj_noisy, RNN_CONFIG, direct=True, seed=seed)
                        elif method == 'esindy':
                            coefs, n_active = run_esindy(traj_noisy, seed=seed)
                        else:
                            raise ValueError(f"Unknown method: {method}")

                        elapsed = time.time() - t0

                        # Autonomous forecast from discovered ODE
                        sim_traj = simulate_polynomial_ode(
                            coefs, h0_forecast, FORECAST_STEPS, DT)
                        forecast_mse, n_valid = compute_forecast_mse(
                            forecast_traj, sim_traj)

                        metrics = compute_metrics(coefs, n_active, forecast_mse, n_valid)
                        metrics.update({
                            'method': method,
                            'noise_frac': noise_frac,
                            'n_steps': n_steps,
                            'seed': seed,
                            'time': elapsed,
                            'sim_traj': sim_traj,
                        })
                        all_results.append(metrics)

                        print(f"  coef_err={metrics['coef_error']:.4f}, "
                              f"F1={metrics['f1']:.3f}, "
                              f"exact={metrics['exact_match']}, "
                              f"forecast_MSE={forecast_mse:.2e}, "
                              f"valid={n_valid}/{FORECAST_STEPS}, "
                              f"terms={n_active}, "
                              f"time={elapsed:.1f}s")

                    except Exception as e:
                        print(f"  FAILED: {e}")
                        all_results.append({
                            'method': method,
                            'noise_frac': noise_frac,
                            'n_steps': n_steps,
                            'seed': seed,
                            'coef_error': float('nan'),
                            'f1': 0.0,
                            'exact_match': False,
                            'forecast_mse': float('nan'),
                            'n_valid': 0,
                            'n_active': 0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'time': 0.0,
                        })

    # Save raw results (without sim_traj arrays)
    results_for_json = [{k: v for k, v in r.items() if k != 'sim_traj'}
                        for r in all_results]
    results_path = 'lorenz_recovery_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_for_json, f, indent=2, default=str)
    print(f"\nRaw results saved to {results_path}")

    # Print summary tables
    print_summary_tables(all_results)

    # Generate plots
    plot_results(all_results)
    plot_forecast_grid(all_results, forecast_traj, save_path='lorenz_recovery_forecasts.png')


if __name__ == '__main__':
    main()
