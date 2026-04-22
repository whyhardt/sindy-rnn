"""Lorenz parameter recovery study: noise × data quantity × method.

Compares three methods for recovering Lorenz ODE coefficients:
  1. sindy-rnn (factored) — implicit regularization from multilinear factorization
  2. sindy-rnn (direct)   — direct learnable polynomial coefficients
  3. pysindy STLSQ        — standard SINDy with sequential thresholded least squares

Ground truth Lorenz system (continuous-time):
    dx/dt = -10*x + 10*y           (sigma=10)
    dy/dt = 28*x - y - x*z         (rho=28)
    dz/dt = -8/3*z + x*y           (beta=8/3)

True active terms: 7 (x: x,y; y: x,y,x*z; z: z,x*y)

Metrics:
  - Coefficient error: ||c_discovered - c_true||_2 / ||c_true||_2  (relative)
  - Structure recovery: fraction of correctly identified active/inactive terms
  - Prediction MSE on clean test data
  - Number of active terms discovered
"""

import sys
import os
import json
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.math = math  # pysindy compat

import torch
import torch.nn.functional as F
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
NOISE_FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
DATA_LENGTHS = [500, 1000, 2000, 5000, 10000]
NUM_SEEDS = 5
METHODS = ['factored', 'direct', 'stlsq']

# Lorenz parameters
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
DT = 0.01

# sindy-rnn shared hyperparameters
RNN_CONFIG = {
    'degree': 2,
    'ensemble_size': 11,
    'epochs': 1000,
    'warmup_steps': 500,
    'window_size': 100,
    'learning_rate': 1e-2,
    'l2': 5e-2,
    'pruning_frequency': 20,
    'pruning_threshold': 0.5,
    'ensemble_pruning_alpha': 0.05,
    'dropout': 0.1,
    'feature_dropout': 0.,
    'refit_epochs': 200,
}

# pysindy hyperparameters
STLSQ_THRESHOLD = 0.1
STLSQ_ALPHA = 0.05  # L2 regularization


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
# Method: sindy-rnn (factored or direct)
# ============================================================

def run_sindy_rnn(trajectory_noisy, trajectory_test, config, direct=False, seed=42):
    torch.manual_seed(seed)

    xs, ys = chunk_trajectory(trajectory_noisy, config['window_size'])
    xs = torch.tensor(xs, dtype=torch.float32).to(DEVICE)
    ys = torch.tensor(ys, dtype=torch.float32).to(DEVICE)

    xs_test, ys_test = chunk_trajectory(trajectory_test, config['window_size'])
    xs_test = torch.tensor(xs_test, dtype=torch.float32).to(DEVICE)
    ys_test = torch.tensor(ys_test, dtype=torch.float32).to(DEVICE)

    model = PolynomialRNN(
        n_states=3, n_controls=0,
        polynomial_degree=config['degree'],
        ensemble_size=config['ensemble_size'],
        state_names=['x', 'y', 'z'],
        dropout=config['dropout'],
        feature_dropout=config['feature_dropout'],
        compiled_forward=False,
        direct=direct,
        decomposed=False,
    ).to(DEVICE)

    fit(model, xs, ys,
        xs_test=xs_test, ys_test=ys_test,
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

    # Extract continuous-time coefficients
    coefs = model.get_coefficients(aggregate=True)
    coef_matrix = np.zeros((3, model.rnn._n_library_terms))
    for i, name in enumerate(model.state_names):
        c_disc = coefs[name].cpu().numpy()
        # Convert to continuous time
        c_cont = c_disc / DT
        self_idx = model.rnn._linear_indices[i].item()
        c_cont[self_idx] = (c_disc[self_idx] - 1.0) / DT
        coef_matrix[i] = c_cont

    # Test MSE on clean data
    with torch.no_grad():
        model.eval()
        E = model.ensemble_size
        x_te = xs_test.unsqueeze(0).expand(E, -1, -1, -1)
        yp_te, _ = model(x_te)
        y_te = ys_test.unsqueeze(0).expand(E, -1, -1, -1)
        test_mse = F.mse_loss(yp_te, y_te).item()

    active = model.count_active_terms()
    n_active = sum(active.values())

    return coef_matrix, test_mse, n_active


# ============================================================
# Method: pysindy STLSQ
# ============================================================

def run_stlsq(trajectory_noisy, trajectory_test, seed=42):
    # Fit SINDy on training data
    z = trajectory_noisy
    z_dot = np.gradient(z, DT, axis=0)

    sindy_model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=STLSQ_THRESHOLD, alpha=STLSQ_ALPHA),
        feature_library=ps.PolynomialLibrary(degree=2),
    )
    sindy_model.fit(z, t=DT, x_dot=z_dot)

    # Extract coefficients (pysindy: (n_states, n_features))
    coef_matrix = sindy_model.coefficients()

    # Test MSE: simulate forward and compare
    z_test = trajectory_test
    z_dot_test = np.gradient(z_test, DT, axis=0)

    # Predict derivatives and compute MSE
    z_dot_pred = sindy_model.predict(z_test)
    test_mse = np.mean((z_dot_pred - z_dot_test) ** 2)

    # For fair comparison, also compute discrete next-step MSE
    # z_{t+1} ≈ z_t + dt * f(z_t)
    z_next_pred = z_test[:-1] + DT * sindy_model.predict(z_test[:-1])
    z_next_true = z_test[1:]
    discrete_mse = np.mean((z_next_pred - z_next_true) ** 2)

    n_active = np.count_nonzero(coef_matrix)

    return coef_matrix, discrete_mse, n_active


# ============================================================
# Evaluation metrics
# ============================================================

def compute_metrics(coef_matrix, n_active, test_mse):
    """Compute coefficient error and structure recovery metrics."""
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
        'test_mse': test_mse,
        'n_active': n_active,
    }


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
    test_trajectories = {}
    for seed in range(NUM_SEEDS):
        for n_steps in DATA_LENGTHS:
            trajectories[(n_steps, seed)] = generate_lorenz(n_steps, dt=DT, seed=seed * 1000)
            test_trajectories[(n_steps, seed)] = generate_lorenz(2000, dt=DT, seed=seed * 1000 + 500)
    print(f"  Generated {len(trajectories)} training + {len(test_trajectories)} test trajectories")

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
                traj_test = test_trajectories[(n_steps, seed)]

                for method in METHODS:
                    run_idx += 1
                    tag = f"[{run_idx}/{total_runs}] noise={noise_frac:.0%}, N={n_steps}, seed={seed}, {method}"
                    print(f"\n{tag}")

                    t0 = time.time()
                    try:
                        if method == 'factored':
                            coefs, mse, n_active = run_sindy_rnn(
                                traj_noisy, traj_test, RNN_CONFIG, direct=False, seed=seed)
                        elif method == 'direct':
                            coefs, mse, n_active = run_sindy_rnn(
                                traj_noisy, traj_test, RNN_CONFIG, direct=True, seed=seed)
                        elif method == 'stlsq':
                            coefs, mse, n_active = run_stlsq(traj_noisy, traj_test, seed=seed)
                        else:
                            raise ValueError(f"Unknown method: {method}")

                        elapsed = time.time() - t0
                        metrics = compute_metrics(coefs, n_active, mse)
                        metrics.update({
                            'method': method,
                            'noise_frac': noise_frac,
                            'n_steps': n_steps,
                            'seed': seed,
                            'time': elapsed,
                        })
                        all_results.append(metrics)

                        print(f"  coef_err={metrics['coef_error']:.4f}, "
                              f"F1={metrics['f1']:.3f}, "
                              f"exact={metrics['exact_match']}, "
                              f"MSE={mse:.6f}, "
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
                            'test_mse': float('nan'),
                            'n_active': 0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'time': 0.0,
                        })

    # Save raw results
    results_path = 'lorenz_recovery_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to {results_path}")

    # ================================================================
    # Print summary tables
    # ================================================================
    print_summary_tables(all_results)

    # ================================================================
    # Generate plots
    # ================================================================
    plot_results(all_results)


def print_summary_tables(results):
    """Print aggregated summary tables."""
    import itertools

    print(f"\n\n{'='*80}")
    print("SUMMARY: Coefficient Error (mean ± std across seeds)")
    print(f"{'='*80}")

    for method in METHODS:
        print(f"\n--- {method} ---")
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
                        and r['n_steps'] == n and not np.isnan(r['coef_error'])]
                if vals:
                    row += f"{np.mean(vals):>8.4f}±{np.std(vals):>5.4f}"
                else:
                    row += f"{'N/A':>16}"
            print(row)

    print(f"\n\n{'='*80}")
    print("SUMMARY: Structure F1 Score (mean across seeds)")
    print(f"{'='*80}")

    for method in METHODS:
        print(f"\n--- {method} ---")
        header = f"{'Noise %':<10}"
        for n in DATA_LENGTHS:
            header += f"{'N='+str(n):>12}"
        print(header)
        print("-" * (10 + 12 * len(DATA_LENGTHS)))

        for nf in NOISE_FRACTIONS:
            row = f"{nf:<10.0%}"
            for n in DATA_LENGTHS:
                vals = [r['f1'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == n]
                if vals:
                    row += f"{np.mean(vals):>12.3f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)

    print(f"\n\n{'='*80}")
    print("SUMMARY: Exact Structure Match Rate (across seeds)")
    print(f"{'='*80}")

    for method in METHODS:
        print(f"\n--- {method} ---")
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


def plot_results(results):
    """Generate publication-quality plots for the parameter recovery study."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    colors = {'factored': '#2196F3', 'direct': '#FF9800', 'stlsq': '#4CAF50'}
    markers = {'factored': 'o', 'direct': 's', 'stlsq': '^'}
    labels = {'factored': 'sindy-rnn (factored)', 'direct': 'sindy-rnn (direct)', 'stlsq': 'PySINDy STLSQ'}

    # --- Top row: Coefficient error vs noise (columns = different data sizes) ---
    data_sizes_to_plot = [1000, 5000, 10000]
    for col, n_steps in enumerate(data_sizes_to_plot):
        ax = axes[0, col]
        for method in METHODS:
            means, stds, nfs = [], [], []
            for nf in NOISE_FRACTIONS:
                vals = [r['coef_error'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == n_steps and not np.isnan(r['coef_error'])]
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

    # --- Bottom row: metrics vs data size (columns = different noise levels) ---
    noise_to_plot = [0.0, 0.05, 0.10]
    for col, nf in enumerate(noise_to_plot):
        ax = axes[1, col]
        for method in METHODS:
            means, stds, ns = [], [], []
            for n_steps in DATA_LENGTHS:
                vals = [r['coef_error'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == n_steps and not np.isnan(r['coef_error'])]
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
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

        # Annotate cells
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


if __name__ == '__main__':
    main()
