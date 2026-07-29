"""Aggregate results/recovery/*.json from recovery_run.py into summary
tables + plots.

Works on whatever grid cells have finished — safe to run against a partial
sweep while a cluster job is still running.

Usage:
    python recovery_aggregate.py
    python recovery_aggregate.py --results_dir results/recovery
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data import generate_lorenz, simulate_polynomial_ode, RESULTS_DIR
from recovery_run import SIGMA, RHO, BETA, DT, FORECAST_STEPS

LABELS = {'factored': 'sindy-rnn (factored)', 'direct': 'sindy-rnn (direct)', 'esindy': 'E-SINDy'}
COLORS = {'factored': '#2196F3', 'direct': '#FF9800', 'esindy': '#4CAF50'}
MARKERS = {'factored': 'o', 'direct': 's', 'esindy': '^'}


def load_results(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, '*.json')))
    results = []
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        r['coef_matrix'] = np.array(r['coef_matrix'])
        results.append(r)
    return results


def print_summary_tables(results, noise_fractions, data_lengths, methods):
    print(f"\n\n{'='*80}")
    print("SUMMARY: Coefficient Error (mean ± std across seeds)")
    print(f"{'='*80}")
    for method in methods:
        print(f"\n--- {LABELS.get(method, method)} ---")
        header = f"{'Noise %':<10}" + ''.join(f"{'N='+str(n):>16}" for n in data_lengths)
        print(header)
        print("-" * (10 + 16 * len(data_lengths)))
        for nf in noise_fractions:
            row = f"{nf:<10.0%}"
            for n in data_lengths:
                vals = [r['coef_error'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == n and not np.isnan(r.get('coef_error', float('nan')))]
                row += f"{np.mean(vals):>8.4f}±{np.std(vals):>5.4f}" if vals else f"{'N/A':>16}"
            print(row)

    print(f"\n\n{'='*80}")
    print("SUMMARY: Exact Structure Match Rate (across seeds)")
    print(f"{'='*80}")
    for method in methods:
        print(f"\n--- {LABELS.get(method, method)} ---")
        header = f"{'Noise %':<10}" + ''.join(f"{'N='+str(n):>12}" for n in data_lengths)
        print(header)
        print("-" * (10 + 12 * len(data_lengths)))
        for nf in noise_fractions:
            row = f"{nf:<10.0%}"
            for n in data_lengths:
                vals = [r['exact_match'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf and r['n_steps'] == n]
                row += f"{np.mean(vals):>10.0%}  " if vals else f"{'N/A':>12}"
            print(row)

    print(f"\n\n{'='*80}")
    print("SUMMARY: Forecast MSE (mean across seeds)")
    print(f"{'='*80}")
    for method in methods:
        print(f"\n--- {LABELS.get(method, method)} ---")
        header = f"{'Noise %':<10}" + ''.join(f"{'N='+str(n):>14}" for n in data_lengths)
        print(header)
        print("-" * (10 + 14 * len(data_lengths)))
        for nf in noise_fractions:
            row = f"{nf:<10.0%}"
            for n in data_lengths:
                vals = [r['forecast_mse'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == n and np.isfinite(r.get('forecast_mse', float('nan')))]
                row += f"{np.mean(vals):>14.2e}" if vals else f"{'div':>14}"
            print(row)


def plot_coef_error(results, noise_fractions, data_lengths, methods, save_path):
    """Coefficient error vs noise (top row) and vs data size (bottom row)."""
    data_sizes_to_plot = [n for n in [1000, 5000, 10000] if n in data_lengths] or data_lengths[:3]
    noise_to_plot = [nf for nf in [0.0, 0.05, 0.10] if nf in noise_fractions] or noise_fractions[:3]

    n_cols = max(len(data_sizes_to_plot), len(noise_to_plot), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9), squeeze=False)
    for col in range(len(data_sizes_to_plot), n_cols):
        axes[0, col].axis('off')
    for col in range(len(noise_to_plot), n_cols):
        axes[1, col].axis('off')
    for col, n_steps in enumerate(data_sizes_to_plot):
        ax = axes[0, col]
        for method in methods:
            means, stds, nfs = [], [], []
            for nf in noise_fractions:
                vals = [r['coef_error'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == n_steps and not np.isnan(r.get('coef_error', float('nan')))]
                if vals:
                    means.append(np.mean(vals)); stds.append(np.std(vals)); nfs.append(nf * 100)
            if means:
                ax.errorbar(nfs, means, yerr=stds, marker=MARKERS.get(method, 'o'),
                           color=COLORS.get(method), label=LABELS.get(method, method),
                           capsize=3, linewidth=1.5, markersize=5)
        ax.set_xlabel('Noise (%)'); ax.set_ylabel('Relative coefficient error')
        ax.set_title(f'N = {n_steps}'); ax.set_yscale('log')
        if col == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for col, nf in enumerate(noise_to_plot):
        ax = axes[1, col]
        for method in methods:
            means, stds, ns = [], [], []
            for n_steps in data_lengths:
                vals = [r['coef_error'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf
                        and r['n_steps'] == n_steps and not np.isnan(r.get('coef_error', float('nan')))]
                if vals:
                    means.append(np.mean(vals)); stds.append(np.std(vals)); ns.append(n_steps)
            if means:
                ax.errorbar(ns, means, yerr=stds, marker=MARKERS.get(method, 'o'),
                           color=COLORS.get(method), label=LABELS.get(method, method),
                           capsize=3, linewidth=1.5, markersize=5)
        ax.set_xlabel('Training data length'); ax.set_ylabel('Relative coefficient error')
        ax.set_title(f'Noise = {nf:.0%}'); ax.set_xscale('log'); ax.set_yscale('log')
        if col == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot saved to {save_path}")


def plot_f1_heatmap(results, noise_fractions, data_lengths, methods, save_path):
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5), squeeze=False)
    axes = axes[0]
    for idx, method in enumerate(methods):
        ax = axes[idx]
        f1_matrix = np.full((len(noise_fractions), len(data_lengths)), np.nan)
        for i, nf in enumerate(noise_fractions):
            for j, ns in enumerate(data_lengths):
                vals = [r['f1'] for r in results
                        if r['method'] == method and r['noise_frac'] == nf and r['n_steps'] == ns]
                if vals:
                    f1_matrix[i, j] = np.mean(vals)

        im = ax.imshow(f1_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
        ax.set_xticks(range(len(data_lengths))); ax.set_xticklabels(data_lengths, fontsize=8)
        ax.set_yticks(range(len(noise_fractions)))
        ax.set_yticklabels([f'{nf:.0%}' for nf in noise_fractions], fontsize=8)
        ax.set_xlabel('Data length'); ax.set_ylabel('Noise fraction')
        ax.set_title(LABELS.get(method, method))
        plt.colorbar(im, ax=ax, label='F1 Score')
        for i in range(len(noise_fractions)):
            for j in range(len(data_lengths)):
                val = f1_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                           color='white' if val < 0.5 else 'black')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"F1 heatmap saved to {save_path}")


def plot_forecast_grid(results, noise_fractions, data_lengths, methods, save_path):
    """Grid of forecast plots: rows=data sizes, cols=noise levels. Recomputes
    sim_traj from each result's saved coef_matrix (cheap, avoids storing
    5000x3 trajectories per grid cell).
    """
    forecast_traj = generate_lorenz(FORECAST_STEPS, DT, SIGMA, RHO, BETA, seed=99999)
    t_full = np.arange(len(forecast_traj)) * DT
    step = max(1, len(t_full) // 500)

    fig, axes = plt.subplots(len(data_lengths), len(noise_fractions),
                             figsize=(3.5 * len(noise_fractions), 2.5 * len(data_lengths)),
                             squeeze=False)

    for row, n_steps in enumerate(data_lengths):
        for col, nf in enumerate(noise_fractions):
            ax = axes[row, col]
            ax.plot(t_full[::step], forecast_traj[::step, 0],
                    color='black', lw=0.8, alpha=0.6, zorder=10)

            for method in methods:
                for r in results:
                    if r['method'] == method and r['noise_frac'] == nf and r['n_steps'] == n_steps:
                        sim = simulate_polynomial_ode(r['coef_matrix'], forecast_traj[0],
                                                      FORECAST_STEPS, DT)
                        t_sim = np.arange(len(sim)) * DT
                        step_sim = max(1, len(t_sim) // 500)
                        ax.plot(t_sim[::step_sim], sim[::step_sim, 0],
                                color=COLORS.get(method), lw=0.4, alpha=0.5)

            x_range = forecast_traj[:, 0]
            margin = (x_range.max() - x_range.min()) * 0.15
            ax.set_ylim(x_range.min() - margin, x_range.max() + margin)
            ax.set_xlim(0, t_full[-1])
            if row == 0:
                ax.set_title(f"noise: {nf:.0%}" if nf > 0 else "noise: clean", fontsize=9)
            if col == 0:
                ax.set_ylabel(f'N={n_steps}', fontsize=9)
            if row == len(data_lengths) - 1:
                ax.set_xlabel('t', fontsize=8)
            ax.tick_params(labelsize=6)

    legend_elements = [Line2D([0], [0], color='black', lw=1, label='ground truth')]
    for method in methods:
        legend_elements.append(Line2D([0], [0], color=COLORS.get(method), lw=1,
                                      label=LABELS.get(method, method)))
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(methods) + 1,
              fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Autonomous Forecast: x(t) from Discovered ODE', fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Forecast grid saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_dir', default=os.path.join(RESULTS_DIR, 'recovery'))
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}. Run recovery_run.py first.")
        return

    noise_fractions = sorted(set(r['noise_frac'] for r in results))
    data_lengths = sorted(set(r['n_steps'] for r in results))
    methods = sorted(set(r['method'] for r in results),
                     key=lambda m: ['factored', 'direct', 'esindy'].index(m)
                     if m in ('factored', 'direct', 'esindy') else 99)

    print(f"Loaded {len(results)} results: "
          f"{len(noise_fractions)} noise levels x {len(data_lengths)} data lengths x "
          f"{len(methods)} methods (partial grids OK)")

    print_summary_tables(results, noise_fractions, data_lengths, methods)
    plot_coef_error(results, noise_fractions, data_lengths, methods,
                    os.path.join(RESULTS_DIR, 'lorenz_recovery_plot.png'))
    plot_f1_heatmap(results, noise_fractions, data_lengths, methods,
                    os.path.join(RESULTS_DIR, 'lorenz_recovery_f1_heatmap.png'))
    plot_forecast_grid(results, noise_fractions, data_lengths, methods,
                       os.path.join(RESULTS_DIR, 'lorenz_recovery_forecasts.png'))


if __name__ == '__main__':
    main()
