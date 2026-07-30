"""Benchmark sindy-rnn (derivative matching) vs sindy-rnn-rollout
(trajectory matching) vs STLSQ (E-SINDy) on Lorenz coefficient recovery.

Loads whichever trained estimators exist in params/ (run train_sindy_rnn.py,
train_sindy_rnn_rollout.py, and/or train_stlsq.py first), evaluates
autonomous forecast MSE + coefficient recovery against the known
ground-truth ODE, and writes figures + a metrics summary to results/.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from examples._common.estimators import PolynomialRNNEstimator, RolloutSINDyRNNEstimator, StlsqEstimator
from examples._common.plotting import plot_trajectory_grid
from data import (
    load_config, generate_or_load_data, compute_forecast_mse,
    TRUE_COEFS, TRUE_ACTIVE, PARAMS_DIR, RESULTS_DIR,
)


def coefficient_metrics(coef_matrix):
    true_norm = np.linalg.norm(TRUE_COEFS)
    coef_error = np.linalg.norm(coef_matrix - TRUE_COEFS) / true_norm

    discovered_active = np.abs(coef_matrix) > 1e-6
    tp = np.sum(discovered_active & TRUE_ACTIVE)
    fp = np.sum(discovered_active & ~TRUE_ACTIVE)
    fn = np.sum(~discovered_active & TRUE_ACTIVE)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    exact_match = bool(np.array_equal(discovered_active, TRUE_ACTIVE))

    return {
        'coef_error': float(coef_error), 'precision': float(precision),
        'recall': float(recall), 'f1': float(f1), 'exact_match': exact_match,
    }


def get_coef_matrix(model, member=None):
    coefs = model.get_coefficients(aggregate=True, member=member)
    coef_matrix = np.zeros((3, model.rnn._n_library_terms))
    for i, name in enumerate(model.state_names):
        coef_matrix[i] = coefs[name].cpu().numpy()
    return coef_matrix


def plot_forecasts(clean_test, sims, save_path):
    """Overlay ground truth vs each method's autonomous forecast, per state."""
    state_names = ['x', 'y', 'z']
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    t_true = np.arange(len(clean_test))
    for d in range(3):
        axes[d].plot(t_true, clean_test[:, d], 'k-', linewidth=1, label='Truth', alpha=0.7)
        for name, sim in sims.items():
            t_sim = np.arange(len(sim))
            axes[d].plot(t_sim, sim[:, d], linewidth=1.2, linestyle='--', label=name)
        axes[d].set_ylabel(state_names[d])
        if d == 0:
            axes[d].legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Step')
    fig.suptitle('Lorenz — Autonomous Forecast Comparison')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_coef_comparison(metrics_by_method, save_path):
    methods = list(metrics_by_method.keys())
    coef_err = [metrics_by_method[m]['coef_error'] for m in methods]
    f1 = [metrics_by_method[m]['f1'] for m in methods]

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(1.5 + 2 * len(methods), 4.5))
    ax.bar(x - width / 2, coef_err, width, label='Relative coefficient error')
    ax.bar(x + width / 2, f1, width, label='Structure F1')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_title('Lorenz — Coefficient Recovery')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    cfg = load_config()
    lcfg = cfg['lorenz']

    print("Lorenz — Analysis")
    print("=" * 60)

    data = generate_or_load_data(cfg)
    clean_test = data['clean_test']
    h0 = clean_test[0]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics = {}
    sims = {}

    # ── sindy-rnn ──
    rnn_path = os.path.join(PARAMS_DIR, 'sindy_rnn.pt')
    if os.path.exists(rnn_path):
        print("\nEvaluating sindy-rnn...")
        est = PolynomialRNNEstimator.load(
            rnn_path, simulate=cfg['sindy_rnn'].get('simulate', 'mean'))
        member = est.model.best_member_idx.item() if est.simulate_mode == 'best' else None
        coef_matrix = get_coef_matrix(est.model, member=member)

        sim = est.simulate(h0[None, :], lcfg['forecast_steps'])[0]
        fore_mse, n_valid = compute_forecast_mse(clean_test, sim)

        m = coefficient_metrics(coef_matrix)
        m.update({'forecast_mse': float(fore_mse), 'n_valid': int(n_valid),
                  'n_active': sum(est.model.count_active_terms(member=member).values()),
                  'equations': est.model.get_equations(member=member)})
        metrics['sindy-rnn'] = m
        sims['sindy-rnn'] = sim

        print(f"  Coef. error: {m['coef_error']:.4f}  F1: {m['f1']:.3f}  "
              f"Exact match: {m['exact_match']}")
        print(f"  Forecast MSE: {fore_mse:.6f} ({n_valid} valid steps)")
        print(f"  Active terms: {m['n_active']}")
    else:
        print(f"\nSkipping sindy-rnn: {rnn_path} not found (run train_sindy_rnn.py first)")

    # ── sindy-rnn-rollout (trajectory matching) ──
    rollout_path = os.path.join(PARAMS_DIR, 'sindy_rnn_rollout.pt')
    if os.path.exists(rollout_path):
        print("\nEvaluating sindy-rnn-rollout...")
        est = RolloutSINDyRNNEstimator.load(
            rollout_path, T_w=1,
            simulate=cfg['sindy_rnn_rollout'].get('simulate', 'mean'))
        member = est.model.best_member_idx.item() if est.simulate_mode == 'best' else None
        coef_matrix = get_coef_matrix(est.model.dynamics, member=member)

        sim = est.simulate(h0[None, :], lcfg['forecast_steps'])
        fore_mse, n_valid = compute_forecast_mse(clean_test, sim)

        m = coefficient_metrics(coef_matrix)
        m.update({'forecast_mse': float(fore_mse), 'n_valid': int(n_valid),
                  'n_active': sum(est.model.count_active_terms(member=member).values()),
                  'equations': est.model.get_equations(member=member)})
        metrics['sindy-rnn-rollout'] = m
        sims['sindy-rnn-rollout'] = sim

        print(f"  Coef. error: {m['coef_error']:.4f}  F1: {m['f1']:.3f}  "
              f"Exact match: {m['exact_match']}")
        print(f"  Forecast MSE: {fore_mse:.6f} ({n_valid} valid steps)")
        print(f"  Active terms: {m['n_active']}")
    else:
        print(f"\nSkipping sindy-rnn-rollout: {rollout_path} not found "
              f"(run train_sindy_rnn_rollout.py first)")

    # ── STLSQ ──
    stlsq_path = os.path.join(PARAMS_DIR, 'stlsq.npz')
    if os.path.exists(stlsq_path):
        print("\nEvaluating STLSQ...")
        est = StlsqEstimator.load(stlsq_path)
        coef_matrix = est.coef_matrix

        sim = est.simulate(h0, lcfg['forecast_steps'])
        fore_mse, n_valid = compute_forecast_mse(clean_test, sim)

        m = coefficient_metrics(coef_matrix)
        m.update({'forecast_mse': float(fore_mse), 'n_valid': int(n_valid),
                  'n_active': int(np.count_nonzero(coef_matrix))})
        metrics['stlsq'] = m
        # Cap to forecast_steps for plotting — clean_test may be longer than
        # the current config's forecast_steps if it came from a stale cache
        # (data/lorenz_cache.npz) generated under a different value.
        sims['stlsq'] = sim[:lcfg['forecast_steps']]

        print(f"  Coef. error: {m['coef_error']:.4f}  F1: {m['f1']:.3f}  "
              f"Exact match: {m['exact_match']}")
        print(f"  Forecast MSE: {fore_mse:.6f} ({n_valid} valid steps)")
        print(f"  Active terms: {m['n_active']}")
    else:
        print(f"\nSkipping STLSQ: {stlsq_path} not found (run train_stlsq.py first)")

    # ── Plots ──
    if sims:
        # STLSQ's truth line is capped to forecast_steps too — clean_test
        # may be longer if it came from a stale cache (data/lorenz_cache.npz)
        # generated under a different forecast_steps value.
        rows = {
            name: (clean_test[:lcfg['forecast_steps']] if name == 'stlsq' else clean_test, sim)
            for name, sim in sims.items()
        }
        plot_trajectory_grid(
            rows, os.path.join(RESULTS_DIR, 'trajectory_comparison.png'),
            state_names=['x', 'y', 'z'], title='Lorenz — Truth vs Simulated')
        plot_forecasts(clean_test, sims, os.path.join(RESULTS_DIR, 'forecast_comparison.png'))
    if len(metrics) > 1:
        plot_coef_comparison(metrics, os.path.join(RESULTS_DIR, 'coefficient_comparison.png'))

    results_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {results_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method, m in metrics.items():
        print(f"  {method}: coef_error={m['coef_error']:.4f}  f1={m['f1']:.3f}  "
              f"forecast_mse={m['forecast_mse']:.4f}  active={m['n_active']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
