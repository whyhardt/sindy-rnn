"""Single grid-cell run of the Lorenz noise/data-size/seed/method parameter
recovery study — the natural unit for a cluster array job.

Runs exactly one (noise_frac, n_steps, seed, method) combination and writes
its own result JSON to results/recovery/. Run recovery_aggregate.py
afterward to combine every finished cell into summary tables + plots.

Example:
    python recovery_run.py --noise_frac 0.05 --n_steps 5000 --seed 0 --method factored
    python recovery_run.py --noise_frac 0.05 --n_steps 5000 --seed 0 --method esindy

Cluster array job (SLURM-style): map --array index to a (noise_frac,
n_steps, seed, method) combination via GRID below, e.g.:
    python recovery_run.py --grid_index $SLURM_ARRAY_TASK_ID
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
np.math = math  # pysindy compat

import torch

from examples._common.estimators import PolynomialRNNEstimator, StlsqEstimator
from data import (
    load_config, generate_lorenz, add_noise, chunk_trajectory, simulate_polynomial_ode,
    compute_forecast_mse, scale_coefficients, TRUE_COEFS, TRUE_ACTIVE, RESULTS_DIR,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters come from config.yaml — the same sindy_rnn:/stlsq: sections
# train_sindy_rnn.py/train_stlsq.py use for the single-run demo — so the
# sweep and the demo never silently drift apart.
_cfg = load_config()
RNN_CONFIG = _cfg['sindy_rnn']
ESINDY_CONFIG = _cfg['stlsq']
# See config.yaml's lorenz.normalize comment / data.py's scale_coefficients().
# run_cell() trains on normalized data but always unscales the discovered
# coef_matrix back to physical units immediately afterward, so every other
# function here (compute_metrics, recovery_aggregate.py's forecast replay)
# stays oblivious to whether normalization happened.
NORMALIZE = _cfg['lorenz'].get('normalize', False)

# Lorenz physical parameters (must match data.py's TRUE_COEFS).
SIGMA, RHO, BETA, DT = _cfg['lorenz']['sigma'], _cfg['lorenz']['rho'], _cfg['lorenz']['beta'], _cfg['lorenz']['dt']
FORECAST_STEPS = _cfg['lorenz']['forecast_steps']

# Grid dimensions — sweep-specific, not part of config.yaml's single-run
# schema. Used both for GRID (cluster array indexing) and for
# recovery_aggregate.py's default axis labels.
NOISE_FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
DATA_LENGTHS = [500, 1000, 2000, 5000, 10000]
NUM_SEEDS = 5
METHODS = ['factored', 'direct', 'esindy']
GRID = list(itertools.product(NOISE_FRACTIONS, DATA_LENGTHS, range(NUM_SEEDS), METHODS))


def get_coef_matrix(model):
    coefs = model.get_coefficients(aggregate=True)
    coef_matrix = np.zeros((3, model.rnn._n_library_terms))
    for i, name in enumerate(model.state_names):
        coef_matrix[i] = coefs[name].cpu().numpy()
    return coef_matrix


def run_sindy_rnn(trajectory_noisy, direct, seed):
    torch.manual_seed(seed)
    xs, ys = chunk_trajectory(trajectory_noisy, RNN_CONFIG['window_size'])

    est = PolynomialRNNEstimator(
        device=DEVICE,
        model_kwargs=dict(
            n_states=3, n_controls=0, polynomial_degree=RNN_CONFIG['degree'],
            ensemble_size=RNN_CONFIG['ensemble_size'], dt=DT,
            state_names=['x', 'y', 'z'],
            compiled_forward=False,
            direct=direct, decomposed=not direct,
        ),
        fit_kwargs=dict(
            epochs=RNN_CONFIG['epochs'], warmup_steps=RNN_CONFIG['warmup_steps'],
            agreement_frac=RNN_CONFIG['agreement_frac'],
            pruning_threshold=RNN_CONFIG['pruning_threshold'],
            pruning_frequency=RNN_CONFIG['pruning_frequency'],
            pruning_method='agreement', learning_rate=RNN_CONFIG['learning_rate'],
            lambda_s=RNN_CONFIG['lambda_s'], refit_epochs=RNN_CONFIG['refit_epochs'],
            verbose=False,
        ),
    )
    est.fit(xs, ys)

    coef_matrix = get_coef_matrix(est.model)
    n_active = sum(est.model.count_active_terms().values())
    return coef_matrix, n_active


def run_esindy(trajectory_noisy):
    est = StlsqEstimator(threshold=ESINDY_CONFIG['threshold'], alpha=ESINDY_CONFIG['alpha'],
                         n_models=ESINDY_CONFIG['n_models'], degree=2, dt=DT,
                         feature_names=['x', 'y', 'z'],
                         simulate=ESINDY_CONFIG.get('simulate', 'mean'))
    est.fit(trajectory_noisy)
    n_active = int(np.count_nonzero(est.coef_matrix))
    return est.coef_matrix, n_active


def compute_metrics(coef_matrix, n_active, forecast_mse, n_valid):
    """Coefficient error, structure recovery (precision/recall/F1), forecast metrics."""
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
        'forecast_mse': float(forecast_mse), 'n_valid': int(n_valid),
        'n_active': int(n_active),
    }


def run_cell(noise_frac, n_steps, seed, method):
    traj_clean = generate_lorenz(n_steps, DT, SIGMA, RHO, BETA, seed=seed * 1000)
    traj_noisy = add_noise(traj_clean, noise_frac, seed=seed * 100 + 1)
    forecast_traj = generate_lorenz(FORECAST_STEPS, DT, SIGMA, RHO, BETA, seed=99999)
    h0_forecast = forecast_traj[0]

    # Per-state std of the observed (noisy) trajectory — same statistic
    # data.py's generate_or_load_data() uses. scale=[1,1,1] (a no-op) when
    # NORMALIZE=False.
    scale = np.std(traj_noisy, axis=0) if NORMALIZE else np.ones(3)
    traj_train = traj_noisy / scale if NORMALIZE else traj_noisy

    t0 = time.time()
    if method == 'factored':
        coef_matrix, n_active = run_sindy_rnn(traj_train, direct=False, seed=seed)
    elif method == 'direct':
        coef_matrix, n_active = run_sindy_rnn(traj_train, direct=True, seed=seed)
    elif method == 'esindy':
        coef_matrix, n_active = run_esindy(traj_train)
    else:
        raise ValueError(f"Unknown method: {method}")
    elapsed = time.time() - t0

    # Rescale back to physical units immediately — everything below
    # (forecast simulation against physical h0_forecast, compute_metrics
    # against physical TRUE_COEFS, and recovery_aggregate.py's later
    # forecast replay from the saved coef_matrix) assumes physical units.
    if NORMALIZE:
        coef_matrix = scale_coefficients(coef_matrix, 1 / scale)

    sim_traj = simulate_polynomial_ode(coef_matrix, h0_forecast, FORECAST_STEPS, DT)
    forecast_mse, n_valid = compute_forecast_mse(forecast_traj, sim_traj)

    metrics = compute_metrics(coef_matrix, n_active, forecast_mse, n_valid)
    metrics.update({
        'method': method, 'noise_frac': noise_frac, 'n_steps': n_steps,
        'seed': seed, 'time': elapsed, 'coef_matrix': coef_matrix.tolist(),
    })
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noise_frac', type=float)
    parser.add_argument('--n_steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--grid_index', type=int,
                        help='Index into the full grid (alternative to specifying '
                             'noise_frac/n_steps/seed/method individually — for '
                             'cluster array jobs, e.g. --grid_index $SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--out_dir', default=os.path.join(RESULTS_DIR, 'recovery'))
    args = parser.parse_args()

    if args.grid_index is not None:
        noise_frac, n_steps, seed, method = GRID[args.grid_index]
    else:
        missing = [n for n in ('noise_frac', 'n_steps', 'seed', 'method')
                  if getattr(args, n) is None]
        if missing:
            parser.error(f"either --grid_index or all of {missing} must be given")
        noise_frac, n_steps, seed, method = args.noise_frac, args.n_steps, args.seed, args.method

    print(f"Lorenz recovery: method={method} noise={noise_frac:.0%} "
          f"N={n_steps} seed={seed}")

    metrics = run_cell(noise_frac, n_steps, seed, method)

    print(f"  coef_err={metrics['coef_error']:.4f} F1={metrics['f1']:.3f} "
          f"exact={metrics['exact_match']} forecast_mse={metrics['forecast_mse']:.2e} "
          f"valid={metrics['n_valid']}/{FORECAST_STEPS} terms={metrics['n_active']} "
          f"time={metrics['time']:.1f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f"{method}_N{n_steps}_noise{noise_frac}_seed{seed}.json")
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved to {out_path}")


if __name__ == '__main__':
    main()
