"""SST hyperparameter optimization for sindy-rnn-SHRED.

Sweeps key hyperparameters with reduced epochs to find the best config,
then runs the best config at full training length.

Observations from baseline (10K epochs, sindy_weight=0.1, L1=0):
  - Recon: 4.53%, Forecast: 13.80% — seasonal patterns visible
  - SINDy loss still decreasing at 10K — more epochs or higher weight may help
  - 12 active terms (no pruning) — need L1 > 0 in refit
  - Recon degraded 0.009→0.010 late — sindy pushes encoder

Key axes to sweep:
  1. sindy_weight: how hard E_sindy nudges encoder (0.01-1.0)
  2. l1: L2 regularization during joint training (0, 1e-4, 1e-3)
  3. sindy_warmup: ramp duration (1000-5000)
  4. refit L1 + threshold: sparsification in refit phase
  5. epochs: training budget (5000-10000)
"""

import sys
import os
import json
import time
import math
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
np.math = math

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

from sindy_rnn import SparseAutoencoderRNN, fit_autoencoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# Fixed architecture (same as sst_benchmark.py)
# ============================================================
DATA_PATH = 'data/SST_data.mat'
NUM_SENSORS = 250
LATENT_DIM = 3
DT = 1 / 52
LAGS = 52
GRU_LAYERS = 2
DECODER_L1 = 350
DECODER_L2 = 400
DROPOUT = 0.1
SEED = 0

# ============================================================
# Sweep configs — each is a complete config dict
# ============================================================

# Baseline from user's run (known: 4.53% recon, 13.80% forecast)
BASELINE = {
    'tag': 'baseline',
    'epochs': 10000, 'sindy_warmup': 5000, 'sindy_weight': 0.1,
    'l1': 0, 'prune_threshold': 0., 'refit': 3000,
    'lr': 1e-3, 'dynamics_lr': 5e-2,
}

# Sweep: vary one or two things at a time from baseline
CONFIGS = [
    # --- sindy_weight sweep (higher = stronger encoder nudge) ---
    {'tag': 'sw0.01', 'epochs': 5000, 'sindy_warmup': 2500, 'sindy_weight': 0.01,
     'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},
    {'tag': 'sw0.05', 'epochs': 5000, 'sindy_warmup': 2500, 'sindy_weight': 0.05,
     'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},
    {'tag': 'sw0.1', 'epochs': 5000, 'sindy_warmup': 2500, 'sindy_weight': 0.1,
     'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},
    {'tag': 'sw0.5', 'epochs': 5000, 'sindy_warmup': 2500, 'sindy_weight': 0.5,
     'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},
    {'tag': 'sw1.0', 'epochs': 5000, 'sindy_warmup': 2500, 'sindy_weight': 1.0,
     'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},

    # --- sindy_warmup sweep (ramp duration) ---
    {'tag': 'warm1k', 'epochs': 5000, 'sindy_warmup': 1000, 'sindy_weight': 0.1,
     'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},
    {'tag': 'warm3k', 'epochs': 5000, 'sindy_warmup': 3000, 'sindy_weight': 0.1,
     'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},

    # --- L1 during joint training (helps polynomial sparsity) ---
    {'tag': 'l1_1e-4', 'epochs': 5000, 'sindy_warmup': 2500, 'sindy_weight': 0.1,
     'l1': 1e-4, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},
    {'tag': 'l1_1e-3', 'epochs': 5000, 'sindy_warmup': 2500, 'sindy_weight': 0.1,
     'l1': 1e-3, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},

    # --- Pruning during joint training ---
    {'tag': 'prune0.1', 'epochs': 5000, 'sindy_warmup': 2500, 'sindy_weight': 0.1,
     'l1': 1e-4, 'prune_threshold': 0.1, 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},

    # --- Combined: higher sindy_weight + L1 ---
    {'tag': 'sw0.5_l1', 'epochs': 5000, 'sindy_warmup': 2500, 'sindy_weight': 0.5,
     'l1': 1e-4, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},
    {'tag': 'sw1.0_l1', 'epochs': 5000, 'sindy_warmup': 2500, 'sindy_weight': 1.0,
     'l1': 1e-4, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},

    # --- dynamics_lr sweep ---
    {'tag': 'dlr0.1', 'epochs': 5000, 'sindy_warmup': 2500, 'sindy_weight': 0.1,
     'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 0.1},

    # --- Longer refit ---
    {'tag': 'refit5k', 'epochs': 5000, 'sindy_warmup': 2500, 'sindy_weight': 0.1,
     'l1': 0, 'prune_threshold': 0., 'refit': 5000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},
]


# ============================================================
# Data loading
# ============================================================

def load_sst_data():
    load_X = loadmat(DATA_PATH)['Z'].T
    mean_X = np.mean(load_X, axis=0)
    sst_locs = np.where(mean_X != 0)[0]
    return load_X[:, sst_locs], sst_locs


def get_sensor_locs(full_dim, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(full_dim, size=NUM_SENSORS, replace=False)


def prepare_data(X_scaled, sensor_locs, lags, train_end):
    N = len(X_scaled)
    sparse_all = X_scaled[:, sensor_locs]

    N_train = train_end - lags + 1
    train_starts = range(N_train)
    sparse_train = np.stack([sparse_all[s:s + lags] for s in train_starts])
    full_target_train = X_scaled[[s + lags - 1 for s in train_starts]]

    N_test = N - train_end
    if N_test >= lags:
        test_starts = range(train_end - lags + 1, N - lags + 1)
        sparse_test = np.stack([sparse_all[s:s + lags] for s in test_starts])
        full_target_test = X_scaled[[s + lags - 1 for s in test_starts]]
    else:
        sparse_test, full_target_test = None, None

    def to_dev(x):
        return torch.tensor(x, dtype=torch.float32).to(DEVICE) if x is not None else None
    return to_dev(sparse_train), to_dev(full_target_train), \
           to_dev(sparse_test), to_dev(full_target_test)


# ============================================================
# Evaluation (window-based to match training)
# ============================================================

def evaluate_recon(model, X_scaled, sensor_locs, scaler, test_frames, X_raw):
    """Same-timestep reconstruction using sliding windows (matches training)."""
    N = X_scaled.shape[0]
    full_dim = X_scaled.shape[1]
    model.eval()

    recons_scaled = np.full((N, full_dim), np.nan)
    with torch.no_grad():
        sparse_all = torch.tensor(
            X_scaled[:, sensor_locs], dtype=torch.float32).to(DEVICE)

        # Process as sliding windows to match training
        chunk = 64
        for batch_start in range(LAGS - 1, N, chunk):
            batch_end = min(batch_start + chunk, N)
            windows = []
            for t in range(batch_start, batch_end):
                windows.append(sparse_all[t - LAGS + 1:t + 1])
            windows = torch.stack(windows)  # (chunk, LAGS, sparse_dim)
            encoded = model.encoder(windows)  # (chunk, LAGS, latent_dim)
            z = encoded[:, -1, :]  # last GRU output per window
            decoded = model.decoder(z)  # (chunk, full_dim)
            recons_scaled[batch_start:batch_end] = decoded.cpu().numpy()

    # Inverse transform only valid rows
    valid_mask = ~np.isnan(recons_scaled[:, 0])
    recons_raw = np.full((N, full_dim), np.nan)
    recons_raw[valid_mask] = scaler.inverse_transform(recons_scaled[valid_mask])

    # Evaluate on test frames
    valid = valid_mask[test_frames]
    valid_frames = test_frames[valid]
    if len(valid_frames) == 0:
        return float('nan'), float('nan')
    r = recons_raw[valid_frames]
    g = X_raw[valid_frames]
    mse = np.mean((r - g) ** 2)
    rel_error = np.linalg.norm(r - g) / np.linalg.norm(g)
    return mse, rel_error


def evaluate_forecast(model, X_scaled, sensor_locs, train_end, scaler,
                      test_frames, X_raw):
    """Autonomous forecast from train boundary using sliding window init."""
    N = X_scaled.shape[0]
    full_dim = X_scaled.shape[1]
    E = model.ensemble_size
    n_forecast = N - train_end
    model.eval()

    forecast_arr = np.full((N, full_dim), np.nan)

    with torch.no_grad():
        sparse_all = torch.tensor(
            X_scaled[:, sensor_locs], dtype=torch.float32).to(DEVICE)

        # Encode last training window
        window = sparse_all[train_end - LAGS:train_end].unsqueeze(0)
        encoded = model.encoder(window)
        z_0 = encoded[:, -1:, :].expand(E, -1, -1)

        theta = model.dynamics.rnn.unfold_polynomial_coefficients()
        h = z_0
        latent_steps = []
        for t in range(n_forecast):
            h = model.dynamics.rnn.forward_polynomial(
                h, None, mask=model.dynamics.coefficient_masks, theta=theta,
                integrator='rk4')
            latent_steps.append(h)

        # Decode in chunks
        chunk_size = 10
        forecast_scaled = []
        for i in range(0, n_forecast, chunk_size):
            chunk = torch.stack(latent_steps[i:i + chunk_size], dim=2)
            decoded = model.decoder(chunk)
            forecast_scaled.append(decoded.mean(0)[0].cpu().numpy())

        forecast_scaled = np.concatenate(forecast_scaled, axis=0)

        # Check divergence
        if np.any(np.abs(forecast_scaled) > 1e6) or np.any(np.isnan(forecast_scaled)):
            return float('nan'), float('nan')

        forecast_arr[train_end:train_end + n_forecast] = \
            scaler.inverse_transform(forecast_scaled)

    valid = ~np.isnan(forecast_arr[test_frames, 0])
    valid_frames = test_frames[valid]
    if len(valid_frames) == 0:
        return float('nan'), float('nan')
    r = forecast_arr[valid_frames]
    g = X_raw[valid_frames]
    mse = np.mean((r - g) ** 2)
    rel_error = np.linalg.norm(r - g) / np.linalg.norm(g)
    return mse, rel_error


# ============================================================
# Single run
# ============================================================

def run_config(config, sparse_train, full_target_train, sparse_test, full_target_test,
               X_scaled, X_raw, sensor_locs, scaler, train_end, full_dim, test_frames):
    """Train with a single config, return metrics."""
    torch.manual_seed(SEED)
    tag = config['tag']

    sindy_warmup = config['sindy_warmup']
    warmup_steps = sindy_warmup + 2000  # pruning starts 2000 epochs after sindy ramp begins

    model = SparseAutoencoderRNN(
        sparse_dim=NUM_SENSORS, full_dim=full_dim, latent_dim=LATENT_DIM,
        ensemble_size=11, polynomial_degree=1,
        dt=DT,
        encoder_type='gru',
        encoder_gru_hidden_dim=None,
        encoder_num_layers=GRU_LAYERS,
        decoder_hidden_dims=[DECODER_L1, DECODER_L2],
        encoder_dropout=DROPOUT, decoder_dropout=DROPOUT,
        dynamics_dropout=DROPOUT,
        state_names=[f'z{i+1}' for i in range(LATENT_DIM)],
    ).to(DEVICE)

    t0 = time.time()
    fit_autoencoder(
        model, sparse_train, full_target_train,
        sparse_obs_test=sparse_test, full_state_target_test=full_target_test,
        epochs=config['epochs'],
        warmup_steps=warmup_steps,
        batch_size=64,
        learning_rate=config['lr'],
        dynamics_learning_rate=config['dynamics_lr'],
        l1=config['l1'],
        pruning_threshold=config['prune_threshold'],
        pruning_method='median',
        pruning_frequency=100,
        dt=DT,
        refit_epochs=config['refit'],
        sindy_weight=config['sindy_weight'],
        sindy_warmup_epochs=sindy_warmup,
        centered_diff=True,
        verbose=True,
    )
    elapsed = time.time() - t0

    # Evaluate
    active = model.count_active_terms()
    n_active = sum(active.values())
    try:
        equations = model.get_continuous_equations()
    except Exception:
        equations = "N/A"

    recon_mse, recon_rel = evaluate_recon(
        model, X_scaled, sensor_locs, scaler, test_frames, X_raw)
    fore_mse, fore_rel = evaluate_forecast(
        model, X_scaled, sensor_locs, train_end, scaler, test_frames, X_raw)

    model.cpu()
    torch.cuda.empty_cache()

    result = {
        'tag': tag,
        'config': {k: v for k, v in config.items() if k != 'tag'},
        'recon_mse': float(recon_mse),
        'recon_rel': float(recon_rel),
        'forecast_mse': float(fore_mse),
        'forecast_rel': float(fore_rel),
        'n_active_terms': n_active,
        'equations': equations,
        'time': elapsed,
    }

    print(f"\n{'='*60}")
    print(f"RESULT [{tag}]")
    print(f"  Recon: {100*recon_rel:.2f}%  |  Forecast: {100*fore_rel:.2f}%"
          if not np.isnan(fore_rel) else f"  Recon: {100*recon_rel:.2f}%  |  Forecast: DIVERGED")
    print(f"  Active terms: {n_active}  |  Time: {elapsed:.0f}s")
    if equations != "N/A":
        for line in equations.strip().split('\n'):
            print(f"  {line}")
    print(f"{'='*60}\n")

    return result


# ============================================================
# Main
# ============================================================

def main():
    print("SST Hyperparameter Optimization")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Configs: {len(CONFIGS)}")

    # Load data once
    print("\nLoading SST data...")
    X_raw, sst_locs = load_sst_data()
    n_time, full_dim = X_raw.shape
    print(f"  Shape: ({n_time}, {full_dim})")

    train_length = 1000
    train_end = train_length + LAGS

    sc = MinMaxScaler()
    sc.fit(X_raw[:train_end])
    X_scaled = sc.transform(X_raw)

    sensor_locs = get_sensor_locs(full_dim, SEED)

    # Test frames (skip val region)
    shred_val_end_frame = (train_length + 30 - 1) + LAGS - 1
    test_start = max(train_end, shred_val_end_frame + 1)
    test_frames = np.arange(test_start, n_time)
    print(f"  Test frames: {test_start}-{n_time-1} ({len(test_frames)} frames)")

    # Prepare data once
    sparse_train, full_target_train, sparse_test, full_target_test = \
        prepare_data(X_scaled, sensor_locs, LAGS, train_end)
    print(f"  Training windows: {sparse_train.shape[0]}")

    results_path = 'sst_hpo_results.json'
    all_results = []

    for i, config in enumerate(CONFIGS):
        print(f"\n\n{'#'*70}")
        print(f"Config [{i+1}/{len(CONFIGS)}]: {config['tag']}")
        print(f"  sindy_weight={config['sindy_weight']}, l1={config['l1']}, "
              f"warmup={config['sindy_warmup']}, epochs={config['epochs']}, "
              f"refit={config['refit']}, prune={config['prune_threshold']}, "
              f"dynamics_lr={config['dynamics_lr']}")
        print(f"{'#'*70}\n")

        try:
            result = run_config(
                config, sparse_train, full_target_train,
                sparse_test, full_target_test,
                X_scaled, X_raw, sensor_locs, sc,
                train_end, full_dim, test_frames)
            all_results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results.append({
                'tag': config['tag'],
                'config': {k: v for k, v in config.items() if k != 'tag'},
                'recon_rel': float('nan'), 'forecast_rel': float('nan'),
                'n_active_terms': 0, 'equations': f'FAILED: {e}',
            })

        # Save after each run
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  (saved to {results_path})")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n\n{'='*70}")
    print("HPO RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Tag':<15} {'sindy_w':>8} {'l1':>8} {'warm':>6} {'ep':>6} "
          f"{'refit':>6} | {'recon%':>8} {'fore%':>8} {'terms':>6} {'time':>6}")
    print("-" * 95)

    # Sort by recon error
    sorted_results = sorted(all_results,
                           key=lambda r: r.get('recon_rel', 999))

    for r in sorted_results:
        c = r.get('config', {})
        recon_pct = 100 * r.get('recon_rel', float('nan'))
        fore_pct = 100 * r.get('forecast_rel', float('nan'))
        terms = r.get('n_active_terms', 0)
        t = r.get('time', 0)
        print(f"{r['tag']:<15} {c.get('sindy_weight', '?'):>8} {c.get('l1', '?'):>8} "
              f"{c.get('sindy_warmup', '?'):>6} {c.get('epochs', '?'):>6} "
              f"{c.get('refit', '?'):>6} | "
              f"{recon_pct:>7.2f}% {fore_pct:>7.2f}% {terms:>6} {t:>5.0f}s")

    # Best config
    valid = [r for r in all_results if not np.isnan(r.get('forecast_rel', float('nan')))]
    if valid:
        def score(r):
            return r['recon_rel'] + 0.5 * r['forecast_rel']
        best = min(valid, key=score)
        print(f"\nBest (recon + 0.5*forecast): {best['tag']}")
        print(f"  Recon: {100*best['recon_rel']:.2f}%, Forecast: {100*best['forecast_rel']:.2f}%")
        if best.get('equations', 'N/A') != 'N/A':
            print(f"  Equations:\n{best['equations']}")

    print(f"\nReference: SINDy-SHRED = 2.01% recon")
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
