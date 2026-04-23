"""SST HPO Phase 2: targeted sweep around best Phase 1 configs.

Phase 1 findings:
  - sw1.0 = only stable forecast (23.1%) — strong encoder nudge is essential
  - dlr0.1 = best recon (4.28%) — higher dynamics lr helps
  - warm1k = good recon (4.38%) — shorter warmup helps encoder
  - Most configs diverge on forecast at 5K epochs

Phase 2: combine sw1.0 + dlr0.1, vary warmup + epochs + refit.
"""

import sys
import os
import json
import time
import math

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

# Fixed architecture
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

# Phase 2 configs: all use sw=1.0 (only stable forecast from Phase 1)
CONFIGS = [
    # A: sw1.0 + dlr0.1 (combine two winners), 5K baseline
    {'tag': 'sw1_dlr01', 'epochs': 5000, 'sindy_warmup': 2500,
     'sindy_weight': 1.0, 'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 0.1},

    # B: same but shorter warmup (1K ramp)
    {'tag': 'sw1_dlr01_w1k', 'epochs': 5000, 'sindy_warmup': 1000,
     'sindy_weight': 1.0, 'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 0.1},

    # C: longer training (10K)
    {'tag': 'sw1_10k', 'epochs': 10000, 'sindy_warmup': 5000,
     'sindy_weight': 1.0, 'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},

    # D: longer training + higher dlr
    {'tag': 'sw1_dlr01_10k', 'epochs': 10000, 'sindy_warmup': 5000,
     'sindy_weight': 1.0, 'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 0.1},

    # E: sw1.0, short warmup, longer refit
    {'tag': 'sw1_w1k_r5k', 'epochs': 5000, 'sindy_warmup': 1000,
     'sindy_weight': 1.0, 'l1': 0, 'prune_threshold': 0., 'refit': 5000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},

    # F: sw1.0 + light L1 (keep sparsity pressure during joint training)
    {'tag': 'sw1_l1e-4', 'epochs': 5000, 'sindy_warmup': 2500,
     'sindy_weight': 1.0, 'l1': 1e-4, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 5e-2},

    # G: sw0.5 + dlr0.1 (moderate encoder nudge + fast polynomial)
    {'tag': 'sw05_dlr01', 'epochs': 5000, 'sindy_warmup': 2500,
     'sindy_weight': 0.5, 'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 0.1},

    # H: sw2.0 (even stronger encoder nudge)
    {'tag': 'sw2_dlr01', 'epochs': 5000, 'sindy_warmup': 2500,
     'sindy_weight': 2.0, 'l1': 0, 'prune_threshold': 0., 'refit': 3000,
     'lr': 1e-3, 'dynamics_lr': 0.1},
]


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


def evaluate_recon(model, X_scaled, sensor_locs, scaler, test_frames, X_raw):
    N = X_scaled.shape[0]
    full_dim = X_scaled.shape[1]
    model.eval()
    recons_scaled = np.full((N, full_dim), np.nan)
    with torch.no_grad():
        sparse_all = torch.tensor(
            X_scaled[:, sensor_locs], dtype=torch.float32).to(DEVICE)
        chunk = 64
        for batch_start in range(LAGS - 1, N, chunk):
            batch_end = min(batch_start + chunk, N)
            windows = torch.stack([sparse_all[t - LAGS + 1:t + 1]
                                   for t in range(batch_start, batch_end)])
            encoded = model.encoder(windows)
            z = encoded[:, -1, :]
            decoded = model.decoder(z)
            recons_scaled[batch_start:batch_end] = decoded.cpu().numpy()
    valid_mask = ~np.isnan(recons_scaled[:, 0])
    recons_raw = np.full((N, full_dim), np.nan)
    recons_raw[valid_mask] = scaler.inverse_transform(recons_scaled[valid_mask])
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
    N = X_scaled.shape[0]
    full_dim = X_scaled.shape[1]
    E = model.ensemble_size
    n_forecast = N - train_end
    model.eval()
    forecast_arr = np.full((N, full_dim), np.nan)
    with torch.no_grad():
        sparse_all = torch.tensor(
            X_scaled[:, sensor_locs], dtype=torch.float32).to(DEVICE)
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
        chunk_size = 10
        forecast_scaled = []
        for i in range(0, n_forecast, chunk_size):
            chunk_t = torch.stack(latent_steps[i:i + chunk_size], dim=2)
            decoded = model.decoder(chunk_t)
            forecast_scaled.append(decoded.mean(0)[0].cpu().numpy())
        forecast_scaled = np.concatenate(forecast_scaled, axis=0)
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


def run_config(config, sparse_train, full_target_train, sparse_test, full_target_test,
               X_scaled, X_raw, sensor_locs, scaler, train_end, full_dim, test_frames):
    torch.manual_seed(SEED)
    tag = config['tag']
    sindy_warmup = config['sindy_warmup']
    warmup_steps = sindy_warmup + 2000

    model = SparseAutoencoderRNN(
        sparse_dim=NUM_SENSORS, full_dim=full_dim, latent_dim=LATENT_DIM,
        ensemble_size=11, polynomial_degree=1, dt=DT,
        encoder_type='gru', encoder_gru_hidden_dim=None,
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
        epochs=config['epochs'], warmup_steps=warmup_steps, batch_size=64,
        learning_rate=config['lr'],
        dynamics_learning_rate=config['dynamics_lr'],
        l1=config['l1'],
        pruning_threshold=config['prune_threshold'],
        pruning_method='median', pruning_frequency=100, dt=DT,
        refit_epochs=config['refit'],
        sindy_weight=config['sindy_weight'],
        sindy_warmup_epochs=sindy_warmup,
        centered_diff=True, verbose=True,
    )
    elapsed = time.time() - t0

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
        'recon_mse': float(recon_mse), 'recon_rel': float(recon_rel),
        'forecast_mse': float(fore_mse), 'forecast_rel': float(fore_rel),
        'n_active_terms': n_active, 'equations': equations, 'time': elapsed,
    }

    print(f"\n{'='*60}")
    print(f"RESULT [{tag}]")
    fr = f'{100*fore_rel:.2f}%' if not math.isnan(fore_rel) else 'DIVERGED'
    print(f"  Recon: {100*recon_rel:.2f}%  |  Forecast: {fr}")
    print(f"  Active terms: {n_active}  |  Time: {elapsed:.0f}s")
    if equations and equations != 'N/A':
        for line in equations.strip().split('\n'):
            print(f"  {line}")
    print(f"{'='*60}\n")
    return result


def main():
    print("SST HPO Phase 2")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Configs: {len(CONFIGS)}")

    X_raw, sst_locs = load_sst_data()
    n_time, full_dim = X_raw.shape
    print(f"Data: ({n_time}, {full_dim})")

    train_length = 1000
    train_end = train_length + LAGS
    sc = MinMaxScaler()
    sc.fit(X_raw[:train_end])
    X_scaled = sc.transform(X_raw)
    sensor_locs = get_sensor_locs(full_dim, SEED)

    shred_val_end_frame = (train_length + 30 - 1) + LAGS - 1
    test_start = max(train_end, shred_val_end_frame + 1)
    test_frames = np.arange(test_start, n_time)
    print(f"Test frames: {test_start}-{n_time-1} ({len(test_frames)})")

    sparse_train, full_target_train, sparse_test, full_target_test = \
        prepare_data(X_scaled, sensor_locs, LAGS, train_end)
    print(f"Training windows: {sparse_train.shape[0]}")

    results_path = 'sst_hpo_phase2_results.json'
    all_results = []

    for i, config in enumerate(CONFIGS):
        print(f"\n{'#'*70}")
        print(f"[{i+1}/{len(CONFIGS)}] {config['tag']}: "
              f"sw={config['sindy_weight']}, dlr={config['dynamics_lr']}, "
              f"warm={config['sindy_warmup']}, ep={config['epochs']}, "
              f"refit={config['refit']}, l1={config['l1']}")
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

        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*70}")
    print("PHASE 2 RESULTS")
    print(f"{'='*70}")
    print(f"{'Tag':<20s} {'recon%':>8s} {'fore%':>8s} {'terms':>6s} {'time':>6s}")
    print("-" * 55)
    for r in sorted(all_results, key=lambda x: x.get('recon_rel', 999)):
        f = 100 * r.get('forecast_rel', float('nan'))
        fs = f'{f:7.1f}%' if not math.isnan(f) else '    DIV'
        print(f"{r['tag']:<20s} {100*r.get('recon_rel',float('nan')):7.2f}% {fs} "
              f"{r.get('n_active_terms',0):6d} {r.get('time',0):5.0f}s")

    print(f"\nRef: baseline=4.53%/13.80%, SINDy-SHRED=2.01%")
    print(f"Results in {results_path}")


if __name__ == '__main__':
    main()
