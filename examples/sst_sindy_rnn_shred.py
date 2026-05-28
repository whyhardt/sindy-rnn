"""SST benchmark: sindy-rnn-shred (ODE encoder) vs SINDy-SHRED vs SHRED.

Uses SparseAutoencoderRNN with encoder_type='ode':
  Encoder: MLP(sensors[t]) -> u[t], z[t+1] = z[t] + dt * P(z[t], u[t])
  Autonomous dynamics: dz/dt = P_auto(z) for derivative matching + discovery
  Decoder: z -> full_state

Training via fit_autoencoder():
  E_id:    decode(z_final) ≈ full_state        # reconstruction
  E_sindy: P_auto(z) ≈ dz/dt from encoder z    # derivative matching
  Refit:   freeze encoder, fit clean equations

Data: NOAA Optimum Interpolation SST V2 (1992-2019)
  - 1,400 weekly snapshots, ~44,000 sea grid points
  - 250 random sensors (0.57% spatial coverage)
"""

import sys
import os
import io
import json
import contextlib
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'sindy-shred'))

import numpy as np
np.math = math

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

from sindy_rnn import SparseAutoencoderRNN, fit_autoencoder
from sindy_shred import SINDySHRED

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# Configuration
# ============================================================

NUM_SEEDS = 1
DATA_PATH = 'data/SST_data.mat'

# Shared architecture
NUM_SENSORS = 250
LATENT_DIM = 3
POLY_ORDER = 3
DT = 1 / 52             # weekly
LAGS = 52               # 1 year of sensor history
DECODER_L1 = 350
DECODER_L2 = 400
DROPOUT = 0.1

# SINDy-SHRED
SHRED_EPOCHS = 1000
SHRED_BATCH_SIZE = 128
SHRED_LR = 1e-3
SHRED_THRESHOLD = 1.0
SHRED_PATIENCE = 5
SHRED_SINDY_REG = 10.0
SHRED_THRES_EPOCH = 100
SHRED_GRU_LAYERS = 2

# sindy-rnn-shred (ODE encoder via SparseAutoencoderRNN)
RNN_SHRED_ENSEMBLE = 11
RNN_SHRED_EPOCHS = 10000
RNN_SHRED_LR = 1e-3
RNN_SHRED_DYNAMICS_LR = 5e-2
RNN_SHRED_BATCH_SIZE = 64
RNN_SHRED_ENCODER_HIDDEN = [128, 64]

# Derivative matching (E_sindy) + L1 + pruning
RNN_SHRED_SINDY_WEIGHT = 1.0      # weight for derivative matching loss
RNN_SHRED_SINDY_WARMUP = 0      # epochs before E_sindy + L1 activate
RNN_SHRED_L1 = 1e-3               # L1 on autonomous dynamics coefficients
RNN_SHRED_PRUNE_THRESHOLD = 0.1   # pruning threshold
RNN_SHRED_PRUNE_FREQ = 1000        # pruning frequency

# Refit (freeze encoder, fit clean equations)
RNN_SHRED_REFIT_EPOCHS = 10000

METHODS = ['sindy-rnn-shred']#, 'sindy-shred', 'shred']


# ============================================================
# Data loading
# ============================================================

def load_sst_data():
    """Load SST data, filter to sea grid points."""
    load_X = loadmat(DATA_PATH)['Z'].T  # (1400, 64800)
    mean_X = np.mean(load_X, axis=0)
    sst_locs = np.where(mean_X != 0)[0]
    return load_X[:, sst_locs], sst_locs


def get_sensor_locs(full_dim, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(full_dim, size=NUM_SENSORS, replace=False)


def prepare_sliding_windows(X_scaled, sensor_locs, lags, train_end):
    """Create sliding-window data (stride=1).

    Returns:
        sparse_train: (N_train, LAGS, sparse_dim)
        full_target_train: (N_train, full_dim)
        sparse_test: (N_test, LAGS, sparse_dim) or None
        full_target_test: (N_test, full_dim) or None
    """
    N = len(X_scaled)
    sparse_all = X_scaled[:, sensor_locs]

    # Training windows
    N_train = train_end - lags + 1
    train_starts = range(N_train)
    sparse_train = np.stack([sparse_all[s:s + lags] for s in train_starts])
    full_target_train = X_scaled[[s + lags - 1 for s in train_starts]]

    # Test windows
    N_test = N - train_end
    if N_test >= lags:
        test_starts = range(train_end - lags + 1, N - lags + 1)
        sparse_test = np.stack([sparse_all[s:s + lags] for s in test_starts])
        full_target_test = X_scaled[[s + lags - 1 for s in test_starts]]
        N_test = len(test_starts)
    else:
        sparse_test, full_target_test = None, None
        N_test = 0

    print(f"  Training windows: {N_train} (lags={lags}, stride=1)")
    if N_test > 0:
        print(f"  Test windows: {N_test}")

    def to_dev(x):
        return torch.tensor(x, dtype=torch.float32).to(DEVICE) if x is not None else None
    return to_dev(sparse_train), to_dev(full_target_train), \
           to_dev(sparse_test), to_dev(full_target_test)


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_reconstructions(recons, X_raw, test_frames):
    """Compute MSE and relative error on test frames (raw space)."""
    valid = ~np.isnan(recons[test_frames, 0])
    valid_frames = test_frames[valid]

    if len(valid_frames) == 0:
        return float('nan'), float('nan'), 0

    r = recons[valid_frames]
    g = X_raw[valid_frames]
    mse = np.mean((r - g) ** 2)
    rel_error = np.linalg.norm(r - g) / np.linalg.norm(g)
    return mse, rel_error, len(valid_frames)


def compute_forecast_mse_per_step(forecast, X_raw, train_end):
    """Per-timestep MSE for autonomous forecast."""
    n_frames = X_raw.shape[0]
    n_forecast = n_frames - train_end
    mse_per_step = np.full(n_forecast, np.nan)

    for k in range(n_forecast):
        frame = train_end + k
        if not np.isnan(forecast[frame, 0]):
            mse_per_step[k] = np.mean((forecast[frame] - X_raw[frame]) ** 2)

    return mse_per_step


# ============================================================
# Reconstruction: same-timestep encode-decode
# ============================================================

def reconstruct_sindy_shred(shred_obj, n_frames, full_dim):
    """SINDy-SHRED/SHRED reconstruction."""
    shred_obj._shred.eval()
    lags = shred_obj._lags
    recons = np.full((n_frames, full_dim), np.nan)

    for data_type, indices in [('train', shred_obj._train_ind),
                                ('validate', shred_obj._val_ind),
                                ('test', shred_obj._test_ind)]:
        if indices is None or len(indices) == 0:
            continue
        recon = shred_obj.sensor_recon(data_type=data_type, return_scaled=False)
        for i, idx in enumerate(indices):
            frame = idx + lags - 1
            if frame < n_frames:
                recons[frame] = recon[i]

    return recons


def reconstruct_sindy_rnn_shred(model, X_scaled, sensor_locs, scaler):
    """sindy-rnn-shred reconstruction: process sliding windows, decode z[-1].

    Each frame t (for t >= LAGS-1) is reconstructed from the window
    [t-LAGS+1, ..., t] by running the MLP+SINDy-RNN forward and decoding.

    Returns (n_frames, full_dim) in raw space.
    """
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
            full_pred, z_final, _ = model(windows)  # (E, B, full_dim)
            recons_scaled[batch_start:batch_end] = full_pred.mean(0).cpu().numpy()

    valid = ~np.isnan(recons_scaled[:, 0])
    recons = np.full((N, full_dim), np.nan)
    recons[valid] = scaler.inverse_transform(recons_scaled[valid])
    return recons


# ============================================================
# Forecast: autonomous rollout
# ============================================================

def forecast_sindy_shred(shred_obj, n_frames, full_dim, train_end):
    """Autonomous forecast using SINDy-SHRED's post-hoc SINDy model."""
    n_forecast = n_frames - train_end
    forecast_arr = np.full((n_frames, full_dim), np.nan)

    if shred_obj._model is None:
        print("  Forecast skipped: no post-hoc SINDy model")
        return forecast_arr

    try:
        forecast_raw = shred_obj.forecast(
            n_steps=n_forecast, init_from="train", return_scaled=False)
        n_actual = min(len(forecast_raw), n_forecast)
        if np.any(np.abs(forecast_raw[:n_actual]) > 1e6) or np.any(np.isnan(forecast_raw[:n_actual])):
            print("  Forecast diverged")
            return forecast_arr
        forecast_arr[train_end:train_end + n_actual] = forecast_raw[:n_actual]
    except Exception as e:
        print(f"  SINDy-SHRED forecast failed: {e}")

    return forecast_arr


def forecast_sindy_rnn_shred(model, X_scaled, sensor_locs, train_end, scaler):
    """Autonomous forecast using discovered polynomial dynamics.

    Gets z_0 by encoding the last training window, then evolves forward
    using autonomous P(z) from the fitted dynamics.
    """
    N = X_scaled.shape[0]
    full_dim = X_scaled.shape[1]
    E = model.ensemble_size
    n_forecast = N - train_end
    model.eval()

    forecast_arr = np.full((N, full_dim), np.nan)

    with torch.no_grad():
        sparse_all = torch.tensor(
            X_scaled[:, sensor_locs], dtype=torch.float32).to(DEVICE)

        # Encode the window ending at train_end-1 (last training frame)
        window = sparse_all[train_end - LAGS:train_end].unsqueeze(0)  # (1, LAGS, sparse_dim)
        encoded = model.encoder(window)  # (1, LAGS, latent_dim)
        z_0 = encoded[:, -1, :]  # (1, latent_dim) — last timestep
        z_0 = z_0.unsqueeze(0).expand(E, -1, -1)  # (E, 1, latent_dim)

        # Autonomous rollout using fitted dynamics
        full_traj, latent_traj = model.forecast(z_0, n_steps=n_forecast)

        # Decode
        forecast_scaled = full_traj.mean(0)[0].cpu().numpy()  # (n_forecast, full_dim)

        # Check for divergence
        if np.any(np.abs(forecast_scaled) > 1e6) or np.any(np.isnan(forecast_scaled)):
            print("  Forecast diverged")
            return forecast_arr

        forecast_arr[train_end:train_end + n_forecast] = \
            scaler.inverse_transform(forecast_scaled)

    return forecast_arr


# ============================================================
# Latent trajectory extraction
# ============================================================

def extract_latent_sindy_rnn_shred(model, X_scaled, sensor_locs, train_end):
    """Extract encoder latent trajectory and autonomous rollout."""
    N = X_scaled.shape[0]
    E = model.ensemble_size
    model.eval()

    with torch.no_grad():
        sparse_all = torch.tensor(
            X_scaled[:, sensor_locs], dtype=torch.float32).to(DEVICE)

        # Extract z from all windows (stride=1)
        z_list = []
        chunk = 64
        for batch_start in range(LAGS - 1, N, chunk):
            batch_end = min(batch_start + chunk, N)
            windows = torch.stack([sparse_all[t - LAGS + 1:t + 1]
                                   for t in range(batch_start, batch_end)])
            encoded = model.encoder(windows)  # (B, LAGS, latent_dim)
            z_list.append(encoded[:, -1, :].cpu().numpy())

        z_encoder = np.full((N, LATENT_DIM), np.nan)
        z_cat = np.concatenate(z_list, axis=0)
        z_encoder[LAGS - 1:] = z_cat

        # Autonomous rollout from train boundary
        z_0_val = z_encoder[train_end - 1]
        z_0 = torch.tensor(z_0_val, dtype=torch.float32).to(DEVICE)
        z_0 = z_0.unsqueeze(0).unsqueeze(0).expand(E, -1, -1)  # (E, 1, latent_dim)

        dynamics = model.dynamics
        theta = dynamics.rnn.unfold_polynomial_coefficients()
        h = z_0
        rollout = [z_0_val]
        n_forecast = N - train_end
        for t in range(n_forecast):
            h = dynamics.rnn.forward_polynomial(
                h, None, mask=dynamics.coefficient_masks, theta=theta,
                integrator='rk4')
            rollout.append(h.mean(0)[0].cpu().numpy())
        z_rollout = np.array(rollout)

    return {
        'z_encoder': z_encoder,
        'z_rollout': z_rollout,
        'rollout_start_frame': train_end - 1,
    }


def extract_latent_sindy_shred(shred_obj, has_sindy=True):
    """Extract encoder and SINDy-integrated latent trajectories."""
    lags = shred_obj._lags
    n_frames = shred_obj._n_time_dim
    latent_dim = shred_obj._latent_dim

    z_encoder = np.full((n_frames, latent_dim), np.nan)
    for data_type, indices in [('train', shred_obj._train_ind),
                                ('validate', shred_obj._val_ind),
                                ('test', shred_obj._test_ind)]:
        if indices is None or len(indices) == 0:
            continue
        z = shred_obj.gru_normalize(data_type=data_type).detach().cpu().numpy()
        for i in range(len(z)):
            idx = indices[i + 1]
            frame = idx + lags - 1
            if frame < n_frames:
                z_encoder[frame] = z[i]

    z_rollout = None
    rollout_start_frame = None
    if has_sindy and shred_obj._model is not None:
        try:
            rollout_start_frame = shred_obj._train_ind[-1] + lags - 1
            n_rollout = n_frames - rollout_start_frame
            t = np.arange(n_rollout) * shred_obj._dt
            z_rollout = shred_obj.sindy_predict(t=t, init_from='train')
            if np.any(np.abs(z_rollout) > 1e6) or np.any(np.isnan(z_rollout)):
                print("  SINDy latent rollout diverged")
                z_rollout = None
                rollout_start_frame = None
        except Exception as e:
            print(f"  SINDy latent rollout failed: {e}")

    return {
        'z_encoder': z_encoder,
        'z_rollout': z_rollout,
        'rollout_start_frame': rollout_start_frame,
    }


# ============================================================
# Plotting
# ============================================================

SST_GRID_ROWS = 180
SST_GRID_COLS = 360


def plot_latent_dynamics(latent_dict, train_end, latent_dim, seed, save_dir, prefix):
    """Plot latent variable dynamics for each method."""
    for method, data in latent_dict.items():
        fig, axes = plt.subplots(latent_dim, 1, figsize=(12, 3 * latent_dim),
                                  sharex=True)
        if latent_dim == 1:
            axes = [axes]

        z_enc = data['z_encoder']
        z_roll = data.get('z_rollout')
        roll_start = data.get('rollout_start_frame')

        for d in range(latent_dim):
            ax = axes[d]
            valid = ~np.isnan(z_enc[:, d])
            frames = np.arange(len(z_enc))
            ax.plot(frames[valid], z_enc[valid, d], 'b-', alpha=0.7,
                    linewidth=1, label='Encoder')
            if z_roll is not None and roll_start is not None:
                roll_frames = np.arange(roll_start, roll_start + len(z_roll))
                ax.plot(roll_frames, z_roll[:, d], 'r--', alpha=0.7,
                        linewidth=1.5, label='Autonomous rollout')
            ax.axvline(x=train_end, color='k', linestyle=':', alpha=0.5,
                       label='Train/Test' if d == 0 else None)
            ax.set_ylabel(f'z{d+1}')
            if d == 0:
                ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel('Frame')
        fig.suptitle(f'{method} — Latent Dynamics ({prefix}, seed {seed})', fontsize=12)
        fig.tight_layout()

        path = os.path.join(save_dir, f'{prefix}_latent_{method}_seed{seed}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved latent dynamics plot: {path}")


def generate_sst_images(X_raw, sst_locs, data_dict, train_end, seed, save_dir,
                        prefix='recon'):
    """Generate composite images for all methods."""
    n_frames = X_raw.shape[0]
    X_mean = np.nanmean(X_raw, axis=0)

    def to_grid(vec):
        grid = np.full(SST_GRID_ROWS * SST_GRID_COLS, np.nan)
        grid[sst_locs] = vec
        return grid.reshape(SST_GRID_ROWS, SST_GRID_COLS)

    for method, data in data_dict.items():
        if prefix == 'forecast':
            start = train_end
            end = n_frames - 1
        else:
            valid_mask = ~np.isnan(data[:, 0])
            if not valid_mask.any():
                continue
            start = int(np.argmax(valid_mask))
            end = n_frames - 1 - int(np.argmax(valid_mask[::-1]))

        frame_indices = np.linspace(start, end,
                                    min(8, end - start + 1), dtype=int)
        n_cols = len(frame_indices)
        fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 7))

        gt_anom = X_raw[frame_indices] - X_mean[np.newaxis, :]
        pred_anom = data[frame_indices] - X_mean[np.newaxis, :]
        vlim = np.nanpercentile(np.abs(gt_anom), 98)
        vmin_val, vmax_val = -vlim, vlim

        row_labels = ['True anomaly',
                      'Recon. anomaly' if prefix == 'recon' else 'Forecast anomaly',
                      '|Error|']

        err_vals = np.abs(gt_anom - pred_anom)
        valid_err = err_vals[~np.isnan(err_vals)]
        err_max = np.nanpercentile(valid_err, 95) if len(valid_err) > 0 else 1.0

        for j, fidx in enumerate(frame_indices):
            region = "test" if fidx >= train_end else "train"
            has_pred = not np.isnan(data[fidx, 0])

            gt_grid = to_grid(gt_anom[j])
            axes[0, j].imshow(gt_grid, cmap='RdBu_r', vmin=vmin_val, vmax=vmax_val,
                              aspect='auto', origin='upper')
            axes[0, j].set_title(f"t={fidx} ({region})", fontsize=8)

            if has_pred:
                rec_grid = to_grid(pred_anom[j])
                err_grid = to_grid(np.abs(gt_anom[j] - pred_anom[j]))
                axes[1, j].imshow(rec_grid, cmap='RdBu_r', vmin=vmin_val, vmax=vmax_val,
                                  aspect='auto', origin='upper')
                axes[2, j].imshow(err_grid, cmap='hot', vmin=0, vmax=err_max,
                                  aspect='auto', origin='upper')
            else:
                for row in [1, 2]:
                    axes[row, j].text(0.5, 0.5, 'N/A', transform=axes[row, j].transAxes,
                                      ha='center', va='center', fontsize=12, color='gray')

            for row in range(3):
                axes[row, j].set_xticks([])
                axes[row, j].set_yticks([])

        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=10)

        fig.suptitle(f'{method} — SST {prefix} anomaly (seed {seed})', fontsize=12)
        fig.tight_layout()

        path = os.path.join(save_dir, f'sst_{prefix}_{method}_seed{seed}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved image: {path}")


def plot_forecast_mse_over_time(mse_dict, seed, save_dir, prefix):
    """Plot per-timestep forecast MSE."""
    fig, ax = plt.subplots(figsize=(8, 4))
    has_data = False

    for method, mse_arr in mse_dict.items():
        valid = ~np.isnan(mse_arr)
        if not valid.any():
            continue
        steps = np.arange(len(mse_arr))
        ax.plot(steps[valid], mse_arr[valid], label=method, linewidth=1.5)
        has_data = True

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel('Forecast step')
    ax.set_ylabel('MSE')
    ax.set_title(f'Forecast MSE vs horizon (seed {seed})')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()

    path = os.path.join(save_dir, f'{prefix}_forecast_mse_seed{seed}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved forecast MSE plot: {path}")


# ============================================================
# Method runners
# ============================================================

def run_sindy_rnn_shred(X, sensor_locs, train_end, full_dim, seed, save_dir=None):
    """Train sindy-rnn-shred (ODE encoder) and return reconstruction + forecast."""
    torch.manual_seed(seed)

    sc = MinMaxScaler()
    sc.fit(X[:train_end])
    X_scaled = sc.transform(X)

    sparse_train, full_target_train, sparse_test, full_target_test = \
        prepare_sliding_windows(X_scaled, sensor_locs, LAGS, train_end)

    model = SparseAutoencoderRNN(
        sparse_dim=NUM_SENSORS,
        full_dim=full_dim,
        latent_dim=LATENT_DIM,
        ensemble_size=RNN_SHRED_ENSEMBLE,
        polynomial_degree=POLY_ORDER,
        dt=DT,
        encoder_type='ode',
        encoder_hidden_dims=RNN_SHRED_ENCODER_HIDDEN,
        decoder_hidden_dims=[DECODER_L1, DECODER_L2],
        encoder_dropout=DROPOUT,
        decoder_dropout=DROPOUT,
        state_names=[f'z{i+1}' for i in range(LATENT_DIM)],
        decomposed=True,
    ).to(DEVICE)

    fit_autoencoder(
        model, sparse_train, full_target_train,
        sparse_obs_test=sparse_test, full_state_target_test=full_target_test,
        epochs=RNN_SHRED_EPOCHS,
        batch_size=RNN_SHRED_BATCH_SIZE,
        learning_rate=RNN_SHRED_LR,
        dynamics_learning_rate=RNN_SHRED_DYNAMICS_LR,
        sindy_weight=RNN_SHRED_SINDY_WEIGHT,
        sindy_warmup_epochs=RNN_SHRED_SINDY_WARMUP,
        l1=RNN_SHRED_L1,
        pruning_threshold=RNN_SHRED_PRUNE_THRESHOLD,
        pruning_frequency=RNN_SHRED_PRUNE_FREQ,
        pruning_method='median',
        refit_epochs=RNN_SHRED_REFIT_EPOCHS,
        dt=DT,
        centered_diff=True,
        verbose=True,
    )

    active = model.count_active_terms()
    n_active = sum(active.values())
    n_params = sum(p.numel() for p in model.parameters())

    try:
        equations = model.get_continuous_equations()
    except Exception:
        equations = "N/A"

    # Metric 1: Reconstruction
    recons = reconstruct_sindy_rnn_shred(model, X_scaled, sensor_locs, sc)

    # Metric 2: Forecast (autonomous rollout)
    forecast = forecast_sindy_rnn_shred(model, X_scaled, sensor_locs, train_end, sc)

    # Extract latent trajectories
    latent = extract_latent_sindy_rnn_shred(model, X_scaled, sensor_locs, train_end)

    model.cpu()
    torch.cuda.empty_cache()

    return recons, forecast, {
        'n_params': n_params,
        'n_active_terms': n_active,
        'equations': equations,
    }, latent


def run_sindy_shred(X, sensor_locs, train_length, validate_length,
                    n_frames, full_dim, seed, sindy_reg=SHRED_SINDY_REG,
                    save_dir=None):
    """Train SINDy-SHRED (or plain SHRED) and return reconstruction + forecast."""
    shred = SINDySHRED(
        latent_dim=LATENT_DIM, poly_order=POLY_ORDER,
        hidden_layers=SHRED_GRU_LAYERS, l1=DECODER_L1, l2=DECODER_L2,
        dropout=DROPOUT, batch_size=SHRED_BATCH_SIZE,
        num_epochs=SHRED_EPOCHS, lr=SHRED_LR,
        threshold=SHRED_THRESHOLD, patience=SHRED_PATIENCE,
        sindy_regularization=sindy_reg, thres_epoch=SHRED_THRES_EPOCH,
        verbose=True, device=DEVICE,
    )

    shred.fit(
        num_sensors=NUM_SENSORS, dt=DT, x_to_fit=X, lags=LAGS,
        train_length=train_length, validate_length=validate_length,
        sensor_locations=sensor_locs, seed=seed,
    )

    n_params = sum(p.numel() for p in shred._shred.parameters())
    train_end = train_length + LAGS

    n_active = 0
    equations = "N/A"
    if sindy_reg > 0:
        try:
            best_thresh, tune_results = shred.auto_tune_threshold(
                metric='bic', verbose=True)
            sindy_model = shred._model
            n_active = int(np.sum(np.abs(sindy_model.coefficients()) > 1e-6))
            lhs = [f"z{i}'" for i in range(LATENT_DIM)]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sindy_model.print(lhs=lhs)
            equations = buf.getvalue().strip()
            print(f"  Post-hoc SINDy (threshold={best_thresh:.4f}, {n_active} terms):")
            print(f"  {equations}")
        except Exception as e:
            print(f"  Post-hoc SINDy failed: {e}")
            import traceback; traceback.print_exc()
            equations = f"Failed: {e}"

    recons = reconstruct_sindy_shred(shred, n_frames, full_dim)

    forecast = np.full((n_frames, full_dim), np.nan)
    if sindy_reg > 0:
        forecast = forecast_sindy_shred(shred, n_frames, full_dim, train_end)

    latent = extract_latent_sindy_shred(shred, has_sindy=(sindy_reg > 0))

    if save_dir is not None:
        method_tag = 'shred' if sindy_reg == 0.0 else 'sindy_shred'
        path = os.path.join(save_dir, f'sst_{method_tag}_seed{seed}.pt')
        torch.save(shred._shred.state_dict(), path)
        print(f"  Saved model to {path}")

    shred._shred.cpu()
    torch.cuda.empty_cache()

    return recons, forecast, {
        'n_params': n_params,
        'n_active_terms': n_active,
        'equations': equations,
    }, latent


# ============================================================
# Main benchmark
# ============================================================

def main():
    print("SST Benchmark: sindy-rnn-shred vs SINDy-SHRED vs SHRED")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {NUM_SEEDS}")
    print(f"Methods: {METHODS}")
    print(f"Architecture:")
    print(f"  sindy-rnn-shred: ODEEncoder(MLP({NUM_SENSORS}->{RNN_SHRED_ENCODER_HIDDEN}->{LATENT_DIM})"
          f"+PolyODE(deg={POLY_ORDER})) + AutonomousDyn(E={RNN_SHRED_ENSEMBLE}) "
          f"+ Decoder({LATENT_DIM}->{DECODER_L1}->{DECODER_L2}->full)")
    print(f"  sindy-shred/shred: GRU({NUM_SENSORS}->{LATENT_DIM}, {SHRED_GRU_LAYERS}L) "
          f"+ Decoder({LATENT_DIM}->{DECODER_L1}->{DECODER_L2}->full)")
    print(f"poly_order={POLY_ORDER}, dt={DT:.4f}, lags={LAGS}")

    # Load data
    print("\nLoading SST data...")
    X, sst_locs = load_sst_data()
    n_time, full_dim = X.shape
    print(f"  Shape: ({n_time}, {full_dim})")

    # Train/val split
    train_length = 1000
    validate_length = 30
    train_end = train_length + LAGS

    shred_val_end_frame = (train_length + validate_length - 1) + LAGS - 1
    test_start = max(train_end, shred_val_end_frame + 1)
    test_frames = np.arange(test_start, n_time)

    print(f"  Train: {train_length} samples, Val: {validate_length} samples")
    print(f"  Test frames: {test_start}-{n_time-1} ({len(test_frames)} frames)")

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'params')
    img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results')
    os.makedirs(save_dir, exist_ok=True)

    all_results = []

    for seed in range(NUM_SEEDS):
        sensor_locs = get_sensor_locs(full_dim, seed)
        seed_recons = {}
        seed_forecasts = {}
        seed_latent = {}

        for method in METHODS:
            tag = f"Seed {seed}, {method}"
            print(f"\n{'='*70}")
            print(f"{tag}")
            print(f"{'='*70}")

            t0 = time.time()
            try:
                if method == 'sindy-rnn-shred':
                    recons, forecast, info, latent = run_sindy_rnn_shred(
                        X, sensor_locs, train_end, full_dim, seed,
                        save_dir=save_dir)
                elif method == 'sindy-shred':
                    recons, forecast, info, latent = run_sindy_shred(
                        X, sensor_locs, train_length, validate_length,
                        n_time, full_dim, seed, sindy_reg=SHRED_SINDY_REG,
                        save_dir=save_dir)
                elif method == 'shred':
                    recons, forecast, info, latent = run_sindy_shred(
                        X, sensor_locs, train_length, validate_length,
                        n_time, full_dim, seed, sindy_reg=0.0,
                        save_dir=save_dir)

                elapsed = time.time() - t0
                seed_recons[method] = recons
                seed_forecasts[method] = forecast
                seed_latent[method] = latent

                recon_mse, recon_rel, n_recon = evaluate_reconstructions(
                    recons, X, test_frames)
                fore_mse, fore_rel, n_fore = evaluate_reconstructions(
                    forecast, X, test_frames)

                fore_mse_per_step = None
                if not np.isnan(fore_mse):
                    fore_mse_per_step = compute_forecast_mse_per_step(
                        forecast, X, train_end)

                result = {
                    'method': method,
                    'seed': seed,
                    'recon_mse': recon_mse,
                    'recon_rel_error': recon_rel,
                    'n_recon_frames': n_recon,
                    'forecast_mse': fore_mse,
                    'forecast_rel_error': fore_rel,
                    'n_forecast_frames': n_fore,
                    'forecast_mse_per_step': fore_mse_per_step.tolist() if fore_mse_per_step is not None else None,
                    'n_active_terms': info.get('n_active_terms', 0),
                    'n_params': info.get('n_params', 0),
                    'time': elapsed,
                    'equations': info.get('equations', 'N/A'),
                }
                all_results.append(result)

                print(f"  Reconstruction MSE: {recon_mse:.6f} ({n_recon} frames), "
                      f"rel.err: {100*recon_rel:.2f}%")
                if not np.isnan(fore_mse):
                    print(f"  Forecast MSE: {fore_mse:.6f} ({n_fore} frames), "
                          f"rel.err: {100*fore_rel:.2f}%")
                else:
                    print(f"  Forecast: N/A (no dynamics)")
                print(f"  Active terms: {info.get('n_active_terms', 0)}")
                print(f"  Time: {elapsed:.1f}s")

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    'method': method, 'seed': seed,
                    'recon_mse': float('nan'), 'recon_rel_error': float('nan'),
                    'n_recon_frames': 0,
                    'forecast_mse': float('nan'), 'forecast_rel_error': float('nan'),
                    'n_forecast_frames': 0,
                    'n_params': 0, 'n_active_terms': 0, 'time': 0.0,
                    'equations': 'FAILED',
                })

        # Generate images
        if seed_recons:
            print(f"\nGenerating images for seed {seed}...")
            generate_sst_images(X, sst_locs, seed_recons, train_end, seed, img_dir,
                                prefix='recon')
            generate_sst_images(X, sst_locs, seed_forecasts, train_end, seed, img_dir,
                                prefix='forecast')

            seed_fore_mse = {}
            for m in seed_forecasts:
                r = [x for x in all_results if x['method'] == m and x['seed'] == seed]
                if r and r[0].get('forecast_mse_per_step') is not None:
                    seed_fore_mse[m] = np.array(r[0]['forecast_mse_per_step'])
            if seed_fore_mse:
                plot_forecast_mse_over_time(seed_fore_mse, seed, img_dir, 'sst_rnn_shred')

            if seed_latent:
                plot_latent_dynamics(seed_latent, train_end, LATENT_DIM, seed,
                                    img_dir, 'sst_rnn_shred')

    # Save results
    results_path = 'sst_sindy_rnn_shred_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ================================================================
    # Summary
    # ================================================================
    def print_metric_table(metric_key, rel_key, title):
        print(f"\n{title}")
        print(f"{'Method':<25} {'MSE (mean±std)':<22} {'Rel.Err %':<18} {'Active':<12} {'Time (s)':<10}")
        print("-" * 87)
        for method in METHODS:
            mses = [r[metric_key] for r in all_results
                    if r['method'] == method and not np.isnan(r[metric_key])]
            rels = [r[rel_key] for r in all_results
                    if r['method'] == method and not np.isnan(r[rel_key])]
            terms = [r['n_active_terms'] for r in all_results if r['method'] == method]
            times = [r['time'] for r in all_results if r['method'] == method]

            if not mses:
                terms_str = f"{np.mean(terms):.1f}" if terms else "—"
                print(f"{method:<25} {'N/A (no dynamics)':<22} {'':18} {terms_str:<12}")
                continue

            mse_str = f"{np.mean(mses):.6f}±{np.std(mses):.6f}"
            rel_str = f"{100*np.mean(rels):.2f}±{100*np.std(rels):.2f}"
            terms_str = f"{np.mean(terms):.1f}" if terms else "—"
            time_str = f"{np.mean(times):.1f}" if times else "—"
            print(f"{method:<25} {mse_str:<22} {rel_str:<18} {terms_str:<12} {time_str:<10}")

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print_metric_table('recon_mse', 'recon_rel_error',
                       'RECONSTRUCTION (same-timestep, all methods)')
    print_metric_table('forecast_mse', 'forecast_rel_error',
                       '\nFORECAST (autonomous rollout, dynamics methods only)')

    # Per-seed details
    print(f"\nPer-seed test relative error (%):")
    print(f"{'Seed':<6}", end='')
    for method in METHODS:
        print(f"  {'recon':>10} {'forecast':>10}", end='')
    print()
    print("-" * (6 + 22 * len(METHODS)))

    for seed in range(NUM_SEEDS):
        row = f"{seed:<6}"
        for method in METHODS:
            r = [x for x in all_results if x['method'] == method and x['seed'] == seed]
            if r:
                recon_rel = r[0].get('recon_rel_error', float('nan'))
                fore_rel = r[0].get('forecast_rel_error', float('nan'))
                recon_str = f"{100*recon_rel:.2f}%" if not np.isnan(recon_rel) else "N/A"
                fore_str = f"{100*fore_rel:.2f}%" if not np.isnan(fore_rel) else "N/A"
                row += f"  {recon_str:>10} {fore_str:>10}"
            else:
                row += f"  {'N/A':>10} {'N/A':>10}"
        print(row)

    # Print autonomous equations
    print(f"\n\n{'='*70}")
    print("DISCOVERED EQUATIONS (autonomous, stage 2)")
    print(f"{'='*70}")
    for r in all_results:
        eq = r.get('equations', 'N/A')
        if eq and eq != 'N/A' and eq != 'FAILED' and not eq.startswith('Failed'):
            print(f"\n--- {r['method']}, seed {r['seed']} ---")
            print(eq)

    print(f"\n\nReference: SINDy-SHRED relative error = 2.01% (Gao et al.)")
    print(f"{'='*70}")
    print("Benchmark complete.")


if __name__ == '__main__':
    main()
