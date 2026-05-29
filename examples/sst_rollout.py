"""SST benchmark with RolloutSINDyRNN.

Architecture:
  z_0 = GRU(sparse_sensors[0:T_w])      # encode 1-year warmup
  z_{t+1} = z_t + dt * P(z_t)            # autonomous polynomial ODE rollout
  x_hat_t = W @ z_t + b                  # linear decoder at every step

Trained end-to-end with per-step reconstruction loss + rollout curriculum.

Data: NOAA OI SST V2 (1992-2019)
  - 1,400 weekly snapshots, ~44,000 sea grid points
  - 250 random sensors (0.57% spatial coverage)
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

from sindy_rnn import RolloutSINDyRNN, fit_rollout

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# Configuration
# ============================================================

DATA_PATH = 'data/SST_data.mat'

# Data
NUM_SENSORS = 250
LATENT_DIM = 3
DT = 1 / 52              # weekly
TRAIN_LENGTH = 1000       # matches SINDy-SHRED paper
LAGS = 52                 # T_w = 1 year warmup

# Model
POLY_DEGREE = 3           # SST dynamics are linear
ENSEMBLE_SIZE = 11
NUM_EULER_STEPS = 3
GRU_LAYERS = 2

# Training
T_MAX = 100               # max rollout = ~2 years
T_START = 1
DELTA_T = 2
EPOCHS = 1000
BATCH_SIZE = 16           # smaller than Lorenz due to large n_full
BATCHES_PER_EPOCH = 8
LR = 5e-3
LAMBDA_0 = 1e-3           # z_0 norm regularization
LAMBDA_S = 1e-3           # L1 sparsity on polynomial coefficients
GRAD_CLIP = 0.5
PRUNING_THRESHOLD = 0.1
PRUNING_FREQUENCY = 100


# ============================================================
# Data loading
# ============================================================

def load_sst_data():
    """Load SST data, filter to sea grid points."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"SST data not found at {DATA_PATH}. "
            f"Download NOAA OI SST V2 and save as {DATA_PATH}")
    load_X = loadmat(DATA_PATH)['Z'].T  # (1400, 64800)
    mean_X = np.mean(load_X, axis=0)
    sst_locs = np.where(mean_X != 0)[0]
    return load_X[:, sst_locs], sst_locs


# ============================================================
# Evaluation
# ============================================================

def reconstruct_rollout(model, X_scaled, sensor_locs, T_w, scaler):
    """Same-timestep reconstruction: encode warmup -> decode z_0.

    For each frame t >= T_w-1, encodes sensor window [t-T_w+1, t+1)
    to get z_0, then decodes to full state. No dynamics involved.

    Returns (n_frames, full_dim) in raw space. NaN for frames < T_w-1.
    """
    N = X_scaled.shape[0]
    full_dim = X_scaled.shape[1]
    device = next(model.parameters()).device
    model.eval()

    recons_scaled = np.full((N, full_dim), np.nan)

    with torch.no_grad():
        sparse_all = torch.tensor(
            X_scaled[:, sensor_locs], dtype=torch.float32)  # CPU

        chunk = 32
        for batch_start in range(T_w - 1, N, chunk):
            batch_end = min(batch_start + chunk, N)
            windows = torch.stack([
                sparse_all[t - T_w + 1:t + 1]
                for t in range(batch_start, batch_end)
            ]).to(device)

            z_0 = model.encode(windows)        # (E, chunk, n_latent)
            x_hat = model.decoder(z_0.mean(0)) # (chunk, n_full)
            recons_scaled[batch_start:batch_end] = x_hat.cpu().numpy()

    valid = ~np.isnan(recons_scaled[:, 0])
    recons = np.full((N, full_dim), np.nan)
    recons[valid] = scaler.inverse_transform(recons_scaled[valid])
    return recons


def forecast_rollout(model, X_scaled, sensor_locs, train_end, T_w, scaler):
    """Autonomous forecast from train boundary.

    Encodes the last training warmup window, then rolls forward
    using only the polynomial dynamics (no sensor input).

    Returns (n_frames, full_dim) in raw space. NaN for training frames.
    """
    N = X_scaled.shape[0]
    full_dim = X_scaled.shape[1]
    n_forecast = N - train_end
    device = next(model.parameters()).device
    model.eval()

    forecast_arr = np.full((N, full_dim), np.nan)

    with torch.no_grad():
        warmup = torch.tensor(
            X_scaled[train_end - T_w:train_end, sensor_locs],
            dtype=torch.float32).unsqueeze(0).to(device)

        x_hat, z_traj = model.forecast(warmup, n_forecast)
        forecast_scaled = x_hat[0].cpu().numpy()  # (n_forecast, n_full)

        # Truncate at first divergent step (keep good early steps)
        step_max = np.max(np.abs(forecast_scaled), axis=1)  # (n_forecast,)
        diverged = np.isnan(step_max) | (step_max > 1e6)
        first_bad = np.argmax(diverged) if diverged.any() else n_forecast
        if first_bad < n_forecast:
            print(f"  Forecast diverged at step {first_bad} of {n_forecast}")
            forecast_scaled = forecast_scaled[:first_bad]

        if len(forecast_scaled) > 0:
            forecast_arr[train_end:train_end + len(forecast_scaled)] = \
                scaler.inverse_transform(forecast_scaled)

    return forecast_arr, z_traj


def evaluate_on_frames(pred, X_raw, test_frames):
    """Compute MSE and relative error on test frames."""
    valid = ~np.isnan(pred[test_frames, 0])
    valid_frames = test_frames[valid]

    if len(valid_frames) == 0:
        return float('nan'), float('nan'), 0

    p = pred[valid_frames]
    g = X_raw[valid_frames]
    mse = np.mean((p - g) ** 2)
    rel = np.linalg.norm(p - g) / np.linalg.norm(g)
    return mse, rel, len(valid_frames)


def extract_latent(model, X_scaled, sensor_locs, train_end, T_w):
    """Extract encoder latent trajectory and autonomous rollout trajectory."""
    N = X_scaled.shape[0]
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        sparse_all = torch.tensor(
            X_scaled[:, sensor_locs], dtype=torch.float32)

        # Encoder z for all valid frames
        z_encoder = np.full((N, LATENT_DIM), np.nan)
        chunk = 32
        for batch_start in range(T_w - 1, N, chunk):
            batch_end = min(batch_start + chunk, N)
            windows = torch.stack([
                sparse_all[t - T_w + 1:t + 1]
                for t in range(batch_start, batch_end)
            ]).to(device)
            z = model.encode(windows).mean(0)  # (chunk, n_latent)
            z_encoder[batch_start:batch_end] = z.cpu().numpy()

        # Autonomous rollout from train boundary
        warmup = torch.tensor(
            X_scaled[train_end - T_w:train_end, sensor_locs],
            dtype=torch.float32).unsqueeze(0).to(device)

        _, z_traj = model.forecast(warmup, N - train_end)
        z_rollout = z_traj.mean(0)[0].cpu().numpy()  # (n_forecast, n_latent)

        # Prepend z_0 (at train_end-1)
        z_0 = z_encoder[train_end - 1]
        z_rollout = np.vstack([z_0[np.newaxis, :], z_rollout])

    return z_encoder, z_rollout


# ============================================================
# Plotting
# ============================================================

SST_GRID_ROWS = 180
SST_GRID_COLS = 360


def generate_sst_images(X_raw, sst_locs, pred, train_end, save_path,
                        prefix='recon'):
    """Generate SST map comparison: truth vs prediction vs error."""
    n_frames = X_raw.shape[0]
    X_mean = np.nanmean(X_raw, axis=0)

    def to_grid(vec):
        grid = np.full(SST_GRID_ROWS * SST_GRID_COLS, np.nan)
        grid[sst_locs] = vec
        return grid.reshape(SST_GRID_ROWS, SST_GRID_COLS)

    if prefix == 'forecast':
        start, end = train_end, n_frames - 1
    else:
        valid_mask = ~np.isnan(pred[:, 0])
        if not valid_mask.any():
            return
        start = int(np.argmax(valid_mask))
        end = n_frames - 1 - int(np.argmax(valid_mask[::-1]))

    frame_indices = np.linspace(start, end,
                                min(8, end - start + 1), dtype=int)
    n_cols = len(frame_indices)

    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 7))

    gt_anom = X_raw[frame_indices] - X_mean[np.newaxis, :]
    pred_anom = pred[frame_indices] - X_mean[np.newaxis, :]

    vlim = np.nanpercentile(np.abs(gt_anom), 98)

    err_vals = np.abs(gt_anom - pred_anom)
    valid_err = err_vals[~np.isnan(err_vals)]
    err_max = np.nanpercentile(valid_err, 95) if len(valid_err) > 0 else 1.0

    label_mid = 'Recon. anomaly' if prefix == 'recon' else 'Forecast anomaly'

    for j, fidx in enumerate(frame_indices):
        region = "test" if fidx >= train_end else "train"
        has_pred = not np.isnan(pred[fidx, 0])

        axes[0, j].imshow(to_grid(gt_anom[j]), cmap='RdBu_r',
                          vmin=-vlim, vmax=vlim, aspect='auto', origin='upper')
        axes[0, j].set_title(f"t={fidx} ({region})", fontsize=8)

        if has_pred:
            axes[1, j].imshow(to_grid(pred_anom[j]), cmap='RdBu_r',
                              vmin=-vlim, vmax=vlim, aspect='auto',
                              origin='upper')
            axes[2, j].imshow(to_grid(np.abs(gt_anom[j] - pred_anom[j])),
                              cmap='hot', vmin=0, vmax=err_max,
                              aspect='auto', origin='upper')
        else:
            for row in [1, 2]:
                axes[row, j].text(0.5, 0.5, 'N/A',
                                  transform=axes[row, j].transAxes,
                                  ha='center', va='center',
                                  fontsize=12, color='gray')

        for row in range(3):
            axes[row, j].set_xticks([])
            axes[row, j].set_yticks([])

    for row, label in enumerate(['True anomaly', label_mid, '|Error|']):
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(f'Rollout — SST {prefix} anomaly', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_latent_dynamics(z_encoder, z_rollout, train_end, save_path):
    """Plot encoder z and autonomous rollout z."""
    fig, axes = plt.subplots(LATENT_DIM, 1, figsize=(12, 3 * LATENT_DIM),
                              sharex=True)
    if LATENT_DIM == 1:
        axes = [axes]

    frames = np.arange(len(z_encoder))
    rollout_frames = np.arange(train_end - 1, train_end - 1 + len(z_rollout))

    for d in range(LATENT_DIM):
        ax = axes[d]
        valid = ~np.isnan(z_encoder[:, d])
        ax.plot(frames[valid], z_encoder[valid, d], 'b-', alpha=0.7,
                linewidth=1, label='Encoder')
        ax.plot(rollout_frames, z_rollout[:, d], 'r--', alpha=0.7,
                linewidth=1.5, label='Autonomous rollout')
        ax.axvline(x=train_end, color='k', linestyle=':', alpha=0.5)
        ax.set_ylabel(f'z{d}')
        if d == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Frame')
    fig.suptitle('Rollout — Latent Dynamics', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_forecast_mse(forecast, X_raw, train_end, save_path):
    """Plot per-timestep forecast MSE."""
    n_forecast = X_raw.shape[0] - train_end
    mse_per_step = np.full(n_forecast, np.nan)

    for k in range(n_forecast):
        frame = train_end + k
        if not np.isnan(forecast[frame, 0]):
            mse_per_step[k] = np.mean((forecast[frame] - X_raw[frame]) ** 2)

    valid = ~np.isnan(mse_per_step)
    if not valid.any():
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(n_forecast)
    ax.plot(steps[valid], mse_per_step[valid], linewidth=1.5)
    ax.set_xlabel('Forecast step (weeks)')
    ax.set_ylabel('MSE')
    ax.set_title('Autonomous Forecast MSE over Time')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("SST Rollout Benchmark")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load data
    print("\nLoading SST data...")
    X, sst_locs = load_sst_data()
    n_time, full_dim = X.shape
    print(f"  Shape: ({n_time}, {full_dim})")

    # Train/test split (matches existing SST benchmarks)
    train_end = TRAIN_LENGTH + LAGS  # 1052
    validate_length = 30  # SINDy-SHRED validation buffer
    shred_val_end = (TRAIN_LENGTH + validate_length - 1) + LAGS - 1  # 1080
    test_start = max(train_end, shred_val_end + 1)  # 1081 for fair comparison
    test_frames = np.arange(test_start, n_time)

    # Scale using training data
    scaler = MinMaxScaler()
    scaler.fit(X[:train_end])
    X_scaled = scaler.transform(X)

    # Sensor locations (seed=0)
    rng = np.random.default_rng(0)
    sensor_locs = rng.choice(full_dim, size=NUM_SENSORS, replace=False)

    # Prepare raw time series for fit_rollout (keep on CPU for memory)
    x_sparse_train = torch.tensor(
        X_scaled[:train_end, sensor_locs], dtype=torch.float32)
    x_full_train = torch.tensor(
        X_scaled[:train_end], dtype=torch.float32)
    # Include T_w warmup frames before test so _eval_test can build windows
    x_sparse_test = torch.tensor(
        X_scaled[train_end - LAGS:, sensor_locs], dtype=torch.float32)
    x_full_test = torch.tensor(
        X_scaled[train_end - LAGS:], dtype=torch.float32)

    print(f"  Train: {train_end} frames")
    print(f"  Test: {n_time - train_end} frames (eval from frame {test_start})")
    print(f"  Sensors: {NUM_SENSORS} of {full_dim}")
    print(f"  Latent dim: {LATENT_DIM}, Poly degree: {POLY_DEGREE}")

    # Build model
    model = RolloutSINDyRNN(
        n_sensors=NUM_SENSORS,
        n_latent=LATENT_DIM,
        n_full=full_dim,
        ensemble_size=ENSEMBLE_SIZE,
        polynomial_degree=POLY_DEGREE,
        dt=DT,
        num_euler_steps=NUM_EULER_STEPS,
        dynamics_dropout=0.1,
        gru_layers=GRU_LAYERS,
        state_names=['z0', 'z1', 'z2'],
        decomposed=True,
        compile_dynamics=True,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dyn = sum(p.numel() for p in model.dynamics.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"\n  Model: {n_params:,} parameters")
    print(f"    Encoder (GRU 2L, {NUM_SENSORS}->{LATENT_DIM}): {n_enc:,}")
    print(f"    Dynamics (PolyRNN deg={POLY_DEGREE}, E={ENSEMBLE_SIZE}): {n_dyn:,}")
    print(f"    Decoder (Linear {LATENT_DIM}->{full_dim}): {n_dec:,}")

    t0 = time.time()

    # ── Train ──
    fit_rollout(
        model,
        x_sparse_train, x_full_train,
        T_w=LAGS,
        T_max=T_MAX,
        T_start=T_START,
        delta_T=DELTA_T,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        batches_per_epoch=BATCHES_PER_EPOCH,
        learning_rate=LR,
        lambda_0=LAMBDA_0,
        lambda_s=LAMBDA_S,
        grad_clip=GRAD_CLIP,
        pruning_threshold=PRUNING_THRESHOLD,
        pruning_frequency=PRUNING_FREQUENCY,
        pruning_method='median',
        lr_patience=100,
        x_sparse_test=x_sparse_test,
        x_full_test=x_full_test,
        verbose=True,
    )

    elapsed = time.time() - t0

    # ── Evaluate ──
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # 1. Same-timestep reconstruction (encoder + decoder only, no dynamics)
    print("\n  Computing reconstruction...")
    recons = reconstruct_rollout(model, X_scaled, sensor_locs, LAGS, scaler)
    recon_mse, recon_rel, n_recon = evaluate_on_frames(recons, X, test_frames)
    print(f"  Reconstruction MSE: {recon_mse:.6f} ({n_recon} frames)")
    print(f"  Reconstruction rel. error: {100 * recon_rel:.2f}%")

    # 2. Autonomous forecast
    print("\n  Computing forecast...")
    forecast, z_traj_fore = forecast_rollout(
        model, X_scaled, sensor_locs, train_end, LAGS, scaler)
    fore_mse, fore_rel, n_fore = evaluate_on_frames(forecast, X, test_frames)
    print(f"  Forecast MSE: {fore_mse:.6f} ({n_fore} frames)")
    print(f"  Forecast rel. error: {100 * fore_rel:.2f}%")

    # 3. Active terms
    active = model.count_active_terms()
    n_active = sum(active.values())
    print(f"\n  Active terms: {n_active} "
          f"({', '.join(f'{k}:{v}' for k, v in active.items())})")

    # 4. Equations
    print("\n  Discovered equations:")
    model.print_equations()

    print(f"\n  Training time: {elapsed:.1f}s")

    # ── Save results ──
    results = {
        'method': 'rollout-sindy-rnn',
        'recon_mse': float(recon_mse),
        'recon_rel_error': float(recon_rel),
        'forecast_mse': float(fore_mse),
        'forecast_rel_error': float(fore_rel),
        'n_active_terms': n_active,
        'n_params': n_params,
        'time': elapsed,
        'config': {
            'latent_dim': LATENT_DIM,
            'poly_degree': POLY_DEGREE,
            'ensemble_size': ENSEMBLE_SIZE,
            'T_w': LAGS,
            'T_max': T_MAX,
            'epochs': EPOCHS,
            'lr': LR,
            'lambda_s': LAMBDA_S,
            'batch_size': BATCH_SIZE,
        },
    }

    try:
        results['equations'] = model.get_equations()
    except Exception:
        pass

    results_path = 'sst_rollout_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # ── Plots ──
    print("\nGenerating plots...")
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    generate_sst_images(X, sst_locs, recons, train_end,
                        os.path.join(results_dir, 'sst_rollout_recon.png'),
                        prefix='recon')

    generate_sst_images(X, sst_locs, forecast, train_end,
                        os.path.join(results_dir, 'sst_rollout_forecast.png'),
                        prefix='forecast')

    z_encoder, z_rollout = extract_latent(
        model, X_scaled, sensor_locs, train_end, LAGS)
    plot_latent_dynamics(z_encoder, z_rollout, train_end,
                        os.path.join(results_dir, 'sst_rollout_latent.png'))

    plot_forecast_mse(forecast, X, train_end,
                      os.path.join(results_dir, 'sst_rollout_forecast_mse.png'))

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Reconstruction rel. error: {100 * recon_rel:.2f}%")
    print(f"  Forecast rel. error:       {100 * fore_rel:.2f}%")
    print(f"  Active terms:              {n_active}")
    print(f"  Training time:             {elapsed:.1f}s")
    print(f"\n  Reference: SINDy-SHRED rel. error = 2.01% (Gao et al.)")
    print("=" * 60)


if __name__ == '__main__':
    main()
