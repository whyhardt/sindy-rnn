"""Cylinder flow benchmark with RolloutSINDyRNN.

Architecture:
  z_0 = GRU(sparse_sensors[0:T_w])      # encode 2-second warmup
  z_{t+1} = z_t + dt * P(z_t)            # autonomous polynomial ODE rollout
  x_hat_t = D(z_t)                       # MLP decoder at every step

Trained end-to-end with per-step reconstruction loss + rollout curriculum.

Data: Flow over cylinder
  - 334 frames, 400x1000 grayscale
  - 200 random sensors (0.05% spatial coverage)
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
from sklearn.preprocessing import MinMaxScaler

from sindy_rnn import RolloutSINDyRNN, fit_rollout, refit_rollout

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# Configuration
# ============================================================

DATA_PATH = 'data/flow_over_cylinder.npy'

# Data
NUM_SENSORS = 200
LATENT_DIM = 4
DT = 1 / 30              # 30 FPS
LAGS = 60                # T_w = 2 seconds warmup

# Model
POLY_DEGREE = 3           # cubic dynamics (reference ODE has cubic terms)
ENSEMBLE_SIZE = 11
NUM_EULER_STEPS = 3
GRU_LAYERS = 2

# Training
T_MAX = 60                # max rollout = 2 seconds
T_START = 1
DELTA_T = 2
EPOCHS = 1000
BATCH_SIZE = 8            # small — full_dim = 400K
BATCHES_PER_EPOCH = 8
LR = 5e-3
LAMBDA_0 = 1e-3           # z_0 norm regularization
LAMBDA_S = 1e-2           # L1 sparsity on polynomial coefficients
GRAD_CLIP = 0.5
PRUNING_THRESHOLD = 0.1
PRUNING_FREQUENCY = 100


# ============================================================
# Data loading
# ============================================================

def load_cylinder_data():
    """Load cylinder flow data, subtract temporal mean, flatten."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Cylinder data not found at {DATA_PATH}. "
            f"Place flow_over_cylinder.npy in data/")
    data = np.load(DATA_PATH)                # (334, 400, 1000)
    mean_frame = data.mean(axis=0)           # temporal mean
    data = data - mean_frame                 # remove static background
    return data.reshape(data.shape[0], -1), mean_frame  # (334, 400000)


# ============================================================
# Evaluation
# ============================================================

def reconstruct_rollout(model, X_scaled, sensor_locs, T_w, scaler):
    """Same-timestep reconstruction: encode warmup -> decode z_0.

    Returns (n_frames, full_dim) in raw space. NaN for frames < T_w-1.
    """
    N = X_scaled.shape[0]
    full_dim = X_scaled.shape[1]
    device = next(model.parameters()).device
    model.eval()

    recons_scaled = np.full((N, full_dim), np.nan)

    with torch.no_grad():
        sparse_all = torch.tensor(
            X_scaled[:, sensor_locs], dtype=torch.float32)

        chunk = 16  # smaller than SST — larger full_dim
        for batch_start in range(T_w - 1, N, chunk):
            batch_end = min(batch_start + chunk, N)
            windows = torch.stack([
                sparse_all[t - T_w + 1:t + 1]
                for t in range(batch_start, batch_end)
            ]).to(device)

            z_0 = model.encode(windows)
            x_hat = model.decoder(z_0.mean(0))
            recons_scaled[batch_start:batch_end] = x_hat.cpu().numpy()

    valid = ~np.isnan(recons_scaled[:, 0])
    recons = np.full((N, full_dim), np.nan)
    recons[valid] = scaler.inverse_transform(recons_scaled[valid])
    return recons


def forecast_rollout(model, X_scaled, sensor_locs, train_end, T_w, scaler):
    """Autonomous forecast from train boundary.

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
        forecast_scaled = x_hat[0].cpu().numpy()

        # Truncate at first divergent step
        step_max = np.max(np.abs(forecast_scaled), axis=1)
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

        z_encoder = np.full((N, LATENT_DIM), np.nan)
        chunk = 16
        for batch_start in range(T_w - 1, N, chunk):
            batch_end = min(batch_start + chunk, N)
            windows = torch.stack([
                sparse_all[t - T_w + 1:t + 1]
                for t in range(batch_start, batch_end)
            ]).to(device)
            z = model.encode(windows).mean(0)
            z_encoder[batch_start:batch_end] = z.cpu().numpy()

        warmup = torch.tensor(
            X_scaled[train_end - T_w:train_end, sensor_locs],
            dtype=torch.float32).unsqueeze(0).to(device)

        _, z_traj = model.forecast(warmup, N - train_end)
        z_rollout = z_traj.mean(0)[0].cpu().numpy()

        z_0 = z_encoder[train_end - 1]
        z_rollout = np.vstack([z_0[np.newaxis, :], z_rollout])

    return z_encoder, z_rollout


# ============================================================
# Plotting
# ============================================================

FRAME_ROWS = 400
FRAME_COLS = 1000


def generate_cylinder_images(X_raw, mean_frame, pred, train_end, save_path,
                             prefix='recon'):
    """Generate cylinder frame comparison: truth vs prediction vs error."""
    n_frames = X_raw.shape[0]

    if prefix == 'forecast':
        start, end = train_end, n_frames - 1
    else:
        valid_mask = ~np.isnan(pred[:, 0])
        if not valid_mask.any():
            return
        start = int(np.argmax(valid_mask))
        end = n_frames - 1 - int(np.argmax(valid_mask[::-1]))

    frame_indices = np.linspace(start, end,
                                min(6, end - start + 1), dtype=int)
    n_cols = len(frame_indices)

    fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 6))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    vlim = np.nanpercentile(np.abs(X_raw[frame_indices]), 98)

    label_mid = 'Recon.' if prefix == 'recon' else 'Forecast'

    for j, fidx in enumerate(frame_indices):
        region = "test" if fidx >= train_end else "train"
        has_pred = not np.isnan(pred[fidx, 0])

        gt_frame = X_raw[fidx].reshape(FRAME_ROWS, FRAME_COLS)
        axes[0, j].imshow(gt_frame, cmap='RdBu_r',
                          vmin=-vlim, vmax=vlim, aspect='auto')
        axes[0, j].set_title(f"t={fidx} ({region})", fontsize=8)

        if has_pred:
            pred_frame = pred[fidx].reshape(FRAME_ROWS, FRAME_COLS)
            err_frame = np.abs(gt_frame - pred_frame)
            axes[1, j].imshow(pred_frame, cmap='RdBu_r',
                              vmin=-vlim, vmax=vlim, aspect='auto')
            axes[2, j].imshow(err_frame, cmap='hot', vmin=0,
                              vmax=vlim * 0.5, aspect='auto')
        else:
            for row in [1, 2]:
                axes[row, j].text(0.5, 0.5, 'N/A',
                                  transform=axes[row, j].transAxes,
                                  ha='center', va='center',
                                  fontsize=12, color='gray')

        for row in range(3):
            axes[row, j].set_xticks([])
            axes[row, j].set_yticks([])

    for row, label in enumerate(['Truth', label_mid, '|Error|']):
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(f'Rollout — Cylinder {prefix}', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_latent_dynamics(z_encoder, z_rollout, train_end, save_path):
    """Plot encoder z and autonomous rollout z."""
    fig, axes = plt.subplots(LATENT_DIM, 1, figsize=(10, 2.5 * LATENT_DIM),
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
    fig.suptitle('Rollout — Latent Dynamics (Cylinder)', fontsize=12)
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
    t_seconds = steps / 30.0
    ax.plot(t_seconds[valid], mse_per_step[valid], linewidth=1.5)
    ax.set_xlabel('Forecast time (seconds)')
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
    print("Cylinder Flow Rollout Benchmark")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load data
    print("\nLoading cylinder data...")
    X, mean_frame = load_cylinder_data()
    n_time, full_dim = X.shape
    print(f"  Shape: ({n_time}, {full_dim})")
    print(f"  Frame size: {FRAME_ROWS}x{FRAME_COLS}")

    # Train/test split (80/20, matching cylinder_benchmark.py)
    train_length = n_time - LAGS - 67  # 207
    train_end = train_length + LAGS    # 267
    test_frames = np.arange(train_end, n_time)

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
    x_sparse_test = torch.tensor(
        X_scaled[train_end:, sensor_locs], dtype=torch.float32)
    x_full_test = torch.tensor(
        X_scaled[train_end:], dtype=torch.float32)

    print(f"  Train: {train_end} frames ({train_length} windows)")
    print(f"  Test: {n_time - train_end} frames")
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
        state_names=[f'z{i}' for i in range(LATENT_DIM)],
        decomposed=True,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dyn = sum(p.numel() for p in model.dynamics.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"\n  Model: {n_params:,} parameters")
    print(f"    Encoder (GRU {GRU_LAYERS}L, {NUM_SENSORS}->{LATENT_DIM}): {n_enc:,}")
    print(f"    Dynamics (PolyRNN deg={POLY_DEGREE}, E={ENSEMBLE_SIZE}): {n_dyn:,}")
    print(f"    Decoder (MLP {LATENT_DIM}->...{full_dim}): {n_dec:,}")

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
        cosine_decay=False,
        x_sparse_test=x_sparse_test,
        x_full_test=x_full_test,
        verbose=True,
    )

    # ── Stage 2: Refit dynamics on frozen encoder latents ──
    refit_rollout(
        model,
        x_sparse_train,
        T_w=LAGS,
        refit_epochs=3000,
        refit_learning_rate=5e-2,
        refit_l2=5e-2,
        refit_pruning_threshold=0.1,
        refit_pruning_frequency=100,
        refit_pruning_method='median',
        centered_diff=True,
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
        'dataset': 'cylinder',
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

    results_path = 'cylinder_rollout_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # ── Plots ──
    print("\nGenerating plots...")
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    generate_cylinder_images(X, mean_frame, recons, train_end,
                             os.path.join(results_dir,
                                          'cylinder_rollout_recon.png'),
                             prefix='recon')

    generate_cylinder_images(X, mean_frame, forecast, train_end,
                             os.path.join(results_dir,
                                          'cylinder_rollout_forecast.png'),
                             prefix='forecast')

    z_encoder, z_rollout = extract_latent(
        model, X_scaled, sensor_locs, train_end, LAGS)
    plot_latent_dynamics(z_encoder, z_rollout, train_end,
                         os.path.join(results_dir,
                                      'cylinder_rollout_latent.png'))

    plot_forecast_mse(forecast, X, train_end,
                      os.path.join(results_dir,
                                   'cylinder_rollout_forecast_mse.png'))

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Reconstruction rel. error: {100 * recon_rel:.2f}%")
    print(f"  Forecast rel. error:       {100 * fore_rel:.2f}%")
    print(f"  Active terms:              {n_active}")
    print(f"  Training time:             {elapsed:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
