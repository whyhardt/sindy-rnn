"""Multi-seed cylinder flow benchmark: sindy-rnn vs SINDy-SHRED vs SHRED.

Runs 5 seeds for each of 3 methods on cylinder flow data, with matched
sensor locations per seed.

Evaluation uses two metrics on the SAME held-out test frames:
  1. Reconstruction: same-timestep encode-decode (sensors_t -> x_t). All 3 methods.
  2. Forecast: autonomous rollout from z_0 at train boundary. sindy-rnn + sindy-shred only.

Methods:
  1. sindy-rnn (SparseAutoencoderRNN with factored polynomial dynamics)
  2. SINDy-SHRED (GRU encoder + E_SINDy dynamics + MLP decoder)
  3. SHRED (SINDy-SHRED with sindy_regularization=0, no dynamics constraint)

Data: 334 frames at 30 FPS, 400x1000 grayscale, 200 random sensors.
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
np.math = math  # pysindy uses np.math.factorial

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sindy_rnn import SparseAutoencoderRNN, fit_autoencoder
from sindy_shred import SINDySHRED

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# Configuration
# ============================================================

NUM_SEEDS = 1
DATA_PATH = 'data/flow_over_cylinder.npy'

# Shared architecture
NUM_SENSORS = 200
LATENT_DIM = 4
POLY_ORDER = 3
DT = 1 / 30
LAGS = 60
GRU_LAYERS = 2
DECODER_L1 = 350
DECODER_L2 = 400
DROPOUT = 0.1

# SINDy-SHRED (paper defaults)
SHRED_EPOCHS = 1000
SHRED_BATCH_SIZE = 64       # appendix: batch size 64
SHRED_LR = 5e-4             # appendix: learning rate 5e-4
SHRED_THRESHOLD = 1e-3      # appendix: thresholds ranging (1e-4, 1e-3)
SHRED_PATIENCE = 20
SHRED_SINDY_REG = 10.0
SHRED_THRES_EPOCH = 300     # appendix: thresholding every 300 epochs

# sindy-rnn
RNN_WINDOW = 30
RNN_ENSEMBLE = 11
RNN_EPOCHS = 1000
RNN_WARMUP = 200
RNN_LR = 1e-3
RNN_L1 = 5e-3
RNN_PRUNE_THRESHOLD = 0.1
RNN_PRUNE_FREQ = 20
RNN_REFIT = 100
RNN_GRU_HIDDEN = None  # match SINDy-SHRED: GRU hidden_size = latent_dim

# METHODS = ['sindy-shred', 'shred', 'sindy-rnn']
METHODS = ['sindy-shred']


# ============================================================
# Data loading and preparation
# ============================================================

def load_data():
    data = np.load(DATA_PATH)             # (334, 400, 1000)
    mean_frame = data.mean(axis=0)        # (400, 1000) — temporal mean
    data = data - mean_frame              # remove static background
    return data.reshape(data.shape[0], -1)


def get_sensor_locs(full_dim, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(full_dim, size=NUM_SENSORS, replace=False)


def prepare_sindy_rnn_data(X_scaled, sensor_locs, window_size, train_end):
    N = len(X_scaled)
    sparse_all = X_scaled[:, sensor_locs]

    usable_train = train_end - 1
    n_train = usable_train // window_size
    sparse_train = sparse_all[:n_train * window_size].reshape(n_train, window_size, -1)
    full_next_train = X_scaled[1:n_train * window_size + 1].reshape(n_train, window_size, -1)

    test_start = train_end
    usable_test = N - test_start - 1
    n_test = usable_test // window_size
    if n_test > 0:
        sparse_test = sparse_all[test_start:test_start + n_test * window_size].reshape(
            n_test, window_size, -1)
        full_next_test = X_scaled[test_start + 1:test_start + n_test * window_size + 1].reshape(
            n_test, window_size, -1)
    else:
        sparse_test, full_next_test = None, None

    def to_dev(x):
        return torch.tensor(x, dtype=torch.float32).to(DEVICE) if x is not None else None
    return to_dev(sparse_train), to_dev(full_next_train), \
           to_dev(sparse_test), to_dev(full_next_test)


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_reconstructions(recons, X_raw, test_frames):
    """Compute MSE and relative error on test frames (raw space).

    Works for both reconstruction and forecast arrays — handles NaN gracefully.
    """
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
    """Compute per-timestep MSE for autonomous forecast.

    Returns (n_forecast,) array where entry k is the MSE at forecast step k
    (i.e., frame train_end + k). NaN where forecast is unavailable.
    """
    n_frames = X_raw.shape[0]
    n_forecast = n_frames - train_end
    mse_per_step = np.full(n_forecast, np.nan)

    for k in range(n_forecast):
        frame = train_end + k
        if not np.isnan(forecast[frame, 0]):
            mse_per_step[k] = np.mean((forecast[frame] - X_raw[frame]) ** 2)

    return mse_per_step


# ============================================================
# Metric 1: Reconstruction (same-timestep encode-decode)
# ============================================================

def reconstruct_sindy_shred(shred_obj):
    """SINDy-SHRED/SHRED same-timestep reconstruction: sensors_t -> z_t -> x_t.

    Returns (n_frames, full_dim) in raw space. NaN for frames without reconstruction.
    """
    shred_obj._shred.eval()
    n_frames = shred_obj._n_time_dim
    lags = shred_obj._lags

    recon_train = shred_obj.sensor_recon(data_type='train', return_scaled=False)
    full_dim = recon_train.shape[1]
    recons = np.full((n_frames, full_dim), np.nan)

    for i, idx in enumerate(shred_obj._train_ind):
        frame = idx + lags - 1
        if frame < n_frames:
            recons[frame] = recon_train[i]

    recon_val = shred_obj.sensor_recon(data_type='validate', return_scaled=False)
    for i, idx in enumerate(shred_obj._val_ind):
        frame = idx + lags - 1
        if frame < n_frames:
            recons[frame] = recon_val[i]

    if shred_obj._test_ind is not None and len(shred_obj._test_ind) > 0:
        recon_test = shred_obj.sensor_recon(data_type='test', return_scaled=False)
        for i, idx in enumerate(shred_obj._test_ind):
            frame = idx + lags - 1
            if frame < n_frames:
                recons[frame] = recon_test[i]

    return recons


def reconstruct_sindy_rnn(model, X_scaled, sensor_locs, scaler):
    """sindy-rnn same-timestep reconstruction: sensors_t -> z_t -> x_t.

    Encodes the full sensor sequence through the GRU, then decodes z_t at each
    frame directly (no dynamics step). Tests encoder-decoder quality only.

    Returns (n_frames, full_dim) in raw space.
    """
    N = X_scaled.shape[0]
    full_dim = X_scaled.shape[1]
    model.eval()

    recons_scaled = np.full((N, full_dim), np.nan)

    with torch.no_grad():
        sparse_all = torch.tensor(
            X_scaled[:, sensor_locs], dtype=torch.float32).to(DEVICE)
        encoded = model.encoder(sparse_all.unsqueeze(0))  # (1, N, latent_dim)

        # Decode in chunks to avoid OOM on large full_dim
        for start in range(0, N, 64):
            end = min(start + 64, N)
            z_chunk = encoded[:, start:end, :]  # (1, chunk, latent_dim)
            decoded = model.decoder(z_chunk)     # (1, chunk, full_dim)
            recons_scaled[start:end] = decoded[0].cpu().numpy()

    recons = scaler.inverse_transform(recons_scaled)
    return recons


# ============================================================
# Metric 2: Forecast (autonomous rollout from z_0)
# ============================================================

def forecast_sindy_shred(shred_obj, n_frames, full_dim, train_end):
    """Autonomous forecast using SINDy-SHRED's post-hoc SINDy model.

    Integrates the discovered ODE z' = f(z) forward from the last training
    GRU state, then decodes to full state.

    Returns (n_frames, full_dim) in raw space. NaN for training frames.
    """
    n_forecast = n_frames - train_end
    forecast_arr = np.full((n_frames, full_dim), np.nan)

    if shred_obj._model is None:
        print("  Forecast skipped: no post-hoc SINDy model")
        return forecast_arr

    try:
        forecast_raw = shred_obj.forecast(
            n_steps=n_forecast, init_from="train", return_scaled=False)
        n_actual = min(len(forecast_raw), n_forecast)
        # Check for divergence
        if np.any(np.abs(forecast_raw[:n_actual]) > 1e6) or np.any(np.isnan(forecast_raw[:n_actual])):
            print("  Forecast diverged")
            return forecast_arr
        forecast_arr[train_end:train_end + n_actual] = forecast_raw[:n_actual]
    except Exception as e:
        print(f"  SINDy-SHRED forecast failed: {e}")

    return forecast_arr


def forecast_sindy_rnn(model, X_scaled, sensor_locs, train_end, scaler):
    """Autonomous forecast using sindy-rnn polynomial dynamics.

    Encodes sensors up to train_end to get z_0, then evolves forward using
    only the polynomial dynamics P(z) — no sensor input.

    Returns (n_frames, full_dim) in raw space. NaN for training frames.
    """
    N = X_scaled.shape[0]
    full_dim = X_scaled.shape[1]
    E = model.ensemble_size
    n_forecast = N - train_end
    model.eval()

    forecast_arr = np.full((N, full_dim), np.nan)

    with torch.no_grad():
        # Encode sensor history up to train boundary
        sparse_train = torch.tensor(
            X_scaled[:train_end, sensor_locs], dtype=torch.float32).to(DEVICE)
        encoded = model.encoder(sparse_train.unsqueeze(0))  # (1, train_end, latent_dim)
        z_0 = encoded[:, -1:, :]  # (1, 1, latent_dim) — last training state
        z_0 = z_0.expand(E, -1, -1)  # (E, 1, latent_dim)

        # Autonomous rollout — get latent trajectory first, decode in chunks
        theta = model.dynamics.rnn.unfold_polynomial_coefficients()
        h = z_0
        latent_steps = []
        for t in range(n_forecast):
            h = model.dynamics.rnn.forward_polynomial(
                h, None, mask=model.dynamics.coefficient_masks, theta=theta)
            latent_steps.append(h)

        # Decode in chunks to avoid OOM
        chunk_size = 10
        forecast_scaled = []
        for i in range(0, n_forecast, chunk_size):
            chunk = torch.stack(latent_steps[i:i + chunk_size], dim=2)  # (E, 1, chunk, latent_dim)
            decoded = model.decoder(chunk)  # (E, 1, chunk, full_dim)
            forecast_scaled.append(decoded.mean(0)[0].cpu().numpy())  # (chunk, full_dim)

        forecast_scaled = np.concatenate(forecast_scaled, axis=0)  # (n_forecast, full_dim)
        forecast_arr[train_end:train_end + n_forecast] = scaler.inverse_transform(forecast_scaled)

    return forecast_arr


# ============================================================
# Method runners (train + return reconstruction + forecast)
# ============================================================

def run_sindy_shred(X, sensor_locs, train_length, validate_length,
                    seed, sindy_reg=SHRED_SINDY_REG, save_dir=None):
    """Train SINDy-SHRED (or plain SHRED) and return reconstruction + forecast."""
    shred = SINDySHRED(
        latent_dim=LATENT_DIM, poly_order=POLY_ORDER,
        hidden_layers=GRU_LAYERS, l1=DECODER_L1, l2=DECODER_L2,
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
    n_frames = shred._n_time_dim
    full_dim = X.shape[1]
    train_end = train_length + LAGS

    # Post-hoc SINDy: auto-tune STLSQ threshold on GRU latent trajectories
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

    # Metric 1: Reconstruction (same-timestep)
    recons = reconstruct_sindy_shred(shred)

    # Metric 2: Forecast (autonomous rollout)
    forecast = forecast_sindy_shred(shred, n_frames, full_dim, train_end)

    # Save model
    if save_dir is not None:
        method_tag = 'shred' if sindy_reg == 0.0 else 'sindy_shred'
        path = os.path.join(save_dir, f'cylinder_{method_tag}_seed{seed}.pt')
        torch.save(shred._shred.state_dict(), path)
        print(f"  Saved model to {path}")

    # Free GPU
    shred._shred.cpu()
    torch.cuda.empty_cache()

    return recons, forecast, {
        'n_params': n_params,
        'n_active_terms': n_active,
        'equations': equations,
    }


def run_sindy_rnn(X, X_scaled, sensor_locs, train_end, full_dim, seed, scaler=None,
                  save_dir=None):
    """Train sindy-rnn and return reconstruction + forecast."""
    torch.manual_seed(seed)

    sparse_train, full_next_train, sparse_test, full_next_test = \
        prepare_sindy_rnn_data(X_scaled, sensor_locs, RNN_WINDOW, train_end)

    model = SparseAutoencoderRNN(
        sparse_dim=NUM_SENSORS, full_dim=full_dim, latent_dim=LATENT_DIM,
        ensemble_size=RNN_ENSEMBLE, polynomial_degree=POLY_ORDER,
        encoder_gru_hidden_dim=RNN_GRU_HIDDEN,
        encoder_num_layers=GRU_LAYERS,
        decoder_hidden_dims=[DECODER_L1, DECODER_L2],
        encoder_dropout=DROPOUT, decoder_dropout=DROPOUT,
        dynamics_dropout=DROPOUT,
        state_names=[f'z{i+1}' for i in range(LATENT_DIM)],
    ).to(DEVICE)

    fit_autoencoder(
        model, sparse_train, full_next_train,
        sparse_obs_test=sparse_test, full_state_next_test=full_next_test,
        epochs=RNN_EPOCHS, warmup_steps=RNN_WARMUP, batch_size=1,
        learning_rate=RNN_LR, l1=RNN_L1,
        pruning_threshold=RNN_PRUNE_THRESHOLD, pruning_method='median',
        pruning_frequency=RNN_PRUNE_FREQ, dt=DT,
        refit_epochs=RNN_REFIT, verbose=True,
    )

    active = model.count_active_terms()
    n_active = sum(active.values())
    n_params = sum(p.numel() for p in model.parameters())

    try:
        equations = model.get_continuous_equations(DT)
    except Exception:
        equations = "N/A"

    # Metric 1: Reconstruction (same-timestep, no dynamics)
    recons = reconstruct_sindy_rnn(model, X_scaled, sensor_locs, scaler)

    # Metric 2: Forecast (autonomous rollout)
    forecast = forecast_sindy_rnn(model, X_scaled, sensor_locs, train_end, scaler)

    # Save model
    if save_dir is not None:
        path = os.path.join(save_dir, f'cylinder_sindy_rnn_seed{seed}.pt')
        model.save(path)
        print(f"  Saved model to {path}")

    # Free GPU
    model.cpu()
    torch.cuda.empty_cache()

    return recons, forecast, {
        'n_params': n_params,
        'n_active_terms': n_active,
        'equations': equations,
    }


# ============================================================
# Image generation
# ============================================================

FRAME_HEIGHT = 400
FRAME_WIDTH = 1000


def generate_cylinder_images(X_raw, data_dict, train_end, seed, save_dir, prefix='recon'):
    """Generate composite images for all methods at a given seed.

    Args:
        X_raw: (n_frames, full_dim) ground truth in raw space
        data_dict: dict mapping method_name -> (n_frames, full_dim) array
        train_end: frame index where test region begins
        seed: seed number (for filename)
        save_dir: directory to save images
        prefix: 'recon' for reconstruction, 'forecast' for autonomous rollout
    """
    n_frames = X_raw.shape[0]

    for method, data in data_dict.items():
        # Determine valid frame range for this method
        valid_mask = ~np.isnan(data[:, 0])
        if not valid_mask.any():
            continue
        first_valid = int(np.argmax(valid_mask))
        last_valid = n_frames - 1 - int(np.argmax(valid_mask[::-1]))

        if prefix == 'forecast':
            start = max(train_end, first_valid)
        else:
            start = first_valid
        frame_indices = np.linspace(start, last_valid,
                                    min(8, last_valid - start + 1), dtype=int)

        n_cols = len(frame_indices)
        fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9))

        gt_frames = X_raw[frame_indices]
        vmax_gt = np.nanpercentile(np.abs(gt_frames), 99)
        vmin_gt = -vmax_gt

        row_labels = ['Ground Truth',
                      'Reconstruction' if prefix == 'recon' else 'Forecast',
                      '|Error|']

        for j, fidx in enumerate(frame_indices):
            gt = X_raw[fidx].reshape(FRAME_HEIGHT, FRAME_WIDTH)
            region = "test" if fidx >= train_end else "train"

            axes[0, j].imshow(gt, cmap='RdBu_r', vmin=vmin_gt, vmax=vmax_gt,
                              aspect='auto')
            axes[0, j].set_title(f"t={fidx} ({region})", fontsize=8)

            rec = data[fidx].reshape(FRAME_HEIGHT, FRAME_WIDTH)
            axes[1, j].imshow(rec, cmap='RdBu_r', vmin=vmin_gt, vmax=vmax_gt,
                              aspect='auto')
            err = np.abs(gt - rec)
            axes[2, j].imshow(err, cmap='hot', vmin=0,
                              vmax=vmax_gt * 0.5, aspect='auto')

            for row in range(3):
                axes[row, j].set_xticks([])
                axes[row, j].set_yticks([])

        # Row labels on leftmost column
        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=11)

        fig.suptitle(f'{method} — Cylinder Flow {prefix} (seed {seed})', fontsize=12)
        fig.tight_layout()

        path = os.path.join(save_dir, f'cylinder_{prefix}_{method}_seed{seed}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved image: {path}")


def plot_forecast_mse_over_time(mse_dict, seed, save_dir, prefix):
    """Plot per-timestep forecast MSE for all methods on a single figure.

    Args:
        mse_dict: dict mapping method_name -> (n_forecast,) MSE array
        seed: seed number (for filename)
        save_dir: directory to save plot
        prefix: 'cylinder' or 'sst'
    """
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


def select_median_and_cleanup(all_results, save_dir, prefix, methods, num_seeds):
    """Keep only median-seed images per method, delete the rest."""
    for method in methods:
        mses = [(r['seed'], r['recon_mse']) for r in all_results
                if r['method'] == method and not np.isnan(r['recon_mse'])]
        if not mses:
            continue
        mses.sort(key=lambda x: x[1])
        median_seed = mses[len(mses) // 2][0]
        print(f"  {method}: median seed = {median_seed} "
              f"(recon MSE = {mses[len(mses) // 2][1]:.6f})")
        for seed in range(num_seeds):
            if seed != median_seed:
                for img_prefix in ['recon', 'forecast']:
                    path = os.path.join(save_dir,
                        f'{prefix}_{img_prefix}_{method}_seed{seed}.png')
                    if os.path.exists(path):
                        os.remove(path)
                path = os.path.join(save_dir,
                    f'{prefix}_forecast_mse_seed{seed}.png')
                if os.path.exists(path):
                    os.remove(path)


# ============================================================
# Main benchmark
# ============================================================

def main():
    print("Multi-Seed Cylinder Flow Benchmark")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {NUM_SEEDS}")
    print(f"Methods: {METHODS}")
    print(f"poly_order={POLY_ORDER}, dt={DT:.4f}, lags={LAGS}")

    # Load data
    print("\nLoading data...")
    X = load_data()
    n_frames, full_dim = X.shape
    print(f"  Shape: ({n_frames}, {full_dim})")

    # Train/val split
    train_length = n_frames - LAGS - 67
    validate_length = n_frames - LAGS - train_length
    train_end = train_length + LAGS

    # Common test frames: everything after train_end
    test_frames = np.arange(train_end, n_frames)
    print(f"  Train frames: 0-{train_end-1}, Test frames: {train_end}-{n_frames-1} ({len(test_frames)} frames)")

    # Scale data once (fit on training frames)
    scaler = MinMaxScaler()
    scaler.fit(X[:train_end])
    X_scaled = scaler.transform(X)

    # Create directories
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

        for method in METHODS:
            tag = f"Seed {seed}, {method}"
            print(f"\n{'='*70}")
            print(f"{tag}")
            print(f"{'='*70}")

            t0 = time.time()
            try:
                if method == 'sindy-shred':
                    recons, forecast, info = run_sindy_shred(
                        X, sensor_locs, train_length, validate_length, seed,
                        sindy_reg=SHRED_SINDY_REG, save_dir=save_dir)
                elif method == 'shred':
                    recons, forecast, info = run_sindy_shred(
                        X, sensor_locs, train_length, validate_length, seed,
                        sindy_reg=0.0, save_dir=save_dir)
                elif method == 'sindy-rnn':
                    recons, forecast, info = run_sindy_rnn(
                        X, X_scaled, sensor_locs, train_end, full_dim, seed,
                        scaler=scaler, save_dir=save_dir)

                elapsed = time.time() - t0
                seed_recons[method] = recons
                seed_forecasts[method] = forecast

                # Metric 1: Reconstruction
                recon_mse, recon_rel, n_recon = evaluate_reconstructions(
                    recons, X, test_frames)
                # Metric 2: Forecast
                fore_mse, fore_rel, n_fore = evaluate_reconstructions(
                    forecast, X, test_frames)

                # Per-step forecast MSE
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
                })

        # Generate images for this seed
        if seed_recons:
            print(f"\nGenerating images for seed {seed}...")
            generate_cylinder_images(X, seed_recons, train_end, seed, img_dir, prefix='recon')
            generate_cylinder_images(X, seed_forecasts, train_end, seed, img_dir, prefix='forecast')

            # Forecast MSE over time plot
            seed_fore_mse = {}
            for m in seed_forecasts:
                r = [x for x in all_results if x['method'] == m and x['seed'] == seed]
                if r and r[0].get('forecast_mse_per_step') is not None:
                    seed_fore_mse[m] = np.array(r[0]['forecast_mse_per_step'])
            if seed_fore_mse:
                plot_forecast_mse_over_time(seed_fore_mse, seed, img_dir, 'cylinder')

    # Save results
    with open('cylinder_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to cylinder_benchmark_results.json")

    # ================================================================
    # Summary
    # ================================================================
    def print_metric_table(metric_key, rel_key, title):
        print(f"\n{title}")
        print(f"{'Method':<20} {'MSE (mean±std)':<22} {'Rel.Err %':<18} {'Active':<12} {'Time (s)':<10}")
        print("-" * 82)
        for method in METHODS:
            mses = [r[metric_key] for r in all_results
                    if r['method'] == method and not np.isnan(r[metric_key])]
            rels = [r[rel_key] for r in all_results
                    if r['method'] == method and not np.isnan(r[rel_key])]
            terms = [r['n_active_terms'] for r in all_results if r['method'] == method]
            times = [r['time'] for r in all_results if r['method'] == method]

            if not mses:
                terms_str = f"{np.mean(terms):.1f}" if terms else "—"
                print(f"{method:<20} {'N/A (no dynamics)':<22} {'':18} {terms_str:<12}")
                continue

            mse_str = f"{np.mean(mses):.6f}±{np.std(mses):.6f}"
            rel_str = f"{100*np.mean(rels):.2f}±{100*np.std(rels):.2f}"
            terms_str = f"{np.mean(terms):.1f}" if terms else "—"
            time_str = f"{np.mean(times):.1f}" if times else "—"
            print(f"{method:<20} {mse_str:<22} {rel_str:<18} {terms_str:<12} {time_str:<10}")

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print_metric_table('recon_mse', 'recon_rel_error',
                       'RECONSTRUCTION (same-timestep encode-decode, all methods)')
    print_metric_table('forecast_mse', 'forecast_rel_error',
                       '\nFORECAST (autonomous rollout from z_0, dynamics methods only)')

    # Print discovered equations
    print(f"\n\n{'='*70}")
    print("DISCOVERED EQUATIONS")
    print(f"{'='*70}")
    for r in all_results:
        eq = r.get('equations', 'N/A')
        if eq and eq != 'N/A' and not eq.startswith('Failed'):
            print(f"\n--- {r['method']}, seed {r['seed']} ---")
            print(eq)

    # Select median seed images, delete the rest
    if NUM_SEEDS > 1:
        print(f"\nSelecting median-seed images...")
        select_median_and_cleanup(all_results, img_dir, 'cylinder', METHODS, NUM_SEEDS)

    print(f"\n{'='*70}")
    print("Benchmark complete.")


if __name__ == '__main__':
    main()
