"""Multi-seed SST benchmark: sindy-rnn vs SINDy-SHRED vs SHRED.

Discovers latent governing equations for weekly sea surface temperature data.
Runs 5 seeds for each of 3 methods with matched sensor locations.

Evaluation uses two metrics on the SAME held-out test frames:
  1. Reconstruction: same-timestep encode-decode (sensors_t -> x_t). All 3 methods.
  2. Forecast: autonomous rollout from z_0 at train boundary. sindy-rnn + sindy-shred only.

Data: NOAA Optimum Interpolation SST V2 (1992-2019)
  - 1,400 weekly snapshots, ~44,000 sea grid points
  - 250 random sensors (0.57% spatial coverage)

SINDy-SHRED reference (Gao et al.):
  3D linear ODE (poly_order=3, cubic terms pruned), reconstruction relative error: 2.01%
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
POLY_ORDER = 1          # appendix: poly_order=3 for E-SINDy (cubic terms get pruned to linear)
DT = 1 / 52             # weekly
LAGS = 52               # 1 year of sensor history
GRU_LAYERS = 2
DECODER_L1 = 350
DECODER_L2 = 400
DROPOUT = 0.1

# SINDy-SHRED (appendix SST settings)
SHRED_EPOCHS = 1000          # appendix: 1000 training epochs
SHRED_BATCH_SIZE = 128
SHRED_LR = 1e-3
SHRED_THRESHOLD = 1.0        # appendix: thresholds from 0.1 to 1.0
SHRED_PATIENCE = 5
SHRED_SINDY_REG = 10.0
SHRED_THRES_EPOCH = 100

# sindy-rnn
RNN_WINDOW = 52         # 1 year windows
RNN_ENSEMBLE = 11
RNN_EPOCHS = 10000
RNN_WARMUP = 500
RNN_LR = 1e-3
RNN_L1 = 0
RNN_PRUNE_THRESHOLD = 0.01
RNN_PRUNE_FREQ = 100
RNN_REFIT = 100
RNN_GRU_HIDDEN = None  # match SINDy-SHRED: GRU hidden_size = latent_dim

# METHODS = ['sindy-shred', 'shred', 'sindy-rnn']
METHODS = ['sindy-rnn']


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


def prepare_sindy_rnn_data(X_scaled, sensor_locs, window_size, train_end):
    """Create windowed training/test data for sindy-rnn."""
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

def reconstruct_sindy_shred(shred_obj, n_frames, full_dim):
    """SINDy-SHRED/SHRED same-timestep reconstruction: sensors_t -> z_t -> x_t.

    Returns (n_frames, full_dim) in raw space. NaN for frames without reconstruction.
    """
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
                    n_frames, full_dim, seed, sindy_reg=SHRED_SINDY_REG,
                    save_dir=None):
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
    recons = reconstruct_sindy_shred(shred, n_frames, full_dim)

    # Metric 2: Forecast (autonomous rollout) — only if we have a SINDy model
    forecast = np.full((n_frames, full_dim), np.nan)
    if sindy_reg > 0:
        forecast = forecast_sindy_shred(shred, n_frames, full_dim, train_end)

    # Save model
    if save_dir is not None:
        method_tag = 'shred' if sindy_reg == 0.0 else 'sindy_shred'
        path = os.path.join(save_dir, f'sst_{method_tag}_seed{seed}.pt')
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


def run_sindy_rnn(X, sensor_locs, train_end, full_dim, seed, save_dir=None):
    """Train sindy-rnn and return reconstruction + forecast (in raw space).

    Creates its own scaler internally — reconstruction and forecast must
    happen before the model/scaler are freed.
    """
    torch.manual_seed(seed)

    # Scale using training data only
    sc = MinMaxScaler()
    sc.fit(X[:train_end])
    X_scaled = sc.transform(X)

    sparse_train, full_next_train, sparse_test, full_next_test = \
        prepare_sindy_rnn_data(X_scaled, sensor_locs, RNN_WINDOW, train_end)

    model = SparseAutoencoderRNN(
        sparse_dim=NUM_SENSORS, full_dim=full_dim, latent_dim=LATENT_DIM,
        ensemble_size=RNN_ENSEMBLE, polynomial_degree=POLY_ORDER,
        dt=DT,
        # encoder_type='gru',
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
        equations = model.get_continuous_equations()
    except Exception:
        equations = "N/A"

    # Metric 1: Reconstruction (same-timestep, no dynamics)
    recons = reconstruct_sindy_rnn(model, X_scaled, sensor_locs, sc)

    # Metric 2: Forecast (autonomous rollout)
    forecast = forecast_sindy_rnn(model, X_scaled, sensor_locs, train_end, sc)

    # Save model
    if save_dir is not None:
        path = os.path.join(save_dir, f'sst_sindy_rnn_seed{seed}.pt')
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

SST_GRID_ROWS = 180
SST_GRID_COLS = 360


def generate_sst_images(X_raw, sst_locs, data_dict, train_end, seed, save_dir,
                        prefix='recon'):
    """Generate composite images for all methods at a given seed.

    Maps sea grid points back to a (180, 360) global grid for 2D visualization.
    Plots anomalies from the temporal mean rather than absolute temperatures,
    since absolute SST is dominated by the static climatological pattern and
    looks nearly identical across timesteps. Anomalies reveal seasonal cycles,
    interannual variability (El Nino/La Nina), and whether the model captures
    temporal evolution.

    Args:
        X_raw: (n_frames, n_sea_points) ground truth
        sst_locs: indices of sea grid points in the full (64800,) grid
        data_dict: dict mapping method_name -> (n_frames, n_sea_points) array
        train_end: frame index where test region begins
        seed: seed number (for filename)
        save_dir: directory to save images
        prefix: 'recon' for reconstruction, 'forecast' for autonomous rollout
    """
    n_frames = X_raw.shape[0]
    X_mean = np.nanmean(X_raw, axis=0)  # temporal mean per grid point

    def to_grid(vec):
        """Map sea-point vector to (180, 360) grid with NaN for land."""
        grid = np.full(SST_GRID_ROWS * SST_GRID_COLS, np.nan)
        grid[sst_locs] = vec
        return grid.reshape(SST_GRID_ROWS, SST_GRID_COLS)

    for method, data in data_dict.items():
        # Select frames spanning the full relevant range
        if prefix == 'forecast':
            # Always span the full test period so seasonal variation is visible,
            # even if the forecast diverges partway through
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

        # Anomalies from temporal mean
        gt_anom = X_raw[frame_indices] - X_mean[np.newaxis, :]
        pred_anom = data[frame_indices] - X_mean[np.newaxis, :]

        # Symmetric color limits from ground truth anomalies
        vlim = np.nanpercentile(np.abs(gt_anom), 98)
        vmin_val, vmax_val = -vlim, vlim

        row_labels = ['True anomaly',
                      'Recon. anomaly' if prefix == 'recon' else 'Forecast anomaly',
                      '|Error|']

        # Shared error colorbar limit (only from valid predictions)
        err_vals = np.abs(gt_anom - pred_anom)
        valid_err = err_vals[~np.isnan(err_vals)]
        err_max = np.nanpercentile(valid_err, 95) if len(valid_err) > 0 else 1.0

        for j, fidx in enumerate(frame_indices):
            region = "test" if fidx >= train_end else "train"
            has_pred = not np.isnan(data[fidx, 0])

            gt_grid = to_grid(gt_anom[j])

            # Row 0: True anomaly (always available)
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
                # Blank panel for diverged/unavailable forecast
                for row in [1, 2]:
                    axes[row, j].text(0.5, 0.5, 'N/A', transform=axes[row, j].transAxes,
                                      ha='center', va='center', fontsize=12, color='gray')

            for row in range(3):
                axes[row, j].set_xticks([])
                axes[row, j].set_yticks([])

        # Row labels on leftmost column
        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=10)

        fig.suptitle(f'{method} — SST {prefix} anomaly (seed {seed})', fontsize=12)
        fig.tight_layout()

        path = os.path.join(save_dir, f'sst_{prefix}_{method}_seed{seed}.png')
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
    print("Multi-Seed SST Benchmark")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {NUM_SEEDS}")
    print(f"Methods: {METHODS}")
    print(f"Architecture: GRU({NUM_SENSORS}->{LATENT_DIM}, {GRU_LAYERS}L) "
          f"+ Decoder({LATENT_DIM}->{DECODER_L1}->{DECODER_L2}->full)")
    print(f"poly_order={POLY_ORDER}, dt={DT:.4f}, lags={LAGS}")

    # Load data
    print("\nLoading SST data...")
    X, sst_locs = load_sst_data()
    n_time, full_dim = X.shape
    print(f"  Shape: ({n_time}, {full_dim})")

    # Train/val split (matching SINDy-SHRED paper)
    train_length = 1000
    validate_length = 30
    train_end = train_length + LAGS

    # Common test frames: after both methods' train+val regions
    # SINDy-SHRED val ends at frame (train_length + validate_length - 1) + lags - 1
    # = (1029) + 51 = 1080
    # sindy-rnn train ends at frame train_end = 1052
    # Use the later of the two as the start of test frames
    shred_val_end_frame = (train_length + validate_length - 1) + LAGS - 1
    test_start = max(train_end, shred_val_end_frame + 1)
    test_frames = np.arange(test_start, n_time)

    print(f"  Train: {train_length} samples, Val: {validate_length} samples")
    print(f"  Test frames: {test_start}-{n_time-1} ({len(test_frames)} frames)")

    # Create directories for saving models and images
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
                        X, sensor_locs, train_length, validate_length,
                        n_time, full_dim, seed, sindy_reg=SHRED_SINDY_REG,
                        save_dir=save_dir)
                elif method == 'shred':
                    recons, forecast, info = run_sindy_shred(
                        X, sensor_locs, train_length, validate_length,
                        n_time, full_dim, seed, sindy_reg=0.0,
                        save_dir=save_dir)
                elif method == 'sindy-rnn':
                    recons, forecast, info = run_sindy_rnn(
                        X, sensor_locs, train_end, full_dim, seed,
                        save_dir=save_dir)

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
                    'equations': 'FAILED',
                })

        # Generate images for this seed
        if seed_recons:
            print(f"\nGenerating images for seed {seed}...")
            generate_sst_images(X, sst_locs, seed_recons, train_end, seed, img_dir,
                                prefix='recon')
            generate_sst_images(X, sst_locs, seed_forecasts, train_end, seed, img_dir,
                                prefix='forecast')

            # Forecast MSE over time plot
            seed_fore_mse = {}
            for m in seed_forecasts:
                r = [x for x in all_results if x['method'] == m and x['seed'] == seed]
                if r and r[0].get('forecast_mse_per_step') is not None:
                    seed_fore_mse[m] = np.array(r[0]['forecast_mse_per_step'])
            if seed_fore_mse:
                plot_forecast_mse_over_time(seed_fore_mse, seed, img_dir, 'sst')

    # Save results
    with open('sst_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to sst_benchmark_results.json")

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

    # Print discovered equations
    print(f"\n\n{'='*70}")
    print("DISCOVERED EQUATIONS")
    print(f"{'='*70}")
    for r in all_results:
        eq = r.get('equations', 'N/A')
        if eq and eq != 'N/A' and eq != 'FAILED' and not eq.startswith('Failed'):
            print(f"\n--- {r['method']}, seed {r['seed']} ---")
            print(eq)

    # Select median seed images, delete the rest
    if NUM_SEEDS > 1:
        print(f"\nSelecting median-seed images...")
        select_median_and_cleanup(all_results, img_dir, 'sst', METHODS, NUM_SEEDS)

    print(f"\n\nReference: SINDy-SHRED relative error = 2.01% (Gao et al.)")
    print(f"{'='*70}")
    print("SST benchmark complete.")


if __name__ == '__main__':
    main()
