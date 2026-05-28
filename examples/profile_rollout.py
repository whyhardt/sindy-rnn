"""Profile one epoch of fit_rollout to find the bottleneck.

Instruments: data prep, encode, unfold, rollout loop, decode, loss, backward, optimizer.
Uses same config as sst_rollout.py.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

from sindy_rnn import RolloutSINDyRNN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Same config as sst_rollout.py
NUM_SENSORS = 250
LATENT_DIM = 3
DT = 1 / 52
TRAIN_LENGTH = 1000
LAGS = 52
POLY_DEGREE = 3
ENSEMBLE_SIZE = 11
NUM_EULER_STEPS = 3
GRU_LAYERS = 2
T_MAX = 104
BATCH_SIZE = 16
BATCHES_PER_EPOCH = 8
LAMBDA_0 = 1e-3
LAMBDA_S = 1e-2
GRAD_CLIP = 0.5
LR = 5e-3


def sync():
    if DEVICE == 'cuda':
        torch.cuda.synchronize()


def main():
    # Load data
    print("Loading SST data...")
    load_X = loadmat('data/SST_data.mat')['Z'].T
    mean_X = np.mean(load_X, axis=0)
    sst_locs = np.where(mean_X != 0)[0]
    X = load_X[:, sst_locs]
    n_time, full_dim = X.shape

    train_end = TRAIN_LENGTH + LAGS
    scaler = MinMaxScaler()
    scaler.fit(X[:train_end])
    X_scaled = scaler.transform(X)

    rng = np.random.default_rng(0)
    sensor_locs = rng.choice(full_dim, size=NUM_SENSORS, replace=False)

    x_sparse_train = torch.tensor(
        X_scaled[:train_end, sensor_locs], dtype=torch.float32)
    x_full_train = torch.tensor(
        X_scaled[:train_end], dtype=torch.float32)

    # Build model
    model = RolloutSINDyRNN(
        n_sensors=NUM_SENSORS, n_latent=LATENT_DIM, n_full=full_dim,
        ensemble_size=ENSEMBLE_SIZE, polynomial_degree=POLY_DEGREE,
        dt=DT, num_euler_steps=NUM_EULER_STEPS, dynamics_dropout=0.1,
        gru_layers=GRU_LAYERS, state_names=['z0', 'z1', 'z2'],
        decomposed=True,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    T_cur = T_MAX  # profile at full rollout length (worst case)
    span = LAGS + T_cur
    N_time = x_sparse_train.shape[0]
    max_start = N_time - span
    n_samples = min(max_start, BATCHES_PER_EPOCH * BATCH_SIZE)
    all_starts = torch.randperm(max_start)[:n_samples]

    print(f"\nProfiling 1 epoch: T_cur={T_cur}, {len(all_starts)} samples, "
          f"batch_size={BATCH_SIZE}")
    print(f"  full_dim={full_dim}, latent_dim={LATENT_DIM}, "
          f"poly_degree={POLY_DEGREE}, E={ENSEMBLE_SIZE}")
    print(f"  Device: {DEVICE}")
    print()

    # Warmup (first batch, discard timing)
    starts = all_starts[:BATCH_SIZE]
    warmup_slices = [x_sparse_train[s:s + LAGS] for s in starts]
    target_slices = [x_full_train[s + LAGS - 1:s + LAGS - 1 + T_cur + 1]
                     for s in starts]
    x_warmup = torch.stack(warmup_slices).to(DEVICE)
    x_targets = torch.stack(target_slices).to(DEVICE)
    x_hat, z_traj = model(x_warmup, T_cur)
    loss = F.mse_loss(x_hat, x_targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    sync()

    # Timed run: break down one full epoch
    timings = {
        'data_prep': 0., 'encode': 0., 'unfold_theta': 0.,
        'rollout_loop': 0., 'stack_traj': 0., 'decode': 0.,
        'loss_fwd': 0., 'backward': 0., 'optimizer_step': 0.,
        'epoch_total': 0.,
    }
    n_batches = 0

    sync()
    t_epoch = time.perf_counter()

    for bi in range(0, len(all_starts), BATCH_SIZE):
        starts = all_starts[bi:bi + BATCH_SIZE]
        B = len(starts)

        # --- Data prep ---
        sync()
        t0 = time.perf_counter()
        warmup_slices = [x_sparse_train[s:s + LAGS] for s in starts]
        target_slices = [x_full_train[s + LAGS - 1:s + LAGS - 1 + T_cur + 1]
                         for s in starts]
        x_warmup = torch.stack(warmup_slices).to(DEVICE)
        x_targets = torch.stack(target_slices).to(DEVICE)
        sync()
        timings['data_prep'] += time.perf_counter() - t0

        # --- Encode ---
        sync()
        t0 = time.perf_counter()
        z = model.encode(x_warmup)  # (E, B, n_latent)
        sync()
        timings['encode'] += time.perf_counter() - t0

        # --- Unfold theta ---
        sync()
        t0 = time.perf_counter()
        theta = model.dynamics.rnn.unfold_polynomial_coefficients()
        theta_masked = theta * model.dynamics.coefficient_masks.float()
        sync()
        timings['unfold_theta'] += time.perf_counter() - t0

        # --- Rollout loop ---
        sync()
        t0 = time.perf_counter()
        rnn = model.dynamics.rnn
        dt_sub = rnn._dt / rnn._num_euler_steps
        traj = [z]
        for _ in range(T_cur):
            for _ in range(rnn._num_euler_steps):
                z = z + dt_sub * rnn._evaluate_rhs_impl(z, None, theta_masked)
            traj.append(z)
        sync()
        timings['rollout_loop'] += time.perf_counter() - t0

        # --- Stack trajectory ---
        sync()
        t0 = time.perf_counter()
        z_stack = torch.stack(traj, dim=2)  # (E, B, T+1, n_latent)
        sync()
        timings['stack_traj'] += time.perf_counter() - t0

        # --- Decode ---
        sync()
        t0 = time.perf_counter()
        z_raw = model._denormalize_z(z_stack)
        x_hat = model.decoder(z_raw.mean(0))  # (B, T+1, n_full)
        sync()
        timings['decode'] += time.perf_counter() - t0

        # --- Loss ---
        sync()
        t0 = time.perf_counter()
        L_rec = F.mse_loss(x_hat, x_targets)
        L_z0 = LAMBDA_0 * (z_stack[:, :, 0] ** 2).mean()
        L_sp = LAMBDA_S * (theta * model.dynamics.coefficient_masks.float()).abs().mean()
        loss = L_rec + L_z0 + L_sp
        sync()
        timings['loss_fwd'] += time.perf_counter() - t0

        # --- Backward ---
        sync()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        sync()
        timings['backward'] += time.perf_counter() - t0

        # --- Optimizer step ---
        sync()
        t0 = time.perf_counter()
        optimizer.step()
        sync()
        timings['optimizer_step'] += time.perf_counter() - t0

        n_batches += 1

    sync()
    timings['epoch_total'] = time.perf_counter() - t_epoch

    # Report
    print(f"{'Component':<20} {'Total (ms)':>12} {'Per-batch (ms)':>14} {'%':>8}")
    print("-" * 58)
    for key in ['data_prep', 'encode', 'unfold_theta', 'rollout_loop',
                'stack_traj', 'decode', 'loss_fwd', 'backward',
                'optimizer_step']:
        total_ms = timings[key] * 1000
        per_batch_ms = total_ms / n_batches
        pct = 100 * timings[key] / timings['epoch_total']
        print(f"  {key:<18} {total_ms:>10.1f}ms {per_batch_ms:>12.1f}ms {pct:>7.1f}%")
    print("-" * 58)
    total_ms = timings['epoch_total'] * 1000
    print(f"  {'TOTAL':<18} {total_ms:>10.1f}ms {total_ms/n_batches:>12.1f}ms")
    print(f"\n  {n_batches} batches, B={BATCH_SIZE}, T_cur={T_cur}")
    print(f"  Rollout: {T_cur} steps x {NUM_EULER_STEPS} sub-steps "
          f"= {T_cur * NUM_EULER_STEPS} RHS evaluations per sample")


if __name__ == '__main__':
    main()
