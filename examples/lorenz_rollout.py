"""Lorenz validation for RolloutSINDyRNN.

Validates that the rollout architecture can recover Lorenz dynamics from
sparse (2 of 3 coordinates) observations:
  - Sparse sensors: x, z (dropping y)
  - Full field: x, y, z
  - GRU encodes warmup window -> z_0
  - Polynomial ODE rolls forward autonomously
  - Linear decoder reconstructs all 3 coordinates at every step

Success criteria:
  1. Low reconstruction error on train and test
  2. Stable autonomous rollout
  3. Correct number of active terms after pruning
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sindy_rnn.rollout import RolloutSINDyRNN, fit_rollout

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def lorenz_rk4(x, dt=0.01, sigma=10, rho=28, beta=8/3):
    def f(x):
        return np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ])
    k1 = f(x)
    k2 = f(x + dt / 2 * k1)
    k3 = f(x + dt / 2 * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def main():
    print("Lorenz Rollout Validation")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ── Generate Lorenz trajectory ──
    dt = 0.01
    N_total = 10001
    x = np.array([1., 1., 1.])
    trajectory = [x]
    for _ in range(N_total - 1):
        x = lorenz_rk4(x, dt=dt)
        trajectory.append(x)
    trajectory = np.array(trajectory)  # (10001, 3)

    # Normalize (z-score on training set)
    train_end = 8000
    mean = trajectory[:train_end].mean(axis=0)
    std = trajectory[:train_end].std(axis=0)
    trajectory_norm = (trajectory - mean) / std

    # Sparse sensors: x and z (indices 0, 2) — drop y
    sensor_idx = [0, 2]
    x_sparse_train = torch.tensor(
        trajectory_norm[:train_end, sensor_idx], dtype=torch.float32)
    x_full_train = torch.tensor(
        trajectory_norm[:train_end], dtype=torch.float32)
    x_sparse_test = torch.tensor(
        trajectory_norm[train_end:, sensor_idx], dtype=torch.float32)
    x_full_test = torch.tensor(
        trajectory_norm[train_end:], dtype=torch.float32)

    print(f"  Trajectory: {N_total} points, dt={dt}")
    print(f"  Train: {train_end}, Test: {N_total - train_end}")
    print(f"  Sensors: {len(sensor_idx)} of 3 (x, z)")
    print(f"  Data mean: {mean}, std: {std}")

    # ── Build model ──
    T_w = 20
    T_max = 50
    model = RolloutSINDyRNN(
        n_sensors=len(sensor_idx),
        n_latent=3,
        n_full=3,
        ensemble_size=1,
        polynomial_degree=2,
        dt=dt,
        num_euler_steps=3,
        gru_layers=1,
        state_names=['z0', 'z1', 'z2'],
        decomposed=True,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params} parameters")
    print(f"  T_w={T_w}, T_max={T_max}")

    # ── Train ──
    fit_rollout(
        model,
        x_sparse_train, x_full_train,
        T_w=T_w,
        T_max=T_max,
        T_start=1,
        delta_T=2,
        epochs=1000,
        batch_size=64,
        batches_per_epoch=16,
        learning_rate=5e-3,
        lambda_0=1e-3,
        lambda_s=5e-2,
        grad_clip=0.5,
        pruning_threshold=0.1,
        pruning_frequency=100,
        pruning_method='median',
        cosine_decay=False,
        x_sparse_test=x_sparse_test,
        x_full_test=x_full_test,
        verbose=True,
    )

    # ── Evaluate ──
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        # Reconstruction on test windows
        n_test_windows = 50
        test_span = T_w + T_max
        max_test_start = len(x_sparse_test) - test_span
        if max_test_start > 0:
            starts = torch.linspace(0, max_test_start - 1, n_test_windows).long()
            rec_errors = []
            for s in starts:
                warmup = x_sparse_test[s:s + T_w].unsqueeze(0).to(DEVICE)
                targets = x_full_test[s + T_w - 1:s + T_w - 1 + T_max + 1]
                x_hat, _ = model(warmup, T_max)
                rec_errors.append(
                    (x_hat[0].cpu() - targets).pow(2).mean().item())
            print(f"  Test reconstruction MSE: {np.mean(rec_errors):.6f}")

        # Autonomous forecast from last training window
        warmup = x_sparse_train[-T_w:].unsqueeze(0).to(DEVICE)
        n_forecast = min(500, len(x_full_test))
        x_hat_forecast, z_traj = model.forecast(warmup, n_forecast)
        x_hat_np = x_hat_forecast[0].cpu().numpy()

        # Un-normalize for comparison
        forecast_raw = x_hat_np * std + mean
        truth_raw = trajectory[train_end:train_end + n_forecast]

        forecast_mse = np.mean((forecast_raw - truth_raw) ** 2)
        forecast_rel = np.linalg.norm(forecast_raw - truth_raw) / np.linalg.norm(truth_raw)
        print(f"  Forecast MSE ({n_forecast} steps): {forecast_mse:.6f}")
        print(f"  Forecast rel. error: {100 * forecast_rel:.2f}%")

        # Check latent norm stability
        z_norms = z_traj.mean(0)[0].norm(dim=-1).cpu().numpy()
        print(f"  Latent norm: min={z_norms.min():.3f}, max={z_norms.max():.3f}, "
              f"final={z_norms[-1]:.3f}")

        # Active terms
        active = model.count_active_terms()
        print(f"  Active terms: {sum(active.values())} "
              f"({', '.join(f'{k}:{v}' for k, v in active.items())})")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(14, 8))

        # Left column: autonomous forecast vs truth
        labels = ['x', 'y', 'z']
        t_forecast = np.arange(n_forecast) * dt
        for i in range(3):
            ax = axes[i, 0]
            ax.plot(t_forecast, truth_raw[:, i], 'b-', alpha=0.7, label='Truth')
            ax.plot(t_forecast, forecast_raw[:, i], 'r--', alpha=0.7, label='Forecast')
            ax.set_ylabel(labels[i])
            if i == 0:
                ax.set_title('Autonomous Forecast')
                ax.legend(fontsize=8)
        axes[2, 0].set_xlabel('Time')

        # Right column: latent trajectory
        z_np = z_traj.mean(0)[0].cpu().numpy()
        for i in range(3):
            ax = axes[i, 1]
            ax.plot(z_np[:, i], 'g-', alpha=0.7)
            ax.set_ylabel(f'z{i}')
            if i == 0:
                ax.set_title('Latent Trajectory (forecast)')
        axes[2, 1].set_xlabel('Step')

        fig.tight_layout()
        save_path = 'lorenz_rollout_validation.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Saved plot: {save_path}")
    except ImportError:
        print("\n  matplotlib not available, skipping plots")


if __name__ == '__main__':
    main()
