"""Train RolloutSINDyRNN on cylinder flow and save the model to params/.

Architecture:
  z_0 = GRU(sparse_sensors[0:T_w])      # encode 2-second warmup
  z_{t+1} = z_t + dt * P(z_t)            # autonomous polynomial ODE rollout
  x_hat_t = D(z_t)                       # MLP decoder at every step

Trained end-to-end with per-step reconstruction loss + rollout curriculum.
Run analyze.py afterward to benchmark reconstruction/forecast error.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from examples._common.estimators import RolloutSINDyRNNEstimator
from data import load_config, load_data, get_sensor_locs, train_test_split, fit_scaler, PARAMS_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    cfg = load_config()
    dcfg = cfg['data']
    rcfg = cfg['sindy_rnn']

    print("Cylinder Flow — train sindy-rnn")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    X, _ = load_data(cfg)
    n_time, full_dim = X.shape
    print(f"  Shape: ({n_time}, {full_dim})")

    train_length, train_end, test_frames = train_test_split(cfg, n_time)
    scaler = fit_scaler(X, train_end)
    X_scaled = scaler.transform(X)
    sensor_locs = get_sensor_locs(cfg, full_dim)

    lags = dcfg['lags']
    x_sparse_train = X_scaled[:train_end, sensor_locs]
    x_full_train = X_scaled[:train_end]
    x_sparse_test = torch.tensor(X_scaled[train_end - lags:, sensor_locs], dtype=torch.float32)
    x_full_test = torch.tensor(X_scaled[train_end - lags:], dtype=torch.float32)

    print(f"  Train: {train_end} frames ({train_length} windows)")
    print(f"  Test: {n_time - train_end} frames")
    print(f"  Sensors: {dcfg['num_sensors']} of {full_dim}")
    print(f"  Latent dim: {dcfg['latent_dim']}, Poly degree: {rcfg['poly_degree']}")

    est = RolloutSINDyRNNEstimator(
        device=DEVICE,
        model_kwargs=dict(
            n_sensors=dcfg['num_sensors'],
            n_latent=dcfg['latent_dim'],
            n_full=full_dim,
            ensemble_size=rcfg['ensemble_size'],
            polynomial_degree=rcfg['poly_degree'],
            dt=dcfg['dt'],
            num_euler_steps=rcfg['num_euler_steps'],
            dynamics_dropout=0.1,
            gru_layers=rcfg['gru_layers'],
            state_names=[f'z{i}' for i in range(dcfg['latent_dim'])],
            decomposed=True,
        ),
        fit_kwargs=dict(
            T_w=lags,
            T_max=rcfg['t_max'],
            T_start=rcfg['t_start'],
            delta_T=rcfg['delta_t'],
            epochs=rcfg['epochs'],
            batch_size=rcfg['batch_size'],
            batches_per_epoch=rcfg['batches_per_epoch'],
            learning_rate=rcfg['lr'],
            lambda_0=rcfg['lambda_0'],
            lambda_s=rcfg['lambda_s'],
            grad_clip=rcfg['grad_clip'],
            pruning_threshold=rcfg['pruning_threshold'],
            pruning_frequency=rcfg['pruning_frequency'],
            pruning_method='agreement',
            lr_patience=100,
            x_sparse_test=x_sparse_test,
            x_full_test=x_full_test,
            verbose=True,
        ),
    )

    t0 = time.time()
    est.fit(x_sparse_train, x_full_train)
    elapsed = time.time() - t0

    print("\n  Discovered equations:")
    est.model.print_equations()
    active = est.model.count_active_terms()
    print(f"  Active terms: {sum(active.values())}")
    print(f"  Training time: {elapsed:.1f}s")

    os.makedirs(PARAMS_DIR, exist_ok=True)
    save_path = os.path.join(PARAMS_DIR, 'sindy_rnn.pt')
    est.save(save_path)
    print(f"\n  Saved model to {save_path}")


if __name__ == '__main__':
    main()
