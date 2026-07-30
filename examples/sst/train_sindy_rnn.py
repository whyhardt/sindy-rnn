"""Train RolloutSINDyRNN on SST and save the model to params/.

Architecture:
  z_0 = GRU(sparse_sensors[0:T_w])      # encode 1-year warmup
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
from data import load_config, load_data, get_sensor_locs, train_test_split, validation_frames, fit_scaler, PARAMS_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    cfg = load_config()
    dcfg = cfg['data']
    rcfg = cfg['sindy_rnn']

    print("SST — train sindy-rnn")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    X, sst_locs = load_data(cfg)
    n_time, full_dim = X.shape
    print(f"  Shape: ({n_time}, {full_dim})")

    train_end, test_frames = train_test_split(cfg, n_time)
    val_frames = validation_frames(cfg, n_time)
    scaler = fit_scaler(X, train_end)
    X_scaled = scaler.transform(X)
    sensor_locs = get_sensor_locs(cfg, full_dim)

    if rcfg.get('T_max') is None:
        rcfg['T_max'] = len(test_frames)
        print(f"  T_max not set in config; defaulting to test length ({rcfg['T_max']})")

    T_w = dcfg['T_w']
    x_sparse_train = X_scaled[:train_end, sensor_locs]
    x_full_train = X_scaled[:train_end]
    x_sparse_val = torch.tensor(X_scaled[train_end - T_w:val_frames[-1] + 1, sensor_locs], dtype=torch.float32)
    x_full_val = torch.tensor(X_scaled[train_end - T_w:val_frames[-1] + 1], dtype=torch.float32)

    print(f"  Train: {train_end} frames")
    print(f"  Validation: {len(val_frames)} frames (monitored during training, frames {val_frames[0]}-{val_frames[-1]})")
    print(f"  Test: {len(test_frames)} frames (eval from frame {test_frames[0]}, held out until analyze.py)")
    print(f"  Sensors: {dcfg['n_sensors']} of {full_dim}")
    print(f"  Latent dim: {dcfg['n_latent']}, Poly degree: {rcfg['polynomial_degree']}")

    est = RolloutSINDyRNNEstimator(
        device=DEVICE,
        n_full=full_dim,
        x_sparse_test=x_sparse_val,
        x_full_test=x_full_val,
        verbose=True,
        **dcfg,
        **rcfg,
    )

    t0 = time.time()
    est.fit(x_sparse_train, x_full_train)
    elapsed = time.time() - t0

    member = est.model.best_member_idx.item() if est.simulate_mode == 'best' else None
    print(f"\n  Discovered equations{' (best member)' if member is not None else ''}:")
    est.model.print_equations(member=member)
    active = est.model.count_active_terms(member=member)
    print(f"  Active terms: {sum(active.values())}")
    print(f"  Training time: {elapsed:.1f}s")

    os.makedirs(PARAMS_DIR, exist_ok=True)
    save_path = os.path.join(PARAMS_DIR, 'sindy_rnn.pt')
    est.save(save_path)
    print(f"\n  Saved model to {save_path}")


if __name__ == '__main__':
    main()
