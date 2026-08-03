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

from sindy_rnn.rollout import RolloutSINDyRNN, select_best_member_bic
from examples._common.estimators import RolloutSINDyRNNEstimator, resolve_member
from data import load_config, load_data, get_sensor_locs, train_test_split, validation_frames, fit_scaler, PARAMS_DIR

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
    val_frames = validation_frames(cfg, n_time)
    scaler = fit_scaler(X, train_end)
    X_scaled = scaler.transform(X)
    sensor_locs = get_sensor_locs(cfg, full_dim)

    if rcfg.get('T_max') is None:
        rcfg['T_max'] = len(test_frames)
        print(f"  T_max not set in config; defaulting to test_length ({rcfg['T_max']})")

    T_w = dcfg['T_w']
    x_sparse_train = X_scaled[:train_end, sensor_locs]
    x_full_train = X_scaled[:train_end]
    x_sparse_val = torch.tensor(X_scaled[train_end - T_w:val_frames[-1] + 1, sensor_locs], dtype=torch.float32)
    x_full_val = torch.tensor(X_scaled[train_end - T_w:val_frames[-1] + 1], dtype=torch.float32)

    print(f"  Train: {train_end} frames ({train_length} windows)")
    print(f"  Validation: {len(val_frames)} frames (monitored during training, frames {val_frames[0]}-{val_frames[-1]})")
    print(f"  Test: {len(test_frames)} frames (eval from frame {test_frames[0]}, held out until analyze.py)")
    print(f"  Sensors: {dcfg['n_sensors']} of {full_dim}")
    print(f"  Latent dim: {dcfg['n_latent']}, Poly degree: {rcfg['polynomial_degree']}")

    os.makedirs(PARAMS_DIR, exist_ok=True)
    save_path = os.path.join(PARAMS_DIR, rcfg['path_model'])

    est = RolloutSINDyRNNEstimator(
        device=DEVICE,
        n_full=full_dim,
        x_sparse_test=x_sparse_val,
        x_full_test=x_full_val,
        verbose=True,
        **dcfg,
        **rcfg,
    )

    if rcfg.get('epochs', 1) == 0:
        # Stage 1 skipped — reload the existing checkpoint and let fit()
        # run only whatever Stage 2.1/2.2 refit epochs are configured on top
        # of it (e.g. to re-run refit with new refit_* settings, or just
        # recompute best/bic selection with both refit stages at 0, without
        # repeating Stage 1's long training run).
        if not os.path.exists(save_path):
            raise FileNotFoundError(
                f"epochs=0 requires an existing checkpoint at {save_path} to reload")
        print(f"\n  epochs=0: reloading {save_path} (Stage 1 skipped)")
        est.model = RolloutSINDyRNN.load(save_path).to(DEVICE)

    t0 = time.time()
    est.fit(x_sparse_train, x_full_train)
    elapsed = time.time() - t0

    x_sparse_train_t = torch.tensor(x_sparse_train, dtype=torch.float32, device=DEVICE)
    x_full_train_t = torch.tensor(x_full_train, dtype=torch.float32, device=DEVICE)
    bic_scores, mse_scores = select_best_member_bic(est.model, x_sparse_train_t, x_full_train_t, T_w, rcfg['T_max'],
                                        batch_size=rcfg.get('batch_size', 32))

    member = resolve_member(est)
    print(f"\n  Discovered equations{' (best member)' if member is not None else ''}:")
    est.model.print_equations(member=member)
    active = est.model.count_active_terms(member=member)
    print(f"  Active terms: {sum(active.values())}")
    print(f"  Training time: {elapsed:.1f}s")

    if rcfg.get('pruning_method') == 'ladder':
        E = est.model.ensemble_size
        exp_step = rcfg.get('ladder_exponent_step', 0.2)
        offset = rcfg.get('ladder_offset', -1.0)
        thresholds = [rcfg['pruning_threshold'] * 10 ** (exp_step * e + offset) for e in range(E)]
        n_coefs = [sum(est.model.count_active_terms(member=e).values()) for e in range(E)]
        print("\n  Pruning ladder:")
        print("  member   |" + "".join(f"{e:>10d}" for e in range(E)))
        print("  threshold|" + "".join(f"{t:>10.2e}" for t in thresholds))
        print("  n_coef   |" + "".join(f"{n:>10d}" for n in n_coefs))
        if mse_scores is not None:
            print("  mse      |" + "".join(f"{m:>10.2e}" for m in mse_scores.tolist()))
        if bic_scores is not None:
            print("  bic      |" + "".join(f"{b:>10.1f}" for b in bic_scores.tolist()))

    est.save(save_path)
    print(f"\n  Saved model to {save_path}")


if __name__ == '__main__':
    main()
