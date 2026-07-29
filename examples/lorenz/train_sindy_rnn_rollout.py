"""Train sindy-rnn on noisy full-state Lorenz via trajectory-matching
rollout (RolloutSINDyRNN with identity=True), instead of derivative matching.

Architecture (identity mode — no encoder/decoder, z IS the observed state):
    z_0 = x_noisy[t]                        # last observed (noisy) frame
    z_{t+1} = z_t + dt * P(z_t)              # autonomous polynomial ODE rollout
    x_hat_t = z_t                            # decoder is the identity

Trained with the same rollout-length curriculum + per-step noise injection
used for cylinder/SST. Trajectory matching against noisy observations
integrates out zero-mean noise over the rollout instead of amplifying it
through finite differences (what train_sindy_rnn.py's derivative-matching
fit() does) — see CLAUDE.md for the tradeoff. Compare against train_sindy_rnn.py
via analyze.py.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from examples._common.estimators import RolloutSINDyRNNEstimator
from data import load_config, generate_or_load_data, PARAMS_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    cfg = load_config()
    lcfg = cfg['lorenz']
    rcfg = cfg['sindy_rnn_rollout']

    print("Lorenz — train sindy-rnn (rollout / trajectory-matching)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"  noise_frac={lcfg['noise_frac']}, n_steps={lcfg['n_steps']}")

    data = generate_or_load_data(cfg)
    x_noisy = data['noisy_train']

    torch.manual_seed(lcfg['seed'])
    est = RolloutSINDyRNNEstimator(
        device=DEVICE,
        model_kwargs=dict(
            n_sensors=3, n_latent=3, n_full=3,
            ensemble_size=rcfg['ensemble_size'],
            polynomial_degree=rcfg['degree'],
            dt=lcfg['dt'],
            state_names=['x', 'y', 'z'],
            decomposed=True,
            identity=True,
        ),
        fit_kwargs=dict(
            T_w=1,                        # identity mode: no recurrent context needed
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
            pruning_method='median',
            rollout_noise=rcfg['rollout_noise'],
            verbose=True,
        ),
    )

    t0 = time.time()
    est.fit(x_noisy, x_noisy)
    elapsed = time.time() - t0

    n_params = sum(p.numel() for p in est.model.parameters())
    print(f"\n  Model: {n_params:,} parameters (dynamics only — identity encoder/decoder)")

    print("\n  Discovered equations:")
    est.model.print_equations()
    active = est.model.count_active_terms()
    print(f"  Active terms: {sum(active.values())}")
    print(f"  Training time: {elapsed:.1f}s")

    os.makedirs(PARAMS_DIR, exist_ok=True)
    save_path = os.path.join(PARAMS_DIR, 'sindy_rnn_rollout.pt')
    est.save(save_path)
    print(f"\n  Saved model to {save_path}")


if __name__ == '__main__':
    main()
