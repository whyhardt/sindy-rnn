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

Trains on states normalized by data.compute_scale() (per-state std, no
mean-centering) — see data.py for why. This also makes lambda_0's z_0-norm
regularization meaningful again: in raw units z_0 is the literal observed
physical state (O(10-30)), so penalizing its norm fought the real data;
normalized, z_0 is O(1) and the regularization does what it's meant to.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from sindy_rnn.polynomial_library import get_library_feature_names
from examples._common.estimators import RolloutSINDyRNNEstimator
from data import load_config, generate_or_load_data, normalize, rescale_coefficients, PARAMS_DIR

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
    scale = data['scale']
    print(f"  Normalization scale (x,y,z): {scale}")
    x_noisy = normalize(data['noisy_train'], scale)

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
            pruning_method='agreement',
            agreement_frac=rcfg['agreement_frac'],
            rollout_noise=rcfg['rollout_noise'],
            refit_epochs=rcfg['refit_epochs'],
            refit_learning_rate=rcfg['refit_learning_rate'],
            lr_patience=rcfg['lr_patience'],
            lr_factor=rcfg['lr_factor'],
            min_lr=rcfg['min_lr'],
            verbose=True,
        ),
    )

    t0 = time.time()
    est.fit(x_noisy, x_noisy)
    elapsed = time.time() - t0

    n_params = sum(p.numel() for p in est.model.parameters())
    print(f"\n  Model: {n_params:,} parameters (dynamics only — identity encoder/decoder)")

    print("\n  Discovered equations (normalized units, z' = z / scale):")
    est.model.print_equations()

    coefs = est.model.get_coefficients(aggregate=True)
    coef_matrix = torch.stack([coefs[n] for n in est.model.dynamics.state_names]).cpu().numpy()
    coef_matrix_raw = rescale_coefficients(coef_matrix, 1 / scale, degree=rcfg['degree'])
    term_names = get_library_feature_names(est.model.dynamics.state_names, rcfg['degree'])
    print("\n  Discovered equations (raw physical units):")
    for i, name in enumerate(est.model.dynamics.state_names):
        terms = ' + '.join(f"{c:.3f}*{t}" for c, t in zip(coef_matrix_raw[i], term_names)
                           if abs(c) > 1e-9)
        print(f"  d{name}/dt = {terms}")

    active = est.model.count_active_terms()
    print(f"\n  Active terms: {sum(active.values())}")
    print(f"  Training time: {elapsed:.1f}s")

    os.makedirs(PARAMS_DIR, exist_ok=True)
    save_path = os.path.join(PARAMS_DIR, 'sindy_rnn_rollout.pt')
    est.save(save_path)
    print(f"\n  Saved model to {save_path}")


if __name__ == '__main__':
    main()
