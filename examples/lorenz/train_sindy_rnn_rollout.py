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

from sindy_rnn.rollout import RolloutSINDyRNN, select_best_member_bic
from examples._common.estimators import RolloutSINDyRNNEstimator, resolve_member
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

    os.makedirs(PARAMS_DIR, exist_ok=True)
    save_path = os.path.join(PARAMS_DIR, rcfg['path_model'])

    torch.manual_seed(lcfg['seed'])
    est = RolloutSINDyRNNEstimator(
        device=DEVICE,
        n_sensors=3, n_latent=3, n_full=3,
        dt=lcfg['dt'],
        state_names=['x', 'y', 'z'],
        identity=True,
        T_w=1,                        # identity mode: no recurrent context needed
        verbose=True,
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
    est.fit(x_noisy, x_noisy)
    elapsed = time.time() - t0

    n_params = sum(p.numel() for p in est.model.parameters())
    print(f"\n  Model: {n_params:,} parameters (dynamics only — identity encoder/decoder)")

    x_noisy_t = torch.tensor(x_noisy, dtype=torch.float32, device=DEVICE)
    bic_scores, mse_scores = select_best_member_bic(
        est.model, x_noisy_t, x_noisy_t, est.T_w, rcfg['T_max'],
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
