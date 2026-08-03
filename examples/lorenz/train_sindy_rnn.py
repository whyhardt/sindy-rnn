"""Train sindy-rnn on a noisy, fully-observed Lorenz trajectory and save
the model to params/.

Unlike cylinder/SST, Lorenz is fully observed (no sparse sensors, no
encoder/decoder) — this matches CLAUDE.md's direct application of
PolynomialRNN + fit() via derivative matching. Run analyze.py afterward to
compare against STLSQ on coefficient recovery + forecast error.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from examples._common.estimators import PolynomialRNNEstimator, resolve_member
from data import load_config, generate_or_load_data, chunk_trajectory, PARAMS_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    cfg = load_config()
    lcfg = cfg['lorenz']
    rcfg = cfg['sindy_rnn']

    print("Lorenz — train sindy-rnn")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"  noise_frac={lcfg['noise_frac']}, n_steps={lcfg['n_steps']}")

    data = generate_or_load_data(cfg)
    xs, ys = chunk_trajectory(data['noisy_train'], rcfg['window_size'])

    torch.manual_seed(lcfg['seed'])
    est = PolynomialRNNEstimator(
        device=DEVICE,
        model_kwargs=dict(
            n_states=3, n_controls=0,
            polynomial_degree=rcfg['degree'],
            ensemble_size=rcfg['ensemble_size'],
            dt=lcfg['dt'],
            state_names=['x', 'y', 'z'],
            compiled_forward=False,
            direct=False,
            decomposed=True,
        ),
        fit_kwargs=dict(
            epochs=rcfg['epochs'],
            warmup_steps=rcfg['warmup_steps'],
            agreement_frac=rcfg['agreement_frac'],
            pruning_threshold=rcfg['pruning_threshold'],
            pruning_frequency=rcfg['pruning_frequency'],
            pruning_method=rcfg.get('pruning_method', 'agreement'),
            ladder_exponent_step=rcfg.get('ladder_exponent_step', 0.2),
            ladder_offset=rcfg.get('ladder_offset', -1.0),
            learning_rate=rcfg['learning_rate'],
            lambda_s=rcfg['lambda_s'],
            weight_decay=rcfg.get('weight_decay', 0.0),
            refit_epochs=rcfg['refit_epochs'],
            lr_patience=rcfg.get('lr_patience', 0),
            lr_factor=rcfg.get('lr_factor', 0.5),
            min_lr=rcfg.get('min_lr', 1e-5),
            verbose=True,
        ),
    )

    t0 = time.time()
    est.fit(xs, ys)
    elapsed = time.time() - t0

    bic_scores, mse_scores = est.model.select_best_member_bic(
        torch.tensor(xs, dtype=torch.float32, device=DEVICE),
        torch.tensor(ys, dtype=torch.float32, device=DEVICE))

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
        print("  mse      |" + "".join(f"{m:>10.2e}" for m in mse_scores.tolist()))
        print("  bic      |" + "".join(f"{b:>10.1f}" for b in bic_scores.tolist()))

    os.makedirs(PARAMS_DIR, exist_ok=True)
    save_path = os.path.join(PARAMS_DIR, rcfg['path_model'])
    est.save(save_path)
    print(f"\n  Saved model to {save_path}")


if __name__ == '__main__':
    main()
