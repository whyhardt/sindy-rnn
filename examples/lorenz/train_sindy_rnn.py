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

from examples._common.estimators import PolynomialRNNEstimator
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
