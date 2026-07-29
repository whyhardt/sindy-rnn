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
            dropout=rcfg['dropout'],
            feature_dropout=rcfg['feature_dropout'],
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
            pruning_method='agreement',
            learning_rate=rcfg['learning_rate'],
            l2=rcfg['l2'],
            refit_epochs=rcfg['refit_epochs'],
            verbose=True,
        ),
    )

    t0 = time.time()
    est.fit(xs, ys)
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
