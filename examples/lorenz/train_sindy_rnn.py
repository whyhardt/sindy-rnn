"""Train sindy-rnn on a noisy, fully-observed Lorenz trajectory and save
the model to params/.

Unlike cylinder/SST, Lorenz is fully observed (no sparse sensors, no
encoder/decoder) — this matches CLAUDE.md's direct application of
PolynomialRNN + fit() via derivative matching. Run analyze.py afterward to
compare against STLSQ on coefficient recovery + forecast error.

Trains on states normalized by data.compute_scale() (per-state std, no
mean-centering — see data.py for why only scale and not centering is safe
here). The model itself, and its saved checkpoint, live entirely in this
normalized frame; analyze.py converts back to raw units for comparison
against the ground-truth ODE.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from sindy_rnn.polynomial_library import get_library_feature_names
from examples._common.estimators import PolynomialRNNEstimator
from data import (
    load_config, generate_or_load_data, chunk_trajectory, normalize,
    rescale_coefficients, PARAMS_DIR,
)

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
    scale = data['scale']
    print(f"  Normalization scale (x,y,z): {scale}")

    noisy_train_norm = normalize(data['noisy_train'], scale)
    xs, ys = chunk_trajectory(noisy_train_norm, rcfg['window_size'])

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

    print("\n  Discovered equations (normalized units, z' = z / scale):")
    est.model.print_equations()

    coefs = est.model.get_coefficients(aggregate=True)
    coef_matrix = torch.stack([coefs[n] for n in est.model.state_names]).cpu().numpy()
    coef_matrix_raw = rescale_coefficients(coef_matrix, 1 / scale, degree=rcfg['degree'])
    term_names = get_library_feature_names(est.model.state_names, rcfg['degree'])
    print("\n  Discovered equations (raw physical units):")
    for i, name in enumerate(est.model.state_names):
        terms = ' + '.join(f"{c:.3f}*{t}" for c, t in zip(coef_matrix_raw[i], term_names)
                           if abs(c) > 1e-9)
        print(f"  d{name}/dt = {terms}")

    active = est.model.count_active_terms()
    print(f"\n  Active terms: {sum(active.values())}")
    print(f"  Training time: {elapsed:.1f}s")

    os.makedirs(PARAMS_DIR, exist_ok=True)
    save_path = os.path.join(PARAMS_DIR, 'sindy_rnn.pt')
    est.save(save_path)
    print(f"\n  Saved model to {save_path}")


if __name__ == '__main__':
    main()
