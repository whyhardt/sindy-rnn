"""Train E-SINDy (ensemble STLSQ + bagging + median aggregation) on the same
noisy Lorenz trajectory sindy-rnn uses, and save coefficients to params/.

Wrapped in StlsqEstimator for the same fit()/predict()/simulate()/save()/
load() interface as the sindy-rnn estimators.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
np.math = math  # pysindy uses np.math.factorial

from examples._common.estimators import StlsqEstimator
from data import load_config, generate_or_load_data, PARAMS_DIR


def main():
    cfg = load_config()
    lcfg = cfg['lorenz']
    scfg = cfg['stlsq']

    print("Lorenz — train STLSQ (E-SINDy)")
    print("=" * 60)
    print(f"  noise_frac={lcfg['noise_frac']}, n_steps={lcfg['n_steps']}")

    data = generate_or_load_data(cfg)
    z = data['noisy_train']

    est = StlsqEstimator(threshold=scfg['threshold'], n_models=scfg['n_models'],
                         degree=2, dt=lcfg['dt'])
    est.fit(z)

    n_active = int(np.count_nonzero(est.coef_matrix))
    print(f"\n  Active terms: {n_active}")
    print(f"  Coefficients:\n{est.coef_matrix}")

    os.makedirs(PARAMS_DIR, exist_ok=True)
    save_path = os.path.join(PARAMS_DIR, 'stlsq.npz')
    est.save(save_path)
    print(f"\n  Saved coefficients to {save_path}")


if __name__ == '__main__':
    main()
