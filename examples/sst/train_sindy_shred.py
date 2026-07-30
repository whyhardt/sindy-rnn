"""Train SINDy-SHRED on SST and save the model to params/.

Uses the reference SINDy-SHRED implementation in sindy-shred/ (GRU encoder +
E_SINDy dynamics + MLP decoder), matching the config in CLAUDE.md, wrapped
in SindyShredEstimator for a save()/load()/predict()/simulate() interface
consistent with the sindy-rnn estimator. Run analyze.py afterward to
benchmark reconstruction/forecast error.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
np.math = math  # pysindy uses np.math.factorial

import torch

from examples._common.estimators import SindyShredEstimator
from data import load_config, load_data, get_sensor_locs, train_test_split, train_length, PARAMS_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    cfg = load_config()
    dcfg = cfg['data']
    scfg = cfg['sindy_shred']

    print("SST — train SINDy-SHRED")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    X, sst_locs = load_data(cfg)
    n_time, full_dim = X.shape
    train_end, test_frames = train_test_split(cfg, n_time)
    sensor_locs = get_sensor_locs(cfg, full_dim)

    est = SindyShredEstimator(
        sensor_locations=sensor_locs, dt=dcfg['dt'], lags=dcfg['T_w'],
        train_length=train_length(cfg, n_time), validate_length=dcfg['validate_length'],
        test_length=dcfg['test_length'],
        seed=dcfg['sensor_seed'], device=DEVICE,
        latent_dim=dcfg['n_latent'], poly_order=scfg['poly_order'],
        hidden_layers=scfg['gru_layers'], l1=scfg['decoder_l1'], l2=scfg['decoder_l2'],
        dropout=scfg['dropout'], batch_size=scfg['batch_size'],
        num_epochs=scfg['epochs'], lr=scfg['lr'],
        threshold=scfg['threshold'], patience=scfg['patience'],
        sindy_regularization=scfg['sindy_reg'], thres_epoch=scfg['thres_epoch'],
        verbose=True,
    )

    t0 = time.time()
    est.fit(X)
    elapsed = time.time() - t0

    n_params = sum(p.numel() for p in est.net.parameters())
    print(f"\n  Model: {n_params:,} parameters")
    print(f"  Training time: {elapsed:.1f}s")

    os.makedirs(PARAMS_DIR, exist_ok=True)
    save_path = os.path.join(PARAMS_DIR, 'sindy_shred.pt')
    est.save(save_path)
    print(f"\n  Saved model to {save_path}")


if __name__ == '__main__':
    main()
