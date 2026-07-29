"""Data loading/prep for SST. Values come from config.yaml so
train_sindy_rnn.py, train_sindy_shred.py, and analyze.py stay in sync.
"""
import os

import numpy as np
import yaml
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(EXAMPLE_DIR, 'params')
RESULTS_DIR = os.path.join(EXAMPLE_DIR, 'results')

SST_GRID_ROWS = 180
SST_GRID_COLS = 360


def load_config():
    with open(os.path.join(EXAMPLE_DIR, 'config.yaml')) as f:
        return yaml.safe_load(f)


def load_data(cfg):
    """Load SST data, filter to sea grid points.

    Returns (X, sst_locs): X is (n_frames, full_dim) of sea points only,
    sst_locs are their flat indices into the (180, 360) lat/lon grid.
    """
    path = os.path.join(EXAMPLE_DIR, cfg['data']['path'])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SST data not found at {path}. Download NOAA OI SST V2 and "
            f"save as examples/sst/data/SST_data.mat")
    load_X = loadmat(path)['Z'].T  # (1400, 64800)
    mean_X = np.mean(load_X, axis=0)
    sst_locs = np.where(mean_X != 0)[0]
    return load_X[:, sst_locs], sst_locs


def get_sensor_locs(cfg, full_dim):
    rng = np.random.default_rng(cfg['data']['sensor_seed'])
    return rng.choice(full_dim, size=cfg['data']['num_sensors'], replace=False)


def train_test_split(cfg, n_time):
    """Return (train_end, test_frames).

    test_frames starts after SINDy-SHRED's own validation buffer so both
    methods are scored on an identical, non-overlapping held-out region.
    """
    dcfg = cfg['data']
    lags = dcfg['lags']
    train_length = dcfg['train_length']
    validate_length = dcfg['validate_length']

    train_end = train_length + lags
    shred_val_end = (train_length + validate_length - 1) + lags - 1
    test_start = max(train_end, shred_val_end + 1)
    test_frames = np.arange(test_start, n_time)
    return train_end, test_frames


def fit_scaler(X, train_end):
    scaler = MinMaxScaler()
    scaler.fit(X[:train_end])
    return scaler


def make_to_image(sst_locs):
    """Return a to_image(vec) closure that scatters sea-point values back
    into the (180, 360) lat/lon grid for imshow.
    """
    def to_image(vec):
        grid = np.full(SST_GRID_ROWS * SST_GRID_COLS, np.nan)
        grid[sst_locs] = vec
        return grid.reshape(SST_GRID_ROWS, SST_GRID_COLS)
    return to_image
