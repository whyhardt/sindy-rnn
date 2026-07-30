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
    return rng.choice(full_dim, size=cfg['data']['n_sensors'], replace=False)


def train_length(cfg, n_time):
    """train_length is inferred from the frame budget: warmup (T_w),
    validate_length, and test_length are all set directly in config;
    whatever's left goes to training.
    """
    dcfg = cfg['data']
    train_length = n_time - dcfg['T_w'] - dcfg['validate_length'] - dcfg['test_length']
    if train_length < 1:
        raise ValueError(
            f"T_w + validate_length + test_length "
            f"({dcfg['T_w'] + dcfg['validate_length'] + dcfg['test_length']}) "
            f"leaves no frames for training out of {n_time} total.")
    return train_length


def train_test_split(cfg, n_time):
    """Return (train_end, test_frames).

    test_frames starts after SINDy-SHRED's own validation buffer so both
    methods are scored on an identical, non-overlapping held-out region.
    """
    dcfg = cfg['data']
    T_w = dcfg['T_w']
    tr_length = train_length(cfg, n_time)
    validate_length = dcfg['validate_length']
    test_length = dcfg['test_length']

    train_end = tr_length + T_w
    shred_val_end = (tr_length + validate_length - 1) + T_w - 1
    test_start = shred_val_end + 1
    test_frames = np.arange(test_start, min(test_start + test_length, n_time))
    return train_end, test_frames


def validation_frames(cfg, n_time):
    """Return the validation buffer frames — held out of training but
    distinct from (and preceding) the test_frames returned by
    train_test_split(). Used for progress monitoring during training so the
    test set stays unseen until analyze.py.
    """
    dcfg = cfg['data']
    T_w = dcfg['T_w']
    tr_length = train_length(cfg, n_time)
    validate_length = dcfg['validate_length']

    train_end = tr_length + T_w
    shred_val_end = (tr_length + validate_length - 1) + T_w - 1
    val_end = min(shred_val_end + 1, n_time)
    return np.arange(train_end, val_end)


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
