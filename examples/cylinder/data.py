"""Data loading/prep for cylinder flow. Values come from config.yaml so
train_sindy_rnn.py, train_sindy_shred.py, and analyze.py stay in sync.
"""
import os

import numpy as np
import yaml
from sklearn.preprocessing import MinMaxScaler

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(EXAMPLE_DIR, 'params')
RESULTS_DIR = os.path.join(EXAMPLE_DIR, 'results')


def load_config():
    with open(os.path.join(EXAMPLE_DIR, 'config.yaml')) as f:
        return yaml.safe_load(f)


def load_data(cfg):
    """Load cylinder flow data, subtract temporal mean, flatten.

    Returns (X, mean_frame): X is (n_frames, full_dim), mean_frame is the
    (rows, cols) temporal-mean frame that was subtracted.
    """
    path = os.path.join(EXAMPLE_DIR, cfg['data']['path'])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cylinder data not found at {path}. "
            f"Place flow_over_cylinder.npy in examples/cylinder/data/")
    data = np.load(path)                     # (334, 400, 1000)
    mean_frame = data.mean(axis=0)           # temporal mean
    data = data - mean_frame                 # remove static background
    return data.reshape(data.shape[0], -1), mean_frame  # (334, 400000)


def get_sensor_locs(cfg, full_dim):
    rng = np.random.default_rng(cfg['data']['sensor_seed'])
    return rng.choice(full_dim, size=cfg['data']['num_sensors'], replace=False)


def train_test_split(cfg, n_time):
    """Return (train_length, train_end, test_frames)."""
    lags = cfg['data']['lags']
    train_length = n_time - lags - cfg['data']['test_holdout']
    train_end = train_length + lags
    test_frames = np.arange(train_end, n_time)
    return train_length, train_end, test_frames


def fit_scaler(X, train_end):
    scaler = MinMaxScaler()
    scaler.fit(X[:train_end])
    return scaler


def to_image(cfg, vec):
    """Flat full_dim vector -> (rows, cols) image for imshow."""
    return vec.reshape(cfg['data']['frame_rows'], cfg['data']['frame_cols'])
