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


def pooled_shape(cfg):
    """(rows, cols) of a frame after config['data']['downsample'] block-mean
    pooling — the shape everything downstream (full_dim, to_image) actually
    uses, not the raw .npy resolution."""
    ds = cfg['data'].get('downsample', 1)
    return cfg['data']['frame_rows'] // ds, cfg['data']['frame_cols'] // ds


def load_data(cfg):
    """Load cylinder flow data, optionally downsample, subtract temporal
    mean, flatten.

    `downsample` (config['data'], default 1) block-mean-pools each frame by
    that factor per side before flattening — e.g. downsample=2 turns
    400x1000 into 200x500, a 4x reduction in full_dim. Meant for prototyping
    on smaller GPUs: the rollout decoder's memory/param count scales
    linearly with full_dim, and RolloutSINDyRNN's autonomous multi-step
    rollout needs the whole (batch, T, full_dim) target tensor resident at
    once, so full_dim is the single biggest lever on GPU memory.

    Returns (X, mean_frame): X is (n_frames, full_dim), mean_frame is the
    (rows, cols) temporal-mean frame that was subtracted (already
    downsampled, if applicable).
    """
    path = os.path.join(EXAMPLE_DIR, cfg['data']['path'])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cylinder data not found at {path}. "
            f"Place flow_over_cylinder.npy in examples/cylinder/data/")
    data = np.load(path)                     # (334, 400, 1000)

    ds = cfg['data'].get('downsample', 1)
    if ds > 1:
        n, rows, cols = data.shape
        data = data.reshape(n, rows // ds, ds, cols // ds, ds).mean(axis=(2, 4))

    mean_frame = data.mean(axis=0)           # temporal mean
    data = data - mean_frame                 # remove static background
    return data.reshape(data.shape[0], -1), mean_frame  # (334, full_dim)


def get_sensor_locs(cfg, full_dim):
    rng = np.random.default_rng(cfg['data']['sensor_seed'])
    return rng.choice(full_dim, size=cfg['data']['n_sensors'], replace=False)


def train_length(cfg, n_time):
    """train_length is inferred from the frame budget: warmup (T_w),
    validate_length, and test_length are all set directly in config;
    whatever's left goes to training.
    """
    dcfg = cfg['data']
    tl = n_time - dcfg['T_w'] - dcfg['validate_length'] - dcfg['test_length']
    if tl < 1:
        raise ValueError(
            f"T_w + validate_length + test_length "
            f"({dcfg['T_w'] + dcfg['validate_length'] + dcfg['test_length']}) "
            f"leaves no frames for training out of {n_time} total.")
    return tl


def train_test_split(cfg, n_time):
    """Return (train_length, train_end, test_frames).

    test_frames starts after the validation buffer (see validation_frames())
    so both methods are scored on an identical, non-overlapping held-out
    region — same convention as examples/sst/data.py.
    """
    dcfg = cfg['data']
    T_w = dcfg['T_w']
    tl = train_length(cfg, n_time)
    validate_length = dcfg['validate_length']
    test_length = dcfg['test_length']

    train_end = tl + T_w
    val_end = (tl + validate_length - 1) + T_w - 1
    test_start = val_end + 1
    test_frames = np.arange(test_start, min(test_start + test_length, n_time))
    return tl, train_end, test_frames


def validation_frames(cfg, n_time):
    """Return the validation buffer frames — held out of training but
    distinct from (and preceding) the test_frames returned by
    train_test_split(). Used for progress monitoring during training so the
    test set stays unseen until analyze.py.
    """
    dcfg = cfg['data']
    T_w = dcfg['T_w']
    tl = train_length(cfg, n_time)
    validate_length = dcfg['validate_length']

    train_end = tl + T_w
    val_end = min((tl + validate_length - 1) + T_w - 1 + 1, n_time)
    return np.arange(train_end, val_end)


def fit_scaler(X, train_end):
    scaler = MinMaxScaler()
    scaler.fit(X[:train_end])
    return scaler


def to_image(cfg, vec):
    """Flat full_dim vector -> (rows, cols) image for imshow."""
    return vec.reshape(*pooled_shape(cfg))
