"""Shared reconstruction/forecast evaluation protocol.

Per CLAUDE.md: all methods are compared on the SAME held-out test frames
using MSE and relative error in scaled space, regardless of each method's
internal training objective.
"""
import numpy as np


def evaluate_on_frames(pred, X_raw, test_frames):
    """Compute MSE and relative error on a set of held-out frames.

    Args:
        pred: (n_frames, full_dim) — reconstruction or forecast array,
            NaN where unavailable.
        X_raw: (n_frames, full_dim) — ground truth.
        test_frames: array of frame indices to evaluate on.

    Returns:
        (mse, rel_error, n_valid_frames)
    """
    valid = ~np.isnan(pred[test_frames, 0])
    valid_frames = test_frames[valid]

    if len(valid_frames) == 0:
        return float('nan'), float('nan'), 0

    p = pred[valid_frames]
    g = X_raw[valid_frames]
    mse = np.mean((p - g) ** 2)
    rel = np.linalg.norm(p - g) / np.linalg.norm(g)
    return mse, rel, len(valid_frames)


def forecast_mse_per_step(forecast, X_raw, train_end):
    """Per-timestep MSE of an autonomous forecast.

    Args:
        forecast: (n_frames, full_dim) — NaN before train_end / after divergence.
        X_raw: (n_frames, full_dim) — ground truth.
        train_end: first forecast frame index.

    Returns:
        (n_forecast,) array, NaN where forecast is unavailable.
    """
    n_frames = X_raw.shape[0]
    n_forecast = n_frames - train_end
    mse_per_step = np.full(n_forecast, np.nan)

    for k in range(n_forecast):
        frame = train_end + k
        if not np.isnan(forecast[frame, 0]):
            mse_per_step[k] = np.mean((forecast[frame] - X_raw[frame]) ** 2)

    return mse_per_step
