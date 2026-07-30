"""Benchmark sindy-rnn vs SINDy-SHRED on cylinder flow.

Loads the trained estimators from params/ (run train_sindy_rnn.py and
train_sindy_shred.py first), evaluates reconstruction + forecast error
on the same held-out test frames per CLAUDE.md's shared evaluation protocol,
and writes figures + a metrics summary to results/.

Known asymmetry: both methods hold out the same validation buffer
(config.yaml's validate_length, preceding the true test_frames), but
SINDy-SHRED uses that span for early-stopping/best-checkpoint selection
during training (its `patience` kwarg), while fit_rollout() only logs
held-out loss without acting on it — sindy-rnn keeps whatever the final
training epoch produces. Not corrected here; treat as a caveat when
comparing results.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from examples._common.estimators import RolloutSINDyRNNEstimator, SindyShredEstimator
from examples._common.eval import evaluate_on_frames, forecast_mse_per_step
from examples._common.plotting import (
    plot_field_comparison, plot_latent_dynamics, plot_forecast_mse,
    plot_method_comparison,
)
from data import load_config, load_data, get_sensor_locs, train_test_split, fit_scaler, to_image, PARAMS_DIR, RESULTS_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _rescale(pred_scaled, scaler):
    """Inverse-transform a (N, full_dim) array with NaN rows preserved."""
    valid = ~np.isnan(pred_scaled[:, 0])
    out = np.full_like(pred_scaled, np.nan)
    if valid.any():
        out[valid] = scaler.inverse_transform(pred_scaled[valid])
    return out


def main():
    cfg = load_config()
    dcfg = cfg['data']

    print("Cylinder Flow — Analysis")
    print("=" * 60)

    X, mean_frame = load_data(cfg)
    n_time, full_dim = X.shape
    train_length, train_end, test_frames = train_test_split(cfg, n_time)
    scaler = fit_scaler(X, train_end)
    X_scaled = scaler.transform(X)
    sensor_locs = get_sensor_locs(cfg, full_dim)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics = {}

    # ── sindy-rnn ──
    rnn_path = os.path.join(PARAMS_DIR, 'sindy_rnn.pt')
    if os.path.exists(rnn_path):
        print("\nEvaluating sindy-rnn...")
        est = RolloutSINDyRNNEstimator.load(
            rnn_path, device=DEVICE, T_w=dcfg['T_w'],
            simulate=cfg['sindy_rnn'].get('simulate', 'mean'))

        recons = _rescale(est.predict(X_scaled[:, sensor_locs]), scaler)
        recon_mse, recon_rel, n_recon = evaluate_on_frames(recons, X, test_frames)

        warmup = X_scaled[train_end - dcfg['T_w']:train_end, sensor_locs]
        forecast_scaled = est.simulate(warmup, n_time - train_end)
        forecast = np.full((n_time, full_dim), np.nan)
        forecast[train_end:] = scaler.inverse_transform(forecast_scaled)
        fore_mse, fore_rel, n_fore = evaluate_on_frames(forecast, X, test_frames)

        member = est.model.best_member_idx.item() if est.simulate_mode == 'best' else None
        active = est.model.count_active_terms(member=member)
        print(f"  Reconstruction rel. error: {100 * recon_rel:.2f}% ({n_recon} frames)")
        print(f"  Forecast rel. error:       {100 * fore_rel:.2f}% ({n_fore} frames)")
        print(f"  Active terms: {sum(active.values())}"
              f"{' (best member)' if member is not None else ''}")

        metrics['sindy-rnn'] = {
            'recon_mse': float(recon_mse), 'recon_rel_error': float(recon_rel),
            'forecast_mse': float(fore_mse), 'forecast_rel_error': float(fore_rel),
            'n_active_terms': sum(active.values()),
            'equations': est.model.get_equations(member=member),
        }

        plot_field_comparison(X, recons, train_end,
                              os.path.join(RESULTS_DIR, 'sindy_rnn_recon.png'),
                              lambda v: to_image(cfg, v), 'sindy-rnn — cylinder', prefix='recon')
        plot_field_comparison(X, forecast, train_end,
                              os.path.join(RESULTS_DIR, 'sindy_rnn_forecast.png'),
                              lambda v: to_image(cfg, v), 'sindy-rnn — cylinder', prefix='forecast')
        plot_forecast_mse(forecast_mse_per_step(forecast, X, train_end),
                          os.path.join(RESULTS_DIR, 'sindy_rnn_forecast_mse.png'),
                          dt=dcfg['dt'], time_label='Forecast time (seconds)',
                          title='sindy-rnn — Forecast MSE over Time')
    else:
        print(f"\nSkipping sindy-rnn: {rnn_path} not found (run train_sindy_rnn.py first)")

    # ── SINDy-SHRED ──
    shred_path = os.path.join(PARAMS_DIR, 'sindy_shred.pt')
    if os.path.exists(shred_path):
        print("\nEvaluating SINDy-SHRED...")
        est = SindyShredEstimator.load(shred_path, device=DEVICE)

        recons = est.predict(X)
        recon_mse, recon_rel, n_recon = evaluate_on_frames(recons, X, test_frames)

        warmup = X[train_end - est.lags:train_end]
        forecast_raw = est.simulate(warmup, n_time - train_end)
        forecast = np.full((n_time, full_dim), np.nan)
        forecast[train_end:] = forecast_raw
        fore_mse, fore_rel, n_fore = evaluate_on_frames(forecast, X, test_frames)

        n_active = (int(np.sum(np.abs(est.sindy_model.coefficients()) > 1e-6))
                   if est.sindy_model is not None else 0)
        print(f"  Reconstruction rel. error: {100 * recon_rel:.2f}% ({n_recon} frames)")
        print(f"  Forecast rel. error:       {100 * fore_rel:.2f}% ({n_fore} frames)")
        print(f"  Active terms: {n_active}")

        metrics['sindy-shred'] = {
            'recon_mse': float(recon_mse), 'recon_rel_error': float(recon_rel),
            'forecast_mse': float(fore_mse), 'forecast_rel_error': float(fore_rel),
            'n_active_terms': n_active,
        }

        plot_field_comparison(X, recons, train_end,
                              os.path.join(RESULTS_DIR, 'sindy_shred_recon.png'),
                              lambda v: to_image(cfg, v), 'SINDy-SHRED — cylinder', prefix='recon')
        plot_field_comparison(X, forecast, train_end,
                              os.path.join(RESULTS_DIR, 'sindy_shred_forecast.png'),
                              lambda v: to_image(cfg, v), 'SINDy-SHRED — cylinder', prefix='forecast')
        plot_forecast_mse(forecast_mse_per_step(forecast, X, train_end),
                          os.path.join(RESULTS_DIR, 'sindy_shred_forecast_mse.png'),
                          dt=dcfg['dt'], time_label='Forecast time (seconds)',
                          title='SINDy-SHRED — Forecast MSE over Time')
    else:
        print(f"\nSkipping SINDy-SHRED: {shred_path} not found (run train_sindy_shred.py first)")

    # ── Comparison ──
    if len(metrics) > 1:
        plot_method_comparison(metrics, os.path.join(RESULTS_DIR, 'method_comparison.png'),
                               title='Cylinder Flow — Method Comparison')

    results_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {results_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method, m in metrics.items():
        print(f"  {method}: recon {100*m['recon_rel_error']:.2f}%  "
              f"forecast {100*m['forecast_rel_error']:.2f}%  "
              f"terms {m['n_active_terms']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
