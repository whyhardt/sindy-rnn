"""Benchmark sindy-rnn vs SINDy-SHRED on SST.

Loads the trained estimators from params/ (run train_sindy_rnn.py and
train_sindy_shred.py first), evaluates reconstruction + forecast error
on the same held-out test frames per CLAUDE.md's shared evaluation protocol,
and writes figures + a metrics summary to results/.

Known asymmetry: both methods hold out the same frame span (config.yaml's
train_length/validate_length), but SINDy-SHRED uses that span for
early-stopping/best-checkpoint selection during training (its `patience`
kwarg), while fit_rollout() only logs held-out loss without acting on it —
sindy-rnn keeps whatever the final training epoch produces. Not corrected
here; treat as a caveat when comparing results.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from examples._common.estimators import RolloutSINDyRNNEstimator, SindyShredEstimator, resolve_member
from examples._common.eval import evaluate_on_frames, forecast_mse_per_step
from examples._common.plotting import (
    plot_field_comparison, plot_forecast_mse, plot_method_comparison,
)
from data import load_config, load_data, get_sensor_locs, train_test_split, fit_scaler, make_to_image, PARAMS_DIR, RESULTS_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _rescale(pred_scaled, scaler):
    valid = ~np.isnan(pred_scaled[:, 0])
    out = np.full_like(pred_scaled, np.nan)
    if valid.any():
        out[valid] = scaler.inverse_transform(pred_scaled[valid])
    return out


def main():
    cfg = load_config()
    dcfg = cfg['data']

    print("SST — Analysis")
    print("=" * 60)

    X, sst_locs = load_data(cfg)
    n_time, full_dim = X.shape
    train_end, test_frames = train_test_split(cfg, n_time)
    scaler = fit_scaler(X, train_end)
    X_scaled = scaler.transform(X)
    sensor_locs = get_sensor_locs(cfg, full_dim)
    to_image = make_to_image(sst_locs)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics = {}

    # ── sindy-rnn ──
    rnn_path = os.path.join(PARAMS_DIR, cfg['sindy_rnn']['path_model'])
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

        member = resolve_member(est)
        active = est.model.count_active_terms(member=member)
        equations = est.get_equations()
        print(f"  Reconstruction rel. error: {100 * recon_rel:.2f}% ({n_recon} frames)")
        print(f"  Forecast rel. error:       {100 * fore_rel:.2f}% ({n_fore} frames)")
        print(f"  Active terms: {sum(active.values())}"
              f"{' (best member)' if member is not None else ''}")
        print(f"  Equations:\n{equations}")

        metrics['sindy-rnn'] = {
            'recon_mse': float(recon_mse), 'recon_rel_error': float(recon_rel),
            'forecast_mse': float(fore_mse), 'forecast_rel_error': float(fore_rel),
            'n_active_terms': sum(active.values()),
            'equations': equations,
        }

        plot_field_comparison(X, recons, train_end,
                              os.path.join(RESULTS_DIR, 'sindy_rnn_recon.png'),
                              to_image, 'sindy-rnn — SST', prefix='recon', anomaly=True)
        plot_field_comparison(X, forecast, train_end,
                              os.path.join(RESULTS_DIR, 'sindy_rnn_forecast.png'),
                              to_image, 'sindy-rnn — SST', prefix='forecast', anomaly=True)
        plot_forecast_mse(forecast_mse_per_step(forecast, X, train_end),
                          os.path.join(RESULTS_DIR, 'sindy_rnn_forecast_mse.png'),
                          dt=1.0, time_label='Forecast step (weeks)',
                          title='sindy-rnn — Forecast MSE over Time')
    else:
        print(f"\nSkipping sindy-rnn: {rnn_path} not found (run train_sindy_rnn.py first)")

    # ── SINDy-SHRED ──
    shred_path = os.path.join(PARAMS_DIR, cfg['sindy_shred']['path_model'])
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
        equations = est.get_equations()
        print(f"  Reconstruction rel. error: {100 * recon_rel:.2f}% ({n_recon} frames)")
        print(f"  Forecast rel. error:       {100 * fore_rel:.2f}% ({n_fore} frames)")
        print(f"  Active terms: {n_active}")
        print(f"  Equations:\n{equations}")

        metrics['sindy-shred'] = {
            'recon_mse': float(recon_mse), 'recon_rel_error': float(recon_rel),
            'forecast_mse': float(fore_mse), 'forecast_rel_error': float(fore_rel),
            'n_active_terms': n_active,
            'equations': equations,
        }

        plot_field_comparison(X, recons, train_end,
                              os.path.join(RESULTS_DIR, 'sindy_shred_recon.png'),
                              to_image, 'SINDy-SHRED — SST', prefix='recon', anomaly=True)
        plot_field_comparison(X, forecast, train_end,
                              os.path.join(RESULTS_DIR, 'sindy_shred_forecast.png'),
                              to_image, 'SINDy-SHRED — SST', prefix='forecast', anomaly=True)
        plot_forecast_mse(forecast_mse_per_step(forecast, X, train_end),
                          os.path.join(RESULTS_DIR, 'sindy_shred_forecast_mse.png'),
                          dt=1.0, time_label='Forecast step (weeks)',
                          title='SINDy-SHRED — Forecast MSE over Time')
    else:
        print(f"\nSkipping SINDy-SHRED: {shred_path} not found (run train_sindy_shred.py first)")

    # ── Comparison ──
    if len(metrics) > 1:
        plot_method_comparison(metrics, os.path.join(RESULTS_DIR, 'method_comparison.png'),
                               title='SST — Method Comparison')

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
