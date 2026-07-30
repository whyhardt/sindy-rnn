"""Shared, dataset-parameterized plotting for the benchmark analysis scripts.

Field comparison and latent-dynamics plots are model-agnostic: they take
plain arrays plus a small dataset-specific `to_image` callback, so the same
functions render sindy-rnn, SINDy-SHRED, or any other method's output.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_field_comparison(X_raw, pred, train_end, save_path, to_image,
                          title, prefix='recon', n_frames_show=6,
                          anomaly=False):
    """Plot truth / prediction / |error| for a handful of frames.

    Args:
        X_raw: (n_frames, full_dim) ground truth, raw space.
        pred: (n_frames, full_dim) prediction, NaN where unavailable.
        train_end: first test-frame index.
        save_path: output PNG path.
        to_image: callable (flat_vector) -> 2D array for imshow.
        title: figure suptitle.
        prefix: 'recon' or 'forecast' — controls frame selection + labeling.
        n_frames_show: number of frames to display.
        anomaly: if True, subtract the temporal mean before plotting
            (useful when raw values are dominated by a large mean field,
            e.g. SST).
    """
    n_frames = X_raw.shape[0]

    if prefix == 'forecast':
        start, end = train_end, n_frames - 1
    else:
        valid_mask = ~np.isnan(pred[:, 0])
        if not valid_mask.any():
            return
        start = int(np.argmax(valid_mask))
        end = n_frames - 1 - int(np.argmax(valid_mask[::-1]))

    frame_indices = np.linspace(start, end,
                                min(n_frames_show, end - start + 1), dtype=int)
    n_cols = len(frame_indices)

    if anomaly:
        X_mean = np.nanmean(X_raw, axis=0)
        gt = X_raw[frame_indices] - X_mean[np.newaxis, :]
        pr = pred[frame_indices] - X_mean[np.newaxis, :]
    else:
        gt = X_raw[frame_indices]
        pr = pred[frame_indices]

    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 6.5))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    vlim = np.nanpercentile(np.abs(gt), 98)
    err_vals = np.abs(gt - pr)
    valid_err = err_vals[~np.isnan(err_vals)]
    err_max = np.nanpercentile(valid_err, 95) if len(valid_err) > 0 else vlim * 0.5

    label_mid = ('Recon.' if prefix == 'recon' else 'Forecast')
    if anomaly:
        label_mid += ' anomaly'

    for j, fidx in enumerate(frame_indices):
        region = "test" if fidx >= train_end else "train"
        has_pred = not np.isnan(pred[fidx, 0])

        axes[0, j].imshow(to_image(gt[j]), cmap='RdBu_r',
                          vmin=-vlim, vmax=vlim, aspect='auto', origin='upper')
        axes[0, j].set_title(f"t={fidx} ({region})", fontsize=8)

        if has_pred:
            axes[1, j].imshow(to_image(pr[j]), cmap='RdBu_r',
                              vmin=-vlim, vmax=vlim, aspect='auto', origin='upper')
            axes[2, j].imshow(to_image(np.abs(gt[j] - pr[j])), cmap='hot',
                              vmin=0, vmax=err_max, aspect='auto', origin='upper')
        else:
            for row in [1, 2]:
                axes[row, j].text(0.5, 0.5, 'N/A', transform=axes[row, j].transAxes,
                                  ha='center', va='center', fontsize=12, color='gray')

        for row in range(3):
            axes[row, j].set_xticks([])
            axes[row, j].set_yticks([])

    gt_label = 'True anomaly' if anomaly else 'Truth'
    for row, label in enumerate([gt_label, label_mid, '|Error|']):
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(f'{title} — {prefix}', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_latent_dynamics(z_encoder, z_rollout, train_end, save_path,
                         n_latent, title='Latent Dynamics'):
    """Plot encoder z alongside the autonomous rollout z, per latent dim."""
    fig, axes = plt.subplots(n_latent, 1, figsize=(10, 2.5 * n_latent),
                              sharex=True)
    if n_latent == 1:
        axes = [axes]

    frames = np.arange(len(z_encoder))
    rollout_frames = np.arange(train_end - 1, train_end - 1 + len(z_rollout))

    for d in range(n_latent):
        ax = axes[d]
        valid = ~np.isnan(z_encoder[:, d])
        ax.plot(frames[valid], z_encoder[valid, d], 'b-', alpha=0.7,
                linewidth=1, label='Encoder')
        ax.plot(rollout_frames, z_rollout[:, d], 'r--', alpha=0.7,
                linewidth=1.5, label='Autonomous rollout')
        ax.axvline(x=train_end, color='k', linestyle=':', alpha=0.5)
        ax.set_ylabel(f'z{d}')
        if d == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Frame')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_forecast_mse(mse_per_step, save_path, dt=1.0, time_label='Forecast step',
                      title='Autonomous Forecast MSE over Time', log_scale=True):
    """Plot per-timestep forecast MSE (single method)."""
    valid = ~np.isnan(mse_per_step)
    if not valid.any():
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(len(mse_per_step))
    t = steps * dt
    ax.plot(t[valid], mse_per_step[valid], linewidth=1.5)
    ax.set_xlabel(time_label)
    ax.set_ylabel('MSE')
    ax.set_title(title)
    if log_scale:
        ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_trajectory_grid(rows, save_path, state_names, title='Trajectory Comparison'):
    """Grid of truth-vs-simulated trajectories: one row per data source
    (method), one column per state — all sources in a single figure.

    Args:
        rows: dict {method_name: (true_traj, sim_traj)}. Each method
            supplies its own truth array too (not just sim) since it may be
            truncated differently per method (e.g. a stale cached ground
            truth longer than the current forecast horizon for one method
            but not another).
        save_path: output PNG path.
        state_names: list of per-state labels, e.g. ['x', 'y', 'z'].
        title: figure suptitle.
    """
    names = list(rows.keys())
    n_rows = len(names)
    n_cols = len(state_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.2 * n_rows),
                             sharex='col', squeeze=False)

    for r, name in enumerate(names):
        true_traj, sim_traj = rows[name]
        t_true = np.arange(len(true_traj))
        t_sim = np.arange(len(sim_traj))
        for c in range(n_cols):
            ax = axes[r][c]
            ax.plot(t_true, true_traj[:, c], 'k-', linewidth=1, label='Truth', alpha=0.7)
            ax.plot(t_sim, sim_traj[:, c], 'r--', linewidth=1.2, label='Simulated')
            if r == 0:
                ax.set_title(state_names[c])
            if r == n_rows - 1:
                ax.set_xlabel('Step')
            if c == 0:
                ax.set_ylabel(name)
            if r == 0 and c == n_cols - 1:
                ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_trajectory_comparison(true_traj, sim_traj, save_path, state_names,
                               title='Trajectory Comparison'):
    """Overlay ground truth vs a single method's autonomous simulation, per state.

    Args:
        true_traj: (n_steps, n_states) ground truth trajectory.
        sim_traj: (n_steps_sim, n_states) simulated trajectory (may be
            shorter than true_traj if it diverged and was truncated).
        save_path: output PNG path.
        state_names: list of per-state labels, e.g. ['x', 'y', 'z'].
    """
    n_states = len(state_names)
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.2 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    t_true = np.arange(len(true_traj))
    t_sim = np.arange(len(sim_traj))
    for d in range(n_states):
        axes[d].plot(t_true, true_traj[:, d], 'k-', linewidth=1, label='Truth', alpha=0.7)
        axes[d].plot(t_sim, sim_traj[:, d], 'r--', linewidth=1.2, label='Simulated')
        axes[d].set_ylabel(state_names[d])
        if d == 0:
            axes[d].legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Step')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_method_comparison(metrics_by_method, save_path,
                           title='Method Comparison'):
    """Bar chart comparing reconstruction/forecast relative error across methods.

    Args:
        metrics_by_method: dict {method_name: {'recon_rel_error': float,
            'forecast_rel_error': float}}
        save_path: output PNG path.
    """
    methods = list(metrics_by_method.keys())
    recon = [100 * metrics_by_method[m].get('recon_rel_error', np.nan) for m in methods]
    forecast = [100 * metrics_by_method[m].get('forecast_rel_error', np.nan) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(1.5 + 2 * len(methods), 4.5))
    ax.bar(x - width / 2, recon, width, label='Reconstruction')
    ax.bar(x + width / 2, forecast, width, label='Forecast')
    ax.set_ylabel('Relative error (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
