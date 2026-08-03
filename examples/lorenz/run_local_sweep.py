"""Laptop-friendly driver for the Lorenz noise/data-length recovery sweep —
runs every grid cell sequentially in one process (no cluster array job, no
SLURM), for sindy-rnn (factored) and STLSQ/E-SINDy only. No rollout method.

Just calls recovery_run.run_cell() in a loop and writes the same per-cell
JSONs recovery_run.py would, so recovery_aggregate.py works unchanged
afterward. Safe to Ctrl-C and re-run: cells whose JSON already exists in
--out_dir are skipped.

Usage:
    python run_local_sweep.py
    python run_local_sweep.py --noise_fracs 0 0.01 0.05 0.1 0.2 \\
        --data_lengths 100 500 1000 5000 10000 --num_seeds 5
    python run_local_sweep.py --methods factored esindy --num_seeds 3
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data import RESULTS_DIR
from recovery_run import run_cell, METHODS as ALL_METHODS

DEFAULT_NOISE_FRACS = [0.0, 0.01, 0.05, 0.1, 0.2]
DEFAULT_DATA_LENGTHS = [100, 500, 1000, 5000, 10000]
DEFAULT_METHODS = ['factored', 'esindy']  # sindy-rnn + STLSQ; no 'direct', no rollout


def cell_path(out_dir, method, n_steps, noise_frac, seed):
    return os.path.join(out_dir, f"{method}_N{n_steps}_noise{noise_frac}_seed{seed}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--noise_fracs', type=float, nargs='+', default=DEFAULT_NOISE_FRACS)
    parser.add_argument('--data_lengths', type=int, nargs='+', default=DEFAULT_DATA_LENGTHS)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS,
                        choices=ALL_METHODS, help="'direct' and rollout are not included by default")
    parser.add_argument('--out_dir', default=os.path.join(RESULTS_DIR, 'recovery'))
    parser.add_argument('--overwrite', action='store_true',
                        help='re-run cells even if a result JSON already exists')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    grid = list(itertools.product(args.methods, args.data_lengths, args.noise_fracs, range(args.num_seeds)))
    todo = [(m, n, nf, s) for (m, n, nf, s) in grid
            if args.overwrite or not os.path.exists(cell_path(args.out_dir, m, n, nf, s))]

    print(f"Grid: {len(args.methods)} methods x {len(args.data_lengths)} data lengths x "
          f"{len(args.noise_fracs)} noise levels x {args.num_seeds} seeds = {len(grid)} cells "
          f"({len(grid) - len(todo)} already done, {len(todo)} to run)")

    t_start = time.time()
    failures = []
    for i, (method, n_steps, noise_frac, seed) in enumerate(todo):
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(todo) - i) if i > 0 else 0
        print(f"\n[{i+1}/{len(todo)}] method={method} N={n_steps} noise={noise_frac:.0%} seed={seed} "
              f"(elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m)")
        try:
            metrics = run_cell(noise_frac, n_steps, seed, method)
        except Exception as e:
            print(f"  FAILED: {e}")
            failures.append((method, n_steps, noise_frac, seed, str(e)))
            continue

        print(f"  coef_err={metrics['coef_error']:.4f} F1={metrics['f1']:.3f} "
              f"exact={metrics['exact_match']} forecast_mse={metrics['forecast_mse']:.2e} "
              f"terms={metrics['n_active']} time={metrics['time']:.1f}s")

        out_path = cell_path(args.out_dir, method, n_steps, noise_frac, seed)
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    print(f"\nDone in {(time.time() - t_start)/60:.1f} minutes. "
          f"{len(todo) - len(failures)}/{len(todo)} cells succeeded.")
    if failures:
        print(f"\n{len(failures)} cells FAILED:")
        for method, n_steps, noise_frac, seed, err in failures:
            print(f"  method={method} N={n_steps} noise={noise_frac:.0%} seed={seed}: {err}")

    print(f"\nResults in {args.out_dir} — run recovery_aggregate.py to summarize:")
    print(f"  python recovery_aggregate.py --results_dir {args.out_dir}")


if __name__ == '__main__':
    main()
