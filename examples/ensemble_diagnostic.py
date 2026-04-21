"""Diagnostic: train E=10 ensemble on Lorenz, print per-member coefficients."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sindy_rnn import PolynomialRNN, fit
from sindy_rnn.pruning import _get_effective_coefficients_raw, _to_continuous

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def lorenz_rk4(x, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0):
    def f(x):
        return np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        ])
    k1 = f(x)
    k2 = f(x + dt/2 * k1)
    k3 = f(x + dt/2 * k2)
    k4 = f(x + dt * k3)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def generate_lorenz_data(n_steps=10000, dt=0.01, seed=42):
    rng = np.random.default_rng(seed)
    x = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        x = lorenz_rk4(x, dt=dt)
        trajectory.append(x.copy())
    return np.array(trajectory)


def chunk_trajectory(trajectory, window_size=100):
    N = len(trajectory) - 1
    n_windows = N // window_size
    xs_all = trajectory[:n_windows * window_size]
    ys_all = trajectory[1:n_windows * window_size + 1]
    xs = xs_all.reshape(n_windows, window_size, -1)
    ys = ys_all.reshape(n_windows, window_size, -1)
    return xs, ys


def main():
    DT = 0.01
    E = 10

    print(f"Device: {DEVICE}")
    print(f"Ensemble size: {E}")
    print(f"\nGenerating Lorenz data...")
    trajectory = generate_lorenz_data(n_steps=10000, dt=DT, seed=42)

    xs_np, ys_np = chunk_trajectory(trajectory, window_size=100)
    xs = torch.tensor(xs_np, dtype=torch.float32).to(DEVICE)
    ys = torch.tensor(ys_np, dtype=torch.float32).to(DEVICE)

    print(f"Data: {xs.shape[0]} windows of {xs.shape[1]} timesteps")

    torch.manual_seed(42)
    model = PolynomialRNN(
        n_states=3, n_controls=0, polynomial_degree=2,
        ensemble_size=E, state_names=['x', 'y', 'z'],
        dropout=0.1, feature_dropout=0.,
        compiled_forward=False,
    ).to(DEVICE)

    # Train WITH median pruning
    fit(model, xs, ys,
        epochs=1000,
        warmup_steps=500,
        learning_rate=1e-2,
        l2=2e-2,  # stronger L2 to push toward minimum-norm solution
        dt=DT,
        pruning_frequency=50,
        pruning_threshold=0.5,
        ensemble_pruning_alpha=0.5,  # min_fraction for median test
        pruning_method='median',
        verbose=True,
    )

    # Now examine per-member continuous-time coefficients
    print("\n\n" + "="*80)
    print("PER-MEMBER CONTINUOUS-TIME COEFFICIENTS (no pruning applied)")
    print("="*80)

    with torch.no_grad():
        model.eval()
        theta_eff = _get_effective_coefficients_raw(model)  # (E, n_states, n_terms)
        theta_cont = _to_continuous(theta_eff, model, DT)   # (E, n_states, n_terms)

    term_names = model.library_terms
    state_names = model.state_names

    # Expected Lorenz continuous-time:
    # dx/dt = -10*x + 10*y
    # dy/dt = 28*x - y - x*z
    # dz/dt = -2.667*z + x*y
    print("\nExpected:")
    print("  dx/dt = -10*x + 10*y")
    print("  dy/dt = 28*x - y - x*z")
    print("  dz/dt = -2.667*z + x*y")

    for state_idx, state_name in enumerate(state_names):
        print(f"\n{'─'*80}")
        print(f"  d{state_name}/dt coefficients per ensemble member")
        print(f"{'─'*80}")

        # Only show terms where at least one member has |coef| > 0.5
        coefs = theta_cont[:, state_idx, :]  # (E, n_terms)
        max_abs = coefs.abs().max(dim=0).values  # (n_terms,)
        active_terms = (max_abs > 0.5).nonzero(as_tuple=True)[0]

        if len(active_terms) == 0:
            print("  No terms > 0.5")
            continue

        # Header
        header = f"  {'Member':<8}"
        for t_idx in active_terms:
            header += f"{term_names[t_idx]:>10}"
        print(header)
        print("  " + "-" * (8 + 10 * len(active_terms)))

        # Per-member values
        for e in range(E):
            row = f"  {e:<8}"
            for t_idx in active_terms:
                val = coefs[e, t_idx].item()
                row += f"{val:>10.3f}"
            print(row)

        # Summary stats
        print("  " + "-" * (8 + 10 * len(active_terms)))
        row_mean = f"  {'mean':<8}"
        row_std = f"  {'std':<8}"
        for t_idx in active_terms:
            vals = coefs[:, t_idx]
            row_mean += f"{vals.mean().item():>10.3f}"
            row_std += f"{vals.std().item():>10.3f}"
        print(row_mean)
        print(row_std)

    # Compare pruning strategies
    from sindy_rnn.pruning import minimum_effect_ci_test, median_effect_test
    mask = model.coefficient_masks

    print("\n\n" + "="*80)
    print("CI TEST RESULTS (alpha=0.05, delta=0.1)")
    print("="*80)
    significant_ci = minimum_effect_ci_test(theta_cont, mask, alpha=0.05, delta=0.1)
    for state_idx, state_name in enumerate(state_names):
        surviving = significant_ci[state_idx].nonzero(as_tuple=True)[0]
        terms_str = [term_names[t.item()] for t in surviving]
        print(f"  d{state_name}/dt surviving terms: {terms_str}")

    print("\n" + "="*80)
    print("MEDIAN TEST RESULTS (delta=0.1)")
    print("="*80)
    significant_med = median_effect_test(theta_cont, mask, delta=0.1)
    for state_idx, state_name in enumerate(state_names):
        surviving = significant_med[state_idx].nonzero(as_tuple=True)[0]
        terms_str = [term_names[t.item()] for t in surviving]
        print(f"  d{state_name}/dt surviving terms: {terms_str}")

    print("\n" + "="*80)
    print("MEDIAN TEST RESULTS (delta=0.5)")
    print("="*80)
    significant_med2 = median_effect_test(theta_cont, mask, delta=0.5)
    for state_idx, state_name in enumerate(state_names):
        surviving = significant_med2[state_idx].nonzero(as_tuple=True)[0]
        terms_str = [term_names[t.item()] for t in surviving]
        print(f"  d{state_name}/dt surviving terms: {terms_str}")

    # Print correlation matrix of input features on training data
    print("\n\n" + "="*80)
    print("FEATURE CORRELATION ON TRAINING DATA")
    print("="*80)
    flat = xs.reshape(-1, 3).cpu().numpy()
    corr = np.corrcoef(flat.T)
    print(f"  {'':>6} {'x':>8} {'y':>8} {'z':>8}")
    for i, name in enumerate(['x', 'y', 'z']):
        print(f"  {name:>6} {corr[i,0]:>8.3f} {corr[i,1]:>8.3f} {corr[i,2]:>8.3f}")


if __name__ == '__main__':
    main()
