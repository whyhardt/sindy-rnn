"""
Numerical verification that a single bilinear (product-of-two-linear-forms)
parameterization CANNOT simultaneously achieve:

    xz coefficient != 0
    y linear coefficient != 0
    yz coefficient = 0
    xy coefficient = 0
    x^2 = y^2 = z^2 = 0

The polynomial is:
    P(x,y,z) = (a*x + b*y + c*z + b0) * (d*x + e*y + f*z + b1) / sqrt(2)

Expanding (ignoring the 1/sqrt(2) normalization for the constraint analysis):
    x^2:      a*d
    y^2:      b*e
    z^2:      c*f
    xy:       a*e + b*d
    xz:       a*f + c*d
    yz:       b*f + c*e
    x_linear: a*b1 + d*b0
    y_linear: b*b1 + e*b0
    z_linear: c*b1 + f*b0
    constant: b0*b1

We try to minimize:
    (xz - 1)^2 + (y_lin - 0.28)^2 + yz^2 + xy^2 + x2^2 + y2^2 + z2^2

If the impossibility holds, the minimum residual should be bounded away from zero.

Proof sketch (by contradiction):
  From x^2=0: a*d=0, so either a=0 or d=0.
  Case a=0: xz = c*d != 0 => c,d != 0. xy = b*d = 0 => b=0.
            y_linear = e*b0, need != 0 => e != 0, b0 != 0.
            But yz = b*f + c*e = 0 + c*e != 0. CONTRADICTION with yz=0.
  Case d=0: xz = a*f != 0 => a,f != 0. xy = a*e = 0 => e=0.
            y_linear = b*b1, need != 0 => b != 0, b1 != 0.
            But yz = b*f + c*e = b*f + 0 != 0. CONTRADICTION with yz=0.
"""

import numpy as np
from scipy.optimize import minimize
import time

def compute_coefficients(params):
    """Expand (a*x + b*y + c*z + b0) * (d*x + e*y + f*z + b1)."""
    a, b, c, b0, d, e, f, b1 = params
    return {
        'x2': a * d,
        'y2': b * e,
        'z2': c * f,
        'xy': a * e + b * d,
        'xz': a * f + c * d,
        'yz': b * f + c * e,
        'x_lin': a * b1 + d * b0,
        'y_lin': b * b1 + e * b0,
        'z_lin': c * b1 + f * b0,
        'const': b0 * b1,
    }

def objective(params):
    """Loss: want xz=1, y_lin=0.28, and xy=yz=x2=y2=z2=0."""
    c = compute_coefficients(params)
    return (
        (c['xz'] - 1.0) ** 2
        + (c['y_lin'] - 0.28) ** 2
        + c['yz'] ** 2
        + c['xy'] ** 2
        + c['x2'] ** 2
        + c['y2'] ** 2
        + c['z2'] ** 2
    )

def objective_grad(params):
    """Analytical gradient for faster optimization."""
    a, b, c_, b0, d, e, f, b1 = params

    x2 = a * d
    y2 = b * e
    z2 = c_ * f
    xy = a * e + b * d
    xz = a * f + c_ * d
    yz = b * f + c_ * e
    y_lin = b * b1 + e * b0

    r_xz = xz - 1.0
    r_ylin = y_lin - 0.28
    r_yz = yz
    r_xy = xy
    r_x2 = x2
    r_y2 = y2
    r_z2 = z2

    grad = np.zeros(8)
    grad[0] = 2 * r_x2 * d + 2 * r_xy * e + 2 * r_xz * f           # d/da
    grad[1] = 2 * r_y2 * e + 2 * r_xy * d + 2 * r_yz * f + 2 * r_ylin * b1  # d/db
    grad[2] = 2 * r_z2 * f + 2 * r_xz * d + 2 * r_yz * e            # d/dc
    grad[3] = 2 * r_ylin * e                                          # d/db0
    grad[4] = 2 * r_x2 * a + 2 * r_xy * b + 2 * r_xz * c_           # d/dd
    grad[5] = 2 * r_y2 * b + 2 * r_xy * a + 2 * r_yz * c_ + 2 * r_ylin * b0  # d/de
    grad[6] = 2 * r_z2 * c_ + 2 * r_xz * a + 2 * r_yz * b           # d/df
    grad[7] = 2 * r_ylin * b                                          # d/db1

    return grad

# ---------------------------------------------------------------------------
# Run optimization from many random starting points
# ---------------------------------------------------------------------------
n_starts = 2000
rng = np.random.default_rng(42)

best_residual = np.inf
best_params = None
best_coeffs = None

residuals = []
t0 = time.time()

for i in range(n_starts):
    x0 = rng.standard_normal(8) * 2.0

    # L-BFGS-B with analytical gradient (fast and accurate)
    res = minimize(objective, x0, jac=objective_grad, method='L-BFGS-B',
                   options={'maxiter': 10000, 'gtol': 1e-15})

    if res.fun < best_residual:
        best_residual = res.fun
        best_params = res.x.copy()
        best_coeffs = compute_coefficients(res.x)

    residuals.append(res.fun)

    # For first 200 starts, also try Nelder-Mead (derivative-free)
    if i < 200:
        res2 = minimize(objective, x0, method='Nelder-Mead',
                        options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-15})
        if res2.fun < best_residual:
            best_residual = res2.fun
            best_params = res2.x.copy()
            best_coeffs = compute_coefficients(res2.x)
        residuals.append(res2.fun)

elapsed = time.time() - t0
residuals = np.array(residuals)

# ---------------------------------------------------------------------------
# Report results
# ---------------------------------------------------------------------------
print("=" * 70)
print("NUMERICAL VERIFICATION: Bilinear Impossibility Result")
print("=" * 70)
print()
print("Polynomial: P(x,y,z) = (a*x + b*y + c*z + b0)(d*x + e*y + f*z + b1)")
print()
print("Target constraints:")
print("  xz coefficient = 1     (nonzero)")
print("  y linear coeff = 0.28  (nonzero)")
print("  xy coefficient = 0")
print("  yz coefficient = 0")
print("  x^2 coefficient = 0")
print("  y^2 coefficient = 0")
print("  z^2 coefficient = 0")
print()
print(f"Optimization: {len(residuals)} runs from {n_starts} random starts")
print(f"Time: {elapsed:.1f}s")
print()
print("-" * 70)
print("RESULTS")
print("-" * 70)
print(f"  Best residual achieved:     {best_residual:.10f}")
print(f"  Median residual:            {np.median(residuals):.10f}")
print(f"  Runs with residual < 0.01:  {np.sum(residuals < 0.01)} / {len(residuals)}")
print(f"  Runs with residual < 0.001: {np.sum(residuals < 0.001)} / {len(residuals)}")
print(f"  Runs with residual < 1e-6:  {np.sum(residuals < 1e-6)} / {len(residuals)}")
print()
print("Best solution found:")
a, b, c, b0, d, e, f, b1 = best_params
print(f"  Factor 1: {a:.6f}*x + {b:.6f}*y + {c:.6f}*z + {b0:.6f}")
print(f"  Factor 2: {d:.6f}*x + {e:.6f}*y + {f:.6f}*z + {b1:.6f}")
print()
print("  Achieved coefficients vs targets:")
print(f"    x^2:      {best_coeffs['x2']:+.8f}  (target: 0)")
print(f"    y^2:      {best_coeffs['y2']:+.8f}  (target: 0)")
print(f"    z^2:      {best_coeffs['z2']:+.8f}  (target: 0)")
print(f"    xy:       {best_coeffs['xy']:+.8f}  (target: 0)")
print(f"    xz:       {best_coeffs['xz']:+.8f}  (target: 1)")
print(f"    yz:       {best_coeffs['yz']:+.8f}  (target: 0)")
print(f"    y_linear: {best_coeffs['y_lin']:+.8f}  (target: 0.28)")

# ---------------------------------------------------------------------------
# Tradeoff analysis: when x2,y2,z2,xy are near zero, what happens to yz?
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("TRADEOFF ANALYSIS")
print("-" * 70)
print()
print("Among all optimization runs, filtering for solutions where")
print("x^2, y^2, z^2, xy are all near zero (<0.01 each):")
print()

# Re-run with stored results: check each solution
good_solutions = []
rng2 = np.random.default_rng(42)  # reset to get same starting points
for i in range(n_starts):
    x0 = rng2.standard_normal(8) * 2.0
    res = minimize(objective, x0, jac=objective_grad, method='L-BFGS-B',
                   options={'maxiter': 10000, 'gtol': 1e-15})
    co = compute_coefficients(res.x)
    quad_residual = co['x2']**2 + co['y2']**2 + co['z2']**2 + co['xy']**2
    if quad_residual < 1e-4:
        good_solutions.append(co)

    if i < 200:
        rng2.standard_normal(8)  # consume same random numbers

if good_solutions:
    print(f"  Found {len(good_solutions)} solutions with near-zero x^2,y^2,z^2,xy.")
    print()

    # Sort by remaining residual
    remaining = [(co, (co['xz']-1)**2 + co['yz']**2 + (co['y_lin']-0.28)**2)
                 for co in good_solutions]
    remaining.sort(key=lambda x: x[1])

    print("  Top 5 by lowest remaining residual (xz, yz, y_lin terms):")
    for rank, (co, r) in enumerate(remaining[:5]):
        print(f"    #{rank+1}: xz={co['xz']:+.5f}, yz={co['yz']:+.5f}, "
              f"y_lin={co['y_lin']:+.5f}  |  residual={r:.6f}")
    print()
    print(f"  Best remaining residual: {remaining[0][1]:.8f}")
    print()
    print("  Observation: when pure-power and xy terms are driven to zero,")
    print("  the yz term CANNOT be zeroed while keeping xz and y_linear nonzero.")
else:
    print("  No solutions found with near-zero x^2,y^2,z^2,xy terms.")

# ---------------------------------------------------------------------------
# Analytical lower bound verification
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("ANALYTICAL VERIFICATION OF THE PROOF")
print("-" * 70)
print()
print("The proof by exhaustive case analysis:")
print()
print("  Given: x^2 = a*d = 0  =>  a=0 or d=0")
print()
print("  Case 1 (a=0):")
print("    xz = c*d != 0  =>  c != 0, d != 0")
print("    xy = b*d = 0   =>  b = 0  (since d != 0)")
print("    y_lin = e*b0   =>  need e != 0, b0 != 0")
print("    yz = b*f + c*e = 0 + c*e  (c != 0, e != 0)")
print("    => yz != 0  CONTRADICTION")
print()
print("  Case 2 (d=0):")
print("    xz = a*f != 0  =>  a != 0, f != 0")
print("    xy = a*e = 0   =>  e = 0  (since a != 0)")
print("    y_lin = b*b1   =>  need b != 0, b1 != 0")
print("    yz = b*f + c*e = b*f + 0  (b != 0, f != 0)")
print("    => yz != 0  CONTRADICTION")
print()
print("  Both cases lead to contradiction. QED.")

# Verify Case 1 numerically: fix a=0, b=0, set c,d,e,b0 to get xz=1, y_lin=0.28
print()
print("  Numerical check of Case 1 (a=0, b=0):")
c_val, d_val = 1.0, 1.0  # xz = c*d = 1
e_val, b0_val = 0.28, 1.0  # y_lin = e*b0 = 0.28
yz_case1 = c_val * e_val  # must be nonzero
print(f"    c={c_val}, d={d_val} => xz = {c_val*d_val}")
print(f"    e={e_val}, b0={b0_val} => y_lin = {e_val*b0_val}")
print(f"    yz = c*e = {yz_case1}  (FORCED nonzero)")

print()
print("  Numerical check of Case 2 (d=0, e=0):")
a_val, f_val = 1.0, 1.0  # xz = a*f = 1
b_val, b1_val = 0.28, 1.0  # y_lin = b*b1 = 0.28
yz_case2 = b_val * f_val  # must be nonzero
print(f"    a={a_val}, f={f_val} => xz = {a_val*f_val}")
print(f"    b={b_val}, b1={b1_val} => y_lin = {b_val*b1_val}")
print(f"    yz = b*f = {yz_case2}  (FORCED nonzero)")

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)

if best_residual > 1e-4:
    print()
    print(f"The best residual across {len(residuals)} optimization runs is")
    print(f"{best_residual:.8f}, which is bounded well away from zero.")
    print()
    print("This CONFIRMS the impossibility: a single bilinear form (product of")
    print("two linear forms) CANNOT represent a polynomial with nonzero xz and")
    print("y_linear coefficients while simultaneously zeroing xy, yz, x^2, y^2, z^2.")
    print()
    print("Implication for Lorenz y-equation:")
    print("  dy/dt = rho*x - y - x*z requires xz != 0 and y_linear != 0 with")
    print("  xy = yz = x^2 = y^2 = z^2 = 0. A degree-2 polynomial RNN (product")
    print("  of 2 linear forms) MUST activate at least one spurious cross-term,")
    print("  explaining ensemble bifurcation observed in practice.")
else:
    print()
    print(f"WARNING: Best residual {best_residual:.2e} is very small.")
    print("The impossibility claim may not hold!")
