"""Equation extraction and printing utilities.

With forward Euler h[t+1] = h[t] + dt * P(h[t]), the polynomial coefficients
directly represent the ODE right-hand side dh/dt = P(h). No conversion needed.
"""

from typing import Dict, Optional

import torch
from torch import Tensor


def get_coefficients(model, aggregate: bool = True) -> Dict[str, Tensor]:
    """Return ODE coefficients for each state dimension.

    Coefficients directly represent the ODE dh/dt = P(h).
    Pruned terms are masked to zero.

    Returns:
        dict mapping state_name -> Tensor
            aggregate=True:  (n_terms,) ensemble mean (NaN-aware, pruned->0)
            aggregate=False: (E, n_terms) per-member
    """
    theta = model.rnn.unfold_ode_coefficients().detach()  # (E, n_states, n_terms)
    mask = model.coefficient_masks.float()  # (E, n_states, n_terms)

    # Apply mask — theta * mask gives active ODE coefficients
    c = theta * mask  # (E, n_states, n_terms)

    results = {}
    for i, name in enumerate(model.state_names):
        c_i = c[:, i, :]      # (E, n_terms)
        mask_i = mask[:, i, :]  # (E, n_terms)

        if aggregate:
            c_agg = c_i.clone()
            # Mark pruned terms as NaN for nanmean
            c_agg = torch.where(
                mask_i == 0,
                torch.full_like(c_agg, float('nan')),
                c_agg,
            )
            c_i = torch.nanmean(c_agg, dim=0)
            c_i = torch.nan_to_num(c_i, nan=0.0)

        results[name] = c_i
    return results


def get_equations(model) -> str:
    """Return discovered ODE equations as a formatted multi-line string.

    Coefficients directly represent dh/dt = P(h).

    Example output:
        dx/dt = -10.000*x + 10.000*y
        dy/dt = 28.000*x - 1.000*y - 1.000*x*z
        dz/dt = -2.667*z + 1.000*x*y
    """
    coefs = get_coefficients(model, aggregate=True)
    term_names = model.library_terms
    lines = []

    for i, state_name in enumerate(model.state_names):
        c = coefs[state_name]  # (n_terms,) — ODE coefficients
        parts = []

        for j, term_name in enumerate(term_names):
            val = c[j].item()
            if abs(val) < 1e-6:
                continue
            sign = '+' if val >= 0 else '-'
            coef_str = f"{abs(val):.3f}"
            if term_name == '1':
                label = coef_str
            else:
                label = f"{coef_str}*{term_name}"
            parts.append((sign, label))

        if not parts:
            rhs = '0'
        else:
            sign0, label0 = parts[0]
            rhs = (f"-{label0}" if sign0 == '-' else label0)
            for sign, label in parts[1:]:
                rhs += f" {sign} {label}"

        lines.append(f"d{state_name}/dt = {rhs}")

    return '\n'.join(lines)


def get_continuous_equations(model, dt: float = None) -> str:
    """Return continuous-time ODE form of discovered equations.

    This is an alias for get_equations(). The dt parameter is kept
    for backward compatibility but is ignored (theta directly
    represents the ODE).
    """
    return get_equations(model)
