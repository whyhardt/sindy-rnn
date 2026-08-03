"""Equation extraction and printing utilities.

With forward Euler h[t+1] = h[t] + dt * P(h[t]), the polynomial coefficients
directly represent the ODE right-hand side dh/dt = P(h). No conversion needed.
"""

from typing import Dict, Optional, Union

import torch
from torch import Tensor


def get_coefficients(model, aggregate: Union[bool, str] = True, member: Optional[int] = None) -> Dict[str, Tensor]:
    """Return ODE coefficients for each state dimension.

    Coefficients directly represent the ODE dh/dt = P(h).
    Pruned terms are masked to zero.

    Args:
        aggregate: ignored when member is not None. True/'mean' (default)
            aggregates with the NaN-aware ensemble mean. 'median' aggregates
            with the NaN-aware ensemble median instead — e.g. for reporting
            equations when simulate='mean' but a single outlier member
            shouldn't be able to skew a displayed coefficient the way it can
            skew a mean. False returns the raw per-member array.
        member: if given, return that single ensemble member's own
            coefficients instead of aggregating — for reporting the model
            actually used when simulate='best' or simulate='bic' (see
            PolynomialRNN.select_best_member()/select_best_member_bic(),
            rollout.select_best_member()/select_best_member_bic()).

    Returns:
        dict mapping state_name -> Tensor
            member given:         (n_terms,) that member's own coefficients
            aggregate=True/'mean': (n_terms,) ensemble mean (NaN-aware, pruned->0)
            aggregate='median':    (n_terms,) ensemble median (NaN-aware, pruned->0)
            aggregate=False:       (E, n_terms) per-member
    """
    theta = model.rnn.unfold_ode_coefficients().detach()  # (E, n_states, n_terms)
    mask = model.coefficient_masks.float()  # (E, n_states, n_terms)

    # Apply mask — theta * mask gives active ODE coefficients
    c = theta * mask  # (E, n_states, n_terms)

    results = {}
    for i, name in enumerate(model.state_names):
        c_i = c[:, i, :]      # (E, n_terms)
        mask_i = mask[:, i, :]  # (E, n_terms)

        if member is not None:
            c_i = c_i[member]
        elif aggregate:
            c_agg = c_i.clone()
            # Mark pruned terms as NaN for nanmean/nanmedian
            c_agg = torch.where(
                mask_i == 0,
                torch.full_like(c_agg, float('nan')),
                c_agg,
            )
            if aggregate == 'median':
                c_i = torch.nanmedian(c_agg, dim=0).values
            else:
                c_i = torch.nanmean(c_agg, dim=0)
            c_i = torch.nan_to_num(c_i, nan=0.0)

        results[name] = c_i
    return results


def get_equations(model, member: Optional[int] = None, aggregate: Union[bool, str] = True) -> str:
    """Return discovered ODE equations as a formatted multi-line string.

    Coefficients directly represent dh/dt = P(h).

    Args:
        member: if given, report that single ensemble member's own
            equations instead of the ensemble aggregate.
        aggregate: ignored when member is given. True/'mean' (default) or
            'median' — see get_coefficients().

    Example output:
        dx/dt = -10.000*x + 10.000*y
        dy/dt = 28.000*x - 1.000*y - 1.000*x*z
        dz/dt = -2.667*z + 1.000*x*y
    """
    coefs = get_coefficients(model, aggregate=aggregate, member=member)
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


def get_continuous_equations(model, dt: float = None, member: Optional[int] = None,
                             aggregate: Union[bool, str] = True) -> str:
    """Return continuous-time ODE form of discovered equations.

    This is an alias for get_equations(). The dt parameter is kept
    for backward compatibility but is ignored (theta directly
    represents the ODE).
    """
    return get_equations(model, member=member, aggregate=aggregate)
