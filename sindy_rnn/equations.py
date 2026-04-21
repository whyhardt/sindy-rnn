"""Equation extraction and printing utilities."""

from typing import Dict, Optional

import torch
from torch import Tensor


def get_coefficients(model, aggregate: bool = True) -> Dict[str, Tensor]:
    """Return effective polynomial coefficients for each state dimension.

    Effective = gate-absorbed + (1-alpha) self-term correction.
    Pruned terms are treated as zero (or NaN before nanmean when aggregate=True).

    Returns:
        dict mapping state_name -> Tensor
            aggregate=True:  (n_terms,) ensemble mean (NaN-aware, pruned->0)
            aggregate=False: (E, n_terms) per-member
    """
    alpha = torch.sigmoid(model.rnn.damping_coefficient).detach()  # (E,)
    alpha_n = alpha.view(-1, 1, 1) if model.rnn.scale_candidate else 1.0

    theta = model.rnn.unfold_polynomial_coefficients().detach()  # (E, n_states, n_terms)
    mask = model.coefficient_masks.float()  # (E, n_states, n_terms)

    c_eff = theta * mask * alpha_n  # (E, n_states, n_terms)

    # Add (1-alpha) to each dimension's self-term (always, regardless of mask)
    for i in range(model.n_states):
        self_idx = model.rnn._linear_indices[i].item()
        c_eff[:, i, self_idx] = c_eff[:, i, self_idx] + (1 - alpha)  # broadcast (E,)

    results = {}
    for i, name in enumerate(model.state_names):
        c_i = c_eff[:, i, :]      # (E, n_terms)
        mask_i = mask[:, i, :]     # (E, n_terms)
        self_idx = model.rnn._linear_indices[i].item()

        if aggregate:
            c_agg = c_i.clone()
            # Mark pruned non-self terms as NaN for nanmean
            non_self = torch.ones(c_i.shape[-1], dtype=torch.bool)
            non_self[self_idx] = False
            c_agg[:, non_self] = torch.where(
                mask_i[:, non_self] == 0,
                torch.full_like(c_agg[:, non_self], float('nan')),
                c_agg[:, non_self],
            )
            c_i = torch.nanmean(c_agg, dim=0)
            c_i = torch.nan_to_num(c_i, nan=0.0)

        results[name] = c_i
    return results


def get_equations(model) -> str:
    """Return discovered equations as a formatted multi-line string.

    Example output:
        x[t+1] = 0.900*x[t] + 0.100*y
        y[t+1] = 0.280*x - 0.100*x*z + 0.900*y[t]
        z[t+1] = 0.100*x*y + 0.973*z[t]

    The [t] suffix marks the self-term of each state variable.
    """
    coefs = get_coefficients(model, aggregate=True)
    term_names = model.library_terms
    lines = []

    for i, state_name in enumerate(model.state_names):
        c = coefs[state_name]  # (n_terms,)
        self_idx = model.rnn._linear_indices[i].item()
        parts = []

        for j, term_name in enumerate(term_names):
            val = c[j].item()
            if abs(val) < 1e-8:
                continue
            sign = '+' if val >= 0 else '-'
            coef_str = f"{abs(val):.3f}"
            if term_name == '1':
                label = coef_str
            elif j == self_idx:
                label = f"{coef_str}*{state_name}[t]"
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

        lines.append(f"{state_name}[t+1] = {rhs}")

    return '\n'.join(lines)


def get_continuous_equations(model, dt: float) -> str:
    """Convert discovered discrete-time equations to continuous-time ODE form.

    The discrete update is:
        h_i[t+1] = (1-α)*h_i[t] + P_i(h[t])

    Subtracting h_i[t] and dividing by dt:
        dh_i/dt ≈ (h_i[t+1] - h_i[t]) / dt
                = ((1-α)*h_i + P_i(h) - h_i) / dt
                = (-α*h_i + P_i(h)) / dt

    For the self-term (h_i in P_i), the effective discrete coef is
    (1-α) + θ_raw_self, so the continuous contribution of h_i is:
        ((1-α) + θ_raw_self - 1) / dt = (θ_raw_self - α) / dt

    For all other terms:
        c_continuous = c_discrete / dt

    Args:
        model: PolynomialRNN
        dt: timestep used in data generation

    Returns:
        Formatted string of continuous-time equations d/dt h_i = ...
    """
    coefs = get_coefficients(model, aggregate=True)
    term_names = model.library_terms
    lines = []

    for i, state_name in enumerate(model.state_names):
        c = coefs[state_name]  # (n_terms,) — discrete-time effective coefficients
        self_idx = model.rnn._linear_indices[i].item()
        parts = []

        for j, term_name in enumerate(term_names):
            discrete_val = c[j].item()

            if j == self_idx:
                # Continuous self-term: (discrete_self - 1) / dt
                cont_val = (discrete_val - 1.0) / dt
            else:
                # Continuous non-self: discrete / dt
                cont_val = discrete_val / dt

            if abs(cont_val) < 1e-6:
                continue
            sign = '+' if cont_val >= 0 else '-'
            coef_str = f"{abs(cont_val):.3f}"
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
