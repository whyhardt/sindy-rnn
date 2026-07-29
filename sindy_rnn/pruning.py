"""Pruning system: ensemble median/agreement tests, patience mechanism, and
threshold fallback.

Pruning operates on polynomial coefficients (theta), which directly represent
the ODE right-hand side dh/dt = P(h) in the forward Euler update
h[t+1] = h[t] + dt * P(h). The pruning threshold (delta) is specified in
ODE units and used directly (no unit conversion needed).
"""

import torch
from torch import Tensor


def agreement_test(
    coefficients: Tensor,
    presence: Tensor,
    delta: float = 0.0,
    agreement_frac: float = 0.5,
) -> Tensor:
    """Ensemble agreement test for term survival.

    Each active member individually "votes" that a term exists iff its own
    |coefficient| > delta. A term survives iff at least agreement_frac of
    active members agree it exists. Simpler and cheaper than the mean-based
    CI test, and — unlike the median test — lets you tune how much of the
    ensemble must agree rather than requiring the single central value to
    clear delta.

    Args:
        coefficients: (E, n_states, n_terms) — ODE coefficients
        presence: (E, n_states, n_terms) — bool mask (which members have term)
        delta: minimum |coefficient| for a member to count as agreeing
        agreement_frac: fraction of active members that must agree

    Returns:
        significant: (n_states, n_terms) bool — True where term survives
    """
    votes = (coefficients.abs() > delta) & presence  # (E, n_states, n_terms)

    n_active = presence.float().sum(dim=0)  # (n_states, n_terms)
    n_agree = votes.float().sum(dim=0)

    significant = n_agree >= agreement_frac * n_active.clamp(min=1)

    # Require at least 2 active members
    significant = significant & (n_active >= 2)

    return significant


def median_effect_test(
    coefficients: Tensor,
    presence: Tensor,
    delta: float = 0.0,
) -> Tensor:
    """Median-based pruning test robust to ensemble bifurcation.

    A term survives iff its median |coefficient| across active members
    exceeds delta. Unlike the mean-based CI test, the median is robust
    to a minority of members finding an alternative parameterization.

    Args:
        coefficients: (E, n_states, n_terms) — ODE coefficients
        presence: (E, n_states, n_terms) — bool mask (which members have term)
        delta: minimum effect size threshold

    Returns:
        significant: (n_states, n_terms) bool — True where term survives
    """
    effective = (coefficients * presence.float()).detach()  # (E, n_states, n_terms)

    median = effective.median(dim=0).values  # (n_states, n_terms)

    n_active = presence.float().sum(dim=0).clamp(min=1)  # (n_states, n_terms)
    significant = (median.abs() > delta)

    # Require at least 2 active members
    significant = significant & (n_active >= 2)

    return significant


def _get_effective_coefficients_raw(model) -> Tensor:
    """Return polynomial coefficients (masked).

    With forward Euler h[t+1] = h[t] + dt * P(h), these coefficients
    directly represent the ODE right-hand side dh/dt = P(h).
    """
    theta = model.rnn.unfold_polynomial_coefficients().detach()  # (E, n_states, n_terms)
    return theta


def ensemble_prune(model, delta: float,
                   method: str = 'agreement', agreement_frac: float = 0.5):
    """Run one pruning step using an ensemble statistical test.

    Args:
        model: PolynomialRNN instance
        delta: minimum effect size threshold (in ODE units).
            Used directly since theta represents the ODE.
        method: 'median' for median test (robust to bifurcation),
                'agreement' for ensemble agreement test
        agreement_frac: fraction of active members that must individually
            exceed delta for a term to survive. Only used for method='agreement'.
    """
    theta = _get_effective_coefficients_raw(model)  # (E, n_states, n_terms)
    mask = model.coefficient_masks  # (E, n_states, n_terms)

    if method == 'agreement':
        significant = agreement_test(
            theta, mask, delta=delta, agreement_frac=agreement_frac
        )
    else:
        significant = median_effect_test(
            theta, mask, delta=delta
        )  # (n_states, n_terms)

    still_active = mask.any(dim=0)  # (n_states, n_terms)
    failed = ~significant & still_active

    # Update patience counters
    counters = model.pruning_patience  # (E, n_states, n_terms)
    failed_e = failed.unsqueeze(0).expand_as(counters)
    counters = torch.where(failed_e, counters + 1, torch.zeros_like(counters))

    # Permanently prune terms with patience >= 2
    prune = (counters[0] >= 2)  # (n_states, n_terms)
    if prune.any():
        model.coefficient_masks = model.coefficient_masks & ~prune.unsqueeze(0).expand_as(mask)
        counters = counters * (~prune.unsqueeze(0).expand_as(counters)).int()

    model.pruning_patience = counters


def threshold_patience_update(model, threshold: float):
    """Increment patience for terms with |coefficient| < threshold.

    Threshold is in ODE units and used directly since theta represents
    the ODE right-hand side.
    """
    theta = _get_effective_coefficients_raw(model)  # (E, n_states, n_terms)
    below = (theta.abs() < threshold) & model.coefficient_masks
    model.pruning_patience = torch.where(
        below,
        model.pruning_patience + 1,
        torch.zeros_like(model.pruning_patience)
    )


def threshold_prune(model, patience_limit: int = 2):
    """Permanently prune terms that exceeded patience_limit."""
    candidates = (model.pruning_patience >= patience_limit) & model.coefficient_masks
    model.coefficient_masks = model.coefficient_masks & ~candidates
    model.pruning_patience = model.pruning_patience * (~candidates).int()
