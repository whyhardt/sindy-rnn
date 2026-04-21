"""Pruning system: ensemble CI test, patience mechanism, and threshold fallback."""

import torch
from torch import Tensor


def minimum_effect_ci_test(
    coefficients: Tensor,
    presence: Tensor,
    alpha: float = 0.05,
    delta: float = 0.0,
) -> Tensor:
    """Minimum-effect confidence interval test for term survival.

    A term survives iff its ensemble mean is statistically distinguishable
    from zero at level alpha with minimum effect size delta.

    Pruned members (presence=False) contribute zero to the mean, naturally
    penalising terms that only a few ensemble members identified.

    Args:
        coefficients: (E, n_states, n_terms) — effective coefficients
        presence: (E, n_states, n_terms) — bool mask (which members have term)
        alpha: significance level (two-sided)
        delta: minimum effect size threshold

    Returns:
        significant: (n_states, n_terms) bool — True where term survives
    """
    import scipy.stats

    effective = (coefficients * presence.float()).detach()  # (E, n_states, n_terms)
    E = effective.shape[0]

    mean = effective.mean(dim=0)  # (n_states, n_terms)
    std = effective.std(dim=0, correction=1)
    se = std / (E ** 0.5)
    t_crit = scipy.stats.t.ppf(1 - alpha / 2, df=E - 1)

    ci_lower = mean.abs() - t_crit * se
    significant = ci_lower > delta

    # Require at least 2 active members
    n_active = presence.float().sum(dim=0)  # (n_states, n_terms)
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
        coefficients: (E, n_states, n_terms) — effective coefficients
        presence: (E, n_states, n_terms) — bool mask (which members have term)
        delta: minimum effect size threshold

    Returns:
        significant: (n_states, n_terms) bool — True where term survives
    """
    effective = (coefficients * presence.float()).detach()  # (E, n_states, n_terms)
    E = effective.shape[0]

    median = effective.median(dim=0).values  # (n_states, n_terms)

    n_active = presence.float().sum(dim=0).clamp(min=1)  # (n_states, n_terms)
    significant = (median.abs() > delta)

    # Require at least 2 active members
    significant = significant & (n_active >= 2)

    return significant


def _get_effective_coefficients_raw(model) -> Tensor:
    """Compute effective coefficients (gate-absorbed + self-term)."""
    theta = model.rnn.unfold_polynomial_coefficients().detach()  # (E, n_states, n_terms)
    gate = torch.sigmoid(model.rnn.damping_coefficient).detach()  # (E,)
    alpha_n = gate.view(-1, 1, 1) if model.rnn.scale_candidate else 1.0

    theta_eff = theta * alpha_n
    for i in range(model.n_states):
        self_idx = model.rnn._linear_indices[i].item()
        theta_eff[:, i, self_idx] = theta_eff[:, i, self_idx] + (1 - gate)  # broadcast (E,)

    return theta_eff


def _to_continuous(theta_eff: Tensor, model, dt: float) -> Tensor:
    """Convert discrete effective coefficients to continuous-time scale.

    Non-self terms: c / dt
    Self-term (dim i): (c - 1) / dt
    """
    theta_cont = theta_eff / dt
    for i in range(model.n_states):
        self_idx = model.rnn._linear_indices[i].item()
        theta_cont[:, i, self_idx] = (theta_eff[:, i, self_idx] - 1.0) / dt
    return theta_cont


def ensemble_prune(model, alpha: float, delta: float, dt: float = None,
                   method: str = 'ci'):
    """Run one pruning step using an ensemble statistical test.

    Args:
        model: PolynomialRNN instance
        alpha: significance level for CI test, or min_fraction for median test
        delta: minimum effect size threshold
        dt: timestep (converts to continuous-time coefficients when provided)
        method: 'ci' for mean-based confidence interval test,
                'median' for median + sign-agreement test (robust to bifurcation)
    """
    theta_eff = _get_effective_coefficients_raw(model)  # (E, n_states, n_terms)
    coefs_for_test = _to_continuous(theta_eff, model, dt) if dt else theta_eff
    mask = model.coefficient_masks  # (E, n_states, n_terms)

    if method == 'median':
        significant = median_effect_test(
            coefs_for_test, mask, delta=delta
        )
    else:
        significant = minimum_effect_ci_test(
            coefs_for_test, mask, alpha=alpha, delta=delta
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


def threshold_patience_update(model, threshold: float, dt: float = None):
    """Increment patience for terms with |effective_coef| < threshold.

    If dt is provided, coefficients are converted to continuous-time scale
    before comparison. Threshold is then in ODE units.
    """
    theta_eff = _get_effective_coefficients_raw(model)  # (E, n_states, n_terms)
    coefs_for_test = _to_continuous(theta_eff, model, dt) if dt else theta_eff
    below = (coefs_for_test.abs() < threshold) & model.coefficient_masks
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
