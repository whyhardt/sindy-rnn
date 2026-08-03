"""Pruning system: ensemble median/agreement/ladder tests, patience
mechanism, and threshold fallback.

Pruning operates on polynomial coefficients (theta), which directly represent
the ODE right-hand side dh/dt = P(h) in the forward Euler update
h[t+1] = h[t] + dt * P(h). The pruning threshold (delta) is specified in
ODE units and used directly (no unit conversion needed).
"""

from typing import Optional

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


def ladder_threshold_test(
    coefficients: Tensor,
    presence: Tensor,
    delta: float,
    exponent_step: float = 0.2,
    offset: float = -1.0,
) -> Tensor:
    """Per-member test with a geometric ladder of thresholds across the
    ensemble, instead of one shared threshold voted/aggregated across members.

    Member e gets its own threshold delta_e = delta * 10 ** (exponent_step *
    e + offset), so low-index members prune aggressively and high-index
    members leniently. Unlike agreement_test/median_effect_test, there is no
    cross-member aggregation step — each member's mask is decided purely by
    its own coefficient against its own rung of the ladder. This lets
    different members converge on different sparse structures instead of
    being pushed toward one consensus structure. Mirrors SINDy-SHRED's
    E_SINDy.thresholding() (sindy-shred/sindy_shred_net.py).

    Args:
        coefficients: (E, n_states, n_terms) — ODE coefficients
        presence: (E, n_states, n_terms) — bool mask (which members have term)
        delta: threshold for the member at exponent_step * e + offset == 0
        exponent_step: log10 spacing between consecutive members' thresholds
        offset: shifts which member index gets exactly delta

    Returns:
        significant: (E, n_states, n_terms) bool — per-member term survival
    """
    E = coefficients.shape[0]
    e_idx = torch.arange(E, device=coefficients.device, dtype=coefficients.dtype)
    delta_e = (delta * 10 ** (exponent_step * e_idx + offset)).view(E, 1, 1)
    return (coefficients.abs() > delta_e) & presence


def compute_prune_budget(n_active_terms: int, n_pruning_events: int) -> Optional[int]:
    """Fixed per-event pruning budget: n_active_terms / n_pruning_events,
    rounded up.

    Computed once, when pruning first activates (all n_pruning_events still
    ahead), and reused unchanged at every event. If every remaining event
    prunes exactly this many terms, the model reaches 0 active terms right
    at the last event — not sooner (a single event can't wipe out most of
    the model based on noisy early statistics) and not later (genuinely
    insignificant terms aren't stuck waiting indefinitely for a slot).

    Returns None if there are no pruning events scheduled (budget is
    meaningless — nothing will ever be pruned).
    """
    if n_pruning_events <= 0:
        return None
    return -(-n_active_terms // n_pruning_events)  # ceil division


def _apply_prune_budget(
    candidates: Tensor, magnitude: Tensor, max_prune: Optional[int],
    per_member: bool = False,
) -> Tensor:
    """Cap how many True entries in `candidates` are finalized this event.

    If more terms qualify than max_prune allows, only the max_prune terms
    with the smallest magnitude (the strongest evidence of being
    negligible) are finalized; the rest are left active for a later event,
    still accumulating patience normally (they're not reset — if they keep
    failing they'll be first in line for the budget next time).

    Args:
        candidates: bool tensor — terms that exceeded patience. Shape
            (n_states, n_terms) for agreement/median (one shared mask), or
            (E, n_states, n_terms) for ladder (per-member masks).
        magnitude: same shape as candidates — ranking score (smaller =
            weaker evidence of belonging in the model = pruned first)
        max_prune: budget for this event, or None for no cap
        per_member: if True, candidates/magnitude have a leading E
            dimension and each member competes only against its own
            candidates for its own max_prune-sized budget (ladder — masks
            genuinely differ per member). If False (default), all entries
            compete for one shared max_prune budget (agreement/median,
            where the whole (n_states, n_terms) tensor is one decision).

    Returns:
        bool tensor, same shape as candidates — the subset to actually prune
    """
    if per_member:
        return torch.stack([
            _apply_prune_budget(candidates[e], magnitude[e], max_prune)
            for e in range(candidates.shape[0])
        ])

    if max_prune is None or not candidates.any():
        return candidates
    n_candidates = int(candidates.sum().item())
    if n_candidates <= max_prune:
        return candidates

    flat_mag = magnitude.masked_fill(~candidates, float('inf')).flatten()
    keep_idx = torch.topk(flat_mag, k=max_prune, largest=False).indices
    prune = torch.zeros_like(candidates).flatten()
    prune[keep_idx] = True
    return prune.view_as(candidates)


def _get_effective_coefficients_raw(model) -> Tensor:
    """Return polynomial coefficients (masked).

    With forward Euler h[t+1] = h[t] + dt * P(h), these coefficients
    directly represent the ODE right-hand side dh/dt = P(h).
    """
    theta = model.rnn.unfold_polynomial_coefficients().detach()  # (E, n_states, n_terms)
    return theta


def ensemble_prune(model, delta: float,
                   method: str = 'agreement', agreement_frac: float = 0.5,
                   ladder_exponent_step: float = 0.2, ladder_offset: float = -1.0,
                   max_prune: Optional[int] = None, patience_limit: int = 2):
    """Run one pruning step using an ensemble statistical test.

    Args:
        model: PolynomialRNN instance
        delta: minimum effect size threshold (in ODE units).
            Used directly since theta represents the ODE.
        method: 'median' for median test (robust to bifurcation),
                'agreement' for ensemble agreement test,
                'ladder' for per-member geometric threshold ladder (no
                cross-member vote — see ladder_threshold_test)
        agreement_frac: fraction of active members that must individually
            exceed delta for a term to survive. Only used for method='agreement'.
        ladder_exponent_step, ladder_offset: only used for method='ladder',
            see ladder_threshold_test.
        max_prune: cap on how many terms get permanently pruned this call.
            If more terms exceed patience than this, only the max_prune
            with the smallest |coefficient| are pruned; the rest wait for a
            later call (see compute_prune_budget, _apply_prune_budget).
            None = no cap (prune everything that qualifies, as before).
            For method='ladder', this budget applies independently to each
            ensemble member (each member's mask is its own, so each gets
            its own max_prune-sized allowance per event); for
            'agreement'/'median' it applies once to the single shared mask.
        patience_limit: number of consecutive failed pruning events a term
            must accumulate before permanent removal. Default 2 (see
            CLAUDE.md §5.3). 1 = prune on the first failure, no patience.
    """
    theta = _get_effective_coefficients_raw(model)  # (E, n_states, n_terms)
    mask = model.coefficient_masks  # (E, n_states, n_terms)

    if method == 'ladder':
        # No cross-member aggregation: each member's survival depends only
        # on its own rung of the ladder, so patience/pruning stay per-member
        # too (unlike agreement/median, which share one decision across E).
        significant = ladder_threshold_test(
            theta, mask, delta=delta,
            exponent_step=ladder_exponent_step, offset=ladder_offset,
        )  # (E, n_states, n_terms)

        failed = ~significant & mask
        counters = torch.where(
            failed, model.pruning_patience + 1,
            torch.zeros_like(model.pruning_patience)
        )

        candidates = counters >= patience_limit
        # Per-member budget: each member's mask is independent (no
        # cross-member vote), so each gets its own max_prune-sized budget
        # rather than competing in one shared pool across the ensemble.
        prune = _apply_prune_budget(candidates, theta.abs(), max_prune, per_member=True)
        if prune.any():
            model.coefficient_masks = model.coefficient_masks & ~prune
            counters = torch.where(prune, torch.zeros_like(counters), counters)

        model.pruning_patience = counters
        return

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

    # Permanently prune terms with patience >= patience_limit, capped to
    # max_prune terms with the smallest |median coefficient| across the
    # whole model.
    candidates = (counters[0] >= patience_limit)  # (n_states, n_terms)
    magnitude = (theta * mask.float()).abs().median(dim=0).values  # (n_states, n_terms)
    prune = _apply_prune_budget(candidates, magnitude, max_prune)

    if prune.any():
        model.coefficient_masks = model.coefficient_masks & ~prune.unsqueeze(0).expand_as(mask)
        counters = torch.where(
            prune.unsqueeze(0).expand_as(counters), torch.zeros_like(counters), counters
        )

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


def threshold_prune(model, patience_limit: int = 2, max_prune: Optional[int] = None):
    """Permanently prune terms that exceeded patience_limit.

    max_prune: cap on how many terms get pruned this call — see
        ensemble_prune's max_prune / compute_prune_budget. When more terms
        qualify, only the max_prune with the smallest |coefficient| are
        pruned; the rest wait for a later call.
    """
    candidates = (model.pruning_patience >= patience_limit) & model.coefficient_masks
    theta = _get_effective_coefficients_raw(model)
    magnitude = theta.abs()
    prune = _apply_prune_budget(candidates, magnitude, max_prune)
    model.coefficient_masks = model.coefficient_masks & ~prune
    model.pruning_patience = model.pruning_patience * (~prune).int()
