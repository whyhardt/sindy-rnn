"""Training loop for PolynomialRNN."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from .pruning import (
    ensemble_prune, threshold_patience_update, threshold_prune,
    compute_prune_budget,
)


def fit(
    model,
    xs: Tensor,
    ys: Tensor,
    xs_test: Optional[Tensor] = None,
    ys_test: Optional[Tensor] = None,
    epochs: int = 500,
    warmup_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: float = 1e-2,
    lambda_s: float = 1e-4,
    weight_decay: float = 0.0,
    pruning_frequency: int = 1,
    pruning_threshold: Optional[float] = None,
    agreement_frac: float = 0.5,
    pruning_method: str = 'agreement',
    ladder_exponent_step: float = 0.2,
    ladder_offset: float = -1.0,
    patience_limit: int = 2,
    include_bias: bool = True,
    interaction_only: bool = False,
    refit_epochs: int = 0,
    refit_learning_rate: Optional[float] = None,
    dynamics_weight: float = 0.0,
    centered_diff: bool = True,
    lr_patience: int = 0,
    lr_factor: float = 0.5,
    min_lr: float = 1e-5,
    verbose: bool = True,
):
    """Train the PolynomialRNN.

    The teacher-forced loss operates in derivative space: P(h) is compared
    directly to the empirical derivative. By default, centered differences
    (h[t+1] - h[t-1]) / (2*dt) are used for O(dt²) accuracy. Set
    centered_diff=False for forward differences (h[t+1] - h[t]) / dt with
    O(dt) accuracy (appropriate for discrete-time systems with large dt).

    When dynamics_weight > 0, an autonomous forecast loss is added: the model
    rolls out from h_0 via h[t+1] = h[t] + dt * P(h[t]) and the trajectory
    is compared to the true states.

    Total loss: E_deriv + dynamics_weight * E_fwd + lambda_s * theta^2

    Args:
        model: PolynomialRNN instance
        xs: (B, T, n_states + n_controls) — observations
        ys: (B, T, n_states) — next-state targets
        xs_test, ys_test: optional held-out validation data
        epochs: total training epochs
        warmup_steps: epochs before pruning begins (default: epochs // 4)
        batch_size: mini-batch size over sequences (None = full batch)
        learning_rate: AdamW learning rate
        lambda_s: coefficient penalty weight on unfolded polynomial coefficients.
            Scaled by ensemble_size internally so per-member penalty
            strength doesn't dilute as ensemble_size grows.
        weight_decay: AdamW weight decay applied to all model parameters,
            including the dynamics/polynomial params (raw, pre-unfolding
            weights under the factored/decomposed parameterizations — this
            is on top of, not instead of, lambda_s directly on theta). 0 = off
            (default).
        pruning_frequency: epochs between pruning events
        pruning_threshold: minimum effect size delta for pruning test.
        agreement_frac: fraction of active ensemble members that must
            individually exceed pruning_threshold. Only used for
            method='agreement'.
        pruning_method: 'agreement' for ensemble agreement test (default),
            'median' for median test (robust to bifurcation), 'ladder' for
            per-member geometric threshold ladder (no cross-member vote —
            each member decided independently against its own rung)
        ladder_exponent_step, ladder_offset: only used for
            pruning_method='ladder'. Member e's threshold is
            pruning_threshold * 10 ** (ladder_exponent_step * e +
            ladder_offset). Defaults mirror SINDy-SHRED's E_SINDy.thresholding().
        patience_limit: consecutive failed pruning events before permanent
            removal (default 2, see CLAUDE.md §5.3). 1 = prune immediately
            on the first failure.
        include_bias: if False, mask out constant term before training
        interaction_only: if True, mask out pure power terms before training
        refit_epochs: additional epochs with lambda_s=0 and frozen mask after pruning.
            Debiases coefficient estimates by removing penalty shrinkage on the
            identified support. 0 = no refit (default).
        refit_learning_rate: learning rate for refit phase (default: same as learning_rate)
        dynamics_weight: weight on autonomous forecast loss relative to
            derivative matching loss. 0.0 = derivative matching only (default).
        centered_diff: if True (default), use centered differences for O(dt²)
            derivative accuracy. Set to False for discrete-time systems (dt~1).
        lr_patience: ReduceLROnPlateau patience on the derivative-matching
            loss (0 = no scheduler, default).
        lr_factor: LR reduction factor on plateau.
        min_lr: minimum learning rate.
        verbose: show a tqdm progress bar with live loss/term-count postfix
    """
    if warmup_steps is None:
        warmup_steps = epochs // 2

    # Apply optional initial mask exclusions
    if not include_bias:
        model.coefficient_masks[:, :, model.rnn._bias_index] = False
    if interaction_only:
        for t_idx, term in enumerate(model.rnn._library_terms):
            if len(term) >= 2 and len(set(term)) == 1:
                model.coefficient_masks[:, :, t_idx] = False

    # Fixed per-event pruning budget: total active terms / total scheduled
    # pruning events, computed once up front so a single noisy early event
    # can't wipe out most of the model — see pruning.compute_prune_budget.
    n_pruning_events = sum(
        1 for e in range(warmup_steps, epochs)
        if pruning_frequency > 0 and e % pruning_frequency == 0
    )
    n_active_terms = int(model.coefficient_masks.any(dim=0).sum().item())
    max_prune = compute_prune_budget(n_active_terms, n_pruning_events)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor,
            patience=lr_patience, min_lr=min_lr)

    if verbose:
        print(f"Derivative-matching training ({epochs} epochs)")
        print(f"  Pruning method={pruning_method}, threshold={pruning_threshold}, "
              f"frequency={pruning_frequency}, activates at epoch {warmup_steps}")
        print(f"  Pruning maximum {max_prune} terms per pruning event "
              f"({n_pruning_events} events, {n_active_terms} active terms)")
        print(f"  lr={learning_rate:.1e}, lambda_s={lambda_s:.1e}, weight_decay={weight_decay:.1e}")

    E = model.ensemble_size
    B, T = xs.shape[0], xs.shape[1]
    n_states = model.n_states
    model_dt = model.rnn._dt  # buffer, stays on correct device

    # Bootstrap: (B, T, F) -> (E, B, T, F), fixed for entire training
    if E > 1:
        indices = torch.randint(0, B, (E, B))
        xs_train = xs[indices]
        ys_train = ys[indices]
    else:
        xs_train = xs.unsqueeze(0)
        ys_train = ys.unsqueeze(0)

    def _derivative_matching_loss(xb, yb, theta_masked):
        """Compute derivative matching loss.

        When centered_diff=True (default): uses (h[t+1] - h[t-1]) / (2*dt)
        for O(dt²) accuracy, evaluating P(h) at interior points t=1,...,T-2.
        When centered_diff=False: uses (h[t+1] - h[t]) / dt with O(dt)
        accuracy, evaluating P(h) at all timesteps.
        """
        if centered_diff:
            # Centered difference: (h[t+1] - h[t-1]) / (2*dt)
            h_prev = xb[:, :, :-2, :n_states]
            h_next = yb[:, :, 1:-1, :]  # yb[t] = h[t+1]
            dh_dt_target = (h_next - h_prev) / (2 * model_dt)

            # Evaluate P(h) at interior points h[1]...h[T-2]
            x_eval = xb[:, :, 1:-1, :n_states + model.rnn.n_controls]
        else:
            # Forward difference: (h[t+1] - h[t]) / dt
            h_all = xb[:, :, :, :n_states]
            dh_dt_target = (yb - h_all) / model_dt

            x_eval = xb[:, :, :, :n_states + model.rnn.n_controls]

        E_, Bb, T_inner, F_ = x_eval.shape
        x_flat = x_eval.reshape(E_, Bb * T_inner, F_)
        library = model.rnn._compute_library(x_flat)
        P_h = torch.einsum('ebt,ent->ebn', library, theta_masked)
        P_h = P_h.reshape(E_, Bb, T_inner, n_states)

        valid = ~torch.isnan(dh_dt_target.sum(dim=-1))
        return F.mse_loss(P_h[valid], dh_dt_target[valid])

    epoch_iter = tqdm(range(epochs), desc="Derivative-matching training", disable=not verbose)
    try:
        for epoch in epoch_iter:
            model.train()

            if batch_size is not None and batch_size < B:
                batch_idx = torch.randperm(B)[:batch_size]
                xb = xs_train[:, batch_idx]
                yb = ys_train[:, batch_idx]
            else:
                xb, yb = xs_train, ys_train

            theta = model.rnn.unfold_polynomial_coefficients()
            theta_masked = theta * model.coefficient_masks.float()

            # Derivative matching loss (E_deriv): MSE(P(h), (y-h)/dt)
            tf_loss = _derivative_matching_loss(xb, yb, theta_masked)

            # Autonomous forecast loss (E_fwd)
            if dynamics_weight > 0:
                h = xb[:, :, 0, :n_states]  # (E, Bb, n_states) — initial state
                fwd_loss = 0.
                T_win = xb.shape[2]
                for k in range(T_win):
                    u_k = xb[:, :, k, n_states:] if xb.shape[-1] > n_states else None
                    h = model.rnn.forward_polynomial(
                        h, u_k, mask=model.coefficient_masks, theta=theta
                    )
                    target = yb[:, :, k, :]
                    valid_k = ~torch.isnan(target.sum(dim=-1))
                    if valid_k.any():
                        fwd_loss = fwd_loss + F.mse_loss(h[valid_k], target[valid_k])
                fwd_loss = fwd_loss / T_win
            else:
                fwd_loss = 0.

            loss = tf_loss + dynamics_weight * fwd_loss

            # Coefficient penalty. Scaled by E so per-member penalty
            # strength is independent of ensemble_size — plain .mean() over
            # (E, n_states, n_terms) would otherwise dilute the penalty by
            # 1/E as E grows (mirrors fit_rollout()'s lambda_s scaling).
            if lambda_s > 0:
                loss = loss + lambda_s * theta.shape[0] * (theta * model.coefficient_masks).abs().mean()
                # loss = loss + lambda_s * theta.shape[0] * (theta * model.coefficient_masks).pow(2).mean()

            if not torch.isfinite(loss):
                # Mirrors fit_rollout()'s guard: an autonomous forecast term
                # (dynamics_weight > 0) can transiently blow up early in
                # training. Skip the update rather than apply nan/inf
                # gradients, which would permanently corrupt every weight.
                if verbose:
                    tqdm.write(f"  Non-finite loss at epoch {epoch}: skipping optimization step")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step(tf_loss.item())

            # Pruning
            if epoch >= warmup_steps and pruning_frequency > 0 and epoch % pruning_frequency == 0:
                with torch.no_grad():
                    n_before = int(model.coefficient_masks.any(dim=0).sum().item())
                    if E > 1:
                        ensemble_prune(model, pruning_threshold or 0.0,
                                       method=pruning_method,
                                       agreement_frac=agreement_frac,
                                       ladder_exponent_step=ladder_exponent_step,
                                       ladder_offset=ladder_offset,
                                       max_prune=max_prune,
                                       patience_limit=patience_limit)
                    elif pruning_threshold and pruning_threshold > 0:
                        threshold_patience_update(model, pruning_threshold)
                        threshold_prune(model, patience_limit=patience_limit, max_prune=max_prune)
                    n_pruned = n_before - int(model.coefficient_masks.any(dim=0).sum().item())
                if verbose and n_pruned > 0:
                    tqdm.write(f"  [prune] epoch {epoch}: removed {n_pruned} term(s) "
                              f"(method={pruning_method}, budget={max_prune})")

            # Logging: postfix updates every epoch (mirrors fit_rollout()).
            if verbose:
                active = model.count_active_terms()
                postfix = {'deriv': f'{tf_loss.item():.6f}'}
                if dynamics_weight > 0:
                    fwd_val = fwd_loss.item() if isinstance(fwd_loss, torch.Tensor) else fwd_loss
                    postfix['fwd'] = f'{fwd_val:.6f}'
                postfix['lr'] = f'{optimizer.param_groups[0]["lr"]:.1e}'
                postfix['terms'] = sum(active.values())
                with torch.no_grad():
                    theta_active = (theta * model.coefficient_masks.float())[model.coefficient_masks].abs()
                    if theta_active.numel() > 0:
                        postfix['|c|_max'] = f'{theta_active.max().item():.3f}'
                        postfix['|c|_mean'] = f'{theta_active.mean().item():.3f}'
                if xs_test is not None and ys_test is not None:
                    with torch.no_grad():
                        model.eval()
                        x_te = xs_test.unsqueeze(0).expand(E, -1, -1, -1)
                        y_te = ys_test.unsqueeze(0).expand(E, -1, -1, -1)
                        theta_te = model.rnn.unfold_polynomial_coefficients()
                        theta_te_m = theta_te * model.coefficient_masks.float()
                        loss_te = _derivative_matching_loss(x_te, y_te, theta_te_m)
                        postfix['test'] = f'{loss_te.item():.6f}'
                epoch_iter.set_postfix(postfix)
    except KeyboardInterrupt:
        if verbose:
            tqdm.write(f"\nTraining interrupted at epoch {epoch}.")

    # Post-pruning refit: train with lambda_s=0 and frozen mask to debias coefficients
    if refit_epochs > 0:
        refit_lr = refit_learning_rate if refit_learning_rate is not None else learning_rate
        refit_optimizer = torch.optim.AdamW(model.parameters(), lr=refit_lr)
        refit_scheduler = None
        if lr_patience > 0:
            refit_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                refit_optimizer, mode='min', factor=lr_factor,
                patience=lr_patience, min_lr=min_lr)

        if verbose:
            active = model.count_active_terms()
            total_active = sum(active.values())
            print(f"\nRefit phase: {refit_epochs} epochs, lr={refit_lr:.1e}, "
                  f"lambda_s=0, mask frozen ({total_active} active terms)")

        refit_iter = tqdm(range(refit_epochs), desc="Refit", disable=not verbose)
        try:
            for epoch in refit_iter:
                model.train()

                if batch_size is not None and batch_size < B:
                    batch_idx = torch.randperm(B)[:batch_size]
                    xb = xs_train[:, batch_idx]
                    yb = ys_train[:, batch_idx]
                else:
                    xb, yb = xs_train, ys_train

                theta = model.rnn.unfold_polynomial_coefficients()
                theta_masked = theta * model.coefficient_masks.float()

                # Derivative matching (no L1, no autonomous)
                loss = _derivative_matching_loss(xb, yb, theta_masked)

                if not torch.isfinite(loss):
                    if verbose:
                        tqdm.write(f"  Non-finite loss in refit epoch {epoch}: skipping optimization step")
                    refit_optimizer.zero_grad(set_to_none=True)
                    continue

                refit_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                refit_optimizer.step()
                if refit_scheduler is not None:
                    refit_scheduler.step(loss.item())

                if verbose:
                    postfix = {'deriv': f'{loss.item():.6f}',
                              'lr': f'{refit_optimizer.param_groups[0]["lr"]:.1e}'}
                    if xs_test is not None and ys_test is not None:
                        with torch.no_grad():
                            model.eval()
                            x_te = xs_test.unsqueeze(0).expand(E, -1, -1, -1)
                            y_te = ys_test.unsqueeze(0).expand(E, -1, -1, -1)
                            theta_te = model.rnn.unfold_polynomial_coefficients()
                            theta_te_m = theta_te * model.coefficient_masks.float()
                            loss_te = _derivative_matching_loss(x_te, y_te, theta_te_m)
                            postfix['test'] = f'{loss_te.item():.6f}'
                    refit_iter.set_postfix(postfix)
        except KeyboardInterrupt:
            if verbose:
                tqdm.write(f"\nRefit interrupted at epoch {epoch}.")

    # Rank ensemble members by their own fit, so predict()/simulate() can
    # switch from the ensemble mean to the single best-fitting member
    # without retraining. Prefer held-out data when given — the training
    # set alone would favor whichever member simply overfits hardest.
    if xs_test is not None and ys_test is not None:
        model.select_best_member(xs_test, ys_test)
    else:
        model.select_best_member(xs, ys)

    # BIC ranking is a separate, parallel selection ('simulate: bic') —
    # its sparsity penalty only means anything on the data the model was
    # actually fit to, so always use xs/ys here (never xs_test/ys_test).
    model.select_best_member_bic(xs, ys)
