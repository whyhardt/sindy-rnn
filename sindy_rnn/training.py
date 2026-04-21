"""Training loop for PolynomialRNN."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .pruning import ensemble_prune, threshold_patience_update, threshold_prune


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
    l2: float = 1e-4,
    pruning_frequency: int = 1,
    pruning_threshold: Optional[float] = None,
    ensemble_pruning_alpha: float = 0.05,
    pruning_method: str = 'ci',
    dt: Optional[float] = None,
    include_bias: bool = True,
    interaction_only: bool = False,
    refit_epochs: int = 0,
    refit_learning_rate: Optional[float] = None,
    verbose: bool = True,
):
    """Train the PolynomialRNN.

    Args:
        model: PolynomialRNN instance
        xs: (B, T, n_states + n_controls) — observations
        ys: (B, T, n_states) — next-state targets
        xs_test, ys_test: optional held-out validation data
        epochs: total training epochs
        warmup_steps: epochs before pruning begins (default: epochs // 4)
        batch_size: mini-batch size over sequences (None = full batch)
        learning_rate: Adam learning rate
        l2: L1 penalty weight on unfolded polynomial coefficients
        pruning_frequency: epochs between pruning events
        pruning_threshold: minimum effect size delta for CI test (and threshold fallback).
            When dt is provided, this is in continuous-time (ODE) units.
        ensemble_pruning_alpha: confidence level alpha for ensemble CI test.
            For method='median', this is the minimum sign-agreement fraction.
        pruning_method: 'ci' for mean-based CI test,
            'median' for median + sign-agreement (robust to bifurcation)
        dt: timestep of the data. When provided, pruning operates on continuous-time
            coefficients (c/dt), making pruning_threshold interpretable in ODE units.
        include_bias: if False, mask out constant term before training
        interaction_only: if True, mask out pure power terms before training
        refit_epochs: additional epochs with l2=0 and frozen mask after pruning.
            Debiases coefficient estimates by removing L1 shrinkage on the
            identified support. 0 = no refit (default).
        refit_learning_rate: learning rate for refit phase (default: learning_rate / 5)
        verbose: print training progress every 50 epochs
    """
    if warmup_steps is None:
        warmup_steps = epochs // 4

    # Apply optional initial mask exclusions
    if not include_bias:
        model.coefficient_masks[:, :, model.rnn._bias_index] = False
    if interaction_only:
        for t_idx, term in enumerate(model.rnn._library_terms):
            if len(term) >= 2 and len(set(term)) == 1:
                model.coefficient_masks[:, :, t_idx] = False

    # Use Adam (no weight decay) — L2 is applied on polynomial coefficients instead
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        # weight_decay=l2,
    )

    E = model.ensemble_size
    B, T = xs.shape[0], xs.shape[1]

    # Bootstrap: (B, T, F) -> (E, B, T, F), fixed for entire training
    if E > 1:
        indices = torch.randint(0, B, (E, B))
        xs_train = xs[indices]
        ys_train = ys[indices]
    else:
        xs_train = xs.unsqueeze(0)
        ys_train = ys.unsqueeze(0)

    try:
        for epoch in range(epochs):
            model.train()

            if batch_size is not None and batch_size < B:
                batch_idx = torch.randperm(B)[:batch_size]
                xb = xs_train[:, batch_idx]
                yb = ys_train[:, batch_idx]
            else:
                xb, yb = xs_train, ys_train

            ys_pred, _ = model(xb)  # (E, B, T, n_states)

            # NaN mask for variable-length sequences
            valid = ~torch.isnan(yb.sum(dim=-1))  # (E, B, T)
            mse_loss = F.mse_loss(ys_pred[valid], yb[valid])

            # L2 penalty on unfolded polynomial coefficients
            if l2 > 0:
                theta = model.rnn.unfold_polynomial_coefficients()  # (E, n_states, n_terms)
                # coeff_penalty = l2 * (theta ** 2).mean()
                coeff_penalty = l2 * theta.abs().mean()
                loss = mse_loss + coeff_penalty
            else:
                loss = mse_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Pruning
            if epoch >= warmup_steps and epoch % pruning_frequency == 0:
                with torch.no_grad():
                    if ensemble_pruning_alpha and E > 1:
                        ensemble_prune(model, ensemble_pruning_alpha,
                                       pruning_threshold or 0.0, dt=dt,
                                       method=pruning_method)
                    elif pruning_threshold and pruning_threshold > 0:
                        threshold_patience_update(model, pruning_threshold, dt=dt)
                        threshold_prune(model, patience_limit=2)

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                active = model.count_active_terms()
                total_active = sum(active.values())
                msg = f"Epoch {epoch:4d} | mse {mse_loss.item():.6f} | active terms: {total_active}"
                if xs_test is not None and ys_test is not None:
                    with torch.no_grad():
                        model.eval()
                        x_te = xs_test.unsqueeze(0).expand(E, -1, -1, -1)
                        yp_te, _ = model(x_te)
                        valid_te = ~torch.isnan(ys_test.sum(dim=-1))
                        y_te_exp = ys_test.unsqueeze(0).expand(E, -1, -1, -1)
                        loss_te = F.mse_loss(
                            yp_te[:, valid_te], y_te_exp[:, valid_te]
                        )
                        msg += f" | test loss {loss_te.item():.6f}"
                print(msg)
    except KeyboardInterrupt:
        if verbose:
            print(f"\nTraining interrupted at epoch {epoch}.")

    # Post-pruning refit: train with l2=0 and frozen mask to debias coefficients
    if refit_epochs > 0:
        refit_lr = refit_learning_rate if refit_learning_rate is not None else learning_rate / 5
        refit_optimizer = torch.optim.Adam(model.parameters(), lr=refit_lr)

        if verbose:
            active = model.count_active_terms()
            total_active = sum(active.values())
            print(f"\nRefit phase: {refit_epochs} epochs, lr={refit_lr:.1e}, "
                  f"l2=0, mask frozen ({total_active} active terms)")

        try:
            for epoch in range(refit_epochs):
                model.train()

                if batch_size is not None and batch_size < B:
                    batch_idx = torch.randperm(B)[:batch_size]
                    xb = xs_train[:, batch_idx]
                    yb = ys_train[:, batch_idx]
                else:
                    xb, yb = xs_train, ys_train

                ys_pred, _ = model(xb)

                valid = ~torch.isnan(yb.sum(dim=-1))
                mse_loss = F.mse_loss(ys_pred[valid], yb[valid])

                refit_optimizer.zero_grad()
                mse_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                refit_optimizer.step()

                if verbose and (epoch % 50 == 0 or epoch == refit_epochs - 1):
                    msg = f"Refit {epoch:4d} | mse {mse_loss.item():.6f}"
                    if xs_test is not None and ys_test is not None:
                        with torch.no_grad():
                            model.eval()
                            x_te = xs_test.unsqueeze(0).expand(E, -1, -1, -1)
                            yp_te, _ = model(x_te)
                            valid_te = ~torch.isnan(ys_test.sum(dim=-1))
                            y_te_exp = ys_test.unsqueeze(0).expand(E, -1, -1, -1)
                            loss_te = F.mse_loss(
                                yp_te[:, valid_te], y_te_exp[:, valid_te]
                            )
                            msg += f" | test loss {loss_te.item():.6f}"
                    print(msg)
        except KeyboardInterrupt:
            if verbose:
                print(f"\nRefit interrupted at epoch {epoch}.")
