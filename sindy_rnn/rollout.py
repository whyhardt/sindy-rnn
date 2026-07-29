"""Rollout SINDy-RNN: GRU encoder -> autonomous polynomial rollout -> decoder.

Architecture:
    z_0 = GRU(x_sparse[0:T_w])              # encode warmup window
    z_{t+1} = z_t + dt * P(z_t)              # autonomous polynomial ODE
    x_hat_t = D(z_t)  for t = 0, ..., T_cur  # decode at every step

Trained end-to-end with per-step reconstruction loss and a rollout-length
curriculum that gradually increases the forecast horizon. The curriculum
prevents the optimizer from needing to stabilize long rollouts from scratch.

The decoder is deliberately linear, forcing the latent space to carry
physical structure rather than letting a nonlinear decoder absorb dynamics.

Identity mode (`identity=True`, requires n_sensors == n_latent == n_full):
skips the encoder/decoder entirely so z IS the observed state. This applies
the same rollout curriculum + per-step noise injection to fully-observed
(e.g. noisy full-state Lorenz) data — trajectory-matching against noisy
observations is more noise-robust than fit()'s derivative matching, at the
cost of a harder optimization landscape. See CLAUDE.md for the tradeoff.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import PolynomialRNN


class RolloutSINDyRNN(nn.Module):
    """GRU encoder -> autonomous polynomial rollout -> linear decoder.

    The GRU encoder processes a warmup window of sparse sensor observations
    to produce an initial latent state z_0. The polynomial dynamics propagates
    z_0 forward autonomously, and a linear decoder reconstructs the full field
    at every rollout step.

    Encoder and decoder are shared; only the polynomial dynamics has E
    independent ensemble members.

    Args:
        n_sensors: number of sparse sensor channels
        n_latent: latent state dimension
        n_full: full state dimension (decoder output)
        ensemble_size: number of ensemble members (dynamics only)
        polynomial_degree: degree of polynomial ODE
        dt: timestep for forward Euler integration
        num_euler_steps: sub-steps per dt interval
        gru_layers: number of GRU layers
        state_names: names for latent state variables
        decomposed: use decomposed polynomial parameterization
        direct: use direct polynomial parameterization
        dynamics_dropout: dropout for polynomial layer
        dynamics_feature_dropout: feature dropout for polynomial layer
        identity: if True, skip the GRU encoder and decoder — z IS the
            observed state. Requires n_sensors == n_latent == n_full.
    """

    def __init__(
        self,
        n_sensors: int,
        n_latent: int,
        n_full: int,
        ensemble_size: int = 1,
        polynomial_degree: int = 2,
        dt: float = 1.0,
        num_euler_steps: int = 1,
        gru_layers: int = 1,
        dec_layers: int = 1,
        state_names: Optional[List[str]] = None,
        decomposed: bool = True,
        direct: bool = False,
        dynamics_dropout: float = 0.,
        dynamics_feature_dropout: float = 0.,
        compile_dynamics: bool = False,
        rollout_noise: float = 0.,
        identity: bool = False,
    ):
        super().__init__()
        if identity and not (n_sensors == n_latent == n_full):
            raise ValueError(
                "identity=True requires n_sensors == n_latent == n_full "
                "(no encoder/decoder means no dimensionality change)")

        self.n_sensors = n_sensors
        self.n_latent = n_latent
        self.n_full = n_full
        self.rollout_noise = rollout_noise
        self.identity = identity

        # GRU encoder: sparse sensors -> z_0. Skipped when identity=True —
        # the observed state IS z, used for fully-observed systems (e.g.
        # noisy full-state Lorenz) where the rollout curriculum + per-step
        # noise injection is used for its trajectory-matching noise
        # robustness, not for sparse-sensor reconstruction.
        self.encoder = None if identity else nn.GRU(
            n_sensors, n_latent,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dynamics_dropout,
        )

        # Autonomous polynomial dynamics: dz/dt = P(z)
        state_names = state_names or [f'z{i}' for i in range(n_latent)]
        self.dynamics = PolynomialRNN(
            n_states=n_latent,
            n_controls=0,
            ensemble_size=ensemble_size,
            polynomial_degree=polynomial_degree,
            dt=dt,
            state_names=state_names,
            dropout=0.,
            feature_dropout=0.,
            decomposed=decomposed,
            direct=direct,
            num_euler_steps=num_euler_steps,
        )

        # Linear decoder (deliberately simple — prevents decoder absorbing dynamics)
        # Skipped when identity=True — z IS the full state.
        self.decoder = nn.Identity() if identity else nn.Linear(n_latent, n_full)
        
        # self.decoder = nn.Sequential(
        #     nn.Linear(n_latent, 150),
        #     nn.ReLU(),
        #     nn.Dropout(dynamics_dropout),
        #     nn.Linear(150, 450),
        #     nn.ReLU(),
        #     nn.Dropout(dynamics_dropout),
        #     nn.Linear(450, n_full),
        # )
        
        # Running z normalization between encoder and dynamics.
        # Polynomial always sees standardized z (≈ mean 0, std 1).
        # Decoder always sees raw z. Updated via EMA during training.
        self.register_buffer('_z_mean', torch.zeros(n_latent))
        self.register_buffer('_z_std', torch.ones(n_latent))
        self.register_buffer('_z_updates', torch.tensor(0, dtype=torch.long))
        self._z_momentum = 0.1

        # Optional: compile the per-step RHS for faster rollouts.
        # Shapes are fixed per step (E, B, n_latent) so no recompilation.
        self._compiled_rhs = None
        # if compile_dynamics:
        #     try:
        #         self._compiled_rhs = torch.compile(
        #             self.dynamics.rnn._evaluate_rhs)
        #     except Exception:
        #         pass

    @property
    def ensemble_size(self):
        return self.dynamics.ensemble_size

    def _normalize_z(self, z_raw: Tensor) -> Tensor:
        """Standardize z using running stats with gradient rescaling.

        Uses the detach trick from sindy_rnn_shred: forward is identity
        (z_raw), backward gradient is O(1) instead of O(1/z_std).

        Args:
            z_raw: (..., n_latent)

        Returns:
            z_norm: (..., n_latent) — standardized
        """
        mu = self._z_mean
        sig = self._z_std

        if self.training:
            # Gradient trick: d(z_scaled)/d(z_raw) = sig,
            # so d(z_norm)/d(z_raw) = sig / sig = 1
            z_scaled = z_raw.detach() + sig * (z_raw - z_raw.detach())
            return (z_scaled - mu) / sig
        else:
            return (z_raw - mu) / sig

    def _denormalize_z(self, z_norm: Tensor) -> Tensor:
        """Un-standardize z back to raw encoder space."""
        return z_norm * self._z_std + self._z_mean

    def encode(self, x_sparse_warmup: Tensor) -> Tensor:
        """Encode warmup window to initial latent state.

        Returns standardized z_0 (polynomial sees normalized input).
        Updates running mean/std via EMA during training.

        Args:
            x_sparse_warmup: (B, T_w, n_sensors)

        Returns:
            z_0: (E, B, n_latent) — standardized
        """
        if self.encoder is None:
            # identity=True: the observed state IS z. No recurrent context
            # needed — take the last (possibly noisy) observed frame.
            z_raw = x_sparse_warmup[:, -1, :]  # (B, n_latent)
        else:
            _, h_n = self.encoder(x_sparse_warmup)  # (num_layers, B, n_latent)
            z_raw = h_n[-1]  # (B, n_latent)
        E = self.ensemble_size
        z_raw = z_raw.unsqueeze(0).expand(E, -1, -1)  # (E, B, n_latent)

        # Update running stats (detached, no gradient)
        # if self.training:
        #     with torch.no_grad():
        #         flat = z_raw.detach().reshape(-1, self.n_latent)  # (E*B, n_latent)
        #         batch_mean = flat.mean(0)
        #         batch_std = flat.std(0).clamp(min=1e-6)
        #         m = self._z_momentum
        #         self._z_mean.mul_(1 - m).add_(m * batch_mean)
        #         self._z_std.mul_(1 - m).add_(m * batch_std)
        #         self._z_updates.add_(1)

        # return self._normalize_z(z_raw)
        return z_raw

    # @torch.compile(dynamic=True)
    def _rollout(self, z: Tensor, n_steps: int, theta_masked: Tensor,
                 include_init: bool = True) -> Tensor:
        """Euler sub-stepping rollout (compiled).

        Entire loop is captured by torch.compile into a single fused graph.
        Each unique n_steps value triggers a retrace (loop unrolls at trace
        time), but within a curriculum phase all epochs reuse the same graph.

        Calls _evaluate_rhs_impl directly (not _evaluate_rhs) to avoid
        nesting compiled functions.

        Args:
            z: (E, B, n_latent) — initial state
            n_steps: number of ODE steps
            theta_masked: (E, n_states, n_terms) — pre-masked coefficients
            include_init: whether to include z_0 in output

        Returns:
            (E, B, T, n_latent) — stacked trajectory, T = n_steps+1 or n_steps
        """
        rnn = self.dynamics.rnn
        dt_sub = rnn._dt / rnn._num_euler_steps

        traj = [z] if include_init else []
        for _ in range(n_steps):
            for _ in range(rnn._num_euler_steps):
                z = z + dt_sub * rnn._evaluate_rhs_impl(z, None, theta_masked)
            if self.training and self.rollout_noise > 0:
                z = z + self.rollout_noise * torch.randn_like(z)
            traj.append(z)
        return torch.stack(traj, dim=2)

    def forward(
        self,
        x_sparse_warmup: Tensor,
        T_cur: int,
    ) -> tuple:
        """Encode warmup -> rollout T_cur steps -> decode all steps.

        Encoder output is standardized before polynomial rollout.
        Trajectory is denormalized before decoding.

        Args:
            x_sparse_warmup: (B, T_w, n_sensors)
            T_cur: number of autonomous rollout steps

        Returns:
            x_hat: (B, T_cur+1, n_full) — ensemble-mean decoded reconstruction
            z_traj: (E, B, T_cur+1, n_latent) — latent trajectory (normalized)
        """
        z = self.encode(x_sparse_warmup)  # (E, B, n_latent) — normalized

        # Cache theta with mask applied (reused for all rollout steps)
        theta = self.dynamics.rnn.unfold_polynomial_coefficients()
        theta_masked = theta * self.dynamics.coefficient_masks.float()

        z_stack = self._rollout(z, T_cur, theta_masked, include_init=True)

        # Denormalize before decoder
        z_raw = self._denormalize_z(z_stack)
        x_hat = self.decoder(z_raw.mean(0))  # (B, T_cur+1, n_full)

        return x_hat, z_stack

    def forecast(
        self,
        x_sparse_warmup: Tensor,
        n_steps: int,
    ) -> tuple:
        """Forecast from a warmup window (no per-step supervision).

        Uses autonomous_dynamics (from refit) if available, otherwise
        uses the jointly-trained dynamics. Encoder z is standardized
        via running stats, denormalized before decoding.

        Args:
            x_sparse_warmup: (B, T_w, n_sensors)
            n_steps: number of forecast steps

        Returns:
            x_hat: (B, n_steps, n_full) — decoded forecast (excludes initial z_0)
            z_traj: (E, B, n_steps, n_latent) — latent trajectory (normalized)
        """
        self.eval()
        dyn = self._eq_dynamics

        with torch.no_grad():
            z = self.encode(x_sparse_warmup)  # (E, B, n_latent) — raw

            # Normalize into the space autonomous_dynamics was trained on
            z = self._normalize_z(z)

            theta = dyn.rnn.unfold_polynomial_coefficients()
            theta_masked = theta * dyn.coefficient_masks.float()

            z_traj = self._rollout(z, n_steps, theta_masked, include_init=False)

            # Denormalize before decoding
            z_raw = self._denormalize_z(z_traj)
            x_hat = self.decoder(z_raw.mean(0))  # (B, n_steps, n_full)

        return x_hat, z_traj

    def extract_latent_trajectory(
        self,
        x_sparse_all: Tensor,
        T_w: int,
        chunk_size: int = 32,
        raw: bool = False,
    ) -> Tensor:
        """Extract encoder latent z at each valid timestep.

        For each frame t >= T_w-1, encodes sensor window [t-T_w+1, t+1)
        to produce z_t.

        Args:
            x_sparse_all: (N_time, n_sensors) — raw sensor time series
            T_w: warmup window length
            chunk_size: batch size for encoding
            raw: if True, return raw GRU output (before normalization)

        Returns:
            z_trajectory: (N_valid, n_latent) where N_valid = N_time - T_w + 1
        """
        N = x_sparse_all.shape[0]
        device = next(self.parameters()).device
        self.eval()
        z_list = []

        with torch.no_grad():
            for batch_start in range(T_w - 1, N, chunk_size):
                batch_end = min(batch_start + chunk_size, N)
                windows = torch.stack([
                    x_sparse_all[t - T_w + 1:t + 1]
                    for t in range(batch_start, batch_end)
                ]).to(device)

                if raw:
                    # Bypass normalization — return raw GRU hidden state
                    if self.encoder is None:
                        z = windows[:, -1, :]  # (chunk, n_latent)
                    else:
                        _, h_n = self.encoder(windows)
                        z = h_n[-1]  # (chunk, n_latent)
                else:
                    z = self.encode(windows).mean(0)  # (chunk, n_latent)
                z_list.append(z.cpu())

        return torch.cat(z_list, dim=0)  # (N_valid, n_latent)

    # --- Equation delegation (uses autonomous_dynamics if available) ---

    @property
    def _eq_dynamics(self):
        """Return autonomous dynamics (refit) if available, else dynamics."""
        return getattr(self, 'autonomous_dynamics', self.dynamics)

    def get_equations(self) -> str:
        return self._eq_dynamics.get_equations()

    def get_continuous_equations(self, dt: float = None) -> str:
        return self._eq_dynamics.get_continuous_equations(dt)

    def get_coefficients(self, aggregate=True) -> Dict[str, Tensor]:
        return self._eq_dynamics.get_coefficients(aggregate=aggregate)

    def count_active_terms(self) -> Dict[str, int]:
        return self._eq_dynamics.count_active_terms()

    def print_equations(self):
        self._eq_dynamics.print_equations()

    def save(self, path: str):
        """Save model weights + constructor config.

        If refit_rollout() has been run, model.autonomous_dynamics (with its
        own sparsity masks/patience) is saved alongside it — state_dict()
        captures it automatically since it's a registered submodule.
        """
        torch.save({
            'state_dict': self.state_dict(),
            'has_autonomous_dynamics': hasattr(self, 'autonomous_dynamics'),
            'config': {
                'n_sensors': self.n_sensors,
                'n_latent': self.n_latent,
                'n_full': self.n_full,
                'ensemble_size': self.ensemble_size,
                'polynomial_degree': self.dynamics.rnn._degree,
                'dt': self.dynamics.rnn._dt.item(),
                'num_euler_steps': self.dynamics.rnn._num_euler_steps,
                'gru_layers': 1 if self.encoder is None else self.encoder.num_layers,
                'state_names': self.dynamics.state_names,
                'decomposed': self.dynamics.rnn._decomposed,
                'direct': self.dynamics.rnn._direct,
                'rollout_noise': self.rollout_noise,
                'identity': self.identity,
            }
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load a saved model. kwargs override saved config."""
        checkpoint = torch.load(path, weights_only=False)
        config = {**checkpoint['config'], **kwargs}
        model = cls(**config)

        if checkpoint['has_autonomous_dynamics']:
            dynamics = model.dynamics
            model.autonomous_dynamics = PolynomialRNN(
                n_states=model.n_latent,
                n_controls=0,
                ensemble_size=model.ensemble_size,
                polynomial_degree=dynamics.rnn._degree,
                dt=dynamics.rnn._dt.item(),
                state_names=dynamics.state_names,
                decomposed=dynamics.rnn._decomposed,
                direct=dynamics.rnn._direct,
                num_euler_steps=dynamics.rnn._num_euler_steps,
            )

        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model


def fit_rollout(
    model: RolloutSINDyRNN,
    x_sparse_all: Tensor,
    x_full_all: Tensor,
    T_w: int = 12,
    T_max: int = 100,
    T_start: int = 1,
    delta_T: int = 2,
    E_step: Optional[int] = None,
    epochs: int = 5000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    lambda_0: float = 1e-3,
    lambda_s: float = 1e-3,
    grad_clip: float = 0.5,
    batches_per_epoch: int = 4,
    pruning_threshold: float = 0.1,
    pruning_frequency: int = 100,
    pruning_method: str = 'agreement',
    agreement_frac: float = 0.5,
    lr_patience: int = 0,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    rollout_noise: float = 0.1,
    x_sparse_test: Optional[Tensor] = None,
    x_full_test: Optional[Tensor] = None,
    verbose: bool = True,
):
    """Train RolloutSINDyRNN with rollout-length curriculum.

    Data format: raw time series (not pre-windowed). Windows are constructed
    on-the-fly during training, which allows T_cur to vary with the curriculum.

    Each training sample is built from a consecutive span of T_w + T_cur frames:
      - Warmup: x_sparse_all[s : s+T_w]           (GRU encoder input)
      - Targets: x_full_all[s+T_w-1 : s+T_w+T_cur]  (T_cur+1 reconstruction targets)

    The first target (at index s+T_w-1) is the same-timestep reconstruction
    of the last warmup frame (z_0 target). Subsequent targets are forecasts.

    Curriculum schedule:
        T_cur(epoch) = min(T_max, T_start + floor(epoch / E_step) * delta_T)

    Sparsity (L1 + pruning) activates only after the curriculum reaches T_max,
    avoiding the moving-target problem from premature sparsification.

    Args:
        model: RolloutSINDyRNN instance (already on target device)
        x_sparse_all: (N_time, n_sensors) — training sensor time series
        x_full_all: (N_time, n_full) — training full state time series
        T_w: warmup window length (GRU encoder input)
        T_max: maximum rollout length (curriculum target)
        T_start: starting rollout length
        delta_T: rollout increment per curriculum step
        E_step: epochs per curriculum step (auto-computed if None)
        epochs: total training epochs
        batch_size: mini-batch size
        learning_rate: Adam learning rate
        lambda_0: z_0 norm regularization weight
        lambda_s: L1 sparsity weight on theta (after curriculum completes)
        grad_clip: gradient clip norm
        batches_per_epoch: max mini-batches per epoch (stride-1 windows are
            heavily redundant; 2-8 batches per epoch is typically sufficient)
        pruning_threshold: pruning threshold delta
        pruning_frequency: epochs between pruning (after sparsity activation)
        pruning_method: 'median' or 'agreement'
        agreement_frac: fraction of active ensemble members that must
            individually exceed pruning_threshold. Only used for method='agreement'.
        lr_patience: ReduceLROnPlateau patience (0 = no scheduler)
        lr_factor: LR reduction factor on plateau
        min_lr: minimum learning rate
        x_sparse_test: (N_test, n_sensors) — optional test sensor time series
        x_full_test: (N_test, n_full) — optional test full state time series
        verbose: print training progress
    """
    from .pruning import (
        ensemble_prune, threshold_patience_update, threshold_prune,
    )

    N_time = x_sparse_all.shape[0]
    device = next(model.parameters()).device
    model.rollout_noise = rollout_noise

    # Auto-compute E_step so curriculum finishes in first 50% of epochs
    if E_step is None:
        n_curriculum_steps = max(1, (T_max - T_start + delta_T - 1) // delta_T)
        E_step = max(1, epochs // (2 * n_curriculum_steps))

    # Epoch when sparsity activates (T_cur first reaches T_max)
    n_steps_to_max = max(0, (T_max - T_start + delta_T - 1) // delta_T)
    sparsity_start_epoch = n_steps_to_max * E_step

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = None
    if lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor,
            patience=lr_patience, min_lr=min_lr)

    if verbose:
        print(f"Rollout curriculum training ({epochs} epochs)")
        print(f"  T_w={T_w}, T_start={T_start}, T_max={T_max}, "
              f"delta_T={delta_T}, E_step={E_step}")
        print(f"  Sparsity activates at epoch {sparsity_start_epoch}")
        print(f"  lr={learning_rate:.1e}, lambda_0={lambda_0:.1e}, "
              f"lambda_s={lambda_s:.1e}, grad_clip={grad_clip}")
        print(f"  N_time={N_time}, batch_size={batch_size}, "
              f"batches_per_epoch={batches_per_epoch}")

    try:
        for epoch in range(epochs):
            model.train()

            # Curriculum: current rollout length
            T_cur = min(T_max, T_start + (epoch // E_step) * delta_T)
            span = T_w + T_cur  # total frames needed per sample
            use_sparsity = (epoch >= sparsity_start_epoch)

            # Valid starting indices for this T_cur
            max_start = N_time - span
            if max_start < 1:
                raise ValueError(
                    f"Not enough data: {N_time} frames, need {span} "
                    f"(T_w={T_w} + T_cur={T_cur})")

            # Sample a limited number of random windows per epoch.
            # Stride-1 windows overlap ~95%+, so most are redundant.
            n_samples = min(max_start, batches_per_epoch * batch_size)
            all_starts = torch.randperm(max_start)[:n_samples]

            epoch_rec_loss = 0.
            epoch_total_loss = 0.
            n_batches = 0

            for bi in range(0, len(all_starts), batch_size):
                starts = all_starts[bi:bi + batch_size]
                B = len(starts)

                # Construct windows on the fly
                warmup_slices = [x_sparse_all[s:s + T_w] for s in starts]
                target_slices = [
                    x_full_all[s + T_w - 1:s + T_w - 1 + T_cur + 1]
                    for s in starts
                ]

                x_warmup = torch.stack(warmup_slices).to(device)
                x_targets = torch.stack(target_slices).to(device)

                # Forward: encode + rollout + decode
                x_hat, z_traj = model(x_warmup, T_cur)

                # Per-step reconstruction loss
                L_rec = F.mse_loss(x_hat, x_targets)

                # z_0 regularization (keeps encoder output bounded)
                L_z0 = lambda_0 * (z_traj[:, :, 0] ** 2).mean()

                # L1 sparsity on polynomial coefficients (always active)
                L_sp = 0.
                if lambda_s > 0:
                    theta = model.dynamics.rnn.unfold_polynomial_coefficients()
                    L_sp = lambda_s * (
                        theta * model.dynamics.coefficient_masks.float()
                    ).abs().mean()

                loss = L_rec + L_z0 + L_sp

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                epoch_rec_loss += L_rec.item()
                epoch_total_loss += loss.item()
                n_batches += 1

            if scheduler is not None:
                scheduler.step(epoch_rec_loss / max(n_batches, 1))

            # Pruning (after curriculum completes)
            if (use_sparsity and pruning_threshold > 0
                    and pruning_frequency > 0
                    and epoch % pruning_frequency == 0):
                    with torch.no_grad():
                        E = model.ensemble_size
                        if E > 1:
                            ensemble_prune(
                                model.dynamics, delta=pruning_threshold,
                                method=pruning_method,
                                agreement_frac=agreement_frac)
                        else:
                            threshold_patience_update(
                                model.dynamics,
                                threshold=pruning_threshold)
                            threshold_prune(model.dynamics)

            # Logging
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                avg_rec = epoch_rec_loss / n_batches
                avg_tot = epoch_total_loss / n_batches

                # Latent norm diagnostic (average ||z_t|| over trajectory)
                with torch.no_grad():
                    # Quick check on last batch's z_traj
                    z_norms = z_traj.mean(0).norm(dim=-1).mean().item()

                lr_now = optimizer.param_groups[0]['lr']
                msg = (f"Epoch {epoch:5d} | T_cur={T_cur:3d} | "
                       f"rec={avg_rec:.6f} | total={avg_tot:.6f} | "
                       f"|z|={z_norms:.3f} | lr={lr_now:.1e}")

                if use_sparsity:
                    active = model.count_active_terms()
                    msg += f" | terms={sum(active.values())}"

                # Test evaluation at T_max (stress test)
                if x_sparse_test is not None and x_full_test is not None:
                    test_loss = _eval_test(
                        model, x_sparse_test, x_full_test, T_w, T_max,
                        batch_size)
                    msg += f" | test={test_loss:.6f}"

                print(msg)

    except KeyboardInterrupt:
        if verbose:
            print(f"\nTraining interrupted at epoch {epoch}.")

    if verbose:
        print("\nDiscovered equations:")
        model.print_equations()


def _eval_test(model, x_sparse_test, x_full_test, T_w, T_max, batch_size):
    """Evaluate reconstruction loss on test time series at T_max rollout."""
    model.eval()
    device = next(model.parameters()).device
    N_test = x_sparse_test.shape[0]
    span = T_w + T_max
    max_start = N_test - span
    if max_start < 1:
        return float('nan')

    # Use evenly spaced test windows (not random)
    n_windows = min(max_start, 50)
    starts = torch.linspace(0, max_start - 1, n_windows).long()

    total_se = 0.
    total_n = 0

    with torch.no_grad():
        for bi in range(0, len(starts), batch_size):
            s_batch = starts[bi:bi + batch_size]
            x_warmup = torch.stack(
                [x_sparse_test[s:s + T_w] for s in s_batch]).to(device)
            x_targets = torch.stack(
                [x_full_test[s + T_w - 1:s + T_w - 1 + T_max + 1]
                 for s in s_batch]).to(device)

            x_hat, _ = model(x_warmup, T_max)
            total_se += F.mse_loss(
                x_hat, x_targets, reduction='sum').item()
            total_n += x_hat.numel()

    model.train()
    return total_se / total_n if total_n > 0 else float('nan')


def refit_rollout(
    model: RolloutSINDyRNN,
    x_sparse_all: Tensor,
    T_w: int,
    refit_epochs: int = 1000,
    refit_learning_rate: float = 5e-2,
    refit_l2: float = 5e-2,
    refit_pruning_threshold: float = 0.1,
    refit_pruning_frequency: int = 100,
    refit_pruning_method: str = 'agreement',
    agreement_frac: float = 0.5,
    centered_diff: bool = True,
    include_bias: bool = True,
    interaction_only: bool = False,
    verbose: bool = True,
):
    """Stage 2: Refit autonomous dynamics on frozen encoder latent trajectory.

    Freezes encoder + decoder, extracts z at every timestep, standardizes,
    then fits a fresh PolynomialRNN with derivative matching (fit()).

    The result is stored as model.autonomous_dynamics with z_mean/z_std
    buffers for forecast normalization.

    Args:
        model: trained RolloutSINDyRNN (after fit_rollout)
        x_sparse_all: (N_time, n_sensors) — training sensor time series
        T_w: warmup window length used during Stage 1
        refit_epochs: epochs for derivative matching
        refit_learning_rate: Adam lr for polynomial fitting
        refit_l2: L1 penalty on polynomial coefficients
        refit_pruning_threshold: pruning threshold delta
        refit_pruning_frequency: epochs between pruning
        refit_pruning_method: 'median' or 'agreement'
        agreement_frac: fraction of active ensemble members that must
            individually exceed refit_pruning_threshold. Only used for
            refit_pruning_method='agreement'.
        centered_diff: use centered differences for derivative estimation
        include_bias: include constant term in library
        interaction_only: exclude pure power terms
        verbose: print progress
    """
    from .training import fit as _fit_dynamics

    dynamics = model.dynamics
    device = next(model.parameters()).device
    dt = dynamics.rnn._dt.item()

    if verbose:
        print(f"\nStage 2: Discovery ({refit_epochs} epochs)")
        print(f"  Freezing encoder + decoder, extracting latent trajectory...")

    # Extract raw latent trajectory from frozen encoder (before normalization)
    z_trajectory = model.extract_latent_trajectory(x_sparse_all, T_w, raw=True)
    z_trajectory = z_trajectory.to(device)

    # Overwrite running stats with exact training statistics.
    # These are used by _normalize_z / _denormalize_z at forecast time.
    z_mean = z_trajectory.mean(0)   # (n_latent,)
    z_std = z_trajectory.std(0).clamp(min=1e-6)
    model._z_mean.copy_(z_mean)
    model._z_std.copy_(z_std)

    # Standardize trajectory for polynomial fitting
    z_trajectory = (z_trajectory - z_mean) / z_std

    if verbose:
        print(f"  Latent trajectory: {z_trajectory.shape[0]} points, "
              f"dim={z_trajectory.shape[1]}")
        print(f"  z mean: {z_mean.cpu().numpy()}, std: {z_std.cpu().numpy()}")

    # Build xs/ys for autonomous fit: z[t] -> z[t+1]
    xs_latent = z_trajectory[:-1].unsqueeze(0)  # (1, N-1, n_latent)
    ys_latent = z_trajectory[1:].unsqueeze(0)   # (1, N-1, n_latent)

    # Create fresh autonomous dynamics
    model.autonomous_dynamics = PolynomialRNN(
        n_states=model.n_latent,
        n_controls=0,
        ensemble_size=model.ensemble_size,
        polynomial_degree=dynamics.rnn._degree,
        dt=dt,
        state_names=dynamics.state_names,
        dropout=0.,
        feature_dropout=0.,
        decomposed=dynamics.rnn._decomposed,
        direct=dynamics.rnn._direct,
        compiled_forward=True,
        num_euler_steps=dynamics.rnn._num_euler_steps,
    ).to(device)

    if verbose:
        print(f"  Fitting dz/dt = P(z) with lr={refit_learning_rate:.1e}, "
              f"L1={refit_l2:.1e}, threshold={refit_pruning_threshold}")

    _fit_dynamics(
        model.autonomous_dynamics, xs_latent, ys_latent,
        epochs=refit_epochs,
        warmup_steps=0,
        learning_rate=refit_learning_rate,
        l2=refit_l2,
        pruning_threshold=refit_pruning_threshold,
        pruning_method=refit_pruning_method,
        pruning_frequency=refit_pruning_frequency,
        agreement_frac=agreement_frac,
        include_bias=include_bias,
        interaction_only=interaction_only,
        centered_diff=centered_diff,
        refit_epochs=refit_epochs // 5,
        verbose=verbose,
    )

    if verbose:
        active = model.count_active_terms()
        total_active = sum(active.values())
        print(f"\n  Discovery complete: {total_active} active terms")
        model.print_equations()
