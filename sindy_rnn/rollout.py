"""Rollout SINDy-RNN: GRU encoder -> autonomous polynomial rollout -> decoder.

Architecture:
    z_0 = GRU(x_sparse[0:T_w])              # encode warmup window
    z_{t+1} = z_t + dt * P(z_t)              # autonomous polynomial ODE
    x_hat_t = D(z_t)  for t = 0, ..., T_cur  # decode at every step

Trained end-to-end with per-step reconstruction loss and a rollout-length
curriculum that gradually increases the forecast horizon. The curriculum
prevents the optimizer from needing to stabilize long rollouts from scratch.

The decoder mirrors SINDy-SHRED's "shallow decoder network": two hidden
ReLU layers (dec_l1, dec_l2) with the same dropout as the GRU encoder, then
a final linear projection to n_full. Nonlinear, so the latent space is not
forced to be a linear (POD-like) coordinate system of the full field.

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
from tqdm import tqdm

from .model import PolynomialRNN


class RolloutSINDyRNN(nn.Module):
    """GRU encoder -> autonomous polynomial rollout -> SHRED-style decoder.

    The GRU encoder processes a warmup window of sparse sensor observations
    to produce an initial latent state z_0. The polynomial dynamics propagates
    z_0 forward autonomously, and a 2-hidden-layer decoder (matching SINDy-
    SHRED's shallow decoder network) reconstructs the full field at every
    rollout step.

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
        encoder_dropout: dropout for the GRU encoder (only has an effect
            when gru_layers > 1 — torch.nn.GRU's own inter-layer dropout)
            AND for the decoder's two hidden layers (always active there).
            Ignored when identity=True (no encoder/decoder).
        dec_l1: decoder's first hidden layer width (SINDy-SHRED default: 350)
        dec_l2: decoder's second hidden layer width (SINDy-SHRED default: 400)
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
        dec_l1: int = 350,
        dec_l2: int = 400,
        state_names: Optional[List[str]] = None,
        decomposed: bool = True,
        direct: bool = False,
        encoder_dropout: float = 0.,
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
            dropout=encoder_dropout,
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
            decomposed=decomposed,
            direct=direct,
            num_euler_steps=num_euler_steps,
        )

        # SHRED-style shallow decoder network: Linear -> Dropout -> ReLU,
        # twice, then a final linear projection. Skipped when identity=True
        # — z IS the full state.
        self.decoder = nn.Identity() if identity else nn.Sequential(
            nn.Linear(n_latent, dec_l1),
            nn.Dropout(encoder_dropout),
            nn.ReLU(),
            nn.Linear(dec_l1, dec_l2),
            nn.Dropout(encoder_dropout),
            nn.ReLU(),
            nn.Linear(dec_l2, n_full),
        )
        
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
        reduction: str = 'mean',
    ) -> tuple:
        """Forecast from a warmup window (no per-step supervision).

        Uses autonomous_dynamics (from refit) if available, otherwise
        uses the jointly-trained dynamics. Encoder z is standardized
        via running stats, denormalized before decoding.

        Args:
            x_sparse_warmup: (B, T_w, n_sensors)
            n_steps: number of forecast steps
            reduction: 'mean' (default) averages the decoded reconstruction
                over the dynamics ensemble. 'best' instead decodes only the
                single member at dyn.best_member_idx (set by
                select_best_member() — see rollout.select_best_member).
                'bic' instead decodes the single member at
                dyn.bic_member_idx (set by select_best_member_bic()).

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
            if reduction == 'best':
                x_hat = self.decoder(z_raw[dyn.best_member_idx])  # (B, n_steps, n_full)
            elif reduction == 'bic':
                x_hat = self.decoder(z_raw[dyn.bic_member_idx])  # (B, n_steps, n_full)
            else:
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

    def get_equations(self, member: Optional[int] = None, aggregate=True) -> str:
        return self._eq_dynamics.get_equations(member=member, aggregate=aggregate)

    def get_continuous_equations(self, dt: float = None, member: Optional[int] = None,
                                 aggregate=True) -> str:
        return self._eq_dynamics.get_continuous_equations(dt, member=member, aggregate=aggregate)

    def get_coefficients(self, aggregate=True, member: Optional[int] = None) -> Dict[str, Tensor]:
        return self._eq_dynamics.get_coefficients(aggregate=aggregate, member=member)

    def count_active_terms(self, member: Optional[int] = None) -> Dict[str, int]:
        return self._eq_dynamics.count_active_terms(member=member)

    def print_equations(self, member: Optional[int] = None, aggregate=True):
        self._eq_dynamics.print_equations(member=member, aggregate=aggregate)

    @property
    def best_member_idx(self):
        """Index set by rollout.select_best_member() on the active dynamics
        module — the same member simulate() uses when simulate='best'."""
        return self._eq_dynamics.best_member_idx

    @property
    def bic_member_idx(self):
        """Index set by rollout.select_best_member_bic() on the active
        dynamics module — the same member simulate() uses when
        simulate='bic'."""
        return self._eq_dynamics.bic_member_idx

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
                'dec_l1': None if self.identity else self.decoder[0].out_features,
                'dec_l2': None if self.identity else self.decoder[3].out_features,
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
    lambda_0: float = 0,
    lambda_s: float = 0,
    weight_decay: float = 0.,
    grad_clip: float = 0.5,
    batches_per_epoch: int = 4,
    pruning_threshold: float = 0.1,
    pruning_frequency: int = 100,
    pruning_method: str = 'agreement',
    agreement_frac: float = 0.5,
    ladder_exponent_step: float = 0.2,
    ladder_offset: float = -1.0,
    patience_limit: int = 2,
    lr_patience: int = 0,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    rollout_noise: float = 0.1,
    refit_epochs: int = 0,
    refit_learning_rate: Optional[float] = None,
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

    L2 shrinkage on theta (lambda_s) is always active, from epoch 0. Only
    the hard prune-to-zero step is deferred until the curriculum reaches
    T_max, avoiding the moving-target problem from pruning against a
    dynamics module that hasn't yet learned to integrate that far.

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
        lambda_s: L2 sparsity weight on theta (after curriculum completes)
        weight_decay: AdamW weight decay applied to all model parameters,
            including the dynamics/polynomial params (raw, pre-unfolding
            weights under the factored/decomposed parameterizations — this
            is on top of, not instead of, lambda_s directly on theta).
            0 = off (default).
        grad_clip: gradient clip norm
        batches_per_epoch: max mini-batches per epoch (stride-1 windows are
            heavily redundant; 2-8 batches per epoch is typically sufficient)
        pruning_threshold: pruning threshold delta
        pruning_frequency: epochs between pruning (after sparsity activation)
        pruning_method: 'median', 'agreement', or 'ladder' (per-member
            geometric threshold ladder, no cross-member vote — see
            pruning.ladder_threshold_test)
        agreement_frac: fraction of active ensemble members that must
            individually exceed pruning_threshold. Only used for method='agreement'.
        ladder_exponent_step, ladder_offset: only used for
            pruning_method='ladder'. Member e's threshold is
            pruning_threshold * 10 ** (ladder_exponent_step * e +
            ladder_offset). Defaults mirror SINDy-SHRED's E_SINDy.thresholding().
        patience_limit: consecutive failed pruning events before permanent
            removal (default 2, see CLAUDE.md §5.3). 1 = prune immediately
            on the first failure.
        lr_patience: ReduceLROnPlateau patience (0 = no scheduler)
        lr_factor: LR reduction factor on plateau
        min_lr: minimum learning rate
        refit_epochs: additional epochs at T_max with lambda_s=0, the mask
            frozen, and the encoder/decoder frozen. Debiases the dynamics
            coefficients for the already-discovered structure, which would
            otherwise stay shrunk by the ongoing L2 penalty. Continues the
            same trajectory-matching objective (unlike refit_rollout(),
            which switches to derivative matching — that would undo the
            noise-robustness of trajectory matching in identity mode).
            0 = no refit (default).
        refit_learning_rate: learning rate for refit phase
            (default: learning_rate / 5)
        x_sparse_test: (N_test, n_sensors) — optional test sensor time series
        x_full_test: (N_test, n_full) — optional test full state time series
        verbose: print training progress
    """
    from .pruning import (
        ensemble_prune, threshold_patience_update, threshold_prune,
        compute_prune_budget,
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

    # Fixed per-event pruning budget: total active terms / total scheduled
    # pruning events, computed once up front so a single noisy early event
    # can't wipe out most of the model — see pruning.compute_prune_budget.
    n_pruning_events = sum(
        1 for e in range(sparsity_start_epoch, epochs)
        if pruning_frequency > 0 and e % pruning_frequency == 0
    )
    n_active_terms = int(model.dynamics.coefficient_masks.any(dim=0).sum().item())
    max_prune = compute_prune_budget(n_active_terms, n_pruning_events)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = None
    if lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor,
            patience=lr_patience, min_lr=min_lr)

    if verbose:
        print(f"Rollout curriculum training ({epochs} epochs)")
        print(f"  T_w={T_w}, T_start={T_start}, T_max={T_max}, "
              f"delta_T={delta_T}, E_step={E_step}")
        print(f"  Pruning activates at epoch {sparsity_start_epoch}")
        print(f"  Pruning maximum {max_prune} terms per pruning event "
              f"({n_pruning_events} events, {n_active_terms} active terms)")
        print(f"  lr={learning_rate:.1e}, lambda_0={lambda_0:.1e}, "
              f"lambda_s={lambda_s:.1e}, weight_decay={weight_decay:.1e}, "
              f"grad_clip={grad_clip}")
        print(f"  N_time={N_time}, batch_size={batch_size}, "
              f"batches_per_epoch={batches_per_epoch}")

    prev_T_cur = None
    last_test_loss = None
    epoch_iter = tqdm(range(epochs), desc="Rollout training", disable=not verbose)
    try:
        for epoch in epoch_iter:
            model.train()

            # Curriculum: current rollout length
            T_cur = min(T_max, T_start + (epoch // E_step) * delta_T)
            span = T_w + T_cur  # total frames needed per sample
            use_sparsity = (epoch >= sparsity_start_epoch)

            # A curriculum step changes the training tensor shape, leaving
            # the old shape's cached blocks stranded (unusable for the new
            # size) in PyTorch's CUDA caching allocator. Defragment on each
            # step rather than letting stale blocks pile up across the run.
            if T_cur != prev_T_cur:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                prev_T_cur = T_cur

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

                # z_0 regularization (keeps encoder output bounded). Scaled
                # by E so per-member penalty strength is independent of
                # ensemble_size — plain .mean() over (E, B, n_latent) would
                # otherwise dilute the penalty by 1/E as E grows.
                E = z_traj.shape[0]
                L_z0 = lambda_0 * E * (z_traj[:, :, 0] ** 2).mean()

                # L2 sparsity on polynomial coefficients (always active).
                # Same E-scaling: penalizes each ensemble member's theta at
                # full lambda_s strength regardless of ensemble_size.
                L_sp = 0.
                if lambda_s > 0:
                    theta = model.dynamics.rnn.unfold_polynomial_coefficients()
                    # L_sp = lambda_s * theta.shape[0] * (
                    #     theta * model.dynamics.coefficient_masks.float()
                    # ).pow(2).mean()
                    L_sp = lambda_s * theta.shape[0] * (
                        theta * model.dynamics.coefficient_masks.float()
                    ).abs().mean()

                loss = L_rec + L_z0 + L_sp

                if not torch.isfinite(loss):
                    # A long autonomous rollout can transiently blow up
                    # (especially early in the curriculum, before the
                    # polynomial has learned anything, on chaotic systems
                    # like Lorenz). Applying nan/inf gradients here would
                    # permanently corrupt every weight, so skip the update
                    # instead and let training continue from the last good
                    # state — mirrors sindy_shred_net.py's own safeguard.
                    if verbose:
                        tqdm.write(f"  Non-finite loss at epoch {epoch}, batch {bi}: "
                                   f"skipping optimization step")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad()
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(total_norm):
                    # loss can look finite while backward() still produces
                    # nan/inf gradients (e.g. overflow inside the polynomial
                    # product that cancels back to a finite loss value) —
                    # the loss-only check above doesn't catch this. Skip the
                    # step instead of applying a nan/inf update to every
                    # parameter (and poisoning Adam's moment buffers for good).
                    if verbose:
                        tqdm.write(f"  Non-finite gradient at epoch {epoch}, batch {bi}: "
                                   f"skipping optimization step")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()

                epoch_rec_loss += L_rec.item()
                epoch_total_loss += loss.item()
                n_batches += 1

            if scheduler is not None and use_sparsity:
                # Gated the same as pruning: before the curriculum reaches
                # T_max, the objective itself changes every E_step epochs
                # (T_cur grows), so a rising loss reflects a harder task,
                # not a plateau. Plateau detection is only meaningful once
                # T_cur is held fixed at T_max.
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
                                agreement_frac=agreement_frac,
                                ladder_exponent_step=ladder_exponent_step,
                                ladder_offset=ladder_offset,
                                max_prune=max_prune,
                                patience_limit=patience_limit)
                        else:
                            threshold_patience_update(
                                model.dynamics,
                                threshold=pruning_threshold)
                            threshold_prune(model.dynamics, max_prune=max_prune,
                                            patience_limit=patience_limit)

            # Logging: postfix, including validation eval, updates every epoch.
            if verbose:
                avg_rec = epoch_rec_loss / max(n_batches, 1)
                avg_tot = epoch_total_loss / max(n_batches, 1)

                # Latent norm diagnostic (average ||z_t|| over trajectory)
                with torch.no_grad():
                    # Quick check on last batch's z_traj
                    z_norms = z_traj.mean(0).norm(dim=-1).mean().item()

                lr_now = optimizer.param_groups[0]['lr']
                active = model.count_active_terms()
                postfix = {
                    'T_cur': T_cur,
                    'rec': f'{avg_rec:.6f}',
                    'total': f'{avg_tot:.6f}',
                    'lr': f'{lr_now:.1e}',
                    'terms': sum(active.values()),
                    '|z|': f'{z_norms:.3f}',
                }

                # Magnitude of present (unmasked) coefficients — diagnoses
                # penalty shrinkage / bifurcation independent of term count.
                with torch.no_grad():
                    theta = model.dynamics.rnn.unfold_polynomial_coefficients()
                    theta_active = theta[model.dynamics.coefficient_masks].abs()
                    if theta_active.numel() > 0:
                        postfix['|c|_max'] = f'{theta_active.max().item():.3f}'
                        postfix['|c|_mean'] = f'{theta_active.mean().item():.3f}'

                # Validation evaluation at T_max (stress test). Defragment
                # before and after — this forces one large contiguous
                # allocation at a size training itself may not have used yet.
                if x_sparse_test is not None and x_full_test is not None:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    last_test_loss = _eval_test(
                        model, x_sparse_test, x_full_test, T_w, T_max,
                        batch_size)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if last_test_loss is not None:
                    postfix['val'] = f'{last_test_loss:.6f}'

                epoch_iter.set_postfix(postfix)

    except KeyboardInterrupt:
        if verbose:
            print(f"\nTraining interrupted at epoch {epoch}.")

    # Post-pruning refit: debias dynamics coefficients for the
    # already-discovered structure. Freezes encoder/decoder (avoids a moving
    # target — the polynomial would otherwise be debiasing against a
    # shifting z) and continues the same trajectory-matching objective with
    # lambda_s=0 and the mask frozen (no more pruning calls).
    if refit_epochs > 0:
        if model.encoder is not None:
            for p in model.encoder.parameters():
                p.requires_grad_(False)
        for p in model.decoder.parameters():
            p.requires_grad_(False)

        refit_lr = refit_learning_rate if refit_learning_rate is not None else learning_rate / 5
        refit_optimizer = torch.optim.AdamW(model.dynamics.parameters(), lr=refit_lr)
        refit_scheduler = None
        if lr_patience > 0:
            refit_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                refit_optimizer, mode='min', factor=lr_factor,
                patience=lr_patience, min_lr=min_lr)

        if verbose:
            active = model.count_active_terms()
            total_active = sum(active.values())
            print(f"\nRefit phase: {refit_epochs} epochs, lr={refit_lr:.1e}, "
                  f"lambda_s=0, mask + encoder/decoder frozen "
                  f"({total_active} active terms)")

        span = T_w + T_cur
        max_start = N_time - span

        refit_iter = tqdm(range(refit_epochs), desc="Refit", disable=not verbose)
        try:
            for r_epoch in refit_iter:
                model.train()

                n_samples = min(max_start, batches_per_epoch * batch_size)
                all_starts = torch.randperm(max_start)[:n_samples]

                epoch_rec_loss = 0.
                n_batches = 0

                for bi in range(0, len(all_starts), batch_size):
                    starts = all_starts[bi:bi + batch_size]
                    warmup_slices = [x_sparse_all[s:s + T_w] for s in starts]
                    target_slices = [
                        x_full_all[s + T_w - 1:s + T_w - 1 + T_cur + 1]
                        for s in starts
                    ]
                    x_warmup = torch.stack(warmup_slices).to(device)
                    x_targets = torch.stack(target_slices).to(device)

                    x_hat, z_traj = model(x_warmup, T_cur)
                    loss = F.mse_loss(x_hat, x_targets)

                    if not torch.isfinite(loss):
                        if verbose:
                            tqdm.write(f"  Non-finite loss in refit epoch {r_epoch}: "
                                       f"skipping optimization step")
                        refit_optimizer.zero_grad(set_to_none=True)
                        continue

                    refit_optimizer.zero_grad()
                    loss.backward()
                    total_norm = torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(), grad_clip)
                    if not torch.isfinite(total_norm):
                        if verbose:
                            tqdm.write(f"  Non-finite gradient in refit epoch {r_epoch}: "
                                       f"skipping optimization step")
                        refit_optimizer.zero_grad(set_to_none=True)
                        continue
                    refit_optimizer.step()

                    epoch_rec_loss += loss.item()
                    n_batches += 1

                avg_rec = epoch_rec_loss / max(n_batches, 1)
                if refit_scheduler is not None:
                    refit_scheduler.step(avg_rec)

                if verbose:
                    postfix = {
                        'T_cur': T_cur, 'rec': f'{avg_rec:.6f}',
                        'lr': f'{refit_optimizer.param_groups[0]["lr"]:.1e}',
                    }
                    with torch.no_grad():
                        theta = model.dynamics.rnn.unfold_polynomial_coefficients()
                        theta_active = theta[model.dynamics.coefficient_masks].abs()
                        if theta_active.numel() > 0:
                            postfix['|c|_max'] = f'{theta_active.max().item():.3f}'
                            postfix['|c|_mean'] = f'{theta_active.mean().item():.3f}'
                    refit_iter.set_postfix(postfix)
        except KeyboardInterrupt:
            if verbose:
                print(f"\nRefit interrupted at epoch {r_epoch}.")

    # Rank ensemble members by their own full-state reconstruction fit —
    # predict()'s same-timestep reconstruction is identical across members
    # (encode() broadcasts one shared z_0 to all E copies), so only the
    # multi-step autonomous rollout differentiates them. Evaluated at T_max
    # since that's the horizon simulate()/forecast() actually uses. Prefer
    # held-out data when given — the training set alone would favor
    # whichever member simply overfits hardest.
    if x_sparse_test is not None and x_full_test is not None:
        select_best_member(model, x_sparse_test, x_full_test, T_w, T_max, batch_size)
    else:
        select_best_member(model, x_sparse_all, x_full_all, T_w, T_max, batch_size)

    # BIC ranking is a separate, parallel selection ('simulate: bic') —
    # its sparsity penalty only means anything on the data the model was
    # actually fit to, so always use the training set here (never test).
    select_best_member_bic(model, x_sparse_all, x_full_all, T_w, T_max, batch_size)

    if verbose:
        print("\nDiscovered equations:")
        model.print_equations()


def select_best_member(model: RolloutSINDyRNN, x_sparse_all: Tensor,
                       x_full_all: Tensor, T_w: int, T_eval: int,
                       batch_size: int = 32, n_windows: int = 50):
    """Evaluate each ensemble member's own autonomous-rollout reconstruction
    fit on training data and store the best (lowest-MSE) member's index on
    the active dynamics module (model._eq_dynamics.best_member_idx).

    Mirrors forward()'s/forecast()'s rollout + decode mechanics but skips
    the .mean(0) ensemble reduction, so every member's decoded
    reconstruction is scored independently. Called once at the end of
    fit_rollout()/refit_rollout() so analyze.py can switch between the
    ensemble mean and this single best-fitting member (config
    `simulate: mean|best`) without retraining.
    """
    model.eval()
    dyn = model._eq_dynamics
    device = next(model.parameters()).device
    E = model.ensemble_size
    N = x_sparse_all.shape[0]
    span = T_w + T_eval
    max_start = N - span
    if max_start < 1:
        return

    n_windows = min(max_start, n_windows)
    starts = torch.linspace(0, max_start - 1, n_windows).long()

    total_se = torch.zeros(E, device=device)
    total_n = 0

    with torch.no_grad():
        for bi in range(0, len(starts), batch_size):
            s_batch = starts[bi:bi + batch_size]
            x_warmup = torch.stack(
                [x_sparse_all[s:s + T_w] for s in s_batch]).to(device)
            x_targets = torch.stack(
                [x_full_all[s + T_w - 1:s + T_w - 1 + T_eval + 1]
                 for s in s_batch]).to(device)

            z = model.encode(x_warmup)  # (E, B, n_latent)
            if dyn is not model.dynamics:
                # autonomous_dynamics (post-refit) was fit on normalized z —
                # mirrors forecast()'s explicit normalize call.
                z = model._normalize_z(z)

            theta = dyn.rnn.unfold_polynomial_coefficients()
            theta_masked = theta * dyn.coefficient_masks.float()
            z_stack = model._rollout(z, T_eval, theta_masked, include_init=True)
            z_raw = model._denormalize_z(z_stack)

            # Decode one ensemble member at a time — decoding all E at once
            # (E, B, T_eval+1, n_full) can be huge for large full_dim (e.g.
            # SST's 44219): with E=10 that's a 10x larger allocation than
            # needed, since each member's squared error is independent.
            for e in range(E):
                x_hat_e = model.decoder(z_raw[e])  # (B, T_eval+1, n_full)
                total_se[e] += ((x_hat_e - x_targets) ** 2).sum()
            total_n += x_targets.numel()

    model.train()
    if total_n > 0:
        member_loss = total_se / total_n
        dyn.best_member_idx.fill_(int(torch.argmin(member_loss).item()))


def select_best_member_bic(model: RolloutSINDyRNN, x_sparse_all: Tensor,
                           x_full_all: Tensor, T_w: int, T_eval: int,
                           batch_size: int = 32, n_windows: int = 50):
    """Score each ensemble member by BIC on its own autonomous-rollout
    reconstruction fit and store the lowest-BIC member's index on the
    active dynamics module (model._eq_dynamics.bic_member_idx).

    Same rollout mechanics as select_best_member() (same windows, same
    decoded reconstruction), but BIC = n*ln(RSS/n) + k*ln(n) trades pure
    fit for fit penalized by the member's own active term count — meant
    to run on training data, where select_best_member()'s ranking alone
    would favor whichever member simply overfits hardest.

    Returns:
        (E,) tensor of per-member BIC scores, or None if there wasn't
        enough data to evaluate any window.
    """
    model.eval()
    dyn = model._eq_dynamics
    device = next(model.parameters()).device
    E = model.ensemble_size
    N = x_sparse_all.shape[0]
    span = T_w + T_eval
    max_start = N - span
    if max_start < 1:
        return None

    n_windows = min(max_start, n_windows)
    starts = torch.linspace(0, max_start - 1, n_windows).long()

    total_se = torch.zeros(E, device=device)
    total_elems = 0
    total_frames = 0

    with torch.no_grad():
        for bi in range(0, len(starts), batch_size):
            s_batch = starts[bi:bi + batch_size]
            x_warmup = torch.stack(
                [x_sparse_all[s:s + T_w] for s in s_batch]).to(device)
            x_targets = torch.stack(
                [x_full_all[s + T_w - 1:s + T_w - 1 + T_eval + 1]
                 for s in s_batch]).to(device)

            z = model.encode(x_warmup)  # (E, B, n_latent)
            if dyn is not model.dynamics:
                z = model._normalize_z(z)

            theta = dyn.rnn.unfold_polynomial_coefficients()
            theta_masked = theta * dyn.coefficient_masks.float()
            z_stack = model._rollout(z, T_eval, theta_masked, include_init=True)
            z_raw = model._denormalize_z(z_stack)

            # Decode one ensemble member at a time — see select_best_member().
            for e in range(E):
                x_hat_e = model.decoder(z_raw[e])  # (B, T_eval+1, n_full)
                total_se[e] += ((x_hat_e - x_targets) ** 2).sum()
            total_elems += x_targets.numel()
            total_frames += x_targets.shape[0] * x_targets.shape[1]

    model.train()
    if total_elems > 0:
        # n = number of time-domain samples (frames), NOT multiplied by
        # full_dim — mirrors sindy-shred.py's auto_tune_threshold(), whose
        # n_samples is len(x_train) (time steps only) even though its mse
        # is averaged over all latent dims. Multiplying n by full_dim (as
        # a naive per-scalar-residual BIC would) makes n*log(mse) swamp
        # k*log(n) by orders of magnitude on high-dim reconstructions,
        # making the sparsity penalty meaningless.
        mse = total_se / total_elems
        n = torch.full((E,), float(total_frames), device=device)
        k = torch.tensor(
            [sum(dyn.count_active_terms(member=e).values()) for e in range(E)],
            dtype=torch.float32, device=device,
        )
        bic = n * torch.log(mse) + k * torch.log(n)
        dyn.bic_member_idx.fill_(int(torch.argmin(bic).item()))
        return bic
    return None


def _eval_test(model, x_sparse_test, x_full_test, T_w, T_max, batch_size):
    """Evaluate reconstruction loss on held-out time series, rolled out as
    far as T_max or the available data allows, whichever is shorter (the
    held-out series passed in for training-time monitoring — typically a
    validation buffer — may be much shorter than T_max, which also doubles
    as the curriculum target)."""
    model.eval()
    device = next(model.parameters()).device
    N_test = x_sparse_test.shape[0]
    T_max = min(T_max, N_test - T_w - 1)
    if T_max < 1:
        return float('nan')
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
    ladder_exponent_step: float = 0.2,
    ladder_offset: float = -1.0,
    patience_limit: int = 2,
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
        refit_l2: L2 penalty on polynomial coefficients
        refit_pruning_threshold: pruning threshold delta
        refit_pruning_frequency: epochs between pruning
        refit_pruning_method: 'median', 'agreement', or 'ladder' (per-member
            geometric threshold ladder, no cross-member vote)
        agreement_frac: fraction of active ensemble members that must
            individually exceed refit_pruning_threshold. Only used for
            refit_pruning_method='agreement'.
        ladder_exponent_step, ladder_offset: only used for
            refit_pruning_method='ladder'. See training.fit().
        patience_limit: consecutive failed pruning events before permanent
            removal (default 2, see CLAUDE.md §5.3). 1 = prune immediately
            on the first failure.
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
        decomposed=dynamics.rnn._decomposed,
        direct=dynamics.rnn._direct,
        compiled_forward=True,
        num_euler_steps=dynamics.rnn._num_euler_steps,
    ).to(device)

    if verbose:
        print(f"  Fitting dz/dt = P(z) with lr={refit_learning_rate:.1e}, "
              f"L2={refit_l2:.1e}, threshold={refit_pruning_threshold}")

    _fit_dynamics(
        model.autonomous_dynamics, xs_latent, ys_latent,
        epochs=refit_epochs,
        warmup_steps=0,
        learning_rate=refit_learning_rate,
        lambda_s=refit_l2,
        pruning_threshold=refit_pruning_threshold,
        pruning_method=refit_pruning_method,
        pruning_frequency=refit_pruning_frequency,
        agreement_frac=agreement_frac,
        ladder_exponent_step=ladder_exponent_step,
        ladder_offset=ladder_offset,
        patience_limit=patience_limit,
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
