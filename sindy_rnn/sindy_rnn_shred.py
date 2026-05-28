"""SINDy-RNN-SHRED: MLP encoder + polynomial ODE temporal integration + SINDy regularizer.

Architecture:
    u[t] = MLP(x_sparse[t])                         # per-timestep encoding
    z[0] = 0
    z[t+1] = z[t] + dt * P_forced(z[t], u[t])       # polynomial ODE with forcing
    X_full = decoder(z[-1])                          # decode final latent state

Following SINDy-SHRED (Gao et al.), an autonomous SINDy module is trained
simultaneously during stage 1 as a regularizer:
    z_pred[t+1] = z[t] + dt * P_auto(z[t])          # autonomous polynomial dynamics
    L_sindy = MSE(z_pred[t+1], z_enc[t+1])          # regularization loss

Total loss = L_recon + sindy_reg * L_sindy + L1 * |theta_sindy| + 0.1 * |mean(z)|

This couples the encoder to polynomial dynamics constraints, ensuring the
latent space is shaped to follow sparse polynomial dynamics from the start.

Stage 2 (optional refit) discovers clean autonomous equations on the fixed
latent trajectories using derivative matching.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import PolynomialRNN
from .autoencoder import MLPEncoder, MLPDecoder


class SINDyRNNSHRED(nn.Module):
    """MLP encoder + polynomial ODE temporal integrator + MLP decoder + SINDy regularizer.

    Architecture:
        u[t] = MLP_encoder(x_sparse[t])              # per-timestep
        z[t+1] = z[t] + dt * P(z[t], u[t])           # polynomial ODE + forcing
        X_full = MLP_decoder(z[-1])                   # decode final state

    The sindy_module is an autonomous PolynomialRNN that predicts z[t+1] from
    z[t] without forcing. Trained simultaneously with the encoder-decoder,
    its loss regularizes the latent space to follow polynomial dynamics.

    The encoder and decoder are shared across ensemble members. Both forced
    dynamics and sindy_module have E independent members.

    Args:
        sparse_dim: dimension of sparse measurement vector
        full_dim: dimension of full state (decoder output)
        latent_dim: dimension of latent state z and control signal u
        ensemble_size: number of independent ensemble members
        polynomial_degree: degree for polynomial ODE
        dt: timestep for forward Euler integration
        encoder_hidden_dims: MLP encoder hidden widths
        decoder_hidden_dims: decoder MLP hidden widths
        encoder_dropout: dropout for encoder
        decoder_dropout: dropout for decoder MLP
        dynamics_dropout: dropout for polynomial layer
        dynamics_feature_dropout: feature dropout for polynomial layer
        state_names: names for latent state variables
        decomposed: use decomposed polynomial layer (for forced dynamics)
        direct: use direct parameterization (for forced dynamics)
        num_euler_steps: sub-steps per dt interval
    """

    def __init__(
        self,
        sparse_dim: int,
        full_dim: int,
        latent_dim: int,
        ensemble_size: int = 1,
        polynomial_degree: int = 2,
        dt: float = 1.0,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        encoder_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
        dynamics_dropout: float = 0.,
        dynamics_feature_dropout: float = 0.,
        state_names: Optional[List[str]] = None,
        decomposed: bool = True,
        direct: bool = False,
        num_euler_steps: int = 1,
    ):
        super().__init__()
        self.sparse_dim = sparse_dim
        self.full_dim = full_dim
        self.latent_dim = latent_dim

        encoder_hidden_dims = encoder_hidden_dims or [128, 64]

        # Per-timestep MLP: sparse sensors -> control signal u
        self.encoder = MLPEncoder(
            sparse_dim, latent_dim, encoder_hidden_dims, encoder_dropout)

        # MLP decoder: latent -> full state
        self.decoder = MLPDecoder(
            latent_dim, full_dim, decoder_hidden_dims, decoder_dropout)

        # Polynomial ODE with forcing: dz/dt = P(z, u)
        # u has same dim as z (n_controls = latent_dim)
        state_names = state_names or [f'z_{i}' for i in range(latent_dim)]
        control_names = [f'u_{i}' for i in range(latent_dim)]

        self.dynamics = PolynomialRNN(
            n_states=latent_dim,
            n_controls=latent_dim,  # MLP outputs serve as control signals
            ensemble_size=ensemble_size,
            polynomial_degree=polynomial_degree,
            dt=dt,
            state_names=state_names,
            control_names=control_names,
            dropout=dynamics_dropout,
            feature_dropout=dynamics_feature_dropout,
            decomposed=decomposed,
            direct=direct,
            compiled_forward=True,  # no pruning on forced dynamics — compile for speed
            num_euler_steps=num_euler_steps,
        )

        # Autonomous SINDy regularizer (E_SINDy module): dz/dt = P_auto(z)
        # - Direct parameterization (coefficients are nn.Parameters, like SINDy-SHRED)
        # - No controls (autonomous)
        # - No dropout (regularized via L1 + thresholding)
        self.sindy_module = PolynomialRNN(
            n_states=latent_dim,
            n_controls=0,  # autonomous
            ensemble_size=ensemble_size,
            polynomial_degree=polynomial_degree,
            dt=dt,
            state_names=state_names,
            dropout=0.,
            feature_dropout=0.,
            decomposed=False,
            direct=True,
            compiled_forward=False,
            num_euler_steps=num_euler_steps,
        )
        # Match SINDy-SHRED's small init (std=0.001 vs our default 0.01)
        with torch.no_grad():
            self.sindy_module.rnn.theta.data.normal_(0, 0.001)

        self._compiled_forward = None

    @property
    def ensemble_size(self):
        return self.dynamics.ensemble_size

    @property
    def n_states(self):
        return self.dynamics.n_states

    def _forward_impl(self, sparse_obs: Tensor):
        """Core forward: encode -> integrate ODE loop -> decode.

        Uses rnn.forward() (basic polynomial layer, no unfolding/masking)
        instead of forward_polynomial. Pruning is only on the sindy_module,
        not the forced dynamics. rnn.forward() is compiled for speed.
        """
        B, T, _ = sparse_obs.shape
        E = self.dynamics.ensemble_size
        device = sparse_obs.device

        # Encode each timestep independently (shared across ensemble)
        u_all = self.encoder(sparse_obs)  # (B, T, latent_dim)

        # Expand for ensemble
        u_exp = u_all.unsqueeze(0).expand(E, -1, -1, -1)  # (E, B, T, latent_dim)

        # Initialize z = 0
        z = torch.zeros(E, B, self.latent_dim, device=device)

        # Basic forward: uses compiled polynomial layer, no masking
        rnn = self.dynamics.rnn
        for t in range(T):
            z = rnn(z, u_exp[:, :, t, :])

        # Decode final state
        full_pred = self.decoder(z)  # (E, B, full_dim)

        return full_pred, z, u_all

    def forward(self, sparse_obs: Tensor):
        """Forward pass: encode sensors per-timestep, integrate polynomial ODE, decode.

        Args:
            sparse_obs: (B, T, sparse_dim) — sensor window

        Returns:
            full_pred: (E, B, full_dim) — decoded full state per ensemble member
            z_final: (E, B, latent_dim) — final latent state per ensemble member
            u_all: (B, T, latent_dim) — MLP control signals (shared across ensemble)
        """
        if self._compiled_forward is not None:
            return self._compiled_forward(sparse_obs)
        return self._forward_impl(sparse_obs)

    def extract_latent_trajectory(self, sparse_obs: Tensor, chunk_size: int = 64):
        """Extract latent z at each timestep from sliding windows.

        Processes all windows and returns the z value at the last timestep
        of each window, forming a continuous trajectory.

        Args:
            sparse_obs: (N, LAGS, sparse_dim) — all sliding windows (stride=1)
            chunk_size: number of windows to process at once

        Returns:
            z_trajectory: (N, latent_dim) — one z per window
        """
        N = sparse_obs.shape[0]
        self.eval()
        z_list = []

        with torch.no_grad():
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                batch = sparse_obs[start:end]
                _, z_final, _ = self.forward(batch)  # (E, B, latent_dim)
                z_list.append(z_final.mean(0).cpu())  # average across ensemble

        return torch.cat(z_list, dim=0)  # (N, latent_dim)

    def forecast(self, z_init: Tensor, n_steps: int,
                 integrator: str = 'rk4') -> tuple:
        """Autonomous forecast using discovered dynamics (no sensor input).

        Uses sindy_module (stage 1) or autonomous_dynamics (stage 2 refit).

        Args:
            z_init: (E, B, latent_dim) — initial latent state
            n_steps: number of steps to forecast
            integrator: 'euler' or 'rk4'

        Returns:
            full_traj: (E, B, n_steps, full_dim) — decoded forecast
            latent_traj: (E, B, n_steps, latent_dim) — latent forecast
        """
        dyn = self._eq_dynamics
        theta = dyn.rnn.unfold_polynomial_coefficients()

        # Standardize init if stage 2 used standardization
        z_mean = getattr(self, 'z_mean', None)
        z_std = getattr(self, 'z_std', None)
        h = (z_init - z_mean) / z_std if z_mean is not None else z_init

        trajectory = []
        for t in range(n_steps):
            h = dyn.rnn.forward_polynomial(
                h, None, mask=dyn.coefficient_masks, theta=theta,
                integrator=integrator)
            trajectory.append(h)

        latent_traj = torch.stack(trajectory, dim=2)  # (E, B, n_steps, latent_dim)

        # Un-standardize before decoding
        if z_mean is not None:
            latent_traj_raw = latent_traj * z_std + z_mean
        else:
            latent_traj_raw = latent_traj

        full_traj = self.decoder(latent_traj_raw)
        return full_traj, latent_traj_raw

    # --- Delegation: equations come from autonomous_dynamics or sindy_module ---

    @property
    def _eq_dynamics(self):
        """Return autonomous dynamics (refit) if available, else sindy_module."""
        return getattr(self, 'autonomous_dynamics', self.sindy_module)

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


def fit_sindy_rnn_shred(
    model: SINDyRNNSHRED,
    sparse_obs: Tensor,
    full_state_target: Tensor,
    sparse_obs_test: Optional[Tensor] = None,
    full_state_target_test: Optional[Tensor] = None,
    epochs: int = 1000,
    batch_size: Optional[int] = None,
    learning_rate: float = 1e-3,
    dynamics_learning_rate: Optional[float] = None,
    sindy_regularization: float = 10.0,
    sindy_warmup_epochs: int = 100,
    l2: float = 1e-3,
    pruning_threshold: float = 0.1,
    pruning_frequency: int = 100,
    pruning_method: str = 'median',
    refit_epochs: int = 1000,
    refit_learning_rate: Optional[float] = None,
    refit_l2: float = 5e-2,
    refit_pruning_threshold: Optional[float] = 0.1,
    refit_pruning_frequency: int = 100,
    refit_pruning_method: str = 'median',
    refit_stability_weight: float = 0.0,
    ensemble_pruning_alpha: float = 0.05,
    dt: Optional[float] = None,
    centered_diff: bool = True,
    include_bias: bool = True,
    interaction_only: bool = False,
    verbose: bool = True,
):
    """Train SINDyRNNSHRED with simultaneous SINDy regularization.

    Stage 1: Train encoder + forced ODE + decoder on reconstruction, while
    simultaneously training an autonomous sindy_module as a regularizer.
    Following SINDy-SHRED, the SINDy loss MSE(z_sindy_pred, z_encoder)
    couples the encoder to polynomial dynamics constraints.

    Stage 2 (optional): Freeze encoder + decoder, extract latent z, refit
    autonomous ODE with derivative matching for clean equation discovery.

    Args:
        model: SINDyRNNSHRED instance
        sparse_obs: (N, LAGS, sparse_dim) — sliding sensor windows, stride=1
        full_state_target: (N, full_dim) — same-timestep reconstruction target
        sparse_obs_test: optional test windows
        full_state_target_test: optional test targets
        epochs: stage 1 training epochs (each epoch = full pass over data)
        batch_size: mini-batch size (None = full batch)
        learning_rate: Adam lr for encoder/decoder
        dynamics_learning_rate: Adam lr for forced dynamics and sindy_module
        sindy_regularization: weight for SINDy dynamics loss (SINDy-SHRED default: 10.0)
        sindy_warmup_epochs: epochs before SINDy regularization + L1 + pruning start
        l2: L1 penalty weight on sindy_module coefficients
        pruning_threshold: pruning threshold for sindy_module (delta)
        pruning_frequency: epochs between pruning of sindy_module
        pruning_method: 'median' or 'ci' for sindy_module pruning
        refit_epochs: stage 2 refit epochs (0 = skip, use sindy_module directly)
        refit_learning_rate: lr for stage 2 (default: 5e-2)
        refit_l2: L1 penalty weight for stage 2
        refit_pruning_threshold: pruning threshold for stage 2
        refit_pruning_frequency: epochs between pruning in stage 2
        refit_pruning_method: 'median' or 'ci'
        refit_stability_weight: stability penalty weight for stage 2
        ensemble_pruning_alpha: confidence level for CI pruning
        dt: timestep between consecutive windows
        centered_diff: use centered differences in stage 2
        include_bias: include constant term in library
        interaction_only: exclude pure power terms
        verbose: print progress
    """
    from .pruning import ensemble_prune

    dynamics = model.dynamics
    sindy = model.sindy_module
    E = dynamics.ensemble_size
    E_sindy = sindy.ensemble_size
    N = sparse_obs.shape[0]

    # Three param groups: encoder/decoder, forced dynamics, sindy_module
    dynamics_lr = dynamics_learning_rate or learning_rate
    sindy_param_ids = set(id(p) for p in sindy.parameters())
    dynamics_param_ids = set(id(p) for p in dynamics.parameters())
    encdec_params = [p for p in model.parameters()
                     if id(p) not in sindy_param_ids
                     and id(p) not in dynamics_param_ids]
    optimizer = torch.optim.Adam([
        {'params': encdec_params, 'lr': learning_rate},
        {'params': list(dynamics.parameters()), 'lr': dynamics_lr},
        {'params': list(sindy.parameters()), 'lr': dynamics_lr},
    ])

    # Effective batch size: +1 for consecutive pairs needed by SINDy loss
    bs = batch_size or N
    if bs >= N:
        bs = N

    def _test_loss():
        """Compute test reconstruction loss in chunks."""
        model.eval()
        N_te = sparse_obs_test.shape[0]
        total_se = 0.
        total_n = 0
        chunk = 64
        for start in range(0, N_te, chunk):
            end = min(start + chunk, N_te)
            full_pred, _, _ = model(sparse_obs_test[start:end])
            decoded = full_pred.mean(0)
            target = full_state_target_test[start:end]
            total_se += F.mse_loss(decoded, target, reduction='sum').item()
            total_n += decoded.numel()
            del decoded, full_pred
        return total_se / total_n

    # ================================================================
    # Stage 1: Reconstruction + SINDy regularization
    # ================================================================
    if verbose:
        print(f"Stage 1: Reconstruction + SINDy regularizer "
              f"({epochs} epochs, lr={learning_rate:.1e})")
        print(f"  sindy_reg={sindy_regularization}, warmup={sindy_warmup_epochs}, "
              f"L1={l2:.1e}")
        print(f"  pruning: threshold={pruning_threshold}, "
              f"freq={pruning_frequency}, method={pruning_method}")

    try:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            sindy_active = (sindy_regularization > 0
                            and epoch >= sindy_warmup_epochs)

            # Sample one random contiguous chunk per epoch (consecutive for SINDy pairs)
            if bs < N - 1:
                start = torch.randint(0, N - bs, (1,)).item()
                obs_batch = sparse_obs[start:start + bs + 1]     # +1 for pairs
                target_batch = full_state_target[start:start + bs + 1]
            else:
                obs_batch = sparse_obs
                target_batch = full_state_target

            # Forward pass through encoder + forced ODE + decoder
            full_pred, z_final, _ = model(obs_batch)

            # Reconstruction loss — average across ensemble members
            recon_loss = F.mse_loss(full_pred.mean(0), target_batch)
            loss = recon_loss

            # SINDy regularization (after warmup)
            sindy_loss_val = 0.
            if sindy_active:
                # Average z across forced dynamics ensemble
                z_avg = z_final.mean(0)  # (B', latent_dim)

                # Standardize z for sindy module (batch stats, detached).
                # Naive standardization has gradient d(z_norm)/d(z_avg) = 1/z_sig,
                # which explodes when z is small (early training), drowning out
                # reconstruction gradients. Use detach trick to cancel 1/z_sig:
                #   forward: z_scaled = z_avg (identity)
                #   backward: d(z_scaled)/d(z_avg) = z_sig, so d(z_norm)/d(z_avg) = 1
                with torch.no_grad():
                    z_mu = z_avg.mean(0, keepdim=True)
                    z_sig = z_avg.std(0, keepdim=True).clamp(min=1e-6)
                z_scaled = z_avg.detach() + z_sig * (z_avg - z_avg.detach())
                z_norm = (z_scaled - z_mu) / z_sig

                z_start = z_norm[:-1]    # (B'-1, latent_dim) — z[t]
                z_target = z_norm[1:]    # (B'-1, latent_dim) — z[t+1]

                # Expand for sindy ensemble
                z_start_exp = z_start.unsqueeze(0).expand(E_sindy, -1, -1)
                z_target_exp = z_target.unsqueeze(0).expand(E_sindy, -1, -1)

                # Autonomous SINDy prediction: z_pred = z[t] + dt * P(z[t])
                theta_s = sindy.rnn.unfold_polynomial_coefficients()
                theta_masked = theta_s * sindy.coefficient_masks.float()
                z_pred = sindy.rnn.forward_polynomial(
                    z_start_exp, None, mask=None, theta=theta_masked)

                sindy_loss = F.mse_loss(
                    z_pred, z_target_exp) * sindy_regularization
                loss = loss + sindy_loss
                sindy_loss_val = sindy_loss.item()

            # L1 on sindy_module coefficients (after warmup)
            l1_loss_val = 0.
            if l2 > 0 and sindy_active:
                theta_s = sindy.rnn.unfold_polynomial_coefficients()
                l1_loss = l2 * (
                    theta_s * sindy.coefficient_masks.float()
                ).abs().mean()
                loss = loss + l1_loss
                l1_loss_val = l1_loss.item()

            # Latent centering penalty (SINDy-SHRED style)
            loss = loss + torch.abs(z_final.mean()) * 0.1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=100.0)
            optimizer.step()

            # Periodic pruning of sindy_module (after warmup + one pruning cycle)
            sindy_epochs_elapsed = epoch - sindy_warmup_epochs
            if (pruning_threshold > 0 and pruning_frequency > 0
                    and sindy_active
                    and sindy_epochs_elapsed >= pruning_frequency
                    and sindy_epochs_elapsed % pruning_frequency == 0):
                with torch.no_grad():
                    ensemble_prune(
                        sindy, delta=pruning_threshold,
                        method=pruning_method, alpha=ensemble_pruning_alpha)

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                msg = (f"Epoch {epoch:4d} | "
                       f"recon {recon_loss.item():.6f}")
                if sindy_active:
                    msg += f" | sindy {sindy_loss_val:.6f}"
                    msg += f" | L1 {l1_loss_val:.6f}"
                if (sparse_obs_test is not None
                        and full_state_target_test is not None):
                    with torch.no_grad():
                        loss_te = _test_loss()
                        msg += f" | test {loss_te:.6f}"
                if sindy_active and pruning_threshold > 0:
                    active = sindy.count_active_terms()
                    msg += f" | terms {sum(active.values())}"
                print(msg)

    except KeyboardInterrupt:
        if verbose:
            print(f"\nStage 1 interrupted at epoch {epoch}.")

    if verbose:
        print("\nStage 1 SINDy module equations:")
        sindy.print_equations()

    # Store global z stats for forecasting (sindy_module trained on standardized z)
    # If stage 2 runs, refit_autonomous() will overwrite these.
    with torch.no_grad():
        z_traj = model.extract_latent_trajectory(sparse_obs)
        z_mean = z_traj.mean(0).to(sparse_obs.device)
        z_std = z_traj.std(0).clamp(min=1e-6).to(sparse_obs.device)
        model.register_buffer('z_mean', z_mean)
        model.register_buffer('z_std', z_std)
        if verbose:
            print(f"  z stats: mean={z_mean.cpu().numpy()}, std={z_std.cpu().numpy()}")

    # ================================================================
    # Stage 2: Discovery — refit autonomous dynamics on fixed latent space
    # ================================================================
    if refit_epochs > 0:
        del optimizer
        refit_autonomous(
            model, sparse_obs,
            refit_epochs=refit_epochs,
            refit_learning_rate=refit_learning_rate,
            refit_l2=refit_l2,
            refit_pruning_threshold=refit_pruning_threshold,
            refit_pruning_frequency=refit_pruning_frequency,
            refit_pruning_method=refit_pruning_method,
            refit_stability_weight=refit_stability_weight,
            ensemble_pruning_alpha=ensemble_pruning_alpha,
            dt=dt,
            centered_diff=centered_diff,
            include_bias=include_bias,
            interaction_only=interaction_only,
            verbose=verbose,
        )


def refit_autonomous(
    model: SINDyRNNSHRED,
    sparse_obs: Tensor,
    refit_epochs: int = 1000,
    refit_learning_rate: Optional[float] = None,
    refit_l2: float = 5e-2,
    refit_pruning_threshold: Optional[float] = 0.1,
    refit_pruning_frequency: int = 100,
    refit_pruning_method: str = 'median',
    refit_stability_weight: float = 0.0,
    ensemble_pruning_alpha: float = 0.05,
    dt: Optional[float] = None,
    centered_diff: bool = True,
    include_bias: bool = True,
    interaction_only: bool = False,
    verbose: bool = True,
):
    """Rerun stage 2 (autonomous discovery) on a trained SINDyRNNSHRED model.

    Freezes encoder + decoder, extracts latent z trajectories from sparse_obs,
    creates a fresh autonomous PolynomialRNN (n_controls=0), and fits it with
    L1 + pruning. The result is stored as ``model.autonomous_dynamics``.

    This is useful for:
    - Loading a model saved after stage 1 and running discovery separately
    - Re-running discovery with different L1/pruning hyperparameters

    Args:
        model: trained SINDyRNNSHRED (encoder + decoder + forced dynamics)
        sparse_obs: (N, LAGS, sparse_dim) — sliding sensor windows (stride=1)
        refit_epochs: training epochs for autonomous discovery
        refit_learning_rate: Adam lr (default: 5e-2)
        refit_l2: L1 penalty weight on polynomial coefficients
        refit_pruning_threshold: pruning threshold (delta)
        refit_pruning_frequency: epochs between pruning
        refit_pruning_method: 'median' or 'ci'
        refit_stability_weight: stability penalty weight
        ensemble_pruning_alpha: confidence level for CI pruning
        dt: timestep between consecutive windows
        centered_diff: use centered differences for derivative estimation
        include_bias: include constant term in library
        interaction_only: exclude pure power terms
        verbose: print progress
    """
    from .training import fit as _fit_dynamics

    dynamics = model.dynamics
    device = sparse_obs.device

    if verbose:
        print(f"\nStage 2: Discovery ({refit_epochs} epochs)")
        print(f"  Freezing encoder + decoder, extracting latent trajectories...")

    # Extract latent trajectory from frozen encoder + forced dynamics
    z_trajectory = model.extract_latent_trajectory(sparse_obs)
    z_trajectory = z_trajectory.to(device)

    # Standardize latent states for better polynomial fitting
    z_mean = z_trajectory.mean(0)   # (latent_dim,)
    z_std = z_trajectory.std(0).clamp(min=1e-6)
    z_trajectory = (z_trajectory - z_mean) / z_std

    # Store on model for forecast (un-standardize before decoding)
    model.register_buffer('z_mean', z_mean)
    model.register_buffer('z_std', z_std)

    if verbose:
        print(f"  Latent trajectory: {z_trajectory.shape[0]} points, "
              f"dim={z_trajectory.shape[1]}")
        print(f"  z mean: {z_mean.cpu().numpy()}, std: {z_std.cpu().numpy()}")

    # Build xs/ys for autonomous fit: z[t] -> z[t+1]
    xs_latent = z_trajectory[:-1].unsqueeze(0)  # (1, N-1, latent_dim)
    ys_latent = z_trajectory[1:].unsqueeze(0)   # (1, N-1, latent_dim)

    # Create fresh autonomous dynamics (n_controls=0) stored separately.
    # model.dynamics (forced) is kept intact for reconstruction.
    model.autonomous_dynamics = PolynomialRNN(
        n_states=model.latent_dim,
        n_controls=0,  # autonomous — no controls
        ensemble_size=model.ensemble_size,
        polynomial_degree=dynamics.rnn._degree,
        dt=dynamics.rnn._dt.item(),
        state_names=dynamics.state_names,
        dropout=0.,
        feature_dropout=0.,
        decomposed=dynamics.rnn._decomposed,
        direct=dynamics.rnn._direct,
        compiled_forward=True,
        num_euler_steps=dynamics.rnn._num_euler_steps,
    ).to(device)

    refit_lr = refit_learning_rate or 5e-2

    if verbose:
        print(f"  Fitting autonomous dz/dt = P(z) with lr={refit_lr:.1e}, "
              f"L2={refit_l2:.1e}, pruning_threshold={refit_pruning_threshold}")

    _fit_dynamics(
        model.autonomous_dynamics, xs_latent, ys_latent,
        epochs=refit_epochs,
        warmup_steps=0,
        learning_rate=refit_lr,
        l2=refit_l2,
        pruning_threshold=refit_pruning_threshold,
        pruning_method=refit_pruning_method,
        pruning_frequency=refit_pruning_frequency,
        ensemble_pruning_alpha=ensemble_pruning_alpha,
        dt=dt,
        include_bias=include_bias,
        interaction_only=interaction_only,
        centered_diff=centered_diff,
        refit_epochs=refit_epochs // 5,
        stability_weight=refit_stability_weight,
        verbose=verbose,
    )

    if verbose:
        active = model.count_active_terms()
        total_active = sum(active.values())
        print(f"\n  Discovery complete: {total_active} active terms")
        model.print_equations()
