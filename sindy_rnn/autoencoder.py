"""Encoder-decoder architecture for sparse observation settings.

Wraps PolynomialRNN with encoder/decoder to learn latent dynamics
from sparse measurements via joint reconstruction + dynamics training:

    L = E_id + sindy_weight * E_sindy + l1 * |theta|

    E_id:    decode(z_i) ≈ full_state_i           # same-timestep reconstruction
    E_sindy: P(z_i) ≈ (z_{i+1} - z_i) / dt       # derivative matching

Training uses SINDy-SHRED-style sliding windows: each sample is a sensor
window of length LAGS (stride=1). The GRU encoder produces one latent per
window (final hidden state), so every latent point has full temporal context.
Dynamics pairs come from consecutive windows' latent outputs.

E_id anchors the latent space (encoder/decoder learn meaningful compression).
E_sindy couples the encoder to the polynomial dynamics — the GRU is pushed
to produce latent trajectories predictable by a sparse polynomial ODE.

After joint training, the refit phase freezes the encoder and runs fit()
on the fixed latent trajectories for clean equation discovery.

Forward Euler update: z_{t+1} = z_t + dt * P(z_t), where theta directly
represents the ODE right-hand side dz/dt = P(z).

Two encoder types:
  - MLP: processes each timestep independently (no temporal context)
  - GRU: accumulates temporal context from the sensor sequence, providing
    better state estimation from sparse measurements (Takens' delay embedding)
"""

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import PolynomialRNN
from .pruning import ensemble_prune, threshold_patience_update, threshold_prune


class MLPEncoder(nn.Module):
    """MLP encoder mapping sparse measurements to latent state.

    Architecture: Linear -> ReLU -> Dropout -> ... -> Linear
    With hidden_dims=[] this reduces to a single linear layer (W @ x + b).

    Processes each timestep independently (no temporal structure).

    Args:
        input_dim: dimension of sparse measurement vector
        latent_dim: dimension of output latent state
        hidden_dims: list of hidden layer widths (default: [128, 64]).
            Pass [] for a single linear layer.
        dropout: dropout rate between layers
    """

    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: Optional[List[int]] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dims = [128, 64] if hidden_dims is None else hidden_dims

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim))

        self.net = nn.Sequential(*layers)

        # Xavier init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., input_dim) — any leading batch dimensions
        Returns:
            z: (..., latent_dim)
        """
        return self.net(x)


class GRUEncoder(nn.Module):
    """GRU encoder mapping sparse sensor sequences to latent states.

    Unlike the MLP encoder which processes each timestep independently,
    the GRU accumulates temporal context from the sensor sequence. This
    provides better state estimation from sparse measurements via temporal
    observability (Takens' delay embedding).

    Architecture: GRU(input_dim -> hidden_dim, num_layers) [-> Linear(hidden_dim -> latent_dim)]
    The projection layer is only added when hidden_dim != latent_dim.

    Args:
        input_dim: dimension of sparse measurement vector
        latent_dim: dimension of output latent state
        hidden_dim: GRU hidden dimension (default: same as latent_dim)
        num_layers: number of stacked GRU layers (default: 2)
        dropout: dropout between GRU layers (applied only when num_layers > 1)
    """

    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dim: Optional[int] = None, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else latent_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.,
        )

        if hidden_dim != latent_dim:
            self.projection = nn.Linear(hidden_dim, latent_dim)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        else:
            self.projection = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., T, input_dim) — any leading batch dimensions
        Returns:
            z: (..., T, latent_dim)
        """
        leading_shape = x.shape[:-2]
        T, F = x.shape[-2], x.shape[-1]
        x_flat = x.reshape(-1, T, F)  # (B_flat, T, input_dim)

        output, _ = self.gru(x_flat)  # (B_flat, T, hidden_dim)

        if self.projection is not None:
            output = self.projection(output)  # (B_flat, T, latent_dim)

        return output.reshape(*leading_shape, T, self.latent_dim)


class ODEEncoder(nn.Module):
    """MLP + forced polynomial ODE encoder.

    Per-timestep MLP encodes sensors to control signal u, then a forced
    polynomial ODE integrates: z[t+1] = z[t] + dt * P(z[t], u[t]).

    Single member (E=1) — ensemble diversity comes from the autonomous
    dynamics in SparseAutoencoderRNN, not from the encoder.

    Interface matches GRUEncoder: (..., T, input_dim) -> (..., T, latent_dim)

    Args:
        input_dim: dimension of sparse measurement vector
        latent_dim: dimension of output latent state (and control signal u)
        hidden_dims: MLP encoder hidden widths (default: [128, 64])
        dropout: dropout for MLP encoder
        polynomial_degree: degree for forced polynomial ODE
        dt: timestep for forward Euler integration
        decomposed: use decomposed polynomial layer
        num_euler_steps: sub-steps per dt interval
    """

    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: Optional[List[int]] = None, dropout: float = 0.1,
                 polynomial_degree: int = 2, dt: float = 1.0,
                 decomposed: bool = True, num_euler_steps: int = 1):
        super().__init__()
        self.mlp = MLPEncoder(input_dim, latent_dim, hidden_dims, dropout)
        self.latent_dim = latent_dim

        # Forced polynomial RNN with E=1
        self.ode = PolynomialRNN(
            n_states=latent_dim,
            n_controls=latent_dim,  # MLP outputs as controls
            ensemble_size=1,
            polynomial_degree=polynomial_degree,
            dt=dt,
            state_names=[f'z_{i}' for i in range(latent_dim)],
            control_names=[f'u_{i}' for i in range(latent_dim)],
            dropout=0.,
            feature_dropout=0.,
            decomposed=decomposed,
            direct=False,
            compiled_forward=True,
            num_euler_steps=num_euler_steps,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., T, input_dim) — sensor window
        Returns:
            z: (..., T, latent_dim) — latent state at each timestep
        """
        *batch_dims, T, _ = x.shape
        u = self.mlp(x)  # (..., T, latent_dim)
        u_flat = u.reshape(-1, T, self.latent_dim)  # (B, T, latent_dim)
        B = u_flat.shape[0]
        u_exp = u_flat.unsqueeze(0)  # (1, B, T, latent_dim)
        z = torch.zeros(1, B, self.latent_dim, device=x.device)

        rnn = self.ode.rnn
        z_traj = []
        for t in range(T):
            z = rnn(z, u_exp[:, :, t, :])
            z_traj.append(z)

        z_out = torch.stack(z_traj, dim=2)  # (1, B, T, latent_dim)
        return z_out[0].reshape(*batch_dims, T, self.latent_dim)


class MLPDecoder(nn.Module):
    """MLP decoder mapping latent state back to full state.

    Symmetric to encoder: Linear -> ReLU -> Dropout -> ... -> Linear

    Args:
        latent_dim: dimension of latent state
        output_dim: dimension of full state
        hidden_dims: list of hidden layer widths (default: [64, 128])
        dropout: dropout rate between layers
    """

    def __init__(self, latent_dim: int, output_dim: int,
                 hidden_dims: Optional[List[int]] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dims = [64, 128] if hidden_dims is None else hidden_dims

        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Xavier init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: (..., latent_dim) — any leading batch dimensions
        Returns:
            x_hat: (..., output_dim)
        """
        return self.net(z)


class SparseAutoencoderRNN(nn.Module):
    """Encoder-decoder wrapper around PolynomialRNN for sparse observation settings.

    Architecture (forward Euler with polynomial ODE):
        z_t = encoder(sparse_obs_t)                          # encode sensors to latent
        z_{t+1} = z_t + dt * P(z_t [, u_t])                 # forward Euler step
        full_pred_{t+1} = decoder(z_{t+1})                   # decode to full state

    Theta directly represents the ODE: dz/dt = P(z).

    The encoder maps sparse sensors to latent coordinates at each timestep.
    The polynomial RNN predicts the next latent state via forward Euler.
    The decoder maps back to full state. Teacher forcing: z_t always comes
    from encoding actual sensors, not from the model's own predictions.

    Three encoder types:
      - 'mlp': Per-timestep MLP, no temporal context
      - 'gru': GRU accumulates temporal context from sensor sequence,
        providing better state estimation from sparse measurements
      - 'ode': MLP + forced polynomial ODE integrates temporal context via
        z[t+1] = z[t] + dt * P(z[t], MLP(sensors[t])). Single member (E=1),
        ensemble diversity from autonomous dynamics only.

    The polynomial P operates only on the latent state z (and optional external
    controls u). The forward Euler step h + dt*P(h) ensures the polynomial
    must actively contribute dynamics — masking all terms gives identity (not decay).

    The encoder and decoder are shared across ensemble members. Only the inner
    PolynomialRNN has E independent members. Pruning and equation extraction
    operate on self.dynamics (the inner PolynomialRNN) directly.

    Args:
        sparse_dim: dimension of sparse measurement vector
        full_dim: dimension of full state (decoder output)
        latent_dim: dimension of latent state (= PolynomialRNN.n_states)
        n_controls: number of external control inputs
        ensemble_size: number of independent ensemble members
        polynomial_degree: degree for PolynomialRNN
        dt: timestep for forward Euler integration
        encoder_type: 'mlp', 'gru', or 'ode'
        encoder_hidden_dims: MLP encoder hidden widths ([] for single linear layer)
        encoder_gru_hidden_dim: GRU hidden dimension (default: latent_dim)
        encoder_num_layers: number of GRU layers (default: 2)
        decoder_hidden_dims: decoder MLP hidden widths
        encoder_dropout: dropout for encoder
        decoder_dropout: dropout for decoder MLP
        state_names: names for latent state variables
        control_names: names for external control variables
        dynamics_dropout: dropout for polynomial layer (per-factor)
        dynamics_feature_dropout: feature dropout for polynomial layer
        decomposed: use decomposed polynomial layer
    """

    def __init__(
        self,
        sparse_dim: int,
        full_dim: int,
        latent_dim: int,
        n_controls: int = 0,
        ensemble_size: int = 1,
        polynomial_degree: int = 2,
        dt: float = 1.0,
        encoder_type: str = 'mlp',
        encoder_hidden_dims: Optional[List[int]] = None,
        encoder_gru_hidden_dim: Optional[int] = None,
        encoder_num_layers: int = 2,
        decoder_hidden_dims: Optional[List[int]] = None,
        encoder_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
        state_names: Optional[List[str]] = None,
        control_names: Optional[List[str]] = None,
        dynamics_dropout: float = 0.,
        dynamics_feature_dropout: float = 0.,
        decomposed: bool = True,
        direct: bool = False,
        num_euler_steps: int = 1,
    ):
        super().__init__()
        self.sparse_dim = sparse_dim
        self.full_dim = full_dim
        self.latent_dim = latent_dim
        self._n_external_controls = n_controls
        self.encoder_type = encoder_type

        if encoder_type == 'gru':
            self.encoder = GRUEncoder(
                sparse_dim, latent_dim,
                hidden_dim=encoder_gru_hidden_dim,
                num_layers=encoder_num_layers,
                dropout=encoder_dropout,
            )
        elif encoder_type == 'ode':
            self.encoder = ODEEncoder(
                sparse_dim, latent_dim, encoder_hidden_dims, encoder_dropout,
                polynomial_degree=polynomial_degree, dt=dt,
                decomposed=decomposed, num_euler_steps=num_euler_steps,
            )
        else:
            self.encoder = MLPEncoder(sparse_dim, latent_dim, encoder_hidden_dims, encoder_dropout)
        self.decoder = MLPDecoder(latent_dim, full_dim, decoder_hidden_dims, decoder_dropout)

        # Polynomial operates on z only (+ optional external controls).
        state_names = state_names or [f'z_{i}' for i in range(latent_dim)]
        ext_control_names = control_names or [f'u_{i}' for i in range(n_controls)]

        self.dynamics = PolynomialRNN(
            n_states=latent_dim,
            n_controls=n_controls,
            ensemble_size=ensemble_size,
            polynomial_degree=polynomial_degree,
            dt=dt,
            state_names=state_names,
            control_names=ext_control_names,
            dropout=dynamics_dropout,
            feature_dropout=dynamics_feature_dropout,
            decomposed=decomposed,
            direct=direct,
            compiled_forward=True,
            num_euler_steps=num_euler_steps,
        )

    @property
    def ensemble_size(self):
        return self.dynamics.ensemble_size

    @property
    def n_states(self):
        return self.dynamics.n_states

    def forward(self, sparse_obs: Tensor, controls: Optional[Tensor] = None):
        """Forward pass: encode sensors -> polynomial dynamics -> decode predictions.

        For GRU/MLP encoders: teacher-forced polynomial dynamics predicts z_{t+1}
        from encoded z_t. For ODE encoder: the encoder already integrates the
        forced ODE, so we decode the last latent state directly.

        Args:
            sparse_obs: (B, T, sparse_dim) or (E, B, T, sparse_dim) — sparse measurements
            controls: (B, T, n_controls) or (E, B, T, n_controls) or None — external controls

        Returns:
            full_pred: decoded predictions
            latent_pred: latent states
            encoded: encoder output
        """
        # Encode sparse observations to latent states
        encoded = self.encoder(sparse_obs)  # (..., T, latent_dim)

        # ODE encoder already integrates temporally — skip teacher-forced dynamics.
        # Unsqueeze to add E=1 dimension for API consistency.
        if isinstance(self.encoder, ODEEncoder):
            z_last = encoded[..., -1, :]  # (..., latent_dim)
            full_pred = self.decoder(z_last)  # (..., full_dim)
            return full_pred.unsqueeze(0), z_last.unsqueeze(0), encoded

        # Build input for PolynomialRNN: [encoded_z, controls]
        if controls is not None:
            dynamics_input = torch.cat([encoded, controls], dim=-1)
        else:
            dynamics_input = encoded

        # PolynomialRNN teacher-forced forward: uses encoded z_t to predict z_{t+1}
        latent_pred, final_state = self.dynamics(dynamics_input)  # (E, B, T, latent_dim)

        # Decode predicted latent states to full state
        full_pred = self.decoder(latent_pred)  # (E, B, T, full_dim)

        return full_pred, latent_pred, encoded

    def forecast(self, z_init: Tensor, n_steps: int,
                 controls: Optional[Tensor] = None,
                 integrator: str = 'rk4') -> tuple:
        """Autonomous forward prediction without sensor input.

        Evolves the latent state using only the polynomial dynamics P(z).
        The encoder provides z_0, then P(z) takes over.

        Args:
            z_init: (E, B, latent_dim) — initial latent state (e.g. from encoder)
            n_steps: number of steps to forecast
            controls: (E, B, n_steps, n_external_controls) or None
            integrator: 'euler' or 'rk4' (default: 'rk4' for forecast accuracy)

        Returns:
            full_traj: (E, B, n_steps, full_dim) — decoded forecast trajectory
            latent_traj: (E, B, n_steps, latent_dim) — latent forecast trajectory
        """
        theta = self.dynamics.rnn.unfold_polynomial_coefficients()
        h = z_init
        trajectory = []
        for t in range(n_steps):
            u_t = controls[:, :, t, :] if controls is not None else None
            h = self.dynamics.rnn.forward_polynomial(
                h, u_t, mask=self.dynamics.coefficient_masks, theta=theta,
                integrator=integrator,
            )
            trajectory.append(h)

        latent_traj = torch.stack(trajectory, dim=2)  # (E, B, n_steps, latent_dim)
        full_traj = self.decoder(latent_traj)  # (E, B, n_steps, full_dim)
        return full_traj, latent_traj

    def encode(self, sparse_obs: Tensor) -> Tensor:
        """Encode sparse observations to latent states.

        Convenience method for getting z to pass to forecast().

        Args:
            sparse_obs: (..., sparse_dim)
        Returns:
            z: (..., latent_dim)
        """
        return self.encoder(sparse_obs)

    # --- Delegation to self.dynamics ---

    def get_equations(self) -> str:
        return self.dynamics.get_equations()

    def get_continuous_equations(self, dt: float = None) -> str:
        return self.dynamics.get_continuous_equations(dt)

    def get_coefficients(self, aggregate=True) -> Dict[str, Tensor]:
        return self.dynamics.get_coefficients(aggregate=aggregate)

    def count_active_terms(self) -> Dict[str, int]:
        return self.dynamics.count_active_terms()

    def print_equations(self):
        self.dynamics.print_equations()

    def save(self, path: str):
        """Save full model state."""
        config = {
            'sparse_dim': self.sparse_dim,
            'full_dim': self.full_dim,
            'latent_dim': self.latent_dim,
            'n_controls': self._n_external_controls,
            'ensemble_size': self.dynamics.ensemble_size,
            'polynomial_degree': self.dynamics.rnn._degree,
            'dt': self.dynamics.rnn._dt.item(),
            'state_names': self.dynamics.state_names,
            'decomposed': self.dynamics.rnn._decomposed,
            'direct': self.dynamics.rnn._direct,
            'num_euler_steps': self.dynamics.rnn._num_euler_steps,
            'encoder_type': self.encoder_type,
        }
        if self.encoder_type == 'gru':
            config['encoder_gru_hidden_dim'] = self.encoder.hidden_dim
            config['encoder_num_layers'] = self.encoder.num_layers
        torch.save({
            'state_dict': self.state_dict(),
            'coefficient_masks': self.dynamics.coefficient_masks,
            'pruning_patience': self.dynamics.pruning_patience,
            'config': config,
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load saved model. kwargs override saved config."""
        checkpoint = torch.load(path, weights_only=False)
        config = {**checkpoint['config'], **kwargs}
        # Backward compat
        if 'direct' not in config:
            config['direct'] = False
        if 'dt' not in config:
            config['dt'] = 1.0
        # Backward compat: remove alpha from old checkpoints
        config.pop('alpha', None)
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.dynamics.coefficient_masks.copy_(checkpoint['coefficient_masks'])
        model.dynamics.pruning_patience.copy_(checkpoint['pruning_patience'])
        return model


def fit_autoencoder(
    model: SparseAutoencoderRNN,
    sparse_obs: Tensor,
    full_state_target: Tensor,
    sparse_obs_test: Optional[Tensor] = None,
    full_state_target_test: Optional[Tensor] = None,
    epochs: int = 500,
    warmup_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: float = 1e-3,
    dynamics_learning_rate: Optional[float] = None,
    l1: float = 1e-4,
    pruning_frequency: int = 1,
    pruning_threshold: Optional[float] = None,
    ensemble_pruning_alpha: float = 0.05,
    pruning_method: str = 'median',
    dt: Optional[float] = None,
    include_bias: bool = True,
    interaction_only: bool = False,
    refit_epochs: int = 0,
    sindy_weight: float = 1.0,
    sindy_warmup_epochs: int = 0,
    stability_weight: float = 0.0,
    centered_diff: bool = True,
    verbose: bool = True,
):
    """Train the SparseAutoencoderRNN with reconstruction + dynamics losses.

    Uses SINDy-SHRED-style sliding windows: each sample is a sensor window
    of length LAGS. The GRU encoder produces one latent per window (final
    hidden state), so every latent point has full temporal context.

    Joint loss: L = E_id + sindy_weight * E_sindy + l1 * |theta|
                    + stability_weight * relu(max|1 + dt*lambda| - 1)

    Staged training: E_sindy and L1 are gated by sindy_warmup_epochs to let
    the encoder/decoder converge before training dynamics.
      Phase 1 [0, sindy_warmup): E_id only — pure autoencoder
      Phase 2 [sindy_warmup, end): E_id + E_sindy + L1

    E_id (reconstruction): decode the final GRU output per window, compare to
    same-timestep full state. Anchors the latent space.

    E_sindy (derivative matching): P(z_i) compared to empirical dz/dt from
    consecutive windows' latent outputs. Couples encoder to polynomial dynamics.

    After joint training, the refit phase freezes the encoder and runs fit()
    on the fixed latent trajectories for clean equation discovery.

    Args:
        model: SparseAutoencoderRNN instance
        sparse_obs: (N, LAGS, sparse_dim) — sliding sensor windows, stride=1
        full_state_target: (N, full_dim) — same-timestep reconstruction target
        sparse_obs_test: (N_te, LAGS, sparse_dim) — optional test windows
        full_state_target_test: (N_te, full_dim) — optional test targets
        epochs: total training epochs
        warmup_steps: epochs before pruning begins (default: epochs // 4)
        batch_size: mini-batch for E_id reconstruction (None = full batch).
            E_sindy always uses the full ordered dataset.
        learning_rate: Adam learning rate for encoder/decoder
        dynamics_learning_rate: Adam learning rate for polynomial dynamics.
            Default: same as learning_rate. Set higher (e.g. 5e-2) to help
            polynomial coefficients reach O(10+) ODE values faster.
        l1: L1 penalty weight on unfolded polynomial coefficients
        pruning_frequency: epochs between pruning events
        pruning_threshold: minimum effect size delta for pruning test
        ensemble_pruning_alpha: confidence level for CI test (unused for median)
        pruning_method: 'ci' or 'median'
        dt: timestep between consecutive windows
        include_bias: if False, mask out constant term
        interaction_only: if True, mask out pure power terms
        refit_epochs: epochs for frozen-encoder refit with fit().
            Resets masks and runs full pruning cycle on fixed latent space.
        sindy_weight: weight on E_sindy relative to E_id (default: 1.0)
        sindy_warmup_epochs: epochs before E_sindy + L1 activate (default: 0).
            During [0, sindy_warmup), only E_id trains.
        stability_weight: weight on discrete Euler stability penalty.
            Penalizes max|1 + dt*lambda| - 1 where lambda are Jacobian eigenvalues.
            Uses detached latent points (encoder gets no gradient from this term).
            0.0 = disabled (default).
        centered_diff: if True (default), use centered differences for O(dt²)
            derivative accuracy. Set to False for forward differences.
        verbose: print progress every 50 epochs
    """
    from .training import fit as _fit_dynamics

    if warmup_steps is None:
        warmup_steps = epochs // 4

    dynamics = model.dynamics
    E = dynamics.ensemble_size
    N = sparse_obs.shape[0]
    n_states = dynamics.n_states
    model_dt = dynamics.rnn._dt  # buffer, stays on correct device

    # Apply optional initial mask exclusions
    if not include_bias:
        dynamics.coefficient_masks[:, :, dynamics.rnn._bias_index] = False
    if interaction_only:
        for t_idx, term in enumerate(dynamics.rnn._library_terms):
            if len(term) >= 2 and len(set(term)) == 1:
                dynamics.coefficient_masks[:, :, t_idx] = False

    # Separate param groups: autonomous dynamics polynomial uses higher lr.
    # Encoder ODE params stay at encoder lr — they're part of the encoding
    # pipeline and high lr destabilizes the multi-step ODE integration.
    dynamics_lr = dynamics_learning_rate if dynamics_learning_rate is not None else learning_rate
    dynamics_param_ids = set(id(p) for p in dynamics.parameters())
    encdec_params = [p for p in model.parameters() if id(p) not in dynamics_param_ids]
    optimizer = torch.optim.Adam([
        {'params': list(dynamics.parameters()), 'lr': dynamics_lr},
        {'params': encdec_params, 'lr': learning_rate},
    ])

    def _encode_last(obs):
        """Encode windows and return LAST GRU output per window.

        Args:
            obs: (N, LAGS, sparse_dim) or (Bb, LAGS, sparse_dim)
        Returns:
            z: (N, latent_dim) or (Bb, latent_dim) — one latent per window
        """
        encoded = model.encoder(obs)  # (..., LAGS, latent_dim)
        return encoded[..., -1, :]    # (..., latent_dim) — last output

    def _test_loss():
        """Compute test reconstruction loss (E_id) in chunks to avoid OOM."""
        model.eval()
        N_te = sparse_obs_test.shape[0]
        total_se = 0.
        total_n = 0
        chunk = 64
        for start in range(0, N_te, chunk):
            end = min(start + chunk, N_te)
            z = _encode_last(sparse_obs_test[start:end])
            decoded = model.decoder(z)
            target = full_state_target_test[start:end]
            total_se += F.mse_loss(decoded, target, reduction='sum').item()
            total_n += decoded.numel()
            del decoded, z
        return total_se / total_n

    def _sindy_loss(z_all, theta_masked):
        """Derivative matching on consecutive latent points.

        Args:
            z_all: (N, latent_dim) — ordered latent trajectory
            theta_masked: (E, n_states, n_terms) — masked polynomial coefficients
        Returns:
            loss: scalar — MSE(P(z), dz/dt)
        """
        # Rescale encoder gradient: dt factor cancels 1/dt from finite diff,
        # giving O(1) encoder gradient. P(z_eval) uses unscaled z_all.
        z_diff = z_all.detach() + model_dt * (z_all - z_all.detach())
        if centered_diff:
            dz_dt = (z_diff[2:] - z_diff[:-2]) / (2 * model_dt)
            z_eval = z_all[1:-1]       # interior points
        else:
            dz_dt = (z_diff[1:] - z_diff[:-1]) / model_dt
            z_eval = z_all[:-1]

        # Expand for ensemble: (E, M, latent_dim)
        M = z_eval.shape[0]
        z_eval_e = z_eval.unsqueeze(0).expand(E, -1, -1)
        library = dynamics.rnn._compute_library(z_eval_e)  # (E, M, n_terms)
        P_z = torch.einsum('ebt,ent->ebn', library, theta_masked)  # (E, M, n_states)

        dz_dt_e = dz_dt.unsqueeze(0).expand(E, -1, -1)
        return F.mse_loss(P_z, dz_dt_e)

    try:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            # --- Phase 1: E_id (reconstruction) ---
            # Encode all windows, take last GRU output per window.
            # Two-phase gradient accumulation: backward() after E_id frees
            # the large decoded tensor before computing E_sindy.
            if batch_size is not None and batch_size < N:
                batch_idx = torch.randperm(N, device=sparse_obs.device)[:batch_size]
                z_recon = _encode_last(sparse_obs[batch_idx])
                target_recon = full_state_target[batch_idx]
            else:
                z_recon = _encode_last(sparse_obs)
                target_recon = full_state_target

            decoded = model.decoder(z_recon)  # (Bb, full_dim)
            recon_loss = F.mse_loss(decoded, target_recon)
            recon_val = recon_loss.item()
            recon_loss.backward()
            del decoded, z_recon, recon_loss

            # --- Phase 2: E_sindy + L1 ---
            # Re-encode full ordered dataset for dynamics (fresh graph).
            # Exponential ramp: sindy_weight grows from ~0 to full value over
            # sindy_warmup_epochs, then stays constant. Avoids sudden shock.
            sindy_val = 0.
            penalty_val = 0.
            stability_val = 0.
            if sindy_weight > 0 and sindy_warmup_epochs > 0 and epoch < sindy_warmup_epochs:
                _ramp = 5.0  # controls curvature (exp(5)-1 ≈ 147x dynamic range)
                progress = epoch / sindy_warmup_epochs
                eff_sindy_w = sindy_weight * (math.exp(_ramp * progress) - 1) / (math.exp(_ramp) - 1)
                eff_l1 = l1 * (math.exp(_ramp * progress) - 1) / (math.exp(_ramp) - 1)
            else:
                eff_sindy_w = sindy_weight
                eff_l1 = l1
            use_sindy = eff_sindy_w > 0
            use_l1 = eff_l1 > 0
            use_stability = stability_weight > 0
            if use_sindy or use_l1 or use_stability:
                z_all = _encode_last(sparse_obs)  # (N, latent_dim)
                theta = dynamics.rnn.unfold_polynomial_coefficients()
                theta_masked = theta * dynamics.coefficient_masks.float()

                loss2 = torch.tensor(0., device=sparse_obs.device)

                if use_sindy:
                    # sindy_weight scales encoder gradient only (gentle nudge);
                    # polynomial gets full derivative-matching gradient.
                    z_sindy = z_all.detach() + eff_sindy_w * (z_all - z_all.detach())
                    s_loss = _sindy_loss(z_sindy, theta_masked)
                    loss2 = loss2 + s_loss
                    sindy_val = s_loss.item()

                if use_l1:
                    pen = eff_l1 * theta.abs().mean()
                    loss2 = loss2 + pen
                    penalty_val = pen.item()

                if use_stability:
                    if dynamics.rnn._degree > 1:
                        z_sub = z_all.detach()[::10].unsqueeze(0).expand(E, -1, -1)
                    else:
                        z_sub = None
                    stab_loss = dynamics.rnn.compute_stability_loss(
                        theta_masked, model_dt, z_sub)
                    loss2 = loss2 + stability_weight * stab_loss
                    stability_val = stab_loss.item()

                if isinstance(loss2, Tensor) and loss2.requires_grad:
                    loss2.backward()
                del z_all, loss2

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()

            # Pruning
            if epoch >= warmup_steps and epoch % pruning_frequency == 0:
                with torch.no_grad():
                    if ensemble_pruning_alpha and E > 1:
                        ensemble_prune(dynamics, ensemble_pruning_alpha,
                                       pruning_threshold or 0.0, dt=dt,
                                       method=pruning_method)
                    elif pruning_threshold and pruning_threshold > 0:
                        threshold_patience_update(dynamics, pruning_threshold, dt=dt)
                        threshold_prune(dynamics, patience_limit=2)

            if verbose:
                if epoch == sindy_warmup_epochs and sindy_weight > 0 and sindy_warmup_epochs > 0:
                    print(f"--- Epoch {epoch}: E_sindy ramp complete (weight={sindy_weight:.4f}) ---")

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                active = dynamics.count_active_terms()
                total_active = sum(active.values())
                msg = f"Epoch {epoch:4d} | recon {recon_val:.6f}"
                if use_sindy:
                    msg += f" | sindy {sindy_val:.6f} (w={eff_sindy_w:.4f})"
                if use_l1:
                    msg += f" | penalty {penalty_val:.6f}"
                if use_stability:
                    msg += f" | stab {stability_val:.6f}"
                msg += f" | active terms: {total_active}"
                if sparse_obs_test is not None and full_state_target_test is not None:
                    with torch.no_grad():
                        loss_te = _test_loss()
                        msg += f" | test {loss_te:.6f}"
                print(msg)
    except KeyboardInterrupt:
        if verbose:
            print(f"\nTraining interrupted at epoch {epoch}.")

    # Refit: freeze encoder, extract latent trajectories, run fit() on dynamics
    if refit_epochs > 0:
        del optimizer

        if verbose:
            active = dynamics.count_active_terms()
            total_active = sum(active.values())
            print(f"\nRefit phase: freezing encoder, extracting latent trajectories...")
            print(f"  Pre-refit active terms: {total_active}")

        # Extract latent trajectory from frozen encoder
        model.eval()
        with torch.no_grad():
            z_all = _encode_last(sparse_obs)  # (N, latent_dim)

        # Construct xs/ys for fit(): xs[t] = z[t], ys[t] = z[t+1]
        xs_latent = z_all[:-1].unsqueeze(0)  # (1, N-1, latent_dim)
        ys_latent = z_all[1:].unsqueeze(0)   # (1, N-1, latent_dim)

        # Reset masks for fresh discovery on fixed latent space
        dynamics.coefficient_masks.fill_(True)
        dynamics.pruning_patience.zero_()
        if not include_bias:
            dynamics.coefficient_masks[:, :, dynamics.rnn._bias_index] = False
        if interaction_only:
            for t_idx, term in enumerate(dynamics.rnn._library_terms):
                if len(term) >= 2 and len(set(term)) == 1:
                    dynamics.coefficient_masks[:, :, t_idx] = False

        if verbose:
            print(f"  Latent trajectory: {z_all.shape[0]} points, "
                  f"calling fit() for {refit_epochs} epochs")

        # Run fit() on the PolynomialRNN with fixed latent data
        _fit_dynamics(
            dynamics, xs_latent, ys_latent,
            epochs=refit_epochs,
            # warmup_steps=refit_epochs//2,
            learning_rate=dynamics_lr,
            l2=l1,
            pruning_threshold=pruning_threshold,
            pruning_method=pruning_method,
            pruning_frequency=pruning_frequency,
            ensemble_pruning_alpha=ensemble_pruning_alpha,
            dt=dt,
            include_bias=include_bias,
            interaction_only=interaction_only,
            centered_diff=centered_diff,
            refit_epochs=refit_epochs // 5,
            stability_weight=stability_weight,
            verbose=verbose,
        )
