"""Encoder-decoder architecture for sparse observation settings.

Wraps PolynomialRNN with encoder/decoder to learn latent dynamics
from sparse measurements via teacher-forced next-state prediction:

    z_t = encoder(sparse_obs_t)                         # encode current sensors to latent
    z_{t+1} = z_t + dt * P(z_t)                         # forward Euler step (P ≈ dz/dt)
    full_pred_{t+1} = decoder(z_{t+1})                  # decode predicted next state

Two encoder types:
  - MLP: processes each timestep independently (no temporal context)
  - GRU: accumulates temporal context from the sensor sequence, providing
    better state estimation from sparse measurements (Takens' delay embedding)

Teacher forcing: at each training step, z_t comes from encoding the ACTUAL
sensors (not the model's own prediction). The polynomial P(z) operates
autonomously on the latent state — no sensor terms in the polynomial library.
The discovered equations are dz/dt = P(z) only.

At forecast time, the encoder provides z_0 from the last observation,
then the polynomial evolves z forward without any sensor input:
    z_0 = encoder(sensors_0)
    z_1 = z_0 + dt * P(z_0)
    z_2 = z_1 + dt * P(z_1)
    ...
"""

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

    Architecture (teacher-forced next-state prediction with forward Euler):
        z_t = encoder(sparse_obs_t)                          # encode sensors to latent
        z_{t+1} = z_t + dt * P(z_t [, u_t])                 # forward Euler (P ≈ dz/dt)
        full_pred_{t+1} = decoder(z_{t+1})                   # decode to full state

    The encoder maps sparse sensors to latent coordinates at each timestep.
    The polynomial RNN predicts the next latent state via forward Euler.
    The decoder maps back to full state. Teacher forcing: z_t always comes
    from encoding actual sensors, not from the model's own predictions.

    Two encoder types:
      - 'mlp': Per-timestep MLP, no temporal context
      - 'gru': GRU accumulates temporal context from sensor sequence,
        providing better state estimation from sparse measurements

    The polynomial P operates only on the latent state z (and optional external
    controls u) and directly represents the ODE dz/dt = P(z). The small dt
    provides implicit stability bias for autonomous forecasting.

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
        encoder_type: 'mlp' or 'gru'
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
        )

    @property
    def ensemble_size(self):
        return self.dynamics.ensemble_size

    @property
    def n_states(self):
        return self.dynamics.n_states

    def forward(self, sparse_obs: Tensor, controls: Optional[Tensor] = None):
        """Forward pass: encode sensors -> polynomial dynamics -> decode predictions.

        Teacher-forced: at each timestep, z_t is encoded from actual sensor
        observations. The polynomial RNN predicts z_{t+1} from z_t, and the
        decoder maps z_{t+1} to the predicted full state.

        This reuses the existing PolynomialRNN teacher-forced forward pass.
        The encoder output serves as the "observed state" that gets fed into
        the polynomial at each step.

        Args:
            sparse_obs: (B, T, sparse_dim) or (E, B, T, sparse_dim) — sparse measurements
            controls: (B, T, n_controls) or (E, B, T, n_controls) or None — external controls

        Returns:
            full_pred: (E, B, T, full_dim) — decoded predicted next states
            latent_pred: (E, B, T, latent_dim) — predicted next latent states
            encoded: encoder output (shape matches sparse_obs minus last dim)
        """
        # Encode sparse observations to latent states
        encoded = self.encoder(sparse_obs)  # (..., T, latent_dim)

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
                 controls: Optional[Tensor] = None) -> tuple:
        """Autonomous forward prediction without sensor input.

        Evolves the latent state using only the polynomial dynamics P(z).
        The encoder provides z_0, then P(z) takes over.

        Args:
            z_init: (E, B, latent_dim) — initial latent state (e.g. from encoder)
            n_steps: number of steps to forecast
            controls: (E, B, n_steps, n_external_controls) or None

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
                h, u_t, mask=self.dynamics.coefficient_masks, theta=theta
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
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.dynamics.coefficient_masks.copy_(checkpoint['coefficient_masks'])
        model.dynamics.pruning_patience.copy_(checkpoint['pruning_patience'])
        return model


def fit_autoencoder(
    model: SparseAutoencoderRNN,
    sparse_obs: Tensor,
    full_state_next: Tensor,
    controls: Optional[Tensor] = None,
    sparse_obs_test: Optional[Tensor] = None,
    full_state_next_test: Optional[Tensor] = None,
    controls_test: Optional[Tensor] = None,
    epochs: int = 500,
    warmup_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: float = 1e-3,
    l1: float = 1e-4,
    pruning_frequency: int = 1,
    pruning_threshold: Optional[float] = None,
    ensemble_pruning_alpha: float = 0.05,
    pruning_method: str = 'median',
    dt: Optional[float] = None,
    include_bias: bool = True,
    interaction_only: bool = False,
    refit_epochs: int = 0,
    refit_learning_rate: Optional[float] = None,
    verbose: bool = True,
):
    """Train the SparseAutoencoderRNN.

    End-to-end training: sparse_obs -> encoder -> PolynomialRNN -> decoder -> full_state_next.
    Loss = prediction MSE + L1 on polynomial coefficients.

    The prediction target is the full state at the NEXT timestep:
        encoder(sparse_obs_t) -> z_t -> P(z_t) -> z_{t+1} -> decoder -> full_state_{t+1}

    Args:
        model: SparseAutoencoderRNN instance
        sparse_obs: (B, T, sparse_dim) — sparse measurements at each timestep
        full_state_next: (B, T, full_dim) — full state prediction targets (next timestep)
        controls: (B, T, n_controls) or None — external control inputs
        sparse_obs_test, full_state_next_test, controls_test: optional test data
        epochs: total training epochs
        warmup_steps: epochs before pruning begins (default: epochs // 4)
        batch_size: mini-batch size over sequences (None = full batch)
        learning_rate: Adam learning rate
        l1: L1 penalty weight on unfolded polynomial coefficients
        pruning_frequency: epochs between pruning events
        pruning_threshold: minimum effect size delta for pruning test
        ensemble_pruning_alpha: confidence level for CI test (unused for median)
        pruning_method: 'ci' or 'median'
        dt: timestep (converts pruning to continuous-time scale)
        include_bias: if False, mask out constant term
        interaction_only: if True, mask out pure power terms
        refit_epochs: post-pruning refit epochs with l1=0
        refit_learning_rate: learning rate for refit phase
        verbose: print progress every 50 epochs
    """
    if warmup_steps is None:
        warmup_steps = epochs // 4

    dynamics = model.dynamics
    E = dynamics.ensemble_size

    # Apply optional initial mask exclusions
    if not include_bias:
        dynamics.coefficient_masks[:, :, dynamics.rnn._bias_index] = False
    if interaction_only:
        for t_idx, term in enumerate(dynamics.rnn._library_terms):
            if len(term) >= 2 and len(set(term)) == 1:
                dynamics.coefficient_masks[:, :, t_idx] = False

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    B = sparse_obs.shape[0]

    def _test_loss():
        """Compute test loss per-window to avoid OOM on high-dim outputs."""
        model.eval()
        B_te = sparse_obs_test.shape[0]
        total_se = 0.
        total_n = 0
        for b in range(B_te):
            s = sparse_obs_test[b:b+1].unsqueeze(0).expand(E, -1, -1, -1)
            c = None
            if controls_test is not None:
                c = controls_test[b:b+1].unsqueeze(0).expand(E, -1, -1, -1)
            fp, _, _ = model(s, controls=c)
            fn = full_state_next_test[b:b+1].unsqueeze(0).expand(E, -1, -1, -1)
            total_se += F.mse_loss(fp, fn, reduction='sum').item()
            total_n += fp.numel()
            del fp, fn, s, c
        return total_se / total_n

    # Bootstrap indices: each ensemble member sees different sequence samples.
    # Stored as indices only — data is indexed lazily to avoid pre-expanding
    # high-dimensional full_state_next (which can be GBs for large full_dim).
    if E > 1:
        bootstrap_indices = torch.randint(0, B, (E, B))  # (E, B)
    else:
        bootstrap_indices = torch.arange(B).unsqueeze(0)  # (1, B)

    def _get_batch(batch_idx=None):
        """Index into data with bootstrap + mini-batch selection."""
        if batch_idx is not None:
            idx = bootstrap_indices[:, batch_idx]  # (E, batch_size)
        else:
            idx = bootstrap_indices  # (E, B)
        sparse_b = sparse_obs[idx]              # (E, Bb, T, sparse_dim)
        full_next_b = full_state_next[idx]      # (E, Bb, T, full_dim)
        ctrl_b = controls[idx] if controls is not None else None
        return sparse_b, full_next_b, ctrl_b

    try:
        for epoch in range(epochs):
            model.train()

            # Mini-batch selection
            if batch_size is not None and batch_size < B:
                batch_idx = torch.randperm(B)[:batch_size]
            else:
                batch_idx = None
            sparse_b, full_next_b, ctrl_b = _get_batch(batch_idx)

            # Forward: encode -> polynomial dynamics -> decode
            full_pred, _, _ = model(sparse_b, controls=ctrl_b)  # (E, Bb, T, full_dim)

            # NaN mask for variable-length sequences
            valid = ~torch.isnan(full_next_b.sum(dim=-1))  # (E, Bb, T)
            pred_loss = F.mse_loss(full_pred[valid], full_next_b[valid])

            # L1 on unfolded polynomial coefficients
            if l1 > 0:
                theta = dynamics.rnn.unfold_polynomial_coefficients()
                coeff_penalty = l1 * theta.abs().mean()
                loss = pred_loss + coeff_penalty
            else:
                loss = pred_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Free training batch tensors to reduce memory for test eval / pruning
            del full_pred, full_next_b, sparse_b, ctrl_b, loss

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

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                active = dynamics.count_active_terms()
                total_active = sum(active.values())
                msg = f"Epoch {epoch:4d} | pred {pred_loss.item():.6f} | active terms: {total_active}"
                if sparse_obs_test is not None and full_state_next_test is not None:
                    with torch.no_grad():
                        loss_te = _test_loss()
                        msg += f" | test {loss_te:.6f}"
                print(msg)
    except KeyboardInterrupt:
        if verbose:
            print(f"\nTraining interrupted at epoch {epoch}.")

    # Post-pruning refit: train with l1=0 and frozen mask to debias coefficients
    if refit_epochs > 0:
        del optimizer  # free first optimizer's states before creating refit optimizer
        refit_lr = refit_learning_rate if refit_learning_rate is not None else learning_rate / 5
        refit_optimizer = torch.optim.Adam(model.parameters(), lr=refit_lr)

        if verbose:
            active = dynamics.count_active_terms()
            total_active = sum(active.values())
            print(f"\nRefit phase: {refit_epochs} epochs, lr={refit_lr:.1e}, "
                  f"l1=0, mask frozen ({total_active} active terms)")

        try:
            for epoch in range(refit_epochs):
                model.train()

                if batch_size is not None and batch_size < B:
                    batch_idx = torch.randperm(B)[:batch_size]
                else:
                    batch_idx = None
                sparse_b, full_next_b, ctrl_b = _get_batch(batch_idx)

                full_pred, _, _ = model(sparse_b, controls=ctrl_b)

                valid = ~torch.isnan(full_next_b.sum(dim=-1))
                pred_loss = F.mse_loss(full_pred[valid], full_next_b[valid])

                refit_optimizer.zero_grad()
                pred_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                refit_optimizer.step()

                if verbose and (epoch % 50 == 0 or epoch == refit_epochs - 1):
                    msg = f"Refit {epoch:4d} | pred {pred_loss.item():.6f}"
                    if sparse_obs_test is not None and full_state_next_test is not None:
                        with torch.no_grad():
                            loss_te = _test_loss()
                            msg += f" | test {loss_te:.6f}"
                    print(msg)
        except KeyboardInterrupt:
            if verbose:
                print(f"\nRefit interrupted at epoch {epoch}.")
