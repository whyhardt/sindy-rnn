"""Core model classes: EnsembleLinear, EnsemblePolynomialLayer, EnsembleRNNModule, PolynomialRNN."""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .polynomial_library import build_library_structure, get_library_feature_names


class EnsembleLinear(nn.Module):
    """Linear layer with independent parameters per ensemble member.

    weight: (E, out_features, in_features)
    bias:   (E, out_features)
    """

    def __init__(self, ensemble_size: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(ensemble_size, out_features, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(ensemble_size, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: (E, B, in_features) -> (E, B, out_features)
        return (
            torch.einsum('eoi,ebi->ebo', self.weight, x)
            + self.bias.unsqueeze(1)
        )


class EnsemblePolynomialLayer(nn.Module):
    """Computes element-wise product of D independent affine projections.

    output_i = prod_{d=0}^{D-1} (W_d[i, :] @ x + b_d[i]) / sqrt(D)

    where W_d in R^{E x n x F}, b_d in R^{E x n}, input x in R^{E x B x F},
    output in R^{E x B x n}.

    Each factor (W_d @ x + b_d) is linear in x. The elementwise product of D
    linear forms is an exact degree-D polynomial in x.
    """

    def __init__(self, ensemble_size: int, input_size: int, output_size: int, degree: int = 2, dropout: float = 0.):
        super().__init__()
        self.degree = degree
        self.input_size = input_size
        self.output_size = output_size

        self.dropout = nn.Dropout(dropout)
        
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(ensemble_size, output_size, input_size))
            for _ in range(degree)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(ensemble_size, output_size))
            for _ in range(degree)
        ])
        for w in self.weights:
            nn.init.xavier_normal_(w, gain=1.0)

    def forward(self, x):
        # x: (E, B, n_features) -> output: (E, B, n_states)
        result = (
            self.dropout(torch.einsum('eni,ebi->ebn', self.weights[0], x)
            + self.biases[0].unsqueeze(1))
        )
        for d in range(1, self.degree):
            factor = (
                self.dropout(torch.einsum('eni,ebi->ebn', self.weights[d], x)
                + self.biases[d].unsqueeze(1))
            )
            result = result * factor
        if self.degree > 1:
            result = result / (self.degree ** 0.5)
        return result


class EnsembleRNNModule(nn.Module):
    """Gated recurrent cell built on EnsemblePolynomialLayer.

    x_t    = concat(h[t], u[t])         # (E, B, n_features)
    c      = PolynomialLayer(x_t)       # (E, B, n_states)
    h[t+1] = (1 - alpha) * h[t] + alpha_n * c
    """

    def __init__(
        self,
        ensemble_size: int,
        n_states: int,
        n_controls: int = 0,
        dropout: float = 0.,
        feature_dropout: float = 0.,
        compiled_forward: bool = True,
        polynomial_degree: int = 2,
    ):
        super().__init__()
        n_features = n_states + n_controls

        self.n_states = n_states
        self.n_controls = n_controls
        self.ensemble_size = ensemble_size

        self.projection = EnsemblePolynomialLayer(
            ensemble_size=ensemble_size,
            input_size=n_features,
            output_size=n_states,
            degree=polynomial_degree,
            dropout=dropout,
        )

        # Per-member damping: sigmoid(-3) ~ 0.047 -> nearly persistent state at init
        self.damping_coefficient = nn.Parameter(torch.full((ensemble_size,), -3.0))
        self.scale_candidate = False  # P(x) is not scaled by alpha

        # self.dropout = nn.Dropout(p=dropout)
        self.feature_dropout_p = feature_dropout
        
        # Precompute library structure over n_features
        lib = build_library_structure(n_features, polynomial_degree)
        self._library_terms = lib['terms']
        self._n_library_terms = lib['n_terms']
        self._bias_index = lib['bias_index']
        self.register_buffer('_mult_table', lib['mult_table'])
        self.register_buffer('_linear_indices', lib['linear_indices'])

        self._compiled_forward = None
        if compiled_forward:
            try:
                self._compiled_forward = torch.compile(self._forward_impl, dynamic=True)
            except Exception:
                self._compiled_forward = None

    def _forward_impl(self, h, u=None):
        """Core forward implementation."""
        x_t = torch.cat([h, u], dim=-1) if u is not None else h
        c = self.projection(x_t)
        # if self.dropout.p > 0:
        #     c = self.dropout(c)
        alpha = torch.sigmoid(self.damping_coefficient).view(-1, 1, 1)  # (E, 1, 1)
        alpha_n = alpha if self.scale_candidate else 1.0
        return (1 - alpha) * h + alpha_n * c

    def forward(self, h, u=None):
        """Standard forward pass (training — gradients flow through polynomial layer).

        Args:
            h: (E, B, n_states) — current hidden state
            u: (E, B, m) — controls, or None
        Returns:
            h_next: (E, B, n_states)
        """
        if self._compiled_forward is not None:
            return self._compiled_forward(h, u)
        return self._forward_impl(h, u)

    def forward_polynomial(self, h, u=None, mask=None, theta=None):
        """Compute next hidden state via the explicit polynomial representation.

        Mask is applied to raw unfolded coefficients theta BEFORE the (1-alpha)
        self-term is added. The (1-alpha) contribution is then added unconditionally
        by the gated update.

        Args:
            h: (E, B, n_states) — current hidden state
            u: (E, B, m) — controls, or None
            mask: (E, n_states, n_terms) — sparsity mask
            theta: (E, n_states, n_terms) — pre-computed polynomial coefficients.
                If None, unfold_polynomial_coefficients() is called.
        """
        if theta is None:
            theta = self.unfold_polynomial_coefficients()  # (E, n_states, n_terms)
        if mask is not None:
            theta = theta * mask.float()

        x_t = torch.cat([h, u], dim=-1) if u is not None else h  # (E, B, n+m)

        # Feature dropout: zero entire feature columns during training
        if self.training and self.feature_dropout_p > 0:
            feat_mask = torch.bernoulli(
                torch.full(x_t.shape[-1:], 1 - self.feature_dropout_p,
                           device=x_t.device)
            )  # (n_features,)
            x_t = x_t * feat_mask / (1 - self.feature_dropout_p)

        library = self._compute_library(x_t)  # (E, B, n_terms)
        n = torch.einsum('ebt,ent->ebn', library, theta)  # (E, B, n_states)

        alpha = torch.sigmoid(self.damping_coefficient).view(-1, 1, 1)  # (E, 1, 1)
        alpha_n = alpha if self.scale_candidate else 1.0
        return (1 - alpha) * h + alpha_n * n

    def unfold_polynomial_coefficients(self) -> Tensor:
        """Unfold weight matrices into polynomial coefficients via recursive expansion.

        Returns:
            theta: (E, n_states, n_terms) — polynomial coefficients in monomial basis
        """
        W_list = list(self.projection.weights)  # D x (E, n_states, n_features)
        b_list = list(self.projection.biases)   # D x (E, n_states)
        degree = self.projection.degree
        n_terms = self._n_library_terms
        E, n = W_list[0].shape[0], W_list[0].shape[1]

        # Initialize with first linear form d=0
        coeffs = torch.zeros(E, n, n_terms,
                             device=W_list[0].device, dtype=W_list[0].dtype)
        coeffs[:, :, self._bias_index] = b_list[0]
        coeffs[:, :, self._linear_indices] = W_list[0]

        # Recursively multiply by linear forms d=1 ... D-1
        for d in range(1, degree):
            new_coeffs = coeffs * b_list[d].unsqueeze(-1)  # (E, n_states, n_terms)

            for f in range(self._mult_table.shape[1]):
                targets = self._mult_table[:, f]  # (n_terms,)
                valid = targets >= 0
                src_idx = torch.where(valid)[0]
                tgt_idx = targets[src_idx]

                w_f = W_list[d][:, :, f].unsqueeze(-1)  # (E, n_states, 1)
                new_coeffs[:, :, tgt_idx] = (
                    new_coeffs[:, :, tgt_idx]
                    + coeffs[:, :, src_idx] * w_f
                )
            coeffs = new_coeffs

        # Degree normalisation
        if degree > 1:
            coeffs = coeffs / (degree ** 0.5)

        return coeffs

    def _compute_library(self, features: Tensor) -> Tensor:
        """Compute monomial library values for all terms.

        Args:
            features: (E, B, n_features)
        Returns:
            library: (E, B, n_terms)
        """
        E, B, _ = features.shape
        library = torch.ones(E, B, self._n_library_terms,
                             device=features.device, dtype=features.dtype)

        # Degree-1 terms
        library[:, :, self._linear_indices] = features

        # Degree-2+ terms: explicit products from term tuple definitions
        for t_idx, term in enumerate(self._library_terms):
            if len(term) >= 2:
                val = features[:, :, term[0]]
                for f_idx in term[1:]:
                    val = val * features[:, :, f_idx]
                library[:, :, t_idx] = val

        return library


class PolynomialRNN(nn.Module):
    """Polynomial RNN for sparse dynamics discovery.

    A single polynomial RNN operating on the full hidden state h in R^n.
    The polynomial layer maps [h, u] -> R^n directly — one independent
    degree-D polynomial per output dimension, no intermediate projection.
    """

    def __init__(
        self,
        n_states: int,
        n_controls: int = 0,
        ensemble_size: int = 1,
        polynomial_degree: int = 2,
        state_names: Optional[List[str]] = None,
        control_names: Optional[List[str]] = None,
        dropout: float = 0.,
        feature_dropout: float = 0.,
        compiled_forward: bool = False,
        initial_state: Union[float, Tensor] = 0.,
    ):
        super().__init__()
        self.n_states = n_states
        self.ensemble_size = ensemble_size
        self.state_names = state_names or [f'h_{i}' for i in range(n_states)]
        self.control_names = control_names or [f'u_{i}' for i in range(n_controls)]

        self.rnn = EnsembleRNNModule(
            ensemble_size=ensemble_size,
            n_states=n_states,
            n_controls=n_controls,
            dropout=dropout,
            feature_dropout=feature_dropout,
            compiled_forward=compiled_forward,
            polynomial_degree=polynomial_degree,
        )

        n_library_terms = self.rnn._n_library_terms

        # Sparsity mask and patience: registered as buffers
        self.register_buffer(
            'coefficient_masks',
            torch.ones(ensemble_size, n_states, n_library_terms, dtype=torch.bool)
        )
        self.register_buffer(
            'pruning_patience',
            torch.zeros(ensemble_size, n_states, n_library_terms, dtype=torch.int32)
        )

        # Library term names
        feature_names = self.state_names + self.control_names
        self.library_terms = get_library_feature_names(feature_names, polynomial_degree)

        # Initial hidden state
        if isinstance(initial_state, (int, float)):
            h0 = torch.full((n_states,), float(initial_state))
        else:
            h0 = torch.as_tensor(initial_state, dtype=torch.float32)
        self.register_buffer('_initial_state', h0)

    def forward(self, x, state=None):
        """Forward pass over a sequence (teacher-forced).

        At each timestep, the observed state x_t[:, :, :n_states] is used as
        input to predict the next state. This is teacher forcing: the model
        sees the true observation at each step rather than its own predictions.

        Args:
            x: (E, B, T, n_states + n_controls) or (B, T, ...) — auto-promoted to 4D.
                First n_states columns are observed states, remaining are controls.
            state: Not used (kept for API compatibility).

        Returns:
            predictions: (E, B, T, n_states) — predicted next states
            final_state: (E, B, n_states) — last prediction
        """
        if x.dim() == 3:
            x = x.unsqueeze(0).expand(self.ensemble_size, -1, -1, -1)
        E, B, T, F = x.shape

        # Unfold theta once for all timesteps (it doesn't change within a forward pass)
        theta = self.rnn.unfold_polynomial_coefficients()  # (E, n_states, n_terms)

        predictions = []
        for t in range(T):
            x_t = x[:, :, t, :]  # (E, B, F)
            h_t = x_t[:, :, :self.n_states]  # observed state (teacher forcing)
            u_t = x_t[:, :, self.n_states:] if self.rnn.n_controls > 0 else None
            h_next = self.rnn.forward_polynomial(
                h_t, u_t, mask=self.coefficient_masks, theta=theta
            )
            predictions.append(h_next)

        predictions = torch.stack(predictions, dim=2)  # (E, B, T, n_states)
        return predictions, predictions[:, :, -1, :]

    def get_equations(self) -> str:
        """Return discovered equations as formatted string."""
        from .equations import get_equations
        return get_equations(self)

    def get_continuous_equations(self, dt: float) -> str:
        """Return continuous-time ODE form of discovered equations."""
        from .equations import get_continuous_equations
        return get_continuous_equations(self, dt)

    def get_coefficients(self, aggregate=True) -> Dict[str, Tensor]:
        """Return effective polynomial coefficients per state dimension."""
        from .equations import get_coefficients
        return get_coefficients(self, aggregate=aggregate)

    def count_active_terms(self) -> Dict[str, int]:
        """Count active (unmasked) polynomial terms per state dimension."""
        result = {}
        for i, name in enumerate(self.state_names):
            result[name] = self.coefficient_masks[:, i, :].any(dim=0).sum().item()
        return result

    def print_equations(self):
        """Print discovered equations to stdout."""
        print(self.get_equations())

    def save(self, path: str):
        """Save model weights + sparsity masks + patience counters."""
        torch.save({
            'state_dict': self.state_dict(),
            'coefficient_masks': self.coefficient_masks,
            'pruning_patience': self.pruning_patience,
            'config': {
                'n_states': self.n_states,
                'n_controls': self.rnn.n_controls,
                'ensemble_size': self.ensemble_size,
                'polynomial_degree': self.rnn.projection.degree,
                'state_names': self.state_names,
                'control_names': self.control_names,
            }
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load saved model. kwargs override saved config."""
        checkpoint = torch.load(path, weights_only=False)
        config = {**checkpoint['config'], **kwargs}
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.coefficient_masks.copy_(checkpoint['coefficient_masks'])
        model.pruning_patience.copy_(checkpoint['pruning_patience'])
        return model
