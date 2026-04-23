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


class DecomposedPolynomialLayer(nn.Module):
    """Degree-decomposed polynomial with independent parameterization per degree.

    Each polynomial degree d has its own set of weight matrices:
    - d=0: learnable bias (E, n_states)
    - d=1: single weight matrix (E, n_states, n_features)
    - d>=2: product of d bias-free linear forms, each (E, n_states, n_features)

    This completely decouples coefficients across degrees. Bias-free products
    produce ONLY degree-d monomials (no lower-degree leakage), eliminating
    the algebraic constraints that arise when a single product-of-forms must
    encode both linear and nonlinear terms through weight-bias interactions.
    """

    def __init__(self, ensemble_size: int, input_size: int, output_size: int,
                 degree: int = 2, dropout: float = 0.):
        super().__init__()
        self.degree = degree
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)

        # Degree 0: constant
        self.constant_bias = nn.Parameter(torch.zeros(ensemble_size, output_size))

        # Degree 1: linear map
        self.linear_weight = nn.Parameter(
            torch.empty(ensemble_size, output_size, input_size)
        )
        nn.init.xavier_normal_(self.linear_weight, gain=1.0)

        # Degree d>=2: d independent weight matrices (bias-free linear forms)
        self.higher_degree_weights = nn.ModuleDict()
        for d in range(2, degree + 1):
            weights = nn.ParameterList([
                nn.Parameter(torch.empty(ensemble_size, output_size, input_size))
                for _ in range(d)
            ])
            for w in weights:
                nn.init.xavier_normal_(w, gain=1.0)
            self.higher_degree_weights[str(d)] = weights

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the decomposed polynomial.

        Args:
            x: (E, B, n_features)
        Returns:
            (E, B, n_states)
        """
        # Degree 0: constant
        result = self.constant_bias.unsqueeze(1).expand(-1, x.shape[1], -1)

        # Degree 1: linear
        result = result + self.dropout(
            torch.einsum('eni,ebi->ebn', self.linear_weight, x)
        )

        # Degree d>=2: product of d bias-free forms
        for d_str, weights in self.higher_degree_weights.items():
            d = int(d_str)
            prod = self.dropout(
                torch.einsum('eni,ebi->ebn', weights[0], x)
            )
            for k in range(1, d):
                factor = self.dropout(
                    torch.einsum('eni,ebi->ebn', weights[k], x)
                )
                prod = prod * factor
            if d > 1:
                prod = prod / (d ** 0.5)
            result = result + prod

        return result


class EnsembleRNNModule(nn.Module):
    """Gated recurrent cell built on a polynomial layer.

    The polynomial P operates in discrete-time with a learnable mixing
    coefficient alpha:
        x_t    = concat(h[t], u[t])         # (E, B, n_features)
        c      = PolynomialLayer(x_t)       # (E, B, n_states)
        h[t+1] = (1 - alpha) * h[t] + alpha * c    # gated update

    The ODE right-hand side is recovered analytically:
        dh/dt = alpha * (P(h) - h) / dt
    where dt is the physical timestep of the data (stored as buffer).
    """

    def __init__(
        self,
        ensemble_size: int,
        n_states: int,
        n_controls: int = 0,
        dt: float = 1.0,
        dropout: float = 0.,
        feature_dropout: float = 0.,
        compiled_forward: bool = True,
        polynomial_degree: int = 2,
        decomposed: bool = True,
        direct: bool = False,
        alpha: float = None,
    ):
        super().__init__()
        n_features = n_states + n_controls

        self.n_states = n_states
        self.n_controls = n_controls
        self.ensemble_size = ensemble_size
        self._decomposed = decomposed
        self._direct = direct
        self._degree = polynomial_degree

        # Physical timestep (for ODE coefficient extraction, not used in forward)
        self.register_buffer('_dt', torch.tensor(float(dt)))

        # Learnable mixing coefficient: h[t+1] = (1 - alpha) * h + alpha * P(h)
        # Store raw logit; apply sigmoid in forward to get alpha in (0, 1)
        if alpha is not None:
            # Inverse sigmoid to initialize at desired alpha value
            alpha = max(min(alpha, 1 - 1e-6), 1e-6)
            logit = torch.log(torch.tensor(alpha / (1 - alpha)))
        else:
            logit = torch.tensor(0.)  # sigmoid(0) = 0.5
        self._alpha_logit = nn.Parameter(logit)

        # Precompute library structure over n_features (needed before direct theta init)
        lib = build_library_structure(n_features, polynomial_degree)
        self._library_terms = lib['terms']
        self._n_library_terms = lib['n_terms']
        self._bias_index = lib['bias_index']
        self.register_buffer('_mult_table', lib['mult_table'])
        self.register_buffer('_linear_indices', lib['linear_indices'])

        # Polynomial parameterization
        if direct:
            # Direct: theta is an nn.Parameter, no projection layer
            self.theta = nn.Parameter(
                torch.zeros(ensemble_size, n_states, self._n_library_terms)
            )
            nn.init.normal_(self.theta, std=0.01)
            self.projection = None
        elif decomposed:
            self.projection = DecomposedPolynomialLayer(
                ensemble_size=ensemble_size,
                input_size=n_features,
                output_size=n_states,
                degree=polynomial_degree,
                dropout=dropout,
            )
        else:
            self.projection = EnsemblePolynomialLayer(
                ensemble_size=ensemble_size,
                input_size=n_features,
                output_size=n_states,
                degree=polynomial_degree,
                dropout=dropout,
            )

        self.feature_dropout_p = feature_dropout

        self._compiled_forward = None
        if compiled_forward and not direct:
            try:
                self._compiled_forward = torch.compile(self._forward_impl, dynamic=True)
            except Exception:
                self._compiled_forward = None

    @property
    def _alpha(self):
        return torch.sigmoid(self._alpha_logit)

    def _forward_impl(self, h, u=None):
        """Core forward implementation: h[t+1] = (1 - alpha) * h[t] + alpha * P(h[t], u[t])."""
        x_t = torch.cat([h, u], dim=-1) if u is not None else h
        if self._direct:
            library = self._compute_library(x_t)
            c = torch.einsum('ebt,ent->ebn', library, self.theta)
        else:
            c = self.projection(x_t)
        return (1 - self._alpha) * h + self._alpha * c

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

        Uses gated update: h[t+1] = (1 - alpha) * h[t] + alpha * P(h[t], u[t])
        where P is the polynomial with coefficients theta (discrete-time).

        Mask is applied to theta before computing the polynomial output.
        The gated identity (1-alpha)*h[t] is always present (architectural,
        not subject to masking).

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

        library = self._compute_library(x_t)  # (E, B, n_terms)
        n = torch.einsum('ebt,ent->ebn', library, theta)  # (E, B, n_states)

        return (1 - self._alpha) * h + self._alpha * n

    def unfold_ode_coefficients(self) -> Tensor:
        """Convert discrete-time polynomial coefficients to ODE coefficients.

        The gated update h[t+1] = (1 - alpha) * h + alpha * P(h) corresponds to:
            dh/dt = alpha * (P(h) - h) / dt

        So the ODE coefficients are:
            theta_ode[i, j] = alpha * theta_P[i, j] / dt       (all terms)
            theta_ode[i, self_i] -= alpha / dt                  (identity correction)

        Returns:
            theta_ode: (E, n_states, n_terms) — ODE coefficients
        """
        theta_p = self.unfold_polynomial_coefficients()  # (E, n_states, n_terms)
        theta_ode = self._alpha * theta_p / self._dt

        # Subtract alpha/dt from diagonal linear terms (identity contribution)
        for i in range(self.n_states):
            lin_idx = self._linear_indices[i].item()
            theta_ode[:, i, lin_idx] = theta_ode[:, i, lin_idx] - self._alpha / self._dt

        return theta_ode

    def unfold_polynomial_coefficients(self) -> Tensor:
        """Return discrete-time polynomial coefficients in monomial basis.

        These are the raw coefficients of P in h[t+1] = (1-alpha)*h + alpha*P(h).
        For ODE coefficients, use unfold_ode_coefficients() instead.

        For direct mode, returns self.theta directly.
        For factored modes, unfolds weight matrices via recursive expansion.

        Returns:
            theta: (E, n_states, n_terms) — polynomial coefficients in monomial basis
        """
        if self._direct:
            return self.theta
        if self._decomposed:
            return self._unfold_decomposed()
        return self._unfold_coupled()

    def _unfold_coupled(self) -> Tensor:
        """Unfold coupled (original) polynomial layer."""
        W_list = list(self.projection.weights)  # D x (E, n_states, n_features)
        b_list = list(self.projection.biases)   # D x (E, n_states)
        degree = self.projection.degree
        n_terms = self._n_library_terms
        E, n = W_list[0].shape[0], W_list[0].shape[1]

        coeffs = torch.zeros(E, n, n_terms,
                             device=W_list[0].device, dtype=W_list[0].dtype)
        coeffs[:, :, self._bias_index] = b_list[0]
        coeffs[:, :, self._linear_indices] = W_list[0]

        for d in range(1, degree):
            new_coeffs = coeffs * b_list[d].unsqueeze(-1)

            for f in range(self._mult_table.shape[1]):
                targets = self._mult_table[:, f]
                valid = targets >= 0
                src_idx = torch.where(valid)[0]
                tgt_idx = targets[src_idx]

                w_f = W_list[d][:, :, f].unsqueeze(-1)
                new_coeffs[:, :, tgt_idx] = (
                    new_coeffs[:, :, tgt_idx]
                    + coeffs[:, :, src_idx] * w_f
                )
            coeffs = new_coeffs

        if degree > 1:
            coeffs = coeffs / (degree ** 0.5)

        return coeffs

    def _unfold_decomposed(self) -> Tensor:
        """Unfold decomposed polynomial layer into monomial coefficients.

        Each degree component is unfolded independently:
        - Degree 0: constant bias -> bias_index slot
        - Degree 1: linear weight -> linear_indices slots
        - Degree d>=2: recursive expansion of d bias-free forms -> degree-d slots only
          (bias-free products produce no lower-degree leakage)
        """
        proj = self.projection
        n_terms = self._n_library_terms
        n_features = self._mult_table.shape[1]
        E = proj.linear_weight.shape[0]
        n = proj.linear_weight.shape[1]
        device = proj.linear_weight.device
        dtype = proj.linear_weight.dtype

        coeffs = torch.zeros(E, n, n_terms, device=device, dtype=dtype)

        # Degree 0: constant
        coeffs[:, :, self._bias_index] = proj.constant_bias

        # Degree 1: linear
        coeffs[:, :, self._linear_indices] = proj.linear_weight

        # Degree d>=2: product of d bias-free forms
        for d_str, weights in proj.higher_degree_weights.items():
            d = int(d_str)
            W_list = list(weights)

            # Initialize with first form's linear terms
            d_coeffs = torch.zeros(E, n, n_terms, device=device, dtype=dtype)
            d_coeffs[:, :, self._linear_indices] = W_list[0]

            # Multiply by remaining bias-free forms (bias=0 -> no lower-degree leakage)
            for k in range(1, d):
                new_d_coeffs = torch.zeros_like(d_coeffs)
                for f in range(n_features):
                    targets = self._mult_table[:, f]
                    valid = targets >= 0
                    src_idx = torch.where(valid)[0]
                    tgt_idx = targets[src_idx]

                    w_f = W_list[k][:, :, f].unsqueeze(-1)
                    new_d_coeffs[:, :, tgt_idx] = (
                        new_d_coeffs[:, :, tgt_idx]
                        + d_coeffs[:, :, src_idx] * w_f
                    )
                d_coeffs = new_d_coeffs

            # Degree normalization
            if d > 1:
                d_coeffs = d_coeffs / (d ** 0.5)

            coeffs = coeffs + d_coeffs

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
        dt: float = 1.0,
        state_names: Optional[List[str]] = None,
        control_names: Optional[List[str]] = None,
        dropout: float = 0.,
        feature_dropout: float = 0.,
        compiled_forward: bool = False,
        initial_state: Union[float, Tensor] = 0.,
        decomposed: bool = True,
        direct: bool = False,
        alpha: float = None,
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
            dt=dt,
            dropout=dropout,
            feature_dropout=feature_dropout,
            compiled_forward=compiled_forward,
            polynomial_degree=polynomial_degree,
            decomposed=decomposed,
            direct=direct,
            alpha=alpha,
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

    def get_continuous_equations(self, dt: float = None) -> str:
        """Return continuous-time ODE form of discovered equations.

        With Euler parameterization, theta directly represents the ODE.
        The dt argument is kept for backward compatibility but is ignored
        (model's stored dt is used).
        """
        from .equations import get_continuous_equations
        return get_continuous_equations(self)

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
                'polynomial_degree': self.rnn._degree,
                'dt': self.rnn._dt.item(),
                'alpha': self.rnn._alpha.item(),
                'state_names': self.state_names,
                'control_names': self.control_names,
                'decomposed': self.rnn._decomposed,
                'direct': self.rnn._direct,
            }
        }, path)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load saved model. kwargs override saved config."""
        checkpoint = torch.load(path, weights_only=False)
        config = {**checkpoint['config'], **kwargs}
        # Backward compat: old checkpoints may not have these keys
        if 'decomposed' not in config:
            config['decomposed'] = False
        if 'direct' not in config:
            config['direct'] = False
        if 'dt' not in config:
            config['dt'] = 1.0
        # alpha defaults to dt for old checkpoints (Euler equivalent)
        if 'alpha' not in config:
            config['alpha'] = config['dt']
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.coefficient_masks.copy_(checkpoint['coefficient_masks'])
        model.pruning_patience.copy_(checkpoint['pruning_patience'])
        return model
