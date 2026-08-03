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

    def __init__(self, ensemble_size: int, input_size: int, output_size: int, degree: int = 2):
        super().__init__()
        self.degree = degree
        self.input_size = input_size
        self.output_size = output_size

        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(ensemble_size, output_size, input_size))
            for _ in range(degree)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(ensemble_size, output_size))
            for _ in range(degree)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weights:
            nn.init.xavier_normal_(w, gain=1.0)
        for b in self.biases:
            nn.init.zeros_(b)

    def forward(self, x):
        # x: (E, B, n_features) -> output: (E, B, n_states)
        result = (
            torch.einsum('eni,ebi->ebn', self.weights[0], x)
            + self.biases[0].unsqueeze(1)
        )
        for d in range(1, self.degree):
            factor = (
                torch.einsum('eni,ebi->ebn', self.weights[d], x)
                + self.biases[d].unsqueeze(1)
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
                 degree: int = 2):
        super().__init__()
        self.degree = degree
        self.input_size = input_size
        self.output_size = output_size

        # Degree 0: constant
        self.constant_bias = nn.Parameter(torch.zeros(ensemble_size, output_size))

        # Degree 1: linear map
        self.linear_weight = nn.Parameter(
            torch.empty(ensemble_size, output_size, input_size)
        )

        # Degree d>=2: d independent weight matrices (bias-free linear forms)
        self.higher_degree_weights = nn.ModuleDict()
        for d in range(2, degree + 1):
            weights = nn.ParameterList([
                nn.Parameter(torch.empty(ensemble_size, output_size, input_size))
                for _ in range(d)
            ])
            self.higher_degree_weights[str(d)] = weights

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.constant_bias)
        nn.init.xavier_normal_(self.linear_weight, gain=1.0)
        for weights in self.higher_degree_weights.values():
            for w in weights:
                nn.init.xavier_normal_(w, gain=1.0)

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
        result = result + torch.einsum('eni,ebi->ebn', self.linear_weight, x)

        # Degree d>=2: product of d bias-free forms
        for d_str, weights in self.higher_degree_weights.items():
            d = int(d_str)
            prod = torch.einsum('eni,ebi->ebn', weights[0], x)
            for k in range(1, d):
                factor = torch.einsum('eni,ebi->ebn', weights[k], x)
                prod = prod * factor
            if d > 1:
                prod = prod / (d ** 0.5)
            result = result + prod

        return result


class EnsembleRNNModule(nn.Module):
    """Forward Euler recurrent cell built on a polynomial layer.

    The polynomial P directly represents the ODE right-hand side dh/dt:
        x_t    = concat(h[t], u[t])         # (E, B, n_features)
        c      = PolynomialLayer(x_t)       # (E, B, n_states) — dh/dt
        h[t+1] = h[t] + dt * c              # forward Euler step

    Supports sub-stepping: each logical timestep is divided into
    num_euler_steps mini-steps with dt_sub = dt / num_euler_steps
    for improved numerical stability.

    Coefficients from unfold_polynomial_coefficients() directly represent
    the ODE — no discrete-to-continuous conversion needed.
    """

    def __init__(
        self,
        ensemble_size: int,
        n_states: int,
        n_controls: int = 0,
        dt: float = 1.0,
        compiled_forward: bool = True,
        polynomial_degree: int = 2,
        decomposed: bool = True,
        direct: bool = False,
        num_euler_steps: int = 1,
    ):
        super().__init__()
        n_features = n_states + n_controls

        self.n_states = n_states
        self.n_controls = n_controls
        self.ensemble_size = ensemble_size
        self._decomposed = decomposed
        self._direct = direct
        self._degree = polynomial_degree
        self._num_euler_steps = num_euler_steps

        # Physical timestep used in forward Euler: h[t+1] = h[t] + dt * P(h[t])
        self.register_buffer('_dt', torch.tensor(float(dt)))

        # Precompute library structure over n_features (needed before direct theta init)
        lib = build_library_structure(n_features, polynomial_degree)
        self._library_terms = lib['terms']
        self._n_library_terms = lib['n_terms']
        self._bias_index = lib['bias_index']
        self.register_buffer('_mult_table', lib['mult_table'])
        self.register_buffer('_linear_indices', lib['linear_indices'])

        # Precompute vectorized library index arrays for fast _compute_library
        self._build_library_index_arrays()

        # Precompute src/tgt index pairs for compile-friendly unfolding
        self._build_unfolding_index_arrays()

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
            )
        else:
            self.projection = EnsemblePolynomialLayer(
                ensemble_size=ensemble_size,
                input_size=n_features,
                output_size=n_states,
                degree=polynomial_degree,
            )

        self._compiled_forward = None
        self._compiled_unfold = None
        self._compiled_evaluate_rhs = None
        if compiled_forward:
            try:
                if not direct:
                    self._compiled_forward = torch.compile(
                        self._forward_impl, dynamic=True)
                self._compiled_unfold = torch.compile(
                    self._unfold_impl, dynamic=True)
                self._compiled_evaluate_rhs = torch.compile(
                    self._evaluate_rhs_impl, dynamic=True)
            except Exception:
                self._compiled_forward = None
                self._compiled_unfold = None
                self._compiled_evaluate_rhs = None

    def reset_dynamics_parameters(self):
        """Reinitialize polynomial coefficients from scratch, in place.

        Used by refit stages that keep a discovered sparsity mask but want
        an unbiased coefficient fit from a fresh initialization (avoids
        carrying over shrinkage/bias accumulated under the discovery-phase
        penalty). Masks/patience live on the owning PolynomialRNN, not here,
        so this only touches the raw weights.
        """
        if self._direct:
            nn.init.normal_(self.theta, std=0.01)
        else:
            self.projection.reset_parameters()

    def _forward_impl(self, h, u=None):
        """Core forward: num_euler_steps sub-steps of h += (dt/N) * P(h, u)."""
        dt_sub = self._dt / self._num_euler_steps
        for _ in range(self._num_euler_steps):
            x_t = torch.cat([h, u], dim=-1) if u is not None else h
            if self._direct:
                library = self._compute_library(x_t)
                c = torch.einsum('ebt,ent->ebn', library, self.theta)
            else:
                c = self.projection(x_t)
            h = h + dt_sub * c
        return h

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

    def _evaluate_rhs_impl(self, h, u, theta):
        """Implementation of RHS evaluation (compile target).

        Args:
            h: (E, B, n_states) — current state
            u: (E, B, m) — controls, or None
            theta: (E, n_states, n_terms) — masked polynomial coefficients
        Returns:
            P_h: (E, B, n_states) — ODE right-hand side
        """
        x_t = torch.cat([h, u], dim=-1) if u is not None else h
        library = self._compute_library(x_t)
        return torch.einsum('ebt,ent->ebn', library, theta)

    def _evaluate_rhs(self, h, u, theta):
        """Evaluate the polynomial RHS: dh/dt = P(h, u).

        Dispatches to compiled version when available.
        """
        if self._compiled_evaluate_rhs is not None:
            return self._compiled_evaluate_rhs(h, u, theta)
        return self._evaluate_rhs_impl(h, u, theta)

    def forward_polynomial(self, h, u=None, mask=None, theta=None,
                           integrator='euler'):
        """Compute next hidden state via the explicit polynomial representation.

        Integrator options:
          - 'euler': Forward Euler with sub-stepping (num_euler_steps mini-steps).
            Default for training (fast, simple gradients).
          - 'rk4': Classic 4th-order Runge-Kutta (single step, no sub-stepping).
            Recommended for autonomous forecasting (4th-order accuracy).

        Mask is applied to theta before computing the polynomial output.
        Masking all terms gives identity h[t+1] = h[t] (not decay).

        Args:
            h: (E, B, n_states) — current hidden state
            u: (E, B, m) — controls, or None
            mask: (E, n_states, n_terms) — sparsity mask
            theta: (E, n_states, n_terms) — pre-computed polynomial coefficients.
                If None, unfold_polynomial_coefficients() is called.
            integrator: 'euler' or 'rk4'
        """
        if theta is None:
            theta = self.unfold_polynomial_coefficients()  # (E, n_states, n_terms)
        if mask is not None:
            theta = theta * mask.float()

        if integrator == 'rk4':
            dt = self._dt
            k1 = self._evaluate_rhs(h, u, theta)
            k2 = self._evaluate_rhs(h + dt / 2 * k1, u, theta)
            k3 = self._evaluate_rhs(h + dt / 2 * k2, u, theta)
            k4 = self._evaluate_rhs(h + dt * k3, u, theta)
            return h + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            dt_sub = self._dt / self._num_euler_steps
            for _ in range(self._num_euler_steps):
                rhs = self._evaluate_rhs(h, u, theta)
                h = h + dt_sub * rhs
            return h

    def unfold_ode_coefficients(self) -> Tensor:
        """Return ODE coefficients (= polynomial coefficients for Euler update).

        With forward Euler h[t+1] = h[t] + dt * P(h), the polynomial
        coefficients directly represent the ODE right-hand side dh/dt = P(h).
        No conversion needed.

        Returns:
            theta_ode: (E, n_states, n_terms) — ODE coefficients
        """
        return self.unfold_polynomial_coefficients()

    def _unfold_impl(self) -> Tensor:
        """Implementation of coefficient unfolding (compile target).

        Dispatches to direct/decomposed/coupled based on mode flags.
        All paths use precomputed index buffers (no torch.where).
        """
        if self._direct:
            return self.theta
        if self._decomposed:
            return self._unfold_decomposed()
        return self._unfold_coupled()

    def unfold_polynomial_coefficients(self) -> Tensor:
        """Return polynomial coefficients in monomial basis.

        These are the coefficients of P in h[t+1] = h[t] + dt * P(h),
        representing the ODE right-hand side dh/dt = P(h) directly.

        For direct mode, returns self.theta directly.
        For factored modes, unfolds weight matrices via recursive expansion.

        Returns:
            theta: (E, n_states, n_terms) — polynomial coefficients in monomial basis
        """
        if self._compiled_unfold is not None:
            return self._compiled_unfold()
        return self._unfold_impl()

    def _unfold_coupled(self) -> Tensor:
        """Unfold coupled (original) polynomial layer.

        Uses precomputed _unfold_src_{f} / _unfold_tgt_{f} buffers instead of
        torch.where, making this function compatible with torch.compile.
        """
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

            for f in range(self._unfold_n_features):
                src_idx = getattr(self, f'_unfold_src_{f}')
                tgt_idx = getattr(self, f'_unfold_tgt_{f}')

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

        Uses precomputed _unfold_src_{f} / _unfold_tgt_{f} buffers instead of
        torch.where, making this function compatible with torch.compile.
        """
        proj = self.projection
        n_terms = self._n_library_terms
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
                for f in range(self._unfold_n_features):
                    src_idx = getattr(self, f'_unfold_src_{f}')
                    tgt_idx = getattr(self, f'_unfold_tgt_{f}')

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

    def _build_library_index_arrays(self):
        """Precompute index arrays for vectorized library computation.

        Groups degree-2+ terms by degree. For each group, stores:
          - term_indices: which library slots to fill
          - factor_indices: (n_terms_at_degree, degree) feature indices to multiply
        """
        for d in range(2, self._degree + 1):
            term_idx_list = []
            factor_list = []
            for t_idx, term in enumerate(self._library_terms):
                if len(term) == d:
                    term_idx_list.append(t_idx)
                    factor_list.append(list(term))
            if term_idx_list:
                self.register_buffer(
                    f'_lib_term_idx_d{d}',
                    torch.tensor(term_idx_list, dtype=torch.long))
                self.register_buffer(
                    f'_lib_factor_idx_d{d}',
                    torch.tensor(factor_list, dtype=torch.long))

    def _build_unfolding_index_arrays(self):
        """Precompute src/tgt index pairs per feature for coefficient unfolding.

        For each feature f, extracts the valid entries from mult_table[:, f]
        (where target >= 0) and stores them as flat buffers. This eliminates
        the torch.where calls that would otherwise cause graph breaks in
        torch.compile.

        Registers buffers:
          _unfold_src_{f}: source term indices (variable length per feature)
          _unfold_tgt_{f}: target term indices (same length as src)
          _unfold_n_features: number of features (for iteration bound)
        """
        n_features = self._mult_table.shape[1]
        self._unfold_n_features = n_features
        for f in range(n_features):
            targets = self._mult_table[:, f]
            valid = targets >= 0
            src_idx = torch.where(valid)[0]
            tgt_idx = targets[src_idx]
            self.register_buffer(f'_unfold_src_{f}', src_idx)
            self.register_buffer(f'_unfold_tgt_{f}', tgt_idx)

    def _compute_library(self, features: Tensor) -> Tensor:
        """Compute monomial library values for all terms (vectorized).

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

        # Degree-2+ terms: vectorized gather + product
        for d in range(2, self._degree + 1):
            term_idx = getattr(self, f'_lib_term_idx_d{d}', None)
            if term_idx is None:
                continue
            factor_idx = getattr(self, f'_lib_factor_idx_d{d}')
            # factor_idx: (n_terms_d, d) — feature indices for each term
            # Gather all factors at once: (E, B, n_terms_d, d)
            vals = features[:, :, factor_idx]  # advanced indexing -> (E, B, n_terms_d, d)
            # Product along the degree axis
            prod = vals.prod(dim=-1)  # (E, B, n_terms_d)
            library[:, :, term_idx] = prod

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
        compiled_forward: bool = False,
        initial_state: Union[float, Tensor] = 0.,
        decomposed: bool = True,
        direct: bool = False,
        num_euler_steps: int = 1,
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
            compiled_forward=compiled_forward,
            polynomial_degree=polynomial_degree,
            decomposed=decomposed,
            direct=direct,
            num_euler_steps=num_euler_steps,
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

        # Index of the ensemble member with the best training-data fit,
        # set by select_best_member() at the end of training. Lets
        # predict()/simulate() switch between the ensemble mean and this
        # single member (examples/_common/estimators.py's `simulate=`
        # mode) without retraining. Defaults to 0 until select_best_member
        # is called (also the value restored for older checkpoints that
        # predate this buffer, via load()'s strict=False).
        self.register_buffer('best_member_idx', torch.zeros((), dtype=torch.long))

        # Index of the ensemble member with the best BIC on training data,
        # set by select_best_member_bic() at the end of training. A third
        # option alongside 'mean'/'best' for predict()/simulate() (see
        # examples/_common/estimators.py's `simulate=` mode): trades
        # best_member_idx's held-out MSE ranking (favors whichever member
        # reconstructs unseen data most accurately, regardless of how many
        # terms it uses) for one that also rewards sparsity, computed from
        # training data alone.
        self.register_buffer('bic_member_idx', torch.zeros((), dtype=torch.long))

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

    def get_equations(self, member: Optional[int] = None, aggregate: Union[bool, str] = True) -> str:
        """Return discovered equations as formatted string.

        member: if given (e.g. self.best_member_idx.item() after
            select_best_member()), report that single ensemble member's own
            equations instead of the ensemble aggregate.
        aggregate: ignored when member is given. True/'mean' (default) or
            'median' — see equations.get_coefficients().
        """
        from .equations import get_equations
        return get_equations(self, member=member, aggregate=aggregate)

    def get_continuous_equations(self, dt: float = None, member: Optional[int] = None,
                                 aggregate: Union[bool, str] = True) -> str:
        """Return continuous-time ODE form of discovered equations.

        With Euler parameterization, theta directly represents the ODE.
        The dt argument is kept for backward compatibility but is ignored
        (model's stored dt is used).
        """
        from .equations import get_continuous_equations
        return get_continuous_equations(self, member=member, aggregate=aggregate)

    def get_coefficients(self, aggregate=True, member: Optional[int] = None) -> Dict[str, Tensor]:
        """Return effective polynomial coefficients per state dimension."""
        from .equations import get_coefficients
        return get_coefficients(self, aggregate=aggregate, member=member)

    def reset_masks(self):
        """Reset sparsity mask to all-active and clear pruning patience.

        Used by refit stages that want pruning to re-vote from scratch
        (e.g. against a newly-frozen encoder) instead of inheriting
        decisions made against a moving target.
        """
        self.coefficient_masks.fill_(True)
        self.pruning_patience.fill_(0)

    def reset_dynamics_parameters(self):
        """Reinitialize polynomial coefficients from scratch, in place.

        Keeps the sparsity mask untouched — use reset_masks() separately
        if a fresh mask is also wanted. See EnsembleRNNModule.reset_dynamics_parameters().
        """
        self.rnn.reset_dynamics_parameters()

    def count_active_terms(self, member: Optional[int] = None) -> Dict[str, int]:
        """Count active (unmasked) polynomial terms per state dimension.

        member: if given, count only that ensemble member's own mask
            instead of the union (`.any(dim=0)`) across all members.
        """
        result = {}
        for i, name in enumerate(self.state_names):
            masks_i = self.coefficient_masks[:, i, :]
            active = masks_i[member] if member is not None else masks_i.any(dim=0)
            result[name] = active.sum().item()
        return result

    def print_equations(self, member: Optional[int] = None, aggregate: Union[bool, str] = True):
        """Print discovered equations to stdout."""
        print(self.get_equations(member=member, aggregate=aggregate))

    def select_best_member(self, xs: Tensor, ys: Tensor) -> int:
        """Score each ensemble member's own next-step prediction fit on
        (xs, ys) and store the best (lowest NaN-masked MSE) member's index
        in best_member_idx.

        Evaluated via forward() — the same call predict() makes — so the
        ranking matches whichever member actually reconstructs the training
        data best, not an internal training objective (e.g. derivative
        matching) that may differ from the prediction task.

        Args:
            xs: (B, T, n_states + n_controls) — same training inputs used to fit()
            ys: (B, T, n_states) — same training targets used to fit()

        Returns:
            Index of the best-fitting ensemble member (also stored on the model).
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            preds, _ = self.forward(xs)  # (E, B, T, n_states)
            target = ys.unsqueeze(0).expand_as(preds)
            valid = ~torch.isnan(target.sum(-1))  # (E, B, T)
            se = ((preds - target) ** 2) * valid.unsqueeze(-1)
            counts = valid.sum(dim=(1, 2)).clamp(min=1) * self.n_states  # (E,)
            member_loss = se.sum(dim=(1, 2, 3)) / counts
            self.best_member_idx.fill_(int(torch.argmin(member_loss).item()))
        self.train(was_training)
        return self.best_member_idx.item()

    def select_best_member_bic(self, xs: Tensor, ys: Tensor) -> Tensor:
        """Score each ensemble member by BIC on (xs, ys) and store the best
        (lowest-BIC) member's index in bic_member_idx.

        BIC = n*ln(RSS/n) + k*ln(n), with RSS/n the same NaN-masked
        next-step prediction MSE select_best_member() uses, and k the
        member's own active (unmasked) term count summed across all state
        equations. Unlike select_best_member() (which ranks purely by fit
        and is meant to run on held-out data), BIC's k*ln(n) penalty
        rewards sparser members, so it is meaningful on training data alone.

        Args:
            xs: (B, T, n_states + n_controls) — training inputs
            ys: (B, T, n_states) — training targets

        Returns:
            ((E,) bic, (E,) mse) tensors of per-member scores (mirrors
            rollout.select_best_member_bic()'s return value, used to print a
            member/threshold/n_coef/mse/bic table — see
            examples/lorenz/train_sindy_rnn.py). The lowest-BIC member's
            index is also stored on the model (bic_member_idx).
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            preds, _ = self.forward(xs)  # (E, B, T, n_states)
            target = ys.unsqueeze(0).expand_as(preds)
            valid = ~torch.isnan(target.sum(-1))  # (E, B, T)
            se = ((preds - target) ** 2) * valid.unsqueeze(-1)
            # n = number of time-domain samples (frames), NOT multiplied by
            # n_states — mirrors sindy-shred.py's auto_tune_threshold(),
            # whose n_samples is time steps only even though its mse is
            # averaged over all state dims. See rollout.select_best_member_bic
            # for why multiplying n by the output dimension is wrong.
            n_frames = valid.sum(dim=(1, 2)).clamp(min=1)  # (E,)
            rss = se.sum(dim=(1, 2, 3))  # (E,)
            mse = rss / (n_frames * self.n_states)
            k = torch.tensor(
                [sum(self.count_active_terms(member=e).values()) for e in range(self.ensemble_size)],
                dtype=torch.float32, device=rss.device,
            )
            bic = n_frames * torch.log(mse) + k * torch.log(n_frames)
            self.bic_member_idx.fill_(int(torch.argmin(bic).item()))
        self.train(was_training)
        return bic, mse

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
                'state_names': self.state_names,
                'control_names': self.control_names,
                'decomposed': self.rnn._decomposed,
                'direct': self.rnn._direct,
                'num_euler_steps': self.rnn._num_euler_steps,
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
        # Backward compat: remove alpha from old checkpoints
        config.pop('alpha', None)
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.coefficient_masks.copy_(checkpoint['coefficient_masks'])
        model.pruning_patience.copy_(checkpoint['pruning_patience'])
        return model
