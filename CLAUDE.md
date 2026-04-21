# Technical Specification: Polynomial RNN for Sparse Nonlinear Dynamics Discovery

## Purpose of This Document

This document specifies every technical detail needed to build a **standalone Python package** that implements a multilinear/polynomial RNN with ensemble-based sparse pruning for discovering interpretable dynamical systems from sequential data.

**Target audience:** An LLM (Claude) that will use these instructions to generate the entire codebase from scratch.

**What to build:** A well-structured GitHub repository (working name: `sindy-rnn`) containing:
1. Core polynomial RNN architecture (`PolynomialRNN` — pure `torch.nn.Module`)
2. Simple training loop with ensemble bootstrap and pruning
3. Polynomial coefficient unfolding (the key contribution)
4. Reproducible examples on benchmark dynamical systems (Lorenz, Duffing, Lotka-Volterra)

**Scope:** This is deliberately minimal. No hierarchical models, no individual differences, no modular architecture. One RNN, one state vector, one set of equations. The contributions are:
1. Multilinear RNN cell = exact polynomial
2. Analytical coefficient unfolding
3. Ensemble CI pruning

---

## 1. Motivation and Positioning

### 1.1 The Problem

Given sequential observations of a dynamical system `x[t+1] = f(x[t], u[t])`, discover a **sparse polynomial** approximation of `f` directly from data. This is the core problem of SINDy (Sparse Identification of Nonlinear Dynamics, Brunton et al. 2016), but existing approaches:
- Require clean state derivative estimates (noise-sensitive)
- Use a two-stage pipeline: first fit, then sparsify (information loss)
- Lack principled uncertainty quantification for term selection

### 1.2 Our Approach

Train an ensemble of **polynomial RNNs** end-to-end on prediction loss. Each RNN cell is architecturally constrained to compute an exact degree-D polynomial, whose coefficients can be **analytically extracted** from the weight matrices (no approximation). Ensemble disagreement provides a natural statistical test for pruning: terms that aren't consistently identified across ensemble members are removed.

### 1.3 Key Technical Contributions

1. **Multilinear RNN cell** — product of D independent linear projections = exact degree-D polynomial. Weights encode polynomial coefficients implicitly.
2. **Analytical coefficient unfolding** — recursive algorithm to expand the multilinear product into monomial-basis coefficients. Fully differentiable.
3. **Ensemble CI pruning** — minimum-effect confidence interval test across ensemble members with patience-based stability.

---

## 2. High-Level Architecture

### 2.1 Overview

The model maintains a hidden state vector `h ∈ R^n` and receives control inputs `u ∈ R^m` at each timestep. A **single** polynomial RNN operates on the full state+control vector:

```
h[t+1] = (1 - α) * h[t] + α_n * P(h[t], u[t])
```

where:
- `P: R^{n+m} → R^n` is an exact degree-D polynomial, parameterized as a product of D linear forms mapping directly to `R^n` — one independent polynomial per output dimension, no intermediate projection
- `α = sigmoid(damping_coefficient)` — learned gate, **per ensemble member** (`damping_coefficient` shape `(E,)`)
- `α_n = 1` (default, `scale_candidate=False`) or `α_n = α` (`scale_candidate=True`)

**Note:** α is per-member but shared across state dimensions within each member. All state dimensions in a given ensemble member decay at the same rate. For systems with qualitatively different time constants per variable this is a simplifying assumption.

The polynomial library is constructed from `n + m` features `[h_1, ..., h_n, u_1, ..., u_m]`. Each output dimension `i` has its own independent rows in the D weight matrices, giving n fully decoupled polynomials — one equation per state variable. The unfolded coefficients are `(E, n_states, n_terms)` directly with no intermediate contraction step.

**After pruning:** When a coefficient is masked at a pruning step, the next forward pass uses the masked polynomial and training simply continues from the current hidden state. There is no hidden state reset. The (1−α) self-term is always present in the gated update regardless of masking — it is architectural, not a learned coefficient.

### 2.2 PolynomialRNN (Top-Level Model)

```python
class PolynomialRNN(nn.Module):
    """Polynomial RNN for sparse dynamics discovery.

    A single polynomial RNN operating on the full hidden state h ∈ R^n.
    The polynomial layer maps [h, u] → R^n directly — one independent
    degree-D polynomial per output dimension, no intermediate projection.

    Args:
        n_states: Number of state variables (hidden state dimensionality)
        n_controls: Number of external control/forcing inputs
        ensemble_size: Number of independent ensemble members (for pruning)
        polynomial_degree: Maximum polynomial degree (1=linear, 2=bilinear, etc.)
        state_names: Optional list of state variable names (default: ['h_0', ...])
        control_names: Optional list of control variable names (default: ['u_0', ...])
        dropout: Dropout rate (applied per-factor in the polynomial layer)
        feature_dropout: Feature dropout rate (zeros entire input feature columns)
        compiled_forward: Use torch.compile for the forward pass
        initial_state: Initial hidden state values, shape (n_states,) or scalar
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

        # Sparsity mask and patience: registered as buffers (auto .to(device))
        self.register_buffer(
            'coefficient_masks',
            torch.ones(ensemble_size, n_states, n_library_terms, dtype=torch.bool)
        )
        self.register_buffer(
            'pruning_patience',
            torch.zeros(ensemble_size, n_states, n_library_terms, dtype=torch.int32)
        )

        # Library term names from features [h_0, ..., h_{n-1}, u_0, ..., u_{m-1}]
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

        Theta is unfolded once before the timestep loop (it doesn't change
        within a forward pass) and passed to forward_polynomial for efficiency.

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

        # Unfold theta once for all timesteps
        theta = self.rnn.unfold_polynomial_coefficients()  # (E, n_states, n_terms)

        predictions = []
        for t in range(T):
            x_t = x[:, :, t, :]                        # (E, B, F)
            h_t = x_t[:, :, :self.n_states]             # observed state (teacher forcing)
            u_t = x_t[:, :, self.n_states:] if self.rnn.n_controls > 0 else None
            h_next = self.rnn.forward_polynomial(
                h_t, u_t, mask=self.coefficient_masks, theta=theta
            )                                           # (E, B, n_states)
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
        """Return effective polynomial coefficients per state dimension.

        Effective = gate-absorbed, with (1-α) added to each dimension's self-term.
        If aggregate: returns (n_terms,) ensemble mean per state (NaN-aware).
        Else: returns (E, n_terms) per-member.
        """
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
```

### 2.3 EnsemblePolynomialLayer

The core building block. Computes the **element-wise product of D independent affine projections** of the input, yielding an exact degree-D polynomial per output dimension.

```
output_i = Π_{d=0}^{D-1} (W_d[i, :] @ x + b_d[i]) / √D
```

where `W_d ∈ R^{E×n×F}`, `b_d ∈ R^{E×n}`, input `x ∈ R^{E×B×F}`, output `∈ R^{E×B×n}`.

Here n = n_states, F = n_features = n_states + n_controls. **Output size always equals n_states — no intermediate projection.** Each output dimension i has its own independent rows `W_d[:, i, :]` across all D factors, giving n fully decoupled polynomials.

**Why this works:** Each factor `(W_d @ x + b_d)` is linear in x. The elementwise product of D linear forms is an exact degree-D polynomial in x — no activation functions, no approximation.

```python
class EnsemblePolynomialLayer(nn.Module):
    def __init__(self, ensemble_size: int, input_size: int, output_size: int,
                 degree: int = 2, dropout: float = 0.):
        # output_size == n_states always — caller passes n_states directly
        self.degree = degree
        self.input_size = input_size   # n_features = n_states + n_controls
        self.output_size = output_size  # n_states
        self.dropout = nn.Dropout(dropout)
        # D independent weight/bias pairs, each (E, n_states, n_features)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(ensemble_size, output_size, input_size))
            for _ in range(degree)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(ensemble_size, output_size))
            for _ in range(degree)
        ])
        # Standard gain — initial polynomial outputs are naturally small because
        # the product of D forms with zero biases is near-zero
        for w in self.weights:
            nn.init.xavier_normal_(w, gain=1.0)

    def forward(self, x):
        # x: (E, B, n_features) → output: (E, B, n_states)
        # Dropout is applied per-factor (after each linear projection)
        result = self.dropout(
            torch.einsum('eni,ebi->ebn', self.weights[0], x)
            + self.biases[0].unsqueeze(1)
        )
        for d in range(1, self.degree):
            factor = self.dropout(
                torch.einsum('eni,ebi->ebn', self.weights[d], x)
                + self.biases[d].unsqueeze(1)
            )
            result = result * factor
        if self.degree > 1:
            result = result / (self.degree ** 0.5)
        return result
```

### 2.4 EnsembleRNNModule

A gated recurrent cell built on `EnsemblePolynomialLayer`. Single-stage computation — the polynomial layer maps directly to the update candidate with no intermediate readout:

```
x_t    = concat(h[t], u[t])         # (E, B, n_features)
c      = PolynomialLayer(x_t)       # (E, B, n_states)  — direct polynomial output
h[t+1] = (1 - α) * h[t] + α_n * c # (E, B, n_states)  — gated update
```

```python
class EnsembleRNNModule(nn.Module):
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
        n_features = n_states + n_controls

        self.n_states = n_states
        self.n_controls = n_controls
        self.ensemble_size = ensemble_size

        # Polynomial layer maps [h, u] → R^{n_states} directly — no intermediate projection
        # Dropout is passed to the polynomial layer (applied per-factor)
        self.projection = EnsemblePolynomialLayer(
            ensemble_size=ensemble_size,
            input_size=n_features,
            output_size=n_states,   # always n_states — no proj_size
            degree=polynomial_degree,
            dropout=dropout,
        )

        # Per-member gate — sigmoid(-3) ≈ 0.047: nearly persistent state at init
        self.damping_coefficient = nn.Parameter(torch.full((ensemble_size,), -3.0))
        self.scale_candidate = False  # P(x) is not scaled by alpha

        # Feature dropout: zeros entire input feature columns during training
        self.feature_dropout_p = feature_dropout

        # Precompute library structure over n_features (for unfolding and library eval)
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
```

**Standard forward (training — gradients flow through polynomial layer):**

```python
def forward(self, h, u=None):
    # h: (E, B, n_states) — current hidden state
    # u: (E, B, m) — controls, or None
    x_t = torch.cat([h, u], dim=-1) if u is not None else h   # (E, B, n_features)
    c = self.projection(x_t)                                    # (E, B, n_states)
    alpha = torch.sigmoid(self.damping_coefficient).view(-1, 1, 1)  # (E, 1, 1)
    alpha_n = alpha if self.scale_candidate else 1.0
    return (1 - alpha) * h + alpha_n * c                       # (E, B, n_states)
```

**Polynomial forward (with sparsity mask — used during training):**

```python
def forward_polynomial(self, h, u=None, mask=None, theta=None):
    """Compute next hidden state via the explicit polynomial representation.

    Mask is applied to raw unfolded coefficients theta BEFORE the (1-α)
    self-term is added. The (1-α) contribution is then added unconditionally
    by the gated update — it is not subject to masking.

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

    x_t = torch.cat([h, u], dim=-1) if u is not None else h   # (E, B, n+m)

    # Feature dropout: zero entire feature columns during training
    if self.training and self.feature_dropout_p > 0:
        feat_mask = torch.bernoulli(
            torch.full(x_t.shape[-1:], 1 - self.feature_dropout_p, device=x_t.device)
        )
        x_t = x_t * feat_mask / (1 - self.feature_dropout_p)

    library = self._compute_library(x_t)                        # (E, B, n_terms)
    n = torch.einsum('ebt,ent->ebn', library, theta)           # (E, B, n_states)

    alpha = torch.sigmoid(self.damping_coefficient).view(-1, 1, 1)  # (E, 1, 1)
    alpha_n = alpha if self.scale_candidate else 1.0
    return (1 - alpha) * h + alpha_n * n                       # (E, B, n_states)
```

---

## 3. Analytical Coefficient Unfolding (KEY CONTRIBUTION)

`unfold_polynomial_coefficients()` converts the implicit polynomial (product of D linear forms, one per output dimension) into explicit monomial-basis coefficients over the n_features input variables.

**Direct structure:** The polynomial layer computes n_states independent polynomials — output dimension i is determined solely by row i of each W_d. The unfolding produces `coeffs: (E, n_states, n_terms)` directly, with no intermediate contraction step. The bias of the polynomial layer (b_d) contributes the constant and lower-degree terms; the weights (W_d) contribute the linear and cross terms.

### 3.1 Library Structure

Build a multiplication table at init time — pure Python/NumPy, called once, results stored as buffers.

```python
def build_library_structure(n_features: int, degree: int) -> dict:
    """Enumerate monomials and build multiplication table for recursive expansion.

    Term ordering: all degree-0 terms, then degree-1, ..., up to degree-D.
    Within each degree: combinations_with_replacement order.

    Example for n_features=2, degree=2:
        terms     = [(), (0,), (1,), (0,0), (0,1), (1,1)]
        names     = ['1', 'h', 'u', 'h^2', 'h*u', 'u^2']
        n_terms   = 6
        bias_index = 0  (the constant term)
        linear_indices = [1, 2]  (indices of h and u)

    mult_table[t, f] = index of the monomial (term_t * x_f), or -1 if the
        product would exceed the maximum degree. Guaranteed unique targets:
        for fixed f, each source term maps to a distinct target (sorted
        multisets are unique), so accumulation via indexing is safe.

    Returns:
        terms: list of tuples (sorted feature index multisets)
        mult_table: (n_terms, n_features) long tensor
        linear_indices: (n_features,) long tensor
        bias_index: int
        n_terms: int
    """
    from itertools import combinations_with_replacement

    terms = [()]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            terms.append(combo)

    term_to_idx = {term: idx for idx, term in enumerate(terms)}
    n_terms = len(terms)

    mult_table = torch.full((n_terms, n_features), -1, dtype=torch.long)
    for t_idx, term in enumerate(terms):
        if len(term) < degree:
            for f in range(n_features):
                product = tuple(sorted(term + (f,)))
                if product in term_to_idx:
                    mult_table[t_idx, f] = term_to_idx[product]

    bias_index = term_to_idx[()]
    linear_indices = torch.tensor(
        [term_to_idx[(f,)] for f in range(n_features)], dtype=torch.long
    )

    return {
        'terms': terms,
        'mult_table': mult_table,
        'linear_indices': linear_indices,
        'bias_index': bias_index,
        'n_terms': n_terms,
    }
```

### 3.2 Recursive Unfolding Algorithm

```python
def unfold_polynomial_coefficients(self) -> Tensor:
    """Unfold weight matrices into polynomial coefficients via recursive expansion.

    Each output dimension i is determined solely by row i of each W_d:
        h_i[t+1] candidate = Π_{d=0}^{D-1} (W_d[e, i, :] @ x + b_d[e, i]) / √D

    This is an exact degree-D polynomial in x ∈ R^{n_features}. The unfolding
    expands it directly into the monomial basis {1, x_0, x_1, ..., x_0*x_1, ...},
    producing theta: (E, n_states, n_terms) — one coefficient vector per
    output dimension per ensemble member. No intermediate contraction needed.

    Fully differentiable: all operations are standard tensor arithmetic on
    nn.Parameters. Autograd traces through without modification.

    NOTE on accumulation safety: mult_table guarantees unique target indices
    per (source_term, feature) pair (sorted multisets are unique). Therefore
    plain index assignment in the accumulation loop is safe — no two source
    terms map to the same target for a given feature f.
    If this ever changes (e.g. custom term orderings), use scatter_add_ instead.

    Returns:
        theta: (E, n_states, n_terms) — polynomial coefficients in monomial basis
    """
    W_list = list(self.projection.weights)   # D × (E, n_states, n_features)
    b_list = list(self.projection.biases)    # D × (E, n_states)
    degree = self.projection.degree
    n_terms = self._n_library_terms
    n_features = self._mult_table.shape[1]
    E, n = W_list[0].shape[0], W_list[0].shape[1]  # n = n_states

    # coeffs: (E, n_states, n_terms) — monomial coefficients, one row per output dim
    # Initialize with first linear form d=0:
    #   constant term  ← b_0[e, i]
    #   linear term x_f ← W_0[e, i, f]
    coeffs = torch.zeros(E, n, n_terms,
                         device=W_list[0].device, dtype=W_list[0].dtype)
    coeffs[:, :, self._bias_index] = b_list[0]          # (E, n_states)
    coeffs[:, :, self._linear_indices] = W_list[0]       # (E, n_states, n_features)

    # Recursively multiply by linear forms d=1 ... D-1
    for d in range(1, degree):
        # new_coeffs = old_coeffs * b_d  (multiply all terms by per-dim scalar bias)
        new_coeffs = coeffs * b_list[d].unsqueeze(-1)    # (E, n_states, n_terms)

        # For each input feature f, multiply existing monomials by x_f
        # and accumulate into the corresponding higher-degree monomial slot
        for f in range(n_features):
            targets = self._mult_table[:, f]              # (n_terms,)
            valid = targets >= 0
            src_idx = torch.where(valid)[0]               # source monomial indices
            tgt_idx = targets[src_idx]                    # target monomial indices

            # W_list[d][:, :, f]: (E, n_states) — weight for feature f in factor d
            # coeffs[:, :, src_idx]: (E, n_states, |src|)
            w_f = W_list[d][:, :, f].unsqueeze(-1)       # (E, n_states, 1)
            new_coeffs[:, :, tgt_idx] = (
                new_coeffs[:, :, tgt_idx]
                + coeffs[:, :, src_idx] * w_f
            )
        coeffs = new_coeffs

    # Degree normalisation (matches EnsemblePolynomialLayer.forward)
    if degree > 1:
        coeffs = coeffs / (degree ** 0.5)

    # coeffs is already (E, n_states, n_terms) — no contraction needed
    return coeffs  # theta: (E, n_states, n_terms)
```

### 3.3 Library Evaluation

```python
def _compute_library(self, features: Tensor) -> Tensor:
    """Compute monomial library values for all terms.

    The library is shared across output dimensions — the same monomial values
    are used for every state equation. Per-dimension differentiation comes
    from theta (the unfolded coefficients) and sparsity masks.

    Args:
        features: (E, B, n_features) — [h_0, ..., h_{n-1}, u_0, ..., u_{m-1}]

    Returns:
        library: (E, B, n_terms)
    """
    E, B, _ = features.shape
    library = torch.ones(E, B, self._n_library_terms,
                         device=features.device, dtype=features.dtype)

    # Degree-1 terms: direct scatter
    library[:, :, self._linear_indices] = features          # (E, B, n_features)

    # Degree-2+ terms: explicit products from term tuple definitions
    for t_idx, term in enumerate(self._library_terms):
        if len(term) >= 2:
            val = features[:, :, term[0]]
            for f_idx in term[1:]:
                val = val * features[:, :, f_idx]
            library[:, :, t_idx] = val

    return library
```

### 3.4 Effective Coefficients and the Gated Update

The full discrete-time update is:

```
h_i[t+1] = (1-α) * h_i[t]  +  α_n * Σ_t θ_raw[i, t] * φ_t(h[t], u[t])
```

The **effective** polynomial coefficient for output dimension `i` and monomial `t` is:

- For `t ≠ h_i_idx` (all terms except the self-term): `c_eff[i, t] = α_n * θ_raw[i, t]`
- For `t = h_i_idx` (the linear self-term h_i): `c_eff[i, h_i_idx] = (1-α) + α_n * θ_raw[i, h_i_idx]`

The self-term index for output dimension `i` is `_linear_indices[i]` (the library index of the monomial `h_i`). Note that `h_i` also appears in the polynomial for output `j ≠ i` as a cross-state term — it does NOT receive the `(1-α)` contribution there.

**The (1-α) contribution is always present, regardless of the sparsity mask.** It is architectural, not a learned coefficient. All reporting, printing, and pruning operates on effective coefficients.

**Mask ordering in forward_polynomial:** The mask zeroes out θ_raw entries before the gated update. The (1-α) self-term is then added by the gated update unconditionally. Masking the h_i coefficient to zero removes only the learned contribution — the (1-α) decay remains.

### 3.5 Key Invariant (Critical Test)

The standard forward path and the polynomial forward path compute **exactly the same function** when `mask=ones`:

```
forward(h, u)  ==  forward_polynomial(h, u, mask=ones)
```

This must hold to numerical precision. It is the single most important unit test in the codebase. Any failure is a full stop — do not proceed until it passes.

---

## 4. Ensemble Infrastructure

### 4.1 EnsembleLinear

```python
class EnsembleLinear(nn.Module):
    """Linear layer with independent parameters per ensemble member.

    weight: (E, out_features, in_features)
    bias:   (E, out_features)
    """
    def __init__(self, ensemble_size: int, in_features: int, out_features: int):
        self.weight = nn.Parameter(
            torch.empty(ensemble_size, out_features, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(ensemble_size, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: (E, B, in_features) → (E, B, out_features)
        return (torch.einsum('eoi,ebi->ebo', self.weight, x)
                + self.bias.unsqueeze(1))
```

### 4.2 Bootstrapping

Each ensemble member trains on a different bootstrap resample of the sequence batch:

```python
if E > 1:
    indices = torch.randint(0, B, (E, B))  # (E, B) — with replacement
    xs_train = xs[indices]                  # (E, B, T, F)
    ys_train = ys[indices]                  # (E, B, T, n_states)
else:
    xs_train = xs.unsqueeze(0)              # (1, B, T, F)
    ys_train = ys.unsqueeze(0)
```

**Bootstrap indices are generated once at the start of training, not per-epoch.**

---

## 5. Pruning System

### 5.1 Sparsity Masks

- `coefficient_masks`: `(E, n_states, n_terms)` bool — which polynomial terms are active per ensemble member, per output dimension. Registered as buffer.
- `pruning_patience`: `(E, n_states, n_terms)` int32 — consecutive pruning failures per term. Registered as buffer.

**Initial state:** All terms active (True), all patience counters at 0.

**Optional initial exclusions (passed to fit):**
- `include_bias=False` → mask out the constant term '1'
- `interaction_only=True` → mask out pure power terms (e.g. `x^2` but keep `x*y`)

### 5.2 Pruning Methods

Two ensemble pruning methods are available. **Median** is recommended as the default.

#### 5.2.1 Median Effect Test (Recommended)

Median-based pruning, robust to ensemble bifurcation (where a minority of members find an alternative parameterization due to multicollinearity). A term survives iff its median |coefficient| across active members exceeds δ.

```python
def median_effect_test(
    coefficients: Tensor,  # (E, n_states, n_terms) — effective coefficients
    presence: Tensor,      # (E, n_states, n_terms) — bool mask (which members have term)
    delta: float = 0.0,    # minimum effect size threshold
) -> Tensor:               # (n_states, n_terms) bool — True where term survives
    """
    Robust to a minority of members finding an alternative parameterization.
    The median ignores outlier clusters. Requires at least 2 active members.
    """
    effective = (coefficients * presence.float()).detach()
    median = effective.median(dim=0).values
    n_active = presence.float().sum(dim=0).clamp(min=1)
    significant = (median.abs() > delta) & (n_active >= 2)
    return significant
```

#### 5.2.2 Mean-Based CI Test

Minimum-effect confidence interval test. A term survives iff its ensemble mean is statistically distinguishable from zero at level α with minimum effect size δ. Less robust than median when ensemble bifurcation occurs.

```python
def minimum_effect_ci_test(
    coefficients: Tensor,  # (E, n_states, n_terms) — effective coefficients
    presence: Tensor,      # (E, n_states, n_terms) — bool mask (which members have term)
    alpha: float = 0.05,   # significance level (two-sided)
    delta: float = 0.0,    # minimum effect size threshold
) -> Tensor:               # (n_states, n_terms) bool — True where term survives
    """
    Survival criterion: |mean(ξ_j)| - t_{α/2, E-1} * SE(ξ_j) > δ
    Requires at least 2 active members.
    """
    import scipy.stats
    effective = (coefficients * presence.float()).detach()
    E = effective.shape[0]
    mean = effective.mean(dim=0)
    std  = effective.std(dim=0, correction=1)
    se   = std / (E ** 0.5)
    t_crit = scipy.stats.t.ppf(1 - alpha / 2, df=E - 1)
    ci_lower = mean.abs() - t_crit * se
    significant = ci_lower > delta
    n_active = presence.float().sum(dim=0)
    significant = significant & (n_active >= 2)
    return significant
```

### 5.3 Continuous-Time Coefficient Conversion

When `dt` is provided to pruning functions, effective coefficients are converted to continuous-time ODE scale before the pruning test. This makes `delta` interpretable in physical units (e.g. δ=0.1 means "prune ODE terms with magnitude < 0.1").

```python
def _to_continuous(theta_eff: Tensor, model, dt: float) -> Tensor:
    """Convert discrete effective coefficients to continuous-time scale.
    Non-self terms: c / dt
    Self-term (dim i): (c - 1) / dt
    """
    theta_cont = theta_eff / dt
    for i in range(model.n_states):
        self_idx = model.rnn._linear_indices[i].item()
        theta_cont[:, i, self_idx] = (theta_eff[:, i, self_idx] - 1.0) / dt
    return theta_cont
```

### 5.4 Patience Mechanism

Terms must fail the pruning test **2 consecutive times** before permanent removal. The `ensemble_prune` function dispatches to either the median or CI test:

```python
def ensemble_prune(model, alpha: float, delta: float, dt: float = None,
                   method: str = 'ci'):
    """Run one pruning step using an ensemble statistical test.

    Args:
        model: PolynomialRNN instance
        alpha: significance level for CI test (unused for median method)
        delta: minimum effect size threshold
        dt: timestep (converts to continuous-time coefficients when provided)
        method: 'ci' for mean-based CI test,
                'median' for median test (robust to bifurcation)
    """
    theta_eff = _get_effective_coefficients_raw(model)
    coefs_for_test = _to_continuous(theta_eff, model, dt) if dt else theta_eff
    mask = model.coefficient_masks

    if method == 'median':
        significant = median_effect_test(coefs_for_test, mask, delta=delta)
    else:
        significant = minimum_effect_ci_test(
            coefs_for_test, mask, alpha=alpha, delta=delta
        )

    still_active = mask.any(dim=0)
    failed = ~significant & still_active

    counters = model.pruning_patience
    failed_e = failed.unsqueeze(0).expand_as(counters)
    counters = torch.where(failed_e, counters + 1, torch.zeros_like(counters))

    prune = (counters[0] >= 2)
    if prune.any():
        model.coefficient_masks = model.coefficient_masks & ~prune.unsqueeze(0).expand_as(mask)
        counters = counters * (~prune.unsqueeze(0).expand_as(counters)).int()

    model.pruning_patience = counters
```

### 5.5 Helper: Effective Coefficients

```python
def _get_effective_coefficients_raw(model) -> Tensor:
    """Compute effective coefficients (gate-absorbed + self-term).
    Handles per-member alpha: damping_coefficient shape (E,)."""
    theta = model.rnn.unfold_polynomial_coefficients().detach()  # (E, n_states, n_terms)
    gate = torch.sigmoid(model.rnn.damping_coefficient).detach()  # (E,)
    alpha_n = gate.view(-1, 1, 1) if model.rnn.scale_candidate else 1.0

    theta_eff = theta * alpha_n
    for i in range(model.n_states):
        self_idx = model.rnn._linear_indices[i].item()
        theta_eff[:, i, self_idx] = theta_eff[:, i, self_idx] + (1 - gate)  # broadcast (E,)

    return theta_eff
```

### 5.6 Fallback: Per-Member Threshold Pruning

For single ensemble member (E=1), hard thresholding with patience:

```python
def threshold_patience_update(model, threshold: float, dt: float = None):
    """Increment patience for terms with |effective_coef| < threshold."""
    theta_eff = _get_effective_coefficients_raw(model)
    coefs_for_test = _to_continuous(theta_eff, model, dt) if dt else theta_eff
    below = (coefs_for_test.abs() < threshold) & model.coefficient_masks
    model.pruning_patience = torch.where(
        below,
        model.pruning_patience + 1,
        torch.zeros_like(model.pruning_patience)
    )

def threshold_prune(model, patience_limit: int = 2):
    """Permanently prune terms that exceeded patience_limit."""
    candidates = (model.pruning_patience >= patience_limit) & model.coefficient_masks
    model.coefficient_masks = model.coefficient_masks & ~candidates
    model.pruning_patience = model.pruning_patience * (~candidates).int()
```

---

## 6. Training

### 6.1 Overview

Single-stage training. No two-stage fitting, no separate SINDy optimization.

1. **MSE loss** on teacher-forced next-state prediction (NaN-masked for variable-length sequences)
2. **L1 penalty on unfolded polynomial coefficients** — `l2 * theta.abs().mean()` applied directly to the unfolded θ tensor, not via AdamW weight decay. L1 promotes sparsity more effectively than L2. The optimizer is plain Adam (no weight decay).
3. **Periodic pruning** of polynomial terms via the median test or CI test (or threshold fallback)

**Teacher forcing:** The forward pass uses observed states as input at each timestep, not the model's own predictions. This ensures gradient flow — a free-running forward from initial_state causes zero gradients when the initial polynomial is near-zero.

### 6.2 Data Format

Plain tensors — no dataset class:
- `xs`: `(B, T, n_states)` or `(B, T, n_states + n_controls)` — observations (+ controls)
- `ys`: `(B, T, n_states)` — next-state targets

For a single trajectory `B=1`. NaN-pad for variable-length sequences.

### 6.3 Training Loop

```python
def fit(
    model: PolynomialRNN,
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
        pruning_threshold: minimum effect size δ for pruning test (and threshold fallback).
            When dt is provided, this is in continuous-time (ODE) units.
        ensemble_pruning_alpha: confidence level α for ensemble CI test.
            For method='median', this parameter is unused.
        pruning_method: 'ci' for mean-based CI test,
            'median' for median test (robust to bifurcation, recommended)
        dt: timestep of the data. When provided, pruning operates on continuous-time
            coefficients (c/dt), making pruning_threshold interpretable in ODE units.
        include_bias: if False, mask out constant term before training
        interaction_only: if True, mask out pure power terms before training
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

    # Use Adam (no weight decay) — L1 is applied on polynomial coefficients instead
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    E = model.ensemble_size
    B, T = xs.shape[0], xs.shape[1]

    # Bootstrap: (B, T, F) → (E, B, T, F), fixed for entire training
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

            ys_pred, _ = model(xb)                         # (E, B, T, n_states)

            # NaN mask for variable-length sequences
            valid = ~torch.isnan(yb.sum(dim=-1))           # (E, B, T)
            mse_loss = F.mse_loss(ys_pred[valid], yb[valid])

            # L1 penalty on unfolded polynomial coefficients
            if l2 > 0:
                theta = model.rnn.unfold_polynomial_coefficients()
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
```

---

## 7. Equation Extraction and Printing

### 7.1 Effective Coefficients

```python
def get_coefficients(model: PolynomialRNN, aggregate: bool = True) -> Dict[str, Tensor]:
    """Return effective polynomial coefficients for each state dimension.

    Effective = gate-absorbed + (1-α) self-term correction.
    Pruned terms are treated as zero (or NaN before nanmean when aggregate=True).

    The (1-α) self-term for dimension i is added to _linear_indices[i] unconditionally.

    Returns:
        dict mapping state_name → Tensor
            aggregate=True:  (n_terms,) ensemble mean (NaN-aware, pruned→0)
            aggregate=False: (E, n_terms) per-member
    """
    alpha = torch.sigmoid(model.rnn.damping_coefficient).detach()  # (E,)
    alpha_n = alpha.view(-1, 1, 1) if model.rnn.scale_candidate else 1.0

    theta = model.rnn.unfold_polynomial_coefficients().detach()  # (E, n_states, n_terms)
    mask  = model.coefficient_masks.float()                       # (E, n_states, n_terms)

    c_eff = theta * mask * alpha_n                               # (E, n_states, n_terms)

    # Add (1-α) to each dimension's self-term (always, regardless of mask)
    for i in range(model.n_states):
        self_idx = model.rnn._linear_indices[i].item()
        c_eff[:, i, self_idx] = c_eff[:, i, self_idx] + (1 - alpha)  # broadcast (E,)

    results = {}
    for i, name in enumerate(model.state_names):
        c_i    = c_eff[:, i, :]    # (E, n_terms)
        mask_i = mask[:, i, :]     # (E, n_terms)
        self_idx = model.rnn._linear_indices[i].item()

        if aggregate:
            c_agg = c_i.clone()
            # Mark pruned non-self terms as NaN for nanmean
            non_self = torch.ones(c_i.shape[-1], dtype=torch.bool)
            non_self[self_idx] = False
            c_agg[:, non_self] = torch.where(
                mask_i[:, non_self] == 0,
                torch.full_like(c_agg[:, non_self], float('nan')),
                c_agg[:, non_self],
            )
            c_i = torch.nanmean(c_agg, dim=0)
            c_i = torch.nan_to_num(c_i, nan=0.0)

        results[name] = c_i
    return results
```

### 7.2 Equation Printing

```python
def get_equations(model: PolynomialRNN) -> str:
    """Return discovered equations as a formatted multi-line string.

    Example output:
        x[t+1] = 0.900*x[t] + 0.100*y
        y[t+1] = 0.280*x - 0.100*x*z + 0.900*y[t]
        z[t+1] = 0.100*x*y + 0.973*z[t]

    The [t] suffix marks the self-term of each state variable.
    Constant and cross-state terms have no time index.
    """
    coefs = get_coefficients(model, aggregate=True)
    term_names = model.library_terms  # list of str, length n_terms
    lines = []

    for i, state_name in enumerate(model.state_names):
        c = coefs[state_name]              # (n_terms,)
        self_idx = model.rnn._linear_indices[i].item()
        parts = []

        for j, term_name in enumerate(term_names):
            val = c[j].item()
            if abs(val) < 1e-8:
                continue
            sign = '+' if val >= 0 else '-'
            coef_str = f"{abs(val):.3f}"
            if term_name == '1':
                label = coef_str
            elif j == self_idx:
                label = f"{coef_str}*{state_name}[t]"
            else:
                label = f"{coef_str}*{term_name}"
            parts.append((sign, label))

        if not parts:
            rhs = '0'
        else:
            sign0, label0 = parts[0]
            rhs = (f"-{label0}" if sign0 == '-' else label0)
            for sign, label in parts[1:]:
                rhs += f" {sign} {label}"

        lines.append(f"{state_name}[t+1] = {rhs}")

    return '\n'.join(lines)
```

### 7.3 Continuous-Time Equation Conversion

```python
def get_continuous_equations(model: PolynomialRNN, dt: float) -> str:
    """Convert discovered discrete-time equations to continuous-time ODE form.

    The discrete update is:
        h_i[t+1] = (1-α)*h_i[t] + P_i(h[t])

    Subtracting h_i[t] and dividing by dt:
        dh_i/dt ≈ (h_i[t+1] - h_i[t]) / dt

    For the self-term (h_i in P_i), the effective discrete coef is
    (1-α) + θ_raw_self, so the continuous contribution of h_i is:
        (discrete_self - 1) / dt

    For all other terms:
        c_continuous = c_discrete / dt

    Returns:
        Formatted string of continuous-time equations d/dt h_i = ...
    """
```

---

## 8. Polynomial Library Utilities

```python
def compute_library_size(n_features: int, degree: int) -> int:
    """Number of monomials up to degree D in n features.
    = C(n_features + degree, degree)
    Example: (n=3, d=2) → 10 terms: {1, x0, x1, x2, x0², x0x1, x0x2, x1², x1x2, x2²}
    """
    from math import comb
    return comb(n_features + degree, degree)

def get_library_feature_names(feature_names: List[str], degree: int) -> List[str]:
    """Generate human-readable monomial names matching build_library_structure order.

    Example: ['h', 'u'], degree=2 → ['1', 'h', 'u', 'h^2', 'h*u', 'u^2']
    """
    from itertools import combinations_with_replacement
    names = ['1']
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(len(feature_names)), d):
            counts = {}
            for idx in combo:
                counts[idx] = counts.get(idx, 0) + 1
            parts = []
            for idx, cnt in sorted(counts.items()):
                name = feature_names[idx]
                parts.append(f"{name}^{cnt}" if cnt > 1 else name)
            names.append('*'.join(parts))
    return names

def get_polynomial_degree_from_term(term: str) -> int:
    """Extract total degree from term string.
    '1'→0, 'x'→1, 'x^2'→2, 'x*y'→2, 'x^2*y'→3
    """
    if term == '1':
        return 0
    total = 0
    for part in term.split('*'):
        if '^' in part:
            total += int(part.split('^')[1])
        else:
            total += 1
    return total
```

---

## 9. Repository Structure

```
sindy-rnn/
├── sindy_rnn/
│   ├── __init__.py                    # Public API: PolynomialRNN, fit
│   ├── model.py                       # PolynomialRNN, EnsembleRNNModule,
│   │                                  #   EnsemblePolynomialLayer, EnsembleLinear
│   ├── training.py                    # fit(), bootstrap helpers
│   ├── pruning.py                     # minimum_effect_ci_test, ensemble_prune,
│   │                                  #   threshold_prune, threshold_patience_update
│   ├── polynomial_library.py          # build_library_structure,
│   │                                  #   compute_library_size,
│   │                                  #   get_library_feature_names,
│   │                                  #   get_polynomial_degree_from_term
│   └── equations.py                   # get_coefficients, get_equations,
│                                      #   get_continuous_equations
├── examples/
│   ├── lorenz.py
│   ├── duffing.py
│   ├── lotka_volterra.py
│   └── linear_system.py              # Sanity check: x[t+1] = 0.9x + 0.1u
├── tests/
│   ├── test_polynomial_layer.py       # forward == forward_polynomial (mask=ones)
│   ├── test_unfolding.py              # Known polynomial recovery
│   ├── test_pruning.py                # CI test, patience, mask updates
│   └── test_training.py              # End-to-end: linear system recovery
├── requirements.txt
└── README.md
```

---

## 10. Implementation Priorities

### Phase 1: Core (must have)
1. `build_library_structure()` and `get_library_feature_names()` — needed by everything else
2. `EnsemblePolynomialLayer` with forward pass
3. `EnsembleRNNModule` — standard and polynomial forward paths
4. `unfold_polynomial_coefficients()` — the key algorithm
5. `PolynomialRNN` — top-level model (save/load, get_equations, get_coefficients)
6. **Unit test: `forward == forward_polynomial(mask=ones)`** — run before proceeding further
7. Training loop (`fit()`) with MSE loss, gradient clipping, NaN masking, bootstrap
8. Ensemble CI pruning with patience

### Phase 2: Polish
9. `EnsembleLinear`
10. `torch.compile` support with fallback
11. Fallback threshold pruning (single ensemble member)
12. `include_bias` / `interaction_only` initial mask exclusions

### Phase 3: Examples & Tests
13. Lorenz system discovery example
14. Linear system sanity check
15. Duffing oscillator example
16. Unit tests for unfolding correctness (known coefficients)
17. Integration test: known polynomial system recovery end-to-end

---

## 11. Critical Implementation Details

### 11.1 Tensor Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Training data (after bootstrap) | `(E, B, T, F)` | F = n_states + n_controls |
| Targets | `(E, B, T, n_states)` | |
| Hidden state | `(E, B, n_states)` | Full state vector |
| RNN input | `(E, B, n_features)` | [h, u] concatenated |
| Polynomial layer output | `(E, B, n_states)` | Direct — no intermediate projection |
| Polynomial coefficients (unfolded) | `(E, n_states, n_terms)` | Direct output of unfold |
| Sparsity mask | `(E, n_states, n_terms)` | Per output dimension |
| Patience counter | `(E, n_states, n_terms)` | Per output dimension |
| Monomial library values | `(E, B, n_terms)` | Shared across output dims |
| Predictions | `(E, B, T, n_states)` | |

### 11.2 Dimension Conventions

| Symbol | Meaning |
|--------|---------|
| E | Ensemble members |
| B | Batch (sequences) |
| T | Timesteps |
| n | n_states |
| m | n_controls |
| F | n_features = n_states + n_controls |
| C | n_terms (monomial library size = C(n_features + degree, degree)) |

### 11.3 Common Pitfalls

1. **`tensor.to(device)` returns a new tensor** — must reassign
2. **Masks and patience counters use `register_buffer`** — automatic `.to(device)`, not `nn.Parameter`
3. **`(1-α)` state decay is architectural** — always present in effective coefficients, never masked
4. **NaN masking before loss** — variable-length sequences are NaN-padded; mask with `~isnan(yb.sum(-1))`
5. **Gradient clipping** (`max_norm=1.0`) is essential for stability with the multiplicative structure
6. **Bootstrap indices are fixed** at the start of training, not re-drawn per epoch
7. **Polynomial forward with mask is used for ALL training steps** (not just after warmup) — the forward pass always uses `forward_polynomial` with the sparsity mask
8. **`EnsemblePolynomialLayer` output_size is always n_states** — no intermediate proj_size, no weight_n readout. The polynomial layer maps directly to R^n.
9. **Xavier init with gain=1.0** for polynomial layer weights — initial polynomial outputs are naturally small because the product of D forms with zero biases is near-zero. Using gain=0.01 caused loss regression.
10. **Damping init `torch.full((E,), -3.0)`** → `sigmoid(-3) ≈ 0.047` — nearly persistent hidden state at init. Per-member (shape `(E,)`), not shared scalar.
11. **Per-dimension self-terms** — `_linear_indices[i]` gives the library index of `h_i`; only dimension i's own equation gets the `(1-α)` addition at that index
12. **Accumulation safety in unfolding** — `mult_table` guarantees unique target indices per (source, feature) pair by construction (sorted multisets). Plain index assignment is safe. If you ever change the term ordering, verify this or switch to `scatter_add_`
13. **Pruning is `torch.no_grad()`** — no gradients needed for the pruning test or mask updates
14. **α is per-member** — `damping_coefficient` shape `(E,)`, shared across state dimensions within each member. Must `.view(-1, 1, 1)` for broadcasting in forward pass.
15. **Teacher forcing is essential** — the forward pass must use observed states as input, not model predictions. A free-running loop from initial_state causes zero gradients when initial polynomial outputs are near-zero.
16. **L1 on unfolded coefficients, not AdamW weight decay** — the optimizer is plain Adam; L1 penalty (`theta.abs().mean()`) is computed on the unfolded θ tensor and added to the MSE loss. This penalises actual polynomial terms rather than raw weights, which have a nonlinear relationship to monomial coefficients.
17. **scale_candidate=False by default** — polynomial output is NOT scaled by α. With α_n=α, the gate suppresses all polynomial terms equally, creating a bottleneck. Setting α_n=1 decouples stability (1-α damping) from polynomial magnitude.
18. **Dropout on linear factors, not on polynomial output** — dropout is applied after each `(W_d @ x + b_d)` projection inside `EnsemblePolynomialLayer`, not on the final polynomial output. This provides better regularization for the multiplicative structure.
19. **Theta caching in forward pass** — `unfold_polynomial_coefficients()` is called once before the timestep loop and the result is passed to `forward_polynomial(theta=theta)`. The unfolded coefficients don't change within a forward pass.
20. **Median pruning for ensembles with bifurcation** — with E=10+, bootstrap resampling can cause different members to find different local minima due to multicollinearity (e.g. x-y correlation ρ=0.88 on Lorenz). The median test is robust to this; the CI test is not.

### 11.4 Feature Ordering

```
features = [h_0, h_1, ..., h_{n-1}, u_0, u_1, ..., u_{m-1}]
```

`_linear_indices[i] = i` for state features (i < n_states). The library is shared across output dimensions; sparsity determines which terms each dimension uses.

---

## 12. Example: Lorenz System Discovery

```python
import torch
import numpy as np
from sindy_rnn import PolynomialRNN, fit

def lorenz_rk4(x, dt=0.01, sigma=10, rho=28, beta=8/3):
    def f(x):
        return np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        ])
    k1 = f(x); k2 = f(x + dt/2*k1)
    k3 = f(x + dt/2*k2); k4 = f(x + dt*k3)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

x = np.array([1., 1., 1.])
trajectory = [x]
for _ in range(5000):
    x = lorenz_rk4(x)
    trajectory.append(x)
trajectory = np.array(trajectory)  # (5001, 3)

xs = torch.tensor(trajectory[:-1], dtype=torch.float32).unsqueeze(0)  # (1, 5000, 3)
ys = torch.tensor(trajectory[1:],  dtype=torch.float32).unsqueeze(0)  # (1, 5000, 3)

model = PolynomialRNN(
    n_states=3,
    n_controls=0,
    polynomial_degree=2,
    ensemble_size=11,
    state_names=['x', 'y', 'z'],
    dropout=0.1,
)

fit(model, xs, ys,
    epochs=1000,
    warmup_steps=500,
    ensemble_pruning_alpha=0.05,
    pruning_threshold=0.1,
    pruning_method='median',
    pruning_frequency=20,
    learning_rate=1e-2,
    l2=5e-2,
    dt=0.01,
    verbose=True,
)

model.print_equations()
# Expected discrete-time equations (approximately):
# x[t+1] = 0.900*x[t] + 0.100*y
# y[t+1] = 0.280*x - 0.100*x*z + 0.900*y[t]
# z[t+1] = 0.100*x*y + 0.973*z[t]

# Continuous-time ODE form:
print(model.get_continuous_equations(dt=0.01))
# Expected:
# dx/dt = -10.000*x + 10.000*y
# dy/dt = 28.000*x - 1.000*y - 1.000*x*z
# dz/dt = -2.667*z + 1.000*x*y
```

---

## 13. Dependencies

```
torch >= 2.0
numpy
scipy       # for scipy.stats.t.ppf in CI test
```

Optional: `matplotlib` for examples.

---

## 14. Testing Strategy

### Unit Tests (Phase 1 gate)

1. **Invariant test (CRITICAL):** `forward(h, u) == forward_polynomial(h, u, mask=ones)` for random weights and random inputs, to floating-point precision (rtol=1e-5). This is the single most important test. Run it after every change to the unfolding algorithm.
2. **Library structure:** Verify `mult_table` for `(n=2, d=2)` against manual enumeration of all monomial products.
3. **Feature names:** Verify name generation matches manual enumeration for small cases.
4. **Unfolding correctness:** Construct a polynomial RNN with known weights; verify `unfold_polynomial_coefficients()` returns the analytically correct coefficients.

### Unit Tests (Phase 2)

5. **CI test:** A term with consistent nonzero mean across ensemble members passes; a term with zero mean fails.
6. **Patience:** Counter increments on consecutive failure, resets on success, pruning fires at count=2.
7. **Self-term masking:** Masking h_i coefficient to zero removes learned contribution but not the (1-α) decay.

### Integration Tests (Phase 3)

8. **Linear system recovery:** Known `x[t+1] = 0.9*x + 0.1*u` → verify coefficients are recovered within tolerance after training.
9. **Sparsity:** Many irrelevant terms are pruned for a sparse ground-truth system.
10. **Ensemble consistency:** Ensemble members discover similar equations (low coefficient variance).

---

## 15. Notes for the Implementing Agent

- **Keep it minimal.** No module system, no config objects, no per-entity logic, no sklearn estimator, no dataset class. One polynomial RNN, one state vector, plain tensors in and out.
- **The core novelty is the multilinear → monomial unfolding.** Get this right and test it thoroughly before writing anything else. The invariant test (forward == forward_polynomial) is the ground truth.
- **`PolynomialRNN`** is the main class — a pure `torch.nn.Module`. `fit()` is a standalone function. Public API is just these two.
- **`coefficient_masks` and `pruning_patience` are registered buffers**, not parameters. They move with the model on `.to(device)` automatically.
- **Do not reset the hidden state after pruning.** Training continues from the current state. The architecture handles this gracefully.
- **No intermediate projection.** `EnsemblePolynomialLayer` maps directly from n_features → n_states. There is no `weight_n`, no `proj_size`, no `bias_n` readout. The unfolding produces `(E, n_states, n_terms)` directly from the D weight matrices.
- **Use `torch.compile` with `dynamic=True`** and a try/except fallback — not all environments support it.
- **`scipy.stats.t.ppf`** is the only scipy dependency — used only at pruning time, not in the forward pass.
