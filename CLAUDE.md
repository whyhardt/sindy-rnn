# Technical Specification: Polynomial RNN for Sparse Nonlinear Dynamics Discovery

## Workflow Rules

**Always sanity-check before large studies.** Before launching multi-seed or grid-search experiments, first run a single-seed quick test to verify that the model achieves expected performance. Compare against known baselines from previous runs. Only scale up once the single run looks correct. Changing hyperparameters "to match baselines" can silently degrade performance — always verify with a quick run first.

**Shared reconstruction-based evaluation for benchmarks.** When comparing different methods (sindy-rnn, SINDy-SHRED, SHRED, etc.), all methods MUST be evaluated using the SAME metric on the SAME held-out data. The standard protocol is:
1. Each method trains with its own internal objective (next-step prediction, SINDy regularization, etc.)
2. After training, each method produces a **full-state reconstruction array** `(n_frames, full_dim)` — the reconstructed field at every timestep.
3. All methods are then evaluated on the **same held-out test frames** using MSE and relative error between reconstruction and ground truth (in scaled space).
4. This is an apples-to-apples comparison regardless of each method's internal training objective.
Never compare internal training/validation losses across methods — they measure different things (next-step prediction vs same-timestep reconstruction, different loss functions, different data splits).

---

## 1. Motivation and Positioning

Given sequential observations of a dynamical system, discover a **sparse polynomial** ODE from data. This is the core problem of SINDy (Brunton et al. 2016), but existing approaches require clean derivative estimates (noise-sensitive) and use a two-stage pipeline (fit then sparsify).

**Our approach:** Train an ensemble of polynomial RNNs end-to-end on prediction loss. Each RNN cell computes an exact degree-D polynomial whose coefficients can be analytically extracted from weight matrices. Ensemble disagreement provides a statistical test for pruning.

**Key contributions:**
1. **Multilinear RNN cell** — product of D linear projections = exact degree-D polynomial
2. **Analytical coefficient unfolding** — recursive expansion into monomial basis, fully differentiable
3. **Ensemble pruning** — median/CI test across ensemble members with patience-based stability

---

## 2. Architecture

### 2.1 Overview

Forward Euler integration with polynomial ODE right-hand side:

```
h[t+1] = h[t] + dt * P(h[t], u[t])
```

where `P: R^{n+m} → R^n` is an exact degree-D polynomial representing `dh/dt = P(h, u)`. `dt` is a buffer (not learned). The polynomial library is constructed from `[h_1, ..., h_n, u_1, ..., u_m]`. Each output dimension has its own independent polynomial — fully decoupled. Unfolded coefficients are `(E, n_states, n_terms)` directly in ODE units.

**After pruning:** Training continues from the current state (no reset). The identity `h[t]` is always present regardless of masking — it is architectural.

### 2.2 PolynomialRNN

> See [model.py](sindy_rnn/model.py) — `class PolynomialRNN`

Top-level `nn.Module`. Constructor params: `n_states, n_controls, ensemble_size, polynomial_degree, dt, state_names, control_names, dropout, feature_dropout, compiled_forward, initial_state, decomposed, direct`.

Key methods: `forward(x)` (teacher-forced), `get_equations()`, `get_coefficients(aggregate)`, `count_active_terms()`, `save(path)`, `load(path)`.

Buffers: `coefficient_masks (E, n_states, n_terms)`, `pruning_patience (E, n_states, n_terms)`.

### 2.3 EnsemblePolynomialLayer

> See [model.py](sindy_rnn/model.py) — `class EnsemblePolynomialLayer`

Element-wise product of D independent affine projections: `output_i = Π_{d=0}^{D-1} (W_d[i,:] @ x + b_d[i]) / √D`. Each factor is linear in x; the product is an exact degree-D polynomial — no activation functions, no approximation. Output size = n_states always (no intermediate projection).

### 2.4 EnsembleRNNModule

> See [model.py](sindy_rnn/model.py) — `class EnsembleRNNModule`

Forward Euler cell:
```
x_t    = concat(h[t], u[t])         # (E, B, n_features)
c      = PolynomialLayer(x_t)       # (E, B, n_states) — P(h,u) ≈ dh/dt
h[t+1] = h[t] + dt * c              # forward Euler step
```

`forward_polynomial(h, u, mask, theta)` applies sparsity mask to theta before polynomial evaluation. The identity `h[t]` is unconditional.

---

## 3. Analytical Coefficient Unfolding (KEY CONTRIBUTION)

> See [model.py](sindy_rnn/model.py) — `unfold_polynomial_coefficients()`, and [polynomial_library.py](sindy_rnn/polynomial_library.py) — `build_library_structure()`

Converts the implicit polynomial (product of D linear forms) into explicit monomial-basis coefficients `(E, n_states, n_terms)`. Fully differentiable.

### 3.1 Library Structure

`build_library_structure(n_features, degree)` enumerates monomials and builds a multiplication table at init time. Term ordering: degree-0 first, then degree-1, ..., up to degree-D, with `combinations_with_replacement` ordering within each degree.

Example for n_features=2, degree=2: `['1', 'h', 'u', 'h^2', 'h*u', 'u^2']`

`mult_table[t, f]` = index of monomial `(term_t * x_f)`, or -1 if exceeding max degree. Guaranteed unique targets per (source, feature) pair.

### 3.2 Recursive Unfolding

Initialize coefficients from the first linear form (d=0): constant from bias, linear terms from weights. Then recursively multiply by each subsequent linear form (d=1...D-1), using `mult_table` to map products to target monomial slots. Final normalization by `√D`.

### 3.3 Library Evaluation

`_compute_library(features)` computes monomial values `(E, B, n_terms)` shared across output dimensions. See [model.py](sindy_rnn/model.py).

### 3.4 Coefficients and the Euler Update

The full update is: `h_i[t+1] = h_i[t] + dt * Σ_j θ[i,j] * φ_j(h[t], u[t])`. Coefficients from `unfold_polynomial_coefficients()` directly represent the ODE right-hand side — no conversion needed. Masking all coefficients gives identity `h_i[t+1] = h_i[t]` (not decay).

### 3.5 Key Invariant (Critical Test)

`forward(h, u) == forward_polynomial(h, u, mask=ones)` must hold to numerical precision. This is the single most important unit test. Any failure is a full stop.

---

## 4. Ensemble Infrastructure

> See [model.py](sindy_rnn/model.py) — `class EnsembleLinear`

Each ensemble member trains on a different bootstrap resample: `indices = torch.randint(0, B, (E, B))`. **Bootstrap indices are generated once at the start of training, not per-epoch.**

---

## 5. Pruning System

> See [pruning.py](sindy_rnn/pruning.py)

### 5.1 Sparsity Masks

- `coefficient_masks`: `(E, n_states, n_terms)` bool — registered buffer
- `pruning_patience`: `(E, n_states, n_terms)` int32 — registered buffer
- Initial: all active, all patience at 0
- Optional: `include_bias=False` masks constant term; `interaction_only=True` masks pure powers

### 5.2 Pruning Methods

**Median (recommended):** Term survives iff `|median(coef)| > δ` across active members. Robust to ensemble bifurcation from multicollinearity.

**CI test:** Term survives iff `|mean| - t_{α/2} * SE > δ`. Less robust when bifurcation occurs.

Both require at least 2 active members. Theta is already in ODE units — no conversion needed.

### 5.3 Patience Mechanism

Terms must fail the pruning test **2 consecutive times** before permanent removal. `ensemble_prune()` dispatches to median or CI test, updates patience counters, and permanently masks terms that exceed patience.

### 5.4 Threshold Fallback

For E=1, `threshold_patience_update()` increments patience for `|coef| < threshold`, then `threshold_prune()` removes terms exceeding patience limit.

---

## 6. Training

> See [training.py](sindy_rnn/training.py) — `fit()`

Single-stage training:
1. **MSE loss** on teacher-forced next-state prediction (NaN-masked for variable-length sequences)
2. **L1 penalty on unfolded θ** — `l2 * theta.abs().mean()` (not AdamW weight decay). Plain Adam optimizer.
3. **Periodic pruning** via median/CI test or threshold fallback

**Teacher forcing is essential** — free-running from initial_state causes zero gradients when initial polynomial is near-zero.

**Data format:** `xs: (B, T, n_states + n_controls)`, `ys: (B, T, n_states)`. NaN-pad for variable-length.

**Key params:** `epochs, warmup_steps, learning_rate, l2 (L1 weight), pruning_frequency, pruning_threshold (δ), pruning_method ('median'/'ci'), dt, refit_epochs`.

---

## 7. Equation Extraction

> See [equations.py](sindy_rnn/equations.py)

- `get_coefficients(model, aggregate)` — returns raw ODE coefficients `θ * mask`. Aggregate: NaN-aware ensemble mean.
- `get_equations(model)` — formatted ODE string: `dx/dt = -10.000*x + 10.000*y`
- `get_continuous_equations(model)` — alias for `get_equations()` (theta is already in ODE units)

---

## 8. Library Utilities

> See [polynomial_library.py](sindy_rnn/polynomial_library.py)

- `compute_library_size(n_features, degree)` — number of monomials = C(n+d, d)
- `get_library_feature_names(feature_names, degree)` — human-readable monomial names
- `get_polynomial_degree_from_term(term)` — total degree from term string

---

## 9. Critical Implementation Details

### 9.1 Tensor Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Training data (after bootstrap) | `(E, B, T, F)` | F = n_states + n_controls |
| Targets | `(E, B, T, n_states)` | |
| Hidden state | `(E, B, n_states)` | Full state vector |
| Polynomial layer output | `(E, B, n_states)` | Direct — no intermediate projection |
| Polynomial coefficients (unfolded) | `(E, n_states, n_terms)` | Direct output of unfold |
| Sparsity mask | `(E, n_states, n_terms)` | Per output dimension |
| Monomial library values | `(E, B, n_terms)` | Shared across output dims |

### 9.2 Dimension Conventions

E=ensemble, B=batch, T=timesteps, n=n_states, m=n_controls, F=n_features=n+m, C=n_terms=C(F+degree, degree)

### 9.3 Common Pitfalls

1. **`tensor.to(device)` returns a new tensor** — must reassign
2. **Masks and patience use `register_buffer`** — auto `.to(device)`, not `nn.Parameter`
3. **Identity is architectural** — `h[t]` in Euler update always present. Masking all terms gives identity (not decay).
4. **NaN masking before loss** — `~isnan(yb.sum(-1))`
5. **Gradient clipping** (`max_norm=1.0`) essential for multiplicative structure
6. **Bootstrap indices fixed** at training start, not per-epoch
7. **`forward_polynomial` with mask used for ALL training steps** (not just after warmup)
8. **Output size always n_states** — no intermediate proj_size
9. **Xavier init gain=1.0** — gain=0.01 caused loss regression
10. **dt as buffer, not parameter** — `register_buffer('_dt', ...)`. Small dt gives implicit stability bias.
11. **Theta IS the ODE** — no discrete-to-continuous conversion needed
12. **Accumulation safety** — `mult_table` guarantees unique targets per (source, feature)
13. **Pruning is `torch.no_grad()`**
14. **Teacher forcing essential** — free-running causes zero gradients
15. **L1 on unfolded θ, not AdamW** — penalizes polynomial terms, not raw weights
16. **Dropout per-factor** — inside `EnsemblePolynomialLayer`, not on final output
17. **Theta caching** — `unfold_polynomial_coefficients()` called once before timestep loop
18. **Median pruning for bifurcation** — E=10+ can bifurcate due to multicollinearity; median is robust
19. **Backward compat in load()** — `strict=False`, defaults for missing keys (`dt=1.0`, `decomposed=False`, `direct=False`)

### 9.4 Feature Ordering

`features = [h_0, ..., h_{n-1}, u_0, ..., u_{m-1}]`. `_linear_indices[i] = i` for states.

---

## 10. Example: Lorenz System Discovery

### 10.1 Key characteristics (direct application with `fit()`)

- **Derivative matching loss**: `fit()` compares `P(h)` to empirical `dh/dt` (not next-step prediction). This is a local per-timestep objective — much easier to optimize than forecasting.
- **Centered differences** (`centered_diff=True`, default): `dh/dt ≈ (h[t+1] - h[t-1]) / (2*dt)` for O(dt²) accuracy. Forward differences (`centered_diff=False`) have O(dt) error, causing ~10% coefficient bias on Lorenz.
- **High learning rate** (`lr=5e-2`): ODE coefficients are O(10–28) (e.g., σ=10, ρ=28). At `lr=1e-2`, Adam takes ~2800 epochs to reach them. `5e-2` converges in ~500 epochs.
- **Gradient clipping** (`max_norm=100.0`): Derivative-scale losses produce gradients ~100x larger than next-step losses. Old `max_norm=1.0` throttled learning by ~1000x.
- **L1 on unfolded θ** (`l2=5e-2`): Drives sparsity. Named `l2` in `fit()` for legacy reasons — it's actually L1 on `theta.abs().mean()`.
- **Batched windows** (`window_size=100`): Long trajectories chunked into non-overlapping windows for GPU parallelism.
- **Sub-stepping** (`num_euler_steps=3`): Multiple Euler steps per dt interval improves integration accuracy during teacher-forced forward pass.
- **dynamics_weight=0**: Disables autonomous rollout loss. Derivative matching alone is sufficient for direct state observation.
- **Known limitation — y absorption**: On Lorenz, x-y correlation ρ=0.88. Gradient descent + L1 prefers 6-term model (25.5*x absorbs 28*x - y in dy/dt). Standard SINDy/STLSQ with OLS finds exact 7-term partition because closed-form least-squares has a unique solution. Prediction MSE is identical with or without -y — this is inherent to gradient-based optimization with correlated features, not a bug.
- **Evaluation**: Autonomous forecast MSE (simulate discovered ODE forward, compare to ground truth). Not next-step prediction MSE (trivially small for small dt) or derivative matching MSE (method-specific).

### 10.2 Working configuration

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

dt = 0.01
x = np.array([1., 1., 1.])
trajectory = [x]
for _ in range(5000):
    x = lorenz_rk4(x, dt=dt)
    trajectory.append(x)
trajectory = np.array(trajectory)  # (5001, 3)

# Chunk into windows for batched training
window_size = 100
N = len(trajectory) - 1
n_windows = N // window_size
xs = torch.tensor(trajectory[:n_windows*window_size].reshape(n_windows, window_size, -1),
                  dtype=torch.float32)
ys = torch.tensor(trajectory[1:n_windows*window_size+1].reshape(n_windows, window_size, -1),
                  dtype=torch.float32)

model = PolynomialRNN(
    n_states=3, n_controls=0, polynomial_degree=2,
    ensemble_size=11, dt=dt,
    state_names=['x', 'y', 'z'],
    dropout=0.1, decomposed=True,
    num_euler_steps=3,
)

fit(model, xs, ys,
    epochs=3000, warmup_steps=1000,
    ensemble_pruning_alpha=0.05,
    pruning_threshold=0.2,
    pruning_method='median',
    pruning_frequency=100,
    learning_rate=5e-2,
    l2=5e-2,              # L1 penalty weight
    dt=dt,
    refit_epochs=500,
    dynamics_weight=0,    # derivative matching only, no autonomous rollout
    verbose=True,
)

model.print_equations()
# Expected ODE (theta directly represents dh/dt):
# dx/dt = -10.000*x + 10.000*y
# dy/dt = 25.500*x - 1.000*x*z       (or 28*x - y - x*z with 7 terms)
# dz/dt = -2.667*z + 1.000*x*y
```

---

## 11. Dependencies

```
torch >= 2.0
numpy
scipy       # for scipy.stats.t.ppf in CI test
```

Optional: `matplotlib` for examples.

---

## 12. Testing Strategy

> See [tests/](tests/)

### Unit Tests
1. **Invariant (CRITICAL):** `forward(h, u) == forward_polynomial(h, u, mask=ones)` — rtol=1e-5
2. **Library structure:** Verify `mult_table` against manual enumeration
3. **Unfolding correctness:** Known weights → analytically correct coefficients
4. **CI test / patience / mask updates**
5. **All-masked identity:** Masking all terms gives `h_i[t+1] = h_i[t]`

### Integration Tests
6. **Linear system recovery:** `x[t+1] = 0.9x + 0.1u` → verify coefficients
7. **Sparsity:** Irrelevant terms pruned for sparse ground-truth
8. **Autoencoder:** Forward, forecast, fit, save/load

---

## 13. Polynomial Parameterizations

> See [model.py](sindy_rnn/model.py)

Three modes via `decomposed` and `direct` flags:

**Factored** (`decomposed=False, direct=False`): Product of D affine projections. Bias interactions mix degrees.

**Decomposed** (`decomposed=True, direct=False`, default): Separate parameterization per degree — d=0 bias, d=1 linear, d>=2 bias-free products. Completely decouples coefficients across degrees.

**Direct** (`direct=True`): `theta = nn.Parameter(E, n_states, n_terms)`. No implicit regularization. Baseline for ablation.

All three implement `unfold_polynomial_coefficients() → (E, n_states, n_terms)`. The key invariant holds for all modes.

---

## 14. SparseAutoencoderRNN

> See [autoencoder.py](sindy_rnn/autoencoder.py)

Encoder-decoder wrapper around `PolynomialRNN` for sparse sensor → latent dynamics → full state reconstruction.

```
z_t = encoder(sparse_obs_t)                         # encode sensors to latent
z_{t+1} = z_t + dt * P(z_t)                         # forward Euler (autonomous)
full_pred_{t+1} = decoder(z_{t+1})                   # decode to full state
```

Constructor: `sparse_dim, full_dim, latent_dim, n_controls, ensemble_size, polynomial_degree, dt, encoder_type ('mlp'/'gru'), encoder_hidden_dims, encoder_gru_hidden_dim, encoder_num_layers, decoder_hidden_dims, encoder_dropout, decoder_dropout, dynamics_dropout, dynamics_feature_dropout, state_names, control_names, decomposed, direct`.

**Encoder types:** MLPEncoder (per-timestep) or GRUEncoder (temporal context via Takens' embedding). **MLPDecoder** maps latent → full state. Encoder/decoder shared across ensemble; only inner dynamics has E members.

### fit_autoencoder() training objective

SINDy-SHRED-style sliding windows: each sample is a sensor window of length LAGS
(stride=1). The GRU encoder produces one latent per window (final hidden state),
so every latent point has full temporal context. Dynamics pairs come from
consecutive windows' latent outputs.

Joint loss: `L = E_id + sindy_weight * E_sindy + l1 * |theta|`

- **E_id** (reconstruction): `decode(z_i) ≈ full_state_i` — same-timestep reconstruction. Anchors latent space.
- **E_sindy** (derivative matching): `P(z_i) ≈ dz/dt` — compares polynomial P to empirical derivatives from consecutive windows' latent outputs. Couples encoder to polynomial dynamics.
- **L1**: sparsity on polynomial coefficients

**Data format:**
- `sparse_obs`: `(N, LAGS, sparse_dim)` — N sliding windows with stride=1
- `full_state_target`: `(N, full_dim)` — same-timestep reconstruction target (one per window)

**Staged training**: E_sindy and L1 are gated by `sindy_warmup_epochs`. During `[0, sindy_warmup)`, only E_id trains.

**Two-phase gradient accumulation**: E_id is computed and backward'd separately (freeing the large decoded tensor) before re-encoding for E_sindy + L1. This prevents OOM on high-dimensional problems.

**Separate param groups**: `dynamics_learning_rate` (default: same as `learning_rate`) controls the polynomial dynamics lr independently. Set higher (e.g. `5e-2`) so ODE coefficients converge faster, while encoder/decoder stay at `1e-3`.

**Refit**: After joint training, freezes the encoder, extracts all latent trajectories, resets masks, and calls `fit()` on the PolynomialRNN for clean equation discovery on fixed latent space.

No bootstrap — all ensemble members see the same shared encoder output (diversity from random init + pruning).

### fit_autoencoder() differences from fit()

| Aspect | `fit()` | `fit_autoencoder()` |
|--------|---------|---------------------|
| L1 param name | `l2` | `l1` |
| Default pruning | `'ci'` | `'median'` |
| Default lr | `1e-2` | `1e-3` (encoder/decoder) |
| Dynamics lr | Same as lr | `dynamics_learning_rate` (separate param group) |
| Bootstrap | Pre-expands data | **No bootstrap** (shared encoder output) |
| Test eval | Full-batch | **Per-window loop** (avoids OOM) |
| Loss | Derivative matching | E_id + E_sindy + L1 (staged via sindy_warmup) |
| Grad clipping | `max_norm=100.0` | `max_norm=100.0` |
| Memory | Single backward | **Two-phase** (E_id freed before E_sindy) |
| Refit | Supported | Freeze encoder → extract z → reset masks → call `fit()` |

---

## 15. Benchmark Study

### 15.1 Overview

Three experimental settings:
1. **Lorenz parameter recovery** — factored vs direct vs STLSQ, 6 noise × 5 data sizes × 5 seeds = 450 experiments
2. **Cylinder flow** — SINDy-RNN-SHRED vs SHRED vs SINDy-SHRED, 400×1000 grayscale from 200 sensors
3. **SST** — same three methods on NOAA weekly SST from 250 ocean sensors

### 15.2 Lorenz Parameter Recovery

**Script:** [lorenz_parameter_recovery.py](examples/lorenz_parameter_recovery.py)

**Key finding:** Factored achieves 100% exact structure match at 5% noise / N>=5000, while both direct and STLSQ achieve 0%.

### 15.3 Cylinder Flow

**Data:** `data/flow_over_cylinder.npy` — 334 frames, 400×1000 (~1GB). Train 80%, Test 20%.

**Config:**
| Parameter | SINDy-RNN-SHRED | SINDy-SHRED | SHRED |
|-----------|-----------------|-------------|-------|
| Encoder | GRU(200→4), 2 layers | Same | Same |
| Decoder | MLP(4→350→400→400K) | Same | Same |
| poly_order/degree | 3 (cubic) | 3 | N/A |
| Ensemble | E=11, median | E=5 | N/A |
| epochs | 1000 | 1000 | 1000 |
| warmup/refit | 200/100 | patience=20 | N/A |
| lr (enc/dec) | 1e-3 | 5e-4 | 5e-4 |
| dynamics_lr | 5e-2 | N/A | N/A |
| batch_size | 64 | 64 | 64 |
| L1/sindy_reg | 1e-4 | 10.0 | N/A |
| pruning_threshold | 0.1 | 1e-3 | N/A |
| dt | 1/30 | N/A | N/A |
| lags | 60 | 60 | 60 |
| sindy_weight | 1.0 | N/A | N/A |
| sindy_warmup | 100 | N/A | N/A |

**GPU memory:** SINDy-SHRED decoder ~160M params. Must `.cpu()` and `torch.cuda.empty_cache()` between methods. Two-phase gradient accumulation in `fit_autoencoder()` prevents OOM by freeing E_id decode tensors before computing E_sindy.

**Script:** [cylinder_benchmark.py](examples/cylinder_benchmark.py)

### 15.4 SST

**Data:** NOAA OI SST V2 (1992–2019), 1400 weekly snapshots, ~44K sea grid points. Train 80%, Test last ~318 frames. 250 random sensors.

**Config:**
| Parameter | SINDy-RNN-SHRED | SINDy-SHRED | SHRED |
|-----------|-----------------|-------------|-------|
| Encoder | GRU(250→3), 2 layers | Same | Same |
| Decoder | MLP(3→350→400→44,219) | Same | Same |
| poly_order/degree | 1 (linear) | 3 | N/A |
| Ensemble | E=11 | E=5 | N/A |
| epochs | 1000 | 1000 | 1000 |
| warmup/refit | 200/1000 | patience=5 | N/A |
| lr (enc/dec) | 1e-3 | 1e-3 | 1e-3 |
| dynamics_lr | 5e-2 | N/A | N/A |
| L1/sindy_reg | 1e-3 | 10.0 | N/A |
| pruning_threshold | 0.1 | 1.0 | N/A |
| dt | 1/52 | N/A | N/A |
| lags | 52 | 52 | 52 |
| sindy_weight | 1.0 | N/A | N/A |
| sindy_warmup | 100 | N/A | N/A |

**Script:** [sst_benchmark.py](examples/sst_benchmark.py)

### 15.5 Evaluation Protocol

All methods evaluated on same held-out test frames: MSE (unscaled) and relative error = ||pred - true||_F / ||true||_F. Teacher-forced reconstruction for all.

### 15.6 SINDy-SHRED Reference

Code in `sindy-shred/` (root level). Key files: `sindy_shred.py`, `sindy_shred_net.py` (E_SINDy, `num_replicates=5` hardcoded), `sindy.py`, `utils.py`.

```python
import math; import numpy as np; np.math = math  # pysindy numpy 2.x fix
sys.path.insert(0, 'sindy-shred/')
from sindy_shred import SINDySHRED
```

---

## 16. Repository Structure

```
sindy-rnn/
├── sindy_rnn/
│   ├── __init__.py                    # Public API
│   ├── model.py                       # PolynomialRNN, EnsembleRNNModule,
│   │                                  #   EnsemblePolynomialLayer,
│   │                                  #   DecomposedPolynomialLayer, EnsembleLinear
│   ├── training.py                    # fit()
│   ├── autoencoder.py                 # SparseAutoencoderRNN, fit_autoencoder
│   ├── pruning.py                     # ensemble_prune, median_effect_test, etc.
│   ├── polynomial_library.py          # build_library_structure, etc.
│   └── equations.py                   # get_coefficients, get_equations, etc.
├── examples/
│   ├── lorenz_parameter_recovery.py   # Lorenz noise/data grid (450 experiments)
│   ├── lorenz_noise_study.py          # Single-noise-level Lorenz study
│   ├── cylinder_benchmark.py          # Cylinder: single seed, 3 methods
│   ├── cylinder_benchmark_seeds.py    # Cylinder: 5 seeds, 3 methods
│   ├── sst_benchmark.py              # SST: 5 seeds, 3 methods
│   └── sanity_check.py               # Quick 1-seed sanity check
├── tests/
│   ├── test_polynomial_layer.py       # forward == forward_polynomial invariant
│   ├── test_unfolding.py              # Known polynomial recovery
│   ├── test_pruning.py                # CI test, patience, mask updates
│   ├── test_training.py               # End-to-end: linear system recovery
│   └── test_autoencoder.py            # Autoencoder tests
├── sindy-shred/                       # SINDy-SHRED reference code (external)
├── data/                              # Dataset files (not in git)
├── requirements.txt
├── CLAUDE.md                          # This file
└── README.md
```
