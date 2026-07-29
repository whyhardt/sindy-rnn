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
3. **Ensemble pruning** — agreement/median test across ensemble members with patience-based stability

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

**Agreement test (default):** Each active member individually votes a term "exists" iff its own `|coef| > δ`. Term survives iff at least `agreement_frac` of active members agree (default 50%). Simple and cheap — no distributional assumptions.

**Median:** Term survives iff `|median(coef)| > δ` across active members. Robust to ensemble bifurcation from multicollinearity.

Both require at least 2 active members. Theta is already in ODE units — no conversion needed.

### 5.3 Patience Mechanism

Terms must fail the pruning test **2 consecutive times** before permanent removal. `ensemble_prune()` dispatches to agreement or median test, updates patience counters, and permanently masks terms that exceed patience.

### 5.4 Threshold Fallback

For E=1, `threshold_patience_update()` increments patience for `|coef| < threshold`, then `threshold_prune()` removes terms exceeding patience limit.

---

## 6. Training

> See [training.py](sindy_rnn/training.py) — `fit()`

Single-stage training:
1. **MSE loss** on teacher-forced next-state prediction (NaN-masked for variable-length sequences)
2. **L1 penalty on unfolded θ** — `l2 * theta.abs().mean()` (not AdamW weight decay). Plain Adam optimizer.
3. **Periodic pruning** via agreement/median test or threshold fallback

**Teacher forcing is essential** — free-running from initial_state causes zero gradients when initial polynomial is near-zero.

**Data format:** `xs: (B, T, n_states + n_controls)`, `ys: (B, T, n_states)`. NaN-pad for variable-length.

**Key params:** `epochs, warmup_steps, learning_rate, l2 (L1 weight), pruning_frequency, pruning_threshold (δ), pruning_method ('agreement'/'median'), agreement_frac, refit_epochs`.

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
    agreement_frac=0.5,
    pruning_threshold=0.2,
    pruning_method='agreement',
    pruning_frequency=100,
    learning_rate=5e-2,
    l2=5e-2,              # L1 penalty weight
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
```

Optional: `matplotlib`, `scipy`, `pyyaml`, `scikit-learn`, `pysindy` for examples.

---

## 12. Testing Strategy

> See [tests/](tests/)

### Unit Tests
1. **Invariant (CRITICAL):** `forward(h, u) == forward_polynomial(h, u, mask=ones)` — rtol=1e-5
2. **Library structure:** Verify `mult_table` against manual enumeration
3. **Unfolding correctness:** Known weights → analytically correct coefficients
4. **Agreement/median test / patience / mask updates**
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

## 14. Benchmark Study

### 14.1 Overview

Each dataset (`lorenz`, `cylinder`, `sst`) lives in its own `examples/<dataset>/` folder with an identical shape:

```
examples/<dataset>/
├── config.yaml           # single source of truth for all tunable values —
│                          #   both training scripts and analyze.py load this,
│                          #   so data prep is guaranteed identical across methods
├── data.py                # data loading/prep functions (take config values as args)
├── data/                  # raw dataset files (cylinder/sst) or cached generated
│                          #   data (lorenz) — gitignored
├── params/                 # trained model checkpoints — gitignored
├── results/                 # figures + metrics.json from analyze.py — gitignored
├── train_sindy_rnn.py       # trains sindy-rnn, saves to params/
├── train_sindy_shred.py     # (cylinder/sst) or train_stlsq.py (lorenz) —
│                            #   trains the baseline, saves to params/
└── analyze.py                # loads both from params/, evaluates on the SAME
                               #   held-out frames, writes figures + metrics.json
                               #   to results/
```

Shared, model-agnostic evaluation/plotting helpers live in `examples/_common/`
(`eval.py`, `plotting.py`) — used by every `analyze.py`.

**Known asymmetry (all three datasets' baselines except STLSQ):** SINDy-SHRED
uses its held-out validation split for early-stopping/best-checkpoint
selection during training; `fit_rollout()` only logs held-out loss without
acting on it, so sindy-rnn keeps whatever the final training epoch produces.
Not corrected — treat as a caveat when comparing results.

#### Estimator wrappers (`examples/_common/estimators.py`)

Every train/analyze script goes through a thin, uniform wrapper rather than
calling `fit()`/`fit_rollout()`/`SINDySHRED`/`pysindy` directly:

```python
est.fit(xs, ys)          # trains; saves nothing on its own
est.predict(xs)           # whatever the model naturally produces when run
                          #   forward — a next-state, a derivative, or a
                          #   decoded full state; the wrapper doesn't care which
est.simulate(x0, n_steps)  # autonomous multi-step rollout from x0
est.save(path); Cls.load(path)
```

Four wrappers, one per method:
- `PolynomialRNNEstimator` — wraps `PolynomialRNN` + `fit()` (Lorenz derivative matching).
- `RolloutSINDyRNNEstimator` — wraps `RolloutSINDyRNN` + `fit_rollout()` (cylinder/SST sparse-sensor, or Lorenz `identity=True`).
- `SindyShredEstimator` — wraps the external `SINDySHRED` reference class.
- `StlsqEstimator` — wraps pysindy's ensemble STLSQ (E-SINDy); `predict()`/`simulate()` use the coefficient matrix directly (Euler-integration-consistent with the other three) rather than pysindy's own `predict()`/`simulate()`.

**`SindyShredEstimator` gives SINDySHRED real save/load**, which the
upstream reference class itself doesn't have — its train/val/test split and
post-hoc SINDy model are normally tied to the live object from `fit()`. The
estimator drops down one level to two independently well-behaved pieces:
the raw `SINDy_SHRED_net` (`.net`, stateless at inference — `predict()`
bypasses SINDySHRED's split-locked `sensor_recon()` and calls the net
directly on arbitrary windows) and the fitted pysindy model on the latent
trajectory (`.sindy_model`, plain picklable object). `save()` persists both;
`load()` reconstructs `predict()`/`simulate()` from them without needing a
live `SINDySHRED` object at all.

### 14.2 Lorenz

Fully observed (no sparse sensors) — two alternative training objectives,
both on the identical noisy trajectory (config.yaml + data.py):

1. **Derivative matching** (`train_sindy_rnn.py`): direct application of
   `PolynomialRNN` + `fit()`, matching §10. Compares `P(h)` to empirical
   `dh/dt` at every point — fast to optimize, but finite-differencing noisy
   data amplifies the noise (derivative matching is noise-sensitive, same
   critique classical SINDy gets — see §1).
2. **Trajectory matching** (`train_sindy_rnn_rollout.py`): `RolloutSINDyRNN`
   with `identity=True` — the same GRU-encoder/decoder rollout mechanism
   used for cylinder/SST, but with no encoder/decoder (`z` IS the observed
   state; requires `n_sensors == n_latent == n_full`). Compares an
   *integrated* multi-step trajectory to noisy observations instead of a
   pointwise derivative, which averages out zero-mean noise instead of
   amplifying it, at the cost of a harder optimization landscape (needs the
   rollout-length curriculum + per-step noise injection from
   `fit_rollout()`). Because Lorenz is chaotic (Lyapunov time ≈ 1 unit ≈
   100 steps at `dt=0.01`), `T_max` is capped near one Lyapunov time in
   `config.yaml`'s `sindy_rnn_rollout` section — beyond that, trajectory MSE
   against one noisy realization stops being a meaningful training signal.

Baseline for both is STLSQ/E-SINDy (`pysindy`), not SINDy-SHRED, since
SHRED's sparse-sensor premise doesn't apply to an already fully-observed
3-state system.

**Scripts:** [examples/lorenz/train_sindy_rnn.py](examples/lorenz/train_sindy_rnn.py), [train_sindy_rnn_rollout.py](examples/lorenz/train_sindy_rnn_rollout.py), [train_stlsq.py](examples/lorenz/train_stlsq.py), [analyze.py](examples/lorenz/analyze.py) (evaluates whichever checkpoints exist in `params/`)

For the full noise/data-size/seed sweep (7 noise levels × 5 data lengths ×
5 seeds × 3 methods = 525 cells; factored vs direct vs STLSQ), see
[recovery_run.py](examples/lorenz/recovery_run.py) and
[recovery_aggregate.py](examples/lorenz/recovery_aggregate.py) — these don't
fit the train-once/analyze-once pattern above and are kept as standalone
sweep scripts in the same folder. `recovery_run.py` runs exactly one grid
cell per invocation and writes its own result JSON to `results/recovery/`
(the natural unit for a cluster array job — `--grid_index $SLURM_ARRAY_TASK_ID`
or explicit `--noise_frac/--n_steps/--seed/--method`); `recovery_aggregate.py`
combines whatever cells have finished (safe to run on a partial sweep) into
summary tables + plots. **Key finding:** factored achieves 100% exact
structure match at 5% noise / N>=5000, while both direct and STLSQ achieve
0%. (This sweep predates trajectory matching; it has not yet been extended
to the rollout objective.)

### 14.3 Cylinder Flow

**Data:** `examples/cylinder/data/flow_over_cylinder.npy` — 334 frames,
400×1000 (~1GB). Train/test split and all hyperparameters are in
[examples/cylinder/config.yaml](examples/cylinder/config.yaml).

**GPU memory:** SINDy-SHRED decoder ~160M params. Must `.cpu()` and
`torch.cuda.empty_cache()` between methods if running both in one process.

### 14.4 SST

**Data:** NOAA OI SST V2 (1992–2019), 1400 weekly snapshots, ~44K sea grid
points, at `examples/sst/data/SST_data.mat`. Train/test split and all
hyperparameters are in [examples/sst/config.yaml](examples/sst/config.yaml).
Test frames start after SINDy-SHRED's own validation buffer
(`validate_length`) so both methods are scored on an identical,
non-overlapping held-out region.

### 14.5 Evaluation Protocol

All methods evaluated on the same held-out test frames: MSE (unscaled) and
relative error = ||pred - true||_F / ||true||_F. See `examples/_common/eval.py`.

### 14.6 SINDy-SHRED Reference

Code in `sindy-shred/` (root level). Key files: `sindy_shred.py`, `sindy_shred_net.py` (E_SINDy, `num_replicates=5` hardcoded), `sindy.py`, `utils.py`.

```python
import math; import numpy as np; np.math = math  # pysindy numpy 2.x fix
sys.path.insert(0, 'sindy-shred/')
from sindy_shred import SINDySHRED
```

---

## 15. Repository Structure

```
sindy-rnn/
├── sindy_rnn/
│   ├── __init__.py                    # Public API
│   ├── model.py                       # PolynomialRNN, EnsembleRNNModule,
│   │                                  #   EnsemblePolynomialLayer,
│   │                                  #   DecomposedPolynomialLayer, EnsembleLinear
│   ├── training.py                    # fit()
│   ├── rollout.py                     # RolloutSINDyRNN, fit_rollout (current best model)
│   ├── pruning.py                     # ensemble_prune, median_effect_test, etc.
│   ├── polynomial_library.py          # build_library_structure, etc.
│   └── equations.py                   # get_coefficients, get_equations, etc.
├── examples/
│   ├── _common/                       # shared across every dataset's scripts
│   │   ├── estimators.py              # fit/predict/simulate/save/load wrappers,
│   │   │                              #   one per method (see §14.1)
│   │   ├── eval.py                    # model-agnostic reconstruction/forecast metrics
│   │   └── plotting.py                # model-agnostic field/latent/MSE plots
│   ├── lorenz/
│   │   ├── config.yaml                # all tunable values (data + all methods)
│   │   ├── data.py                    # generate_lorenz, add_noise, ground truth
│   │   ├── train_sindy_rnn.py         # PolynomialRNN + fit(), derivative matching
│   │   ├── train_sindy_rnn_rollout.py # RolloutSINDyRNN(identity=True) + fit_rollout(),
│   │   │                              #   trajectory matching (noise-robust alternative)
│   │   ├── train_stlsq.py             # E-SINDy (ensemble STLSQ + bagging), via pysindy
│   │   ├── analyze.py                 # coefficient recovery + forecast comparison
│   │   ├── recovery_run.py            # single grid-cell noise/data-size/seed sweep run
│   │   │                              #   (cluster array job unit) -> results/recovery/
│   │   └── recovery_aggregate.py      # combines results/recovery/*.json into tables + plots
│   ├── cylinder/
│   │   ├── config.yaml
│   │   ├── data.py
│   │   ├── train_sindy_rnn.py         # RolloutSINDyRNN + fit_rollout()
│   │   ├── train_sindy_shred.py       # reference SINDySHRED
│   │   └── analyze.py
│   └── sst/
│       ├── config.yaml
│       ├── data.py
│       ├── train_sindy_rnn.py
│       ├── train_sindy_shred.py
│       └── analyze.py
├── tests/
│   ├── test_polynomial_layer.py       # forward == forward_polynomial invariant
│   ├── test_unfolding.py              # Known polynomial recovery
│   ├── test_pruning.py                # Agreement/median test, patience, mask updates
│   ├── test_training.py               # End-to-end: linear system recovery
│   ├── test_rollout.py                # RolloutSINDyRNN identity mode (no encoder/decoder)
│   └── test_estimators.py             # examples/_common/estimators.py wrappers
├── sindy-shred/                       # SINDy-SHRED reference code (external)
├── requirements.txt
├── CLAUDE.md                          # This file
└── README.md
```
