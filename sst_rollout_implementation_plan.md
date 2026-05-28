# SST Rollout Architecture — Implementation Plan

A SHRED-style sea surface temperature reconstruction model where a GRU encoder produces an initial latent state, a polynomial SINDy-RNN propagates that state autonomously, and a decoder reconstructs the full field at every step.

---

## 1. Architecture

### 1.1 Computational graph

```
x_sparse[0:T_w]  ──►  GRU encoder  ──►  z_0
                                          │
                                          ▼
                                    z_1 = f(z_0)
                                          │
                                          ▼
                                    z_2 = f(z_1)
                                          │
                                          ⋮
                                          ▼
                                    z_{T_cur} = f(z_{T_cur-1})
                                          │
                       Decoder applied at every step:
                                          │
                            x̂_t = D(z_t)  for t = 0, …, T_cur
```

### 1.2 Forward equations

$$
\mathbf{z}_0 = E_\phi(\mathbf{x}_{0:T_w}^{\text{sparse}})
$$

$$
\mathbf{z}_{t+1} = f_\theta(\mathbf{z}_t), \quad t = 0, \dots, T_{\text{cur}} - 1
$$

$$
\hat{\mathbf{x}}_t^{\text{full}} = D_\psi(\mathbf{z}_t), \quad t = 0, \dots, T_{\text{cur}}
$$

### 1.3 Module specification

| Module | Type | Input | Output | Notes |
|---|---|---|---|---|
| `E_φ` | small GRU | $\mathbf{x}_{0:T_w}^{\text{sparse}} \in \mathbb{R}^{T_w \times s}$ | $\mathbf{z}_0 \in \mathbb{R}^n$ | hidden = $n$; take last hidden state |
| `f_θ` | `EnsembleRNNModule` (existing) | $\mathbf{z}_t$ | $\mathbf{z}_{t+1}$ | `n_states = n`, `n_controls = 0`, autonomous |
| `D_ψ` | linear or shallow MLP | $\mathbf{z}_t$ | $\hat{\mathbf{x}}_t^{\text{full}} \in \mathbb{R}^{F}$ | start linear; upgrade only if needed |

Keep `D_ψ` deliberately weak. A linear decoder forces $\mathbf{z}_t$ to encode the full-field structure linearly, which prevents the "decoder absorbs dynamics" shortcut.

---

## 2. Training objective

### 2.1 Loss

$$
\mathcal{L} = \underbrace{\frac{1}{T_{\text{cur}} + 1}\sum_{t=0}^{T_{\text{cur}}}\|\hat{\mathbf{x}}_t^{\text{full}} - \mathbf{x}_t^{\text{full}}\|^2}_{\text{per-step reconstruction}} \;+\; \lambda_0\|\mathbf{z}_0\|^2 \;+\; \lambda_s\,\|\boldsymbol{\Theta}_f\|_1
$$

The per-step term is critical — supervising only the final step makes the architecture nearly untrainable.

### 2.2 Hyperparameter defaults

| Knob | Value | Reasoning |
|---|---|---|
| $T_w$ | 8–16 | warmup window; ~1–2 seasons for weekly SST |
| $n$ (latent dim) | 8–16 | start small; grow if reconstruction plateaus |
| $\lambda_0$ | $10^{-3}$ | keep $\mathbf{z}_0$ numerically tame |
| $\lambda_s$ | 0 → $10^{-3}$ | sparsity ramp; off until curriculum finishes |
| optimizer | AdamW, lr $10^{-3}$ | standard |
| lr schedule | cosine decay | standard |
| gradient clip | $\|\nabla\| \le 0.5$ | rollouts produce occasional spikes |
| polynomial degree | 2 (start) | match Lorenz; raise if underfitting |
| integrator | Euler in early curriculum, RK4 once $T_{\text{cur}} \ge 16$ | accuracy matters at long horizons |

---

## 3. Rollout-length curriculum

### 3.1 Schedule

$$
T_{\text{cur}}(e) = \min\!\left(T_{\max},\; T_{\text{start}} + \left\lfloor \tfrac{e}{E_{\text{step}}} \right\rfloor \cdot \Delta T \right)
$$

| Phase | $T_{\text{cur}}$ | What's happening |
|---|---|---|
| 1 (one-step) | 1–2 | Lorenz-regime training; $f$ learns local map |
| 2 (short rollout) | 4–16 | $f$ starts seeing accumulated error |
| 3 (medium) | 16–48 | dynamics structure emerges |
| 4 (long, target) | 48–100+ | identifiability, sparsity ramp on |

Defaults: $T_{\text{start}} = 1$, $\Delta T = 2$, $E_{\text{step}}$ chosen so curriculum completes in the first 50% of total epochs.

### 3.2 Sparsity coupling

Turn on $\lambda_s$ only after the curriculum reaches $T_{\max}$. Sparsifying a still-shaping $f$ is the moving-target trap from the earlier discussion.

---

## 4. Implementation steps

### Step 1 — Data pipeline

- Load weekly SST cube, shape `(N_time, H, W)`.
- Choose sparse sensor mask of $s$ pixels (random or POD-greedy); `x_sparse` is the masked subset, `x_full` is the flattened full field.
- Normalize: per-pixel z-score using training-set statistics.
- Create sequences of length $T_w + T_{\max} + 1$. Stride flexibly; for SST, weekly stride is fine.
- Splits: chronological train/val/test (no shuffling across the time axis).

### Step 2 — Model class

```python
class SSTRolloutModel(nn.Module):
    def __init__(self, n_sensors, n_latent, n_full,
                 polynomial_degree=2, T_w=12):
        super().__init__()
        self.T_w = T_w
        self.encoder = nn.GRU(n_sensors, n_latent, batch_first=True)
        self.dynamics = EnsembleRNNModule(
            ensemble_size=1,           # E=1 for now; can grow later
            n_states=n_latent,
            n_controls=0,              # autonomous
            polynomial_degree=polynomial_degree,
            decomposed=True,
        )
        self.decoder = nn.Linear(n_latent, n_full)

    def forward(self, x_sparse_warmup, T_cur, integrator='euler'):
        # x_sparse_warmup: (B, T_w, s)
        _, h_n = self.encoder(x_sparse_warmup)         # (1, B, n)
        z = h_n.squeeze(0).unsqueeze(0)                # (E=1, B, n)

        z_traj = [z]
        for _ in range(T_cur):
            z = self.dynamics.forward_polynomial(z, integrator=integrator)
            z_traj.append(z)

        z_stack = torch.stack(z_traj, dim=2)           # (E, B, T_cur+1, n)
        x_hat = self.decoder(z_stack.squeeze(0))       # (B, T_cur+1, F)
        return x_hat, z_stack.squeeze(0)
```

Notes:
- `forward_polynomial` is preferable over the fast `forward` because it goes through the unfolded coefficients — same path that sparsity acts on, no fast/slow drift.
- Ensemble size 1 to start; the architecture supports $E > 1$ later for uncertainty.

### Step 3 — Training loop skeleton

```python
for epoch in range(E_total):
    T_cur = min(T_max, T_start + (epoch // E_step) * dT)
    integrator = 'rk4' if T_cur >= 16 else 'euler'
    use_sparsity = T_cur >= T_max

    for batch in train_loader:
        x_sparse = batch['x_sparse'][:, :T_w]                     # warmup
        x_full   = batch['x_full'][:, :T_cur+1]                   # target

        x_hat, z_traj = model(x_sparse, T_cur, integrator)

        L_rec = F.mse_loss(x_hat, x_full)
        L_z0  = lambda_0 * (z_traj[:, 0]**2).mean()
        L_sp  = 0.0
        if use_sparsity:
            theta = model.dynamics.unfold_polynomial_coefficients()
            L_sp = lambda_s * theta.abs().mean()

        L = L_rec + L_z0 + L_sp
        opt.zero_grad()
        L.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
```

### Step 4 — Sanity check on Lorenz first

Before SST, validate on Lorenz with known ground-truth coefficients:

- Generate Lorenz trajectory (length ~10k), add 20–50% Gaussian noise to match your prior experiments.
- "Sparse sensors" = 1 or 2 of the 3 Lorenz coordinates.
- "Full field" = all 3 coordinates.
- Train; at convergence, unfold $\boldsymbol{\Theta}_f$ and compare to true Lorenz coefficients.

If Lorenz works, graduate to SST. If not, the bug is in the training setup — debug there, not on SST where the answer is unknown.

### Step 5 — Diagnostics to log every epoch

| Metric | What to watch for |
|---|---|
| reconstruction loss (train / val) | should decrease monotonically; spikes during $T_{\text{cur}}$ jumps are normal |
| latent norm $\langle\|\mathbf{z}_t\|\rangle$ over $t$ | should stay bounded; growing-with-$t$ means $f$ is unstable |
| linear DMD residual on $\mathbf{z}_{0:T_{\text{cur}}}$ | tells you whether dynamics are non-linear at all |
| $\|\boldsymbol{\Theta}_f\|_0$ after sparsity ramp | sparsity progression |
| pure-rollout val loss vs. shorter-rollout val loss | gap should shrink as curriculum progresses |
| stability eigenvalue $\max\|1 + \Delta t \lambda(J)\|$ | use existing `compute_stability_loss`; should stay near 1 |

### Step 6 — Forecasting at inference

```python
def forecast(model, x_sparse_warmup, n_steps):
    model.eval()
    _, h_n = model.encoder(x_sparse_warmup)
    z = h_n.squeeze(0).unsqueeze(0)
    preds = []
    for _ in range(n_steps):
        z = model.dynamics.forward_polynomial(z, integrator='rk4')
        preds.append(model.decoder(z.squeeze(0)))
    return torch.stack(preds, dim=1)
```

---

## 5. Failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Training diverges immediately at long $T_{\text{cur}}$ | curriculum too aggressive | smaller $\Delta T$ or larger $E_{\text{step}}$ |
| Latent norm explodes during rollout | $f$ has unstable eigenvalues | add stability penalty (already implemented), reduce lr |
| Reconstruction good, $\boldsymbol{\Theta}_f$ near zero | decoder absorbing dynamics | weaken decoder (linear, smaller $n$) |
| $f$ recovers identity after sparsity | $\lambda_s$ too high | reduce; consider $\ell_{1/2}$ instead of $\ell_1$ |
| Val loss diverges from train loss after curriculum step | overfitting to short-horizon shortcut | longer at each $T_{\text{cur}}$ stage |
| Lorenz coefficients wrong but reconstruction good | latent space is rotated/scaled version of true state | expected — coefficients are correct up to coordinate transform |

---

## 6. Milestones

1. **Lorenz validation.** Architecture recovers Lorenz coefficients (up to coordinate transform) from sparse, noisy observations with the curriculum.
2. **SST baseline.** Train on SST with $n = 8$, linear decoder, 1-week-ahead reconstruction. Compare to your existing SHRED/GRU baseline.
3. **SST forecasting.** Evaluate autonomous rollout from $\mathbf{z}_0$ on held-out time windows.
4. **Sparsity readout.** With $\lambda_s$ on, examine which polynomial terms survive — this is the discovered dynamics on SST.

Milestones 1–2 establish the architecture works. Milestones 3–4 are the actual scientific output.
