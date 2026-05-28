# Plan: SINDy-RNN-SHRED — MLP + Polynomial ODE replacing GRU

## Concept

Replace GRU temporal processing in SHRED with MLP (per-timestep) + SINDy-RNN (polynomial ODE integration):

```
Stage 1 (Sensing/Reconstruction):
  x_sparse[t] → MLP → u[t]                     # per-timestep encoding
  z[0] = 0
  z[t+1] = z[t] + dt * P(z[t], u[t])            # polynomial ODE with forcing
  X_full = decoder(z[LAGS-1])                    # decode final state
  Loss = MSE(X_full, X_full_target)              # reconstruction only, no L1/pruning

Stage 2 (Discovery):
  Freeze encoder + decoder
  Extract z trajectories from all sliding windows
  Refit autonomous SINDy-RNN: dz/dt = P(z)      # with L1 + pruning
```

Key design decisions:
- dim(u) = latent_dim (same as z)
- z[0] = 0 (forcing u drives state from zero)
- Stage 1: no SINDy regularization — polynomial structure is the only inductive bias
- Stage 2: autonomous refit on extracted z trajectories

## Files to create

### 1. `sindy_rnn/sindy_rnn_shred.py` — New model + training

**Class `SINDyRNNSHRED(nn.Module)`:**
- `sensor_encoder`: `MLPEncoder(sparse_dim, latent_dim, hidden_dims, dropout)`
- `dynamics`: `PolynomialRNN(n_states=latent_dim, n_controls=latent_dim, ...)` — u is forcing
- `decoder`: `MLPDecoder(latent_dim, full_dim, hidden_dims, dropout)`
- `forward(sparse_obs)`:
  - `u = sensor_encoder(sparse_obs)` — `(B, LAGS, latent_dim)`
  - Expand u to `(E, B, LAGS, latent_dim)` for ensemble
  - `z = zeros(E, B, latent_dim)` — initial state
  - Loop t=0..LAGS-1: `z = dynamics.rnn.forward_polynomial(z, u[:,:,t,:], mask, theta)`
  - `full_pred = decoder(z)` — `(E, B, full_dim)`
  - Return `full_pred, z, u`
- `extract_latent_trajectory(sparse_obs)`:
  - Process all sliding windows, collect z at each timestep
  - Returns `(N_frames, latent_dim)` trajectory
- Save/load, equation delegation (same pattern as SparseAutoencoderRNN)

**Function `fit_sindy_rnn_shred()`:**

Stage 1 training loop:
- Data: sliding windows `(N, LAGS, sparse_dim)` + targets `(N, full_dim)`
- Mini-batch over windows
- Forward pass through model → decoded full state (mean across ensemble)
- Loss = MSE(decoded, target) — no L1, no SINDy regularization
- Adam optimizer, gradient clipping
- Test loss tracking

Stage 2 refit:
- Freeze encoder + decoder
- Extract z trajectory: run all windows through model, collect final z per window
- These z values are consecutive (stride-1 windows), forming a smooth trajectory
- Build xs/ys for `fit()`: `xs = z[:-1]`, `ys = z[1:]` (autonomous, no controls)
- Create new PolynomialRNN(n_states=latent_dim, n_controls=0) for autonomous discovery
- Call `fit()` with L1, pruning, centered_diff

### 2. `examples/sst_sindy_rnn_shred.py` — SST benchmark

Structure mirrors existing `sst_benchmark.py`:
- Same data loading, scaling, sensor selection
- Same sliding window preparation
- Same evaluation protocol (reconstruction MSE/rel_error on test frames)
- Runs: sindy-rnn-shred, sindy-shred, shred (3 methods)
- Plots: reconstruction images, latent dynamics, forecast MSE

Config for sindy-rnn-shred:
- Encoder: MLP(250 → 128 → 64 → 3)
- Dynamics: PolynomialRNN(n_states=3, n_controls=3, degree=3, E=11)
- Decoder: MLP(3 → 350 → 400 → 44K)
- Stage 1: ~1000 epochs, lr=1e-3, no L1, no pruning
- Stage 2 refit: ~1000 epochs, lr=5e-2, L1=5e-2, pruning

### 3. `sindy_rnn/__init__.py` — Update exports

Add `SINDyRNNSHRED` and `fit_sindy_rnn_shred` to exports.

## Implementation order

1. Create `sindy_rnn/sindy_rnn_shred.py` with model class + training function
2. Update `sindy_rnn/__init__.py`
3. Create `examples/sst_sindy_rnn_shred.py`
4. Sanity test: single-seed quick run to verify training works
