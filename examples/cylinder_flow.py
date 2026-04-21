"""Cylinder flow dynamics discovery from sparse pixel measurements.

Discovers latent governing equations for von Karman vortex shedding from
sparse pixel measurements of a dyed water channel experiment.

Data: GoPro video of flow over cylinder at Re=171
  - 334 frames at 30 FPS, 400x1000 grayscale pixels
  - 200 random sensor pixels (0.05% spatial coverage)

Architecture (teacher-forced next-state prediction):
  For each timestep:
    z_t = encoder(sparse_pixels_t)                               # R^200 -> R^4
    z_{t+1} = (1-α)*z_t + P(z_t)                               # polynomial dynamics in latent space
    full_pred_{t+1} = decoder(z_{t+1})                           # R^4 -> R^400000

  Training loss: MSE(full_pred_{t+1}, true_frame_{t+1})  (predict NEXT frame)
  Forecast: encoder provides z_0, then evolve z_{t+1} = (1-α)*z_t + P(z_t), decode

SINDy-SHRED reference results (Gao et al., PNAS 2026):
  4D nonlinear ODE (Eq. 5):
    z1_dot = -0.69*z2 + 0.98*z3 - 0.40*z4
    z2_dot =  1.00*z1 - 0.78*z3 - 0.31*z2*z3^2
    z3_dot = -1.029*z1 + 0.59*z2 + 0.41*z4
    z4_dot = -0.26*z1^2 - 0.29*z2^2*z3 - 0.39*z3^2
  Pixel-space MSE: 0.030
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from sindy_rnn import SparseAutoencoderRNN, fit_autoencoder


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_cylinder_data(path='data/flow_over_cylinder.npy'):
    """Load cylinder flow data and flatten spatial dimensions."""
    data = np.load(path)  # (334, 400, 1000)
    n_frames = data.shape[0]
    data_flat = data.reshape(n_frames, -1)  # (334, 400000)
    return data_flat


def prepare_data(X, sensor_locs, window_size):
    """Chunk trajectory into windowed (sparse_obs, full_state_next) pairs.

    Prediction targets are the NEXT timestep: sparse sensors at time t,
    full state at time t+1.

    Args:
        X: (N, full_dim) — full normalized field
        sensor_locs: array of sensor indices
        window_size: timesteps per window

    Returns:
        sparse_obs: (B, T, sparse_dim) — sensors at time t
        full_state_next: (B, T, full_dim) — full state at time t+1
    """
    N = len(X)
    usable = N - 1  # lose 1 frame for next-step targets
    n_windows = usable // window_size

    sparse_all = X[:n_windows * window_size, sensor_locs]
    full_next_all = X[1:n_windows * window_size + 1]

    sparse_obs = sparse_all.reshape(n_windows, window_size, -1)
    full_state_next = full_next_all.reshape(n_windows, window_size, -1)
    return sparse_obs, full_state_next


def main():
    print("Cylinder Flow Dynamics Discovery with Encoder-Decoder Polynomial RNN")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ---- Configuration ----
    config = {
        'seed': 0,
        'num_sensors': 200,
        'latent_dim': 4,
        'polynomial_degree': 2,       # nonlinear (quadratic) dynamics
        'ensemble_size': 11,
        'window_size': 30,             # ~1 second of video at 30 FPS
        'train_fraction': 0.8,
        'dt': 1 / 30,                  # 30 FPS
        'encoder_hidden_dims': [350, 400],
        'decoder_hidden_dims': [400, 350],
        'encoder_dropout': 0.1,
        'decoder_dropout': 0.1,
        'dynamics_dropout': 0.1,
        'epochs': 500,
        'warmup_steps': 200,
        'learning_rate': 1e-3,
        'l1': 1e-3,
        'pruning_threshold': 0.05,
        'pruning_method': 'median',
        'pruning_frequency': 20,
        'refit_epochs': 100,
        'batch_size': 1,              # mini-batch: full_dim=400k needs small batches
    }

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # ---- Load and preprocess data ----
    print("\nLoading cylinder flow data...")
    X = load_cylinder_data()
    n_frames, full_dim = X.shape
    print(f"  Shape: ({n_frames}, {full_dim})")
    print(f"  Range: [{X.min():.4f}, {X.max():.4f}]")

    # Select random sensor pixels
    sensor_locs = np.random.choice(full_dim, size=config['num_sensors'], replace=False)
    print(f"  Sensors: {config['num_sensors']} / {full_dim} ({100*config['num_sensors']/full_dim:.3f}%)")

    # Train/test split (temporal)
    train_end = int(n_frames * config['train_fraction'])
    print(f"  Train frames: {train_end}, Test frames: {n_frames - train_end}")

    # MinMax normalize (fit on training split only)
    scaler = MinMaxScaler()
    scaler.fit(X[:train_end])
    X_scaled = scaler.transform(X)

    # ---- Prepare train/test data ----
    X_train = X_scaled[:train_end]
    X_test = X_scaled[train_end:]

    sparse_train, full_next_train = prepare_data(X_train, sensor_locs, config['window_size'])
    sparse_test, full_next_test = prepare_data(X_test, sensor_locs, config['window_size'])

    print(f"\n  Train windows: {sparse_train.shape[0]} x {sparse_train.shape[1]} timesteps")
    print(f"  Test windows:  {sparse_test.shape[0]} x {sparse_test.shape[1]} timesteps")

    sparse_train_t = torch.tensor(sparse_train, dtype=torch.float32).to(DEVICE)
    full_next_train_t = torch.tensor(full_next_train, dtype=torch.float32).to(DEVICE)
    sparse_test_t = torch.tensor(sparse_test, dtype=torch.float32).to(DEVICE)
    full_next_test_t = torch.tensor(full_next_test, dtype=torch.float32).to(DEVICE)

    # ---- Build model ----
    print(f"\nBuilding model (latent_dim={config['latent_dim']}, "
          f"degree={config['polynomial_degree']}, E={config['ensemble_size']})...")

    model = SparseAutoencoderRNN(
        sparse_dim=config['num_sensors'],
        full_dim=full_dim,
        latent_dim=config['latent_dim'],
        ensemble_size=config['ensemble_size'],
        polynomial_degree=config['polynomial_degree'],
        encoder_hidden_dims=config['encoder_hidden_dims'],
        decoder_hidden_dims=config['decoder_hidden_dims'],
        encoder_dropout=config['encoder_dropout'],
        decoder_dropout=config['decoder_dropout'],
        dynamics_dropout=config['dynamics_dropout'],
        state_names=[f'z{i+1}' for i in range(config['latent_dim'])],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    print(f"  Polynomial library: {model.dynamics.rnn._n_library_terms} terms (autonomous, z only)")

    # ---- Train ----
    print(f"\nTraining for {config['epochs']} epochs...")
    fit_autoencoder(
        model, sparse_train_t, full_next_train_t,
        sparse_obs_test=sparse_test_t,
        full_state_next_test=full_next_test_t,
        epochs=config['epochs'],
        warmup_steps=config['warmup_steps'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        l1=config['l1'],
        pruning_threshold=config['pruning_threshold'],
        pruning_method=config['pruning_method'],
        pruning_frequency=config['pruning_frequency'],
        dt=config['dt'],
        refit_epochs=config['refit_epochs'],
        verbose=True,
    )

    # ---- Results ----
    print(f"\n{'='*60}")
    print("DISCOVERED EQUATIONS")
    print(f"{'='*60}")

    print("\nDiscrete-time equations:")
    model.print_equations()

    print(f"\nContinuous-time ODE (dt={config['dt']:.4f}):")
    print(model.get_continuous_equations(config['dt']))

    active = model.count_active_terms()
    total_active = sum(active.values())
    print(f"\nActive terms: {total_active} (per state: {active})")

    # ---- Evaluation: Prediction ----
    print(f"\n{'='*60}")
    print("EVALUATION: Next-State Prediction")
    print(f"{'='*60}")

    model.eval()
    E = model.ensemble_size
    with torch.no_grad():
        s_te = sparse_test_t.unsqueeze(0).expand(E, -1, -1, -1)
        fp_te, latent_te, _ = model(s_te)

        # Prediction MSE (in normalized space)
        fn_te = full_next_test_t.unsqueeze(0).expand(E, -1, -1, -1)
        pred_mse = torch.nn.functional.mse_loss(fp_te, fn_te).item()
        print(f"\n  Test prediction MSE (normalized): {pred_mse:.6f}")

        # Relative error
        residual = (fp_te.mean(0) - full_next_test_t).reshape(-1)
        rel_err = residual.norm() / full_next_test_t.reshape(-1).norm()
        print(f"  Test relative error: {rel_err.item():.4f} ({100*rel_err.item():.2f}%)")

    # ---- Evaluation: Autonomous Forecast ----
    print(f"\n{'='*60}")
    print("EVALUATION: Autonomous Forecast")
    print(f"{'='*60}")

    with torch.no_grad():
        # Use last latent state from prediction as initial condition
        z_init = latent_te[:, :, -1, :]  # (E, B, latent_dim) — last timestep
        n_forecast = min(30, sparse_test_t.shape[1])
        full_forecast, latent_forecast = model.forecast(z_init, n_steps=n_forecast)

        print(f"  Forecast steps: {n_forecast}")
        print(f"  Forecast latent shape: {latent_forecast.shape}")
        print(f"  Forecast full shape: {full_forecast.shape}")

    print(f"\nSINDy-SHRED reference: Pixel-space MSE=0.030")

    print(f"\n{'='*60}")
    print("Cylinder flow discovery complete.")


if __name__ == '__main__':
    main()
