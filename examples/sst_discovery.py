"""Sea Surface Temperature dynamics discovery from sparse sensors.

Discovers latent governing equations for weekly SST data from NOAA using
sparse sensor measurements and an encoder-decoder polynomial RNN.

Data: NOAA Optimum Interpolation SST V2 (1992-2019)
  - 1,400 weekly snapshots, 44,219 sea grid points
  - 250 random sensors (0.57% spatial coverage)

Architecture (teacher-forced next-state prediction):
  For each timestep:
    z_t = encoder(sparse_sensors_t)                              # R^250 -> R^3
    z_{t+1} = (1-α)*z_t + P(z_t)                               # polynomial dynamics in latent space
    full_pred_{t+1} = decoder(z_{t+1})                           # R^3 -> R^44219

  Training loss: MSE(full_pred_{t+1}, true_field_{t+1})  (predict NEXT field)
  Forecast: encoder provides z_0, then evolve z_{t+1} = (1-α)*z_t + P(z_t), decode

SINDy-SHRED reference results (Gao et al., PNAS 2026):
  Discovered 3D linear ODE with annual oscillation:
    z1_dot =  4.68*z2 - 2.37*z3
    z2_dot = -3.10*z1 + 3.25*z3
    z3_dot =  2.72*z1 - 5.55*z2
  Reconstruction MSE: 0.57, relative error: 2.01%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

from sindy_rnn import SparseAutoencoderRNN, fit_autoencoder


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_sst_data(path='data/SST_data.mat'):
    """Load SST data, filter to sea grid points."""
    load_X = loadmat(path)['Z'].T  # (1400, 64800)
    mean_X = np.mean(load_X, axis=0)
    sst_locs = np.where(mean_X != 0)[0]
    return load_X[:, sst_locs], sst_locs  # (1400, 44219)


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
    print("SST Dynamics Discovery with Encoder-Decoder Polynomial RNN")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ---- Configuration ----
    config = {
        'seed': 0,
        'num_sensors': 250,
        'latent_dim': 3,
        'polynomial_degree': 1,      # linear (Koopman-type)
        'ensemble_size': 11,
        'window_size': 52,            # 1 year of weekly data
        'train_length': 1000,
        'validate_length': 30,
        'dt': 1 / 52,                 # weekly timestep
        'encoder_hidden_dims': [350, 400],
        'decoder_hidden_dims': [400, 350],
        'encoder_dropout': 0.1,
        'decoder_dropout': 0.1,
        'dynamics_dropout': 0.,
        'epochs': 500,
        'warmup_steps': 200,
        'learning_rate': 1e-3,
        'l1': 1e-3,
        'pruning_threshold': 0.05,
        'pruning_method': 'median',
        'pruning_frequency': 20,
        'refit_epochs': 100,
    }

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # ---- Load and preprocess data ----
    print("\nLoading SST data...")
    X, sst_locs = load_sst_data()
    n_time, full_dim = X.shape
    print(f"  Shape: ({n_time}, {full_dim})")

    # Select random sensor locations
    sensor_locs = np.random.choice(full_dim, size=config['num_sensors'], replace=False)
    print(f"  Sensors: {config['num_sensors']} / {full_dim} ({100*config['num_sensors']/full_dim:.2f}%)")

    # MinMax normalize (fit on training split only)
    lags = config['window_size']
    train_end = config['train_length'] + lags
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
        z_init = latent_te[:, :, -1, :]  # (E, B, latent_dim)
        n_forecast = min(52, sparse_test_t.shape[1])  # up to 1 year
        full_forecast, latent_forecast = model.forecast(z_init, n_steps=n_forecast)

        print(f"  Forecast steps: {n_forecast} ({n_forecast/52:.1f} years)")
        print(f"  Forecast latent shape: {latent_forecast.shape}")
        print(f"  Forecast full shape: {full_forecast.shape}")

    print(f"\nSINDy-SHRED reference: MSE=0.57, relative error=2.01%")

    print(f"\n{'='*60}")
    print("SST discovery complete.")


if __name__ == '__main__':
    main()
