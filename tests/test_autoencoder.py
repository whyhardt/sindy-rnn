"""Tests for the encoder-decoder autoencoder architecture."""

import torch
import pytest

from sindy_rnn import SparseAutoencoderRNN, GRUEncoder, fit_autoencoder, ensemble_prune


class TestMLPEncoderDecoder:
    """Test encoder and decoder shape correctness."""

    def test_encoder_shapes(self):
        model = SparseAutoencoderRNN(sparse_dim=10, full_dim=50, latent_dim=3)
        x = torch.randn(4, 20, 10)  # (B, T, sparse_dim)
        z = model.encoder(x)
        assert z.shape == (4, 20, 3)

    def test_decoder_shapes(self):
        model = SparseAutoencoderRNN(sparse_dim=10, full_dim=50, latent_dim=3)
        z = torch.randn(4, 20, 3)  # (B, T, latent_dim)
        x_hat = model.decoder(z)
        assert x_hat.shape == (4, 20, 50)

    def test_decoder_handles_ensemble_dim(self):
        model = SparseAutoencoderRNN(sparse_dim=10, full_dim=50, latent_dim=3,
                                     ensemble_size=5)
        z = torch.randn(5, 4, 20, 3)  # (E, B, T, latent_dim)
        x_hat = model.decoder(z)
        assert x_hat.shape == (5, 4, 20, 50)

    def test_custom_hidden_dims(self):
        model = SparseAutoencoderRNN(
            sparse_dim=10, full_dim=50, latent_dim=3,
            encoder_hidden_dims=[64, 32],
            decoder_hidden_dims=[32, 64],
        )
        x = torch.randn(2, 10, 10)
        z = model.encoder(x)
        assert z.shape == (2, 10, 3)
        x_hat = model.decoder(z)
        assert x_hat.shape == (2, 10, 50)

    def test_single_linear_encoder(self):
        """hidden_dims=[] gives a single linear layer (W @ x + b)."""
        model = SparseAutoencoderRNN(
            sparse_dim=10, full_dim=50, latent_dim=3,
            encoder_hidden_dims=[],
        )
        x = torch.randn(2, 10, 10)
        z = model.encoder(x)
        assert z.shape == (2, 10, 3)
        # Only one Linear layer in encoder
        linears = [m for m in model.encoder.net if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 1


class TestSparseAutoencoderRNN:
    """Test the full autoencoder-dynamics wrapper."""

    def test_forward_shapes_single_ensemble(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=1, polynomial_degree=2,
        )
        sparse_obs = torch.randn(8, 50, 5)
        full_pred, latent_pred, projected = model(sparse_obs)
        assert full_pred.shape == (1, 8, 50, 20)
        assert latent_pred.shape == (1, 8, 50, 3)
        assert projected.shape == (8, 50, 3)

    def test_forward_shapes_ensemble(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=4, polynomial_degree=2,
        )
        sparse_obs = torch.randn(8, 50, 5)
        full_pred, latent_pred, projected = model(sparse_obs)
        assert full_pred.shape == (4, 8, 50, 20)
        assert latent_pred.shape == (4, 8, 50, 3)
        assert projected.shape == (8, 50, 3)

    def test_forward_with_external_controls(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            n_controls=2, ensemble_size=3, polynomial_degree=2,
        )
        sparse_obs = torch.randn(4, 30, 5)
        controls = torch.randn(4, 30, 2)
        full_pred, latent_pred, projected = model(sparse_obs, controls=controls)
        assert full_pred.shape == (3, 4, 30, 20)
        assert latent_pred.shape == (3, 4, 30, 3)

        # Dynamics should have n_controls = n_external only (not latent_dim + n_external)
        assert model.dynamics.rnn.n_controls == 2

    def test_forward_with_4d_input(self):
        """When sparse_obs is already (E, B, T, sparse_dim)."""
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=3, polynomial_degree=2,
        )
        sparse_obs = torch.randn(3, 4, 20, 5)  # (E, B, T, sparse_dim)
        full_pred, latent_pred, projected = model(sparse_obs)
        assert full_pred.shape == (3, 4, 20, 20)
        assert latent_pred.shape == (3, 4, 20, 3)
        assert projected.shape == (3, 4, 20, 3)

    def test_properties(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=7,
        )
        assert model.ensemble_size == 7
        assert model.n_states == 3
        assert model.sparse_dim == 5
        assert model.full_dim == 20
        assert model.latent_dim == 3

    def test_dynamics_autonomous(self):
        """The inner dynamics should have n_controls=0 (no encoder projection controls)."""
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=1, polynomial_degree=2,
        )
        # n_controls = 0 (encoder output is the observed state, not a control)
        assert model.dynamics.rnn.n_controls == 0

        # Library features: only [z_0, z_1, z_2], no p_i terms
        terms = model.dynamics.library_terms
        for t in terms:
            assert 'p_' not in t, f"Found projection term '{t}' in library"

    def test_equations_autonomous(self):
        """Equations should reference only z terms, no projection variables."""
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=1, polynomial_degree=1,
            state_names=['z1', 'z2', 'z3'],
        )
        terms = model.dynamics.library_terms
        # Should only have: ['1', 'z1', 'z2', 'z3']
        assert terms == ['1', 'z1', 'z2', 'z3']

    def test_recurrent_hidden_state(self):
        """Forward should produce different outputs than encoder alone (recurrent evolution)."""
        torch.manual_seed(42)
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=1, polynomial_degree=2,
        )
        model.eval()
        sparse_obs = torch.randn(2, 10, 5)

        with torch.no_grad():
            _, latent_pred, projected = model(sparse_obs)

        # Latent predictions should differ from encoder projections
        # (they incorporate recurrent polynomial dynamics)
        assert latent_pred.shape == (1, 2, 10, 3)
        assert not torch.allclose(latent_pred[0], projected, atol=1e-3)


class TestForecast:
    """Test autonomous forecasting."""

    def test_forecast_shapes(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=4, polynomial_degree=2,
        )
        model.eval()
        z_init = torch.randn(4, 2, 3)  # (E, B, latent_dim)
        with torch.no_grad():
            full_traj, latent_traj = model.forecast(z_init, n_steps=10)
        assert full_traj.shape == (4, 2, 10, 20)
        assert latent_traj.shape == (4, 2, 10, 3)

    def test_forecast_with_controls(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            n_controls=2, ensemble_size=3, polynomial_degree=2,
        )
        model.eval()
        z_init = torch.randn(3, 2, 3)
        controls = torch.randn(3, 2, 10, 2)
        with torch.no_grad():
            full_traj, latent_traj = model.forecast(z_init, n_steps=10, controls=controls)
        assert full_traj.shape == (3, 2, 10, 20)
        assert latent_traj.shape == (3, 2, 10, 3)

    def test_forecast_is_autonomous(self):
        """Forecast should NOT use encoder — pure polynomial dynamics."""
        torch.manual_seed(42)
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=1, polynomial_degree=2,
        )
        model.eval()
        z_init = torch.randn(1, 2, 3)

        with torch.no_grad():
            _, latent_traj = model.forecast(z_init, n_steps=5)

        # Verify by manually running the polynomial dynamics
        theta = model.dynamics.rnn.unfold_polynomial_coefficients()
        h = z_init
        manual_traj = []
        for t in range(5):
            h = model.dynamics.rnn.forward_polynomial(
                h, None, mask=model.dynamics.coefficient_masks, theta=theta
            )
            manual_traj.append(h)
        manual_latent = torch.stack(manual_traj, dim=2)

        torch.testing.assert_close(latent_traj, manual_latent, rtol=1e-5, atol=1e-6)


class TestPruningCompatibility:
    """Test that pruning works on the inner dynamics model."""

    def test_pruning_works_on_inner_dynamics(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=5, polynomial_degree=2,
        )
        assert model.dynamics.coefficient_masks.all()

        with torch.no_grad():
            ensemble_prune(model.dynamics, alpha=0.05, delta=0.5)

        assert model.dynamics.pruning_patience.shape == model.dynamics.coefficient_masks.shape

    def test_mask_shapes_correct(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=4, polynomial_degree=2,
        )
        E = 4
        n_states = 3
        n_terms = model.dynamics.rnn._n_library_terms
        assert model.dynamics.coefficient_masks.shape == (E, n_states, n_terms)
        assert model.dynamics.pruning_patience.shape == (E, n_states, n_terms)

        # With latent_dim=3, n_controls=0, n_features=3, degree=2:
        # n_terms = C(3+2, 2) = C(5,2) = 10
        assert n_terms == 10


class TestEquationExtraction:
    """Test equation extraction via delegation."""

    def test_get_equations(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=1, polynomial_degree=2,
            state_names=['z1', 'z2', 'z3'],
        )
        eqs = model.get_equations()
        # With Euler parameterization, equations are in ODE form
        assert 'dz1/dt' in eqs
        assert 'dz2/dt' in eqs
        assert 'dz3/dt' in eqs

    def test_get_continuous_equations(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=1, polynomial_degree=2,
            state_names=['z1', 'z2', 'z3'],
        )
        eqs = model.get_continuous_equations()
        assert 'dz1/dt' in eqs
        assert 'dz2/dt' in eqs
        assert 'dz3/dt' in eqs

    def test_count_active_terms(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=1, polynomial_degree=2,
            state_names=['z1', 'z2', 'z3'],
        )
        active = model.count_active_terms()
        assert set(active.keys()) == {'z1', 'z2', 'z3'}
        # All terms active initially
        n_terms = model.dynamics.rnn._n_library_terms
        for v in active.values():
            assert v == n_terms

    def test_get_coefficients(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=1, polynomial_degree=2,
            state_names=['z1', 'z2', 'z3'],
        )
        coefs = model.get_coefficients(aggregate=True)
        assert set(coefs.keys()) == {'z1', 'z2', 'z3'}
        n_terms = model.dynamics.rnn._n_library_terms
        for v in coefs.values():
            assert v.shape == (n_terms,)


class TestInnerDynamicsInvariant:
    """The forward == forward_polynomial invariant must still hold."""

    @pytest.mark.parametrize("decomposed", [True, False])
    def test_invariant_preserved_no_controls(self, decomposed):
        """Invariant with autonomous dynamics (no controls)."""
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=3, polynomial_degree=2,
            decomposed=decomposed,
        )
        rnn = model.dynamics.rnn
        E, B = 3, 4
        h = torch.randn(E, B, 3)

        rnn.eval()
        h_standard = rnn._forward_impl(h, None)
        mask = torch.ones(E, 3, rnn._n_library_terms, dtype=torch.bool)
        h_poly = rnn.forward_polynomial(h, None, mask=mask)

        torch.testing.assert_close(h_standard, h_poly, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("decomposed", [True, False])
    def test_invariant_preserved_with_external_controls(self, decomposed):
        """Invariant with external controls."""
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            n_controls=2, ensemble_size=3, polynomial_degree=2,
            decomposed=decomposed,
        )
        rnn = model.dynamics.rnn
        E, B = 3, 4
        h = torch.randn(E, B, 3)
        u = torch.randn(E, B, 2)  # external controls only

        rnn.eval()
        h_standard = rnn._forward_impl(h, u)
        mask = torch.ones(E, 3, rnn._n_library_terms, dtype=torch.bool)
        h_poly = rnn.forward_polynomial(h, u, mask=mask)

        torch.testing.assert_close(h_standard, h_poly, rtol=1e-4, atol=1e-5)


class TestFitAutoencoder:
    """Test end-to-end training."""

    def test_fit_runs_without_error(self):
        torch.manual_seed(42)
        B, T = 5, 20
        sparse_dim, full_dim, latent_dim = 4, 15, 3

        sparse_obs = torch.randn(B, T, sparse_dim)
        full_state = torch.randn(B, T, full_dim)

        model = SparseAutoencoderRNN(
            sparse_dim=sparse_dim, full_dim=full_dim, latent_dim=latent_dim,
            ensemble_size=3, polynomial_degree=2,
        )
        fit_autoencoder(model, sparse_obs, full_state,
                        epochs=10, verbose=False)

    def test_fit_loss_decreases(self):
        torch.manual_seed(42)
        B, T = 10, 30
        sparse_dim, full_dim, latent_dim = 4, 15, 3

        # Simple linear relationship for easy fitting
        W = torch.randn(sparse_dim, full_dim) * 0.1
        full_state = torch.randn(B, T, full_dim)
        sparse_obs = full_state @ W.T  # (B, T, sparse_dim) - rough projection

        model = SparseAutoencoderRNN(
            sparse_dim=sparse_dim, full_dim=full_dim, latent_dim=latent_dim,
            ensemble_size=1, polynomial_degree=1,
        )

        # Record initial loss
        model.eval()
        with torch.no_grad():
            fp, _, _ = model(sparse_obs)
            loss_init = torch.nn.functional.mse_loss(fp, full_state.unsqueeze(0)).item()

        # Train
        fit_autoencoder(model, sparse_obs, full_state,
                        epochs=100, learning_rate=1e-3, l1=0, verbose=False)

        # Record final loss
        model.eval()
        with torch.no_grad():
            fp, _, _ = model(sparse_obs)
            loss_final = torch.nn.functional.mse_loss(fp, full_state.unsqueeze(0)).item()

        assert loss_final < loss_init, f"Loss did not decrease: {loss_init:.6f} -> {loss_final:.6f}"

    def test_fit_with_external_controls(self):
        torch.manual_seed(42)
        B, T = 5, 20

        sparse_obs = torch.randn(B, T, 4)
        full_state = torch.randn(B, T, 15)
        controls = torch.randn(B, T, 2)

        model = SparseAutoencoderRNN(
            sparse_dim=4, full_dim=15, latent_dim=3,
            n_controls=2, ensemble_size=2, polynomial_degree=2,
        )
        fit_autoencoder(model, sparse_obs, full_state,
                        controls=controls, epochs=5, verbose=False)

    def test_fit_with_test_data(self):
        torch.manual_seed(42)
        B, T = 5, 20

        sparse_obs = torch.randn(B, T, 4)
        full_state_next = torch.randn(B, T, 15)

        model = SparseAutoencoderRNN(
            sparse_dim=4, full_dim=15, latent_dim=3,
            ensemble_size=2, polynomial_degree=2,
        )
        fit_autoencoder(model, sparse_obs, full_state_next,
                        sparse_obs_test=sparse_obs[:2],
                        full_state_next_test=full_state_next[:2],
                        epochs=5, verbose=False)

    def test_fit_with_refit(self):
        torch.manual_seed(42)
        B, T = 5, 20

        sparse_obs = torch.randn(B, T, 4)
        full_state = torch.randn(B, T, 15)

        model = SparseAutoencoderRNN(
            sparse_dim=4, full_dim=15, latent_dim=3,
            ensemble_size=2, polynomial_degree=2,
        )
        fit_autoencoder(model, sparse_obs, full_state,
                        epochs=10, refit_epochs=5, verbose=False)


class TestSaveLoad:
    """Test save/load functionality."""

    def test_save_load_roundtrip(self, tmp_path):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=2, polynomial_degree=2,
            state_names=['a', 'b', 'c'],
        )
        path = str(tmp_path / "model.pt")
        model.save(path)

        loaded = SparseAutoencoderRNN.load(path)
        assert loaded.sparse_dim == 5
        assert loaded.full_dim == 20
        assert loaded.latent_dim == 3
        assert loaded.ensemble_size == 2
        assert loaded.dynamics.state_names == ['a', 'b', 'c']

        # Check weights match
        x = torch.randn(2, 10, 5)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            out1, _, _ = model(x)
            out2, _, _ = loaded(x)
        torch.testing.assert_close(out1, out2)

    def test_save_load_with_external_controls(self, tmp_path):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            n_controls=2, ensemble_size=2, polynomial_degree=2,
        )
        path = str(tmp_path / "model.pt")
        model.save(path)

        loaded = SparseAutoencoderRNN.load(path)
        assert loaded._n_external_controls == 2
        assert loaded.dynamics.rnn.n_controls == 2  # external only

        # Check forward works
        x = torch.randn(2, 10, 5)
        u = torch.randn(2, 10, 2)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            out1, _, _ = model(x, controls=u)
            out2, _, _ = loaded(x, controls=u)
        torch.testing.assert_close(out1, out2)


class TestGRUEncoder:
    """Test the GRU encoder."""

    def test_gru_encoder_shapes(self):
        enc = GRUEncoder(input_dim=10, latent_dim=4, num_layers=2)
        x = torch.randn(3, 20, 10)  # (B, T, input_dim)
        z = enc(x)
        assert z.shape == (3, 20, 4)

    def test_gru_encoder_with_projection(self):
        """When hidden_dim != latent_dim, a projection layer is added."""
        enc = GRUEncoder(input_dim=10, latent_dim=4, hidden_dim=64, num_layers=2)
        assert enc.projection is not None
        x = torch.randn(3, 20, 10)
        z = enc(x)
        assert z.shape == (3, 20, 4)

    def test_gru_encoder_no_projection(self):
        """When hidden_dim == latent_dim (default), no projection layer."""
        enc = GRUEncoder(input_dim=10, latent_dim=4)
        assert enc.projection is None
        assert enc.hidden_dim == 4

    def test_gru_encoder_handles_ensemble_dim(self):
        """GRU should handle (E, B, T, input_dim) inputs."""
        enc = GRUEncoder(input_dim=10, latent_dim=4, num_layers=2)
        x = torch.randn(5, 3, 20, 10)  # (E, B, T, input_dim)
        z = enc(x)
        assert z.shape == (5, 3, 20, 4)

    def test_gru_temporal_context(self):
        """GRU output at time t should depend on inputs at t' < t (unlike MLP)."""
        torch.manual_seed(42)
        enc = GRUEncoder(input_dim=10, latent_dim=4, num_layers=2)
        enc.eval()

        x = torch.randn(1, 20, 10)
        with torch.no_grad():
            z_full = enc(x)  # (1, 20, 4)

        # Modify an early timestep
        x_mod = x.clone()
        x_mod[:, 5, :] += 10.0
        with torch.no_grad():
            z_mod = enc(x_mod)

        # Output at t=5 should change
        assert not torch.allclose(z_full[:, 5], z_mod[:, 5], atol=1e-3)
        # Output at t>5 should also change (temporal context propagates)
        assert not torch.allclose(z_full[:, 10], z_mod[:, 10], atol=1e-3)
        # Output at t<5 should NOT change (GRU is causal)
        torch.testing.assert_close(z_full[:, :5], z_mod[:, :5])


class TestSparseAutoencoderRNNWithGRU:
    """Test the autoencoder with GRU encoder."""

    def test_forward_shapes_gru(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=4, polynomial_degree=2,
            encoder_type='gru',
        )
        sparse_obs = torch.randn(8, 50, 5)
        full_pred, latent_pred, encoded = model(sparse_obs)
        assert full_pred.shape == (4, 8, 50, 20)
        assert latent_pred.shape == (4, 8, 50, 3)
        assert encoded.shape == (8, 50, 3)

    def test_forward_shapes_gru_with_hidden_dim(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=2, polynomial_degree=2,
            encoder_type='gru', encoder_gru_hidden_dim=64,
        )
        sparse_obs = torch.randn(4, 30, 5)
        full_pred, latent_pred, encoded = model(sparse_obs)
        assert full_pred.shape == (2, 4, 30, 20)
        assert latent_pred.shape == (2, 4, 30, 3)
        assert encoded.shape == (4, 30, 3)

    def test_gru_dynamics_still_autonomous(self):
        """GRU encoder shouldn't add control terms to the polynomial."""
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=1, polynomial_degree=2,
            encoder_type='gru',
        )
        assert model.dynamics.rnn.n_controls == 0
        terms = model.dynamics.library_terms
        for t in terms:
            assert 'p_' not in t

    def test_forecast_with_gru_model(self):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=3, polynomial_degree=2,
            encoder_type='gru',
        )
        model.eval()
        z_init = torch.randn(3, 2, 3)
        with torch.no_grad():
            full_traj, latent_traj = model.forecast(z_init, n_steps=10)
        assert full_traj.shape == (3, 2, 10, 20)
        assert latent_traj.shape == (3, 2, 10, 3)

    def test_encoder_type_stored(self):
        model_mlp = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3, encoder_type='mlp')
        model_gru = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3, encoder_type='gru')
        assert model_mlp.encoder_type == 'mlp'
        assert model_gru.encoder_type == 'gru'
        assert isinstance(model_gru.encoder, GRUEncoder)

    def test_fit_with_gru_encoder(self):
        torch.manual_seed(42)
        B, T = 5, 20
        sparse_dim, full_dim, latent_dim = 4, 15, 3

        sparse_obs = torch.randn(B, T, sparse_dim)
        full_state = torch.randn(B, T, full_dim)

        model = SparseAutoencoderRNN(
            sparse_dim=sparse_dim, full_dim=full_dim, latent_dim=latent_dim,
            ensemble_size=2, polynomial_degree=2,
            encoder_type='gru',
        )
        fit_autoencoder(model, sparse_obs, full_state,
                        epochs=10, verbose=False)

    def test_save_load_gru(self, tmp_path):
        model = SparseAutoencoderRNN(
            sparse_dim=5, full_dim=20, latent_dim=3,
            ensemble_size=2, polynomial_degree=2,
            encoder_type='gru', encoder_gru_hidden_dim=32,
            encoder_num_layers=3,
        )
        path = str(tmp_path / "model_gru.pt")
        model.save(path)

        loaded = SparseAutoencoderRNN.load(path)
        assert loaded.encoder_type == 'gru'
        assert isinstance(loaded.encoder, GRUEncoder)
        assert loaded.encoder.hidden_dim == 32
        assert loaded.encoder.num_layers == 3

        # Check weights match
        x = torch.randn(2, 10, 5)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            out1, _, _ = model(x)
            out2, _, _ = loaded(x)
        torch.testing.assert_close(out1, out2)
