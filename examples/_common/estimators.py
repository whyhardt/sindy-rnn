"""Thin, uniform wrappers around each method: fit(xs, ys), predict(xs),
simulate(x0, n_steps), save(path)/load(path). Whatever the underlying model
naturally produces (a derivative, a next-state, a decoded full state) is
just returned as-is — the wrapper doesn't care which.
"""
import inspect
import math
import os

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sindy_rnn import PolynomialRNN, RolloutSINDyRNN, fit as fit_polynomial_rnn, fit_rollout

_FIT_ROLLOUT_PARAMS = set(inspect.signature(fit_rollout).parameters)
_ROLLOUT_MODEL_PARAMS = set(inspect.signature(RolloutSINDyRNN.__init__).parameters) - {'self'}


def _to_tensor(x, device='cpu'):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=device)


def resolve_member(est):
    """Member index for get_equations()/count_active_terms(), consistent
    with est.simulate_mode: 'mean' -> None (report the ensemble aggregate),
    'best' -> est.model.best_member_idx, 'bic' -> est.model.bic_member_idx.
    Works for both PolynomialRNNEstimator and RolloutSINDyRNNEstimator,
    whose .model exposes the same two index buffers/properties."""
    if est.simulate_mode == 'best':
        return est.model.best_member_idx.item()
    if est.simulate_mode == 'bic':
        return est.model.bic_member_idx.item()
    return None


class PolynomialRNNEstimator:
    """Wraps PolynomialRNN + fit() (derivative matching, full-state).

    `simulate='mean'` (default) reduces the ensemble dimension with a plain
    mean, as before. `simulate='best'` instead indexes the single ensemble
    member with the best held-out (or, absent that, training) fit
    (model.best_member_idx, set by PolynomialRNN.select_best_member() at
    the end of fit() — see sindy_rnn/training.py). `simulate='bic'` instead
    indexes the member with the best BIC on training data
    (model.bic_member_idx, set by select_best_member_bic()) — same
    training-data fit as 'best' falls back to, but penalized by term count,
    so it also rewards sparsity rather than fit alone. All three are
    determined at training time regardless of which mode is requested at
    predict()/simulate() time, so switching between them doesn't require
    retraining.
    """

    def __init__(self, device='cpu', **kwargs):
        self.device = device
        self.simulate_mode = kwargs.pop('simulate', 'mean')
        self.model_kwargs = kwargs.pop('model_kwargs', {})
        self.fit_kwargs = kwargs.pop('fit_kwargs', kwargs)
        self.model = None

    def _reduce_ensemble(self, tensor):
        """tensor: (E, ...) -> (...) via mean or a stored member index."""
        if self.simulate_mode == 'best':
            return tensor[self.model.best_member_idx]
        if self.simulate_mode == 'bic':
            return tensor[self.model.bic_member_idx]
        return tensor.mean(0)

    def fit(self, xs, ys):
        xs, ys = _to_tensor(xs), _to_tensor(ys)
        self.model = PolynomialRNN(**self.model_kwargs).to(self.device)
        fit_polynomial_rnn(self.model, xs.to(self.device), ys.to(self.device), **self.fit_kwargs)
        return self

    def predict(self, xs):
        xs = _to_tensor(xs, self.device)
        self.model.eval()
        with torch.no_grad():
            y_hat, _ = self.model(xs)  # (E, B, T, n_states)
        return self._reduce_ensemble(y_hat).cpu().numpy()

    def simulate(self, x0, n_steps):
        """Autonomous rollout from x0 (B, n_states) for n_steps."""
        x0 = _to_tensor(x0, self.device)
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0).expand(self.model.ensemble_size, -1, -1)
        self.model.eval()
        with torch.no_grad():
            theta = self.model.rnn.unfold_polynomial_coefficients()
            theta = theta * self.model.coefficient_masks.float()
            h = x0
            traj = [h]
            for _ in range(n_steps):
                h = self.model.rnn.forward_polynomial(
                    h, None, mask=self.model.coefficient_masks, theta=theta,
                    integrator='rk4')
                traj.append(h)
            traj = torch.stack(traj, dim=2)  # (E, B, n_steps+1, n_states)
        return self._reduce_ensemble(traj)[:, 1:].cpu().numpy()

    def get_equations(self):
        """Equations for whichever member/aggregate predict()/simulate()
        resolve to: the selected member's own equations for 'best'/'bic',
        or the ensemble median for 'mean' (a single outlier member
        shouldn't be able to skew a displayed coefficient the way it can
        skew a plain mean)."""
        member = resolve_member(self)
        if member is not None:
            return self.model.get_equations(member=member)
        return self.model.get_equations(aggregate='median')

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, path, device='cpu', simulate='mean'):
        est = cls(device=device, simulate=simulate)
        est.model = PolynomialRNN.load(path).to(device)
        return est


class RolloutSINDyRNNEstimator:
    """Wraps RolloutSINDyRNN + fit_rollout() (sparse-sensor or identity).

    Both RolloutSINDyRNN's constructor kwargs (n_sensors, n_latent, ...,
    dec_l1, ...) and fit_rollout()'s hyperparameters (T_w, T_max, ...) are
    passed directly here, flat — no model_kwargs=/fit_kwargs= nesting. Each
    name is routed to whichever of the two signatures it matches; anything
    matching neither (e.g. a data-loading-only config key like `path` or
    `sensor_seed`) is silently ignored, so a whole config section can be
    forwarded with `**dcfg, **rcfg` without hand-picking which keys apply.

    This means a typo'd or renamed kwarg fails silently (dropped, not
    raised) rather than erroring — the tradeoff for not needing to touch
    the call site every time a config key is added or renamed.

    `rollout_noise` is the one name both fit_rollout() and the constructor
    accept; it routes to fit_rollout() here, since fit_rollout()
    unconditionally overwrites model.rollout_noise at fit time regardless
    of what the constructor was given — the constructor's own default would
    otherwise silently take effect only if you never call fit().

    `simulate='mean'` (default) reduces the dynamics ensemble with a plain
    mean during autonomous rollout, as before. `simulate='best'` instead
    uses the single ensemble member with the best training-data
    reconstruction fit (model._eq_dynamics.best_member_idx, set by
    rollout.select_best_member() at the end of fit_rollout() — see
    sindy_rnn/rollout.py). predict()'s same-timestep reconstruction doesn't
    depend on this — encode() broadcasts one shared z_0 identically across
    every ensemble member, so mean and best agree there; only simulate()'s
    multi-step autonomous rollout actually diverges across members.
    """

    def __init__(self, device='cpu', **kwargs):
        self.device = device
        self.simulate_mode = kwargs.pop('simulate', 'mean')
        self.fit_kwargs = {k: v for k, v in kwargs.items() if k in _FIT_ROLLOUT_PARAMS}
        self.model_kwargs = {k: v for k, v in kwargs.items()
                             if k in _ROLLOUT_MODEL_PARAMS and k not in _FIT_ROLLOUT_PARAMS}
        self.T_w = self.fit_kwargs.get('T_w', 1)
        self.model = None

    def fit(self, xs, ys):
        """xs: (N_time, n_sensors) sparse observations. ys: (N_time, n_full)."""
        xs, ys = _to_tensor(xs), _to_tensor(ys)
        self.model = RolloutSINDyRNN(**self.model_kwargs).to(self.device)
        fit_rollout(self.model, xs.to(self.device), ys.to(self.device), **self.fit_kwargs)
        return self

    def predict(self, xs):
        """Same-timestep reconstruction at every valid frame.

        xs: (N_time, n_sensors). Returns (N_time, n_full), NaN before T_w-1.
        """
        xs = _to_tensor(xs, self.device)
        N, n_full = xs.shape[0], self.model.n_full
        out = np.full((N, n_full), np.nan, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for t in range(self.T_w - 1, N):
                window = xs[t - self.T_w + 1:t + 1].unsqueeze(0)
                z_0 = self.model.encode(window)
                out[t] = self.model.decoder(z_0.mean(0))[0].cpu().numpy()
        return out

    def simulate(self, x0, n_steps):
        """x0: (T_w, n_sensors) warmup window. Returns (n_steps, n_full)."""
        x0 = _to_tensor(x0, self.device).unsqueeze(0)
        self.model.eval()
        x_hat, _ = self.model.forecast(x0, n_steps, reduction=self.simulate_mode)
        return x_hat[0].cpu().numpy()

    def get_equations(self):
        """Equations for whichever member/aggregate predict()/simulate()
        resolve to: the selected member's own equations for 'best'/'bic',
        or the ensemble median for 'mean' (a single outlier member
        shouldn't be able to skew a displayed coefficient the way it can
        skew a plain mean)."""
        member = resolve_member(self)
        if member is not None:
            return self.model.get_equations(member=member)
        return self.model.get_equations(aggregate='median')

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, path, device='cpu', T_w=1, simulate='mean'):
        est = cls(device=device, fit_kwargs=dict(T_w=T_w), simulate=simulate)
        est.model = RolloutSINDyRNN.load(path).to(device)
        return est


class SindyShredEstimator:
    """Wraps the external SINDySHRED reference implementation.

    fit() delegates to SINDySHRED.fit(); ys is ignored (SINDySHRED derives
    its own reconstruction targets from the single full-state array it's
    given, same as the real SINDySHRED API).
    """

    def __init__(self, sensor_locations, dt, lags, train_length, validate_length,
                test_length=None, seed=0, device='cpu', **shred_kwargs):
        self.sensor_locations = sensor_locations
        self.dt = dt
        self.lags = lags
        self.train_length = train_length
        self.validate_length = validate_length
        self.test_length = test_length
        self.seed = seed
        self.device = device
        self.shred_kwargs = shred_kwargs
        self.net = None          # SINDy_SHRED_net (raw nn.Module, stateless at inference)
        self.sindy_model = None  # fitted pysindy SINDy object on the latent trajectory

    def fit(self, xs, ys=None):
        """xs: (N_time, n_full) full-state array."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'sindy-shred'))
        np.math = math
        from sindy_shred import SINDySHRED

        shred = SINDySHRED(device=self.device, **self.shred_kwargs)
        shred.fit(
            num_sensors=len(self.sensor_locations), dt=self.dt, x_to_fit=np.asarray(xs),
            lags=self.lags, train_length=self.train_length,
            validate_length=self.validate_length, test_length=self.test_length,
            sensor_locations=self.sensor_locations, seed=self.seed,
        )
        try:
            shred.auto_tune_threshold(metric='bic', verbose=False)
        except Exception as e:
            print(f"  Post-hoc SINDy failed: {e}")
        self.net = shred._shred
        self.sindy_model = shred._model
        return self

    def predict(self, xs):
        """Same-timestep reconstruction via the raw network (bypasses
        SINDySHRED's train/val/test-locked sensor_recon()).

        xs: (N_time, n_full). Returns (N_time, n_full), NaN before lags-1.
        """
        self.net.eval()
        xs = np.asarray(xs)
        N, n_full = xs.shape
        sparse = xs[:, self.sensor_locations]
        out = np.full((N, n_full), np.nan, dtype=np.float32)
        device = next(self.net.parameters()).device
        with torch.no_grad():
            for t in range(self.lags - 1, N):
                window = torch.tensor(
                    sparse[t - self.lags + 1:t + 1], dtype=torch.float32,
                    device=device).unsqueeze(0)
                out[t] = self.net(window)[0].cpu().numpy()
        return out

    def simulate(self, x0, n_steps):
        """Autonomous forecast: encode x0's sensor view to a latent state,
        integrate self.sindy_model forward, decode via the net's own
        decoder layers. Doesn't depend on a live SINDySHRED wrapper object
        (only self.net + self.sindy_model, both of which survive save/load).

        x0: (lags, n_full) warmup window. Returns (n_steps, n_full).
        """
        n_full = self.net.linear3.out_features
        if self.sindy_model is None:
            print("  Simulate skipped: no post-hoc SINDy model")
            return np.full((n_steps, n_full), np.nan, dtype=np.float32)

        self.net.eval()
        device = next(self.net.parameters()).device
        sparse = np.asarray(x0)[:, self.sensor_locations]
        window = torch.tensor(sparse, dtype=torch.float32, device=device).unsqueeze(0)
        h_0 = torch.zeros((self.net.hidden_layers, 1, self.net.hidden_size), device=device)
        with torch.no_grad():
            _, h_out = self.net.gru(window, h_0)
        z0 = h_out[-1, 0].cpu().numpy()  # (hidden_size,)

        t = np.arange(n_steps + 1) * self.dt
        try:
            z_traj = self.sindy_model.simulate(z0, t)  # includes z0 as first row
        except Exception as e:
            print(f"  SINDy-SHRED simulate failed: {e}")
            return np.full((n_steps, n_full), np.nan, dtype=np.float32)

        with torch.no_grad():
            zt = torch.tensor(z_traj[1:], dtype=torch.float32, device=device)  # drop z0
            out = torch.relu(self.net.linear1(zt))
            out = torch.relu(self.net.linear2(out))
            out = self.net.linear3(out)
        out = out.cpu().numpy()

        if len(out) < n_steps:
            pad = np.full((n_steps - len(out), n_full), np.nan, dtype=np.float32)
            out = np.concatenate([out, pad], axis=0)
        return out[:n_steps]

    def get_equations(self):
        """Format the post-hoc SINDy model as an ODE string, same layout as
        the other three estimators' get_equations(). No member/aggregate
        choice here — self.sindy_model is already the single model
        SINDySHRED's own E_SINDy (see sindy_shred_net.py) resolved to."""
        if self.sindy_model is None:
            return '(no post-hoc SINDy model)'
        names = self.sindy_model.feature_names
        eqns = self.sindy_model.equations()
        return '\n'.join(f"({name})' = {eqn}" for name, eqn in zip(names, eqns))

    def save(self, path):
        """Saves the raw network weights + post-hoc SINDy model.

        SINDySHRED itself has no save/load — its train/val/test split lives
        on the live object. This drops down one level to the raw nn.Module
        (stateless at inference) and the raw pysindy model (picklable),
        which together are enough to reconstruct predict()/simulate()
        without the live SINDySHRED wrapper.
        """
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'net_init_args': dict(
                input_size=len(self.sensor_locations),
                output_size=self.net.linear3.out_features,
                hidden_size=self.net.hidden_size, hidden_layers=self.net.hidden_layers,
                l1=self.net.linear1.out_features, l2=self.net.linear2.out_features,
                dt=self.net.dt,
                library_dim=(self.sindy_model.n_output_features_
                            if self.sindy_model is not None else 1),
                poly_order=self.shred_kwargs.get('poly_order'),
            ),
            'sindy_model': self.sindy_model,
            'sensor_locations': self.sensor_locations,
            'lags': self.lags,
            'dt': self.dt,
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'sindy-shred'))
        import sindy_shred_net

        checkpoint = torch.load(path, weights_only=False)
        est = cls(sensor_locations=checkpoint['sensor_locations'], dt=checkpoint['dt'],
                  lags=checkpoint['lags'], train_length=0, validate_length=0, device=device)

        net = sindy_shred_net.SINDy_SHRED_net(device=device, **checkpoint['net_init_args'])
        net.load_state_dict(checkpoint['net_state_dict'])
        net.to(device)

        est.net = net
        est.sindy_model = checkpoint['sindy_model']
        return est


class StlsqEstimator:
    """Wraps pysindy's E-SINDy (ensemble STLSQ + bagging). predict()/
    simulate() reuse a coefficient matrix directly rather than pysindy's own
    predict()/simulate(), so all four estimators' outputs are
    Euler-integration-consistent with each other.

    Each bagged member is its own self-consistent (coefficients, sparsity
    structure) fit — elementwise median/mean across members that disagree on
    *which* terms are active can synthesize a coefficient vector no single
    member actually produced (e.g. member A keeps x*z and zeros x*y, member
    B does the opposite; the elementwise median/mean keeps a nonzero blend
    of both, misrepresenting either fit). So the full ensemble
    (`coef_stack`, one self-consistent matrix per member) is what gets
    saved/loaded — mirroring RolloutSINDyRNN/PolynomialRNN's own
    save-the-whole-ensemble pattern — and reduced to a single coef_matrix
    only at simulate()/predict() time, via one of:

    `simulate='best'` (recommended over 'mean' for this reason): each
    member's own one-step Euler prediction MSE is scored on validation data
    (`xs_val`/`ys_val` passed to fit(), if given) or, if none is given, on
    the training data itself — the single best-fitting member's own matrix
    is then used untouched (self-consistent, no cross-member mixing).

    `simulate='mean'` (default, matches the other three estimators'
    default): plain elementwise mean across coef_stack, then thresholded —
    still susceptible to the structural-mixing issue above, kept as the
    cheap/simple default.
    """

    def __init__(self, threshold=0.2, alpha=0.05, n_models=11, degree=2, dt=1.0,
                feature_names=None, simulate='mean'):
        self.threshold = threshold
        self.alpha = alpha
        self.n_models = n_models
        self.degree = degree
        self.dt = dt
        self.feature_names = feature_names
        self.simulate_mode = simulate
        self.coef_stack = None      # (n_models, n_states, n_terms) — full ensemble
        self.best_member_idx = 0
        self.coef_matrix = None     # resolved (n_states, n_terms), per simulate_mode

    def fit(self, xs, ys=None, xs_val=None, ys_val=None):
        """xs: (N_time, n_states). ys: optional precomputed derivative
        (N_time, n_states); computed via np.gradient if not given.

        xs_val: optional held-out state trajectory used to pick the best
            ensemble member (simulate='best'). Falls back to xs (training
            data) if not given.
        """
        import pysindy as ps
        np.math = math

        z = np.asarray(xs)
        z_dot = np.asarray(ys) if ys is not None else np.gradient(z, self.dt, axis=0)

        ensemble_optimizer = ps.EnsembleOptimizer(
            opt=ps.STLSQ(threshold=self.threshold, alpha=self.alpha),
            bagging=True, n_models=self.n_models)
        sindy_model = ps.SINDy(optimizer=ensemble_optimizer, feature_library=ps.PolynomialLibrary(degree=self.degree))
        sindy_model.fit(z, t=self.dt, x_dot=z_dot)

        self.coef_stack = np.array(ensemble_optimizer.coef_list)  # (n_models, n_states, n_terms)

        z_eval = np.asarray(xs_val) if xs_val is not None else z
        self.best_member_idx = self._select_best_member(z_eval)
        self._resolve_coef_matrix()

        print("\nDiscovered equations:")
        print(self.get_equations())

        return self

    def _select_best_member(self, z_eval):
        """Score each ensemble member's own one-step Euler prediction MSE on
        z_eval and return the index of the lowest-MSE member."""
        if len(self.coef_stack) == 1 or len(z_eval) < 2:
            return 0
        library = np.stack([self._library(h) for h in z_eval])  # (N, n_terms)
        best_idx, best_mse = 0, np.inf
        for i, coef in enumerate(self.coef_stack):
            pred = z_eval + self.dt * (library @ coef.T)
            mse = np.mean((pred[:-1] - z_eval[1:]) ** 2)
            if mse < best_mse:
                best_idx, best_mse = i, mse
        return best_idx

    def _resolve_coef_matrix(self):
        if self.simulate_mode == 'best':
            coef = self.coef_stack[self.best_member_idx].copy()
        else:
            coef = self.coef_stack.mean(axis=0)
            coef[np.abs(coef) < self.threshold] = 0
        self.coef_matrix = coef

    def _library(self, h):
        from itertools import combinations_with_replacement
        n_states = len(h)
        terms = [1.0]
        for d in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(n_states), d):
                val = 1.0
                for idx in combo:
                    val *= h[idx]
                terms.append(val)
        return np.array(terms)

    def get_equations(self):
        """Format the resolved coefficients as an ODE string, unlike
        pysindy's own sindy_model.print() (a separate, differently-
        aggregated coefficient source).

        For simulate='best' this is self.coef_matrix — self-consistent with
        what predict()/simulate() actually use. For simulate='mean' this
        instead reports the elementwise ensemble *median* across
        coef_stack: predict()/simulate() still use the plain mean (changing
        that would alter forecast/reconstruction outputs, not just the
        printed report), but a mean lets one outlier bagged member skew a
        displayed coefficient the way a median doesn't.
        """
        if self.simulate_mode == 'best':
            coef_matrix = self.coef_matrix
        else:
            coef_matrix = np.median(self.coef_stack, axis=0)
            coef_matrix[np.abs(coef_matrix) < self.threshold] = 0

        from sindy_rnn.polynomial_library import get_library_feature_names
        names = self.feature_names or ([f'x{i}' for i in range(coef_matrix.shape[0])])
        term_names = get_library_feature_names(names, self.degree)
        lines = []
        for i, state_name in enumerate(names):
            parts = []
            for j, term_name in enumerate(term_names):
                val = coef_matrix[i, j]
                if abs(val) < 1e-6:
                    continue
                sign = '+' if val >= 0 else '-'
                label = f"{abs(val):.3f}" if term_name == '1' else f"{abs(val):.3f}*{term_name}"
                parts.append((sign, label))
            if not parts:
                rhs = '0'
            else:
                sign0, label0 = parts[0]
                rhs = f"-{label0}" if sign0 == '-' else label0
                for sign, label in parts[1:]:
                    rhs += f" {sign} {label}"
            lines.append(f"d{state_name}/dt = {rhs}")
        return '\n'.join(lines)

    def predict(self, xs):
        """One Euler step forward at each row of xs. Returns (N_time, n_states)."""
        xs = np.asarray(xs)
        out = np.zeros_like(xs)
        for t, h in enumerate(xs):
            dh = self._library(h) @ self.coef_matrix.T
            out[t] = h + self.dt * dh
        return out

    def _deriv(self, h):
        return self._library(h) @ self.coef_matrix.T

    def simulate(self, x0, n_steps):
        """Autonomous RK4 integration from x0 (n_states,) for n_steps.

        RK4 (not Euler) to match PolynomialRNNEstimator.simulate()'s
        integrator — otherwise forecast MSE partly measures integrator
        truncation error instead of coefficient quality (see CLAUDE.md
        §14.1's shared-evaluation-protocol rule).
        """
        h = np.asarray(x0, dtype=np.float64).copy()
        traj = []
        for _ in range(n_steps):
            k1 = self._deriv(h)
            k2 = self._deriv(h + self.dt / 2 * k1)
            k3 = self._deriv(h + self.dt / 2 * k2)
            k4 = self._deriv(h + self.dt * k3)
            h = h + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            traj.append(h.copy())
            if np.any(np.abs(h) > 1e6):
                break
        return np.array(traj)

    def save(self, path):
        np.savez_compressed(
            path, coef_stack=self.coef_stack, dt=self.dt, degree=self.degree,
            threshold=self.threshold, best_member_idx=self.best_member_idx,
            simulate_mode=self.simulate_mode,
            feature_names=np.array(self.feature_names) if self.feature_names else np.array([]))

    @classmethod
    def load(cls, path, simulate=None):
        cached = np.load(path)
        feature_names = list(cached['feature_names']) if cached['feature_names'].size else None
        est = cls(
            dt=float(cached['dt']), degree=int(cached['degree']),
            threshold=float(cached['threshold']), feature_names=feature_names,
            simulate=simulate if simulate is not None else str(cached['simulate_mode'].item()))
        est.coef_stack = cached['coef_stack']
        est.best_member_idx = int(cached['best_member_idx'])
        est._resolve_coef_matrix()
        return est
