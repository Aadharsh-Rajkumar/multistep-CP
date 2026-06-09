#!/usr/bin/env python3
"""
Integrated ALPaCA + DtACI: Adaptive Ellipsoid Prediction System
================================================================

Combines:
  1. ALPaCA (one-step meta-learner) → predicts μ, Σ_alpaca for next error
  2. DtACI expert framework → adaptively controls:
       - α_t  (error rate / miscoverage)
       - Σ_T  (ellipsoid shape via exponentially weighted avg of ε·εᵀ)
       - s_t  (ellipsoid scale factor)
  3. Σ optimization: min det(Σ) + x·‖Σ − Σ_T‖_F   s.t. Σ ≻ 0

At each timestep the system produces a coverage ellipsoid  s_t · Σ_opt
and checks whether the true error ε_t falls inside it.
"""

import os
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from copy import deepcopy


# ═══════════════════════════════════════════════════════════════════════════════
#  PID-INSPIRED SATURATION
# ═══════════════════════════════════════════════════════════════════════════════

def _mytan(x):
    """Safe tangent that returns ±inf at ±π/2 boundaries."""
    if x >= np.pi / 2:
        return np.inf
    elif x <= -np.pi / 2:
        return -np.inf
    else:
        return np.tan(x)


def saturation_fn(integrator_sum, t, C_sat=0.5, K_I=1.0):
    """
    PID-paper saturation function (log variant) to prevent integrator windup.
    
    Maps the cumulative coverage error through a bounded nonlinearity:
        sat(x, t) = K_I * tan( x * log(t+1) / (C_sat * (t+1)) )
    
    This guarantees the integrator's contribution stays bounded as t → ∞,
    which is required for the long-run coverage guarantee in the PID paper.
    
    Args:
        integrator_sum: cumulative sum of (α_target - err_t) over all t
        t: current timestep (0-indexed)
        C_sat: saturation constant (controls how aggressively to bound)
        K_I: integrator gain
    
    Returns:
        Saturated integrator output (bounded scalar)
    """
    if K_I == 0:
        return 0.0
    arg = integrator_sum * np.log(t + 2) / (C_sat * (t + 2))
    return K_I * _mytan(arg)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_multidim_regime_shift(T, gap, Sigma1, Sigma2, seed=42):
    """
    Generate T samples of d-dimensional errors with regime switches.

    Every `gap` steps the generating covariance toggles between Sigma1 and Sigma2.
    Returns: (T, d) array of error vectors.
    """
    rng = np.random.default_rng(seed)
    d = Sigma1.shape[0]
    errors = np.zeros((T, d))
    L1 = np.linalg.cholesky(Sigma1)
    L2 = np.linalg.cholesky(Sigma2)
    regime = 1
    for t in range(T):
        z = rng.standard_normal(d)
        errors[t] = z @ (L1 if regime == 1 else L2).T
        if (t + 1) % gap == 0:
            regime = 3 - regime          # toggle 1 ↔ 2
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
#  DtACI EXPERT FRAMEWORK  (ported from R)
# ═══════════════════════════════════════════════════════════════════════════════

class DtACIExpertPool:
    """
    Generic DtACI expert pool that adapts a scalar quantity via
        q_{t+1} = q_t + γ · (target − err_t)
    using a pool of K candidate gammas with exponential weighting.

    This is the building block used for α_t, scale s_t, etc.

    The update direction can be flipped with `invert=True` for quantities
    that should INCREASE on miss (like scale), unlike α which DECREASES on miss.
    """

    def __init__(self, gammas, target, init_val=None,
                 sigma_mix=1/1000, eta=2.72, clamp=None, invert=False):
        """
        Args:
            gammas:    array of candidate learning rates
            target:    target miscoverage rate (α)
            init_val:  starting value for all experts (default = target)
            sigma_mix: mixing weight for stability
            eta:       learning rate for expert weighting
            clamp:     (lo, hi) tuple to clamp values, or None
            invert:    if True, update direction is reversed
                       (value increases on miss, decreases on cover)
        """
        self.gammas = np.array(gammas, dtype=np.float64)
        self.K = len(gammas)
        self.target = target
        self.sigma_mix = sigma_mix
        self.eta = eta
        self.clamp = clamp
        self.invert = invert

        val0 = target if init_val is None else init_val
        self.expert_vals = np.full(self.K, val0, dtype=np.float64)
        self.expert_ws = np.ones(self.K)
        self.expert_probs = np.ones(self.K) / self.K
        self.cur_expert = np.random.randint(self.K)

    @property
    def value(self):
        """Current value from the selected expert."""
        return self.expert_vals[self.cur_expert]

    def update(self, err_t, beta_t=None):
        """
        Update after observing coverage error (1 = missed, 0 = covered).

        When beta_t is provided, uses the pinball loss exactly as in DtACI.R:
            loss_j = pinball(beta_t - expert_val_j, alpha)
        Otherwise uses coverage-based loss: |err_t - target|.

        Returns the value that was active *before* this update.
        """
        val_before = self.value

        # Compute per-expert losses (pinball loss as in the R code)
        if beta_t is not None:
            # pinball(u, alpha) = alpha*u - min(u, 0)
            u = beta_t - self.expert_vals
            expert_losses = self.target * u - np.minimum(u, 0)
        else:
            expert_losses = np.abs(err_t - self.target) * np.ones(self.K)

        # Update each expert's value: q_{j,t+1} = q_{j,t} + γ_j * (target − err_t)
        # If inverted: q_{j,t+1} = q_{j,t} + γ_j * (err_t − target)
        direction = -1.0 if self.invert else 1.0
        for j in range(self.K):
            self.expert_vals[j] += self.gammas[j] * direction * (self.target - err_t)
            if self.clamp is not None:
                self.expert_vals[j] = np.clip(self.expert_vals[j],
                                              self.clamp[0], self.clamp[1])

        # Expert weighting (exponential with mixing)
        if self.eta < np.inf:
            bar_ws = self.expert_ws * np.exp(-self.eta * expert_losses)
            if np.any(np.isnan(bar_ws)) or bar_ws.sum() == 0:
                bar_ws = np.ones(self.K)
            next_ws = (1 - self.sigma_mix) * bar_ws / bar_ws.sum() \
                      + self.sigma_mix / self.K
            self.expert_probs = next_ws / next_ws.sum()
            if np.any(np.isnan(self.expert_probs)):
                self.expert_probs = np.ones(self.K) / self.K
            self.cur_expert = np.random.choice(self.K, p=self.expert_probs)
            self.expert_ws = next_ws
        else:
            self.expert_cum_losses = getattr(self, 'expert_cum_losses',
                                             np.zeros(self.K))
            self.expert_cum_losses += expert_losses
            self.cur_expert = np.argmin(self.expert_cum_losses)

        return val_before


class DtACISigmaPool:
    """
    DtACI expert pool for the shape matrix Σ_T.

    Each expert maintains its own Σ_T updated via
        Σ_T^{t+1} = (1 − γ) Σ_T^t + γ · ε_t ε_tᵀ

    Expert selection is based on coverage loss of the resulting ellipsoid.
    """

    def __init__(self, gammas, d, sigma_init=None, warmup_data=None,
                 sigma_mix=1/1000, eta=2.72):
        self.gammas = np.array(gammas, dtype=np.float64)
        self.K = len(gammas)
        self.d = d
        self.sigma_mix = sigma_mix
        self.eta = eta

        # Initialize Σ
        if sigma_init is not None:
            S0 = sigma_init.copy()
        elif warmup_data is not None and len(warmup_data) > 0:
            S0 = np.cov(warmup_data.T) + 1e-4 * np.eye(d)
        else:
            S0 = np.eye(d)

        self.expert_sigmas = [S0.copy() for _ in range(self.K)]
        self.expert_ws = np.ones(self.K)
        self.expert_probs = np.ones(self.K) / self.K
        self.cur_expert = np.random.randint(self.K)

    @property
    def value(self):
        """Current shape Σ_T from the selected expert."""
        return self.expert_sigmas[self.cur_expert].copy()

    def update(self, eps, err_t, scale_factor):
        """
        Update all experts' Σ after observing error vector eps.
        err_t: 1 if point was outside ellipsoid, 0 if inside.
        scale_factor: current scale (s_t * chi2 quantile) for loss evaluation.
        """
        sigma_before = self.value
        outer = np.outer(eps, eps)
        reg = 1e-6 * np.eye(self.d)

        # Compute per-expert coverage loss
        expert_losses = np.zeros(self.K)
        target = 0.1  # will be overridden from outside; placeholder
        for j in range(self.K):
            # Check coverage with this expert's Sigma
            try:
                S_scaled = scale_factor * self.expert_sigmas[j]
                mahal = eps @ np.linalg.solve(S_scaled, eps)
                missed = float(mahal >= 1.0)
            except np.linalg.LinAlgError:
                missed = 1.0
            expert_losses[j] = abs(missed - target)

            # Update the expert's Sigma
            g = self.gammas[j]
            new_sigma = (1 - g) * self.expert_sigmas[j] + g * outer + reg
            # NaN guard: reset expert sigma if update produces invalid values
            if np.any(np.isnan(new_sigma)) or np.any(np.isinf(new_sigma)):
                new_sigma = outer + reg
            self.expert_sigmas[j] = new_sigma

        # Expert weighting
        if self.eta < np.inf:
            bar_ws = self.expert_ws * np.exp(-self.eta * expert_losses)
            if np.any(np.isnan(bar_ws)) or bar_ws.sum() == 0:
                bar_ws = np.ones(self.K)
            next_ws = (1 - self.sigma_mix) * bar_ws / bar_ws.sum() \
                      + self.sigma_mix / self.K
            self.expert_probs = next_ws / next_ws.sum()
            if np.any(np.isnan(self.expert_probs)):
                self.expert_probs = np.ones(self.K) / self.K
            self.cur_expert = np.random.choice(self.K, p=self.expert_probs)
            self.expert_ws = next_ws

        return sigma_before


# ═══════════════════════════════════════════════════════════════════════════════
#  Σ OPTIMIZATION:  min  det(Σ) + x · ‖Σ − Σ_T‖_F,   Σ ≻ 0
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_sigma(sigma_T, lam, d, alpha_t, recent_residuals, max_iter=200, min_eig_frac=0.3, prob_method="empirical", prob_weight=10.0):
    """
    Find Σ* that minimizes  |pr(eΣ^{-1}e < 1) - (1 -\alpha)| +  det(Σ) + λ · ‖Σ − Σ_T‖_F
    subject to Σ being symmetric positive-definite.

    We parametrize Σ = L Lᵀ where L is lower-triangular with positive diagonal
    (diagonal entries = softplus(param) + min_diag to prevent collapse).
    min_diag is set adaptively as min_eig_frac * sqrt(min_eigenvalue(Σ_T)).

    Args:
        sigma_T:          (d, d) current rolling shape matrix
        lam:              regularization weight (the "x" from the formula)
        d:                dimensionality
        alpha_t:          target adaptive error rate
        recent_residuals: (n, d) matrix of recent error offsets
        max_iter:         optimizer iterations
        min_eig_frac:     fraction of min eigenvalue of Σ_T for L's diagonal floor
        prob_method:      "empirical", "moment", "davies", "none"

    Returns:
        sigma_opt: (d, d) optimized PSD matrix
    """


    # Compute adaptive floor from sigma_T's eigenvalues
    eigvals_T = np.linalg.eigvalsh(sigma_T)
    min_eig = max(eigvals_T.min(), 1e-8)
    min_diag = min_eig_frac * np.sqrt(min_eig)

    # Initialize L from Cholesky of sigma_T
    try:
        L0 = np.linalg.cholesky(sigma_T)
    except np.linalg.LinAlgError:
        shift = max(0, -eigvals_T.min()) + 1e-6
        L0 = np.linalg.cholesky(sigma_T + shift * np.eye(d))

    tril_idx = np.tril_indices(d)
    diag_pos = []
    k = 0
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                diag_pos.append(k)
            k += 1
    diag_pos = np.array(diag_pos)

    def softplus(x):
        return np.log1p(np.exp(np.clip(x, -20, 20)))

    def inv_softplus(y):
        y = np.maximum(y - min_diag, 1e-8)
        return np.log(np.expm1(np.maximum(y, 1e-8)))

    def pack_L(L):
        params = L[tril_idx].copy()
        for k_idx, p in enumerate(diag_pos):
            params[p] = inv_softplus(L[k_idx, k_idx])
        return params

    def unpack_L(params):
        L = np.zeros((d, d))
        L[tril_idx] = params
        for k_idx, p in enumerate(diag_pos):
            L[k_idx, k_idx] = softplus(params[p]) + min_diag
        return L

    def objective(params):
        L = unpack_L(params)
        Sigma = L @ L.T
        det_val = np.linalg.det(Sigma)
        frob = np.linalg.norm(Sigma - sigma_T, 'fro')
        
        # Evaluate coverage probability penalty
        if prob_method == "none":
            coverage_penalty = 0.0
        elif len(recent_residuals) > 0:
            inv_Sigma = np.linalg.pinv(Sigma)
            q_alpha = chi2.ppf(1 - alpha_t, df=d) if alpha_t is not None else chi2.ppf(0.9, df=d)
            
            if prob_method == "empirical":
                # Differentiable sigmoid approximation
                mahal_dists = np.einsum('ij,jk,ik->i', recent_residuals, inv_Sigma, recent_residuals)
                pr = np.mean(1.0 / (1.0 + np.exp(-10.0 * (q_alpha - mahal_dists))))
                coverage_penalty = prob_weight * (pr - (1.0 - alpha_t))**2
                
            # This is the best option as of now for real-time online CP
            elif prob_method == "moment":
                if len(recent_residuals) > 1:
                    # Uncentered second moment captures both variance and origin-offset bias
                    Sigma_e = (recent_residuals.T @ recent_residuals) / len(recent_residuals)
                else:
                    Sigma_e = np.eye(d)
                Sigma_e = Sigma_e + 1e-6 * np.eye(d)

                AS = inv_Sigma @ Sigma_e
                AS = 0.5 * (AS + AS.T)

                E_Q = np.trace(AS)
                Var_Q = 2.0 * np.trace(AS @ AS)

                E_Q = np.clip(E_Q, 1e-8, 1e6)
                Var_Q = np.clip(Var_Q, 1e-8, 1e6)

                k = (E_Q ** 2) / (Var_Q + 1e-8)
                theta = (Var_Q + 1e-8) / (E_Q + 1e-8)

                try:
                    import scipy.stats
                    pr = scipy.stats.gamma.cdf(q_alpha, a=k, scale=theta)
                except:
                    Q = np.einsum('ij,jk,ik->i', recent_residuals, inv_Sigma, recent_residuals)
                    pr = np.mean(Q < q_alpha)

                coverage_penalty = prob_weight * (pr - (1.0 - alpha_t))**2
                    
            elif prob_method == "davies":
                # Exact Gil-Pelaez CF Numerical Integration
                if len(recent_residuals) > 1:
                    Sigma_e = (recent_residuals.T @ recent_residuals) / len(recent_residuals)
                else:
                    Sigma_e = np.eye(d)
                # Force zero-mean inside the algorithm since uncentered Sigma_e incorporates the bias
                mu_e = np.zeros(d)
                A = inv_Sigma
                try:
                    L_e = np.linalg.cholesky(Sigma_e)
                except np.linalg.LinAlgError:
                    L_e = np.linalg.cholesky(Sigma_e + 1e-6 * np.eye(d))
                
                A_scaled = L_e.T @ A @ L_e
                lambdas, V = np.linalg.eigh(A_scaled)
                b_vec = V.T @ L_e.T @ A @ mu_e
                c_val = mu_e.T @ A @ mu_e
                
                def cf_vec(t):
                    res = np.exp(1j * t * c_val)
                    for j in range(len(lambdas)):
                        lj = lambdas[j]
                        if abs(lj) < 1e-10:
                            res *= np.exp(2j * t * b_vec[j])
                        else:
                            dj = b_vec[j] / lj
                            term1 = (1 - 2j * t * lj)**(-0.5)
                            term2 = np.exp((1j * t * lj * dj**2) / (1 - 2j * t * lj))
                            term3 = np.exp(-1j * t * lj * dj**2)
                            res *= term1 * term2 * term3
                    return res
                
                t_vals = np.linspace(1e-5, 20.0, 150)
                y_vals = np.imag(cf_vec(t_vals) * np.exp(-1j * t_vals * q_alpha)) / t_vals
                integral = np.trapz(y_vals, t_vals)
                
                pr = np.clip(0.5 - (1.0 / np.pi) * integral, 0.0, 1.0)
                coverage_penalty = prob_weight * (pr - (1.0 - alpha_t))**2
                
        else:
            coverage_penalty = 0.0

        return coverage_penalty + det_val + lam * frob

    def gradient(params):
        h = 1e-7
        grad = np.zeros_like(params)
        f0 = objective(params)
        for i in range(len(params)):
            p = params.copy()
            p[i] += h
            grad[i] = (objective(p) - f0) / h
        return grad

    x0 = pack_L(L0)
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12})

    L_opt = unpack_L(result.x)
    sigma_opt = L_opt @ L_opt.T

    # NaN guard (PID paper principle: never propagate invalid states)
    if np.any(np.isnan(sigma_opt)) or np.any(np.isinf(sigma_opt)):
        sigma_opt = sigma_T.copy()

    return sigma_opt


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-DIMENSIONAL ALPaCA WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class MultiDimErrorTaskDataset:
    """Wraps multi-dimensional error sequences into ALPaCA's sample() interface."""

    def __init__(self, errors, n_context, alpha_dim=True):
        """
        Args:
            errors: (T, d) array of error vectors
            n_context: number of past steps as context
            alpha_dim: if True, input includes α_t as an extra dimension
        """
        self.errors = errors
        self.n_context = n_context
        self.d = errors.shape[1]
        self.alpha_dim = alpha_dim

        # Build sliding windows
        # Input: flatten [ε_{t-n},...,ε_{t-1}] + optionally α_t
        # Output: ε_t
        T = len(errors)
        self.x_dim = n_context * self.d + (1 if alpha_dim else 0)
        self.y_dim = self.d

        if T <= n_context:
            self.X = np.zeros((0, self.x_dim))
            self.Y = np.zeros((0, self.y_dim))
            return

        self.X = np.zeros((T - n_context, self.x_dim))
        self.Y = errors[n_context:]

        for i in range(T - n_context):
            context = errors[i:i + n_context].flatten()
            if alpha_dim:
                # Default α placeholder (will be set during actual use)
                self.X[i] = np.concatenate([context, [0.1]])
            else:
                self.X[i] = context

    def sample(self, n_funcs, n_samples):
        """Sample batches for ALPaCA meta-training."""
        T = self.X.shape[0]
        x = np.zeros((n_funcs, n_samples, self.x_dim))
        y = np.zeros((n_funcs, n_samples, self.y_dim))

        for i in range(n_funcs):
            if T >= n_samples:
                start = np.random.randint(0, T - n_samples + 1)
                inds = np.arange(start, start + n_samples)
            else:
                inds = np.random.choice(T, n_samples, replace=True)
            x[i] = self.X[inds]
            y[i] = self.Y[inds]
        return x, y


def build_multidim_alpaca_config(d, n_context, alpha_dim=True, **kwargs):
    """Build ALPaCA config for multi-dimensional error prediction."""
    x_dim = n_context * d + (1 if alpha_dim else 0)
    return {
        "lr": kwargs.get("lr", 0.001),
        "x_dim": x_dim,
        "y_dim": d,
        "sigma_eps": kwargs.get("sigma_eps", 0.1),
        "data_horizon": n_context,
        "test_horizon": max(kwargs.get("test_horizon", 10), 1),
        "meta_batch_size": kwargs.get("meta_batch_size", 10),
        "nn_layers": kwargs.get("nn_layers", [128, 128, 32]),
        "activation": kwargs.get("activation", "tanh"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  INTEGRATED SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class IntegratedAlpacaDtACI:
    """
    Full integrated ALPaCA + DtACI system.

    At each timestep:
      1. ALPaCA predicts (μ, Σ_alpaca) conditioned on recent errors + α_t
      2. Observe true error ε_t
      3. Optimize Σ: min det(Σ) + x·‖Σ − Σ_T‖_F
      4. Check coverage with final ellipsoid s_t · Σ_opt
      5. Update all DtACI controllers (α_t, Σ_T, s_t)
      6. Feed updated α_t back to ALPaCA
    """

    def __init__(self,
                 d,
                 alpha_target=0.1,
                 n_context=10,
                 lam=0.5,
                 alpha_gamma=0.01,
                 scale_gamma=0.01,
                 warmup=50,
                 alpaca_train_iters=100,
                 alpaca_kwargs=None,
                 replay_buffer_size=0,
                 prob_method="empirical",
                 prob_weight=10.0,
                 force_zero_mean=False):
        """
        d: dimensionality of the time series
        alpha_target: target expected error rate (e.g., 0.1 for 90% coverage)
        n_context: number of historical points ALPaCA uses to predict mu
        lam: regularization weight for optimize_sigma (λ · ||Σ - Σ_T||)
        alpha_gamma: step size for adjusting α_t
        scale_gamma: step size for adjusting s_t
        warmup: steps to skip tracking metrics to allow buffers to fill
        alpaca_train_iters: epochs for offline meta-learning loop
        alpaca_kwargs: dict for ALPaCA init (layers, eps)
        replay_buffer_size: steps to replay dynamically to ghost calibrate DtACI threshold tracking
        prob_method: "empirical", "moment", "davies", or "none"
        """
        self.d = d
        self.alpha_target = alpha_target
        self.n_context = n_context
        self.lam = lam
        self.alpha_gamma = alpha_gamma
        self.scale_gamma = scale_gamma
        self.warmup = warmup
        self.alpaca_train_iters = alpaca_train_iters
        self.alpaca_kwargs = alpaca_kwargs or {}
        self.replay_buffer_size = replay_buffer_size
        self.prob_method = prob_method
        self.prob_weight = prob_weight
        self.force_zero_mean = force_zero_mean

        self.gammas = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]

        # --- DtACI Controllers ---

        # 1) α_t: simple scalar (matching R EllipsoidDtACI.R)
        #    α_{t+1} = α_t + alphaGamma * (α_target − err_t)
        self.alpha_t = alpha_target

        # 2) Σ_T controller: DtACI expert pool (the core DtACI component)
        self.sigma_ctrl = DtACISigmaPool(
            gammas=self.gammas, d=d, sigma_init=None,
            sigma_mix=1/1000, eta=2.72
        )

        # 3) s_t: simple scalar
        #    s_{t+1} = s_t + scaleGamma * (err_t − α_target)
        #    increases on miss, decreases on cover
        self.scale_t = 1.0

        # PID-inspired integrator state (from Conformal PID paper)
        # Tracks cumulative coverage error for saturation-based anti-windup
        self.integrator_sum = 0.0
        self.C_sat = 0.5     # saturation constant
        self.K_I = 0.5       # integrator gain

        # ALPaCA model (lazy initialization — trained on warmup data)
        self.alpaca_model = None
        self.alpaca_config = None

    def _init_alpaca(self, warmup_errors):
        """Initialize and train ALPaCA on warmup errors."""
        PYTORCH_DIR = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ALPaCA", "pytorch"
        )
        sys.path.insert(0, PYTORCH_DIR)
        from main.alpaca import ALPaCA
        sys.path.pop(0)

        config = build_multidim_alpaca_config(
            self.d, self.n_context, alpha_dim=False,
            **self.alpaca_kwargs
        )
        self.alpaca_config = config

        dataset = MultiDimErrorTaskDataset(warmup_errors, self.n_context,
                                           alpha_dim=False)
        if dataset.X.shape[0] == 0:
            raise ValueError(
                f"Not enough warm-up data ({len(warmup_errors)}) "
                f"for n_context={self.n_context}"
            )

        model = ALPaCA(config)
        model.construct_model()
        print(f"  Training ALPaCA (d={self.d}) for {self.alpaca_train_iters} iters "
              f"on {len(warmup_errors)} warmup points...")
        model.train_model(dataset, self.alpaca_train_iters)
        self.alpaca_model = model

    def _alpaca_predict(self, context_errors):
        """
        Use ALPaCA to predict μ and Σ_alpaca for the next step.

        Args:
            context_errors: (n_context, d) recent errors

        Returns:
            mu_pred:    (d,) predicted mean
            sig_pred:   (d, d) predicted covariance
        """
        if self.alpaca_model is None:
            # Fallback: return zeros and identity
            return np.zeros(self.d), np.eye(self.d)

        # Build input: [flatten(context)]
        ctx_flat = context_errors.flatten()
        x_input = ctx_flat.reshape(1, 1, -1)

        # Use empty context for BLR (online single-step)
        x_ctx = np.zeros((1, 0, self.alpaca_config['x_dim']))
        y_ctx = np.zeros((1, 0, self.d))

        mu, sig = self.alpaca_model.test(x_ctx, y_ctx, x_input)
        mu_pred = mu[0, 0, :]                  # (d,)
        sig_pred = sig[0, 0, :, :]             # (d, d)

        if getattr(self, 'force_zero_mean', False):
            mu_pred = np.zeros(self.d)

        return mu_pred, sig_pred

    def run(self, errors, verbose=True):
        """
        Run the full integrated system on a sequence of errors.

        Args:
            errors: (T, d) array of multi-dimensional error vectors

        Returns:
            dict with keys:
                coverage_seq:    (T,) binary coverage (1=covered, 0=missed)
                alpha_seq:       (T,) adaptive α_t
                scale_seq:       (T,) adaptive s_t
                sigma_T_seq:     list of T (d,d) rolling shape matrices
                sigma_opt_seq:   list of T (d,d) optimized shape matrices
                volume_seq:      (T,) det(s_t · Σ_opt)
                mahal_seq:       (T,) Mahalanobis distances
                mu_pred_seq:     (T, d) ALPaCA predicted means
                gamma_alpha_seq: (T,) selected gamma for α
                gamma_sigma_seq: (T,) selected gamma for Σ_T
                gamma_scale_seq: (T,) selected gamma for scale
        """
        T, d = errors.shape
        assert d == self.d, f"Expected d={self.d}, got {d}"

        # --- Warm-up: initialize Σ_T and ALPaCA ---
        warmup_end = min(self.warmup, T // 4)
        warmup_end = max(warmup_end, self.n_context + 5)

        warmup_errors = errors[:warmup_end]

        # Initialize Σ_T controller with warm-up data
        S0 = np.cov(warmup_errors.T)
        if S0.ndim < 2:
            S0 = np.array([[S0]])
        S0 += 1e-4 * np.eye(d)
        self.sigma_ctrl = DtACISigmaPool(
            gammas=self.sigma_ctrl.gammas, d=d, sigma_init=S0,
            sigma_mix=self.sigma_ctrl.sigma_mix, eta=self.sigma_ctrl.eta
        )

        # Train ALPaCA on warm-up errors
        try:
            self._init_alpaca(warmup_errors)
        except Exception as e:
            print(f"  ALPaCA init failed: {e}. Running without ALPaCA predictions.")

        # --- Output storage ---
        coverage_seq = np.zeros(T)
        alpha_seq = np.zeros(T)
        scale_seq = np.zeros(T)
        volume_seq = np.zeros(T)
        mahal_seq = np.zeros(T)
        mu_pred_seq = np.zeros((T, d))
        sigma_T_seq = [None] * T
        sigma_opt_seq = [None] * T
        gamma_alpha_seq = np.zeros(T)
        gamma_sigma_seq = np.zeros(T)
        gamma_scale_seq = np.zeros(T)

        # --- Main loop ---
        for t, eps in enumerate(errors):
            alpha_t = self.alpha_t
            s_t = self.scale_t
            sigma_T = self.sigma_ctrl.value

            sigma_T_seq[t] = sigma_T.copy()
            gamma_alpha_seq[t] = self.alpha_gamma
            gamma_sigma_seq[t] = self.sigma_ctrl.gammas[self.sigma_ctrl.cur_expert]
            gamma_scale_seq[t] = self.scale_gamma

            # 1) ALPaCA prediction (after warmup)
            if t >= self.n_context:
                ctx = errors[max(0, t - self.n_context):t]
                mu_pred, sig_alpaca = self._alpaca_predict(ctx)
                mu_pred_seq[t] = mu_pred
            else:
                mu_pred = np.zeros(d)
                sig_alpaca = np.eye(d)
                mu_pred_seq[t] = mu_pred

            # 2) Optimize Σ:
            try:
                recent_residuals = ctx - mu_pred if t >= self.n_context else np.zeros((0, d))
                sigma_opt = optimize_sigma(sigma_T, self.lam, d, alpha_t, recent_residuals, prob_method=self.prob_method, prob_weight=self.prob_weight)
            except Exception:
                sigma_opt = sigma_T.copy()
            sigma_opt_seq[t] = sigma_opt.copy()

            # 3) Ghost Replay Buffer
            alpha_b = alpha_t
            scale_b = s_t
            if self.replay_buffer_size > 0 and t >= self.replay_buffer_size:
                try:
                    B = self.replay_buffer_size
                    # Replay via simulated residual geometry mapping
                    res_hist = errors[t-B:t] - mu_pred_seq[t-B:t]
                    inv_sig = np.linalg.pinv(sigma_opt)
                    dists = np.einsum('ij,jk,ik->i', res_hist, inv_sig, res_hist)
                    for b in range(B):
                        a_clamp = np.clip(alpha_b, 0.001, 0.999)
                        q = chi2.ppf(1 - a_clamp, df=d)
                        if dists[b] > scale_b * q:
                            # Missed
                            alpha_b = np.clip(alpha_b + self.alpha_gamma * (self.alpha_target - 1.0), 0.001, 0.999)
                            scale_b = np.clip(scale_b + self.scale_gamma * (1.0 - self.alpha_target), 0.1, 50.0)
                        else:
                            # Covered
                            alpha_b = np.clip(alpha_b + self.alpha_gamma * (self.alpha_target - 0.0), 0.001, 0.999)
                            scale_b = np.clip(scale_b + self.scale_gamma * (0.0 - self.alpha_target), 0.1, 50.0)
                except Exception:
                    pass

            # Log jump-started boundaries for visualization
            alpha_seq[t] = alpha_b
            scale_seq[t] = scale_b

            # 4) Final ellipsoid: ghost_s_t · ghost_chi2_quantile · Σ_opt
            alpha_clamp = np.clip(alpha_b, 0.001, 0.999)
            chi2_scale = chi2.ppf(1 - alpha_clamp, df=d)
            final_sigma = scale_b * chi2_scale * sigma_opt

            # Calculate residual relative to ALPaCA's predicted shift
            residual_t = eps - mu_pred

            # 4) Coverage check: rᵀ (final_Σ)⁻¹ r < 1 → covered
            try:
                mahal = residual_t @ np.linalg.solve(final_sigma, residual_t)
            except np.linalg.LinAlgError:
                mahal = np.inf

            mahal_seq[t] = mahal
            covered = float(mahal < 1.0)
            coverage_seq[t] = covered
            err_t = 1.0 - covered       # 1 if missed

            volume_seq[t] = max(np.linalg.det(final_sigma), 1e-12)

            # 5) Update controllers with PID-inspired saturation

            #    P-term: α_t proportional update
            #    α_{t+1} = α_t + αGamma * (α_target − err_t)
            #    If covered (err=0): α increases → chi2_scale decreases → tighter
            #    If missed  (err=1): α decreases → chi2_scale increases → wider
            p_update = self.alpha_gamma * (self.alpha_target - err_t)

            #    I-term: Saturated integrator to prevent windup
            #    Accumulates coverage error, but bounded via tan(·) saturation
            self.integrator_sum += (self.alpha_target - err_t)
            i_update = saturation_fn(
                self.integrator_sum, t,
                C_sat=self.C_sat, K_I=self.K_I
            )

            #    Combined PID α update (P + saturated-I)
            new_alpha = self.alpha_target + p_update + self.alpha_gamma * i_update
            
            #    NaN guard (PID paper: skip update if NaN, reset integrator)
            if np.isnan(new_alpha) or np.isinf(new_alpha):
                new_alpha = self.alpha_target
                self.integrator_sum = 0.0  # Reset integrator on NaN
            
            self.alpha_t = np.clip(new_alpha, 0.001, 0.999)

            #    Σ_T: DtACI expert pool (shape adaptation uses residual variance)
            self.sigma_ctrl.update(residual_t, err_t, scale_b * chi2_scale)

            #    s_t: scalar update with NaN guard
            new_scale = self.scale_t + self.scale_gamma * (err_t - self.alpha_target)
            if np.isnan(new_scale) or np.isinf(new_scale):
                new_scale = 1.0  # Reset scale on NaN
            self.scale_t = np.clip(new_scale, 0.1, 50.0)

            # Progress
            if verbose and (t + 1) % 200 == 0:
                rolling_cov = coverage_seq[max(0, t - 199):t + 1].mean()
                print(f"  Step {t+1}/{T} | rolling_cov: {rolling_cov:.3f} | "
                      f"α_t (ghost): {alpha_b:.4f} | s_t (ghost): {scale_b:.4f} | "
                      f"det(Σ_opt): {np.linalg.det(sigma_opt):.4e}")

        results = {
            "coverage_seq": coverage_seq,
            "alpha_seq": alpha_seq,
            "scale_seq": scale_seq,
            "sigma_T_seq": sigma_T_seq,
            "sigma_opt_seq": sigma_opt_seq,
            "volume_seq": volume_seq,
            "mahal_seq": mahal_seq,
            "mu_pred_seq": mu_pred_seq,
            "gamma_alpha_seq": gamma_alpha_seq,
            "gamma_sigma_seq": gamma_sigma_seq,
            "gamma_scale_seq": gamma_scale_seq,
        }

        if verbose:
            final_cov = coverage_seq.mean()
            print(f"\n  Final coverage: {final_cov:.4f} (target: {1 - self.alpha_target:.4f})")

        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Integrated ALPaCA + DtACI — Quick Test (2D Simulated)")
    print("=" * 60)

    # Regime-shift 2D data
    Sigma1 = np.array([[4.0, 0.5],
                       [0.5, 0.3]])
    Sigma2 = np.array([[0.3, -0.2],
                       [-0.2, 4.0]])

    T = 2000
    gap = 500
    errors = generate_multidim_regime_shift(T, gap, Sigma1, Sigma2)
    print(f"Generated {T} error vectors (d=2) with regime switches every {gap} steps\n")

    system = IntegratedAlpacaDtACI(
        d=2, alpha_target=0.1,
        n_context=10, lam=5.0,
        alpha_gamma=0.05, scale_gamma=0.05,
        warmup=100, alpaca_train_iters=200,
        alpaca_kwargs={"nn_layers": [64, 32], "lr": 0.001, "sigma_eps": 0.1}
    )

    results = system.run(errors, verbose=True)
    print(f"\nDone. Coverage = {results['coverage_seq'].mean():.4f}")

