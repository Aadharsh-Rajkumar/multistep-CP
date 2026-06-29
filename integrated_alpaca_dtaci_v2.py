#!/usr/bin/env python3
"""
Integrated ALPaCA + DtACI v2: Trajectory-Level Conformal Risk Allocation
=========================================================================

Extends v1 with:
  1. Trajectory-level risk: P(∪ A_t) ≤ α_target over horizon N
  2. Pairwise overlap learning: ω(Δt) = P(V_s=1, V_t=1)
  3. Kounias–Hunter max-weight forest correction (§5)
  4. Adaptive joint risk budget α_joint^live (§8)
  5. Less conservative ellipsoids via overlap-corrected Bonferroni

Reference: "Online Dependence-Aware Conformal Risk Allocation for
Receding-Horizon Control" (Srinivasan, 2024)
"""

import os
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from copy import deepcopy

# Import all v1 components (no duplication)
from integrated_alpaca_dtaci import (
    generate_multidim_regime_shift,
    saturation_fn, _mytan,
    DtACIExpertPool, DtACISigmaPool,
    optimize_sigma,
    MultiDimErrorTaskDataset,
    build_multidim_alpaca_config,
    IntegratedAlpacaDtACI,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  OVERLAP TABLE  
# ═══════════════════════════════════════════════════════════════════════════════

class OverlapTable:
    """
    Discretized table for learning pairwise violation overlaps online.

    Minimal-prototype version: indexed only by Δt (lag between steps).
    Maintains Welford online statistics per cell:
        n_j, ω̂_j (running mean), M2_j (Welford accumulator)

    Provides conservative lower bounds via Hoeffding margins.
    """

    def __init__(self, K, delta_conf=0.05):
        """
        Args:
            K:          maximum lag to track (Δt ∈ {1, …, K})
            delta_conf: confidence budget for Hoeffding margins
        """
        self.K = K
        self.delta_conf = delta_conf
        self.n = np.zeros(K, dtype=np.int64)
        self.omega_hat = np.zeros(K)
        self.M2 = np.zeros(K)

    def update(self, delta_t, Y):
        """Update cell for lag *delta_t* with overlap indicator Y ∈ {0,1}."""
        if delta_t < 1 or delta_t > self.K:
            return
        idx = delta_t - 1
        self.n[idx] += 1
        old_mean = self.omega_hat[idx]
        self.omega_hat[idx] += (Y - self.omega_hat[idx]) / self.n[idx]
        self.M2[idx] += (Y - old_mean) * (Y - self.omega_hat[idx])

    def get_variance(self, delta_t):
        """Empirical variance for lag delta_t."""
        idx = delta_t - 1
        if self.n[idx] < 2:
            return 0.25
        return self.M2[idx] / max(1, self.n[idx] - 1)

    def get_safe_omega(self, delta_t):
        """
        Conservative lower bound:  ω^safe = max{0, ω̂ − M_H}.

        Hoeffding margin: M_H = sqrt( log(K/δ) / (2n) )
        """
        if delta_t < 1 or delta_t > self.K:
            return 0.0
        idx = delta_t - 1
        if self.n[idx] < 2:
            return 0.0
        delta_j = self.delta_conf / self.K
        M = np.sqrt(np.log(1.0 / delta_j) / (2.0 * self.n[idx]))
        return max(0.0, self.omega_hat[idx] - M)

    def get_all_safe(self):
        """Return dict {Δt: ω^safe} for Δt = 1…K."""
        return {dt: self.get_safe_omega(dt) for dt in range(1, self.K + 1)}

    def summary(self):
        """Print a diagnostic summary of the table."""
        lines = ["  OverlapTable summary (lag → ω̂, ω^safe, n):"]
        for dt in range(1, self.K + 1):
            idx = dt - 1
            ws = self.get_safe_omega(dt)
            lines.append(
                f"    Δt={dt:2d}  ω̂={self.omega_hat[idx]:.4f}  "
                f"ω^safe={ws:.4f}  n={self.n[idx]}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAX-WEIGHT FOREST  (Kounias–Hunter)
# ═══════════════════════════════════════════════════════════════════════════════

def max_weight_forest(N, K, safe_omegas):
    """
    Select a maximum-weight forest over lag-truncated edges using Kruskal's.

    For horizon {1,…,N} and max lag K, candidate edges are
        E_K = {(s,t) : 1 ≤ s < t ≤ N, t−s ≤ K}
    with weight ω^safe_{s,t}.

    Returns:
        forest:     list of (s,t) edges in the forest
        correction: total weight  Σ_{(s,t)∈F} ω^safe
    """
    # Build candidate edges
    edges = []
    for s in range(1, N + 1):
        for t in range(s + 1, min(s + K + 1, N + 1)):
            w = safe_omegas.get(t - s, 0.0)
            if w > 0:
                edges.append((w, s, t))

    edges.sort(reverse=True)  # greedy: heaviest first

    # Union-Find
    parent = list(range(N + 2))
    rank = [0] * (N + 2)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    forest = []
    correction = 0.0
    for w, s, t in edges:
        if union(s, t):
            forest.append((s, t))
            correction += w

    return forest, correction


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAJECTORY-LEVEL SYSTEM  (v2)
# ═══════════════════════════════════════════════════════════════════════════════

class TrajectoryAlpacaDtACI:
    """
    v2 trajectory-level conformal risk allocation.

    At each step, α_m is computed from:
        α_m = (α_joint^live + forest_correction) / N

    instead of the per-step DtACI update used in v1.
    All other components (ALPaCA, Σ optimisation, scale_t,
    DtACISigmaPool) are reused from v1.
    """

    def __init__(self,
                 d,
                 alpha_target=0.1,
                 N=10,
                 K=5,
                 n_context=10,
                 lam=0.5,
                 alpha_gamma=0.01,
                 scale_gamma=0.01,
                 warmup=50,
                 alpaca_train_iters=100,
                 alpaca_kwargs=None,
                 prob_method="moment",
                 prob_weight=10.0,
                 force_zero_mean=False,
                 delta_conf=0.05,
                 eta_joint=0.01,
                 base_matrix_mode="sigma_T"):
        """
        Args (new in v2):
            N:          MPC horizon length
            K:          max lag for overlap learning (K ≤ N)
            delta_conf: confidence budget for overlap bounds
            delta_conf: confidence budget for overlap bounds
            eta_joint:  learning rate for α_joint^live update
            base_matrix_mode: "sigma_T", "alpaca_1alpha", or "alpaca_1sd"
        """
        self.d = d
        self.alpha_target = alpha_target
        self.N = N
        self.K = min(K, N - 1)
        self.n_context = n_context
        self.lam = lam
        self.alpha_gamma = alpha_gamma
        self.scale_gamma = scale_gamma
        self.warmup = warmup
        self.alpaca_train_iters = alpaca_train_iters
        self.alpaca_kwargs = alpaca_kwargs or {}
        self.prob_method = prob_method
        self.prob_weight = prob_weight
        self.force_zero_mean = force_zero_mean
        self.eta_joint = eta_joint
        self.base_matrix_mode = base_matrix_mode

        # ── Overlap table (§6) ──
        self.overlap_table = OverlapTable(self.K, delta_conf=delta_conf)

        # ── Trajectory-level budget (§8) ──
        self.alpha_joint_live = alpha_target

        # ── Shared v1 components ──
        self.gammas = [0.001, 0.002, 0.004, 0.008,
                       0.016, 0.032, 0.064, 0.128]
        self.sigma_ctrl = DtACISigmaPool(
            gammas=self.gammas, d=d, sigma_init=None,
            sigma_mix=1 / 1000, eta=2.72,
        )
        self.scale_t = 1.0
        self.integrator_sum = 0.0
        self.C_sat = 0.5
        self.K_I = 0.5

        # ALPaCA (lazy init)
        self.alpaca_model = None
        self.alpaca_config = None

    # ------------------------------------------------------------------
    #  ALPaCA helpers (reuse v1 logic)
    # ------------------------------------------------------------------
    def _init_alpaca(self, warmup_errors):
        """Initialize and train ALPaCA on warmup errors."""
        PYTORCH_DIR = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ALPaCA", "pytorch",
        )
        sys.path.insert(0, PYTORCH_DIR)
        from main.alpaca import ALPaCA
        sys.path.pop(0)

        config = build_multidim_alpaca_config(
            self.d, self.n_context, alpha_dim=False,
            **self.alpaca_kwargs,
        )
        self.alpaca_config = config

        dataset = MultiDimErrorTaskDataset(
            warmup_errors, self.n_context, alpha_dim=False,
        )
        if dataset.X.shape[0] == 0:
            raise ValueError("Not enough warm-up data")

        model = ALPaCA(config)
        model.construct_model()
        print(f"  [v2] Training ALPaCA (d={self.d}) for "
              f"{self.alpaca_train_iters} iters …")
        model.train_model(dataset, self.alpaca_train_iters)
        self.alpaca_model = model

    def _alpaca_predict(self, context_errors):
        if self.alpaca_model is None:
            return np.zeros(self.d), np.eye(self.d)

        ctx_flat = context_errors.flatten()
        x_input = ctx_flat.reshape(1, 1, -1)
        x_ctx = np.zeros((1, 0, self.alpaca_config["x_dim"]))
        y_ctx = np.zeros((1, 0, self.d))
        mu, sig = self.alpaca_model.test(x_ctx, y_ctx, x_input)
        mu_pred = mu[0, 0, :]
        sig_pred = sig[0, 0, :, :]

        if self.force_zero_mean:
            mu_pred = np.zeros(self.d)
        return mu_pred, sig_pred

    # ------------------------------------------------------------------
    #  Main run loop
    # ------------------------------------------------------------------
    def run(self, errors, verbose=True):
        """
        Run v2 trajectory-level system on error sequence.

        Returns the same dict keys as v1 plus:
            alpha_m_seq:           per-step allocated α_m
            alpha_joint_live_seq:  trajectory budget over time
            correction_seq:        forest correction at each step
            traj_violation_seq:    per-window trajectory violation
        """
        T, d = errors.shape
        assert d == self.d

        # ── Warm-up ──
        warmup_end = min(self.warmup, T // 4)
        warmup_end = max(warmup_end, self.n_context + 5)
        warmup_errors = errors[:warmup_end]

        S0 = np.cov(warmup_errors.T)
        if S0.ndim < 2:
            S0 = np.array([[S0]])
        S0 += 1e-4 * np.eye(d)
        self.sigma_ctrl = DtACISigmaPool(
            gammas=self.gammas, d=d, sigma_init=S0,
            sigma_mix=1 / 1000, eta=2.72,
        )

        try:
            self._init_alpaca(warmup_errors)
        except Exception as e:
            print(f"  [v2] ALPaCA init failed: {e}. Continuing without.")

        # ── Output storage ──
        coverage_seq = np.zeros(T)
        alpha_m_seq = np.zeros(T)
        alpha_joint_live_seq = np.zeros(T)
        correction_seq = np.zeros(T)
        scale_seq = np.zeros(T)
        volume_seq = np.zeros(T)
        mahal_seq = np.zeros(T)
        mu_pred_seq = np.zeros((T, d))
        sigma_T_seq = [None] * T
        sigma_opt_seq = [None] * T
        violation_seq = np.zeros(T)       # V_t per step
        traj_violation_seq = np.zeros(T)   # per-window trajectory indicator

        # ── Main loop ──
        for t in range(T):
            eps = errors[t]
            sigma_T = self.sigma_ctrl.value
            sigma_T_seq[t] = sigma_T.copy()

            # 1) ALPaCA prediction
            if t >= self.n_context:
                ctx = errors[max(0, t - self.n_context):t]
                mu_pred, sig_alpaca = self._alpaca_predict(ctx)
            else:
                mu_pred = np.zeros(d)
                sig_alpaca = np.eye(d)
            mu_pred_seq[t] = mu_pred

            # 2) Query overlap table → forest correction
            safe_omegas = self.overlap_table.get_all_safe()
            _, correction = max_weight_forest(self.N, self.K, safe_omegas)
            correction_seq[t] = correction

            # 3) Compute per-step α_m using joint target formula (Eq. 11)
            alpha_m = (self.alpha_joint_live + correction) / self.N
            alpha_m = np.clip(alpha_m, 0.001, 0.5)
            alpha_m_seq[t] = alpha_m
            alpha_joint_live_seq[t] = self.alpha_joint_live

            # 4) Compute Base Matrix and Optimize Σ
            if self.base_matrix_mode == "sigma_T":
                base_anchor = sigma_T
            elif self.base_matrix_mode == "alpaca_1sd":
                # Strategy 1: ALPaCA provides the 1-SD base shape
                base_anchor = sig_alpaca.copy()
            elif self.base_matrix_mode == "alpaca_1alpha":
                # Strategy 2: ALPaCA 1-alpha ellipsoid IS the base shape
                chi2_scale_base = chi2.ppf(1 - alpha_m, df=d)
                base_anchor = chi2_scale_base * sig_alpaca.copy()
            else:
                base_anchor = sigma_T

            try:
                residuals = (ctx - mu_pred) if t >= self.n_context \
                    else np.zeros((0, d))
                sigma_opt = optimize_sigma(
                    base_anchor, self.lam, d, alpha_m, residuals,
                    prob_method=self.prob_method,
                    prob_weight=self.prob_weight,
                )
            except Exception:
                sigma_opt = base_anchor.copy()
                
            sigma_opt_seq[t] = sigma_opt.copy()

            # 5) Final ellipsoid
            if self.base_matrix_mode == "alpaca_1alpha":
                # The chi2 scale was already baked into the base shape
                final_sigma = self.scale_t * sigma_opt
            else:
                chi2_scale = chi2.ppf(1 - alpha_m, df=d)
                final_sigma = self.scale_t * chi2_scale * sigma_opt

            # 6) Coverage check
            residual_t = eps - mu_pred
            try:
                mahal = residual_t @ np.linalg.solve(final_sigma, residual_t)
            except np.linalg.LinAlgError:
                mahal = np.inf
            mahal_seq[t] = mahal
            covered = float(mahal < 1.0)
            coverage_seq[t] = covered
            err_t = 1.0 - covered
            violation_seq[t] = err_t

            volume_seq[t] = max(np.linalg.det(final_sigma), 1e-12)
            scale_seq[t] = self.scale_t

            # 7) Update overlap table (§6): for each lag Δ ∈ {1,…,K}
            for delta in range(1, min(self.K + 1, t + 1)):
                Y = violation_seq[t - delta] * violation_seq[t]
                self.overlap_table.update(delta, Y)

            # 8) PID update on budget using trajectory-level errors (§8)
            # Evaluate trajectory violation E_k for the window ending at t
            if (t + 1) % self.N == 0 and t >= self.N - 1:
                window_start = t - self.N + 1
                window_violations = violation_seq[window_start:t + 1]
                E_k = float(np.any(window_violations > 0))
                traj_violation_seq[t] = E_k
                
                # PID tracks the joint target
                err_traj = self.alpha_target - E_k
                self.integrator_sum += err_traj
                self.integrator_sum = saturation_fn(self.integrator_sum, t, self.C_sat)
                
                P_term = self.eta_joint * err_traj
                I_term = self.K_I * self.eta_joint * self.integrator_sum
                new_budget = self.alpha_joint_live + P_term + I_term
            else:
                new_budget = self.alpha_joint_live
            
            if np.isnan(new_budget) or np.isinf(new_budget):
                new_budget = self.alpha_target
            self.alpha_joint_live = np.clip(new_budget, 0.001, 0.999)

            # 9) Update Σ_T expert pool & scale (shared with v1)
            chi2_scale_for_update = chi2.ppf(1 - alpha_m, df=d)
            self.sigma_ctrl.update(residual_t, err_t,
                                   self.scale_t * chi2_scale_for_update)

            new_scale = (self.scale_t
                         + self.scale_gamma * (err_t - self.alpha_target))
            if np.isnan(new_scale) or np.isinf(new_scale):
                new_scale = 1.0
            self.scale_t = np.clip(new_scale, 0.1, 50.0)

            # Progress
            if verbose and (t + 1) % 200 == 0:
                rc = coverage_seq[max(0, t - 199):t + 1].mean()
                print(f"  [v2] Step {t+1}/{T} | cov={rc:.3f} | "
                      f"α_m={alpha_m:.4f} | α_joint={self.alpha_joint_live:.4f} | "
                      f"corr={correction:.4f} | det={np.linalg.det(sigma_opt):.4e}")

        results = {
            "coverage_seq": coverage_seq,
            "alpha_seq": alpha_m_seq,        # per-step allocated α
            "alpha_joint_live_seq": alpha_joint_live_seq,
            "correction_seq": correction_seq,
            "scale_seq": scale_seq,
            "volume_seq": volume_seq,
            "mahal_seq": mahal_seq,
            "mu_pred_seq": mu_pred_seq,
            "sigma_T_seq": sigma_T_seq,
            "sigma_opt_seq": sigma_opt_seq,
            "violation_seq": violation_seq,
            "traj_violation_seq": traj_violation_seq,
        }

        if verbose:
            final_cov = coverage_seq.mean()
            print(f"\n  [v2] Final coverage: {final_cov:.4f} "
                  f"(target: {1 - self.alpha_target:.4f})")
            print(self.overlap_table.summary())

        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Trajectory-Level ALPaCA + DtACI v2 — Quick Test (2D)")
    print("=" * 60)

    Sigma1 = np.array([[4.0, 0.5], [0.5, 0.3]])
    Sigma2 = np.array([[0.3, -0.2], [-0.2, 4.0]])

    T = 2000
    errors = generate_multidim_regime_shift(T, 500, Sigma1, Sigma2)

    system = TrajectoryAlpacaDtACI(
        d=2, alpha_target=0.1, N=10, K=5,
        n_context=10, lam=5.0,
        alpha_gamma=0.05, scale_gamma=0.05,
        warmup=100, alpaca_train_iters=200,
        alpaca_kwargs={"nn_layers": [64, 32], "lr": 0.001, "sigma_eps": 0.1},
    )

    results = system.run(errors, verbose=True)
    print(f"\nDone. Coverage = {results['coverage_seq'].mean():.4f}")
