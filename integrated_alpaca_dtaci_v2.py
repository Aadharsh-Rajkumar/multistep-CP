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
from collections import deque

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

    def __init__(self, K, delta_conf=0.05, c_bins=None, alpha_bins=None):
        """
        Args:
            K:          maximum lag to track (Δt ∈ {1, …, K})
            delta_conf: confidence budget for Hoeffding margins
        """
        self.K = K
        self.delta_conf = delta_conf
        self.c_bins = np.array(
            c_bins if c_bins is not None
            else [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf],
            dtype=float,
        )
        self.alpha_bins = np.array(
            alpha_bins if alpha_bins is not None
            else [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf],
            dtype=float,
        )
        self.stats = {}
        self.lag_stats = {}

    def _bin_value(self, value, bins):
        idx = np.searchsorted(bins, float(value), side="right") - 1
        return int(np.clip(idx, 0, len(bins) - 2))

    def make_key(self, c_s, c_t, delta_t, alpha_s, alpha_t):
        """Create a discretized overlap key."""
        if delta_t < 1 or delta_t > self.K:
            return None
        return (
            self._bin_value(c_s, self.c_bins),
            self._bin_value(c_t, self.c_bins),
            int(delta_t),
            self._bin_value(alpha_s, self.alpha_bins),
            self._bin_value(alpha_t, self.alpha_bins),
        )

    def _update_stats(self, table, key, Y):
        entry = table.get(key)
        if entry is None:
            entry = {"n": 0, "omega_hat": 0.0, "M2": 0.0}
            table[key] = entry
        entry["n"] += 1
        old_mean = entry["omega_hat"]
        entry["omega_hat"] += (Y - entry["omega_hat"]) / entry["n"]
        entry["M2"] += (Y - old_mean) * (Y - entry["omega_hat"])

    def update(self, c_s, c_t, delta_t, alpha_s, alpha_t, Y):
        """Update the binned overlap table and lag-only fallback table."""
        key = self.make_key(c_s, c_t, delta_t, alpha_s, alpha_t)
        if key is None:
            return
        self._update_stats(self.stats, key, Y)
        self._update_stats(self.lag_stats, int(delta_t), Y)

    def _get_entry(self, c_s, c_t, delta_t, alpha_s, alpha_t):
        key = self.make_key(c_s, c_t, delta_t, alpha_s, alpha_t)
        if key is None:
            return None
        entry = self.stats.get(key)
        if entry is not None:
            return entry
        return self.lag_stats.get(int(delta_t))

    def get_variance(self, c_s, c_t, delta_t, alpha_s, alpha_t):
        """Empirical variance for a queried overlap cell."""
        entry = self._get_entry(c_s, c_t, delta_t, alpha_s, alpha_t)
        if entry is None or entry["n"] < 2:
            return 0.25
        return entry["M2"] / max(1, entry["n"] - 1)

    def get_hat_omega(self, c_s, c_t, delta_t, alpha_s, alpha_t):
        """Empirical overlap estimate ω̂ for the requested cell."""
        entry = self._get_entry(c_s, c_t, delta_t, alpha_s, alpha_t)
        if entry is None:
            return 0.0
        return float(entry["omega_hat"])

    def get_safe_omega(self, c_s, c_t, delta_t, alpha_s, alpha_t):
        """
        Conservative lower bound:  ω^safe = max{0, ω̂ − M_H}.

        Hoeffding margin: M_H = sqrt( log(K/δ) / (2n) )
        """
        entry = self._get_entry(c_s, c_t, delta_t, alpha_s, alpha_t)
        if entry is None or entry["n"] < 2:
            return 0.0
        delta_j = self.delta_conf / max(1, self.K)
        M = np.sqrt(np.log(1.0 / delta_j) / (2.0 * entry["n"]))
        return max(0.0, float(entry["omega_hat"]) - M)

    def get_omega(self, c_s, c_t, delta_t, alpha_s, alpha_t, mode="safe"):
        """Query either raw ω̂ or conservative ω^safe."""
        if mode == "raw":
            return self.get_hat_omega(c_s, c_t, delta_t, alpha_s, alpha_t)
        return self.get_safe_omega(c_s, c_t, delta_t, alpha_s, alpha_t)

    def summary(self):
        """Print a diagnostic summary of the table."""
        lines = ["  OverlapTable summary (lag → ω̂, ω^safe, n):"]
        for dt in range(1, self.K + 1):
            entry = self.lag_stats.get(dt, {"n": 0, "omega_hat": 0.0, "M2": 0.0})
            ws = self.get_safe_omega(1.0, 1.0, dt, 0.01, 0.01)
            lines.append(
                f"    Δt={dt:2d}  ω̂={entry['omega_hat']:.4f}  "
                f"ω^safe={ws:.4f}  n={entry['n']}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAX-WEIGHT FOREST  (Kounias–Hunter)
# ═══════════════════════════════════════════════════════════════════════════════

def max_weight_forest(N, edge_weights):
    """
    Select a maximum-weight forest over lag-truncated edges using Kruskal's.

    For horizon {1,…,N}, candidate edges are any (s,t) pairs present in
    edge_weights with positive weight.

    Returns:
        forest:     list of (s,t) edges in the forest
        correction: total weight  Σ_{(s,t)∈F} ω^safe
    """
    edges = [
        (w, s, t)
        for (s, t), w in edge_weights.items()
        if 1 <= s < t <= N and w > 0
    ]

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
                 base_matrix_mode="sigma_T",
                 overlap_mode="safe"):
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
        self.overlap_mode = overlap_mode

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
        self.history_buffer = deque()
        self.trajectory_buffer = deque()

    def _scalar_multiplier(self, alpha_val):
        """Return the scalar c used to scale the planned base ellipsoid."""
        if self.base_matrix_mode == "alpaca_1alpha":
            return self.scale_t
        return self.scale_t * chi2.ppf(1 - alpha_val, df=self.d)

    def _build_edge_weights(self, horizon_len, alpha_guess):
        """Query overlap weights on the current warm-start plan."""
        if self.K <= 0 or horizon_len <= 1:
            return {}

        c_guess = self._scalar_multiplier(alpha_guess)
        edge_weights = {}
        for s in range(1, horizon_len + 1):
            for t in range(s + 1, min(horizon_len, s + self.K) + 1):
                delta_t = t - s
                edge_weights[(s, t)] = self.overlap_table.get_omega(
                    c_guess,
                    c_guess,
                    delta_t,
                    alpha_guess,
                    alpha_guess,
                    mode=self.overlap_mode,
                )
        return edge_weights

    def _plan_horizon(self, errors, t_start, sigma_T, alpha_m):
        """Build the planned horizon ellipsoids from the information available at t_start."""
        T = len(errors)
        horizon_len = min(self.N, T - t_start)
        if horizon_len <= 0:
            return []

        d = self.d
        ctx_hist = errors[max(0, t_start - self.n_context):t_start].copy()
        pred_context = [row.copy() for row in ctx_hist]
        plans = []
        shared_sigma_opt = None

        for h in range(horizon_len):
            if len(pred_context) >= self.n_context:
                ctx = np.array(pred_context[-self.n_context:])
                mu_pred, sig_alpaca = self._alpaca_predict(ctx)
            else:
                ctx = np.array(pred_context[-self.n_context:]) if len(pred_context) > 0 else np.zeros((0, d))
                mu_pred = np.zeros(d)
                sig_alpaca = np.eye(d)

            if self.base_matrix_mode == "sigma_T":
                base_anchor = sigma_T
            elif self.base_matrix_mode == "alpaca_1sd":
                base_anchor = sig_alpaca.copy()
            elif self.base_matrix_mode == "alpaca_1alpha":
                chi2_scale_base = chi2.ppf(1 - alpha_m, df=d)
                base_anchor = chi2_scale_base * sig_alpaca.copy()
            else:
                base_anchor = sigma_T

            if shared_sigma_opt is None:
                try:
                    residuals = (ctx - mu_pred) if ctx.shape[0] > 0 else np.zeros((0, d))
                    shared_sigma_opt = optimize_sigma(
                        base_anchor,
                        self.lam,
                        d,
                        alpha_m,
                        residuals,
                        prob_method=self.prob_method,
                        prob_weight=self.prob_weight,
                    )
                except Exception:
                    shared_sigma_opt = base_anchor.copy()
            sigma_opt = shared_sigma_opt.copy()

            if self.base_matrix_mode == "alpaca_1alpha":
                c_scalar = self.scale_t
                final_sigma = c_scalar * sigma_opt
            else:
                c_scalar = self.scale_t * chi2.ppf(1 - alpha_m, df=d)
                final_sigma = c_scalar * sigma_opt

            plans.append({
                "h": h + 1,
                "abs_time": t_start + h,
                "mu_pred": mu_pred.copy(),
                "sigma_opt": sigma_opt.copy(),
                "final_sigma": final_sigma.copy(),
                "alpha": alpha_m,
                "c_scalar": c_scalar,
                "scale_t": self.scale_t,
            })

            pred_context.append(mu_pred.copy())

        return plans

    def _push_planned_records(self, plan_time, horizon_plans):
        """Store planned pair events and trajectory records until their observations resolve."""
        horizon_len = len(horizon_plans)
        if horizon_len == 0:
            return

        if self.K > 0:
            for i in range(horizon_len):
                step_s = horizon_plans[i]
                for j in range(i + 1, min(horizon_len, i + self.K + 1)):
                    step_t = horizon_plans[j]
                    self.history_buffer.append({
                        "plan_time": plan_time,
                        "s": step_s["h"],
                        "t": step_t["h"],
                        "abs_s": step_s["abs_time"],
                        "abs_t": step_t["abs_time"],
                        "c_s": step_s["c_scalar"],
                        "c_t": step_t["c_scalar"],
                        "alpha_s": step_s["alpha"],
                        "alpha_t": step_t["alpha"],
                        "mu_s": step_s["mu_pred"].copy(),
                        "mu_t": step_t["mu_pred"].copy(),
                        "Sigma_s": step_s["final_sigma"].copy(),
                        "Sigma_t": step_t["final_sigma"].copy(),
                    })

        self.trajectory_buffer.append({
            "plan_time": plan_time,
            "final_abs": horizon_plans[-1]["abs_time"],
            "steps": [{
                "abs_time": step["abs_time"],
                "mu_pred": step["mu_pred"].copy(),
                "final_sigma": step["final_sigma"].copy(),
            } for step in horizon_plans],
        })

    def _resolve_overlap_records(self, errors, t_now):
        """Resolve all buffered overlap samples whose second timestamp is now observable."""
        unresolved = deque()
        while self.history_buffer:
            record = self.history_buffer.popleft()
            if record["abs_t"] > t_now:
                unresolved.append(record)
                continue

            eps_s = errors[record["abs_s"]] - record["mu_s"]
            eps_t = errors[record["abs_t"]] - record["mu_t"]
            try:
                V_s = float(eps_s @ np.linalg.solve(record["Sigma_s"], eps_s) >= 1.0)
            except np.linalg.LinAlgError:
                V_s = 1.0
            try:
                V_t = float(eps_t @ np.linalg.solve(record["Sigma_t"], eps_t) >= 1.0)
            except np.linalg.LinAlgError:
                V_t = 1.0

            self.overlap_table.update(
                record["c_s"],
                record["c_t"],
                record["t"] - record["s"],
                record["alpha_s"],
                record["alpha_t"],
                V_s * V_t,
            )
        self.history_buffer = unresolved

    def _resolve_trajectory_records(self, errors, t_now):
        """Resolve any full planned trajectories that have completely materialized."""
        resolved = []
        unresolved = deque()
        while self.trajectory_buffer:
            record = self.trajectory_buffer.popleft()
            if record["final_abs"] > t_now:
                unresolved.append(record)
                continue

            traj_violation = 0.0
            for step in record["steps"]:
                residual = errors[step["abs_time"]] - step["mu_pred"]
                try:
                    missed = float(
                        residual @ np.linalg.solve(step["final_sigma"], residual) >= 1.0
                    )
                except np.linalg.LinAlgError:
                    missed = 1.0
                traj_violation = max(traj_violation, missed)
            resolved.append((record["plan_time"], traj_violation))
        self.trajectory_buffer = unresolved
        return resolved

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
            alpha_horizon_seq:     per-horizon α allocation at each step
            alpha_joint_live_seq:  trajectory budget over time
            alpha_total_seq:       sum of horizon α after overlap subtraction
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
        alpha_horizon_seq = np.zeros((T, self.N))
        alpha_joint_live_seq = np.zeros(T)
        alpha_total_seq = np.zeros(T)
        correction_seq = np.zeros(T)
        scale_seq = np.zeros(T)
        volume_seq = np.zeros(T)
        mahal_seq = np.zeros(T)
        mu_pred_seq = np.zeros((T, d))
        sigma_T_seq = [None] * T
        sigma_opt_seq = [None] * T
        violation_seq = np.zeros(T)       # V_t per step
        traj_violation_seq = np.zeros(T)   # per-window trajectory indicator
        overlap_mode_seq = [self.overlap_mode] * T

        # ── Main loop ──
        for t in range(T):
            eps = errors[t]
            sigma_T = self.sigma_ctrl.value
            sigma_T_seq[t] = sigma_T.copy()

            # 1) Query overlap table on a frozen warm-start plan
            horizon_len = min(self.N, T - t)
            alpha_guess = np.clip(self.alpha_joint_live / max(1, horizon_len), 0.001, 0.5)
            edge_weights = self._build_edge_weights(horizon_len, alpha_guess)
            _, correction = max_weight_forest(horizon_len, edge_weights)
            correction_seq[t] = correction

            # 2) Compute per-step α_m using the joint target formula
            alpha_m = (self.alpha_joint_live + correction) / max(1, horizon_len)
            alpha_m = np.clip(alpha_m, 0.001, 0.5)
            alpha_m_seq[t] = alpha_m
            alpha_horizon_seq[t, :] = alpha_m
            alpha_joint_live_seq[t] = self.alpha_joint_live
            alpha_total_seq[t] = horizon_len * alpha_m - correction

            # 3) Build the horizon plan from the information available at t
            horizon_plans = self._plan_horizon(errors, t, sigma_T, alpha_m)
            if len(horizon_plans) == 0:
                continue
            current_plan = horizon_plans[0]
            mu_pred = current_plan["mu_pred"]
            sigma_opt = current_plan["sigma_opt"]
            final_sigma = current_plan["final_sigma"]
            mu_pred_seq[t] = mu_pred
            sigma_opt_seq[t] = sigma_opt.copy()

            # 4) Coverage check for the first planned step
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

            # 5) Store original planned sets, then resolve any completed records
            self._push_planned_records(t, horizon_plans)
            self._resolve_overlap_records(errors, t)
            resolved_trajs = self._resolve_trajectory_records(errors, t)

            # 6) PID update on resolved trajectory-level errors (§8)
            if len(resolved_trajs) > 0:
                plan_idx, E_k = resolved_trajs[-1]
                traj_violation_seq[plan_idx] = E_k
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

            # 7) Update Σ_T expert pool & scale (shared with v1)
            chi2_scale_for_update = chi2.ppf(1 - alpha_m, df=d)
            self.sigma_ctrl.update(residual_t, err_t,
                                   self.scale_t * chi2_scale_for_update,
                                   target=alpha_m)

            new_scale = (self.scale_t
                         + self.scale_gamma * (err_t - alpha_m))
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
            "alpha_horizon_seq": alpha_horizon_seq,
            "alpha_joint_live_seq": alpha_joint_live_seq,
            "alpha_total_seq": alpha_total_seq,
            "correction_seq": correction_seq,
            "scale_seq": scale_seq,
            "volume_seq": volume_seq,
            "mahal_seq": mahal_seq,
            "mu_pred_seq": mu_pred_seq,
            "sigma_T_seq": sigma_T_seq,
            "sigma_opt_seq": sigma_opt_seq,
            "violation_seq": violation_seq,
            "traj_violation_seq": traj_violation_seq,
            "overlap_mode_seq": overlap_mode_seq,
            "N": self.N,
            "K": self.K,
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
