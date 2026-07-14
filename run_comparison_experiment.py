#!/usr/bin/env python3
"""
Comparison Experiment: v1 (per-step DtACI) vs v2 (trajectory-level)
====================================================================

Runs both systems on the same data and produces side-by-side plots:
  - Coverage rolling average
  - Alpha trajectories
  - Ellipsoid snapshots (with volume annotations)
  - Volume comparison
  - Scale trajectories
  - Correction size (v2 only)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import datetime

from integrated_alpaca_dtaci import IntegratedAlpacaDtACI
from integrated_alpaca_dtaci_v2 import TrajectoryAlpacaDtACI
from scripts.generate_drone_tracking import generate_drone_trajectory
from scripts.generate_useful_plots import rolling_window, rolling_std


def plot_ellipse_pts(ax, mu, sigma, color="blue", alpha=0.3, label=None):
    """Draw a 2D ellipse from mu and covariance sigma."""
    try:
        eigvals, eigvecs = np.linalg.eigh(sigma)
        eigvals = np.maximum(eigvals, 1e-10)
        width, height = 2 * np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = Ellipse(
            xy=mu, width=width, height=height, angle=angle,
            facecolor=color, edgecolor=color, alpha=alpha, label=label,
        )
        ax.add_patch(ell)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARISON PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(
    tracking_errors, r1, r2, r2b, shift_t, output_dir, mode="dtaci",
    title_prefix="Multirotor", window=200
):
    """Generate all side-by-side comparison plots."""

    T = len(tracking_errors)
    os.makedirs(output_dir, exist_ok=True)

    c_v1, c_v2, c_v2b = "#2196F3", "#FF5722", "#4CAF50"
    
    if mode == "alpaca":
        label2 = "v2 (ALPaCA 1-α base)"
        label2b = "v2 (ALPaCA 1-SD base)"
        title_extra = "(v1 vs ALPaCA Base Matrices)"
    else:
        label2 = "v2 (Joint + Overlap)"
        label2b = "v2 (Joint Bonferroni)"
        title_extra = "(v1 vs v2 Overlap vs v2 Bonferroni)"

    # ─────────────────────────────────────────────
    # 1. Coverage rolling average
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    cov1 = rolling_window(r1["coverage_seq"], window)
    cov2 = rolling_window(r2["coverage_seq"], window)
    cov2b = rolling_window(r2b["coverage_seq"], window)
    ax.plot(cov1, color=c_v1, label="v1 (Marginal)")
    ax.plot(cov2, color=c_v2, label=label2)
    ax.plot(cov2b, color=c_v2b, label=label2b)
    ax.axhline(0.9, color="black", ls="--", alpha=0.4, label="90% target")
    ax.axvline(shift_t, color="red", ls=":", lw=2, label="Regime shift")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Rolling Coverage")
    ax.set_title(f"{title_prefix}: Coverage Comparison (window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cmp_coverage.png"), dpi=150)
    plt.close()

    # ─────────────────────────────────────────────
    # 2. Alpha trajectories
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(r1["alpha_seq"], color=c_v1, alpha=0.6, label="v1 α_t")
    ax.plot(r2["alpha_seq"], color=c_v2, alpha=0.6, label=label2 + " α_m")
    ax.plot(r2b["alpha_seq"], color=c_v2b, alpha=0.6, label=label2b + " α_m")
    if "alpha_joint_live_seq" in r2:
        ax.plot(r2["alpha_joint_live_seq"], color=c_v2, ls="--",
                alpha=0.5, label="v2 α_joint^live")
    ax.axhline(0.1, color="black", ls=":", alpha=0.3)
    ax.axvline(shift_t, color="red", ls=":", lw=2, label="Regime shift")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("α")
    ax.set_title(f"{title_prefix}: Alpha Trajectory Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cmp_alpha.png"), dpi=150)
    plt.close()

    # ─────────────────────────────────────────────
    # 3. Volume comparison (log scale)
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax = axes[0]
    v1_vol = np.log10(np.maximum(r1["volume_seq"], 1e-20))
    v2_vol = np.log10(np.maximum(r2["volume_seq"], 1e-20))
    v2b_vol = np.log10(np.maximum(r2b["volume_seq"], 1e-20))
    ax.plot(v1_vol, color=c_v1, alpha=0.5, label="v1 log₁₀(vol)")
    ax.plot(v2_vol, color=c_v2, alpha=0.5, label=label2 + " log₁₀(vol)")
    ax.plot(v2b_vol, color=c_v2b, alpha=0.5, label=label2b + " log₁₀(vol)")
    ax.axvline(shift_t, color="red", ls=":", lw=2)
    ax.set_ylabel("log₁₀(Volume)")
    ax.set_title(f"{title_prefix}: Ellipsoid Volume Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ratio_a = v2_vol - v2b_vol
    ratio_b = v2_vol - v1_vol
    ax.plot(ratio_a, color=c_v2, alpha=0.6, label="v2 / v2_bonf (Overlap improvement)")
    ax.plot(ratio_b, color="black", alpha=0.4, label="v2 / v1 ratio")
    ax.axhline(0, color="black", ls="--", alpha=0.3)
    ax.axvline(shift_t, color="red", ls=":", lw=2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("log₁₀ Ratio")
    ax.set_title("Volume Ratio (negative = v2 smaller = better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cmp_volume.png"), dpi=150)
    plt.close()

    # ─────────────────────────────────────────────
    # 4. Scale trajectories
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(r1["scale_seq"], color=c_v1, alpha=0.6, label="v1 scale_t")
    ax.plot(r2["scale_seq"], color=c_v2, alpha=0.6, label=label2 + " scale_t")
    ax.plot(r2b["scale_seq"], color=c_v2b, alpha=0.6, label=label2b + " scale_t")
    ax.axvline(shift_t, color="red", ls=":", lw=2, label="Regime shift")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Scale Factor")
    ax.set_title(f"{title_prefix}: Scale Factor Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cmp_scale.png"), dpi=150)
    plt.close()

    # ─────────────────────────────────────────────
    # 5. Ellipsoid snapshots (CHUNKED OUTPUT)
    # ─────────────────────────────────────────────

    import math

    start_t, end_t = 900, 1900
    step = 5
    chunk_size = 50  # each PNG spans 50 timesteps

    all_times = list(range(start_t, end_t, step))
    all_times = [t for t in all_times if 0 <= t < T]

    warmup_pts = max(50, min(100, T // 4))

    chunks = [
        all_times[i:i + chunk_size // step]
        for i in range(0, len(all_times), chunk_size // step)
    ]

    ell_dir = os.path.join(output_dir, "cmp_ellipsoids")
    os.makedirs(ell_dir, exist_ok=True)

    for chunk_idx, snapshot_times in enumerate(chunks):

        if len(snapshot_times) == 0:
            continue

        fig, axes = plt.subplots(
            3, len(snapshot_times),
            figsize=(2.2 * len(snapshot_times), 12)
        )

        if len(snapshot_times) == 1:
            axes = axes.reshape(3, 1)

        for col, t in enumerate(snapshot_times):

            for row, (res, label, color) in enumerate([
                (r1, "v1 (Marginal)", c_v1),
                (r2, label2, c_v2),
                (r2b, label2b, c_v2b),
            ]):

                ax = axes[row, col]

                recent = tracking_errors[max(0, t - 30):t + 1]

                if recent.shape[0] > 0:
                    if "Stock" in title_prefix:
                        ax.scatter(recent[:, 0], recent[:, 1],
                                   color="gray", alpha=0.4, s=10)
                    else:
                        ax.plot(recent[:, 0], recent[:, 1],
                                color="gray", alpha=0.4, lw=1)

                ax.scatter(
                    tracking_errors[t, 0],
                    tracking_errors[t, 1],
                    color="black",
                    s=40,
                    zorder=5
                )

                sigma_opt = res["sigma_opt_seq"][t]
                mu = res["mu_pred_seq"][t]

                if sigma_opt is not None:
                    alpha_val = res["alpha_seq"][t]
                    s = res["scale_seq"][t]

                    chi2_s = chi2.ppf(
                        1 - np.clip(alpha_val, 0.001, 0.999),
                        df=tracking_errors.shape[1]
                    )

                    final = s * chi2_s * sigma_opt
                    plot_ellipse_pts(ax, mu, final, color=color, alpha=0.25)

                    vol = max(np.linalg.det(final), 1e-12)
                    ax.set_xlabel(f"{vol:.2e}", fontsize=8)

                ax.set_title(f"{label} t={t}", fontsize=9)

                # auto bounds
                pad = 0.1
                if sigma_opt is not None:
                    rx = np.sqrt(final[0, 0])
                    ry = np.sqrt(final[1, 1])

                    min_x = min(recent[:, 0].min(), mu[0] - rx)
                    max_x = max(recent[:, 0].max(), mu[0] + rx)
                    min_y = min(recent[:, 1].min(), mu[1] - ry)
                    max_y = max(recent[:, 1].max(), mu[1] + ry)
                else:
                    min_x, max_x = recent[:, 0].min(), recent[:, 0].max()
                    min_y, max_y = recent[:, 1].min(), recent[:, 1].max()

                ax.set_xlim(min_x - pad, max_x + pad)
                ax.set_ylim(min_y - pad, max_y + pad)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.2)

        fig.suptitle(
            f"{title_prefix}: Ellipsoids {snapshot_times[0]}–{snapshot_times[-1]}",
            fontsize=14
        )

        plt.tight_layout()

        out_path = os.path.join(
            ell_dir,
            f"ellipsoids_{snapshot_times[0]}_{snapshot_times[-1]}.png"
        )

        plt.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close()

    # ─────────────────────────────────────────────
    # 6. Correction size (v2 only)
    # ─────────────────────────────────────────────
    if "correction_seq" in r2:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(r2["correction_seq"], color=c_v2, alpha=0.6, label="v2 correction")
        ax.axvline(shift_t, color="red", ls=":", lw=2, label="Regime shift")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Forest Correction Σ ω^safe")
        ax.set_title(f"{title_prefix}: Overlap Correction (v2)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cmp_correction.png"), dpi=150)
        plt.close()

    # ─────────────────────────────────────────────
    # 7. Summary stats text file
    # ─────────────────────────────────────────────
    v1_cov = r1["coverage_seq"].mean()
    v2a_cov = r2["coverage_seq"].mean()
    v2b_cov = r2b["coverage_seq"].mean()
    v1_vol_mean = np.mean(v1_vol)
    v2a_vol_mean = np.mean(v2_vol)
    v2b_vol_mean = np.mean(v2b_vol)

    summary = (
        f"{'='*50}\n"
        f"  {title_prefix} Comparison Summary\n"
        f"{'='*50}\n"
        f"  v1  coverage:       {v1_cov:.4f}\n"
        f"  v2  (overlap) cov:  {v2a_cov:.4f}\n"
        f"  v2_bonf coverage:   {v2b_cov:.4f}\n"
        f"  v1  mean log₁₀(vol): {v1_vol_mean:.4f}\n"
        f"  v2  mean log₁₀(vol): {v2a_vol_mean:.4f}\n"
        f"  v2_b mean log₁₀(vol): {v2b_vol_mean:.4f}\n"
        f"  Overlap vs Bonf vol reduction:  {v2b_vol_mean - v2a_vol_mean:.4f} (pos=smaller)\n"
        f"  v2 vs v1 vol diff (v2 expected larger): {v2a_vol_mean - v1_vol_mean:.4f}\n"
        f"{'='*50}\n"
    )
    print(summary)
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary)


# ═══════════════════════════════════════════════════════════════════════════════
#  ROBOTICS EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_robotics_comparison(mode="dtaci"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "results", f"{timestamp}_v1_vs_v2_{mode}_robotics")

    print("=" * 70)
    print(f"  V1 vs V2 COMPARISON — MULTIROTOR TRACKING (Mode: {mode})")
    print(f"  Output: {output_dir}")
    print("=" * 70 + "\n")

    T = 2000
    shift_t = 1000
    positions, references = generate_drone_trajectory(T=T, shift_t=shift_t)
    tracking_errors = positions - references

    lam = 1.0
    shared_kwargs = dict(
        d=2, alpha_target=0.1, n_context=15, lam=lam,
        alpha_gamma=0.01, scale_gamma=0.01,
        warmup=100, alpaca_train_iters=300,
        prob_method="moment", prob_weight=10.0,
        force_zero_mean=True,
        alpaca_kwargs={"nn_layers": [128, 64], "lr": 0.001, "sigma_eps": 0.1},
    )

    # ── v1 ──
    print("  Running v1 (per-step DtACI) …")
    kwargs_v1 = shared_kwargs.copy()
    kwargs_v1["alpha_target"] = shared_kwargs["alpha_target"] / 10.0
    sys_v1 = IntegratedAlpacaDtACI(
        replay_buffer_size=0,
        base_matrix_mode="alpaca_1sd",
        **kwargs_v1,
    )
    r1 = sys_v1.run(tracking_errors, verbose=True)

    if mode == "alpaca":
        # ── v2 (ALPaCA 1-α base) ──
        print(f"\n  Running v2 (trajectory-level, ALPaCA 1-α base) …")
        kwargs_v2a = shared_kwargs.copy()
        kwargs_v2a["lam"] = 0.5
        kwargs_v2a["scale_gamma"] = 0.01
        sys_v2a = TrajectoryAlpacaDtACI(
            N=10, K=5, delta_conf=0.05, eta_joint=0.01,
            base_matrix_mode="alpaca_1alpha",
            **kwargs_v2a,
        )
        r2 = sys_v2a.run(tracking_errors, verbose=True)

        # ── v2b (ALPaCA 1-SD base) ──
        print(f"\n  Running v2b (trajectory-level, ALPaCA 1-SD base) …")
        kwargs_v2b = shared_kwargs.copy()
        kwargs_v2b["lam"] = 0.5
        kwargs_v2b["scale_gamma"] = 0.01
        sys_v2b = TrajectoryAlpacaDtACI(
            N=10, K=5, delta_conf=0.05, eta_joint=0.01,
            base_matrix_mode="alpaca_1sd",
            **kwargs_v2b,
        )
        r2b = sys_v2b.run(tracking_errors, verbose=True)
    else:
        # ── v2 (Joint PID + Overlap) ──
        print("\n  Running v2 (Joint + Overlap, ALPaCA 1-SD base) …")
        kwargs_v2a = shared_kwargs.copy()
        kwargs_v2a["lam"] = 0.5
        kwargs_v2a["scale_gamma"] = 0.01
        sys_v2a = TrajectoryAlpacaDtACI(
            N=10, K=5, delta_conf=0.05, eta_joint=0.01,
            base_matrix_mode="alpaca_1sd",
            **kwargs_v2a,
        )
        r2 = sys_v2a.run(tracking_errors, verbose=True)

        # ── v2_bonf (Static Bonferroni + Overlap) ──
        print("\n  Running v2_bonf (Joint Bonferroni, ALPaCA 1-SD base, NO overlap) …")
        kwargs_v2b = shared_kwargs.copy()
        kwargs_v2b["lam"] = 0.5
        kwargs_v2b["scale_gamma"] = 0.01
        # Set eta_joint=0.0 to disable PID (Static)
        sys_v2b = TrajectoryAlpacaDtACI(
            N=10, K=0, delta_conf=0.05, eta_joint=0.01,
            base_matrix_mode="alpaca_1sd",
            **kwargs_v2b,
        )
        r2b = sys_v2b.run(tracking_errors, verbose=True)

    plot_comparison(tracking_errors, r1, r2, r2b, shift_t, output_dir, mode=mode,
                    title_prefix="Multirotor")

    print(f"\n✓ Saved comparison plots to {output_dir}")
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  STOCKS EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_stocks_comparison(mode="dtaci"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "results", f"{timestamp}_v1_vs_v2_{mode}_stocks")

    print("=" * 70)
    print(f"  V1 vs V2 COMPARISON — SIMULATED 2D STOCK RETURNS (Mode: {mode})")
    print(f"  Output: {output_dir}")
    print("=" * 70 + "\n")

    Sigma1 = np.array([[4.0, 0.5], [0.5, 0.3]])
    Sigma2 = np.array([[0.3, -0.2], [-0.2, 4.0]])
    T = 2000
    shift_t = 1000

    from integrated_alpaca_dtaci import generate_multidim_regime_shift
    errors = generate_multidim_regime_shift(T, shift_t, Sigma1, Sigma2)

    lam = 1.0
    shared_kwargs = dict(
        d=2, alpha_target=0.1, n_context=10, lam=lam,
        alpha_gamma=0.01, scale_gamma=0.01,
        warmup=100, alpaca_train_iters=200,
        prob_method="moment", prob_weight=10.0,
        alpaca_kwargs={"nn_layers": [64, 32], "lr": 0.001, "sigma_eps": 0.1},
    )

    print("  Running v1 (per-step DtACI) …")
    kwargs_v1 = shared_kwargs.copy()
    kwargs_v1["alpha_target"] = shared_kwargs["alpha_target"] / 10.0
    sys_v1 = IntegratedAlpacaDtACI(
        replay_buffer_size=0,
        base_matrix_mode="alpaca_1sd",
        **kwargs_v1,
    )
    r1 = sys_v1.run(errors, verbose=True)

    if mode == "alpaca":
        print("\n  Running v2 (trajectory-level, ALPaCA 1-α base) …")
        kwargs_v2a = shared_kwargs.copy()
        kwargs_v2a["lam"] = 0.5
        kwargs_v2a["scale_gamma"] = 0.01
        sys_v2a = TrajectoryAlpacaDtACI(
            N=10, K=5, delta_conf=0.05, eta_joint=0.01,
            base_matrix_mode="alpaca_1alpha",
            **kwargs_v2a,
        )
        r2 = sys_v2a.run(errors, verbose=True)

        print("\n  Running v2b (trajectory-level, ALPaCA 1-SD base) …")
        kwargs_v2b = shared_kwargs.copy()
        kwargs_v2b["lam"] = 0.5
        kwargs_v2b["scale_gamma"] = 0.01
        sys_v2b = TrajectoryAlpacaDtACI(
            N=10, K=5, delta_conf=0.05, eta_joint=0.01,
            base_matrix_mode="alpaca_1sd",
            **kwargs_v2b,
        )
        r2b = sys_v2b.run(errors, verbose=True)
    else:
        print("\n  Running v2 (Joint + Overlap, ALPaCA 1-SD base) …")
        kwargs_v2a = shared_kwargs.copy()
        kwargs_v2a["lam"] = 0.5
        kwargs_v2a["scale_gamma"] = 0.01
        sys_v2a = TrajectoryAlpacaDtACI(
            N=10, K=5, delta_conf=0.05, eta_joint=0.01,
            base_matrix_mode="alpaca_1sd",
            **kwargs_v2a,
        )
        r2 = sys_v2a.run(errors, verbose=True)

        print("\n  Running v2_bonf (Joint Bonferroni, ALPaCA 1-SD base, NO overlap) …")
        kwargs_v2b = shared_kwargs.copy()
        kwargs_v2b["lam"] = 0.5
        kwargs_v2b["scale_gamma"] = 0.01
        sys_v2b = TrajectoryAlpacaDtACI(
            N=10, K=0, delta_conf=0.05, eta_joint=0.01,
            base_matrix_mode="alpaca_1sd",
            **kwargs_v2b,
        )
        r2b = sys_v2b.run(errors, verbose=True)

    plot_comparison(errors, r1, r2, r2b, shift_t, output_dir, mode=mode,
                    title_prefix="Simulated 2D Stock")

    print(f"\n✓ Saved comparison plots to {output_dir}")
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run v1 vs v2 comparison experiments",
    )
    parser.add_argument(
        "--experiment", choices=["robotics", "stocks", "both"],
        default="both",
        help="Which experiment to run (default: both)",
    )
    parser.add_argument(
        "--mode", choices=["dtaci", "alpaca"], default="dtaci",
        help="Which systems to compare (dtaci vs granular alpaca diagnosis)"
    )
    args = parser.parse_args()

    if args.experiment in ["robotics", "both"]:
        run_robotics_comparison(mode=args.mode)
    if args.experiment in ["stocks", "both"]:
        run_stocks_comparison(mode=args.mode)
