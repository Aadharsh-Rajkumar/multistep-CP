#!/usr/bin/env python3
"""
Experiment Runner for Integrated ALPaCA + DtACI
================================================

Sweeps over multiple values of the regularization weight λ (x) in
    det(Σ) + λ · ‖Σ − Σ_T‖_F
and generates comparison plots.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba
import matplotlib.cm as cm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integrated_alpaca_dtaci import (
    IntegratedAlpacaDtACI,
    generate_multidim_regime_shift,
    optimize_sigma,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def ellipse_from_sigma(Sigma, center=(0, 0), n_std=1.0):
    """Create a matplotlib Ellipse patch from a 2x2 covariance matrix."""
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-10)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_std * np.sqrt(eigvals[0])
    height = 2 * n_std * np.sqrt(eigvals[1])
    return Ellipse(xy=center, width=width, height=height, angle=angle)


def plot_ellipse_points(Sigma, center=(0, 0), npts=100):
    """Return (x, y) arrays tracing the boundary of the ellipsoid."""
    angles = np.linspace(0, 2 * np.pi, npts)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
    pts = circle @ L.T
    return pts[:, 0] + center[0], pts[:, 1] + center[1]


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: Simulated 2D Regime Shift
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulated_experiment(output_dir, lambda_values=None, T=2000, gap=500):
    """Run on simulated 2D regime-shift data with different λ values."""

    os.makedirs(output_dir, exist_ok=True)

    # True covariance regimes
    Sigma1 = np.array([[4.0, 0.5],
                       [0.5, 0.3]])
    Sigma2 = np.array([[0.3, -0.2],
                       [-0.2, 4.0]])

    errors = generate_multidim_regime_shift(T, gap, Sigma1, Sigma2)

    if lambda_values is None:
        lambda_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    all_results = {}
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: Simulated 2D Regime Shift (T={T}, gap={gap})")
    print(f"  Lambda values: {lambda_values}")
    print(f"{'='*70}")

    for lam in lambda_values:
        print(f"\n{'─'*50}")
        print(f"  λ = {lam}")
        print(f"{'─'*50}")

        system = IntegratedAlpacaDtACI(
            d=2, alpha_target=0.1,
            n_context=10, lam=lam,
            alpha_gamma=0.05, scale_gamma=0.05,
            warmup=100, alpaca_train_iters=300,
            alpaca_kwargs={
                "nn_layers": [64, 32],
                "lr": 0.001,
                "sigma_eps": 0.1,
                "meta_batch_size": 10,
                "test_horizon": 5,
            }
        )

        results = system.run(errors, verbose=True)
        results["lambda"] = lam
        all_results[lam] = results

    # ── PLOT 1: Rolling Coverage Comparison ──
    fig, ax = plt.subplots(figsize=(14, 5))
    window = 200
    colors = cm.viridis(np.linspace(0.1, 0.9, len(lambda_values)))

    for i, lam in enumerate(lambda_values):
        cov = all_results[lam]["coverage_seq"]
        rolling = np.convolve(cov, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(len(rolling)), rolling,
                color=colors[i], linewidth=1.5, label=f"λ={lam}")

    ax.axhline(y=0.9, color='black', linestyle='--', linewidth=1.5, label='Target (90%)')
    switch_times = np.arange(gap, T, gap)
    for st in switch_times:
        ax.axvline(x=st, color='grey', linestyle=':', alpha=0.5)
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("Rolling Coverage", fontsize=13)
    ax.set_title("Rolling Coverage vs λ (Regularization Weight)", fontsize=15)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coverage_comparison.png"), dpi=150)
    plt.close()
    print(f"\n  Saved: coverage_comparison.png")

    # ── PLOT 2: Adaptive α_t Trajectories ──
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, lam in enumerate(lambda_values):
        ax.plot(all_results[lam]["alpha_seq"], color=colors[i],
                linewidth=0.8, alpha=0.7, label=f"λ={lam}")
    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=1.5)
    for st in switch_times:
        ax.axvline(x=st, color='grey', linestyle=':', alpha=0.5)
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("α_t", fontsize=13)
    ax.set_title("Adaptive Error Rate α_t", fontsize=15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "alpha_trajectories.png"), dpi=150)
    plt.close()
    print(f"  Saved: alpha_trajectories.png")

    # ── PLOT 3: Scale Factor s_t Trajectories ──
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, lam in enumerate(lambda_values):
        ax.plot(all_results[lam]["scale_seq"], color=colors[i],
                linewidth=0.8, alpha=0.7, label=f"λ={lam}")
    for st in switch_times:
        ax.axvline(x=st, color='grey', linestyle=':', alpha=0.5)
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("s_t", fontsize=13)
    ax.set_title("Adaptive Scale Factor s_t", fontsize=15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scale_trajectories.png"), dpi=150)
    plt.close()
    print(f"  Saved: scale_trajectories.png")

    # ── PLOT 4: Ellipsoid Volume (det) Over Time ──
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, lam in enumerate(lambda_values):
        vols = np.clip(all_results[lam]["volume_seq"], 0, np.percentile(
            all_results[lam]["volume_seq"], 99))
        ax.plot(vols, color=colors[i], linewidth=0.8, alpha=0.7, label=f"λ={lam}")
    for st in switch_times:
        ax.axvline(x=st, color='grey', linestyle=':', alpha=0.5)
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("det(final Σ)", fontsize=13)
    ax.set_title("Ellipsoid Volume Over Time", fontsize=15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "volume_trajectories.png"), dpi=150)
    plt.close()
    print(f"  Saved: volume_trajectories.png")

    # ── PLOT 5: Ellipsoid Snapshots for Best λ ──
    # Find λ with coverage closest to target
    best_lam = min(lambda_values,
                   key=lambda l: abs(all_results[l]["coverage_seq"].mean() - 0.9))
    best = all_results[best_lam]
    print(f"\n  Best λ = {best_lam} (coverage = {best['coverage_seq'].mean():.4f})")

    snapshots = [200, 400, 600, 800, 1200, 1500, 1800, 1950]
    snapshots = [s for s in snapshots if s < T]
    window_pts = 50

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, t_snap in enumerate(snapshots[:8]):
        ax = axes[i]
        regime = "Regime 1" if ((t_snap - 1) // gap) % 2 == 0 else "Regime 2"
        true_sigma = Sigma1 if regime == "Regime 1" else Sigma2

        # Scatter of recent errors
        idx_start = max(0, t_snap - window_pts)
        scatter = errors[idx_start:t_snap]
        ax.scatter(scatter[:, 0], scatter[:, 1], c='grey', alpha=0.4, s=15, zorder=1)

        # Learned ellipsoid (Σ_opt)
        if best["sigma_opt_seq"][t_snap] is not None:
            s_t = best["scale_seq"][t_snap]
            from scipy.stats import chi2
            alpha_t = best["alpha_seq"][t_snap]
            mu_pred = best["mu_pred_seq"][t_snap]
            chi2_q = chi2.ppf(1 - np.clip(alpha_t, 0.001, 0.999), df=2)
            final_S = s_t * chi2_q * best["sigma_opt_seq"][t_snap]
            try:
                ex, ey = plot_ellipse_points(final_S, center=mu_pred)
                ax.plot(ex, ey, 'b-', linewidth=2, label='Learned', zorder=3)
            except Exception:
                pass

        # True ellipsoid (for reference)
        true_scale = chi2.ppf(0.9, df=2)
        true_S = true_scale * true_sigma
        tx, ty = plot_ellipse_points(true_S)
        ax.plot(tx, ty, 'r--', linewidth=1.5, label='True', zorder=2)

        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        ax.set_title(f"t={t_snap} ({regime})", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f"Ellipsoid Evolution (λ={best_lam}, Blue=Learned, Red=True)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ellipsoid_snapshots.png"), dpi=150)
    plt.close()
    print(f"  Saved: ellipsoid_snapshots.png")

    # ── PLOT 6: Summary Bar Chart ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    coverages = [all_results[l]["coverage_seq"].mean() for l in lambda_values]
    avg_volumes = [np.mean(all_results[l]["volume_seq"]) for l in lambda_values]
    frob_diffs = []
    for l in lambda_values:
        diffs = []
        for t in range(T):
            if (all_results[l]["sigma_opt_seq"][t] is not None and
                    all_results[l]["sigma_T_seq"][t] is not None):
                diffs.append(np.linalg.norm(
                    all_results[l]["sigma_opt_seq"][t] - all_results[l]["sigma_T_seq"][t], 'fro'))
        frob_diffs.append(np.mean(diffs) if diffs else 0)

    lam_labels = [str(l) for l in lambda_values]

    axes[0].bar(lam_labels, coverages, color=[colors[i] for i in range(len(lambda_values))])
    axes[0].axhline(y=0.9, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_ylabel("Mean Coverage", fontsize=12)
    axes[0].set_xlabel("λ", fontsize=12)
    axes[0].set_title("Coverage by λ", fontsize=13)
    axes[0].set_ylim(0.7, 1.0)

    axes[1].bar(lam_labels, avg_volumes, color=[colors[i] for i in range(len(lambda_values))])
    axes[1].set_ylabel("Mean Volume", fontsize=12)
    axes[1].set_xlabel("λ", fontsize=12)
    axes[1].set_title("Avg Ellipsoid Volume by λ", fontsize=13)

    axes[2].bar(lam_labels, frob_diffs, color=[colors[i] for i in range(len(lambda_values))])
    axes[2].set_ylabel("Mean ‖Σ_opt − Σ_T‖_F", fontsize=12)
    axes[2].set_xlabel("λ", fontsize=12)
    axes[2].set_title("Σ_opt − Σ_T Difference by λ", fontsize=13)

    plt.suptitle("Hyperparameter Summary", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lambda_summary.png"), dpi=150)
    plt.close()
    print(f"  Saved: lambda_summary.png")

    # ── PLOT 7: Mahalanobis Distances ──
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(np.arange(T), best["mahal_seq"], c='grey', alpha=0.15, s=3)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='Threshold')
    for st in switch_times:
        ax.axvline(x=st, color='grey', linestyle=':', alpha=0.5)
    ax.set_ylim(0, np.clip(np.percentile(best["mahal_seq"], 99), 5, 20))
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("Mahalanobis Distance", fontsize=13)
    ax.set_title(f"Mahalanobis Distances (λ={best_lam})", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mahalanobis_distances.png"), dpi=150)
    plt.close()
    print(f"  Saved: mahalanobis_distances.png")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  λ SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'λ':>8} | {'Coverage':>10} | {'Avg Volume':>12} | {'‖Σ_opt-Σ_T‖':>14}")
    print(f"  {'─'*8}-+-{'─'*10}-+-{'─'*12}-+-{'─'*14}")
    for i, lam in enumerate(lambda_values):
        print(f"  {lam:>8.2f} | {coverages[i]:>10.4f} | {avg_volumes[i]:>12.4e} | "
              f"{frob_diffs[i]:>14.4e}")
    print(f"{'='*70}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: Stock Returns
# ═══════════════════════════════════════════════════════════════════════════════

def run_stock_experiment(output_dir, lambda_values=None):
    """Run on AMD/Nvidia daily returns if data files are available."""

    stock_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "DtACI", "MajorDailyReturns")

    amd_file = os.path.join(stock_dir, "AMDDailyPrices.csv")
    nvidia_file = os.path.join(stock_dir, "NividiaDailyPrices.csv")

    if not (os.path.isfile(amd_file) and os.path.isfile(nvidia_file)):
        print("  Stock data not found. Skipping stock experiment.")
        return None

    import pandas as pd

    amd = pd.read_csv(amd_file)
    nvidia = pd.read_csv(nvidia_file)

    prices_amd = amd[' Open'].values[::-1]
    returns_amd = np.diff(prices_amd) / prices_amd[:-1]

    prices_nvidia = nvidia[' Open'].values[::-1]
    returns_nvidia = np.diff(prices_nvidia) / prices_nvidia[:-1]

    min_len = min(len(returns_amd), len(returns_nvidia))
    returns_amd = returns_amd[-min_len:]
    returns_nvidia = returns_nvidia[-min_len:]

    # Errors = actual returns (predicting 0 as baseline)
    stock_errors = np.column_stack([returns_amd, returns_nvidia])

    # Remove any NaN/Inf
    mask = np.all(np.isfinite(stock_errors), axis=1)
    stock_errors = stock_errors[mask]

    if lambda_values is None:
        lambda_values = [0.1, 1.0, 5.0]

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: Stock Returns (AMD/Nvidia), T={len(stock_errors)}")
    print(f"{'='*70}")

    all_results = {}
    for lam in lambda_values:
        print(f"\n  λ = {lam}")
        system = IntegratedAlpacaDtACI(
            d=2, alpha_target=0.1,
            n_context=10, lam=lam,
            alpha_gamma=0.05, scale_gamma=0.05,
            warmup=100, alpaca_train_iters=300,
            alpaca_kwargs={"nn_layers": [64, 32], "lr": 0.001, "sigma_eps": 0.1}
        )
        results = system.run(stock_errors, verbose=True)
        all_results[lam] = results

    # Quick coverage plot
    T = len(stock_errors)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = cm.viridis(np.linspace(0.1, 0.9, len(lambda_values)))
    window = 200
    for i, lam in enumerate(lambda_values):
        cov = all_results[lam]["coverage_seq"]
        rolling = np.convolve(cov, np.ones(window)/window, mode='valid')
        ax.plot(rolling, color=colors[i], linewidth=1.5, label=f"λ={lam}")
    ax.axhline(y=0.9, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("Rolling Coverage", fontsize=13)
    ax.set_title("Stock Returns (AMD/Nvidia): Rolling Coverage", fontsize=15)
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stock_coverage.png"), dpi=150)
    plt.close()
    print(f"\n  Saved: stock_coverage.png")

    # Find best lam
    best_lam = min(lambda_values, key=lambda l: abs(all_results[l]["coverage_seq"].mean() - 0.9))
    best = all_results[best_lam]

    # Plot Ellipsoid Snapshots for best lambda
    snapshots = [500, 1000, 2000, 3000, 4000, 5000]
    snapshots = [s for s in snapshots if s < T]
    window_pts = 50

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, t_snap in enumerate(snapshots[:6]):
        ax = axes[i]
        
        idx_start = max(0, t_snap - window_pts)
        scatter = stock_errors[idx_start:t_snap]
        ax.scatter(scatter[:, 0], scatter[:, 1], c='grey', alpha=0.4, s=15, zorder=1)

        if best["sigma_opt_seq"][t_snap] is not None:
            s_t = best["scale_seq"][t_snap]
            from scipy.stats import chi2
            alpha_t = best["alpha_seq"][t_snap]
            mu_pred = best["mu_pred_seq"][t_snap]
            chi2_q = chi2.ppf(1 - np.clip(alpha_t, 0.001, 0.999), df=2)
            final_S = s_t * chi2_q * best["sigma_opt_seq"][t_snap]
            try:
                ex, ey = plot_ellipse_points(final_S, center=mu_pred)
                ax.plot(ex, ey, 'b-', linewidth=2, label=f'Learned (λ={best_lam})', zorder=3)
            except Exception:
                pass
        
        # dynamic limits based on scatter
        max_extent = np.max(np.abs(scatter)) if len(scatter) > 0 else 0.1
        lim = max_extent * 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(f"t={t_snap}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f"Stock Returns Ellipsoid Evolution (λ={best_lam})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stock_ellipsoid_snapshots.png"), dpi=150)
    plt.close()
    print(f"  Saved: stock_ellipsoid_snapshots.png")

    # Plot Lambda Comparison at t=5000
    t_comp = min(5000, T - 1)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    idx_start = max(0, t_comp - window_pts)
    scatter = stock_errors[idx_start:t_comp]
    ax.scatter(scatter[:, 0], scatter[:, 1], c='grey', alpha=0.4, s=15, zorder=1)
    
    for i, lam in enumerate(lambda_values):
        res = all_results[lam]
        if res["sigma_opt_seq"][t_comp] is not None:
            s_t = res["scale_seq"][t_comp]
            from scipy.stats import chi2
            alpha_t = res["alpha_seq"][t_comp]
            mu_pred = res["mu_pred_seq"][t_comp]
            chi2_q = chi2.ppf(1 - np.clip(alpha_t, 0.001, 0.999), df=2)
            final_S = s_t * chi2_q * res["sigma_opt_seq"][t_comp]
            try:
                ex, ey = plot_ellipse_points(final_S, center=mu_pred)
                ax.plot(ex, ey, linewidth=2, color=colors[i], label=f'λ={lam}', zorder=3)
            except Exception:
                pass

    max_extent = np.max(np.abs(scatter)) if len(scatter) > 0 else 0.1
    lim = max_extent * 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_title(f"Lambda Comparison at t={t_comp}", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stock_lambda_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved: stock_lambda_comparison.png")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run integrated ALPaCA + DtACI experiments")
    parser.add_argument("--experiment", type=str, default="simulated",
                        choices=["simulated", "stocks", "all"],
                        help="Which experiment to run")
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=None,
                        help="Lambda values to sweep (default: [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])")
    parser.add_argument("--T", type=int, default=2000,
                        help="Number of timesteps for simulated data")
    parser.add_argument("--gap", type=int, default=500,
                        help="Regime switch gap for simulated data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir is None:
        output_dir = os.path.join(base_dir, "results", "integrated")
    else:
        output_dir = args.output_dir

    if args.experiment in ("simulated", "all"):
        sim_dir = os.path.join(output_dir, "simulated_2d")
        run_simulated_experiment(sim_dir, lambda_values=args.lambdas,
                                 T=args.T, gap=args.gap)

    if args.experiment in ("stocks", "all"):
        stock_dir = os.path.join(output_dir, "stocks")
        run_stock_experiment(stock_dir, lambda_values=args.lambdas)

    print("\n✓ All experiments complete!")
