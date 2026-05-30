#!/usr/bin/env python3
"""
Generate Useful Comparison Plots
=================================
Produces clean side-by-side rolling coverage and coverage std-dev plots
for all three probability methods (Empirical, Moment Matching, Davies).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Add parent directory to path for project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrated_alpaca_dtaci import IntegratedAlpacaDtACI, generate_multidim_regime_shift
from run_integrated_experiment import plot_ellipse_points


def get_stock_dataset():
    """Load AMD/Nvidia daily returns."""
    import pandas as pd
    stock_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "DtACI", "MajorDailyReturns")
    amd_file = os.path.join(stock_dir, "AMDDailyPrices.csv")
    nvidia_file = os.path.join(stock_dir, "NividiaDailyPrices.csv")

    if not (os.path.isfile(amd_file) and os.path.isfile(nvidia_file)):
        return None

    amd = pd.read_csv(amd_file)
    nvidia = pd.read_csv(nvidia_file)
    prices_amd = amd[' Open'].values[::-1]
    returns_amd = np.diff(prices_amd) / prices_amd[:-1]
    prices_nvidia = nvidia[' Open'].values[::-1]
    returns_nvidia = np.diff(prices_nvidia) / prices_nvidia[:-1]

    min_len = min(len(returns_amd), len(returns_nvidia))
    stock_errors = np.column_stack([returns_amd[-min_len:], returns_nvidia[-min_len:]])
    mask = np.all(np.isfinite(stock_errors), axis=1)
    return stock_errors[mask]


def run_all_methods(data, d=2, lam=10.0, label="dataset"):
    """Run all three probability methods on the same dataset and return results."""
    methods = ["empirical", "moment", "davies"]
    results = {}

    for method in methods:
        print(f"\n{'='*50}")
        print(f"  {label} — Method: {method.upper()}")
        print(f"{'='*50}")

        system = IntegratedAlpacaDtACI(
            d=d, alpha_target=0.1,
            n_context=10, lam=lam,
            alpha_gamma=0.05, scale_gamma=0.05,
            warmup=100, alpaca_train_iters=300,
            replay_buffer_size=0,
            prob_method=method,
            prob_weight=10.0,
            alpaca_kwargs={"nn_layers": [64, 32], "lr": 0.001, "sigma_eps": 0.1}
        )

        res = system.run(data, verbose=True)
        results[method] = res

    return results


def rolling_window(arr, window):
    """Compute rolling mean using convolution."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def rolling_std(arr, window):
    """Compute rolling standard deviation."""
    T = len(arr)
    out = np.zeros(T - window + 1)
    for i in range(len(out)):
        out[i] = np.std(arr[i:i + window])
    return out


def plot_coverage_comparison(results, output_dir, dataset_name, window=200):
    """
    Plot 1: Rolling coverage for all three methods as 3 stacked subplots.
    """
    methods = ["empirical", "moment", "davies"]
    method_labels = {"empirical": "Empirical (Sigmoid)", "moment": "Moment Matching", "davies": "Davies Method"}
    colors = {"empirical": "#e74c3c", "moment": "#2ecc71", "davies": "#3498db"}

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, sharey=True)

    for i, method in enumerate(methods):
        ax = axes[i]
        cov = results[method]["coverage_seq"]
        rolling_cov = rolling_window(cov, window)
        T_roll = len(rolling_cov)

        ax.plot(np.arange(T_roll), rolling_cov,
                color=colors[method], linewidth=1.5, label=method_labels[method])
        ax.axhline(y=0.9, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label="Target (0.90)")
        ax.fill_between(np.arange(T_roll), 0.85, 0.95,
                         color='grey', alpha=0.08, label="±5% band")

        final_cov = cov.mean()
        ax.set_ylabel("Rolling Coverage", fontsize=12)
        ax.set_title(f"{method_labels[method]}  —  Final Coverage: {final_cov:.4f}",
                      fontsize=13, fontweight='bold', pad=8)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.55, 1.05)

    axes[-1].set_xlabel("Timestep", fontsize=13)
    plt.suptitle(f"Rolling Coverage Comparison (window={window}) — {dataset_name}",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(output_dir, f"rolling_coverage_{dataset_name.lower().replace(' ', '_').replace('/', '_')}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_stdev_comparison(results, output_dir, dataset_name, window=200):
    """
    Plot 2: Rolling standard deviation of coverage for all three methods as 3 stacked subplots.
    """
    methods = ["empirical", "moment", "davies"]
    method_labels = {"empirical": "Empirical (Sigmoid)", "moment": "Moment Matching", "davies": "Davies Method"}
    colors = {"empirical": "#e74c3c", "moment": "#2ecc71", "davies": "#3498db"}

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, sharey=True)

    for i, method in enumerate(methods):
        ax = axes[i]
        cov = results[method]["coverage_seq"]
        roll_std = rolling_std(cov, window)
        T_roll = len(roll_std)

        ax.plot(np.arange(T_roll), roll_std,
                color=colors[method], linewidth=1.5, label=method_labels[method])

        mean_std = roll_std.mean()
        ax.axhline(y=mean_std, color=colors[method], linestyle=':', linewidth=1.2, alpha=0.7,
                    label=f"Mean σ = {mean_std:.4f}")

        ax.set_ylabel("Rolling Std Dev", fontsize=12)
        ax.set_title(f"{method_labels[method]}  —  Mean σ: {mean_std:.4f}",
                      fontsize=13, fontweight='bold', pad=8)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Timestep", fontsize=13)
    plt.suptitle(f"Coverage Std Dev Over Time (window={window}) — {dataset_name}",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(output_dir, f"coverage_stdev_{dataset_name.lower().replace(' ', '_').replace('/', '_')}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "results", "useful_plots")
    os.makedirs(output_dir, exist_ok=True)

    # ── Simulated 2D Data ──
    print("\n" + "=" * 60)
    print("  SIMULATED 2D REGIME SHIFT DATA")
    print("=" * 60)

    np.random.seed(42)
    T, gap, d = 2000, 500, 2
    Sigma1 = np.array([[3.0, 0.8], [0.8, 0.5]])
    Sigma2 = np.array([[0.5, -0.2], [-0.2, 4.0]])
    sim_data = np.zeros((T, d))
    for t in range(T):
        if (t // gap) % 2 == 0:
            sim_data[t] = np.random.multivariate_normal(np.zeros(d), Sigma1)
        else:
            sim_data[t] = np.random.multivariate_normal(np.zeros(d), Sigma2)

    sim_results = run_all_methods(sim_data, d=d, lam=10.0, label="Simulated 2D")

    plot_coverage_comparison(sim_results, output_dir, "Simulated 2D", window=200)
    plot_stdev_comparison(sim_results, output_dir, "Simulated 2D", window=200)

    # ── Stock Returns Data ──
    print("\n" + "=" * 60)
    print("  STOCK RETURNS (AMD/NVIDIA)")
    print("=" * 60)

    stock_data = get_stock_dataset()
    if stock_data is not None:
        stock_results = run_all_methods(stock_data, d=2, lam=10.0, label="Stocks (AMD/Nvidia)")

        plot_coverage_comparison(stock_results, output_dir, "Stocks AMD-Nvidia", window=200)
        plot_stdev_comparison(stock_results, output_dir, "Stocks AMD-Nvidia", window=200)
    else:
        print("  Stock data not found — skipping.")

    print(f"\n✓ All useful plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
