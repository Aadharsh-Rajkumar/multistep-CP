import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import chi2

# Add parent directory to path to allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrated_alpaca_dtaci import IntegratedAlpacaDtACI
from run_integrated_experiment import plot_ellipse_points

def generate_identical_dataset(T=2000, gap=500, d=2):
    np.random.seed(42)
    # Identical regime logic
    mu1 = np.zeros(d)
    Sigma1 = np.array([[3.0, 0.8], [0.8, 0.5]])

    mu2 = np.zeros(d)
    Sigma2 = np.array([[0.5, -0.2], [-0.2, 4.0]])

    data = np.zeros((T, d))
    for t in range(T):
        if (t // gap) % 2 == 0:
            data[t] = np.random.multivariate_normal(mu1, Sigma1)
        else:
            data[t] = np.random.multivariate_normal(mu2, Sigma2)
    
    return data, Sigma1, Sigma2

def get_stock_dataset():
    import pandas as pd
    stock_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
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

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "results", "method_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    T = 2000
    gap = 500
    d = 2
    lam = 10.0
    buffer_size = 0 # Baseline comparison
    
    print("Generating identical dataset...")
    data, Sigma1, Sigma2 = generate_identical_dataset(T, gap, d)
    
    methods = ["empirical", "moment", "davies"]
    all_results = {}
    
    for method in methods:
        print(f"\n=====================================")
        print(f" Executing Method: {method.upper()}")
        print(f"=====================================")
        
        system = IntegratedAlpacaDtACI(
            d=d, alpha_target=0.1,
            n_context=10, lam=lam,
            alpha_gamma=0.05, scale_gamma=0.05,
            warmup=100, alpaca_train_iters=300,
            replay_buffer_size=buffer_size,
            prob_method=method,
            prob_weight=10.0,
            alpaca_kwargs={"nn_layers": [64, 32], "lr": 0.001, "sigma_eps": 0.1}
        )
        
        res = system.run(data, verbose=True)
        all_results[method] = res
        
    print("\nGenerating comparative PDF plot...")
    
    snapshots = [200, 400, 600, 800, 1200, 1500, 1800, 1950]
    window_pts = 50
    
    fig, axes = plt.subplots(3, 8, figsize=(28, 11))
    
    for row_idx, method in enumerate(methods):
        res = all_results[method]
        cov_mean = res["coverage_seq"].mean()
        
        for col_idx, t_snap in enumerate(snapshots):
            ax = axes[row_idx, col_idx]
            
            regime = "Regime 1" if ((t_snap - 1) // gap) % 2 == 0 else "Regime 2"
            true_sigma = Sigma1 if regime == "Regime 1" else Sigma2
            
            idx_start = max(0, t_snap - window_pts)
            scatter = data[idx_start:t_snap]
            ax.scatter(scatter[:, 0], scatter[:, 1], c='grey', alpha=0.4, s=15, zorder=1)
            
            if res["sigma_opt_seq"][t_snap] is not None:
                s_t = res["scale_seq"][t_snap]
                alpha_t = res["alpha_seq"][t_snap]
                mu_pred = res["mu_pred_seq"][t_snap]
                chi2_q = chi2.ppf(1 - np.clip(alpha_t, 0.001, 0.999), df=d)
                final_S = s_t * chi2_q * res["sigma_opt_seq"][t_snap]
                try:
                    ex, ey = plot_ellipse_points(final_S, center=mu_pred)
                    ax.plot(ex, ey, 'b-', linewidth=2.5, label='Learned', zorder=3)
                except Exception:
                    pass
            
            true_scale = chi2.ppf(0.9, df=d)
            true_S = true_scale * true_sigma
            tx, ty = plot_ellipse_points(true_S)
            ax.plot(tx, ty, 'r--', linewidth=2, label='True', zorder=2)
            
            ax.set_xlim(-8, 8)
            ax.set_ylim(-8, 8)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            if row_idx == 0:
                ax.set_title(f"t={t_snap} ({regime})", fontsize=14, fontweight='bold')
                
            if col_idx == 0:
                ax.set_ylabel(f"{method.capitalize()}\n(Cov: {cov_mean:.3f})", fontsize=15, fontweight='bold')
                ax.legend(fontsize=10)

    plt.suptitle("Probability Evaluation Methods Comparison (λ=10.0, B=0) - Simulated", fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf_path = os.path.join(output_dir, "prob_method_comparison.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"Saved exact comparative PDF to {pdf_path}")
    
    # ====================================================================
    # STOCKS EXECUTION BLOCK
    # ====================================================================
    print("\n\n" + "="*50)
    print(" LOADING STOCKS DATASET")
    print("="*50)
    
    stock_data = get_stock_dataset()
    if stock_data is None:
        print("Stock data not found, skipping stock evaluation.")
        return
        
    T_stock = len(stock_data)
    print(f"Stock dataset length: {T_stock}")
    
    stock_results = {}
    for method in methods:
        print(f"\n=====================================")
        print(f" Stock Method: {method.upper()}")
        print(f"=====================================")
        
        system = IntegratedAlpacaDtACI(
            d=d, alpha_target=0.1,
            n_context=10, lam=lam,
            alpha_gamma=0.05, scale_gamma=0.05,
            warmup=100, alpaca_train_iters=300,
            replay_buffer_size=buffer_size,
            prob_method=method,
            prob_weight=10.0,
            alpaca_kwargs={"nn_layers": [64, 32], "lr": 0.001, "sigma_eps": 0.1}
        )
        
        res = system.run(stock_data, verbose=True)
        stock_results[method] = res

    print("\nGenerating stock comparative PDF plot...")
    
    stock_snapshots = [500, 1000, 2000, 3000, 4000, 5000]
    stock_snapshots = [s for s in stock_snapshots if s < T_stock]
    n_cols = len(stock_snapshots)
    
    fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 11))
    
    for row_idx, method in enumerate(methods):
        res = stock_results[method]
        cov_mean = res["coverage_seq"].mean()
        
        for col_idx, t_snap in enumerate(stock_snapshots):
            ax = axes[row_idx, col_idx] if n_cols > 1 else axes[row_idx]
            
            idx_start = max(0, t_snap - window_pts)
            scatter = stock_data[idx_start:t_snap]
            ax.scatter(scatter[:, 0], scatter[:, 1], c='grey', alpha=0.4, s=15, zorder=1)
            
            if res["sigma_opt_seq"][t_snap] is not None:
                s_t = res["scale_seq"][t_snap]
                alpha_t = res["alpha_seq"][t_snap]
                mu_pred = res["mu_pred_seq"][t_snap]
                chi2_q = chi2.ppf(1 - np.clip(alpha_t, 0.001, 0.999), df=d)
                final_S = s_t * chi2_q * res["sigma_opt_seq"][t_snap]
                try:
                    ex, ey = plot_ellipse_points(final_S, center=mu_pred)
                    ax.plot(ex, ey, 'b-', linewidth=2.5, label='Learned', zorder=3)
                except Exception:
                    pass
            
            max_extent = np.max(np.abs(scatter)) if len(scatter) > 0 else 0.1
            lim = max_extent * 1.5
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            if row_idx == 0:
                ax.set_title(f"t={t_snap}", fontsize=14, fontweight='bold')
                
            if col_idx == 0:
                ax.set_ylabel(f"{method.capitalize()}\n(Cov: {cov_mean:.3f})", fontsize=15, fontweight='bold')
                ax.legend(fontsize=10)

    plt.suptitle("Probability Evaluation Methods Comparison - Stocks (AMD/Nvidia)", fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    stock_pdf_path = os.path.join(output_dir, "stock_prob_method_comparison.pdf")
    plt.savefig(stock_pdf_path, format='pdf', bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"Saved stock comparative PDF to {stock_pdf_path}")

if __name__ == "__main__":
    main()
