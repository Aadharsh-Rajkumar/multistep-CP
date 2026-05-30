#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import chi2
import datetime
from matplotlib.collections import LineCollection

from integrated_alpaca_dtaci import IntegratedAlpacaDtACI
from run_integrated_experiment import plot_ellipse_points
from scripts.generate_drone_tracking import generate_drone_trajectory
from scripts.generate_useful_plots import rolling_window, rolling_std


def run_multirotor_experiment():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        base_dir, "results",
        f"{timestamp}_multirotor_moment_matching"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  ADAPTNC MULTIROTOR TRACKING EXPERIMENT (Moment Matching)")
    print(f"  Output Dir: {output_dir}")
    print("=" * 70 + "\n")

    # ─────────────────────────────
    # DATA — tracking errors, NOT raw positions
    # ─────────────────────────────
    T = 2000
    shift_t = 1000

    print("  Generating 2D drone trajectory with actuator degradation...")
    positions, references = generate_drone_trajectory(T=T, shift_t=shift_t)

    # Tracking errors: small, zero-centered residuals (like stock returns)
    tracking_errors = positions - references
    print(f"  Tracking error stats — pre-shift std: {tracking_errors[:shift_t].std(axis=0)}")
    print(f"                         post-shift std: {tracking_errors[shift_t:].std(axis=0)}")

    # ─────────────────────────────
    # CONFIGS
    # ─────────────────────────────
    lambda_values = [0.1, 1.0, 5.0, 10.0]
    all_results = {}

    for lam in lambda_values:
        key = f"λ={lam}"
        print(f"\n  Running config: {key}")

        system = IntegratedAlpacaDtACI(
            d=2,
            alpha_target=0.1,
            n_context=15,
            lam=lam,
            alpha_gamma=0.01,
            scale_gamma=0.01,
            warmup=100,
            alpaca_train_iters=300,
            replay_buffer_size=0,
            prob_method="moment",
            prob_weight=10.0,
            alpaca_kwargs={
                "nn_layers": [128, 64],
                "lr": 0.001,
                "sigma_eps": 0.1
            }
        )

        all_results[key] = system.run(tracking_errors, verbose=True)

    colors = cm.viridis(np.linspace(0.1, 0.9, len(lambda_values)))
    window = 200

    print("\n  Generating visualizations...")

    # ─────────────────────────────
    # PLOT 0: Full trajectory overview
    # ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(references[:shift_t, 0], references[:shift_t, 1],
            'k--', alpha=0.5, label='Reference')
    ax.plot(positions[:shift_t, 0], positions[:shift_t, 1],
            color='#2ecc71', alpha=0.7, label='Pre-Shift (Healthy)')
    ax.plot(positions[shift_t:, 0], positions[shift_t:, 1],
            color='#e74c3c', alpha=0.7, label='Post-Shift (Degraded)')
    ax.set_title("Drone Trajectory (Position Space)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(tracking_errors[:shift_t, 0], tracking_errors[:shift_t, 1],
               s=3, alpha=0.4, color='#2ecc71', label='Pre-Shift Errors')
    ax.scatter(tracking_errors[shift_t:, 0], tracking_errors[shift_t:, 1],
               s=3, alpha=0.4, color='#e74c3c', label='Post-Shift Errors')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title("Tracking Errors (What Our System Sees)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drone_trajectory_overview.png"), dpi=150)
    plt.close()

    # ─────────────────────────────
    # PLOT 1: Rolling Coverage
    # ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, lam in enumerate(lambda_values):
        key = f"λ={lam}"
        rolling = rolling_window(all_results[key]["coverage_seq"], window)
        ax.plot(rolling, color=colors[i], label=key)
    ax.axhline(0.9, color='black', linestyle='--', linewidth=1.5, label='Target (90%)')
    ax.axvline(shift_t, color='red', linestyle=':', linewidth=2, label='Actuator Degradation')
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Rolling Coverage", fontsize=12)
    ax.set_title("Multirotor Tracking: Rolling Coverage", fontsize=14)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drone_coverage.png"), dpi=150)
    plt.close()

    # ─────────────────────────────
    # PLOT 2: Coverage Std Dev
    # ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, lam in enumerate(lambda_values):
        key = f"λ={lam}"
        roll_std = rolling_std(all_results[key]["coverage_seq"], window)
        ax.plot(roll_std, color=colors[i], label=key)
    ax.axvline(shift_t, color='red', linestyle=':', linewidth=2, label='Actuator Degradation')
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Rolling Std Dev", fontsize=12)
    ax.set_title("Multirotor Tracking: Coverage Stability (Std Dev)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drone_stdev_coverage.png"), dpi=150)
    plt.close()

    # ─────────────────────────────
    # PLOT 3: Alpha Trajectory
    # ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, lam in enumerate(lambda_values):
        key = f"λ={lam}"
        ax.plot(all_results[key]["alpha_seq"], color=colors[i], alpha=0.7, label=key)
    ax.axhline(0.1, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(shift_t, color='red', linestyle=':', linewidth=2)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("α_t", fontsize=12)
    ax.set_title("Adaptive Error Rate α_t", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drone_alpha_trajectories.png"), dpi=150)
    plt.close()

    # ─────────────────────────────
    # PLOT 4: Scale Factor
    # ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, lam in enumerate(lambda_values):
        key = f"λ={lam}"
        ax.plot(all_results[key]["scale_seq"], color=colors[i], alpha=0.7, label=key)
    ax.axvline(shift_t, color='red', linestyle=':', linewidth=2)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("s_t", fontsize=12)
    ax.set_title("Adaptive Scale Factor s_t", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drone_scale_trajectories.png"), dpi=150)
    plt.close()

    # ─────────────────────────────
    # PLOT 5: Ellipsoid Volume
    # ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, lam in enumerate(lambda_values):
        key = f"λ={lam}"
        vols = all_results[key]["volume_seq"]
        ax.plot(vols, color=colors[i], alpha=0.7, label=key)
    ax.axvline(shift_t, color='red', linestyle=':', linewidth=2)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("det(final Σ)", fontsize=12)
    ax.set_title("Ellipsoid Volume Over Time", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drone_volume_trajectories.png"), dpi=150)
    plt.close()

    # ─────────────────────────────
    # PLOT 6: Ellipsoid Snapshots (in error space)
    # ─────────────────────────────
    best_lam = min(
        lambda_values,
        key=lambda l: abs(all_results[f"λ={l}"]["coverage_seq"].mean() - 0.9)
    )
    best = all_results[f"λ={best_lam}"]

    snapshots = [200, 500, 800, 1100, 1400, 1800]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    window_pts = 80

    for i, t in enumerate(snapshots):
        ax = axes[i]
        regime = "Pre-Shift" if t < shift_t else "Post-Shift"

        # Show recent tracking errors as a fading path
        start = max(0, t - window_pts)
        recent_errors = tracking_errors[start:t+1]
        
        points = np.array([recent_errors[:, 0], recent_errors[:, 1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        alphas = np.linspace(0.05, 0.8, len(segments))
        lc = LineCollection(segments, colors='grey', linewidths=2, zorder=1)
        # Apply fading alpha directly to the colors array
        colors_arr = np.zeros((len(segments), 4))
        colors_arr[:, 3] = alphas # Set alpha channel
        colors_arr[:, 0:3] = 0.5  # Set RGB to grey (0.5, 0.5, 0.5)
        lc.set_color(colors_arr)
        ax.add_collection(lc)

        # Plot dummy line for legend
        ax.plot([], [], color='grey', alpha=0.8, linewidth=2, label='Recent Path')

        # Current point
        ax.scatter(tracking_errors[t, 0], tracking_errors[t, 1],
                   color='black', s=50, zorder=4, label='Current Point (t)')

        # Compute auto-fitting limits from the scatter
        pad = 0.05
        x_min = np.min(recent_errors[:, 0]) - pad
        x_max = np.max(recent_errors[:, 0]) + pad
        y_min = np.min(recent_errors[:, 1]) - pad
        y_max = np.max(recent_errors[:, 1]) + pad

        # Draw the learned ellipsoid
        if best["sigma_opt_seq"][t] is not None:
            mu = best["mu_pred_seq"][t]
            alpha_t = best["alpha_seq"][t]
            s_t = best["scale_seq"][t]
            q = chi2.ppf(1 - np.clip(alpha_t, 0.001, 0.999), df=2)
            S = s_t * q * best["sigma_opt_seq"][t]

            try:
                ex, ey = plot_ellipse_points(S, center=mu)
                ax.plot(ex, ey, 'b-', linewidth=2.5, zorder=3, label=f'Ellipsoid (λ={best_lam})')

                # Expand limits to fit ellipsoid
                x_min = min(x_min, np.min(ex) - pad)
                x_max = max(x_max, np.max(ex) + pad)
                y_min = min(y_min, np.min(ey) - pad)
                y_max = max(y_max, np.max(ey) + pad)
            except Exception:
                pass

        ax.axhline(0, color='black', linewidth=0.3, alpha=0.3)
        ax.axvline(0, color='black', linewidth=0.3, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f"t={t} ({regime})", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8, loc='upper left')

    plt.suptitle(f"Tracking Error Ellipsoid Evolution (λ={best_lam})",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "drone_ellipsoid_snapshots.png"), dpi=150)
    plt.close()

    # ─────────────────────────────
    # PLOT 7: Lambda Comparison
    # ─────────────────────────────
    t_comp = 1400
    fig, ax = plt.subplots(figsize=(8, 8))

    start = max(0, t_comp - window_pts)
    recent_errors = tracking_errors[start:t_comp+1]
    
    points = np.array([recent_errors[:, 0], recent_errors[:, 1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    alphas = np.linspace(0.05, 0.8, len(segments))
    lc = LineCollection(segments, colors='grey', linewidths=2, zorder=1)
    colors_arr = np.zeros((len(segments), 4))
    colors_arr[:, 3] = alphas
    colors_arr[:, 0:3] = 0.5
    lc.set_color(colors_arr)
    ax.add_collection(lc)
    ax.plot([], [], color='grey', alpha=0.8, linewidth=2, label='Recent Path')
    ax.scatter(tracking_errors[t_comp, 0], tracking_errors[t_comp, 1],
               color='black', s=50, zorder=4, label='Current Point (t)')

    pad = 0.05
    x_min = np.min(recent_errors[:, 0]) - pad
    x_max = np.max(recent_errors[:, 0]) + pad
    y_min = np.min(recent_errors[:, 1]) - pad
    y_max = np.max(recent_errors[:, 1]) + pad

    for i, lam in enumerate(lambda_values):
        res = all_results[f"λ={lam}"]
        if res["sigma_opt_seq"][t_comp] is not None:
            mu = res["mu_pred_seq"][t_comp]
            alpha_t = res["alpha_seq"][t_comp]
            s_t = res["scale_seq"][t_comp]
            q = chi2.ppf(1 - np.clip(alpha_t, 0.001, 0.999), df=2)
            S = s_t * q * res["sigma_opt_seq"][t_comp]
            try:
                ex, ey = plot_ellipse_points(S, center=mu)
                ax.plot(ex, ey, linewidth=2, color=colors[i], label=f"λ={lam}", zorder=3)
                x_min = min(x_min, np.min(ex) - pad)
                x_max = max(x_max, np.max(ex) + pad)
                y_min = min(y_min, np.min(ey) - pad)
                y_max = max(y_max, np.max(ey) + pad)
            except Exception:
                pass

    ax.axhline(0, color='black', linewidth=0.3, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.3, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title(f"Lambda Comparison at t={t_comp} (Post-Shift)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drone_lambda_comparison.png"), dpi=150)
    plt.close()

    print(f"\n✓ Saved all plots to {output_dir}")


if __name__ == "__main__":
    run_multirotor_experiment()
