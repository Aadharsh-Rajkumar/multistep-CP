#!/usr/bin/env python3
"""
Generalized 4-Ellipsoid Comparison: Old vs New Hyperparameters
===============================================================

Evaluates:

    Old Bonferroni
    Old Joint+Overlap
    New Bonferroni
    New Joint+Overlap

Changes:
    1. Restricts PyTorch to a single CPU thread to avoid
       OpenMP / MKL native crashes on macOS.
    2. Trains ALPaCA exactly once per dataset.
    3. Reuses the same trained ALPaCA model across all
       four configurations.

Outputs structured directories:

    results/
        <timestamp>_generalized_comparison/
            robotics/
                cmp_ellipsoids/
                volume_cmp_robotics.png

            stocks/
                cmp_ellipsoids/
                volume_cmp_stocks.png
"""

import os
import sys
import datetime


# ============================================================
# PyTorch / macOS thread configuration
# ============================================================
#
# These environment variables MUST be set before importing
# NumPy / PyTorch so the underlying native libraries see them
# during initialization.
#
# This is intended to reduce the chance of OpenMP / MKL
# thread-runtime conflicts causing a segmentation fault.
# ============================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ============================================================
# Imports
# ============================================================

import numpy as np
import torch

# Force PyTorch to use one CPU thread.
torch.set_num_threads(1)

# Force PyTorch's inter-op parallelism to one thread as well.
torch.set_num_interop_threads(1)

import matplotlib

# Use a non-interactive backend because this script generates
# plots directly to files.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from scipy.stats import chi2

from integrated_alpaca_dtaci_v2 import TrajectoryAlpacaDtACI

from scripts.generate_drone_tracking import (
    generate_drone_trajectory,
)

from integrated_alpaca_dtaci import (
    generate_multidim_regime_shift,
)


# ============================================================
# Reproducibility
# ============================================================

# Set global seeds once at program startup.
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# Ellipsoid plotting helper
# ============================================================

def plot_ellipse_pts(
    ax,
    mu,
    sigma,
    color="blue",
    alpha=0.3,
    label=None,
):
    """
    Plot a 2D covariance ellipse.

    Args:
        ax:
            Matplotlib axis.

        mu:
            Ellipse center, shape (2,).

        sigma:
            2x2 covariance / shape matrix.

        color:
            Ellipse color.

        alpha:
            Ellipse transparency.

        label:
            Optional legend label.
    """

    try:
        # Eigen-decompose the covariance matrix.
        eigvals, eigvecs = np.linalg.eigh(sigma)

        # Prevent negative / zero eigenvalues from causing
        # invalid square roots.
        eigvals = np.maximum(eigvals, 1e-10)

        # Convert eigenvalues into ellipse diameters.
        width, height = 2 * np.sqrt(eigvals)

        # Compute the rotation angle of the principal axis.
        angle = np.degrees(
            np.arctan2(
                eigvecs[1, 0],
                eigvecs[0, 0],
            )
        )

        # Create the ellipse patch.
        ell = Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            label=label,
        )

        # Add ellipse to the plot.
        ax.add_patch(ell)

    except Exception:
        # Never allow plotting failure to terminate the experiment.
        pass


# ============================================================
# Train ALPaCA exactly once
# ============================================================

def train_alpaca_once(
    errors,
    shared_kwargs,
):
    """
    Train ALPaCA exactly once for a dataset.

    The resulting ALPaCA model and configuration are returned
    so that all four comparison configurations can reuse them.

    Args:
        errors:
            Error sequence with shape (T, d).

        shared_kwargs:
            Shared configuration parameters.

    Returns:
        alpaca_model:
            Trained ALPaCA model.

        alpaca_config:
            ALPaCA configuration dictionary.
    """

    # Get dimensionality of the dataset.
    d = errors.shape[1]

    # Reproduce the same random initialization used by the
    # original experiment.
    torch.manual_seed(42)
    np.random.seed(42)

    # Copy shared arguments so the original dictionary is
    # not modified.
    kw = shared_kwargs.copy()

    # The dimensionality must match this dataset.
    kw["d"] = d

    # Create one system solely for ALPaCA training.
    #
    # The values of lam, scale_gamma, K, and overlap_mode
    # do not affect ALPaCA training.
    #
    # Therefore, we can train ALPaCA once and reuse it.
    alpaca_system = TrajectoryAlpacaDtACI(
        N=10,
        K=0,
        delta_conf=0.05,
        eta_joint=0.01,
        base_matrix_mode="alpaca_1sd",
        overlap_mode="raw",
        **kw,
    )

    # Calculate the exact same warmup boundary that the v2
    # implementation uses internally.
    T = errors.shape[0]

    warmup_end = min(
        alpaca_system.warmup,
        T // 4,
    )

    warmup_end = max(
        warmup_end,
        alpaca_system.n_context + 5,
    )

    # Extract the warmup data used for ALPaCA training.
    warmup_errors = errors[:warmup_end]

    print(
        f"  > Training ALPaCA once "
        f"(d={d}, "
        f"warmup={len(warmup_errors)}, "
        f"iters={alpaca_system.alpaca_train_iters})"
    )

    # Train ALPaCA.
    #
    # This calls the exact same initialization logic used
    # internally by TrajectoryAlpacaDtACI.run().
    alpaca_system._init_alpaca(warmup_errors)

    # Return the trained model and its configuration.
    return (
        alpaca_system.alpaca_model,
        alpaca_system.alpaca_config,
    )


# ============================================================
# Run one configuration
# ============================================================

def run_single_config(
    errors,
    lam,
    scale_gamma,
    K,
    overlap_mode,
    shared_kwargs,
    alpaca_model,
    alpaca_config,
):
    """
    Run one DtACI configuration using an already-trained
    ALPaCA model.

    ALPaCA is NOT retrained here.

    Args:
        errors:
            Dataset error sequence.

        lam:
            Sigma optimization regularization parameter.

        scale_gamma:
            Scale controller learning rate.

        K:
            Maximum overlap lag.

        overlap_mode:
            Overlap handling mode.

        shared_kwargs:
            Shared experiment parameters.

        alpaca_model:
            Already-trained ALPaCA model.

        alpaca_config:
            Configuration corresponding to alpaca_model.

    Returns:
        Experiment result dictionary.
    """

    # Create a fresh random seed for deterministic experiment
    # initialization.
    #
    # NOTE:
    # We do NOT retrain ALPaCA, so this seed only controls
    # other stochastic components of the experiment.
    torch.manual_seed(42)
    np.random.seed(42)

    # Copy the shared configuration.
    kw = shared_kwargs.copy()

    # Override the configuration-specific hyperparameters.
    kw["lam"] = lam
    kw["scale_gamma"] = scale_gamma
    kw["d"] = errors.shape[1]

    # Tell the system that ALPaCA is already trained.
    kw["skip_alpaca_init"] = True

    # Create a fresh DtACI system.
    #
    # Each configuration gets its own DtACI state:
    #
    #   alpha controller
    #   sigma controller
    #   scale controller
    #   overlap table
    #   trajectory buffer
    #
    # But all four configurations share the exact same
    # trained ALPaCA model.
    sys_obj = TrajectoryAlpacaDtACI(
        N=10,
        K=K,
        delta_conf=0.05,
        eta_joint=0.01,
        base_matrix_mode="alpaca_1sd",
        overlap_mode=overlap_mode,
        **kw,
    )

    # Inject the already-trained ALPaCA model.
    sys_obj.alpaca_model = alpaca_model

    # Inject the matching ALPaCA configuration.
    sys_obj.alpaca_config = alpaca_config

    # Run the actual experiment.
    #
    # Because skip_alpaca_init=True, the v2 run method will
    # NOT retrain ALPaCA.
    return sys_obj.run(
        errors,
        verbose=True,
    )


# ============================================================
# Volume comparison plot
# ============================================================

def plot_volume_comparison(
    results,
    configs,
    shift_t,
    title_prefix,
    out_dir,
):
    """
    Generate volume comparison plots for all four configurations.
    """

    # Create two vertically stacked plots.
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
    )

    # ========================================================
    # Top plot: Absolute volume
    # ========================================================

    ax = axes[0]

    for (
        label,
        lam,
        sg,
        K,
        om,
        color,
    ) in configs:

        # Convert volume to log10 scale.
        vol = np.log10(
            np.maximum(
                results[label]["volume_seq"],
                1e-20,
            )
        )

        # Plot the volume sequence.
        ax.plot(
            vol,
            color=color,
            alpha=0.6,
            lw=1.2,
            label=label,
        )

    # Mark regime shift.
    ax.axvline(
        shift_t,
        color="red",
        ls=":",
        lw=2,
        label="Regime shift",
    )

    ax.set_ylabel("log₁₀(Volume)")

    ax.set_title(
        f"{title_prefix}: Volume Comparison (Old vs New)"
    )

    ax.legend(fontsize=8)

    ax.grid(
        True,
        alpha=0.3,
    )

    # ========================================================
    # Bottom plot: Overlap vs Bonferroni
    # ========================================================

    ax = axes[1]

    # Extract old overlap volume.
    old_o = np.log10(
        np.maximum(
            results["Old Joint+Overlap"]["volume_seq"],
            1e-20,
        )
    )

    # Extract old Bonferroni volume.
    old_b = np.log10(
        np.maximum(
            results["Old Bonferroni"]["volume_seq"],
            1e-20,
        )
    )

    # Extract new overlap volume.
    new_o = np.log10(
        np.maximum(
            results["New Joint+Overlap"]["volume_seq"],
            1e-20,
        )
    )

    # Extract new Bonferroni volume.
    new_b = np.log10(
        np.maximum(
            results["New Bonferroni"]["volume_seq"],
            1e-20,
        )
    )

    # Plot old overlap-minus-Bonferroni.
    ax.plot(
        old_o - old_b,
        color="#FF5722",
        alpha=0.6,
        lw=1.2,
        label="Old: Overlap vs Bonf (neg=overlap wins)",
    )

    # Plot new overlap-minus-Bonferroni.
    ax.plot(
        new_o - new_b,
        color="#E91E63",
        alpha=0.6,
        lw=1.2,
        label="New: Overlap vs Bonf (neg=overlap wins)",
    )

    # Zero means equal volume.
    ax.axhline(
        0,
        color="black",
        ls="--",
        alpha=0.3,
    )

    # Mark regime shift.
    ax.axvline(
        shift_t,
        color="red",
        ls=":",
        lw=2,
    )

    ax.set_xlabel("Timestep")

    ax.set_ylabel("log₁₀ Ratio")

    ax.set_title(
        f"{title_prefix}: Invariant Check (Overlap < Bonf)"
    )

    ax.legend(fontsize=8)

    ax.grid(
        True,
        alpha=0.3,
    )

    # Automatically adjust spacing.
    plt.tight_layout()

    # Save the plot.
    plt.savefig(
        os.path.join(
            out_dir,
            f"volume_cmp_{title_prefix}.png",
        ),
        dpi=150,
    )

    # Release Matplotlib resources.
    plt.close()


# ============================================================
# Generate 4-ellipsoid comparison grids
# ============================================================

def generate_ellipsoids(
    results,
    configs,
    tracking_errors,
    shift_t,
    out_dir,
):
    """
    Generate 4-row ellipsoid comparison grids.

    Each row corresponds to one configuration.
    """

    # Total number of timesteps.
    T = tracking_errors.shape[0]

    # Snapshot range.
    start_t = 900
    end_t = 1900
    step = 5

    # Number of timesteps per image.
    chunk_size = 50

    # Generate valid snapshot times.
    all_times = [
        t
        for t in range(
            start_t,
            end_t,
            step,
        )
        if 0 <= t < T
    ]

    # Split snapshot times into groups.
    chunks = [
        all_times[i:i + chunk_size // step]
        for i in range(
            0,
            len(all_times),
            chunk_size // step,
        )
    ]

    # Output directory for ellipsoid images.
    ell_dir = os.path.join(
        out_dir,
        "cmp_ellipsoids",
    )

    # Create directory if needed.
    os.makedirs(
        ell_dir,
        exist_ok=True,
    )

    # Process each chunk.
    for chunk_idx, snapshot_times in enumerate(chunks):

        # Skip empty chunks.
        if not snapshot_times:
            continue

        # Create 4 rows, one per configuration.
        fig, axes = plt.subplots(
            4,
            len(snapshot_times),
            figsize=(
                2.2 * len(snapshot_times),
                16,
            ),
        )

        # Handle the single-column case.
        if len(snapshot_times) == 1:
            axes = axes.reshape(
                4,
                1,
            )

        # Iterate over snapshot columns.
        for col, t in enumerate(snapshot_times):

            # Iterate over four configurations.
            for row, (
                label,
                lam,
                sg,
                K,
                om,
                color,
            ) in enumerate(configs):

                # Select subplot.
                ax = axes[row, col]

                # Retrieve results for this configuration.
                res = results[label]

                # Get recent tracking errors.
                recent = tracking_errors[
                    max(0, t - 30):t + 1
                ]

                # Plot recent trajectory.
                if recent.shape[0] > 0:
                    ax.plot(
                        recent[:, 0],
                        recent[:, 1],
                        color="gray",
                        alpha=0.4,
                        lw=1,
                    )

                # Plot current error point.
                ax.scatter(
                    tracking_errors[t, 0],
                    tracking_errors[t, 1],
                    color="black",
                    s=40,
                    zorder=5,
                )

                # Get optimized covariance matrix.
                sigma_opt = res[
                    "sigma_opt_seq"
                ][t]

                # Get ALPaCA mean.
                mu = res[
                    "mu_pred_seq"
                ][t]

                # Only draw ellipsoid if covariance exists.
                if sigma_opt is not None:

                    # Get adaptive alpha.
                    alpha_val = res[
                        "alpha_seq"
                    ][t]

                    # Get adaptive scale.
                    s = res[
                        "scale_seq"
                    ][t]

                    # Compute chi-square threshold.
                    chi2_s = chi2.ppf(
                        1
                        - np.clip(
                            alpha_val,
                            0.001,
                            0.999,
                        ),
                        df=2,
                    )

                    # Construct final ellipsoid matrix.
                    final = (
                        s
                        * chi2_s
                        * sigma_opt
                    )

                    # Plot the ellipse.
                    plot_ellipse_pts(
                        ax,
                        mu,
                        final,
                        color=color,
                        alpha=0.25,
                    )

                    # Calculate determinant / volume.
                    vol = max(
                        np.linalg.det(final),
                        1e-12,
                    )

                    # Display volume beneath subplot.
                    ax.set_xlabel(
                        f"{vol:.2e}",
                        fontsize=7,
                    )

                    # Estimate ellipse radii.
                    rx = np.sqrt(
                        final[0, 0]
                    )

                    ry = np.sqrt(
                        final[1, 1]
                    )

                    # Plot padding.
                    pad = 0.1

                    # Determine x-axis bounds.
                    min_x = min(
                        recent[:, 0].min(),
                        mu[0] - rx,
                    )

                    max_x = max(
                        recent[:, 0].max(),
                        mu[0] + rx,
                    )

                    # Determine y-axis bounds.
                    min_y = min(
                        recent[:, 1].min(),
                        mu[1] - ry,
                    )

                    max_y = max(
                        recent[:, 1].max(),
                        mu[1] + ry,
                    )

                    # Set plot limits.
                    ax.set_xlim(
                        min_x - pad,
                        max_x + pad,
                    )

                    ax.set_ylim(
                        min_y - pad,
                        max_y + pad,
                    )

                # Set subplot title.
                ax.set_title(
                    f"{label} t={t}",
                    fontsize=7,
                )

                # Keep equal aspect ratio.
                ax.set_aspect(
                    "equal"
                )

                # Add grid.
                ax.grid(
                    True,
                    alpha=0.2,
                )

        # Add overall title.
        fig.suptitle(
            f"Ellipsoids "
            f"{snapshot_times[0]}–"
            f"{snapshot_times[-1]}",
            fontsize=14,
        )

        # Adjust layout.
        plt.tight_layout()

        # Save image.
        plt.savefig(
            os.path.join(
                ell_dir,
                f"ellipsoids_"
                f"{snapshot_times[0]}_"
                f"{snapshot_times[-1]}.png",
            ),
            dpi=200,
            bbox_inches="tight",
        )

        # Release resources.
        plt.close()


# ============================================================
# Run all four configurations on one dataset
# ============================================================

def run_experiment_on_dataset(
    dataset_name,
    errors,
    shift_t,
    configs,
    shared_kwargs,
    base_dir,
):
    """
    Run all four configurations on a dataset.

    ALPaCA is trained exactly once for this dataset.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"  RUNNING DATASET: "
        f"{dataset_name} "
        f"(d={errors.shape[1]})"
    )

    print(
        "=" * 60
    )

    # Create dataset output directory.
    out_dir = os.path.join(
        base_dir,
        dataset_name.lower(),
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    # ========================================================
    # Train ALPaCA ONCE
    # ========================================================

    print(
        "\n  ============================================"
    )

    print(
        "  TRAINING ALPaCA ONCE FOR THIS DATASET"
    )

    print(
        "  ============================================"
    )

    alpaca_model, alpaca_config = train_alpaca_once(
        errors,
        shared_kwargs,
    )

    print(
        "  ✓ ALPaCA training complete"
    )

    print(
        "  ✓ Reusing same ALPaCA model for all 4 configs"
    )

    # ========================================================
    # Run four configurations
    # ========================================================

    results = {}

    for (
        label,
        lam,
        sg,
        K,
        om,
        color,
    ) in configs:

        print(
            f"\n  > {label} "
            f"(lam={lam}, "
            f"sg={sg}, "
            f"K={K}, "
            f"overlap_mode={om})"
        )

        # Run configuration using the already-trained model.
        results[label] = run_single_config(
            errors=errors,
            lam=lam,
            scale_gamma=sg,
            K=K,
            overlap_mode=om,
            shared_kwargs=shared_kwargs,
            alpaca_model=alpaca_model,
            alpaca_config=alpaca_config,
        )

    # ========================================================
    # Generate volume comparison
    # ========================================================

    plot_volume_comparison(
        results,
        configs,
        shift_t,
        dataset_name,
        out_dir,
    )

    # ========================================================
    # Generate ellipsoid grids for 2D data
    # ========================================================

    if errors.shape[1] == 2:

        generate_ellipsoids(
            results,
            configs,
            errors,
            shift_t,
            out_dir,
        )

    # ========================================================
    # Print summary
    # ========================================================

    print(
        f"\n  SUMMARY: {dataset_name}"
    )

    for (
        label,
        lam,
        sg,
        K,
        om,
        color,
    ) in configs:

        # Compute mean log-volume.
        vol = np.mean(
            np.log10(
                np.maximum(
                    results[label]["volume_seq"],
                    1e-20,
                )
            )
        )

        print(
            f"    {label:20s}: "
            f"Mean log₁₀(Vol) = {vol:.4f}"
        )

    # ========================================================
    # Compare old configurations
    # ========================================================

    old_o = np.mean(
        np.log10(
            np.maximum(
                results[
                    "Old Joint+Overlap"
                ]["volume_seq"],
                1e-20,
            )
        )
    )

    old_b = np.mean(
        np.log10(
            np.maximum(
                results[
                    "Old Bonferroni"
                ]["volume_seq"],
                1e-20,
            )
        )
    )

    # ========================================================
    # Compare new configurations
    # ========================================================

    new_o = np.mean(
        np.log10(
            np.maximum(
                results[
                    "New Joint+Overlap"
                ]["volume_seq"],
                1e-20,
            )
        )
    )

    new_b = np.mean(
        np.log10(
            np.maximum(
                results[
                    "New Bonferroni"
                ]["volume_seq"],
                1e-20,
            )
        )
    )

    # Check old invariant.
    old_ok = old_o < old_b

    # Check new invariant.
    new_ok = new_o < new_b

    print(
        f"\n    Old invariant holds "
        f"(overlap < bonf): "
        f"{old_ok}"
    )

    print(
        f"    New invariant holds "
        f"(overlap < bonf): "
        f"{new_ok}"
    )

    print(
        f"    Overlap volume reduction "
        f"(Old - New): "
        f"{old_o - new_o:.4f} "
        f"(pos=better)"
    )


# ============================================================
# Main
# ============================================================

def main():

    # Generate timestamp for output directory.
    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # Create output path.
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        f"{timestamp}_generalized_comparison",
    )

    # Create output directory.
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ========================================================
    # Shared experiment parameters
    # ========================================================

    shared_kwargs = dict(

        # Target error rate.
        alpha_target=0.1,

        # Number of historical error vectors ALPaCA uses.
        n_context=15,

        # Default lambda.
        #
        # This is overwritten per configuration below.
        lam=1.0,

        # Alpha controller learning rate.
        alpha_gamma=0.01,

        # Default scale controller learning rate.
        #
        # This is overwritten per configuration below.
        scale_gamma=0.01,

        # Number of warmup points.
        warmup=100,

        # Number of ALPaCA training iterations.
        alpaca_train_iters=300,

        # Probability calculation method.
        prob_method="moment",

        # Weight for probability penalty.
        prob_weight=10.0,

        # Force ALPaCA predicted mean to zero.
        force_zero_mean=True,

        # ALPaCA neural-network parameters.
        alpaca_kwargs={
            "nn_layers": [128, 64],
            "lr": 0.001,
            "sigma_eps": 0.1,
        },
    )

    # ========================================================
    # Old hyperparameters
    # ========================================================

    OLD_LAM = 0.5
    OLD_GAMMA = 0.01

    # ========================================================
    # New hyperparameters (modify these to compare against original)
    # ========================================================

    NEW_LAM = 0.6
    NEW_GAMMA = 0.012

    # ========================================================
    # Four comparison configurations
    # ========================================================

    configs = [

        # Old Bonferroni configuration.
        (
            "Old Bonferroni",
            OLD_LAM,
            OLD_GAMMA,
            0,
            "raw",
            "#4CAF50",
        ),

        # Old Joint + Overlap configuration.
        (
            "Old Joint+Overlap",
            OLD_LAM,
            OLD_GAMMA,
            5,
            "raw",
            "#FF5722",
        ),

        # New Bonferroni configuration.
        (
            "New Bonferroni",
            NEW_LAM,
            NEW_GAMMA,
            0,
            "raw",
            "#1565C0",
        ),

        # New Joint + Overlap configuration.
        (
            "New Joint+Overlap",
            NEW_LAM,
            NEW_GAMMA,
            5,
            "raw",
            "#E91E63",
        ),
    ]

    # ========================================================
    # Experiment header
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "  GENERALIZED 4-ELLIPSOID COMPARISON"
    )

    print(
        f"  Old Hyperparams: "
        f"lam={OLD_LAM}, "
        f"scale_gamma={OLD_GAMMA}"
    )

    print(
        f"  New Hyperparams: "
        f"lam={NEW_LAM}, "
        f"scale_gamma={NEW_GAMMA}"
    )

    print(
        f"  Output: {output_dir}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 1. Robotics dataset
    # ========================================================

    # Reset seeds before generating robotics data.
    torch.manual_seed(42)
    np.random.seed(42)

    # Robotics dataset length.
    T_rob = 2000

    # Robotics regime-shift timestep.
    shift_t_rob = 1000

    # Generate drone trajectory.
    pos, ref = generate_drone_trajectory(
        T=T_rob,
        shift_t=shift_t_rob,
    )

    # Calculate tracking error.
    err_rob = pos - ref

    # Run four configurations.
    #
    # ALPaCA is trained once here.
    run_experiment_on_dataset(
        "robotics",
        err_rob,
        shift_t_rob,
        configs,
        shared_kwargs,
        output_dir,
    )

    # ========================================================
    # 2. Stocks dataset
    # ========================================================

    # Reset seeds before generating stock data.
    torch.manual_seed(42)
    np.random.seed(42)

    # Stock dataset length.
    T_stk = 2000

    # Stock regime-shift timestep.
    shift_t_stk = 1000

    # Covariance before regime shift.
    Sigma1 = np.array([
        [4.0, 0.5],
        [0.5, 0.3],
    ])

    # Covariance after regime shift.
    Sigma2 = np.array([
        [0.3, -0.2],
        [-0.2, 4.0],
    ])

    # Generate multidimensional regime-shift data.
    err_stk = generate_multidim_regime_shift(
        T_stk,
        shift_t_stk,
        Sigma1,
        Sigma2,
    )

    # Run four configurations.
    #
    # ALPaCA is trained once here.
    run_experiment_on_dataset(
        "stocks",
        err_stk,
        shift_t_stk,
        configs,
        shared_kwargs,
        output_dir,
    )

    # ========================================================
    # Finished
    # ========================================================

    print(
        f"\n✓ Saved all results and plots to "
        f"{output_dir}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
