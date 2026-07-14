# Integrated ALPaCA and DtACI v2

This repository integrates **ALPaCA** (Adaptive Learning for Probabilistic and Conformal Analytics) with **DtACI** (Dynamically Tuned Adaptive Conformal Inference) to construct a highly robust, multi-dimensional time-series forecasting and uncertainty quantification system.

The intended use case is for autonomous trajectory planning for robots controlled by model-based controllers, where safety must be guaranteed across a rolling horizon of predictions.

---

## System Architecture Overview

### 1. ALPaCA (Meta-Learning Forecaster)
ALPaCA provides the **spatial targeting mechanism**. It is a meta-learning methodology designed for few-shot regression and probabilistic forecasting. 
By utilizing a small moving window of recent context, ALPaCA smoothly adapts to localized trends out-of-sample and dynamically forecasts the **expected mean ($\mu$)**. It focuses strictly on placing the "center" of our uncertainty region appropriately over shifting data trends.

### 2. DtACI v1 (Marginal Conformal Bounds)
DtACI v1 provides the strict **probabilistic scale tracking** for single-step (marginal) predictions. 
*   Adapts the empirical error rate margin ($\alpha_t$) depending on localized violations via expert tracking.
*   Estimates and tracks multi-variable correlations utilizing a rolling geometry scale $s_t$ alongside an Exponential Moving Average (EMA) empirical covariance estimate ($\Sigma_T$).
*   Optimizes the predicted ellipsoid matrix ($\Sigma$) to bound exactly $1 - \alpha_t$ proportion of the recent contextual residuals.

### 3. DtACI v2 (Dependence-Aware Trajectory Allocation)
DtACI v2 extends the system to guarantee safety over an entire **$N$-step trajectory**, resolving the fundamental issue that marginal (single-step) prediction bounds compound and fail catastrophically over multi-step horizons.
*   **Adaptive Joint Risk Budget**: Uses a continuous PID controller to dynamically adapt the baseline joint target ($\alpha_{\text{joint}}^{\text{live}}$) based on observed empirical trajectory violations.
*   **Pairwise Overlap Tracking**: Tracks the intersection probabilities of violation events over recent windows.
*   **Kounias-Hunter Correction**: Subtracts the tracked overlaps from the raw Bonferroni union bound, tightening the conformal ellipsoids safely and significantly reducing volume bloat without sacrificing the joint trajectory safety guarantee.

---

## Setup and Installation

### Conda Environment Setup
Run the following commands in your terminal to mirror the exact system state used to compile this architecture:

```bash
# 1. Create a fresh conda environment
conda create -n dtaci_alpaca python=3.13 -y

# 2. Activate the environment
conda activate dtaci_alpaca

# 3. Install the primary library engines
pip install torch numpy scipy pandas matplotlib
```

---

## Execution Commands

The repository uses the evaluation suite `run_comparison_experiment.py` to compare the system versions.

### Mode 1: DtACI Volume Comparison (`--mode dtaci`)
Compare the v1 Marginal baseline against v2 Joint Bonferroni and v2 Joint + Overlap. Both v2 variants use the optimized `sigma_T` base shape. This mode mathematically proves that the Kounias-Hunter overlap correction reduces ellipsoid volume compared to the naive Bonferroni approach.

```bash
python run_comparison_experiment.py --experiment both --mode dtaci
```

### Mode 2: ALPaCA Base Shape Diagnosis (`--mode alpaca`)
Compare the v1 baseline against v2 tracking using ALPaCA's purely predicted covariance matrices (ALPaCA 1-α base and ALPaCA 1-SD base). This mode produces granular horizontal ellipsoid snapshots in steps of 5 timesteps to visually diagnose ALPaCA's failure to capture non-spherical spatial correlations.

```bash
python run_comparison_experiment.py --experiment both --mode alpaca
```

*(All plots and metrics generated from executions are automatically exported into `results/`)*

---

## Key Codebase Functions

- **`IntegratedAlpacaDtACI.run()`** *(in `integrated_alpaca_dtaci.py`)*
  The central loop driving the original v1 online tracking framework for single-step bounds.
- **`TrajectoryAlpacaDtACI.run()`** *(in `integrated_alpaca_dtaci_v2.py`)*
  The v2 execution engine. Implements the temporal overlap tables, computes the Kounias-Hunter corrections, manages the trajectory-level PID budget scaling, and dictates exactly what scalar $\alpha_m$ is fed into the geometric optimizer.
- **`optimize_sigma()`** *(in `integrated_alpaca_dtaci.py`)*
  The heart of the conformal geometric bounds structure. Utilizing `scipy.optimize.minimize`, this enforces that exactly $\sim 90\%$ of ALPaCA residuals fall strictly inside the $e^T\Sigma^{-1}e < 1$ covariance bounds while minimizing total volume.

---

## Acknowledgements

This integrated system builds upon the core frameworks established in the following foundational repositories. We credit the original authors:
- **ALPaCA**: [StanfordASL/ALPaCA](https://github.com/StanfordASL/ALPaCA)
- **DtACI**: [isgibbs/DtACI](https://github.com/isgibbs/DtACI)
- **Dependence-Aware Risk Allocation**: Based on "Online Dependence-Aware Conformal Risk Allocation for Receding-Horizon Control" (Srinivasan, 2024)
