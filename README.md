# Integrated ALPaCA and DtACI

This repository integrates **ALPaCA** (Adaptive Learning for Probabilistic and Conformal Analytics) with **DtACI** (Dynamically Tuned Adaptive Conformal Inference) to construct a highly robust, multi-dimensional time-series forecasting and uncertainty quantification system.

The system is uniquely resilient to dynamic regime shifts, volatility clustering, and localized distribution changes mapping out complex multi-variable environments. The intended use case is for autonomous trajectory planning for robots controlled by model-based controllers.

---

## System Architecture Overview

### 1. ALPaCA (Meta-Learning Forecaster)
ALPaCA is essentially the **spatial targeting mechanism**. It is a meta-learning methodology designed for few-shot regression and probabilistic forecasting. 
By utilizing a small moving window of recent context (the immediate history of the time series), ALPaCA smoothly adapts to localized trends out-of-sample and dynamically forecasts the **expected mean ($\mu$)** of the distribution. It focuses strictly on placing the "center" of our uncertainty region appropriately over shifting data trends.

### 2. DtACI (Adaptive Conformal Bounds)
DtACI provides the strict **probabilistic scale tracking**. Pure traditional conformal inference breaks down when data is temporally correlated or distribution shifts occur. DtACI leverages an online learning framework (expert algorithms) to dynamically adapt the target conformal geometry:
*   Adapts the empirical error rate margin ($\alpha_t$) depending on localized violations.
*   Estimates and tracks multi-variable correlations utilizing a rolling geometry scale $s_t$ alongside an Exponential Moving Average (EMA) empirical covariance estimate ($\Sigma_T$).

### 3. The Combined Ensemble System
The integration yields an architecture far stronger than the sum of its parts. 
While ALPaCA isolates the shifting mean ($\mu$), the integrated system offloads the prediction of uncertainty covariance and bounds back into DtACI.

The core breakthrough rests in the optimization of the final predicted ellipsoid matrix ($\Sigma$) using a **differentiable empirical formulation**:
```math
\min_{\Sigma} \quad \left| P(e^T \Sigma^{-1} e < 1) - (1 -\alpha_t) \right| + \det(\Sigma) + \lambda \|\Sigma - \Sigma_T\|_F
```
This forces the boundaries to shrink their area (`det`) strictly while dynamically expanding to trap exactly $1 - \alpha_t$ proportion of the recent contextual residuals, all anchored against the fundamental rotation axes detected by DtACI ($\Sigma_T$).

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

The repository is pre-configured with a master evaluation suite (`run_integrated_experiment.py`) for rapidly verifying performance profiles under varying regularizations.

### Simulated Regime Shifts
Test the underlying optimizations rapidly against a localized 2D testing array containing highly aggressive structural gaps simulating large-scale distribution shifts:

```bash
python run_integrated_experiment.py --experiment simulated --lambdas 10.0 --T 1500 --gap 500
```

### Testing On Live Stock Rollouts (Multi-dimensional Proof of Concept)
A much larger execution targeting multi-year, multi-dimensional correlations against the real-life volatility index profiles found inside historical AMD/Nvidia openings.

```bash
python run_integrated_experiment.py --experiment stocks --lambdas 10.0
```
*(All plots and metrics generated from executions are automatically exported into `results/integrated/`)*

---

## Key Codebase Functions

- **`IntegratedAlpacaDtACI.run()`** *(in `integrated_alpaca_dtaci.py`)*
  The absolute central loop driving the entire online tracking framework. Dictates ALPaCA predictions, runs error tracking, loops DtACI scaling calculations, and maintains the context windows simultaneously.
- **`optimize_sigma()`** *(in `integrated_alpaca_dtaci.py`)*
  The heart of the conformal geometric bounds structure. Utilizing `scipy.optimize.minimize` alongside a fully differentiable sigmoid margin mapping to prevent non-differentiable $L_1$ step crashes, this function strictly enforces that exactly $\sim 90\%$ of ALPaCA residuals fall strictly inside the $e^T\Sigma^{-1}e < 1$ covariance bounds.
- **`self._alpaca_predict(context_errors)`** *(in `integrated_alpaca_dtaci.py`)*
  The feature bridge safely delivering purely scaled temporal points to the underlying ALPaCA PyTorch models and harvesting $\mu$ placements.

---

## Acknowledgements

This integrated system builds upon the core frameworks established in the following foundational repositories. We deeply credit the original authors:
- **ALPaCA**: [StanfordASL/ALPaCA](https://github.com/StanfordASL/ALPaCA)
- **DtACI**: [isgibbs/DtACI](https://github.com/isgibbs/DtACI)
