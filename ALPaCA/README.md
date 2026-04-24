# ALPaCA — Dual TensorFlow / PyTorch Implementation

Code for "Meta-Learning Priors for Efficient Online Bayesian Regression" by James Harrison, Apoorva Sharma, and Marco Pavone.

## Directory Structure

```
ALPaCA/
├── pytorch/          # PyTorch implementation (current, Python 3.10+)
│   ├── main/         # Core ALPaCA code
│   ├── maml/         # MAML baseline
│   ├── configs/      # YAML experiment configs
│   └── *.ipynb       # Experiment notebooks
├── tensorflow/       # Original TensorFlow implementation (needs Python 3.9)
│   ├── main/         # Core ALPaCA code (TF 1.x with TF2 compat shims)
│   ├── maml/         # MAML code
│   └── configs/      # YAML experiment configs
├── datasets/         # Drop your CSV files here
├── results/          # Auto-generated results (per-dataset, per-backend)
└── run_tests.py      # Centralized testing script
```

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy scipy matplotlib scikit-learn pyyaml gymnasium
# For TensorFlow (requires Python ≤3.12):
pip install tensorflow
```

### 2. Run on a dataset

Drop a CSV file into `datasets/` and run:

```bash
# Run both backends
python run_tests.py --dataset my_data --backend both

# PyTorch only
python run_tests.py --dataset my_data --backend torch

# TensorFlow only (use pid env)
/path/to/miniconda3/envs/pid/bin/python run_tests.py --dataset my_data --backend tensorflow

# Run all datasets
python run_tests.py --dataset all
```

### 3. Tweakable parameters

```bash
python run_tests.py --dataset my_data \
    --n_context 5 \          # Past datapoints for context/theta
    --sigma_eps 0.1 \        # Observation noise variance
    --train_iters 5000 \     # Meta-training iterations
    --nn_layers 128,128,32 \ # Network architecture
    --activation tanh        # Activation function
```

### 4. Results format

```
results/
└── my_data/
    ├── torch/
    │   ├── predictions.csv   # Per-point predictions
    │   └── summary.csv       # MSE summary
    └── tensorflow/
        ├── predictions.csv
        └── summary.csv
```

## Running the Notebooks

```bash
cd pytorch && jupyter notebook
```
