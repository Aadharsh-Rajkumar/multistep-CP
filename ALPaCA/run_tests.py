#!/usr/bin/env python3
"""
ALPaCA Centralized Testing Pipeline
====================================
Runs ALPaCA on datasets in the datasets/ folder using PyTorch and/or TensorFlow
backends, saving CSV results to results/<dataset_name>/torch/ and
results/<dataset_name>/tensorflow/.

Usage:
    python run_tests.py --dataset <name>          # Run both backends on one dataset
    python run_tests.py --dataset all             # Run on all datasets in datasets/
    python run_tests.py --dataset <name> --backend torch
    python run_tests.py --dataset <name> --backend tensorflow
    python run_tests.py --dataset <name> --n_context 5 --train_iters 5000

Expected dataset format:
    datasets/<name>.csv  — A CSV file with numeric columns. The first column(s)
    are treated as x (input) and the last column(s) as y (output). You can also
    configure this with --x_cols and --y_cols.

    OR datasets/<name>/ — A folder containing multiple CSV files (one per task).

Parameters you'll tweak later:
    --n_context      Number of past datapoints used as control signal/theta (default: 5)
    --sigma_eps      Observation noise variance (default: from config or 0.1)
    --train_iters    Number of meta-training iterations (default: 5000)
    --nn_layers      Hidden layer sizes (default: 128,128,32)
    --activation     Activation function (default: tanh)
"""

import argparse
import csv
import json
import os
import sys
import time
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PYTORCH_DIR = os.path.join(ROOT_DIR, "pytorch")
TENSORFLOW_DIR = os.path.join(ROOT_DIR, "tensorflow")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(dataset_name, args, x_cols=None, y_cols=None):
    """
    Load a dataset from datasets/<name>.CSV or datasets/<name>/.
    Returns:
        tasks: list of (X, Y) tuples, each X is (N, x_dim), Y is (N, y_dim)
        x_dim, y_dim: inferred dimensions
    """
    import pandas as pd
    import pickle
    
    csv_path = os.path.join(DATASETS_DIR, f"{dataset_name}.csv")
    pkl_path = os.path.join(DATASETS_DIR, f"{dataset_name}.pkl")
    folder_path = os.path.join(DATASETS_DIR, dataset_name)

    def process_dataframe(df):
        # Find the target column seamlessly
        target_candidates = ['target', 'actual', 'log_value']
        y_col = None
        for col in target_candidates:
            if col in df.columns:
                y_col = col
                break
                
        # Find grouping columns like item_id or geo_value to prevent state bleed
        group_candidates = ['item_id', 'geo_value', 'site_id']
        group_col = None
        for col in group_candidates:
            if col in df.columns:
                group_col = col
                break

        def extract_xy(sub_df):
            if y_col is not None:
                # Drop rows where target is missing
                sub_df = sub_df.dropna(subset=[y_col])
                Y = sub_df[[y_col]].values
            else:
                # Fallback to the last numeric column
                numeric_df = sub_df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
                if len(numeric_df.columns) > 0:
                    Y = numeric_df.iloc[:, -1:].values
                else:
                    Y = sub_df.iloc[:, -1:].values
            
            Y = np.nan_to_num(Y, nan=0.0)
            
            # If not using past n as state, we just pull the remainder as X
            if not args.use_past_n_as_state:
                if y_col is not None:
                    numeric_df = sub_df.select_dtypes(include=[np.number]).drop(columns=[y_col])
                    X = numeric_df.values
                else:
                    X = sub_df.select_dtypes(include=[np.number]).iloc[:, :-1].values
                X = np.nan_to_num(X, nan=0.0)
                return X, Y
            else:
                # Provide dummy X; will be overwritten later
                return np.zeros_like(Y), Y

        if group_col is not None:
            tasks_local = []
            for _, group in df.groupby(group_col):
                X, Y = extract_xy(group)
                if len(Y) > 0:
                    tasks_local.append((X, Y))
            return tasks_local
        else:
            X, Y = extract_xy(df)
            return [(X, Y)]

    tasks = []

    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        tasks.extend(process_dataframe(df))

    elif os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'X' in data and 'Y' in data:
            tasks.append((data['X'], data['Y']))
        else:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, dict):
                df = pd.DataFrame(data.get('data', data))
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], tuple):
                tasks = data
                df = None
            else:
                df = pd.DataFrame(data)
                
            if df is not None:
                tasks.extend(process_dataframe(df))

    elif os.path.isdir(folder_path):
        for f in sorted(os.listdir(folder_path)):
            p = os.path.join(folder_path, f)
            if f.endswith(".csv"):
                df = pd.read_csv(p)
                tasks.extend(process_dataframe(df))
            elif f.endswith(".pkl"):
                with open(p, "rb") as pfile:
                    data = pickle.load(pfile)
                df = pd.DataFrame(data if not isinstance(data, dict) else data.get('data', data))
                tasks.extend(process_dataframe(df))
    else:
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found.")

    if args.use_past_n_as_state:
        # We rewrite X to be the past `args.n_context` raw Y values.
        # This means for a given time t, X[t] = [Y_{t-n}, Y_{t-n+1}, ..., Y_{t-1}].
        # We discard the first `n_context` rows of both X and Y since we don't have enough history to form X.
        n = args.n_context
        new_tasks = []
        for X_old, Y_old in tasks:
            T = Y_old.shape[0]
            if T <= n:
                print(f"Warning: Task has fewer timesteps ({T}) than n_context ({n}). Skipping.")
                continue
            
            # Build sliding window for X
            # Y_old shape is (T, y_dim). We want X_new shape to be (T-n, n * y_dim).
            X_new = np.zeros((T - n, n * Y_old.shape[1]))
            for i in range(T - n):
                X_new[i] = Y_old[i : i + n].flatten()
            
            Y_new = Y_old[n:]
            Y_new = Y_old[n:]
            
            if args.normalize:
                # We normalize the X and Y blocks dynamically across tasks
                X_mean, X_std = np.mean(X_new, axis=0), np.std(X_new, axis=0)
                X_std[X_std == 0] = 1.0
                X_new = (X_new - X_mean) / X_std
                
                Y_mean, Y_std = np.mean(Y_new, axis=0), np.std(Y_new, axis=0)
                Y_std[Y_std == 0] = 1.0
                Y_new = (Y_new - Y_mean) / Y_std

            new_tasks.append((X_new, Y_new))
        
        if not new_tasks:
            raise ValueError(f"No tasks have enough timesteps (>{n}) for sliding window.")
        tasks = new_tasks

    x_dim = tasks[0][0].shape[1]
    y_dim = tasks[0][1].shape[1]
    return tasks, x_dim, y_dim


def build_config(x_dim, y_dim, args):
    """Build an ALPaCA config dict from CLI args and data dimensions."""
    nn_layers = [int(x) for x in args.nn_layers.split(",")]
    return {
        "lr": args.lr,
        "x_dim": x_dim,
        "y_dim": y_dim,
        "sigma_eps": args.sigma_eps,
        "data_horizon": args.n_context,
        "test_horizon": max(args.test_horizon, 1),
        "meta_batch_size": args.meta_batch_size,
        "nn_layers": nn_layers,
        "activation": args.activation,
        # Dataset params for SinusoidDataset (not used for CSV/PKL, but needed by config)
        "phase_range": [0, 3.14],
        "freq_range": [0.999, 1.0],
        "amp_range": [0.1, 5.0],
        "x_range": [-5.0, 5.0],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET ADAPTER — wraps CSV data for ALPaCA's training interface
# ═══════════════════════════════════════════════════════════════════════════════

class CSVTaskDataset:
    """
    Wraps loaded CSV tasks into the sample(n_funcs, n_samples) interface
    that ALPaCA's train_model() expects.
    """
    def __init__(self, tasks, n_context):
        """
        tasks: list of (X, Y) numpy arrays
        n_context: how many past points to use as control signal / theta
        """
        self.tasks = tasks
        self.n_context = n_context
        self.x_dim = tasks[0][0].shape[1]
        self.y_dim = tasks[0][1].shape[1]

    def sample(self, n_funcs, n_samples):
        """
        Sample n_funcs tasks, each with n_samples data points.
        Returns x (n_funcs, n_samples, x_dim), y (n_funcs, n_samples, y_dim)
        """
        x = np.zeros((n_funcs, n_samples, self.x_dim))
        y = np.zeros((n_funcs, n_samples, self.y_dim))
        for i in range(n_funcs):
            task_idx = np.random.randint(len(self.tasks))
            X_task, Y_task = self.tasks[task_idx]
            T = X_task.shape[0]
            if n_samples > T:
                # Resample with replacement if not enough data
                inds = np.random.choice(T, n_samples, replace=True)
            else:
                inds = np.random.choice(T, n_samples, replace=False)
            x[i] = X_task[inds]
            y[i] = Y_task[inds]
        return x, y


# ═══════════════════════════════════════════════════════════════════════════════
#  PYTORCH BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

def run_pytorch(tasks, config, args):
    """Train and evaluate ALPaCA using PyTorch backend."""
    sys.path.insert(0, PYTORCH_DIR)
    from main.alpaca import ALPaCA
    sys.path.pop(0)

    print("\n" + "=" * 60)
    print("  PyTorch Backend")
    print("=" * 60)

    dataset = CSVTaskDataset(tasks, args.n_context)
    
    # ── Sigma Epsilon Calibration (Pass 1) ──
    if args.calibrate_sigma:
        print("  [Calibration Pass] Training with base sigma_eps to find true variance...")
        # Use the base config with a small default variance (0.001) as per Appendix B.3
        calib_config = config.copy()
        
        calib_model = ALPaCA(calib_config)
        calib_model.construct_model()
        # Train for a fraction of the iterations to get a decent base model for MSE
        calib_iters = max(100, args.train_iters // 2)
        calib_model.train_model(dataset, calib_iters)
        
        # Calculate Validation MSE across all tasks
        print("  [Calibration Pass] Evaluating validation MSE...")
        squared_errors = []
        for X, Y in tasks:
            T = X.shape[0]
            n_ctx = min(args.n_context, T - 1)
            if n_ctx < 1: n_ctx = 1
            
            x_ctx = X[:n_ctx].reshape(1, n_ctx, -1)
            y_ctx = Y[:n_ctx].reshape(1, n_ctx, -1)
            
            # evaluate on the rest
            x_test = X[n_ctx:].reshape(1, -1, X.shape[1])
            mu_pred, _ = calib_model.test(x_ctx, y_ctx, x_test)
            
            # accumulate squared error per dimension
            # mu_pred: (1, T-n_ctx, y_dim), Y[n_ctx:]: (T-n_ctx, y_dim)
            se = (mu_pred[0] - Y[n_ctx:]) ** 2
            squared_errors.append(se)
        
        all_se = np.vstack(squared_errors) # (sum(T-n_ctx), y_dim)
        mse_per_dim = np.mean(all_se, axis=0).tolist()
        
        # Enforce a small floor to prevent rank deficiency
        mse_per_dim = [max(1e-6, m) for m in mse_per_dim]
        
        print(f"  [Calibration Pass] Found Sigma Epsilon (Variance): {mse_per_dim}")
        # Update actual config with the calibrated sigma vector
        config["sigma_eps"] = mse_per_dim

    # ── Actual Training Pass (Pass 2) ──
    model = ALPaCA(config)
    model.construct_model()

    print(f"Training for {args.train_iters} iterations...")
    t0 = time.time()
    model.train_model(dataset, args.train_iters)
    train_time = time.time() - t0
    print(f"Training took {train_time:.1f}s")

    # Evaluate
    all_results = []
    for task_idx, (X, Y) in enumerate(tasks):
        T = X.shape[0]
        n_ctx = min(args.n_context, T - 1)
        if n_ctx < 1:
            n_ctx = 1

        x_ctx = X[:n_ctx].reshape(1, n_ctx, -1)
        y_ctx = Y[:n_ctx].reshape(1, n_ctx, -1)
        x_test = X[n_ctx:].reshape(1, -1, X.shape[1])

        mu_pred, sig_pred = model.test(x_ctx, y_ctx, x_test)

        for j in range(x_test.shape[1]):
            row = {
                "task": task_idx,
                "timestep": n_ctx + j,
            }
            # Input columns
            for d in range(X.shape[1]):
                row[f"x_{d}"] = float(x_test[0, j, d])
            # True output
            for d in range(Y.shape[1]):
                row[f"y_true_{d}"] = float(Y[n_ctx + j, d])
            # Predicted output
            for d in range(Y.shape[1]):
                row[f"y_pred_{d}"] = float(mu_pred[0, j, d])
            # Predicted variance (diagonal)
            for d in range(Y.shape[1]):
                row[f"var_pred_{d}"] = float(sig_pred[0, j, d, d])

            all_results.append(row)

    return all_results, train_time


# ═══════════════════════════════════════════════════════════════════════════════
#  TENSORFLOW BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

def run_tensorflow(tasks, config, args):
    """Train and evaluate ALPaCA using TensorFlow backend."""
    sys.path.insert(0, TENSORFLOW_DIR)

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    from main.alpaca import ALPaCA

    print("\n" + "=" * 60)
    print("  TensorFlow Backend")
    print("=" * 60)

    dataset = CSVTaskDataset(tasks, args.n_context)

    # ── Sigma Epsilon Calibration (Pass 1) ──
    if args.calibrate_sigma:
        print("  [Calibration Pass] Training with base sigma_eps to find true variance...")
        # Needs its own tf graph to cleanly reset
        tf.reset_default_graph()
        calib_sess = tf.InteractiveSession()
        calib_config = config.copy()

        calib_model = ALPaCA(calib_config, calib_sess)
        calib_model.construct_model()

        calib_iters = max(100, args.train_iters // 2)
        calib_model.train(dataset, calib_iters)

        print("  [Calibration Pass] Evaluating validation MSE...")
        squared_errors = []
        for X, Y in tasks:
            T = X.shape[0]
            n_ctx = min(args.n_context, T - 1)
            if n_ctx < 1: n_ctx = 1

            x_ctx = X[:n_ctx].reshape(1, n_ctx, -1)
            y_ctx = Y[:n_ctx].reshape(1, n_ctx, -1)
            x_test = X[n_ctx:].reshape(1, -1, X.shape[1])

            mu_pred, _ = calib_model.test(x_ctx, y_ctx, x_test)
            se = (mu_pred[0] - Y[n_ctx:]) ** 2
            squared_errors.append(se)

        all_se = np.vstack(squared_errors)
        mse_per_dim = np.mean(all_se, axis=0).tolist()
        mse_per_dim = [max(1e-6, m) for m in mse_per_dim]

        print(f"  [Calibration Pass] Found Sigma Epsilon (Variance): {mse_per_dim}")
        config["sigma_eps"] = mse_per_dim
        calib_sess.close()

    # ── Actual Training Pass (Pass 2) ──
    # Reset TF graph for clean state
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    model = ALPaCA(config, sess)
    model.construct_model()

    # Meta-train
    print(f"Training for {args.train_iters} iterations...")
    t0 = time.time()
    model.train(dataset, args.train_iters)
    train_time = time.time() - t0
    print(f"Training took {train_time:.1f}s")

    # Evaluate
    all_results = []
    for task_idx, (X, Y) in enumerate(tasks):
        T = X.shape[0]
        n_ctx = min(args.n_context, T - 1)
        if n_ctx < 1:
            n_ctx = 1

        x_ctx = X[:n_ctx].reshape(1, n_ctx, -1)
        y_ctx = Y[:n_ctx].reshape(1, n_ctx, -1)
        x_test = X[n_ctx:].reshape(1, -1, X.shape[1])

        mu_pred, sig_pred = model.test(x_ctx, y_ctx, x_test)

        for j in range(x_test.shape[1]):
            row = {
                "task": task_idx,
                "timestep": n_ctx + j,
            }
            for d in range(X.shape[1]):
                row[f"x_{d}"] = float(x_test[0, j, d])
            for d in range(Y.shape[1]):
                row[f"y_true_{d}"] = float(Y[n_ctx + j, d])
            for d in range(Y.shape[1]):
                row[f"y_pred_{d}"] = float(mu_pred[0, j, d])
            for d in range(Y.shape[1]):
                row[f"var_pred_{d}"] = float(sig_pred[0, j, d, d])

            all_results.append(row)

    sess.close()
    sys.path.pop(0)
    return all_results, train_time


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS I/O
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(results, output_dir, dataset_name, backend_name, train_time):
    """Save results as CSV to the appropriate directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Predictions CSV
    pred_path = os.path.join(output_dir, "predictions.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(pred_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"  Saved predictions → {pred_path}")

    # Summary CSV
    summary_path = os.path.join(output_dir, "summary.csv")
    if results:
        # Compute MSE per output dim
        y_true_cols = [k for k in results[0] if k.startswith("y_true_")]
        y_pred_cols = [k for k in results[0] if k.startswith("y_pred_")]
        mse_per_dim = {}
        for yt, yp in zip(y_true_cols, y_pred_cols):
            true_vals = np.array([r[yt] for r in results])
            pred_vals = np.array([r[yp] for r in results])
            mse_per_dim[yt.replace("y_true_", "mse_")] = float(np.mean((true_vals - pred_vals) ** 2))

        summary = {
            "dataset": dataset_name,
            "backend": backend_name,
            "train_time_s": round(train_time, 2),
            "n_predictions": len(results),
            **mse_per_dim,
        }
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        print(f"  Saved summary   → {summary_path}")


def print_comparison(torch_results, tf_results, dataset_name):
    """Print a side-by-side MSE comparison."""
    if not torch_results or not tf_results:
        return

    print("\n" + "=" * 60)
    print(f"  COMPARISON: {dataset_name}")
    print("=" * 60)

    y_true_cols = [k for k in torch_results[0] if k.startswith("y_true_")]
    y_pred_cols = [k for k in torch_results[0] if k.startswith("y_pred_")]

    for yt, yp in zip(y_true_cols, y_pred_cols):
        dim = yt.replace("y_true_", "")
        torch_mse = np.mean([(r[yt] - r[yp]) ** 2 for r in torch_results])
        tf_mse = np.mean([(r[yt] - r[yp]) ** 2 for r in tf_results])
        print(f"  Dim {dim}:  PyTorch MSE = {torch_mse:.6f}  |  TF MSE = {tf_mse:.6f}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_datasets():
    """List all datasets in the datasets/ directory."""
    if not os.path.isdir(DATASETS_DIR):
        return []
    names = set()
    for item in os.listdir(DATASETS_DIR):
        if item.startswith("."):
            continue
        if item.endswith(".csv") or item.endswith(".pkl"):
            names.add(item[:-4])
        elif os.path.isdir(os.path.join(DATASETS_DIR, item)):
            names.add(item)
    return sorted(names)


def run_single_dataset(dataset_name, args):
    """Run the pipeline on a single dataset."""
    print(f"\n{'#' * 60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'#' * 60}")

    tasks, x_dim, y_dim = load_dataset(dataset_name, args, args.x_cols, args.y_cols)
    print(f"  Loaded {len(tasks)} task(s), x_dim={x_dim}, y_dim={y_dim}")
    print(f"  Total data points: {sum(X.shape[0] for X, Y in tasks)}")

    config = build_config(x_dim, y_dim, args)

    torch_results = None
    tf_results = None

    # PyTorch
    if args.backend in ("torch", "both"):
        try:
            torch_results, torch_time = run_pytorch(tasks, config, args)
            out_dir = os.path.join(RESULTS_DIR, dataset_name, "torch")
            save_results(torch_results, out_dir, dataset_name, "torch", torch_time)
        except Exception as e:
            print(f"\n  ⚠ PyTorch failed: {e}")
            import traceback
            traceback.print_exc()

    # TensorFlow
    if args.backend in ("tensorflow", "both"):
        try:
            tf_results, tf_time = run_tensorflow(tasks, config, args)
            out_dir = os.path.join(RESULTS_DIR, dataset_name, "tensorflow")
            save_results(tf_results, out_dir, dataset_name, "tensorflow", tf_time)
        except Exception as e:
            print(f"\n  ⚠ TensorFlow failed: {e}")
            import traceback
            traceback.print_exc()

    # Comparison
    if torch_results and tf_results:
        print_comparison(torch_results, tf_results, dataset_name)


def main():
    parser = argparse.ArgumentParser(
        description="ALPaCA Testing Pipeline — Run PyTorch & TensorFlow on CSV datasets"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (without .csv) or 'all'")
    parser.add_argument("--backend", type=str, default="both",
                        choices=["torch", "tensorflow", "both"],
                        help="Which backend(s) to run")

    # Model / training hyperparameters
    parser.add_argument("--n_context", type=int, default=5,
                        help="Number of past datapoints for context/theta (tweakable)")
    parser.add_argument("--sigma_eps", type=float, default=0.1,
                        help="Observation noise variance σ_ε")
    parser.add_argument("--train_iters", type=int, default=5000,
                        help="Number of meta-training iterations")
    parser.add_argument("--test_horizon", type=int, default=20,
                        help="Test horizon (number of query points per batch)")
    parser.add_argument("--meta_batch_size", type=int, default=10,
                        help="Meta batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--nn_layers", type=str, default="128,128,32",
                        help="Hidden layer sizes, comma-separated")
    parser.add_argument("--activation", type=str, default="tanh",
                        choices=["relu", "tanh", "sigmoid"],
                        help="Activation function")

    # Data column spec
    parser.add_argument("--x_cols", type=str, default=None,
                        help="Comma-separated column indices for x (0-indexed)")
    parser.add_argument("--y_cols", type=str, default=None,
                        help="Comma-separated column indices for y (0-indexed)")

    parser.add_argument("--use_past_n_as_state", action="store_true",
                        help="If set, reshapes the input X to be a sliding window of the past n_context raw Y observations (Control Signal Theta)")
    parser.add_argument("--calibrate_sigma", action="store_true",
                        help="If set, runs a 2-pass training to calibrate sigma_eps per Dataset (ALPaCA paper B.3)")
    parser.add_argument("--normalize", action="store_true",
                        help="If set, z-score normalizes the data to mean 0 std 1. Prevents numeric scaling crashes.")

    args = parser.parse_args()

    # Validate
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.dataset == "all":
        datasets = get_all_datasets()
        if not datasets:
            print(f"No datasets found in {DATASETS_DIR}/")
            print("Drop CSV files there and re-run.")
            sys.exit(1)
        print(f"Found {len(datasets)} dataset(s): {', '.join(datasets)}")
        for name in datasets:
            if args.backend == "both":
                import subprocess
                cmd_torch = [sys.executable, __file__, "--dataset", name, "--backend", "torch"] + \
                            [x for x in sys.argv[1:] if not x.startswith("--dataset") and x != args.dataset and not x.startswith("--backend") and x != "both"]
                cmd_tf = [sys.executable, __file__, "--dataset", name, "--backend", "tensorflow"] + \
                         [x for x in sys.argv[1:] if not x.startswith("--dataset") and x != args.dataset and not x.startswith("--backend") and x != "both"]
                subprocess.run(cmd_torch, check=False)
                subprocess.run(cmd_tf, check=False)
            else:
                run_single_dataset(name, args)
    else:
        if args.backend == "both":
            import subprocess
            cmd_torch = [sys.executable, __file__, "--dataset", args.dataset, "--backend", "torch"] + \
                        [x for x in sys.argv[1:] if not x.startswith("--dataset") and x != args.dataset and not x.startswith("--backend") and x != "both"]
            cmd_tf = [sys.executable, __file__, "--dataset", args.dataset, "--backend", "tensorflow"] + \
                     [x for x in sys.argv[1:] if not x.startswith("--dataset") and x != args.dataset and not x.startswith("--backend") and x != "both"]
            subprocess.run(cmd_torch, check=False)
            subprocess.run(cmd_tf, check=False)
        else:
            run_single_dataset(args.dataset, args)

    print("\n✓ Done. Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
