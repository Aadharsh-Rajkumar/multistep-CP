#!/usr/bin/env python3

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PYTORCH_DIR = os.path.join(BASE_DIR, "pytorch")

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (reused from run_tests.py with minor adaptations)
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_tasks(dataset_name):
    """
    Load a dataset and return raw 1D time-series per task.
    Returns list of 1D numpy arrays (one per task/group).
    """
    csv_path = os.path.join(DATASETS_DIR, f"{dataset_name}.csv")
    pkl_path = os.path.join(DATASETS_DIR, f"{dataset_name}.pkl")

    target_candidates = ['target', 'actual', 'log_value']
    group_candidates = ['item_id', 'geo_value', 'site_id']

    def extract_series(df):
        """Extract target column and optional grouping, return list of 1D arrays."""
        y_col = None
        for col in target_candidates:
            if col in df.columns:
                y_col = col
                break

        group_col = None
        for col in group_candidates:
            if col in df.columns:
                group_col = col
                break

        def get_y(sub_df):
            if y_col is not None:
                sub_df = sub_df.dropna(subset=[y_col])
                return sub_df[y_col].values.astype(np.float64)
            else:
                numeric = sub_df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
                if len(numeric.columns) > 0:
                    return numeric.iloc[:, -1].values.astype(np.float64)
                return sub_df.iloc[:, -1].values.astype(np.float64)

        if group_col is not None:
            series_list = []
            for _, group in df.groupby(group_col):
                y = get_y(group)
                y = np.nan_to_num(y, nan=0.0)
                if len(y) > 0:
                    series_list.append(y)
            return series_list
        else:
            y = get_y(df)
            y = np.nan_to_num(y, nan=0.0)
            return [y]

    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        return extract_series(df)

    elif os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            if 'X' in data and 'Y' in data:
                return [data['Y'].flatten().astype(np.float64)]
            df = pd.DataFrame(data.get('data', data))
        else:
            df = pd.DataFrame(data)
        return extract_series(df)

    else:
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found in {DATASETS_DIR}")


def get_all_datasets():
    """Find all dataset names in the datasets directory."""
    names = set()
    for f in os.listdir(DATASETS_DIR):
        if f.startswith('.'):
            continue
        if f.endswith('.csv') or f.endswith('.pkl'):
            name = f.rsplit('.', 1)[0]
            if name != 'dummy_test':
                names.add(name)
    return sorted(names)


# ═══════════════════════════════════════════════════════════════════════════════
#  FORECASTER MODELS (via darts)
# ═══════════════════════════════════════════════════════════════════════════════

def train_forecaster(series_d1, forecaster_name):
    """
    Train a darts forecaster on D1.
    Returns the fitted model.
    """
    from darts import TimeSeries
    from darts.models import LinearRegressionModel, Theta

    ts = TimeSeries.from_values(series_d1.reshape(-1, 1))

    if forecaster_name == 'ar':
        model = LinearRegressionModel(lags=3, output_chunk_length=1)
        model.fit(ts)

    elif forecaster_name == 'theta':
        # Use additive mode to handle datasets with negative/zero values
        from darts.utils.utils import SeasonalityMode
        model = Theta(theta=2, season_mode=SeasonalityMode.ADDITIVE)
        model.fit(ts)

    elif forecaster_name == 'prophet':
        try:
            from darts.models import Prophet
        except ImportError:
            raise ImportError("Prophet not installed. Run: pip install prophet")
        model = Prophet()
        model.fit(ts)

    elif forecaster_name == 'transformer':
        from darts.models import TransformerModel
        model = TransformerModel(
            input_chunk_length=10,
            output_chunk_length=1,
            n_epochs=20,
            model_name="transformer_forecaster",
            force_reset=True,
            save_checkpoints=False,
            log_tensorboard=False,
            pl_trainer_kwargs={
                "enable_progress_bar": False,
                "accelerator": "cpu",
            },
        )
        model.fit(ts)
    else:
        raise ValueError(f"Unknown forecaster: {forecaster_name}")

    return model


def predict_on_splits(model, series, idx1, idx2, forecaster_name):
    """
    Generate predictions for D2 and D3.
    """
    from darts import TimeSeries

    n_d2 = idx2 - idx1
    n_d3 = len(series) - idx2
    n_target = n_d2 + n_d3

    if forecaster_name in ('ar', 'transformer'):
        ts_full = TimeSeries.from_values(series.reshape(-1, 1))

        preds = model.historical_forecasts(
            ts_full,
            start=idx1,
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=False,
        )
        y_hat_all = preds.values().flatten()

        if len(y_hat_all) < n_target:
            pad = np.full(n_target - len(y_hat_all), y_hat_all[0] if len(y_hat_all) > 0 else 0)
            y_hat_all = np.concatenate([pad, y_hat_all])
        
        y_hat_all = y_hat_all[:n_target]
        y_hat_d2 = y_hat_all[:n_d2]
        y_hat_d3 = y_hat_all[n_d2:]
        return y_hat_d2, y_hat_d3

    elif forecaster_name in ('theta', 'prophet'):
        preds = model.predict(n_target)
        y_hat_all = preds.values().flatten()[:n_target]
        y_hat_d2 = y_hat_all[:n_d2]
        y_hat_d3 = y_hat_all[n_d2:]
        return y_hat_d2, y_hat_d3


# ═══════════════════════════════════════════════════════════════════════════════
#  ALPaCA ON ERRORS
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorTaskDataset:
    """Wraps error sequences into ALPaCA's sample() interface."""
    def __init__(self, tasks, n_context):
        self.tasks = tasks
        self.n_context = n_context
        self.x_dim = tasks[0][0].shape[1]
        self.y_dim = tasks[0][1].shape[1]

    def sample(self, n_funcs, n_samples):
        x = np.zeros((n_funcs, n_samples, self.x_dim))
        y = np.zeros((n_funcs, n_samples, self.y_dim))
        for i in range(n_funcs):
            task_idx = np.random.randint(len(self.tasks))
            X_task, Y_task = self.tasks[task_idx]
            T = X_task.shape[0]
            if T >= n_samples:
                start_idx = np.random.randint(0, T - n_samples + 1)
                inds = np.arange(start_idx, start_idx + n_samples)
            else:
                # Fallback if task is shorter than required horizon
                inds = np.random.choice(T, n_samples, replace=True)
            x[i] = X_task[inds]
            y[i] = Y_task[inds]
        return x, y


def normalize_errors(errors, normalize=True, E_mean=None, E_std=None):
    if normalize:
        if E_mean is None:
            E_mean = np.mean(errors)
            E_std = np.std(errors)
            if E_std == 0: E_std = 1.0
        return (errors - E_mean) / E_std, E_mean, E_std
    return errors, 0.0, 1.0

def build_error_tasks(E_norm, n_context):
    """Build tasks from a 1D sequence of errors."""
    T = E_norm.shape[0]
    if T <= n_context:
        # This can happen if the D2 errors are too short for n_context
        # In this case, we'll just return an empty list or handle it upstream
        return [] 
    
    X = np.zeros((T - n_context, n_context))
    for i in range(T - n_context):
        X[i] = E_norm[i:i + n_context].flatten()
    Y = E_norm[n_context:]
    return [(X, Y)]


def build_alpaca_config(n_context, args):
    """Build ALPaCA config for error prediction."""
    nn_layers = [int(x) for x in args.nn_layers.split(",")]
    return {
        "lr": args.lr,
        "x_dim": n_context,
        "y_dim": 1,
        "sigma_eps": args.sigma_eps,
        "data_horizon": n_context,
        "test_horizon": max(args.test_horizon, 1),
        "meta_batch_size": args.meta_batch_size,
        "nn_layers": nn_layers,
        "activation": args.activation,
        "phase_range": [0, 3.14],
        "freq_range": [0.999, 1.0],
        "amp_range": [0.1, 5.0],
        "x_range": [-5.0, 5.0],
    }


def run_alpaca_on_errors(errors_d2, errors_d3, config, args):
    """Train ALPaCA on D2 errors and return eval predictions on D3 errors."""
    sys.path.insert(0, PYTORCH_DIR)
    from main.alpaca import ALPaCA
    sys.path.pop(0)

    # Normalize D2
    E_norm_d2, E_mean, E_std = normalize_errors(errors_d2, args.normalize)
    error_tasks_d2 = build_error_tasks(E_norm_d2.reshape(-1, 1), args.n_context)
    
    if not error_tasks_d2:
        raise ValueError(f"D2 error sequence too short ({len(errors_d2)}) for n_context ({args.n_context}) to build tasks.")

    dataset = ErrorTaskDataset(error_tasks_d2, args.n_context)

    # ── Sigma Epsilon Calibration ──
    if args.calibrate_sigma:
        print("    [Calibration] Training base model...")
        calib_config = config.copy()
        calib_model = ALPaCA(calib_config)
        calib_model.construct_model()
        calib_iters = max(100, args.train_iters // 2)
        calib_model.train_model(dataset, calib_iters)

        print("    [Calibration] Evaluating MSE on D2...")
        squared_errors = []
        for X, Y in error_tasks_d2:
            T = X.shape[0]
            for j in range(T):
                start_c = max(0, j - args.n_context)
                if j == 0:
                    x_ctx = np.zeros((1, 0, X.shape[1]))
                    y_ctx = np.zeros((1, 0, Y.shape[1]))
                else:
                    x_ctx = X[start_c:j].reshape(1, j - start_c, -1)
                    y_ctx = Y[start_c:j].reshape(1, j - start_c, -1)
                x_test_j = X[j].reshape(1, 1, -1)
                mu_pred, _ = calib_model.test(x_ctx, y_ctx, x_test_j)
                se = (mu_pred[0, 0, 0] - Y[j, 0]) ** 2
                squared_errors.append([se])

        all_se = np.vstack(squared_errors)
        mse_per_dim = [max(1e-6, float(m)) for m in np.mean(all_se, axis=0)]
        print(f"    [Calibration] Sigma Epsilon: {mse_per_dim}")
        config["sigma_eps"] = mse_per_dim

    # ── Training ──
    model = ALPaCA(config)
    model.construct_model()
    print(f"    Training ALPaCA for {args.train_iters} iterations on D2...")
    t0 = time.time()
    model.train_model(dataset, args.train_iters)
    train_time = time.time() - t0
    print(f"    Training took {train_time:.1f}s")

    # ── Evaluation on D3 ──
    print(f"    Evaluating ALPaCA sequentially on D3...")
    results = []
    
    # Concatenate the last n_context elements from D2 and all of D3
    errors_eval_raw = np.concatenate([errors_d2[-args.n_context:], errors_d3])
    E_norm_eval, _, _ = normalize_errors(errors_eval_raw, args.normalize, E_mean, E_std)
    error_tasks_d3 = build_error_tasks(E_norm_eval.reshape(-1, 1), args.n_context)
    
    X_eval, Y_eval = error_tasks_d3[0]
    T_eval = X_eval.shape[0]
    
    for j in range(T_eval):
        start_c = max(0, j - args.n_context)
        x_ctx = X_eval[start_c:j].reshape(1, j - start_c, -1) if j > 0 else np.zeros((1, 0, X_eval.shape[1]))
        y_ctx = Y_eval[start_c:j].reshape(1, j - start_c, -1) if j > 0 else np.zeros((1, 0, Y_eval.shape[1]))
        x_test_j = X_eval[j].reshape(1, 1, -1)

        mu_pred, sig_pred = model.test(x_ctx, y_ctx, x_test_j)

        results.append({
            "timestep": j, 
            "error_true": float(Y_eval[j, 0]),
            "error_pred": float(mu_pred[0, 0, 0]),
            "error_var": float(sig_pred[0, 0, 0, 0]),
        })

    return results, E_mean, E_std, train_time


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def get_display_name(dataset_name):
    """Clean dataset name for display."""
    name = dataset_name.replace('-fc', '').replace('_proc', '').replace('_ca', '')
    if name.lower() == 'cases':
        return 'COVID'
    return name.upper() if len(name) <= 5 else name.title()


def generate_plots(dataset_name, forecaster_name, d3_values, y_hat_d3, errors_d3,
                   alpaca_results, y_mean_norm, y_std_norm, save_dir):
    """Generate three plots: confidence intervals, raw errors, ALPaCA error predictions."""
    os.makedirs(save_dir, exist_ok=True)
    display = get_display_name(dataset_name)

    limit = 500
    n_alpaca = len(alpaca_results)
    plot_end = min(n_alpaca, limit)
    
    d3_slice = d3_values[:plot_end]
    yhat_slice = y_hat_d3[:plot_end]

    mu_errors = np.array([r['error_pred'] for r in alpaca_results[:plot_end]])
    var_errors = np.array([r['error_var'] for r in alpaca_results[:plot_end]])
    mu_errors_raw = mu_errors * y_std_norm + y_mean_norm
    std_errors_raw = np.sqrt(np.abs(var_errors)) * y_std_norm

    reconstructed = yhat_slice + mu_errors_raw
    ci_lower = yhat_slice + mu_errors_raw - 2 * std_errors_raw
    ci_upper = yhat_slice + mu_errors_raw + 2 * std_errors_raw

    t = np.arange(len(d3_slice))
    plt.figure(figsize=(14, 6))
    plt.plot(t, d3_slice, 'k-', alpha=0.6, linewidth=1, label='Actual (D3)')
    plt.plot(t, yhat_slice, 'g--', alpha=0.5, linewidth=1, label=f'{forecaster_name.upper()} Forecast')
    plt.plot(t, reconstructed, 'b-', linewidth=1.5, label='Corrected (ŷ + ALPaCA)')
    plt.fill_between(t, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% CI')
    plt.title(f"{display} — {forecaster_name.upper()} + ALPaCA Confidence Intervals (Test / D3)", fontsize=14)
    plt.xlabel("Timestep (D3)", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_dir, f"{display.lower()}_confidence.png"), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(14, 4))
    err_plot = errors_d3[:plot_end]
    t_err = np.arange(len(err_plot))
    plt.plot(t_err, err_plot, 'r-', alpha=0.7, linewidth=1)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.fill_between(t_err, 0, err_plot, color='red', alpha=0.15)
    plt.title(f"{display} — {forecaster_name.upper()} Prediction Errors (y − ŷ)", fontsize=14)
    plt.xlabel("Timestep (D3)", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_dir, f"{display.lower()}_errors.png"), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(14, 5))
    true_errs = np.array([r['error_true'] for r in alpaca_results[:plot_end]])
    t_a = np.arange(len(true_errs))
    plt.plot(t_a, true_errs, 'k-', alpha=0.6, linewidth=1, label='True Error (normalized)')
    plt.plot(t_a, mu_errors, 'b-', linewidth=1.5, label='ALPaCA Predicted Error')
    plt.fill_between(t_a, mu_errors - 2*np.sqrt(np.abs(var_errors)), mu_errors + 2*np.sqrt(np.abs(var_errors)),
                     color='blue', alpha=0.2, label='95% CI')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title(f"{display} — ALPaCA Error Predictions ({forecaster_name.upper()} residuals)", fontsize=14)
    plt.xlabel("Timestep (D3)", fontsize=12)
    plt.ylabel("Error (normalized)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_dir, f"{display.lower()}_alpaca_errors.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Plots saved to {save_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(dataset_name, forecaster_name, args):
    """Run the full D1 → Forecaster → D2/D3 → Errors → ALPaCA pipeline."""
    print(f"\n{'#' * 60}")
    print(f"  Dataset: {dataset_name} | Forecaster: {forecaster_name.upper()}")
    print(f"{'#' * 60}")

    raw_tasks = load_raw_tasks(dataset_name)
    print(f"  Loaded {len(raw_tasks)} task(s)")

    all_alpaca_results = []
    all_forecaster_results = []

    for task_idx, series in enumerate(raw_tasks):
        T = len(series)
        if T < 30:
            print(f"  Task {task_idx}: Too short ({T} points). Skipping.")
            continue

        split1 = int(T * args.split_ratios[0])
        split2 = int(T * (args.split_ratios[0] + args.split_ratios[1]))
        d1 = series[:split1]
        d2 = series[split1:split2]
        d3 = series[split2:]
        print(f"  Task {task_idx}: T={T}, D1={len(d1)}, D2={len(d2)}, D3={len(d3)}")

        print(f"    Training {forecaster_name.upper()} on D1...")
        t0 = time.time()
        model = train_forecaster(d1, forecaster_name)
        print(f"    Forecaster trained in {time.time()-t0:.1f}s")

        print(f"    Generating forecaster predictions on out-of-sample data...")
        y_hat_d2, y_hat_d3 = predict_on_splits(model, series, split1, split2, forecaster_name)

        errors_d2 = d2[:len(y_hat_d2)] - y_hat_d2
        errors_d3 = d3[:len(y_hat_d3)] - y_hat_d3
        print(f"    Error stats (D2): mean={errors_d2.mean():.4f}, std={errors_d2.std():.4f}")

        for j in range(len(y_hat_d3)):
            all_forecaster_results.append({
                "task": task_idx,
                "timestep": split2 + j,
                "y_true": float(d3[j]),
                "y_hat": float(y_hat_d3[j]),
                "error": float(errors_d3[j]),
            })

        try:
            config = build_alpaca_config(args.n_context, args)
            alpaca_results, E_mean, E_std, _ = run_alpaca_on_errors(errors_d2, errors_d3, config, args)
            for r in alpaca_results:
                r["task"] = task_idx
                r["timestep"] = split2 + r["timestep"]
            all_alpaca_results.extend(alpaca_results)
        except ValueError as e:
            print(f"    {e}. Skipping task {task_idx}.")
            continue

        save_dir = os.path.join(RESULTS_DIR, dataset_name, 'forecaster_pipeline', forecaster_name, 'plots')
        generate_plots(
            dataset_name, forecaster_name,
            d3, y_hat_d3, errors_d3,
            alpaca_results, float(E_mean), float(E_std),
            save_dir
        )

    csv_dir = os.path.join(RESULTS_DIR, dataset_name, 'forecaster_pipeline', forecaster_name)
    os.makedirs(csv_dir, exist_ok=True)

    if all_forecaster_results:
        pd.DataFrame(all_forecaster_results).to_csv(
            os.path.join(csv_dir, 'forecaster_predictions_d3.csv'), index=False)
        pd.DataFrame(all_forecaster_results).to_csv(
            os.path.join(csv_dir, 'errors_d3.csv'), index=False)
    if all_alpaca_results:
        pd.DataFrame(all_alpaca_results).to_csv(
            os.path.join(csv_dir, 'alpaca_error_predictions_d3.csv'), index=False)

    print(f"  ✓ Results saved to {csv_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Forecaster → ALPaCA Error Prediction Pipeline"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name or 'all'")
    parser.add_argument("--forecaster", nargs="+", type=str, default=["ar"],
                        choices=["ar", "theta", "prophet", "transformer"],
                        help="Forecaster model(s) to use")
    parser.add_argument("--split_ratios", nargs=2, type=float, default=[0.4, 0.3],
                        help="Fractions of data for D1 (forecaster train) and D2 (ALPaCA train). D3 gets rest.")

    # ALPaCA hyperparameters
    parser.add_argument("--n_context", type=int, default=10,
                        help="Number of past errors as control signal")
    parser.add_argument("--sigma_eps", type=float, default=0.1,
                        help="Base observation noise variance")
    parser.add_argument("--train_iters", type=int, default=500,
                        help="ALPaCA meta-training iterations")
    parser.add_argument("--test_horizon", type=int, default=20,
                        help="Test horizon")
    parser.add_argument("--meta_batch_size", type=int, default=10,
                        help="Meta batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--nn_layers", type=str, default="128,128,32",
                        help="Hidden layer sizes")
    parser.add_argument("--activation", type=str, default="tanh",
                        choices=["relu", "tanh", "sigmoid"])

    parser.add_argument("--calibrate_sigma", action="store_true",
                        help="Auto-calibrate sigma_eps via 2-pass")
    parser.add_argument("--normalize", action="store_true",
                        help="Z-score normalize errors")

    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Determine datasets
    if args.dataset == "all":
        datasets = get_all_datasets()
    else:
        datasets = [args.dataset]

    print(f"Datasets: {datasets}")
    print(f"Forecasters: {args.forecaster}")
    print()

    for ds in datasets:
        for fc in args.forecaster:
            try:
                run_pipeline(ds, fc, args)
            except Exception as e:
                print(f"\n  ✗ Error on {ds}/{fc}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("  All pipelines complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
