import os
import pandas as pd
import numpy as np

RESULTS_DIR = 'results'

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory '{RESULTS_DIR}' not found.")
        return

    datasets = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    
    print("=" * 70)
    print(f"{'Dataset':<15} | {'Column':<12} | {'Mean Abs Diff':<15} | {'Max Abs Diff':<15}")
    print("=" * 70)

    for dataset in sorted(datasets):
        torch_path = os.path.join(RESULTS_DIR, dataset, 'torch', 'predictions.csv')
        tf_path = os.path.join(RESULTS_DIR, dataset, 'tensorflow', 'predictions.csv')

        if not os.path.exists(torch_path) or not os.path.exists(tf_path):
            continue

        df_torch = pd.read_csv(torch_path)
        df_tf = pd.read_csv(tf_path)
        
        # Merge on task and timestep to ensure perfect row alignment
        try:
            df = pd.merge(df_torch, df_tf, on=['task', 'timestep'], suffixes=('_torch', '_tf'))
        except KeyError:
            print(f"{dataset:<15} | Error: missing task/timestep columns")
            continue
            
        if len(df) == 0:
            print(f"{dataset:<15} | No overlapping data")
            continue

        # Find the prediction columns
        y_pred_cols = [c.replace('_torch', '') for c in df.columns if c.startswith('y_pred_') and c.endswith('_torch')]
        var_pred_cols = [c.replace('_torch', '') for c in df.columns if c.startswith('var_pred_') and c.endswith('_torch')]

        # Calculate metrics for predicted means
        for col in y_pred_cols:
            diff = np.abs(df[f"{col}_torch"] - df[f"{col}_tf"])
            mean_diff = diff.mean()
            max_diff = diff.max()
            print(f"{dataset:<15} | {col:<12} | {mean_diff:<15.6e} | {max_diff:<15.6e}")

        # Calculate metrics for predicted variances
        for col in var_pred_cols:
            diff = np.abs(df[f"{col}_torch"] - df[f"{col}_tf"])
            mean_diff = diff.mean()
            max_diff = diff.max()
            print(f"{dataset:<15} | {col:<12} | {mean_diff:<15.6e} | {max_diff:<15.6e}")
            
    print("=" * 70)

if __name__ == '__main__':
    main()
