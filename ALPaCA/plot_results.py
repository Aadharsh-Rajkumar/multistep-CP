import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
PLOTS_DIR = 'plots'

def plot_dataset(dataset_name):
    torch_file = os.path.join(RESULTS_DIR, dataset_name, 'torch', 'predictions.csv')
    tf_file = os.path.join(RESULTS_DIR, dataset_name, 'tensorflow', 'predictions.csv')

    if not os.path.exists(torch_file) or not os.path.exists(tf_file):
        print(f"Skipping {dataset_name}: Missing result files.")
        return

    # Create subfolder inside the dataset results folder
    dataset_plots_dir = os.path.join(RESULTS_DIR, dataset_name, 'plots')
    if not os.path.exists(dataset_plots_dir):
        os.makedirs(dataset_plots_dir)

    df_torch = pd.read_csv(torch_file)
    df_tf = pd.read_csv(tf_file)

    # Filter for the first task
    task_id = df_torch['task'].unique()[0]
    df_torch = df_torch[df_torch['task'] == task_id].sort_values('timestep')
    df_tf = df_tf[df_tf['task'] == task_id].sort_values('timestep')
    
    limit = 500
    if len(df_torch) > limit:
        df_torch = df_torch.iloc[:limit]
        df_tf = df_tf.iloc[:limit]

    plt.figure(figsize=(12, 6))
    
    t = df_torch['timestep']

    # Ground Truth
    plt.plot(t, df_torch['y_true_0'], 'k--', alpha=0.5, label='Actual (Ground Truth)', linewidth=1)

    # PyTorch: mean and 95% CI
    mu_torch = df_torch['y_pred_0']
    std_torch = np.sqrt(df_torch['var_pred_0'])
    plt.plot(t, mu_torch, color='blue', label='PyTorch Prediction', linewidth=1.5)
    plt.fill_between(t, mu_torch - 2*std_torch, mu_torch + 2*std_torch, color='blue', alpha=0.2, label='PyTorch 95% CI')

    # TensorFlow: mean and 95% CI
    mu_tf = df_tf['y_pred_0']
    std_tf = np.sqrt(df_tf['var_pred_0'])
    plt.plot(t, mu_tf, color='red', label='TensorFlow Prediction', linewidth=1.5, alpha=0.8)
    plt.fill_between(t, mu_tf - 2*std_tf, mu_tf + 2*std_tf, color='red', alpha=0.15, label='TensorFlow 95% CI')

    # Clean up title: e.g. Amazon-fc -> Amazon, cases -> COVID
    display_name = dataset_name.replace('-fc', '').replace('_proc', '').replace('_ca', '')
    if display_name.lower() == 'cases':
        display_name = 'COVID'
    
    plt.title(f"ALPaCA Backend Comparison: {display_name}", fontsize=14)
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Value (Normalized)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save as <dataset_name>.png in the plots subfolder
    filename = f"{display_name.lower()}.png"
    save_path = os.path.join(dataset_plots_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated plot: {save_path}")

def main():
    datasets = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    
    for d in sorted(datasets):
        plot_dataset(d)

if __name__ == '__main__':
    main()
