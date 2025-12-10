import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import time


def set_seed(seed: int = 42):
    """
    Set a random seed to ensure reproducibility.
    The default value is 42.
    """
    random.seed(seed)
    np.random.seed(seed)


def select_preprocessing():
    """
    Allows the user to select the type of normalization to use
    through a terminal menu.
    """
    while True:
        print("[Preprocessing Selection Menu]")
        print("[1] StandardScaler")
        print("[2] RobustScaler")
        print("[3] MinMaxScaler")
        opt = input("Choose an option: ").strip()

        if opt in {"1", "2", "3"}:
            types = {
                "1": "StandardScaler",
                "2": "RobustScaler",
                "3": "MinMaxScaler"
            }
            print(f"✅ Selected: {types[opt]}")
            return int(opt)
        else:
            print("⚠️  Invalid option, please try again...")
            time.sleep(0.5)
            os.system("clear" if os.name != "nt" else "cls")


def select_prior_function():
    """
    Allows the user to select the prior sampling function
    through a terminal menu.
    """
    while True:
        print("[Sampling Function Selection Menu]")
        print("[1] Gaussian")
        print("[2] Laplace")
        print("[3] Student-t")
        opt = input("Choose an option: ").strip()

        if opt in {"1", "2", "3"}:
            types = {
                "1": "Gaussian",
                "2": "Laplace",
                "3": "Student-t"
            }
            print(f"✅ Selected: {types[opt]}")
            return int(opt)
        else:
            print("⚠️  Invalid option, please try again...")
            time.sleep(0.5)
            os.system("clear" if os.name != "nt" else "cls")


def plot_posterior_distributions(
    samples, w_mean, w_std, n_dim,
    scaler_opt: int, prior_func_opt: int,
    col_names, save_dir: str = 'reports/figures'
):
    """
    Plot posterior distributions of parameters obtained via MCMC, along with
    their theoretical normal distributions for comparison.
    """
    n_rows = 3
    n_cols = 4

    # Map scaler option to name
    if scaler_opt == 1:
        scaler_name = 'StandardScaler'
    elif scaler_opt == 2:
        scaler_name = 'RobustScaler'
    else:
        scaler_name = 'MinMaxScaler'

    # Map prior option to name
    if prior_func_opt == 1:
        prior_func_name = 'Gaussian'
    elif prior_func_opt == 2:
        prior_func_name = 'Laplace'
    else:
        prior_func_name = 'Student-t'

    report_name = f'{save_dir}/{scaler_name}_{prior_func_name}_posterior_distributions.png'

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 12))
    axes_flat = axes.flatten()

    # Plot each dimension
    for i in range(n_dim):
        ax = axes_flat[i]

        # Histogram of MCMC samples
        ax.hist(samples[:, i], bins=30, density=True, alpha=0.6, label='MCMC Samples')

        # Theoretical normal distribution
        x_vals = np.linspace(w_mean[i] - 3 * w_std[i],
                             w_mean[i] + 3 * w_std[i], 100)
        ax.plot(
            x_vals,
            norm.pdf(x_vals, w_mean[i], w_std[i]),
            label='Theoretical Normal Distribution',
            color='red',
            linestyle='--'
        )

        ax.set_title(f'Posterior Distribution of {col_names[i]}')
        ax.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(report_name), exist_ok=True)
    plt.savefig(report_name)
    plt.close(fig)
