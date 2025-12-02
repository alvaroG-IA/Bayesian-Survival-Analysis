import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import time


def set_seed(seed:int = 42):
    random.seed(seed)
    np.random.seed(seed)


def seleccionar_preprocesado():
    """
    Función encargada de permitir al usuario seleccionar el algoritmo de manifold learning
    a utilizar mediante un menu por terminal
    """
    while True:
        print("[Menú selección del tipo de preprocesado]")
        print("[1] StandardScaler")
        print("[2] RobustScaler")
        print("[3] MinMaxScaler")
        opt = input("Opción a elegir: ").strip()

        if opt in {"1", "2", "3"}:
            tipos = {
                "1": "StandardScaler",
                "2": "RobustScaler",
                "3": "MinMaxScaler"
            }
            print(f"✅ Seleccionado: {tipos[opt]}")
            return int(opt)
        else:
            print("⚠️  Opción no válida, intenta de nuevo...")
            time.sleep(0.5)
            os.system("clear" if os.name != "nt" else "cls")


def seleccionar_prior_func():
    """
    Función encargada de permitir al usuario seleccionar el algoritmo de manifold learning
    a utilizar mediante un menu por terminal
    """
    while True:
        print("[Menú selección de la función de muestreo]")
        print("[1] Gaussiana")
        print("[2] Laplace")
        print("[3] Student-t")
        opt = input("Opción a elegir: ").strip()

        if opt in {"1", "2", "3"}:
            tipos = {
                "1": "Gaussiana",
                "2": "Laplace",
                "3": "Student-t"
            }
            print(f"✅ Seleccionado: {tipos[opt]}")
            return int(opt)
        else:
            print("⚠️  Opción no válida, intenta de nuevo...")
            time.sleep(0.5)
            os.system("clear" if os.name != "nt" else "cls")


def plot_post_distribuitions(samples, w_mean, w_std, n_dim,
                             scaler_opt: int, prior_func_opt: int,
                             col_names, save_dir: str = 'reports/figures'):
    # --- CONFIGURACIÓN DE GRILLE FIJA ---
    n_rows = 3
    n_cols = 4

    # Lógica de nombres (se mantiene para el filename)
    if scaler_opt == 1:
        scaler_name = 'StandardScaler'
    elif scaler_opt == 2:
        scaler_name = 'RobustScaler'
    else:
        scaler_name = 'MinMaxScaler'

    if prior_func_opt == 1:
        prior_func_name = 'Gaussiana'
    elif prior_func_opt == 2:
        prior_func_name = 'Laplace'
    else:
        prior_func_name = 'Student-t'

    report_name = f'{save_dir}/{scaler_name}_{prior_func_name}_post_distribuitions.png'

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 12))

    axes_flat = axes.flatten()

    for i in range(n_dim):
        ax = axes_flat[i]

        ax.hist(samples[:, i], bins=30, density=True, alpha=0.6, label='Muestras MCMC')

        x_vals = np.linspace(w_mean[i] - 3 * w_std[i], w_mean[i] + 3 * w_std[i], 100)
        ax.plot(x_vals, norm.pdf(x_vals, w_mean[i], w_std[i]),
                label='Distribución Normal Teórica', color='red', linestyle='--')

        ax.set_title(f'Distr. posterior de {col_names[i]}')
        ax.legend(loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(report_name), exist_ok=True)
    plt.savefig(report_name)
    plt.close(fig)
