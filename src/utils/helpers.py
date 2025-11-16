import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os # Necesario para la gestión de directorios

def plot_post_distribuitions(samples, w_mean, w_std, n_dim, col_names, save_dir: str='reports/figures/post_distributions.png'):

    n_rows = 2
    n_cols = (n_dim + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 8))
    
    if n_dim == 1:
        axes = [axes]

    for i in range(n_dim):
        
        r = i // n_cols 
        c = i % n_cols  
        
        ax = axes[r, c]

        ax.hist(samples[:, i], bins=30, density=True, alpha=0.6, label='Muestras MCMC')
        
        x_vals = np.linspace(w_mean[i] - 3 * w_std[i], w_mean[i] + 3 * w_std[i], 100)
        ax.plot(x_vals, norm.pdf(x_vals, w_mean[i], w_std[i]), 
                     label='Distribución Normal Teórica', color='red', linestyle='--')
        
        ax.set_title(f'Distribución posterior de {col_names[i]}')
        ax.legend()
    
    plt.tight_layout() 
    
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    plt.savefig(save_dir)
    plt.close(fig)