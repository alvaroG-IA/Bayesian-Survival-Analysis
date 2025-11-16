import numpy as np
from scipy.stats import norm

def sigmoid(z):
    """
    Función sigmoide
    """
    return 1 / (1 + np.exp(-z))

def log_likelihood(w, X, y):
    """
    Log-verosimilitud para regresión logística
    """
    z = np.dot(X, w)
    return np.sum(y * z - np.log1p(np.exp(-np.abs(z))) - np.maximum(z, 0))

def log_prior(w, mu=0, sigma=10):
    """
    Log-prior gaussiano
    """
    return np.sum(norm.logpdf(w, mu, sigma))

def analyze_posterior(samples, confidence_level=95):
    """
    Función que nos ayudará a analizar los valores obtenidos por las muestras, para conocer:
    - El valor medio
    - Su desviación estádar, la cual no ayudará a ver como de fiables es ese valor
    - Intervalos de credibilidad, por defecto se usará un 95% de fiabilidad
    """
    
    # Calcular la media (Valor Central)
    w_mean = np.mean(samples, axis=0)
    
    # Calcular la desviación estándar (Incertidumbre / Fiabilidad)
    w_std = np.std(samples, axis=0)
    
    # Calcular los límites del Intervalo de Credibilidad (CI, ej. 95%)
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    
    w_ci_lower = np.percentile(samples, lower_percentile, axis=0)
    w_ci_upper = np.percentile(samples, upper_percentile, axis=0)
    
    return {
        'mean': w_mean,
        'std': w_std,
        'ci_lower': w_ci_lower,
        'ci_upper': w_ci_upper
    }
