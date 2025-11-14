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
    return np.sum(y * z - np.log(1 + np.exp(z)))

def log_prior(w, mu=0, sigma=10):
    """
    Log-prior gaussiano
    """
    return np.sum(norm.logpdf(w, mu, sigma))
