import numpy as np
from scipy.stats import norm, laplace, t


def sigmoid(z):
    """
    Sigmoid function.
    """
    return 1 / (1 + np.exp(-z))


def log_likelihood(w, X, y):
    """
    Log-likelihood for logistic regression.
    """
    z = X @ w
    return np.sum(y * z - np.log(1 + np.exp(z)))


def log_prior_normal(w, mu=0, sigma=1):
    """
    Gaussian log-prior.
    """
    return np.sum(norm.logpdf(w, mu, sigma))


def log_prior_laplace(w, mu=0, b=0.5):
    """
    Laplace log-prior.
    """
    return np.sum(laplace.logpdf(w, loc=mu, scale=b))


def log_prior_student_t(w, df=3, loc=0, scale=2.5):
    """
    Student-t log-prior.
    """
    return np.sum(t.logpdf(w, df=df, loc=loc, scale=scale))


def analyze_posterior(samples, confidence_level=95):
    """
    Function that helps analyze the sampled posterior values.
    Computes:
    - The mean value for each parameter.
    - Its standard deviation, indicating the reliability/uncertainty of the mean.
    - Credible intervals (CI), using a default confidence level of 95%.
    """
    
    # Compute the mean (Central Value)
    w_mean = np.mean(samples, axis=0)
    
    # Compute the standard deviation (Uncertainty / Reliability)
    w_std = np.std(samples, axis=0)
    
    # Compute the bounds of the Credible Interval (CI), e.g. 95%
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
