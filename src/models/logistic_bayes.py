import numpy as np
from src.utils.math_utils import log_likelihood, log_prior, analyze_posterior, sigmoid
from src.utils.mcmc_utils import metropolis_hastings
from sklearn.linear_model import LogisticRegression

class LogisticBayesModel:
    def __init__(self, prior_mu=0, prior_sigma=10):
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.samples_ = None
        self.w_mean_ = None
        self.w_std_ = None
        self.w_ci_ = None

    def fit(self, X, y, iterations=6000, burn_in=1000, proposal_width=0.1):
        
        lr = LogisticRegression()
        lr.fit(X, y)
        initial_w = lr.coef_[0]

        prior_func = lambda w: log_prior(w, self.prior_mu, self.prior_sigma)

        self.samples_, self.acceptance_ratio = metropolis_hastings(
            log_likelihood_func=log_likelihood,
            log_prior_func=prior_func,
            initial_w=initial_w,
            X=X,
            y=y,
            iterations=iterations,
            burn_in=burn_in,
            proposal_width=proposal_width
        )

        stats = analyze_posterior(samples=self.samples_)

        self.w_mean_ = stats['mean']
        self.w_std_ = stats['std']
        self.w_ci_ = (stats['ci_lower'], stats['ci_upper'])

        return self
    
    def predict_proba(self, data):
        if self.w_mean_ is None:
            raise RuntimeError("Debes de entrenar el modelo .fit() antes de predecir")
        z = np.dot(data, self.w_mean_)
        prob = sigmoid(z)
        return prob
    
    def predict(self, data, threshold=0.5):
        probs = self.predict_proba(data)
        return (probs >= threshold).astype(int)