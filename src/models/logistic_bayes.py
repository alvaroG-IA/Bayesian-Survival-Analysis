import numpy as np
from utils.math_utils import log_likelihood, log_prior
from utils.mcmc_utils import metropolis_hastings
from sklearn.linear_model import LogisticRegression

class LogisticBayesModel:
    def __init__(self, prior_mu=0, prior_sigma=10):
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def fit(self, X, y, iterations=6000, burn_in=1000, proposal_width=0.1):
        
        lr = LogisticRegression()
        lr.fit(X, y)
        initial_w = lr.coef_[0]

        prior_func = lambda w: log_prior(w, self.prior_mu, self.prior_sigma)

        self.samples = metropolis_hastings(
            log_likelihood_func=log_likelihood,
            log_prior_func=prior_func,
            initial_w=initial_w,
            X=X,
            y=y,
            iterations=iterations,
            burn_in=burn_in,
            proposal_width=proposal_width
        )

        self.w_mean = np.mean(self.samples, axis=0)
        return self.w_mean