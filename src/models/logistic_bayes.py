import numpy as np
from src.utils.math_utils import (
    log_likelihood,
    log_prior_normal,
    log_prior_laplace,
    log_prior_student_t,
    analyze_posterior,
    sigmoid
)
from src.utils.mcmc_utils import metropolis_hastings
from sklearn.linear_model import LogisticRegression


class LogisticBayesModel:
    def __init__(self, prior_func_opt):
        self.prior_func_opt = prior_func_opt
        self.samples_ = None
        self.w_mean_ = None
        self.w_std_ = None
        self.w_ci_ = None

        self.acceptance_ratio = None

        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, iterations=10000, burn_in=1000, proposal_width=0.1, use_lr_init=False):

        if use_lr_init:
            # Initialization using standard logistic regression
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X, y)
            initial_w = np.concatenate([lr.intercept_, lr.coef_.ravel()])
        else:
            # Zero initialization
            initial_w = np.zeros(X.shape[1] + 1)

        # Select prior function
        if self.prior_func_opt == 1:
            prior_func = lambda w: log_prior_normal(w, mu=0, sigma=1)
        elif self.prior_func_opt == 2:
            prior_func = lambda w: log_prior_laplace(w, mu=0, b=0.5)
        else:
            prior_func = lambda w: log_prior_student_t(w, df=3, loc=0, scale=3)

        # Add intercept term to data
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])

        self.samples_, self.acceptance_ratio = metropolis_hastings(
            log_likelihood_func=log_likelihood,
            log_prior_func=prior_func,
            initial_w=initial_w,
            X=X_aug,
            y=y,
            iterations=iterations,
            burn_in=burn_in,
            proposal_width=proposal_width
        )

        stats = analyze_posterior(samples=self.samples_)

        self.w_mean_ = stats['mean']
        self.w_std_ = stats['std']
        self.w_ci_ = (stats['ci_lower'], stats['ci_upper'])

        self.intercept_ = np.array([self.w_mean_[0]])
        self.coef_ = np.array([self.w_mean_[1:]])

        return self
    
    def predict_proba(self, data):
        if self.w_mean_ is None:
            raise RuntimeError("You must train the model using .fit() before predicting.")
        z = self.intercept_[0] + np.dot(data, self.coef_.ravel())
        p1 = sigmoid(z)
        p0 = 1 - p1
        return np.column_stack([p0, p1])
    
    def predict(self, data, threshold=0.5):
        probs = self.predict_proba(data)[:, 1]
        return (probs >= threshold).astype(int)
