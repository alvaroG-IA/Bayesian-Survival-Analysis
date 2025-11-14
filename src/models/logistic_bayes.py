import numpy as np

class LogisticBayesModel:
    def __init__(self, prior_mu=0, prior_sigma=10):
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
    
    def fit(self, X, y):
        self.X = X
        self.y = y

        initial_w = np.zeros(self.X.shape[1])
        