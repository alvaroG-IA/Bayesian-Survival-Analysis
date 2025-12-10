import numpy as np


def metropolis_hastings(log_likelihood_func, log_prior_func, initial_w, X, y,
                        iterations=5000, burn_in=0.05, proposal_width=0.1):
    """
    Implementation of the Metropolis-Hastings algorithm.

    :param log_likelihood_func: Function that computes the log-likelihood -> log(p(y | X, w)).
    :param log_prior_func: Function that computes the log-prior -> log(p(w)).
    :param initial_w: Initial weight sample to start the MCMC chain.
    :param X: Input data.
    :param y: Input labels.
    :param iterations: Number of desired MCMC iterations.
    :param burn_in: Percentage of initial samples to discard to ensure samples come
                    from the stationary distribution.
    :param proposal_width: Standard deviation of the proposal distribution, which
                           controls the jump size in parameter space.
    :return:
            samples: Posterior samples after applying burn-in.
            acceptance_ratio: Percentage of proposed moves that were accepted.
    """

    w = initial_w
    samples = [w]

    # Current value of the log-posterior
    log_current = log_likelihood_func(w, X, y) + log_prior_func(w)
    accepted = 0

    for _ in range(iterations):

        # Propose a new candidate
        w_new = w + np.random.normal(scale=proposal_width, size=w.shape)

        # Compute log-posterior of the proposed sample
        log_new = log_likelihood_func(w_new, X, y) + log_prior_func(w_new)
        log_accept = log_new - log_current

        # Accept or reject
        if np.log(np.random.rand()) < log_accept:
            w = w_new
            log_current = log_new
            accepted += 1

        samples.append(w)

    # Burn-in removal
    burn_in_samples = int(burn_in * len(samples))
    samples = samples[burn_in_samples:]

    return np.array(samples), (accepted / iterations) * 100
