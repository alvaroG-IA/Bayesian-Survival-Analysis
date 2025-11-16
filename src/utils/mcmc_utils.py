import numpy as np

def metropolis_hastings(log_likelihood_func, log_prior_func, initial_w, X, y,
                        iterations=5000, burn_in=1000, proposal_width=0.1):
    w = initial_w
    samples = [w]
    
    accepted = 0

    for _ in range(iterations):
        w_new = w + np.random.normal(scale=proposal_width, size=w.shape)

        log_accept = (
            log_likelihood_func(w_new, X, y) + log_prior_func(w_new)
            -
            log_likelihood_func(w, X, y) - log_prior_func(w)
        )

        if np.log(np.random.rand()) < log_accept:
            w = w_new
            accepted+=1

        samples.append(w)
    
    if burn_in > 0:
        samples = samples[burn_in:]

    return np.array(samples), (accepted/iterations) * 100
