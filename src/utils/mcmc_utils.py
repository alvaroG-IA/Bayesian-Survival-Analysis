import numpy as np


def metropolis_hastings(log_likelihood_func, log_prior_func, initial_w, X, y,
                        iterations=5000, burn_in=0.05, proposal_width=0.1):
    """
    Implementación del algoritmo Metropolis-Hasting.
    :param log_likelihood_func: Función que calcula el logaritmo de la verosimilitud -> log(p(y|X, w)).
    :param log_prior_func: Función que calcula el logaritmo del prior -> log(p(w)).
    :param initial_w: Muestra inicial de los pesos para iniciar la cadena MCMC.
    :param X: Datos de entrada.
    :param y: Etiquetas de entrada.
    :param iterations: Número de iteraciones de la cadena MCMC deseadas.
    :param burn_in: Número de muestras iniciales que se descartan con el fin de que la cadena alcance la distribución
                    estacionaria.
    :param proposal_width: Valor de la desviación estándar de la distribución de propuesta, el cual controla el
                           tamaño de los saltos en el espacio de parámetros
    :return:
            samples: Muestras obtenidas de la distribución posterior después de aplicar burn-in.
            acceptance_ratio: Porcentaje de propuestas que son aceptadas.
    """
    w = initial_w
    samples = [w]

    # Valor actual del logaritmo de la posterior
    log_current = log_likelihood_func(w, X, y) + log_prior_func(w)
    accepted = 0

    for _ in range(iterations):

        w_new = w + np.random.normal(scale=proposal_width, size=w.shape)

        log_new = log_likelihood_func(w_new, X, y) + log_prior_func(w_new)
        log_accept = log_new - log_current

        if np.log(np.random.rand()) < log_accept:
            w = w_new
            log_current = log_new
            accepted += 1

        samples.append(w)

    # Eliminación de muestras del burn-in
    burn_in_samples = int(burn_in * len(samples))
    samples = samples[burn_in_samples:]

    return np.array(samples), (accepted/iterations) * 100
