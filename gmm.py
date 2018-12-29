import numpy as np
from scipy.stats import multivariate_normal


def e_step(x, pi, mean, cov):
    """Expectation Step.

    :param x: (n_samples, n_features) features.
    :param pi: (n_components,) mixing coefficients.
    :param mean: (n_components, n_features) means.
    :param cov: (n_components, n_features, n_features) covariances.
    :return: (n_samples, n_components) responsibilities.
    """
    n_samples = len(x)
    n_components = len(pi)

    gamma = np.zeros((n_samples, n_components))
    for k in range(n_components):
        prob_k = multivariate_normal.pdf(x, mean[k], cov[k])
        gamma[:, k] = pi[k] * prob_k

    gamma /= np.sum(gamma, axis=1, keepdims=True)
    return gamma


def e_step_understandable(x, pi, mean, cov):
    """Expectation Step.

    :param x: (n_samples, n_features) features.
    :param pi: (n_components,) mixing coefficients.
    :param mean: (n_components, n_features) means.
    :param cov: (n_components, n_features, n_features) covariances.
    :return: (n_samples, n_components) responsibilities.
    """
    n_samples = len(x)
    n_components = len(pi)

    gamma = np.zeros((n_samples, n_components))

    for n in range(n_samples):
        pdf_n = pi * np.array([multivariate_normal.pdf(x[n], mean_k, cov_k)
                               for mean_k, cov_k in zip(mean, cov)])
        gamma[n] = pdf_n / np.sum(pdf_n)
    return gamma


def m_step(x, gamma):
    """Maximization step.

    :param x: (n_samples, n_features) features.
    :param gamma: (n_samples, n_components) responsibilities.
    :return: Estimated GMM parameters.
        pi: (n_components,) mixing coefficients
        mean: (n_components, n_features) means
        cov: (n_components, n_features, n_features) covariances
    """
    n_samples, n_features = x.shape
    _, n_components = gamma.shape

    n_effect = np.sum(gamma, axis=0)
    pi = n_effect / n_samples
    mean = np.dot(x.T, gamma) / n_effect

    # covariance computation may be more simple.
    cov = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        d_k = x - mean[k]
        cov[k] = np.dot(d_k.T, gamma[:, k:k + 1] * d_k) / n_effect[k]

    return pi, mean, cov


def m_step_understandable(x, gamma):
    """Maximization step.

    This function is understandable for GMM maximization step.

    :param x:
    :param gamma:
    :return:
    """
    n_samples, n_features = x.shape
    _, n_components = gamma.shape

    # (9.27)
    n_effect = np.zeros(n_components)
    for k in range(n_components):
        for n in range(n_samples):
            n_effect[k] += gamma[n, k]

    # (9.26)
    pi = np.zeros(n_components)
    for k in range(n_components):
        pi[k] = n_effect[k] / n_samples

    # (9.24)
    mean = np.zeros((n_components, n_features))
    for k in range(n_components):
        for n in range(n_samples):
            mean[k] += gamma[n, k] * x[n]
        mean[k] /= n_effect[k]

    # (9.25)
    cov = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        for n in range(n_samples):
            cov[k] += gamma[n, k] * np.dot(
                np.expand_dims(x[n] - mean[k], axis=-1),
                np.expand_dims(x[n] - mean[k], axis=-1).T)
        cov[k] /= n_effect[k]
    cov = np.stack(cov)

    return pi, mean, cov


# def log_likelihood(x, pi, mean, cov):
#     pass
#     n_components = len(pi)
#
#     res = 0.0
#     for k in range(n_components):
#         multivariate_normal.pdf(x, )


def log_likelihood_understandable(x, pi, mean, cov):
    n_samples, _ = x.shape

    log_prob = 0.0
    for n in range(n_samples):
        pdf_n = pi * np.array([multivariate_normal.pdf(x[n], mean_k, cov_k)
                               for mean_k, cov_k in zip(mean, cov)])
        log_prob += np.log(np.sum(pdf_n))
    return log_prob
