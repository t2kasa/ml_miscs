import numpy as np
from scipy.stats import multivariate_normal

from k_means import KMeans


class GMM:
    def __init__(self, n_components, max_iters=100, tol=1e-3, pi_init=None,
                 mean_init=None, cov_init=None):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.pi = pi_init
        self.mean = mean_init
        self.cov = cov_init
        self.history = []

    def _init_params(self, x):
        """Initialize GMM parameters by K-means.

        :param x: (n_samples, n_features) features.
        :param n_components: the number of components.
        :return: Initialized GMM parameters:
            pi: (n_components,) mixing coefficients
            mean: (n_components, n_features) means
            cov: (n_components, n_features, n_features) covariances
        """
        n_samples, n_features = x.shape

        k_means = KMeans(self.n_components)
        assigned_indices = k_means.fit_predict(x)
        mean_init = k_means.centers

        pi_init = np.zeros(self.n_components)
        cov_init = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            cond = assigned_indices == k
            d_k = x[cond] - mean_init[k]
            pi_init[k] = np.sum(cond) / n_samples
            cov_init[k] = np.dot(d_k.T, d_k) / np.sum(cond)

        return pi_init, mean_init, cov_init

    def fit(self, x):
        """Estimates model parameters.

        :param x: (n_samples, n_features) features.
        :return: self
        """

        # initialize params unless all params are not None.
        if any([self.pi is None, self.mean is None, self.cov is None]):
            self.pi, self.mean, self.cov = self._init_params(x)

        gamma = self._e_step(x, self.pi, self.mean, self.cov)
        log_likelihood = self._log_likelihood(x, self.pi, self.mean, self.cov)
        self.history = [{'gamma': gamma, 'pi': self.pi, 'mean': self.mean,
                         'cov': self.cov, 'log_likelihood': log_likelihood}]

        prev_log_likelihood = None
        for i in range(self.max_iters):
            gamma = self._e_step(x, self.pi, self.mean, self.cov)
            self.pi, self.mean, self.cov = self._m_step(x, gamma)
            log_likelihood = self._log_likelihood(x, self.pi, self.mean,
                                                  self.cov)

            self.history.append(
                {'gamma': gamma, 'pi': self.pi, 'mean': self.mean,
                 'cov': self.cov, 'log_likelihood': log_likelihood})

            # convergence check
            if prev_log_likelihood is not None and \
                    np.abs(prev_log_likelihood - log_likelihood) < self.tol:
                break

            prev_log_likelihood = log_likelihood

        return self

    @staticmethod
    def _e_step(x, pi, mean, cov):
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

    @staticmethod
    def _m_step(x, gamma):
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

        mean = np.stack(
            [np.sum(gamma[:, k:k + 1] * x, axis=0) / n_effect[k]
             for k in range(n_components)])

        # covariance computation may be more simple.
        cov = np.zeros((n_components, n_features, n_features))
        for k in range(n_components):
            d_k = x - mean[k]
            cov[k] = np.dot(d_k.T, gamma[:, k:k + 1] * d_k) / n_effect[k]

        return pi, mean, cov

    @staticmethod
    def _log_likelihood(x, pi, mean, cov):
        """Computes log-likelihood.

        :param x: (n_samples, n_features) features.
        :param pi: (n_components,) mixing coefficients
        :param mean: (n_components, n_features) means
        :param cov: (n_components, n_features, n_features) covariances
        :return: Computed log-likelihood.
        """
        n_samples = len(x)
        n_components = len(pi)

        pdfs = np.zeros((n_samples, n_components))
        for k in range(n_components):
            pdfs[:, k] = pi[k] * multivariate_normal.pdf(x, mean[k], cov[k])

        score = np.sum(np.log(np.sum(pdfs, axis=1)))
        return score

    @staticmethod
    def _e_step_understandable(x, pi, mean, cov):
        n_samples = len(x)
        n_components = len(pi)

        gamma = np.zeros((n_samples, n_components))
        for n in range(n_samples):
            pdf_n = pi * np.array(
                [multivariate_normal.pdf(x[n], mean_k, cov_k)
                 for mean_k, cov_k in zip(mean, cov)])
            gamma[n] = pdf_n / np.sum(pdf_n)
        return gamma

    @staticmethod
    def _m_step_understandable(x, gamma):
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
