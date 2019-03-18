from functools import partial

import numpy as np


class SVC:
    """Support Vector Classifier.

    This implementation is based on Algorithm 6.2, SMO with Maximum Violating
    Pair working set selection, from the following paper:
        https://leon.bottou.org/publications/pdf/lin-2006.pdf

    """

    def __init__(self, C=1.0, kernel='rbf', gamma='auto', tol=1e-3):
        self.C = C
        self.kernel = Kernel(kernel=kernel, gamma=gamma)
        self.gamma = gamma
        self.tol = tol

    def _smo(self, x_train, y_train):
        k = self.kernel(x_train)

        # initialize params
        n_samples = len(y_train)
        self._init_params(n_samples)

        while True:
            if self._smo_step(k, y_train):
                break

    def _init_params(self, n_samples):
        self.n_iters = 0
        self.a = np.zeros(n_samples)
        self.g = np.ones(n_samples)
        self.b = 0

    def _smo_step(self, k, y):
        n_samples = len(y)
        C = self.C

        self.n_iters += 1

        high = ((self.a < C) & (y == 1)) | ((0 < self.a) & (y == -1))
        low = ((self.a < C) & (y == -1)) | ((0 < self.a) & (y == 1))
        i = np.argmax([y[n] * self.g[n] if high[n] else -np.inf
                       for n in range(n_samples)])
        j = np.argmin([y[n] * self.g[n] if low[n] else np.inf
                       for n in range(n_samples)])

        self.b = (y[i] * self.g[i] + y[j] * self.g[j]) / 2

        # convergence check
        # print('y[i]g[i], y[j]g[j]', y[i] * self.g[i], y[j] * self.g[j])
        if y[i] * self.g[i] < y[j] * self.g[j] + self.tol:
            return True

        B_i = C if y[i] == 1 else 0
        A_j = 0 if y[j] == 1 else -C

        lam = np.min([
            B_i - y[i] * self.a[i], y[j] * self.a[j] - A_j,
            (y[i] * self.g[i] - y[j] * self.g[j]) / (
                    k[i, i] + k[j, j] - 2 * k[i, j])
        ])

        # Update gradient
        for n in range(n_samples):
            self.g[n] -= lam * y[n] * (k[i, n] - k[j, n])

        # Update coefficients
        self.a[i] += lam * y[i]
        self.a[j] -= lam * y[j]

        # Return `False` to indicate not converge in this step
        return False

    def decision_function(self, x_test):
        """Evaluate the decision function for the samples in `x_test`.

        :param x_test: (n_test_samples, n_features)
        :return: (n_test_samples,)
        """
        score = np.dot(self.a * self.y_train,
                       self.kernel(self.x_train, x_test)) + self.b
        return score

    def fit(self, x_train, y_train):
        """Fit the SVM model according to the given training data.

        :param x_train: (n_samples, n_features)
        :param y_train: (n_samples,)
        :return: self
        """
        self.x_train = x_train
        self.y_train = y_train

        self._smo(x_train, y_train)
        # self.a, self.b = self._smo(x_train, y_train)
        return self

    def predict(self, x_test):
        """Perform classification on samples in `x_test`.

        :param x_test: (n_test_samples, n_features)
        :return: (n_test_samples,)
        """
        score = self.decision_function(x_test)
        y_pred = 2 * (0 < score) - 1
        return y_pred


class Kernel:
    def __init__(self, kernel='rbf', gamma='auto'):
        self.metric = self._select_kernel(kernel, gamma=gamma)

    def _select_kernel(self, kernel, **kwargs):
        if kernel == 'linear':
            return partial(linear_kernel, **kwargs)
        if kernel == 'rbf':
            return partial(rbf_kernel, **kwargs)
        return kernel

    def __call__(self, x1, x2=None):
        """Compute pairwise kernel values.

        :param x1: (n_samples1, n_features)
        :param x2: (n_samples2, n_features)
        :return:
            if x2 is None: (n_samples1, n_samples1)
            if x2 is not None: (n_samples1, n_samples2)
        """

        if x2 is None:
            x2 = x1

        n_samples1, n_features1 = x1.shape
        n_samples2, n_features2 = x2.shape

        if n_features1 != n_features2:
            raise ValueError(
                'n_features of x1 and n_features of x2 must be same.')

        k = np.zeros((n_samples1, n_samples2))

        for r in range(n_samples1):
            for c in range(n_samples2):
                k[r, c] = self.metric(x1[r], x2[c])

        return k


def linear_kernel(u, v, **kwargs):
    """Linear kernel.

    :param u: (n_features,)
    :param v: (n_features,)
    :return: Computed linear kernel value.
    """
    return np.dot(u, v)


def rbf_kernel(u, v, gamma='auto', **kwargs):
    """RBF kernel.

    :param u: (n_features,)
    :param v: (n_features,)
    :param gamma:
    :return: Computed RBF kernel value.
    """
    if gamma == 'auto':
        n_features = len(u)
        gamma = 1 / n_features

    return np.exp(-gamma * np.sum(np.square(u - v)))
