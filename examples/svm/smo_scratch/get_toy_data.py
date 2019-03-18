import numpy as np
from scipy.stats import multivariate_normal


def get_toy_data(n_features, n_samples_per_label, seed=0):
    rs = np.random.RandomState(seed)

    means = np.stack([rs.randn(n_features), rs.randn(n_features)])
    covs = np.stack([np.eye(n_features, dtype=np.float32),
                     np.eye(n_features, dtype=np.float32)])

    x = np.concatenate([
        multivariate_normal.rvs(means[0], covs[0], size=n_samples_per_label),
        multivariate_normal.rvs(means[1], covs[1], size=n_samples_per_label)
    ])

    y = np.concatenate([np.ones(n_samples_per_label),
                        -np.ones(n_samples_per_label)])

    return x, y
