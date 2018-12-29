import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

from scipy.stats import chi2

from k_means import KMeans

from gmm import *


def normalize_each_axis(x: np.ndarray):
    _, n_features = x.shape

    x_norm = np.empty_like(x)
    for i in range(n_features):
        x_norm[:, i] = (x[:, i] - x[:, i].mean()) / x[:, i].std()
    return x_norm


def get_data():
    data_file = 'D:/Dropbox/Projects/_GitHub/PRML/datasets/faithful.csv'
    x = np.genfromtxt(data_file, delimiter=',', skip_header=True)
    x = normalize_each_axis(x)
    return x


def _eig_sort(a)
    values, vectors = np.linalg.eig(a)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    return values, vectors


def plot_gmm_contours(mu, sigma):
    n_components = len(mu)

    confidence = 0.95
    alpha = np.sqrt(chi2(2).ppf(confidence))

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.set_xlim(-2.0, 2.0)
    axes.set_ylim(-2.0, 2.0)

    for k in range(n_components):
        values, vectors = _eig_sort(sigma[k])

        width, height = 2 * alpha * np.sqrt(values)
        angle = np.degrees(np.arctan2(vectors[0, 1], vectors[0, 0]))
        theta = np.degrees(np.arctan((values[0] - values[1]) / sigma[k][0, 1]))
        theta2 = np.degrees(np.arctan2(*vectors[:, 0][::-1]))

        print(k, width, height, angle, theta, theta2)

        line1 = np.stack([mu[k], mu[k] + values[0] * vectors[0]])
        line2 = np.stack([mu[k], mu[k] + values[1] * vectors[1]])
        axes.plot(line1[:, 0], line1[:, 1], 'green')
        axes.plot(line2[:, 0], line2[:, 1], 'orange')

        e = Ellipse(mu[k], width, height, angle, color='blue', fill=False)
        axes.add_artist(e)
    fig.show()


def main():
    n_iters = 20
    n_components = 2

    x = get_data()
    n_samples, n_features = x.shape

    # initialize model params
    k_means = KMeans(n_components)
    assigned_indices = k_means.fit_predict(x)
    mean = k_means.centers

    pi = np.zeros(n_components)
    cov = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        cond = assigned_indices == k
        d_k = x[cond] - mean[k]
        pi[k] = np.sum(cond) / n_samples
        cov[k] = np.dot(d_k.T, d_k) / np.sum(cond)

    plot_gmm_contours(mean, cov)

    for _ in range(n_iters):
        gamma = e_step_understandable(x, pi, mean, cov)
        pi, mean, cov = m_step_understandable(x, gamma)
        print(log_likelihood_understandable(x, pi, mean, cov))
    plot_gmm_contours(mean, cov)


if __name__ == '__main__':
    main()
