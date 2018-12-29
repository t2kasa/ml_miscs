import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from gmm import *


def _normalize(x: np.ndarray):
    _, n_features = x.shape

    x_norm = np.empty_like(x)
    for i in range(n_features):
        x_norm[:, i] = (x[:, i] - x[:, i].mean()) / x[:, i].std()
    return x_norm


def get_data():
    data_file = 'D:/Dropbox/Projects/_GitHub/PRML/datasets/faithful.csv'
    x = np.genfromtxt(data_file, delimiter=',', skip_header=True)
    x = _normalize(x)
    return x


def _eig_sort(a):
    values, vectors = np.linalg.eig(a)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    return values, vectors


def plot_gmm_confidence_ellipses(mean, cov, ellipse_colors, confidence=0.95,
                                 plot_eigenvectors=True):
    n_components = len(mean)
    alpha = np.sqrt(chi2(2).ppf(confidence))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)

    for k in range(n_components):
        # plot ellipse from covariance
        values, vectors = _eig_sort(cov[k])
        w, h = 2 * alpha * np.sqrt(values)
        angle = np.degrees(np.arctan2(vectors[0, 1], vectors[0, 0]))
        ax.add_artist(
            Ellipse(mean[k], w, h, angle, color=ellipse_colors[k], fill=False))

        # plot eigenvectors if needed
        if plot_eigenvectors:
            eigen_lines = [np.stack([mean[k], mean[k] + vectors[:, 0] * w / 2]),
                           np.stack([mean[k], mean[k] + vectors[:, 1] * h / 2])]
            for line in eigen_lines:
                ax.plot(line[:, 0], line[:, 1], 'black')

    return fig, ax


def main():
    n_iters = 20
    n_components = 2
    x = get_data()

    gmm = GMM(n_components)
    history = gmm.fit(x).history

    # initialize model params
    pi, mean, cov = init_params(x, n_components)

    for _ in range(n_iters):
        gamma = e_step_understandable(x, pi, mean, cov)
        pi, mean, cov = m_step_understandable(x, gamma)
        print(compute_log_likelihood(x, pi, mean, cov),
              log_likelihood_understandable(x, pi, mean, cov))

    fig, ax = plot_gmm_confidence_ellipses(mean, cov, ellipse_colors=('b', 'r'))
    fig.show()


if __name__ == '__main__':
    main()
