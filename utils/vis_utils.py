import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2


def plot_gmm_confidence_ellipses(ax, means, covariances, colors,
                                 confidence=0.95, plot_eigenvectors=True):
    """Plots ellipses for gmm covariances.

    :param ax:
    :param means: (n_components, n_features) means.
    :param covariances: (n_components, n_features, n_features) covariances.
    :param colors: ellipse colors.
    :param confidence:
    :param plot_eigenvectors:
    :return:
    """

    n_components, n_features = means.shape
    alpha = np.sqrt(chi2(n_features).ppf(confidence))

    for k in range(n_components):
        # plot ellipse from covariance
        values, vectors = _eig_sort(covariances[k])
        w, h = 2 * alpha * np.sqrt(values)
        angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
        ax.add_artist(
            Ellipse(means[k], w, h, angle, color=colors[k], fill=False))

        # plot eigenvectors if needed
        if plot_eigenvectors:
            arrow_params = {'color': colors[k], 'length_includes_head': True,
                            'head_width': 0.05, 'head_length': 0.1}
            ax.arrow(*means[k], *(vectors[:, 0] * w / 2), **arrow_params)
            ax.arrow(*means[k], *(vectors[:, 1] * h / 2), **arrow_params)


def _eig_sort(a):
    values, vectors = np.linalg.eig(a)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    return values, vectors
