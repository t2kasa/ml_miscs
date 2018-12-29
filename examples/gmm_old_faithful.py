import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import BASE_COLORS, to_rgb
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from gmm import GMM


def _normalize(x: np.ndarray):
    _, n_features = x.shape

    x_norm = np.empty_like(x)
    for i in range(n_features):
        x_norm[:, i] = (x[:, i] - x[:, i].mean()) / x[:, i].std()
    return x_norm


def _get_faithful_data():
    from pathlib import Path
    data_file = Path(__file__).parent / '../datasets/faithful.csv'
    x = np.genfromtxt(str(data_file), delimiter=',', skip_header=True)
    x = _normalize(x)
    return x


def _eig_sort(a):
    values, vectors = np.linalg.eig(a)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    return values, vectors


def plot_gmm_confidence_ellipses(ax, mean, cov, colors, confidence=0.95,
                                 plot_eigenvectors=True):
    n_components, n_features = mean.shape
    alpha = np.sqrt(chi2(n_features).ppf(confidence))

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)

    for k in range(n_components):
        # plot ellipse from covariance
        values, vectors = _eig_sort(cov[k])
        w, h = 2 * alpha * np.sqrt(values)
        angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
        ax.add_artist(
            Ellipse(mean[k], w, h, angle, color=colors[k], fill=False))

        # plot eigenvectors if needed
        if plot_eigenvectors:
            arrow_params = {'color': colors[k], 'length_includes_head': True,
                            'head_width': 0.05, 'head_length': 0.1}
            ax.arrow(*mean[k], *(vectors[:, 0] * w / 2), **arrow_params)
            ax.arrow(*mean[k], *(vectors[:, 1] * h / 2), **arrow_params)


def plot_data_samples(ax, x, gamma, colors):
    """Plot data samples.

    :param ax: axes.
    :param x: (n_samples, n_features) features.
    :param gamma: (n_samples, n_components) responsibilities.
    :param colors: component colors.
    """
    color_values = np.array([to_rgb(c) for c in colors])
    n_samples, n_components = gamma.shape

    mixed_plot_colors = np.sum([gamma[:, k:k + 1] * color_values[k] for k in
                                range(n_components)], axis=0)
    ax.scatter(x[:, 0], x[:, 1], c=mixed_plot_colors)


def save_history_as_video_file(x, n_components, history, save_video_file):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    def animate(t):
        nonlocal history, ax
        ax.clear()

        colors = list(BASE_COLORS)[:n_components]

        plot_gmm_confidence_ellipses(ax, history[t]['mean'], history[t]['cov'],
                                     colors=colors)
        plot_data_samples(ax, x, history[t]['gamma'], colors=colors)

        ax.set_title('iteration = {:02d}, log_likelihood = {:.3f}'.format(
            t, history[t]['log_likelihood']))

        return ax,

    writer = FFMpegWriter(fps=1)
    with writer.saving(fig, save_video_file, dpi=100):
        for t in range(len(history)):
            animate(t)
            writer.grab_frame()


def main():
    n_components = 2
    x = _get_faithful_data()
    _, n_features = x.shape

    pi_init = np.random.uniform(size=n_components)
    pi_init = pi_init / np.sum(pi_init)
    mean_init = np.random.randn(n_components, n_features)
    cov_init = np.stack([np.random.uniform() * np.eye(n_features),
                         np.random.uniform() * np.eye(n_features)])
    gmm = GMM(n_components, pi_init=pi_init, mean_init=mean_init,
              cov_init=cov_init)
    history = gmm.fit(x).history
    save_history_as_video_file(x, n_components, history, 'gmm_em.mp4')


if __name__ == '__main__':
    main()
