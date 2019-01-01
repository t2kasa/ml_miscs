import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import BASE_COLORS, to_rgb

from gmm import GMM
from utils.vis_utils import plot_gmm_confidence_ellipses


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

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)

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
