import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from matplotlib.colors import BASE_COLORS
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


def plot_gmm_confidence_ellipses(ax, mean, cov, colors, confidence=0.95,
                                 plot_eigenvectors=True):
    n_components = len(mean)
    alpha = np.sqrt(chi2(2).ppf(confidence))

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

            # eigen_lines = [np.stack([mean[k], mean[k] + vectors[:, 0] * w / 2]),
            #                np.stack([mean[k], mean[k] + vectors[:, 1] * h / 2])]
            # for line in eigen_lines:
            #     ax.arrow(*mean, color=colors[k])
            #     # ax.plot(line[:, 0], line[:, 1], colors[k])
            #

def save_history_as_video_file(history, save_video_file):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    def animate(t):
        nonlocal history, ax
        ax.clear()
        plot_gmm_confidence_ellipses(ax, history[t]['mean'], history[t]['cov'],
                                     colors=list(BASE_COLORS.keys()))
        return ax,

    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=1)
    with writer.saving(fig, save_video_file, dpi=100):
        for t in range(len(history)):
            animate(t)
            writer.grab_frame()


def main():
    n_components = 4
    x = get_data()
    gmm = GMM(n_components)

    # gmm = GMM(n_components,
    #           pi_init=np.ones(n_components) / n_components,
    #           mean_init=np.stack([[0.0, 1.0], [0.0, -1.0]]),
    #           cov_init=np.stack([0.1 * np.eye(2), 0.1 * np.eye(2)]))
    history = gmm.fit(x).history
    save_history_as_video_file(history, 'gmm.mp4')


if __name__ == '__main__':
    main()
