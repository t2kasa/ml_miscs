import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from gmm import e_step, m_step


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


def plot_gmm_contours(mu, sigma):
    xx, yy = np.meshgrid(np.linspace(-2.0, 2.0, 100),
                         np.linspace(-2.0, 2.0, 100))
    xy = np.stack([xx.flat, yy.flat], axis=1)

    n_components = len(mu)
    cmaps = [plt.cm.winter, plt.cm.autumn]

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.set_xlim(-2.0, 2.0)
    axes.set_ylim(-2.0, 2.0)

    for k in range(n_components):
        z = multivariate_normal.pdf(xy, mu[k], sigma[k])
        zz = np.reshape(z, xx.shape)

        values, vectors = np.linalg.eig(cov[0])
        angle = np.angle(np.complex(vectors[0, 0], vectors[0, 1]))
        e = Ellipse(mean[0], values[0], values[1], angle, color='blue',
                    fill=False)
        ax.add_artist(e)
        fig.show()

        axes.contour(xx, yy, zz, levels=0.12, cmap=cmaps[k])

    fig.show()


def main():
    n_iters = 20
    n_components = 2

    x = get_data()
    n_samples, n_features = x.shape

    # initialize model params
    pi = np.ones(n_components) / n_components
    mean = np.array([[-1.0, 1.0], [1.0, -1.0]])
    cov = np.tile(np.eye(n_features), (n_components, 1, 1))

    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots()
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)

    values, vectors = np.linalg.eig(cov[0])
    angle = np.angle(np.complex(vectors[0, 0], vectors[0, 1]))
    e = Ellipse(mean[0], values[0], values[1], angle, color='blue', fill=False)
    ax.add_artist(e)
    fig.show()

    # for _ in range(n_iters):
    #     gamma = e_step(x, pi, mean, cov)
    #     pi, mean, cov = m_step(x, gamma)
    #     plot_gmm_contours(mean, cov)


if __name__ == '__main__':
    main()
