import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def plot_curve_fitting(x_data, t_data, M, w, ax):
    """Plot samples, polynomial curve, and true curve.

    :param x_data: x samples
    :param t_data: t samples
    :param M: polynomial order
    :param w: polynomial coefficients
    :param ax: axes to plot
    :return: plotted axes
    """
    # Figure appearance
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1.5, 1.5)
    red_patch = mpatches.Patch(label="$M = {}$".format(M))
    ax.legend(handles=[red_patch], framealpha=False,
              handlelength=0, handletextpad=0)

    # --------------------
    # Plot
    # --------------------

    # observation samples
    ax.scatter(x_data, t_data, facecolors="none", edgecolors="b")

    # generate x samples
    x_points = np.linspace(-1, 1, 300)

    # polynomial curve
    exponents = np.arange(M + 1)
    X = np.power(x_points.repeat(M + 1).reshape(-1, M + 1), exponents)
    t_pred = X.dot(w)

    ax.plot(x_points, t_pred, c="r")

    # true curve
    t_true = np.sin(2 * np.pi * x_points)
    ax.plot(x_points, t_true, c="lightgreen")

    return ax


def fit_polynomial_curve(x_data, t_data, M):
    """Polynomial Curve Fitting.

    :param x_data: x samples
    :param t_data: t samples
    :param M: polynomial order
    :return: polynomial coefficients
    """
    exponents = np.arange(M + 1)
    X = np.power(x_data.repeat(M + 1).reshape(-1, M + 1), exponents)

    # w = (X^T X)^{-1} X^T t
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(t_data)

    return w


if __name__ == '__main__':
    # Load x, t
    data_csv = os.path.join(os.path.dirname(__file__),
                            "../datasets/curve_fitting.csv")
    x_data, t_data = np.hsplit(np.loadtxt(data_csv, delimiter=","), [1])

    # order values
    M_values = np.array([0, 1, 3, 9])

    fig = plt.figure(figsize=(8, 4))
    fig.suptitle("Polynomial Curve Fitting")

    for i, M in enumerate(M_values):
        w = fit_polynomial_curve(x_data, t_data, M)

        ax = plt.subplot(2, 2, i + 1)
        plot_curve_fitting(x_data, t_data, M, w, ax)

    plt.show()
