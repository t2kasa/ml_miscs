import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def plot_curve_fitting(x_data, t_data, M, lambda_, w, ax):
    """Plot samples, polynomial curve, and true curve.

    :param x_data: x samples
    :param t_data: t samples
    :param M: polynomial order
    :param lambda_: regularization parameter
    :param w: polynomial coefficients
    :param ax: axes to plot
    :return: plotted axes
    """
    # Figure appearance
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1.5, 1.5)

    # Ignore RuntimeWarning from np.log(0) intentionally
    with np.errstate(divide="ignore"):
        patches = [Patch(label=r"$M = {}$".format(M)),
                   Patch(label=r"$\ln \lambda = {}$".format(np.log(lambda_)))]

    ax.legend(handles=patches, framealpha=False,
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


def fit_polynomial_curve_ridge(x_data, t_data, M, lambda_):
    """Polynomial Curve Fitting with Ridge Regression.

    :param x_data: x samples
    :param t_data: t samples
    :param M: polynomial order
    :param lambda_: regularization parameter
    :return: polynomial coefficients
    """
    exponents = np.arange(M + 1)
    X = np.power(x_data.repeat(M + 1).reshape(-1, M + 1), exponents)

    # w = (X^T X + lambda I)^{-1} X^T t
    lambda_matrix = lambda_ * np.identity(M + 1)
    w = np.linalg.inv(X.T.dot(X) + lambda_matrix).dot(X.T).dot(t_data)

    return w


if __name__ == '__main__':
    # Load x, t
    data_csv = os.path.join(os.path.dirname(__file__),
                            "../datasets/curve_fitting.csv")
    x_data, t_data = np.hsplit(np.loadtxt(data_csv, delimiter=","), [1])

    # Polynomial order and regularization params
    M_values = np.array([0, 1, 3, 9])
    lambda_values = np.exp([-np.inf, -18.0, 0.0])

    fig = plt.figure(figsize=(2.5 * len(M_values), 3 * len(lambda_values)))
    fig.suptitle("Polynomial Curve Fitting (Ridge Regression)")

    for i, (M, lambda_) in enumerate(
            itertools.product(M_values, lambda_values)):
        w = fit_polynomial_curve_ridge(x_data, t_data, M, lambda_)

        ax = plt.subplot(len(M_values), len(lambda_values), i + 1)
        plot_curve_fitting(x_data, t_data, M, lambda_, w, ax)

    plt.show()
