import cvxopt
import matplotlib.pyplot as plt
import numpy as np


def create_dummy_data(n_samples):
    half = n_samples // 2

    x1 = np.random.multivariate_normal(
        [-1.5, -1.5], [[1.0, 0.4], [0.4, 1.0]], half)
    x2 = np.random.multivariate_normal(
        [1.5, 1.5], [[1.0, 0.4], [0.4, 1.0]], half)
    x = np.vstack([x1, x2])
    t = np.concatenate([np.ones(half), -np.ones(half)])
    return x, t


def plot_samples(x, t, w, b, support_cond=None):
    plt.figure(figsize=(8, 8))
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # plot boundary
    xx = np.linspace(-3, 3, 200)
    yy = - (w[0] * xx + b) / w[1]
    plt.plot(xx, yy)

    for label in np.unique(t):
        x_l = x[t == label]
        plt.plot(x_l[:, 0], x_l[:, 1], linewidth=0, marker='o', markersize=8)

    if support_cond is not None:
        x_s = x[support_cond]
        plt.plot(x_s[:, 0], x_s[:, 1], linewidth=0, marker='o', markersize=12,
                 markeredgecolor='red', markerfacecolor='none')

    plt.show()


def main():
    np.random.seed(1231)
    penalty_param = 1.0
    n_samples = 500

    x, t = create_dummy_data(n_samples)
    tt = np.outer(t, t)

    # solve QP by cvxopt
    zeros = np.zeros(n_samples)
    ones = np.ones(n_samples)
    sol = cvxopt.solvers.qp(
        P=cvxopt.matrix(tt * np.dot(x, x.T)),
        q=cvxopt.matrix(-ones),
        G=cvxopt.matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)])),
        h=cvxopt.matrix(np.concatenate([zeros, penalty_param * ones])),
        A=cvxopt.matrix(t.astype(np.float64), (1, n_samples)),
        b=cvxopt.matrix(0.0))
    alpha = np.array(sol['x']).ravel()

    support_cond = (0 <= alpha) & (1e-6 < np.abs(alpha))
    x_s = x[support_cond]
    t_s = t[support_cond]
    alpha_s = alpha[support_cond]

    w_pred = np.dot(x_s.T, alpha_s * t_s)
    b_pred = (t_s - np.dot(x_s, w_pred)).mean()

    print(w_pred, b_pred)
    plot_samples(x, t, w_pred, b_pred, support_cond)


if __name__ == '__main__':
    main()
