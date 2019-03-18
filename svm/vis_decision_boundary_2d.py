import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def vis_decision_boundary_2d(clf, ax=None, tol_support=1e-6):
    x_train = clf.x_train
    y_train = clf.y_train
    a = clf.a

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    grid_xx, grid_yy = np.meshgrid(
        np.linspace(x_train[:, 0].min() - 1, x_train[:, 0].max() + 1, 200),
        np.linspace(x_train[:, 1].min() - 1, x_train[:, 1].max() + 1, 200))

    x_test = np.vstack([grid_xx.ravel(), grid_yy.ravel()]).T
    score = clf.decision_function(x_test).reshape(grid_yy.shape)
    y_pred = 2 * (0 < score) - 1

    # scatter
    base_colors = ['C1', 'C2']
    for label, c in zip(np.unique(y_pred), cycler(color=base_colors)):
        x_l = x_train[y_train == label]
        ax.plot(x_l[:, 0], x_l[:, 1], linewidth=0, marker='o',
                markersize=8, **c)

    # decision boundary
    cm = LinearSegmentedColormap.from_list('orange_green', base_colors)
    ax.contourf(grid_xx, grid_yy, y_pred, cmap=cm, alpha=0.25)
    ax.contour(grid_xx, grid_yy, score, levels=[0], colors='black')

    # support vectors
    support_cond = (0 <= a) & (tol_support < np.abs(a))
    ax.plot(x_train[support_cond, 0], x_train[support_cond, 1], linewidth=0,
            marker='o', markersize=12, markeredgecolor='red',
            markerfacecolor='none')

    return ax
