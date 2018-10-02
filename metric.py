import numpy as np


def euclidean(u, v, squared=True):
    d = np.sum((u - v) ** 2)
    if not squared:
        d = np.sqrt(d)
    return d


def squared_euclidean(u, v):
    return euclidean(u, v, squared=True)


def pairwise(us, vs, metric=squared_euclidean):
    """Applies pairwise metric function.

    :param us: (m, n_features) feature vectors.
    :param vs: (n, n_features) feature vectors.
    :return: (m, n) computed distance matrix. (i, j)-th element is the distance
        between i-th vector of `us` and j-th vector of `vs`.
    """
    m, _ = us.shape
    n, _ = vs.shape
    dist_matrix = np.empty((m, n))

    for r, u in enumerate(us):
        for c, v in enumerate(vs):
            dist_matrix[r, c] = metric(u, v)

    return dist_matrix
