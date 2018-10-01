import numpy as np
import pandas as pd

from metric import pairwise


def init(n_clusters=5, seed=None):
    x = np.arange(200).reshape(-1, 2, 10)
    n_samples = len(x)

    rs = np.random.RandomState(seed)

    center_indices = None
    while True:
        # 1a. take one center c_1, chosen uniformly at random from x.
        if center_indices is None:
            center_indices = rs.choice(n_samples, 1)
            continue

        # compute probability which the sample is selected as a new center.
        pair_dist = pairwise(x, x[center_indices])
        d_min = np.min(pair_dist, axis=1)
        prob = d_min / np.sum(d_min)

        # 1b. take a new center c_i, choosing with the above probability.
        new_center_index = rs.choice(n_samples, 1, p=prob)
        center_indices = np.concatenate([center_indices, new_center_index])

        # 1c. repeat step 1b until we have chosen a total of k centers.
        if len(center_indices) == n_clusters:
            break

    return x[center_indices]


def k_means(x, n_clusters):
    n_samples, n_features = x.shape

    # initialization
    centers = x[np.random.choice(n_samples, n_clusters, replace=False)]
    prev_assigned_indices, curr_assigned_indices = None, None

    curr_iter = 0
    while True:
        curr_iter += 1
        print("iter:", curr_iter)

        # maximization step
        distance = pairwise(x, centers)
        curr_assigned_indices = np.argmin(distance, axis=1)

        # convergence check
        if prev_assigned_indices is not None and \
                np.array_equal(prev_assigned_indices, curr_assigned_indices):
            break

        # expectation step
        for ci in range(n_clusters):
            centers[ci, :] = np.mean(x[curr_assigned_indices == ci], axis=0)

        prev_assigned_indices = curr_assigned_indices

    return centers


def main():
    iris_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/' \
               'iris/iris.data'
    df = pd.read_csv(iris_url)
    print(df)
    x = np.array(df)[:, :-1].astype(np.float32)

    n_clusters = 5
    centers = k_means(x, n_clusters)
    print(centers)


if __name__ == '__main__':
    main()
