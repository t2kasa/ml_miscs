import numpy as np

from metric import pairwise


class KMeans:
    def __init__(self, n_centers=8, init='k-means++', random_state=None,
                 verbose=True):
        self._n_centers = n_centers
        self._init = self._validate_init(init)
        self._random_state = random_state or np.random.RandomState()
        self._verbose = verbose

        self._centers = None

    def _validate_init(self, init):
        if init not in ('k-means++', 'random'):
            raise ValueError('init: {} is not supported.'.format(init))
        return init

    def _init_centers(self, x):
        if self._init == 'k-means++':
            self._init_centers_k_means_plus_plus(x)
        else:
            self._init_centers_random(x)

    def _init_centers_k_means_plus_plus(self, x):
        n_samples = len(x)
        if n_samples < self._n_centers:
            raise ValueError('the `n_samples` are less than the `n_centers`.')

        center_indices = None
        while True:
            # 1a. take one center c_1, chosen uniformly at random from x.
            if center_indices is None:
                center_indices = self._random_state.choice(n_samples, 1)
                continue

            # compute probability which the sample is selected as a new center.
            pair_dist = pairwise(x, x[center_indices])
            d_min = np.min(pair_dist, axis=1)
            prob = d_min / np.sum(d_min)

            # 1b. take a new center c_i, choosing with the above probability.
            new_center_index = self._random_state.choice(n_samples, 1, p=prob)
            center_indices = np.concatenate([center_indices, new_center_index])

            # 1c. repeat step 1b until we have chosen a total of k centers.
            if len(center_indices) == self._n_centers:
                break

        self._centers = x[center_indices]

    def _init_centers_random(self, x):
        n_samples = len(x)
        if n_samples < self._n_centers:
            raise ValueError('the `n_samples` are less than the `n_centers`.')

        self._centers = self._random_state.choice(
            n_samples, self._n_centers, replace=False)

    def fit(self, x):
        self._init_centers(x)

        # initialization
        prev_assigned_indices, curr_assigned_indices = None, None

        t_iter = 0
        while True:
            t_iter += 1
            if self._verbose:
                print("iter:", t_iter)

            # maximization step
            distance = pairwise(x, self._centers)
            curr_assigned_indices = np.argmin(distance, axis=1)

            # convergence check
            if prev_assigned_indices is not None and \
                    np.all(prev_assigned_indices == curr_assigned_indices):
                break

            # expectation step
            for ci in range(self._n_centers):
                self._centers[ci, :] = np.mean(
                    x[curr_assigned_indices == ci], axis=0)

            prev_assigned_indices = curr_assigned_indices

        return self._centers

    @property
    def centers(self):
        return self._centers

    @property
    def n_centers(self):
        return self._n_centers
