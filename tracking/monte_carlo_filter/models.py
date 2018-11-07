from abc import ABCMeta

import numpy as np
from scipy.stats import multivariate_normal


class TransitionModelABC(metaclass=ABCMeta):
    def __call__(self, state_post_prev):
        """Transits `state_post_prev`.

        :param state_post_prev: previous post particles.
        :return: transited particles.
        """
        pass


class ObservationModelABC(metaclass=ABCMeta):
    def __call__(self, state_pre, observation):
        """Resamples `state_pre`.

        :param state_pre: current pre particles.
        :param observation: current observation.
        :return: resampled particles.
        """
        pass


class GaussianTransitionModel(TransitionModelABC):
    def __init__(self, transition_matrix, noise_cov):
        self.transition_matrix = transition_matrix
        self.noise_cov = noise_cov

    def __call__(self, state_post_prev):
        state_post_prev = np.array(state_post_prev)
        n_particles, n_states = state_post_prev.shape
        # assume that noise are generated by multivariate gaussian
        noise = multivariate_normal.rvs(cov=self.noise_cov, size=n_particles)
        noise = noise.reshape(n_particles, n_states)
        state_pre = np.dot(self.transition_matrix, state_post_prev.T).T + noise
        return state_pre


class GaussianObservationModel(ObservationModelABC):
    def __init__(self, observation_matrix, noise_cov):
        self.observation_matrix = observation_matrix
        self.noise_cov = noise_cov

    def __call__(self, state_pre, observation):
        state_pre = np.array(state_pre)
        n_particles, n_states = state_pre.shape

        # first compute negative log likelihood of multivariate gaussian
        obs_pred = np.dot(self.observation_matrix, state_pre.T).T
        nll = -multivariate_normal.logpdf(obs_pred, mean=observation,
                                          cov=self.noise_cov)
        # then compute particle weights
        weights = 1 / nll
        weights /= np.sum(weights)

        # resample
        indices = np.random.choice(n_particles, size=n_particles, p=weights)
        state_post = state_pre[indices]
        return state_post