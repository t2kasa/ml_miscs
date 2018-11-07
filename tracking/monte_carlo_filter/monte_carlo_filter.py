import numpy as np

from tracking.monte_carlo_filter import GaussianObservationModel
from tracking.monte_carlo_filter import GaussianTransitionModel


class MonteCarloFilter:
    def __init__(self, n_particles, n_states,
                 particle_initializer=np.random.randn):
        """Creates an instance of `MonteCarloFilter`.

        :param n_particles: the number of particles.
        :param n_states: the number of states.
        :param particle_initializer: particle initializer method.
        """
        assert n_states == 1

        self.n_particles = n_particles
        self.n_states = n_states
        self.state_post = particle_initializer(self.n_particles, self.n_states)

        # (n_states, n_states)
        self.transition_model = GaussianTransitionModel(
            np.eye(self.n_states, dtype=np.float32),
            np.eye(self.n_states, dtype=np.float32))

        self.observation_model = GaussianObservationModel(
            np.eye(self.n_states, dtype=np.float32),
            np.eye(self.n_states, dtype=np.float32))

    def predict(self):
        self.state_pre = self.transition_model(self.state_post)
        return self.state_pre

    def resample(self, observation):
        self.state_post = self.observation_model(self.state_pre, observation)
        return self.state_post
