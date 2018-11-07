import numpy as np


class MonteCarloFilter:
    """Monte Carlo Filter, which is a special case of Particle Filter."""

    def __init__(self, n_particles, n_states, transition_model,
                 observation_model, particle_initializer=np.random.randn):
        """Creates an instance of `MonteCarloFilter`.

        :param n_particles: the number of particles.
        :param n_states: the number of states.
        :param transition_model: assumed transition model.
        :param observation_model: assumed observation model.
        :param particle_initializer: particle initializer method.
        """
        assert n_states == 1

        self.n_particles = n_particles
        self.n_states = n_states
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.state_pre = particle_initializer(self.n_particles, self.n_states)
        self.state_post = particle_initializer(self.n_particles, self.n_states)

    def predict(self):
        self.state_pre = self.transition_model(self.state_post)
        return self.state_pre

    def resample(self, observation):
        self.state_post = self.observation_model(self.state_pre, observation)
        return self.state_post
