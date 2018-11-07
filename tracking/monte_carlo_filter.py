import numpy as np
from scipy.stats import multivariate_normal


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
        self.particles = particle_initializer(self.n_particles, self.n_states)

        # (n_states, n_states)
        self.transition_matrix = np.eye(self.n_states, dtype=np.float32)

    def predict(self, cov=1.0):
        # assume that noise are generated by multivariate gaussian
        noise = multivariate_normal.rvs(cov=cov, size=self.n_particles)
        noise = noise.reshape(self.n_particles, self.n_states)

        # random walk
        # TODO: can set transition model
        self.state_pre = np.dot(self.transition_matrix,
                                self.particles.T).T + noise
        return self.state_pre

    def resample(self, observation, sigma=1.0):
        # random walk: w_k^(i) = h(y_k|x_{k|k}^(i)) = x_{k|k}^(i) + noise
        # TODO: can set observation model
        nll = -multivariate_normal.logpdf(self.state_pre, mean=observation,
                                          cov=sigma)
        # nll = -norm.logpdf(self.state_pre, loc=observation, scale=sigma)
        weights = 1 / nll
        weights /= np.sum(weights)
        # normalize particle weights
        weights /= np.sum(weights)

        # resample
        indices = np.random.choice(self.n_particles, size=self.n_particles,
                                   p=weights)
        self.particles = self.state_pre[indices]
        return self.particles
