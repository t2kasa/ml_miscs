import matplotlib

# to show animation in PyCharm
matplotlib.use('Qt5Agg')  # NOQA

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.stats import norm, gaussian_kde


class RandomWalker1D:
    """1D Random walk model.

    x_k = x_{k - 1} + v_k, v_k \sim N(0, \sigma^2)
    """

    def __init__(self, x_0=0.0, scale=1.0):
        """Creates an instance.

        :param x_0: start position.
        :param scale: noise scale.
        """
        self.mean = x_0
        self.sigma = scale
        self.x = x_0

    def step(self):
        self.x += np.random.randn() * self.sigma
        return self.x


def predict(particles, sigma=1.0):
    n_particles = len(particles)
    # random walk
    x_pre = particles + np.random.randn(n_particles) * sigma
    return x_pre


def resample(x_pre, observation, sigma=1.0):
    n_particles = len(x_pre)

    # random walk: w_k^(i) = h(y_k|x_{k|k}^(i)) = x_{k|k}^(i) + noise
    nll = -norm.logpdf(x_pre, loc=observation, scale=sigma)
    weights = 1 / nll
    weights /= np.sum(weights)

    # normalize particle weights
    weights /= np.sum(weights)

    # resample
    indices = np.random.choice(n_particles, size=n_particles, p=weights)
    resampled_particles = x_pre[indices]
    return resampled_particles


class World:
    def __init__(self):
        self.fig, self.axes = plt.subplots(1, 1)
        self.toy = RandomWalker1D()
        self.n_particles = 300
        self.particles = np.random.randn(self.n_particles)

    def update(self, t, show_animation=True):
        # predict
        x_pre = predict(self.particles)
        # observe
        observation = self.toy.step()
        # resample
        self.particles = resample(x_pre, observation)
        if show_animation:
            self.animate(t)

    def animate(self, t):
        kde = gaussian_kde(self.particles)
        x = np.linspace(self.particles.min(), self.particles.max(), 1000)
        y = kde.pdf(x)
        self.axes.clear()
        self.axes.plot(x, y)
        self.axes.plot([self.toy.x, self.toy.x], [y.min(), y.max()])
        return self.axes,


def main():
    w = World()
    ani = FuncAnimation(w.fig, w.update)
    plt.show()
    exit(0)

    n_particles = 300
    particles = np.random.randn(n_particles)
    kde = gaussian_kde(particles)
    x = np.linspace(-5, 5, 100)
    y = kde.pdf(x)
    plt.plot(x, y)
    plt.show()

    toy = RandomWalker1D()
    # initialize particles

    n_steps = 100
    for t in range(n_steps):
        # predict
        x_pre = predict(particles)
        observation = toy.step()
        # resample particles
        particles = resample(x_pre, observation)
        print(observation)
        print(particles)


if __name__ == '__main__':
    main()
