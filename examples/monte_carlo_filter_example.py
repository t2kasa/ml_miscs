import matplotlib

# to show animation in PyCharm
matplotlib.use('Qt5Agg')  # NOQA

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import gaussian_kde

from tracking.monte_carlo_filter import MonteCarloFilter


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


class HistoryItem:
    def __init__(self, t, o_t, line, pdf):
        self.t = t
        self.o_t = o_t
        self.line = line
        self.pdf = pdf


class ToySimulator:
    def __init__(self, n_particles=300, history_size=5):
        self.n_particles = n_particles
        self.history_size = history_size

        self.fig, self.axes = plt.subplots(1, 1)
        self.toy = RandomWalker1D()
        self.particles = np.random.randn(self.n_particles)
        self.history = []
        self.filter = MonteCarloFilter(n_particles, 1)

    def update(self, t):
        # predict
        self.filter.predict()
        # observe
        observation = self.toy.step()
        # resample
        self.particles = self.filter.resample(observation)
        return self.animate(t)

    def animate(self, t):
        self.axes.clear()

        n_particles = self.n_particles
        particles = self.particles
        o_t = self.toy.x
        kde = gaussian_kde(particles.ravel())

        line = np.linspace(particles.min(), particles.max(), 1000)
        pdf = kde.pdf(line)

        # add/remove a history item
        if len(self.history) == self.history_size:
            self.history = self.history[1:]
        self.history.append(HistoryItem(t, o_t, line, pdf))

        # plot
        for item in self.history:
            self.axes.scatter([item.t] * n_particles, particles.ravel(), c='g',
                              marker='x', alpha=0.25)
            self.axes.plot(item.t + np.array([item.pdf.min(), item.pdf.max()]),
                           [item.o_t] * 2)
            self.axes.plot(item.t + item.pdf, item.line)

        return self.axes,


def main():
    sim = ToySimulator(n_particles=50)
    ani = FuncAnimation(sim.fig, sim.update)
    plt.show()
    exit(0)

    n_steps = 50
    video_file_name = 'monte_carlo_filter_example'
    extensions = ['.mp4', '.h264']
    for ext in extensions:
        writer = FFMpegWriter()
        with writer.saving(sim.fig, video_file_name + ext, dpi=100):
            for t in range(n_steps):
                sim.update(t)
                writer.grab_frame()


if __name__ == '__main__':
    main()
