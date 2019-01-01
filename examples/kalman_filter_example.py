import matplotlib

from tracking.kalman_filter.kalman_filter import KalmanFilter

# to show animation in PyCharm
matplotlib.use('Qt5Agg')  # noqa

from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import multivariate_normal
from matplotlib.colors import BASE_COLORS


def _data_generator(start_point, velocity, cov):
    x = np.array(start_point)
    velocity = np.array(velocity)
    mvn = multivariate_normal(cov=cov)

    def step():
        nonlocal x, velocity, mvn
        x += velocity + mvn.rvs()
        return x.copy()

    return step


class _History:
    def __init__(self, history_size):
        self.history_size = history_size
        self.history = []

    def append(self, **item):
        if len(self.history) == self.history_size:
            self.history = self.history[1:]
        self.history.append(item)


class _ToySimulator:
    def __init__(self, kf: KalmanFilter, gen, history_size=3):
        self.kf = kf
        self.gen = gen
        self.history = _History(history_size)
        self.fig, self.axes = plt.subplots(1, 1, figsize=(16 * 0.5, 9 * 0.5))

    def _plot_contours(self, ax, x_range, y_range, mean, cov):
        x = np.linspace(*x_range, 400)
        y = np.linspace(*y_range, 400)
        xx, yy = np.meshgrid(x, y)
        xy = np.stack([xx.flat, yy.flat], axis=1)
        zz = multivariate_normal.pdf(xy, mean, cov).reshape(xx.shape)

        ax.contourf(xx, yy, zz, cmap=plt.cm.bone_r)

    def update(self, t):
        self.axes.clear()
        colors = list(BASE_COLORS)[:len(self.history.history)]

        self.kf.predict()
        measurement = self.gen()
        measurement_pred = self.kf.estimate_measurement(
            self.kf.update(measurement))
        self.history.append(measurement=measurement,
                            measurement_pred=measurement_pred,
                            error_cov_post=self.kf.error_cov_post[:2, :2])

        # plot
        measurements = []
        means = []
        covariances = []
        for item in self.history.history:
            measurements.append(item['measurement'])
            means.append(item['measurement_pred'])
            covariances.append(item['error_cov_post'])

        measurements = np.array(measurements)
        means = np.array(means)
        covariances = np.array(covariances)

        x_min, x_max = measurements[:, 0].min(), measurements[:, 0].max()
        y_min, y_max = measurements[:, 1].min(), measurements[:, 1].max()
        x_range = (x_min - 2.0, x_max + 2.0)
        y_range = (y_min - 2.0, y_max + 2.0)

        self._plot_contours(self.axes, x_range, y_range, means[-1],
                            covariances[-1])
        self.axes.plot(measurements[:, 0], measurements[:, 1], 'r', marker='x')

        self.axes.set_title('kalman filter')
        self.axes.set_xlim(*x_range)
        self.axes.set_ylim(*y_range)
        return self.axes,


def build_kf():
    # assumptions:
    #     states are two positions and two velocities.
    #     sum of current position and velocity equals to next position
    kf = KalmanFilter(n_states=4, n_measurements=2, n_controls=0)
    kf.transition_matrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
    kf.observation_noise_cov = 0.5 * np.eye(*kf.observation_noise_cov.shape)
    kf.process_noise_cov = 0.5 * np.eye(*kf.process_noise_cov.shape)

    return kf


def main():
    parser = ArgumentParser()
    parser.add_argument('--animation_command', type=str, default='show',
                        choices=('show', 'save'))
    args = parser.parse_args()

    gen = _data_generator(np.zeros(2), [2.2, 0.5], np.eye(2))
    kf = build_kf()
    sim = _ToySimulator(kf, gen)

    if args.animation_command == 'show':
        ani = FuncAnimation(sim.fig, sim.update)
        plt.show()
    elif args.animation_command == 'save':
        n_steps = 50
        video_file_name = 'kf.mp4'
        writer = FFMpegWriter()
        with writer.saving(sim.fig, video_file_name, dpi=100):
            for t in range(n_steps):
                sim.update(t)
                writer.grab_frame()


if __name__ == '__main__':
    main()
