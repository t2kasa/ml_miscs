from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from examples.svm.smo_scratch.get_toy_data import get_toy_data
from svm.svc import SVC
from svm.vis_decision_boundary_2d import vis_decision_boundary_2d


class SVCAnimator:
    def __init__(self, svc, x_train, y_train):
        self.svc = svc
        self.fig, self.axes = plt.subplots(1, 1)

        self.svc._init_params(len(x_train))
        self.svc.x_train = x_train
        self.svc.y_train = y_train
        self.k = self.svc.kernel(x_train)
        self.y = y_train

        self.convergence = False

    def update(self, t):
        self.convergence = self.svc._smo_step(self.k, self.y)
        return self.animate(t)

    def animate(self, t):
        self.axes.clear()
        vis_decision_boundary_2d(self.svc, self.axes)
        self.axes.set_title(f'n_iters: {self.svc.n_iters:06d}, '
                            f'convergence: {self.convergence}')
        return self.axes,


def main():
    parser = ArgumentParser()
    # SVC hyper-parameters
    parser.add_argument('--C', type=float, default=10.0)
    parser.add_argument('--kernel', type=str, default='rbf',
                        choices=('linear', 'rbf'))
    parser.add_argument('--gamma', type=str, default='auto')
    # toy data config
    parser.add_argument('--n_samples_per_label', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    # Animation config
    parser.add_argument('--animation_command', type=str, default='save',
                        choices=('show', 'save'))
    parser.add_argument('--out_video_name', type=str, default='svc_smo.mp4')
    args = parser.parse_args()

    x, y = get_toy_data(n_features=2,
                        n_samples_per_label=args.n_samples_per_label,
                        seed=args.seed)

    clf = SVC(args.C, args.kernel, args.gamma)
    ani = SVCAnimator(clf, x, y)

    if args.animation_command == 'show':
        animation = FuncAnimation(ani.fig, ani.update)
        plt.show()
        return

    if args.animation_command == 'save':
        writer = FFMpegWriter()
        with writer.saving(ani.fig, args.out_video_name, dpi=100):
            t = 0
            while True:
                t += 1
                print(t)
                ani.update(t)
                writer.grab_frame()

                if ani.convergence:
                    break


if __name__ == '__main__':
    main()
