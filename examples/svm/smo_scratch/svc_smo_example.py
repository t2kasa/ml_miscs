from argparse import ArgumentParser

import matplotlib.pyplot as plt

from examples.svm.smo_scratch.get_toy_data import get_toy_data
from svm.svc import SVC
from svm.vis_decision_boundary_2d import vis_decision_boundary_2d

if __name__ == '__main__':
    parser = ArgumentParser()
    # SVC hyper-parameters
    parser.add_argument('--C', type=float, default=10.0)
    parser.add_argument('--kernel', type=str, default='rbf',
                        choices=('linear', 'rbf'))
    parser.add_argument('--gamma', type=str, default='auto')
    # toy data config
    parser.add_argument('--n_features', type=int, default=2)
    parser.add_argument('--n_samples_per_label', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    x_train, y_train = get_toy_data(
        n_features=args.n_features,
        n_samples_per_label=args.n_samples_per_label,
        seed=args.seed)

    clf = SVC(args.C, args.kernel, args.gamma)
    clf.fit(x_train, y_train)
    if args.n_features == 2:
        ax = vis_decision_boundary_2d(clf)
        plt.show()
