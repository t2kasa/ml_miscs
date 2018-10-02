import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from k_means import KMeans


def main():
    iris_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/' \
               'iris/iris.data'

    x_col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    y_col_name = 'label'
    iris_df = pd.read_csv(iris_url, names=x_col_names + [y_col_name])

    x_data = np.array(iris_df[x_col_names])

    # perform k-means clustering
    k_means = KMeans(n_centers=3, init='k-means++',
                     random_state=np.random.RandomState(0))
    y_pred = k_means.fit_predict(x_data)
    centers = k_means.centers

    # plot
    plot_colors = ['r', 'g', 'b']
    for ci in range(k_means.n_centers):
        plt.scatter(x_data[y_pred == ci, 0], x_data[y_pred == ci, 1],
                    c=plot_colors[ci])

    plt.scatter(centers[:, 0], centers[:, 1], c='y', label='centers')

    plt.title('k-means example on the iris dataset')
    plt.xlabel(x_col_names[0])
    plt.ylabel(x_col_names[1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
