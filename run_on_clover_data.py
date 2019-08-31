from ngca_algorithm import run as run_ngca_algorithm
import numpy as np
import utilities
import matplotlib
from matplotlib import pyplot as plt
import itertools
import math


def plot_2d_data(clover_data, shuffled_data, result_data):

    f = plt.figure()
    f, axes = plt.subplots(ncols=3)
    sc = axes[0].scatter(clover_data[:, 0], clover_data[:, 1], c='blue', alpha=0.5)
    axes[0].set_xlabel('Clover', labelpad=5)

    axes[1].scatter(shuffled_data[:, 0], shuffled_data[:, 1], c='blue', alpha=0.5)
    axes[1].set_xlabel('Shuffled', labelpad=5)

    axes[2].scatter(result_data[:, 0], result_data[:, 1], c='blue', alpha=0.5)
    axes[2].set_xlabel('Result', labelpad=5)

    plt.savefig('results/clover.png')


def generate_clover_data(n):
    X = np.empty((2, 0), float)
    count = 0

    while count < n:
        d = 2 * np.random.rand(2, n) - 1
        index = np.argwhere(np.sum(np.power(d, 2), axis=0) < np.sqrt(np.abs(d[0, :] * d[1, :])))
        index = list(itertools.chain(*index))
        count = count + np.size(index)
        X = np.append(X, np.take(d, index, axis=1), axis=1)

    result = math.sqrt(10) * X[:, :n]
    return result


def generate_shuffled_data(data):
    scaling = np.power(10, np.arange(-1, 1.2, 0.2))

    shuffled_data = data
    s = 0
    while s < np.size(scaling):
        shuffled_data = scaling[s] * np.append(shuffled_data, np.random.randn(1, np.shape(shuffled_data)[1]), axis=0)
        s = s + 1

    Q, R = np.linalg.qr(np.random.randn(13, 13))

    shuffled_data = np.dot(Q, shuffled_data)
    return shuffled_data.T


def main():

    algorithm_params = {
        'alpha1': 0.7,
        'alpha2': 0.3,
        'beta1': 0.2,
        'beta2': 0.64,
    }

    n = 1000

    clover_data = generate_clover_data(n)
    shuffled_data = generate_shuffled_data(clover_data)

    # Implementation of algorithm in the paper
    approx_ng_subspace = run_ngca_algorithm(shuffled_data[:int(n/2), :], shuffled_data[int(n/2):, :], algorithm_params)
    # approx_ng_subspace = run_ngca_algorithm(shuffled_data, shuffled_data, algorithm_params)

    projected_data = np.dot(shuffled_data, approx_ng_subspace)

    plot_2d_data(clover_data.T, shuffled_data, projected_data)

    return 0


if __name__ == "__main__":
    main()
