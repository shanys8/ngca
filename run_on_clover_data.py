from ngca_algorithm import run as run_ngca_algorithm
import numpy as np
import utilities
import matplotlib
from matplotlib import pyplot as plt
import itertools
import math
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import constant
import nevergrad as ng
from ngca_algorithm import score_ngca_algorithm_on_clover_dataset as score_ngca_algorithm_on_clover_dataset


def plot_2d_data(clover_data, shuffled_data, result_data):

    kmeans_clover = KMeans(n_clusters=4, random_state=0).fit(clover_data)  # Get 4 clusters
    clover_kmeans_labels = kmeans_clover.labels_

    kmeans_result = KMeans(n_clusters=4, random_state=0).fit(result_data)  # Get 4 clusters
    result_kmeans_labels = kmeans_result.labels_

    # evaluate data clustering by algorithm
    score = adjusted_rand_score(clover_kmeans_labels, result_kmeans_labels)
    # score result
    utilities.print_score(score)

    f = plt.figure()
    f, axes = plt.subplots(ncols=4)
    sc = axes[0].scatter(clover_data[:, 0], clover_data[:, 1], c=clover_kmeans_labels, cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_4_COLORS))
    axes[0].set_xlabel('Clover kmeans labels', labelpad=5)

    axes[1].scatter(shuffled_data[:, 0], shuffled_data[:, 1], c='blue', alpha=0.5)
    axes[1].set_xlabel('Shuffled', labelpad=5)

    axes[2].scatter(result_data[:, 0], result_data[:, 1], c=clover_kmeans_labels, cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_4_COLORS))
    axes[2].set_xlabel('Result by \nclover labels', labelpad=5)

    axes[3].scatter(result_data[:, 0], result_data[:, 1], c=result_kmeans_labels, cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_4_COLORS))
    axes[3].set_xlabel('Result by \nkmeans labels', labelpad=5)

    plt.savefig('results/clover.png')


def scoring():
    algorithm_params = {
        'alpha1': 0.7,
        'alpha2': 0.3,
        'beta1': 0.2,
        'beta2': 0.64,
    }

    n = 1000
    clover_data = utilities.generate_clover_data(n)
    # np.savetxt('datasets/clover.txt', clover_data.T, delimiter=' ')

    # kmeans_clover = KMeans(n_clusters=4, random_state=0).fit(clover_data.T)  # Get 4 clusters
    # np.savetxt('datasets/clover_labels.txt', kmeans_clover.labels_, fmt='%d', delimiter=' ')

    shuffled_data = utilities.generate_shuffled_data(clover_data)
    # np.savetxt('datasets/shuffled_clover.txt', shuffled_data, delimiter=' ')

    # Implementation of algorithm in the paper
    approx_ng_subspace = run_ngca_algorithm(shuffled_data[:int(n/2), :], shuffled_data[int(n/2):, :], algorithm_params)
    # approx_ng_subspace = run_ngca_algorithm(shuffled_data, shuffled_data, algorithm_params)

    projected_data = np.dot(shuffled_data, approx_ng_subspace)

    plot_2d_data(clover_data.T, shuffled_data, projected_data)


def main():

    instrum = ng.Instrumentation(alpha1=ng.var.Array(1).asscalar(),
                                 alpha2=ng.var.Array(1).asscalar(),
                                 beta1=ng.var.Array(1).asscalar(),
                                 beta2=ng.var.Array(1).asscalar())
    optimizer = ng.optimizers.CMA(instrumentation=instrum, budget=100)

    for i in range(optimizer.budget):
        try:
            print('{} out of {}'.format(i, optimizer.budget))
            x = optimizer.ask()
            value = score_ngca_algorithm_on_clover_dataset(*x.args, **x.kwargs)
            optimizer.tell(x, value)
        except:
            print('Error')

    recommendation = optimizer.provide_recommendation()

    print('Optimal params:')
    print(recommendation.kwargs)

    print('Optimal Score on optimal params:')
    # score result
    score = score_ngca_algorithm_on_clover_dataset(recommendation.kwargs['alpha1'],
                                                recommendation.kwargs['alpha2'],
                                                recommendation.kwargs['beta1'],
                                                recommendation.kwargs['beta2'])
    utilities.print_score_fixed(score)


if __name__ == "__main__":
    main()
