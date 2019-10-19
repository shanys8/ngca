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
from ngca_algorithm import score_ngca_algorithm_on_clover_data_by_kmeans as score_ngca_algorithm_on_clover_data_by_kmeans


def evaluate_test_data_by_kmeans(algorithm_params):
    samples, samples_copy = utilities.download_data('blanchard_clover_shuffled_full', separate_data=True)
    shuffled_data_full = utilities.download_data('blanchard_clover_shuffled_full')
    clover_data = utilities.download('blanchard_clover_data')

    # run NGCA on shuffled data
    approx_ng_subspace = run_ngca_algorithm(samples, samples_copy, algorithm_params)

    # Project shuffled_data on the result subspace
    proj_data = np.dot(shuffled_data_full, approx_ng_subspace)

    # plot data in 2D
    plot_2d_data(clover_data, shuffled_data_full, proj_data)

    return


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
    f, axes = plt.subplots(ncols=3)
    sc = axes[0].scatter(clover_data[:, 0], clover_data[:, 1], c=clover_kmeans_labels, cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_4_COLORS))
    axes[0].set_xlabel('Clover kmeans labels', labelpad=5)

    axes[1].scatter(shuffled_data[:, 0], shuffled_data[:, 1], c=clover_kmeans_labels, cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_4_COLORS))
    axes[1].set_xlabel('Shuffled', labelpad=5)

    axes[2].scatter(result_data[:, 0], result_data[:, 1], c=clover_kmeans_labels, cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_4_COLORS))
    axes[2].set_xlabel('Result by \nclover labels', labelpad=5)

    plt.savefig('results/clover/clover_kmeans_optimized.png')


def main():

    instrum = ng.Instrumentation(alpha1=ng.var.Array(1).asscalar(),
                                 alpha2=ng.var.Array(1).asscalar(),
                                 beta1=ng.var.Array(1).asscalar(),
                                 beta2=ng.var.Array(1).asscalar())
    optimizer = ng.optimizers.CMA(instrumentation=instrum, budget=100)

    for i in range(optimizer.budget):
        try:
            x = optimizer.ask()
            value = score_ngca_algorithm_on_clover_data_by_kmeans(*x.args, **x.kwargs)
            print('{} out of {} - value {}'.format(i, optimizer.budget, value))
            optimizer.tell(x, value)
        except:
            print('{} out of {} - error'.format(i, optimizer.budget))

    recommendation = optimizer.provide_recommendation()

    print('Optimal params:')
    print(recommendation.kwargs)

    print('Optimal Score on optimal params:')
    # score result
    score = score_ngca_algorithm_on_clover_data_by_kmeans(recommendation.kwargs['alpha1'],
                                                recommendation.kwargs['alpha2'],
                                                recommendation.kwargs['beta1'],
                                                recommendation.kwargs['beta2'])
    utilities.print_score_fixed(score)

    evaluate_test_data_by_kmeans(recommendation.kwargs)


if __name__ == "__main__":
    main()
