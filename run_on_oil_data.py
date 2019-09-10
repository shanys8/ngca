import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time
from ngca_algorithm import run as run_ngca_algorithm
from ngca_algorithm import score_ngca_algorithm_on_oil_dataset as score_ngca_algorithm_on_oil_dataset
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.cluster import adjusted_rand_score
import constant
import nevergrad as ng
from concurrent import futures
import utilities


def evaluate_test_data(algorithm_params):

    # get samples from test data
    test_samples, test_samples_copy = utilities.download_data('DataTst', separate_data=True)

    # get samples and labels from test data
    test_data = utilities.download_data('DataTst')
    test_labels = utilities.download_labels('DataTst')

    # Run algorithm on samples from train data
    approx_ng_subspace = run_ngca_algorithm(test_samples, test_samples_copy, algorithm_params)

    # Project test data on the result subspace
    proj_data = np.dot(test_data, approx_ng_subspace)

    # evaluate data clustering by algorithm
    score = utilities.get_result_score(proj_data, test_labels, 3)

    print('Optimal Score on test data:')
    utilities.print_score_fixed(score)

    # plot data in 2D & 3D
    plot_2d_data(proj_data, algorithm_params, test_labels)
    plot_3d_data(proj_data, algorithm_params, test_labels)

    return


def plot_2d_data(proj_data, params, labels):
    data_for_clusters = proj_data[:, 0:2]
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data_for_clusters)  # Get 3 clusters
    centers_by_labels = utilities.calculate_centers_by_labels(data_for_clusters, labels)

    # plot first two dimensions of data
    plt.scatter(proj_data[:, 0], proj_data[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_3_COLORS))
    # plot centers of labels
    plt.scatter(centers_by_labels[:, 0], centers_by_labels[:, 1], c='yellow', marker='*', s=128)
    # plot centers of 3 clusters by kmeans result
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='orange', marker='*', s=128)
    # plt.legend()
    # plt.show()
    plt.savefig('results/oil_data_2D_{}.png'.format(utilities.algorithm_params_to_print(params)))


def plot_3d_data(proj_data, params, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    data_for_clusters = proj_data[:, 0:3]
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data_for_clusters)  # Get 3 clusters
    centers_by_labels = utilities.calculate_centers_by_labels(data_for_clusters, labels)

    # plot first three dimensions of data
    ax.scatter(proj_data[:, 0], proj_data[:, 1], proj_data[:, 2], c=labels,
               cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_3_COLORS))
    # plot centers of labels
    ax.scatter(centers_by_labels[:, 0], centers_by_labels[:, 1], centers_by_labels[:, 2], c='yellow', marker='*', s=128)
    # plot centers of 3 clusters by kmeans result
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               kmeans.cluster_centers_[:, 2], c='orange', marker='*', s=128)

    plt.savefig('results/oil_data_3D_{}.png'.format(utilities.algorithm_params_to_print(params)))


def evaluate_ng_subspace(algorithm_params, train_samples, train_samples_copy, validation_data, validation_labels, plot_data=False):

    # Run algorithm on samples from train data
    approx_ng_subspace = run_ngca_algorithm(train_samples, train_samples_copy, algorithm_params)

    # Project validation data on the result subspace
    proj_data = np.dot(validation_data, approx_ng_subspace)

    # evaluate data clustering by algorithm
    score = utilities.get_result_score(proj_data, validation_labels, 3)

    # score result
    utilities.print_score_fixed(score)

    # plot data in 2D & 3D
    if plot_data:
        plot_2d_data(proj_data, algorithm_params, validation_labels)
        plot_3d_data(proj_data, algorithm_params, validation_labels)

    return score


def scoring():
    # get samples from train data
    train_samples, train_samples_copy = utilities.download_data('DataTrn', separate_data=True)

    # get samples and labels from validation data
    validation_data = utilities.download_data('DataVdn')
    validation_labels = utilities.download_labels('DataVdn')

    algorithm_params = {
        'alpha1': 0.7,
        'alpha2': 0.3,
        'beta1': 0.34,
        'beta2': 0.64,
    }

    score = evaluate_ng_subspace(algorithm_params, train_samples, train_samples_copy, validation_data, validation_labels, plot_data=True)
    return score


def main():
    # Optimize params on test and validation datasets
    instrum = ng.Instrumentation(alpha1=ng.var.Array(1).asscalar(),
                                 alpha2=ng.var.Array(1).asscalar(),
                                 beta1=ng.var.Array(1).asscalar(),
                                 beta2=ng.var.Array(1).asscalar())
    optimizer = ng.optimizers.OnePlusOne(instrumentation=instrum, budget=100)
    # recommendation = optimizer.minimize(score_ngca_algorithm_on_oil_dataset)

    # ask and tell
    for i in range(optimizer.budget):
        try:
            print('{} out of {}'.format(i, optimizer.budget))
            x = optimizer.ask()
            value = score_ngca_algorithm_on_oil_dataset(*x.args, **x.kwargs)
            optimizer.tell(x, value)
        except:
            print('Error')

    recommendation = optimizer.provide_recommendation()

    print('Optimal params:')
    print(recommendation.kwargs)

    print('Optimal Score on train and validation data:')
    # score result
    score = score_ngca_algorithm_on_oil_dataset(recommendation.kwargs['alpha1'],
                                                recommendation.kwargs['alpha2'],
                                                recommendation.kwargs['beta1'],
                                                recommendation.kwargs['beta2'])
    utilities.print_score_fixed(score)

    # Run algorithm with optimal params on test data and evaluate score
    # Plot projected test data on the result NG subspace
    evaluate_test_data(recommendation.kwargs)

    return


if __name__ == "__main__":
    main()
