import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time
from ngca_algorithm import run as run_ngca_algorithm
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


def download_oil_data(file_name, separate_data=False):
    result = np.loadtxt(fname="datasets/{}.txt".format(file_name))
    return (result[:500], result[500:]) if separate_data else result


def download_labels(file_name):
    int_labels = []
    labels = np.loadtxt(fname="datasets/{}Lbls.txt".format(file_name))
    for label in labels:
        if np.array_equal(label.astype(int), [1, 0, 0]):
            int_labels = np.append(int_labels, 0)
        if np.array_equal(label.astype(int), [0, 1, 0]):
            int_labels = np.append(int_labels, 1)
        if np.array_equal(label.astype(int), [0, 0, 1]):
            int_labels = np.append(int_labels, 2)
    return int_labels.astype(int)


def calculate_centers_by_labels(X, labels):
    res = np.concatenate((X[labels == 0, :].mean(axis=0)[np.newaxis], X[labels == 1, :].mean(axis=0)[np.newaxis]), axis=0)
    res = np.concatenate((res, X[labels == 2, :].mean(axis=0)[np.newaxis]), axis=0)
    return res


def plot_2d_data(proj_data, params, colors, labels):
    data_for_clusters = proj_data[:, 0:2]
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data_for_clusters)  # Get 3 clusters
    centers_by_labels = calculate_centers_by_labels(data_for_clusters, labels)

    # plot first two dimensions of data
    plt.scatter(proj_data[:, 0], proj_data[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    # plot centers of labels
    plt.scatter(centers_by_labels[:, 0], centers_by_labels[:, 1], c='yellow', marker='*', s=128)
    # plot centers of 3 clusters by kmeans result
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='orange', marker='*', s=128)
    # plt.legend()
    # plt.show()
    plt.savefig('results/2D_alpha1={}|alpha2={}|beta1={}|beta1={}.png'.format(params['alpha1'], params['alpha2'], params['beta1'], params['beta2']))


def plot_3d_data(proj_data, params, colors, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    data_for_clusters = proj_data[:, 0:3]
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data_for_clusters)  # Get 3 clusters
    centers_by_labels = calculate_centers_by_labels(data_for_clusters, labels)

    # plot first three dimensions of data
    ax.scatter(proj_data[:, 0], proj_data[:, 1], proj_data[:, 2], c=labels,
               cmap=matplotlib.colors.ListedColormap(colors))
    # plot centers of labels
    ax.scatter(centers_by_labels[:, 0], centers_by_labels[:, 1], centers_by_labels[:, 2], c='yellow', marker='*', s=128)
    # plot centers of 3 clusters by kmeans result
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               kmeans.cluster_centers_[:, 2], c='orange', marker='*', s=128)

    plt.savefig('results/3D_alpha1={}|alpha2={}|beta1={}|beta1={}.png'.format(params['alpha1'], params['alpha2'], params['beta1'], params['beta2']))


def evaluate_ng_subspace(proj_data, true_labels):
    kmeans = KMeans(n_clusters=3, random_state=0).fit(proj_data)

    score = 0
    return score


def main():

    algorithm_params = {
        'alpha1': 0.7,
        'alpha2': 0.3,
        'beta1': 0.34,
        'beta2': 0.64,
    }
    colors = ['red', 'green', 'blue'] # color per label

    # get samples from train data
    train_data_file_name = 'DataTrn'
    train_samples, train_samples_copy = download_oil_data(train_data_file_name, separate_data=True)

    # Run algorithm on samples from train data
    start = time.time()
    approx_ng_subspace = run_ngca_algorithm(train_samples, train_samples_copy, algorithm_params)
    end = time.time()
    duration = end - start

    # get samples and labels from validation data
    validation_data_file_name = 'DataVdn'
    validation_data = download_oil_data(validation_data_file_name)
    validation_labels = download_labels(validation_data_file_name)

    # Project validation data on the result subspace
    proj_data = np.dot(validation_data, approx_ng_subspace)

    # 2d plot
    plot_2d_data(proj_data, algorithm_params, colors, validation_labels)

    # 3d plot
    plot_3d_data(proj_data, algorithm_params, colors, validation_labels)

    # evaluate data clustering by algorithm
    score = evaluate_ng_subspace(proj_data, validation_labels)


if __name__ == "__main__":
    main()
