import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time
from ngca_algorithm import run as run_ngca_algorithm
from sklearn.cluster import KMeans


# separate data into samples and samples_copy
def download_oil_data(file_name):
    result = np.loadtxt(fname="datasets/{}.txt".format(file_name))
    return result[:500], result[500:]


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


def main():

    alpha1 = 0.7
    alpha2 = 0.3
    beta1 = 0.34
    beta2 = 0.64

    colors = ['red', 'green', 'blue']

    # get samples from train data
    train_data_file_name = 'DataTrn'
    train_samples, train_samples_copy = download_oil_data(train_data_file_name)

    # Run algorithm on samples from train data
    start = time.time()
    approx_ng_subspace = run_ngca_algorithm(train_samples, train_samples_copy, alpha1, alpha2, beta1, beta2)
    end = time.time()
    duration = end - start

    # get samples and labels from validation data
    validation_data_file_name = 'DataVdn'
    validation_samples, validation_samples_copy = download_oil_data(validation_data_file_name)
    validation_labels = download_labels(validation_data_file_name)

    # Project validation data on the result subspace
    proj_data = np.concatenate((np.dot(validation_samples, approx_ng_subspace),
                                np.dot(validation_samples_copy, approx_ng_subspace)), axis=0)

    # 2d plot
    data_for_clusters = proj_data[:, 0:2]
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data_for_clusters) # Get 3 clusters center
    centers_by_labels = calculate_centers_by_labels(data_for_clusters, validation_labels)

    # plot first two dimensions of data
    plt.scatter(proj_data[:, 0], proj_data[:, 1], c=validation_labels, cmap=matplotlib.colors.ListedColormap(colors))
    # plot centers of labels
    plt.scatter(centers_by_labels[:, 0], centers_by_labels[:, 1], c='yellow')
    # plot centers of 3 clusters by kmeans result
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='orange')
    # plt.legend()
    # plt.show()
    plt.savefig('2d_results/alpha1={}|alpha2={}|beta1={}|beta1={}.png'.format(alpha1, alpha2, beta1, beta2))

    # 3d plot
    data_for_clusters = proj_data[:, 0:2]
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data_for_clusters) # Get 3 clusters center
    centers_by_labels = calculate_centers_by_labels(data_for_clusters, validation_labels)

    # plot first three dimensions of data
    plt.scatter(proj_data[:, 0], proj_data[:, 1], c=validation_labels, cmap=matplotlib.colors.ListedColormap(colors))
    # plot centers of labels
    plt.scatter(centers_by_labels[:, 0], centers_by_labels[:, 1], c='yellow')
    # plot centers of 3 clusters by kmeans result
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='orange')

    plt.savefig('3d_results/alpha1={}|alpha2={}|beta1={}|beta1={}.png'.format(alpha1, alpha2, beta1, beta2))

if __name__ == "__main__":
    main()
