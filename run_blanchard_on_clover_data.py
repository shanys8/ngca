import numpy as np
import utilities
import matplotlib
from matplotlib import pyplot as plt
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import constant
from sklearn.svm import SVC


def plot_2d_data(clover_data, shuffled_data, result_data):

    kmeans_clover = KMeans(n_clusters=4, random_state=0).fit(clover_data)  # Get 4 clusters labels
    clover_kmeans_labels = kmeans_clover.labels_

    # build SVM classifier - fit by train data and check predication of validation data
    clf = SVC(gamma='auto')
    clf.fit(clover_data, clover_kmeans_labels)

    # assign score
    predicted_result_labels = clf.predict(result_data)
    score = clf.score(result_data, clover_kmeans_labels)  # score by SVM model
    score_adjust_rand = adjusted_rand_score(clover_kmeans_labels, predicted_result_labels)

    f = plt.figure()
    f, axes = plt.subplots(ncols=3)
    sc = axes[0].scatter(clover_data[:, 0], clover_data[:, 1], c=clover_kmeans_labels, cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_4_COLORS))
    axes[0].set_xlabel('Clover kmeans labels', labelpad=5)

    axes[1].scatter(shuffled_data[:, 0], shuffled_data[:, 1], c=clover_kmeans_labels, cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_4_COLORS))
    axes[1].set_xlabel('Shuffled', labelpad=5)

    axes[2].scatter(result_data[:, 0], result_data[:, 1], c=clover_kmeans_labels, cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_4_COLORS))
    axes[2].set_xlabel('Result by \nclover labels', labelpad=5)

    plt.savefig('results/blanchard/blanchard_clover.png')


def main():

    clover_data = utilities.download_data('blanchard_clover_data')
    shuffled_data = utilities.download_data('blanchard_clover_shuffled')
    projected_data = utilities.download_data('blanchard_clover_result')
    plot_2d_data(clover_data, shuffled_data, projected_data)


if __name__ == "__main__":
    main()
