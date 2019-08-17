import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time
from ngca_algorithm import run as run_ngca_algorithm


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


def main():

    alpha1 = 0.7
    alpha2 = 0.3
    beta1 = 0.34
    beta2 = 0.64

    data_file_name = 'DataTrn'
    samples, samples_copy = download_oil_data(data_file_name)
    labels = download_labels(data_file_name)
    colors = ['red', 'green', 'blue']

    # Run algorithm
    start = time.time()

    approx_ng_subspace = run_ngca_algorithm(samples, samples_copy, alpha1, alpha2, beta1, beta2)
    end = time.time()
    print('runtime')
    print(end - start)

    # Project data on result subspace
    proj_data = np.concatenate((np.dot(samples, approx_ng_subspace), np.dot(samples_copy, approx_ng_subspace)),
                               axis=0)

    # plot first two dimensions
    plt.scatter(proj_data[:, 0], proj_data[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))

    plt.title(data_file_name)
    # plt.legend()
    # plt.show()
    plt.savefig('results/{}_alpha1={}|alpha2={}|beta1={}|beta1={}.png'.format(data_file_name, alpha1, alpha2, beta1, beta2))


if __name__ == "__main__":
    main()
