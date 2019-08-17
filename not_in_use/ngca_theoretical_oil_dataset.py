import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from not_in_use import ngca_old
import random
import time


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


def tuning():
    return {
        'alpha1': random.uniform(1e-10, 10),
        'alpha2': random.uniform(1e-10, 10),
        'beta1': random.uniform(1e-10, 1),
        'beta2': random.uniform(1e-10, 1)
    }


def params_to_title(params):
    return 'alpha1: {} alpha2: {} \n beta1: {} beta2: {}'.format(params['alpha1'], params['alpha2'], params['beta1'], params['beta2'])


def main():

    # Theoretical NGCA optimal params
    params = {'alpha1': 0.6754445940381727, 'alpha2': 0.29744739800298886, 'beta1': 0.3403472323546272,
              'beta2': 0.6441926407645018}

    data_file_name = 'DataVdn'

    # Get data
    samples, samples_copy = download_oil_data(data_file_name)

    # Get labels
    labels = download_labels(data_file_name)
    colors = ['red', 'green', 'blue']

    # Run algorithm
    start = time.time()

    approx_NG_subspace = ngca_old.run_ngca_algorithm(samples,
                                                     samples_copy,
                                                     params['alpha1'],
                                                     params['alpha2'], params['beta1'],
                                                     params['beta2'])
    end = time.time()
    print('runtime')
    print(end - start)

    # Project data on result subspace
    proj_data = np.concatenate((np.dot(samples, approx_NG_subspace), np.dot(samples_copy, approx_NG_subspace)),
                               axis=0)

    # plot first two dimensions
    plt.scatter(proj_data[:, 0], proj_data[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))

    plt.title(data_file_name)
    # plt.legend()
    # plt.show()
    plt.savefig('results/{}.png'.format(data_file_name))


if __name__ == "__main__":
    main()
