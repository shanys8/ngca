import numpy as np
from matplotlib import pyplot as plt
import ngca_theoretical
import random


# separate data into samples and samples_copy
def download_oil_data():
    result = np.loadtxt(fname="3PhData/DataTst.txt")
    return result[:500], result[500:]


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

    # try several params values
    params = tuning()

    # Get data
    samples, samples_copy = download_oil_data()

    # Run algorithm
    approx_NG_subspace = ngca_theoretical.run_ngca_algorithm(samples,
                                                                  samples_copy,
                                                                  params['alpha1'],
                                                                  params['alpha2'], params['beta1'],
                                                                  params['beta2'])

    # Project data on result subspace
    proj_data = np.concatenate((np.dot(approx_NG_subspace.T, samples), np.dot(approx_NG_subspace.T, samples_copy)),
                               axis=1)

    # plot first two dimensions
    plt.scatter(proj_data[0, :], proj_data[1, :], c='r')
    plt.title(params_to_title(params))
    # plt.legend()
    # plt.show()
    plt.savefig('oil_data_results/{}.png'.format(params))


if __name__ == "__main__":
    main()