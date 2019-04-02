import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import utilities
import ngca_blanchard
import ngca_theoretical
import random
import time

# N num of samples
# n dimesion of samples
# d dimension of NG subspace
def generate_synthetic_isotropic_samples(N, n, d):

    G = utilities.generate_gaussian_subspace(n, n - d)
    Q, _ = LA.qr(G)  # QR decomposition from Gaussian Matrix size: n X (n-d)

    # generate subspace orthogonal to the gaussian (non gaussian) - Matrix size: n X d (REQUESTED E)
    Q_orthogonal = utilities.orthogonal_complement(Q, normalize=True)

    samples = np.empty((n, 0), float)

    # Samples should be of the isotripic model
    for _ in range(N):
        # each sample should have mean zero - np.dot(Q, np.random.randn(n - d, 1)) has zero expectancy (gaussian),
        # np.random.rand(d, 1) is normalize by 0.5 - expectancy of random vector in range (0,1)
        sample = np.dot(Q, np.random.randn(n - d, 1)) + np.dot(Q_orthogonal, (np.random.rand(d, 1) - 0.5))  # X = S + N
        samples = np.append(samples, sample, axis=1)

    return samples, Q_orthogonal


def tuning():
    return {
        'alpha1': random.uniform(1e-10, 10),
        'alpha2': random.uniform(1e-10, 10),
        'beta1': random.uniform(1e-10, 1),
        'beta2': random.uniform(1e-10, 1)
    }


def main():

    n = 10  # dimension (number of features) should be 10
    d = 3  # subspace dimension - requested dimension of NG data
    N = 1000  # number of samples to generate should be 1000
    iterations_num = 100

    theoretical_errors = np.empty(0, float)
    params_array = []
    runtimes = []

    # Generate synthetic samples
    samples, NG_subspace = generate_synthetic_isotropic_samples(N, n, d)
    theoretical_samples = samples[:, :int(N / 2)]
    theoretical_samples_copy = samples[:, int(N / 2):]
    # print('NG_subspace')
    # print(NG_subspace)

    for _ in range(iterations_num):
        try:
            # generate algorithm params
            params = tuning()
            params_array.append(params)
            start = time.time()
            # Theoretical NGCA algorithm
            approximate_theoretical_NG_subspace = ngca_theoretical.run_ngca_algorithm(theoretical_samples,
                                                                                      theoretical_samples_copy,
                                                                                      params['alpha1'],
                                                                                      params['alpha2'], params['beta1'],
                                                                                      params['beta2'])
            end = time.time()
            runtimes.append(end - start)
            theoretical_algorithm_error = utilities.subspace_distance(approximate_theoretical_NG_subspace, NG_subspace)
            theoretical_errors = np.append(theoretical_errors, theoretical_algorithm_error)

        except ValueError as e:
            print(e)
            theoretical_errors = np.append(theoretical_errors, 5)


    # print('theoretical_errors')
    # print(theoretical_errors)
    print('avg runtime')
    print(np.average(runtimes))
    # print('blanchard_errors')
    # print(blanchard_errors)

    index_of_min_error = np.argmin(theoretical_errors)
    optimal_error = theoretical_errors[index_of_min_error]
    optimal_params = params_array[index_of_min_error]
    print('optimal_error')
    print(optimal_error)
    print('optimal_params')
    print(optimal_params)
    axes = plt.gca()
    axes.set_xlim([0, 5])
    plt.scatter(theoretical_errors, np.zeros(iterations_num), c='r')
    plt.title('Errors for tuning')
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    main()
