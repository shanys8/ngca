import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import utilities
from not_in_use import ngca_old, ngca_blanchard
import math

# N num of samples
# n dimesion of samples
# d dimension of NG subspace
def generate_synthetic_isotropic_samples(N, n, d):

    G = utilities.generate_gaussian_subspace(n, n - d)
    Q, R = LA.qr(G)  # QR decomposition from Gaussian Matrix size: n X (n-d)

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


def main():

    n = 8  # dimension (number of features) should be 10
    d = 3  # subspace dimension - requested dimension of NG data
    N = 1000  # number of samples to generate should be 1000
    iterations_num = 30
    # Theoretical NGCA input

    # Blanchard NGCA input
    epsilon = 0.3  # should be 1.5
    T = 10  # should be 10
    num_of_samples_in_range = 5  # range divided into num - should be 1000

    theoretical_errors = np.empty(0, float)
    blanchard_errors = np.empty(0, float)

    for i in range(iterations_num):
        print('---------------------------- Round {} out of {} ----------------------------'.format(i, iterations_num))
        # Generate synthetic samples
        samples, NG_subspace = generate_synthetic_isotropic_samples(N, n, d)
        theoretical_samples = samples[:, :int(N/2)]
        theoretical_samples_copy = samples[:, int(N/2):]

        try:
            # Blanchard NGCA algorithm
            approximate_blanchard_NG_subspace = ngca_blanchard.run_ngca_algorithm(samples, n, T, epsilon,
                                                                                  num_of_samples_in_range, d)
            # print('approximate_blanchard_NG_subspace')
            # print(approximate_blanchard_NG_subspace)
            blanchard_algorithm_error = utilities.blanchard_subspace_distance(approximate_blanchard_NG_subspace,
                                                                              NG_subspace)

            if blanchard_algorithm_error > 10 or math.isnan(blanchard_algorithm_error):
                print('error too big or nan')
                continue
            blanchard_errors = np.append(blanchard_errors, blanchard_algorithm_error)

            # Theoretical NGCA algorithm
            params = {'alpha1': 0.6754445940381727, 'alpha2': 0.29744739800298886, 'beta1': 0.3403472323546272,
                      'beta2': 0.6441926407645018}
            approximate_theoretical_NG_subspace = ngca_old.run_ngca_algorithm(theoretical_samples,
                                                                              theoretical_samples_copy,
                                                                              params['alpha1'],
                                                                              params['alpha2'], params['beta1'],
                                                                              params['beta2'])

            theoretical_algorithm_error = utilities.subspace_distance(approximate_theoretical_NG_subspace, NG_subspace)
            theoretical_errors = np.append(theoretical_errors, theoretical_algorithm_error)
        except ValueError as e:
            print(e)

    blanchard_errors = blanchard_errors[:len(theoretical_errors)]
    print('theoretical_errors')
    print(theoretical_errors)
    print('blanchard_errors')
    print(blanchard_errors)

    axes = plt.gca()
    x_max_range = np.max(theoretical_errors) + 0.1
    y_max_range = np.max(blanchard_errors) + (np.max(blanchard_errors) - np.min(blanchard_errors))
    axes.set_xlim([np.min(theoretical_errors) - 0.1, x_max_range])
    axes.set_ylim([np.min(blanchard_errors) - 1e-16, y_max_range])
    plt.scatter(theoretical_errors, blanchard_errors, c='r')
    plt.xlabel('theoretical_errors')
    plt.ylabel('blanchard_errors')
    plt.title('Errors comparison')
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    main()
