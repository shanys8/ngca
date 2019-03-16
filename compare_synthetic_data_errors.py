import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import utilities
import ngca_blanchard
import ngca_theoretical


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


def main():

    n = 8  # dimension (number of features) should be 10
    d = 3  # subspace dimension - requested dimension of NG data
    N = 10  # number of samples to generate should be 1000

    # Theoretical NGCA input
    alpha1 = 2
    alpha2 = 2
    beta1 = 0.5
    beta2 = 0.5

    # Blanchard NGCA input
    epsilon = 0.4  # should be 1.5
    T = 10  # should be 10
    num_of_samples_in_range = 3  # range divided into num - should be 1000

    theoretical_errors = np.empty(0, float)
    blanchard_errors = np.empty(0, float)

    for _ in range(10):
        # Generate synthetic samples
        samples, NG_subspace = generate_synthetic_isotropic_samples(N, n, d)
        theoretical_samples = samples[:, :int(N/2)]
        theoretical_samples_copy = samples[:, int(N/2):]

        # Theoretical NGCA algorithm
        approximate_theoretical_NG_subspace = ngca_theoretical.run_ngca_algorithm(theoretical_samples, theoretical_samples_copy, alpha1, alpha2, beta1, beta2)
        print('approximate_theoretical_NG_subspace')
        print(approximate_theoretical_NG_subspace)
        theoretical_algorithm_error = utilities.subspace_distance(approximate_theoretical_NG_subspace, NG_subspace)
        theoretical_errors = np.append(theoretical_errors, theoretical_algorithm_error)

        # Blanchard NGCA algorithm
        approximate_blanchard_NG_subspace = ngca_blanchard.run_ngca_algorithm(samples, n, T, epsilon, num_of_samples_in_range, d)
        print('approximate_blanchard_NG_subspace')
        print(approximate_blanchard_NG_subspace)
        blanchard_algorithm_error = utilities.blanchard_subspace_distance(approximate_blanchard_NG_subspace, NG_subspace)
        blanchard_errors = np.append(blanchard_errors, blanchard_algorithm_error)

    print('theoretical_errors')
    print(theoretical_errors)
    print('blanchard_errors')
    print(blanchard_errors)

    plt.scatter(theoretical_errors, blanchard_errors, c='r')
    plt.xlabel('theoretical_errors')
    plt.ylabel('blanchard_errors')
    plt.title('Errors comparison')
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    main()
