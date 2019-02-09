import numpy as np
import pandas as pd
import math
from numpy import linalg as LA
from scipy import linalg
from scipy import stats

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from utils.mnist_reader import load_mnist


def print_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def compute_matrix_phi(samples, samples_copy, alpha):
    return (1 / compute_z_phi(samples, alpha)) * \
           (np.array([(math.exp((-1) * alpha * math.pow(LA.norm(sample), 2)) * np.dot(sample, sample))
                      for sample in samples]).sum())


def compute_z_phi(samples, alpha):
    return np.array([(math.exp((-1) * alpha * math.pow(LA.norm(sample), 2))) for sample in samples]).sum()


def compute_matrix_psi(samples, samples_copy, alpha):
    samples_tuple_array = np.array((samples, samples_copy)).T
    return (1 / compute_z_psi(samples, samples_copy, alpha)) * \
           (np.array([(math.exp((-1) * alpha * np.dot(samples_tuple[0], samples_tuple[1])) *
                       (np.dot(samples_tuple[0], samples_tuple[1].T) + np.dot(samples_tuple[1], samples_tuple[0].T)))
                      for samples_tuple in samples_tuple_array]).sum())


def compute_z_psi(samples, samples_copy, alpha):
    samples_tuple_array = np.array((samples, samples_copy)).T
    return 2 * np.array([(math.exp((-1) * alpha * np.dot(samples_tuple[0], samples_tuple[1])))
                         for samples_tuple in samples_tuple_array]).sum()


def get_matrix_relevant_eigenvalues(matrix_eigenvalues, gaussian_eigenvalue, threshold):
    return np.where(math.fabs(matrix_eigenvalues - gaussian_eigenvalue) > threshold)


def calculate_gaussian_phi_eigenvalue(alpha):
    return math.pow(2 * alpha + 1, -1)


def calculate_gaussian_psi_eigenvalue(alpha):
    return alpha * math.pow(alpha * alpha - 1, -1)


def get_matrix_relevant_eigenvectors(matrix, gaussian_eigenvalue, threshold):
    eigenvalues, eigenvectors = LA.eig(matrix)

    relevant_eigenvectors = np.array()
    for i in range(len(eigenvalues)):
        if math.fabs(eigenvalues[i] - gaussian_eigenvalue) > threshold:
            relevant_eigenvectors = np.append(relevant_eigenvectors, eigenvectors[i])

    return relevant_eigenvectors


def calculate_approximate_non_gaussian_space(e1, e2):
    intersection = np.intersect1d(e1, e2)
    return np.where(LA.norm(intersection) == 1)


def polynom(degree_r, n_param, epsilon_param, delta_param, D_param, K_param):
    return degree_r


def generate_gaussian_subspace(rows, cols):
    mu, sigma = 0, 1  # mean and standard deviation
    return np.random.normal(mu, sigma, (rows, cols))


def generate_samples(G, N, n, d):
    Q, _ = np.linalg.qr(G)  # QR decomposition from Gaussian Matrix size: n X (n-d)
    # Q_orthogonal = linalg.orth(Q)   # Orthonormal subspace for Q size: n X d

    # print('Q')
    # print_matrix(Q)
    Q_orthogonal = orthogonal_complement(Q, normalize=True)

    # print('Q_orthogonal')
    # print_matrix(Q_orthogonal)
    samples = np.empty((n, 0), float)
    samples_copy = np.empty((n, 0), float)

    for _ in range(N):
        sample = stats.zscore(np.dot(Q, np.random.rand(n - d, 1)) + \
                              np.dot(Q_orthogonal, np.random.rand(d, 1)))

        # TODO sample should be of the isotripic model
        print(np.dot(sample, sample.T))  # should by eye

        samples = np.append(samples, sample, axis=1)

    for _ in range(N):
        sample = stats.zscore(np.dot(Q, np.random.rand(n - d, 1)) + \
                              np.dot(Q_orthogonal, np.random.rand(d, 1)))
        samples_copy = np.append(samples_copy, sample, axis=1)

    return samples, samples_copy


def orthogonal_complement(x, normalize=True, threshold=1e-15):
    """Compute orthogonal complement of a matrix

    this works along axis zero, i.e. rank == column rank,
    or number of rows > column rank
    otherwise orthogonal complement is empty

    TODO possibly: use normalize='top' or 'bottom'

    """
    x = np.asarray(x)
    r, c = x.shape
    if r < c:
        import warnings
        warnings.warn('fewer rows than columns', UserWarning)

    # we assume svd is ordered by decreasing singular value, o.w. need sort
    s, v, d = np.linalg.svd(x)
    rank = (v > threshold).sum()

    oc = s[:, rank:]

    if normalize:
        k_oc = oc.shape[1]
        oc = oc.dot(np.linalg.inv(oc[:k_oc, :]))
    return oc


def main():
    # phi - represent the X 2-norm gaussian measure
    # psi - represent the <X,X'> gaussian measure

    # input
    n = 5  # dimension
    d = 2  # subspace dimension
    epsilon = 0.2  # how close result vectors will be from E
    delta = 0.4  # probability 1-delta to success
    D = 3  # deviation from gaussian moments
    K = 1  # sub gaussian norm max bound
    r = 4  # polynom degree - calculation D from r
    alpha1 = 0.3
    alpha2 = 0.4
    beta1 = 0.5
    beta2 = 0.6

    N = polynom(r, n, 1 / epsilon, math.log(1 / delta), 1 / D, K)  # number of samples to generate

    G = generate_gaussian_subspace(n, n - d)

    samples, samples_copy = generate_samples(G, N, n, d)

    # Calculate matrices
    matrix_phi = compute_matrix_phi(samples, samples_copy, alpha1) #TODO bug - return integer instead of matrix
    matrix_psi = compute_matrix_psi(samples, samples_copy, alpha2)

    # Calculate the gaussian eigenvalue for each matrix
    gaussian_phi_eigenvalue = calculate_gaussian_phi_eigenvalue(alpha1)
    gaussian_psi_eigenvalue = calculate_gaussian_psi_eigenvalue(alpha2)

    # Calculate corresponding eigenvectors for the relevant eigenvalues -
    # those which are far away beta from the gaussian eigenvalues
    matrix_phi_relevant_eigenvectors = get_matrix_relevant_eigenvectors(matrix_phi,
                                                                        gaussian_phi_eigenvalue, beta1)
    matrix_psi_relevant_eigenvectors = get_matrix_relevant_eigenvectors(matrix_psi,
                                                                        gaussian_psi_eigenvalue, beta2)

    # Calculate E space - non gaussian space
    approximate_non_gaussian_space = \
        calculate_approximate_non_gaussian_space(matrix_phi_relevant_eigenvectors,
                                                 matrix_psi_relevant_eigenvectors)

    return approximate_non_gaussian_space


if __name__ == "__main__":
    main()
