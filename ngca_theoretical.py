import numpy as np
import pandas as pd
import math
from numpy import linalg as LA
from scipy import linalg
from scipy import stats
# from scipy.cluster.vq import whiten
import seaborn as sns

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import time

import utilities

def plotDataAndCov(data):
    ACov = np.cov(data, rowvar=False, bias=True)
    print('\nCovariance matrix:\n', ACov)


def print_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def compute_matrix_phi(samples, samples_copy, alpha):
    return (1 / compute_z_phi(samples, alpha)) * \
           (np.array([(math.exp((-1) * alpha * math.pow(LA.norm(sample), 2)) *
                       np.dot(sample[:, np.newaxis], sample[np.newaxis, :]))
                      for sample in samples.T]).sum(axis=0))


def compute_z_phi(samples, alpha):
    return np.array([(math.exp((-1) * alpha * math.pow(LA.norm(sample), 2))) for sample in samples.T]).sum()


def compute_matrix_psi(samples, samples_copy, alpha):
    return_val = 0
    i = 0
    while i < len(samples.T):
        return_val += math.exp((-1) * alpha * np.dot(samples.T[i], samples_copy.T[i])) * \
                      (np.dot(samples.T[i][:, np.newaxis], samples_copy.T[i][np.newaxis, :]) +
                       np.dot(samples_copy.T[i][:, np.newaxis], samples.T[i][np.newaxis, :]))
        i += 1

    return (1 / compute_z_psi(samples, samples_copy, alpha)) * return_val


def compute_z_psi(samples, samples_copy, alpha):
    return_val = 0
    i = 0
    while i < len(samples.T):
        return_val += math.exp((-1) * alpha * np.dot(samples.T[i], samples_copy.T[i]))
        i += 1

    return 2 * return_val


def get_matrix_relevant_eigenvalues(matrix_eigenvalues, gaussian_eigenvalue, threshold):
    return np.where(math.fabs(matrix_eigenvalues - gaussian_eigenvalue) > threshold)


def calculate_gaussian_phi_eigenvalue(alpha):
    return math.pow(2 * alpha + 1, -1)


def calculate_gaussian_psi_eigenvalue(alpha):
    return alpha * math.pow(alpha * alpha - 1, -1)


def get_matrix_relevant_eigenvectors(matrix, gaussian_eigenvalue, threshold):
    min_eigenvalue_to_count = 1e-5
    eigenvalues, eigenvectors = LA.eigh(matrix)
    if eigenvectors.dtype.name != 'float64':
        raise ValueError('eigenvalues are not all floats')
    relevant_eigenvectors = np.empty((eigenvectors.shape[0], 0), float)
    i = 0
    while i < len(eigenvalues):
        if eigenvalues[i] > min_eigenvalue_to_count and math.fabs(eigenvalues[i] - gaussian_eigenvalue) > threshold:
            relevant_eigenvectors = np.append(relevant_eigenvectors, eigenvectors[:, i][:, np.newaxis], axis=1)
        i += 1

    return relevant_eigenvectors


def union_subspace(e1, e2):
    return np.concatenate((e1, e2), axis=1)


def polynom(degree_r, n_param, epsilon_param, delta_param, D_param, K_param):
    return 2


def center(X):
    newX = X - np.mean(X, axis=0)
    return newX


def whiten(X):
    # newX = center(X)
    cov = X.T.dot(X) / float(X.shape[0])
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigVals, eigVecs = np.linalg.eig(cov)
    # Apply the eigenvectors to X
    decorrelated = X.dot(eigVecs)
    # Rescale the decorrelated data
    whitened = decorrelated / np.sqrt(eigVals + 1e-5)
    return whitened


def generate_synthetic_isotropic_samples(N, n, d):

    G = utilities.generate_gaussian_subspace(n, n - d)
    Q, _ = np.linalg.qr(G)  # QR decomposition from Gaussian Matrix size: n X (n-d)

    # generate subspace orthogonal to the gaussian (non gaussian) - Matrix size: n X d (REQUESTED E)
    Q_orthogonal = utilities.orthogonal_complement(Q, normalize=True)

    assert_all_columns_unit_vectors(Q_orthogonal)

    samples = np.empty((n, 0), float)
    samples_copy = np.empty((n, 0), float)

    # Samples should be of the isotripic model

    for _ in range(N):
        # each sample should have mean zero
        sample = np.dot(Q, np.random.randn(n - d, 1)) + np.dot(Q_orthogonal, (np.random.rand(d, 1) - 0.5))  # X = S + N
        samples = np.append(samples, sample, axis=1)

    for _ in range(N):
        sample = np.dot(Q, np.random.randn(n - d, 1)) + np.dot(Q_orthogonal, (np.random.rand(d, 1) - 0.5))  # X = S + N
        samples_copy = np.append(samples_copy, sample, axis=1)

    return samples, samples_copy, Q_orthogonal


def assert_isotropic_model(X):
    assert (np.allclose(np.mean(X, axis=0), np.zeros(X.shape[1]), rtol=1.e-2,
                        atol=1.e-2))  # each column vector should have mean zero
    cov_X = np.cov(X, rowvar=False, bias=True)
    # print_matrix(cov_X)
    assert (cov_X.shape[0] == cov_X.shape[1]) and np.allclose(cov_X, np.eye(cov_X.shape[0]), rtol=1.e-1,
                                                              atol=1.e-1)  # covariance matrix should by identity


def run_ngca_algorithm(samples, samples_copy, alpha1, alpha2, beta1, beta2):
    # Calculate matrices
    matrix_phi = compute_matrix_phi(samples, samples_copy, alpha1)

    matrix_psi = compute_matrix_psi(samples, samples_copy, alpha2)

    # Calculate the gaussian eigenvalue for each matrix
    gaussian_phi_eigenvalue = calculate_gaussian_phi_eigenvalue(alpha1)
    gaussian_psi_eigenvalue = calculate_gaussian_psi_eigenvalue(alpha2)

    # print('\ngaussian_phi_eigenvalue: ', gaussian_phi_eigenvalue)
    # print('\ngaussian_psi_eigenvalue: ', gaussian_psi_eigenvalue)

    # Calculate corresponding eigenvectors for the relevant eigenvalues -
    # those which are far away beta from the gaussian eigenvalues
    matrix_phi_relevant_eigenvectors = get_matrix_relevant_eigenvectors(matrix_phi,
                                                                        gaussian_phi_eigenvalue, beta1)
    matrix_psi_relevant_eigenvectors = get_matrix_relevant_eigenvectors(matrix_psi,
                                                                        gaussian_psi_eigenvalue, beta2)

    # print('\nmatrix_phi_relevant_eigenvectors')
    # print_matrix(matrix_phi_relevant_eigenvectors)
    #
    # print('\nmatrix_psi_relevant_eigenvectors')
    # print_matrix(matrix_psi_relevant_eigenvectors)

    # Calculate E space - non gaussian space
    result_space = union_subspace(matrix_phi_relevant_eigenvectors, matrix_psi_relevant_eigenvectors)

    assert_all_columns_unit_vectors(result_space)

    return result_space


def assert_all_columns_unit_vectors(matrix):
    i = 0
    while i < matrix.shape[1]:
        assert (is_unit_vector(matrix[:, i][:, np.newaxis]))
        i += 1


def is_unit_vector(vector):
    return math.isclose(LA.norm(vector),  1.0, rel_tol=1e-2)


def check_if_epsilon_close_to_vector_in_subspace(vector, subspace):
    i = 0
    while i < subspace.shape[1]:
        col = subspace[:, i][:, np.newaxis]
        dist = LA.norm(vector - col)
        # print('for vector ', vector, ' dist is ', dist)
        i += 1


def validate_approximation(approx_subspace, subspace):
    print('distance between approx and real subspace ')

    print(utilities.subspace_distance(approx_subspace, subspace))

    # i = 0
    # while i < approx_subspace.shape[1]:
    #     col = approx_subspace[:, i][:, np.newaxis]
    #     check_if_epsilon_close_to_vector_in_subspace(col, subspace)
    #     i += 1
    # i = 0
    # random_spanned_vector = np.zeros((approximate_E.shape[0], 1), float)
    # while i < approximate_E.shape[1]:
    #     random_spanned_vector += np.random.random_sample() * approximate_E[:, i][:, np.newaxis]
    #     i += 1

    # linear_combination_of_Q_orthogonal_columns = np.linalg.lstsq(E, random_spanned_vector)

    # if linear_combination_of_Q_orthogonal_columns exist then we can find for each vector spanned by approximate_E
    # a linear combination of columns of E so it is also belongs to E (need to check if it is epsilon close to vector in E)
    # print(linear_combination_of_Q_orthogonal_columns)


def main():
    # phi - represent the X 2-norm gaussian measure
    # psi - represent the <X,X'> gaussian measure

    # input
    n = 4  # dimension
    d = 2  # subspace dimension
    epsilon = 0.2  # how close result vectors will be from E
    delta = 0.4  # probability 1-delta to success
    D = 3  # deviation from gaussian moments
    K = 1  # sub gaussian norm max bound
    r = 2  # polynom degree - calculation D from r
    alpha1 = 2
    alpha2 = 2
    beta1 = 0.5
    beta2 = 0.5

    N = polynom(r, n, 1 / epsilon, math.log(1 / delta), 1 / D, K)  # number of samples to generate

    samples, samples_copy, E = generate_synthetic_isotropic_samples(N, n, d)

    # Implementation of algorithm in the paper
    approximate_E = run_ngca_algorithm(samples, samples_copy, alpha1, alpha2, beta1, beta2)

    print('\napproximate_E')
    print_matrix(approximate_E)
    #
    # print('\nE')
    # print_matrix(E)

    validate_approximation(approximate_E, E)

    return approximate_E


if __name__ == "__main__":
    main()
