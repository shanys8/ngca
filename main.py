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
    return np.array([(math.exp((-1) * alpha * math.pow(LA.norm(sample), 2))) for sample in samples]).sum()


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
    eigenvalues, eigenvectors = LA.eig(matrix)
    relevant_eigenvectors = np.empty((eigenvectors.shape[0], 0), float)
    i = 0
    while i < len(eigenvalues):
        if math.fabs(eigenvalues[i] - gaussian_eigenvalue) > threshold:
            relevant_eigenvectors = np.append(relevant_eigenvectors, eigenvectors[:, i][:, np.newaxis], axis=1)

        i += 1

    return relevant_eigenvectors


# check whether this is a vector that belongs to both eigenspaces e1, e2
def union_subspace(e1, e2):
    return np.concatenate((e1, e2), axis=1)


def polynom(degree_r, n_param, epsilon_param, delta_param, D_param, K_param):
    return 2


def generate_gaussian_subspace(rows, cols):
    mu, sigma = 0, 1  # mean and standard deviation
    np.random.seed(1234)
    return np.random.normal(mu, sigma, (rows, cols))


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


def generate_isotropic_samples(Q, Q_orthogonal, N, n, d):
    samples = np.empty((n, 0), float)
    samples_copy = np.empty((n, 0), float)

    # Samples should be of the isotripic model

    for _ in range(N):
        # each sample should have mean zero
        sample = np.dot(Q, np.random.rand(n - d, 1)) + np.dot(Q_orthogonal, np.random.rand(d, 1))
        samples = np.append(samples, sample, axis=1)

    whiten_samples = whiten(center(samples))

    for _ in range(N):
        sample = np.dot(Q, np.random.rand(n - d, 1)) + np.dot(Q_orthogonal, np.random.rand(d, 1))
        samples_copy = np.append(samples_copy, sample, axis=1)

    whiten_samples_copy = whiten(center(samples_copy))

    return whiten_samples, whiten_samples_copy


def orthogonal_complement(x: object, normalize: object = True, threshold: object = 1e-15) -> object:
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


def assert_isotropic_model(X):
    assert (np.allclose(np.mean(X, axis=0), np.zeros(X.shape[1]), rtol=1.e-2,
                        atol=1.e-2))  # each column vector should have mean zero
    cov_X = np.cov(X, rowvar=False, bias=True)
    print_matrix(cov_X)
    assert (cov_X.shape[0] == cov_X.shape[1]) and np.allclose(cov_X, np.eye(cov_X.shape[0]), rtol=1.e-1,
                                                              atol=1.e-1)  # covariance matrix should by identity


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
    alpha1 = 0.3
    alpha2 = 0.4
    beta1 = 0.5
    beta2 = 0.6

    N = polynom(r, n, 1 / epsilon, math.log(1 / delta), 1 / D, K)  # number of samples to generate

    G = generate_gaussian_subspace(n, n - d)

    # verify that G is gaussian is we expect
    # sns.distplot(G[:, 0], color="#53BB04")
    # plt.show()

    # generate gaussian subspace
    Q, _ = np.linalg.qr(G)  # QR decomposition from Gaussian Matrix size: n X (n-d)

    # generate subspace orthogonal to the gaussian (non gaussian) - Matrix size: n X d (REQUESTED E)
    Q_orthogonal = orthogonal_complement(Q, normalize=True)

    samples, samples_copy = generate_isotropic_samples(Q, Q_orthogonal, N, n, d)

    assert_isotropic_model(samples)
    assert_isotropic_model(samples_copy)

    # Calculate matrices
    matrix_phi = compute_matrix_phi(samples, samples_copy, alpha1)
    print('\nmatrix_phi')
    print_matrix(matrix_phi)
    matrix_psi = compute_matrix_psi(samples, samples_copy, alpha2)
    print('\nmatrix_psi')
    print_matrix(matrix_psi)

    # Calculate the gaussian eigenvalue for each matrix
    gaussian_phi_eigenvalue = calculate_gaussian_phi_eigenvalue(alpha1)
    gaussian_psi_eigenvalue = calculate_gaussian_psi_eigenvalue(alpha2)

    print('\ngaussian_phi_eigenvalue: ', gaussian_phi_eigenvalue)
    print('\ngaussian_psi_eigenvalue: ', gaussian_psi_eigenvalue)

    # Calculate corresponding eigenvectors for the relevant eigenvalues -
    # those which are far away beta from the gaussian eigenvalues
    matrix_phi_relevant_eigenvectors = get_matrix_relevant_eigenvectors(matrix_phi,
                                                                        gaussian_phi_eigenvalue, beta1)
    matrix_psi_relevant_eigenvectors = get_matrix_relevant_eigenvectors(matrix_psi,
                                                                        gaussian_psi_eigenvalue, beta2)

    # Calculate E space - non gaussian space
    result_space = union_subspace(matrix_phi_relevant_eigenvectors, matrix_psi_relevant_eigenvectors)



    print('\nresult_space')
    print_matrix(result_space)

    i = 0
    random_spanned_vector = np.zeros((result_space.shape[0], 1), float)
    while i < result_space.shape[1]:
        random_spanned_vector += np.random.random_sample() * result_space[:, i][:, np.newaxis]
        i += 1


    print('\nE')
    print_matrix(Q_orthogonal)

    linear_combination_of_Q_orthogonal_columns = np.linalg.lstsq(Q_orthogonal, random_spanned_vector)

    # if linear_combination_of_Q_orthogonal_columns exist then we can find for each vector spanned by result_space
    # a linear combination of columns of E so it is also belongs to E (need to check if it is epsilon close to vector in E)
    print(linear_combination_of_Q_orthogonal_columns)

    return result_space


if __name__ == "__main__":
    main()
