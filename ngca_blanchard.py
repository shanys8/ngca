import numpy as np
import pandas as pd
import math
from numpy import linalg as LA
from scipy import linalg
from scipy import stats
import seaborn as sns
from numpy.linalg import matrix_power
from scipy.linalg import fractional_matrix_power

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


def generate_gaussian_subspace(rows, cols):
    mu, sigma = 0, 1  # mean and standard deviation
    np.random.seed(1234)
    return np.random.normal(mu, sigma, (rows, cols))


def center(X):
    newX = X - np.mean(X, axis=0)
    return newX


def generate_synthetic_isotropic_samples(N, n, d):

    G = generate_gaussian_subspace(n, n - d)

    # verify that G is gaussian is we expect
    # sns.distplot(G[:, 0], color="#53BB04")
    # plt.show()

    # generate gaussian subspace
    Q, _ = np.linalg.qr(G)  # QR decomposition from Gaussian Matrix size: n X (n-d)

    # generate subspace orthogonal to the gaussian (non gaussian) - Matrix size: n X d (REQUESTED E)
    Q_orthogonal = orthogonal_complement(Q, normalize=True)

    # print('\n Q')
    # print_matrix(Q)
    # print('\n Q_orthogonal')
    # print_matrix(Q_orthogonal)

    assert_all_columns_unit_vectors(Q_orthogonal)

    samples = np.empty((n, 0), float)
    samples_copy = np.empty((n, 0), float)

    # Samples should be of the isotripic model

    for _ in range(N):
        # each sample should have mean zero
        sample = np.dot(Q, np.random.rand(n - d, 1)) + np.dot(Q_orthogonal, np.random.rand(d, 1))
        samples = np.append(samples, sample, axis=1)

    centered_samples = center(samples)

    # assert_isotropic_model(whiten_samples)

    return centered_samples, Q_orthogonal


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

    oc, _ = np.linalg.qr(oc)

    return oc


def assert_isotropic_model(X):
    assert (np.allclose(np.mean(X, axis=0), np.zeros(X.shape[1]), rtol=1.e-2,
                        atol=1.e-2))  # each column vector should have mean zero
    cov_X = np.cov(X, rowvar=False, bias=True)
    # print_matrix(cov_X)
    assert (cov_X.shape[0] == cov_X.shape[1]) and np.allclose(cov_X, np.eye(cov_X.shape[0]), rtol=1.e-1,
                                                              atol=1.e-1)  # covariance matrix should by identity


def assert_all_columns_unit_vectors(matrix):
    i = 0
    while i < matrix.shape[1]:
        assert (is_unit_vector(matrix[:, i][:, np.newaxis]))
        i += 1


def is_unit_vector(vector):
    return math.isclose(LA.norm(vector),  1.0, rel_tol=1e-2)


def generate_lambdas(num_of_samples_in_range):
    sigma_values = np.sqrt(get_values_list_in_rage(0.5, 5, num_of_samples_in_range))
    a_values = get_values_list_in_rage(0, 4, num_of_samples_in_range)
    b_values = get_values_list_in_rage(0, 5, num_of_samples_in_range)

    gauss_pow3 = lambda sigma: lambda z: math.pow(z, 3) * math.exp(((-1)*math.pow(z, 2)) / 2*math.pow(sigma, 2))
    list_of_gauss_pow3_lambdas = [gauss_pow3(sigma) for sigma in sigma_values]
    Fourier = lambda a: lambda z: complex(0, math.sin(a*z))
    list_of_fourier_lambdas = [Fourier(a) for a in a_values]
    hyperbolic_tangent = lambda b: lambda z: np.tanh(b*z)
    list_of_hyperbolic_tangent_lambdas = [hyperbolic_tangent(b) for b in b_values]

    lambdas = (list_of_gauss_pow3_lambdas, list_of_fourier_lambdas, list_of_hyperbolic_tangent_lambdas)

    return np.concatenate(lambdas)

def run_ngca_algorithm(samples, T, epsilon, num_of_samples_in_range):
    lambdas = generate_lambdas(num_of_samples_in_range)
    # for lambda_function in lambdas:
    #     w0 = generate_unit_vector()
    #     i = 1
    #     for i <= T
    #         i += 1

    return True


def get_values_list_in_rage(min, max, num_of_samples):
    return np.arange(min, max, (max-min)/num_of_samples)


def whiten_covariance(samples, sigma_circumflex):
    return np.dot(fractional_matrix_power(sigma_circumflex, -0.5), samples)

def main():

    # input
    n = 4  # dimension
    d = 2  # subspace dimension
    N = 3  # number of samples to generate
    m = 3  # requested dimension of NG data
    epsilon = 1.5
    T = 10
    num_of_samples_in_range = 10


    samples, NG_subspace = generate_synthetic_isotropic_samples(N, n, d)

    sigma_circumflex = np.cov(samples)

    whiten_samples = whiten_covariance(samples, sigma_circumflex)

    # Implementation of algorithm in the paper
    approximate_NG_subspace = run_ngca_algorithm(whiten_samples, T, epsilon, num_of_samples_in_range)

    print('\napproximate_NG_subspace')
    print_matrix(approximate_NG_subspace)

    print('\nNG_subspace')
    print_matrix(NG_subspace)

    return approximate_NG_subspace


if __name__ == "__main__":
    main()
