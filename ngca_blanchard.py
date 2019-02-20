import numpy as np
import math
from numpy import linalg as LA
from scipy import linalg
from scipy import stats
import seaborn as sns
from numpy.linalg import matrix_power
from scipy.linalg import fractional_matrix_power
from random import gauss
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
from sklearn import preprocessing


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
    Q, _ = LA.qr(G)  # QR decomposition from Gaussian Matrix size: n X (n-d)

    # generate subspace orthogonal to the gaussian (non gaussian) - Matrix size: n X d (REQUESTED E)
    Q_orthogonal = orthogonal_complement(Q, normalize=True)

    assert_all_columns_unit_vectors(Q_orthogonal)

    samples = np.empty((n, 0), float)

    # Samples should be of the isotripic model
    for _ in range(N):
        # each sample should have mean zero
        sample = np.dot(Q, np.random.rand(n - d, 1)) + np.dot(Q_orthogonal, np.random.rand(d, 1))  # X = S + N
        samples = np.append(samples, sample, axis=1)

    centered_samples = center(samples)

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


def generate_derivative_lambdas(num_of_samples_in_range):

    sigma_values = np.sqrt(get_values_list_in_rage(0.5, 5, num_of_samples_in_range))
    a_values = get_values_list_in_rage(0, 4, num_of_samples_in_range)
    b_values = get_values_list_in_rage(0, 5, num_of_samples_in_range)

    gauss_pow3_derivate = lambda sigma: lambda z: 3 * math.pow(z, 2) * math.exp(((-1)*math.pow(z, 2)) / (2*math.pow(sigma, 2))) + math.pow(z, 3) * (-1) * z * (1 / math.pow(sigma, 2)) * math.exp(((-1)*math.pow(z, 2)) / (2*math.pow(sigma, 2)))
    list_of_gauss_pow3_derivative_lambdas = [gauss_pow3_derivate(sigma) for sigma in sigma_values]
    Fourier_sin_derivative = lambda a: lambda z: a * math.cos(a*z)
    Fourier_cos_derivative = lambda a: lambda z: (-1) * a * math.sin(a*z)
    list_of_fourier_sin_derivate_lambdas = [Fourier_sin_derivative(a) for a in a_values]
    list_of_fourier_cos_derivate_lambdas = [Fourier_cos_derivative(a) for a in a_values]
    hyperbolic_tangent_derivate = lambda b: lambda z: b * (1.0 - np.tanh(b*z) ** 2)
    list_of_hyperbolic_tangent_derivate_ambdas = [hyperbolic_tangent_derivate(b) for b in b_values]

    lambdas = (list_of_gauss_pow3_derivative_lambdas, list_of_fourier_sin_derivate_lambdas, list_of_fourier_cos_derivate_lambdas, list_of_hyperbolic_tangent_derivate_ambdas)

    return np.concatenate(lambdas)


def generate_lambdas(num_of_samples_in_range):

    sigma_values = np.sqrt(get_values_list_in_rage(0.5, 5, num_of_samples_in_range))
    a_values = get_values_list_in_rage(0, 4, num_of_samples_in_range)
    b_values = get_values_list_in_rage(0, 5, num_of_samples_in_range)

    gauss_pow3 = lambda sigma: lambda z: math.pow(z, 3) * math.exp(((-1)*math.pow(z, 2)) / 2*math.pow(sigma, 2))
    list_of_gauss_pow3_lambdas = [gauss_pow3(sigma) for sigma in sigma_values]
    Fourier_sin = lambda a: lambda z: math.sin(a*z)
    Fourier_cos = lambda a: lambda z: math.cos(a*z)
    list_of_fourier_sin_lambdas = [Fourier_sin(a) for a in a_values]
    list_of_fourier_cos_lambdas = [Fourier_cos(a) for a in a_values]
    hyperbolic_tangent = lambda b: lambda z: np.tanh(b*z)
    list_of_hyperbolic_tangent_lambdas = [hyperbolic_tangent(b) for b in b_values]

    lambdas = (list_of_gauss_pow3_lambdas, list_of_fourier_sin_lambdas, list_of_fourier_cos_lambdas, list_of_hyperbolic_tangent_lambdas)

    return np.concatenate(lambdas)


def generate_unit_vector(dimension):
    vec = [gauss(0, 1) for i in range(dimension)]
    mag = sum(x**2 for x in vec) ** .5
    return np.array([x/mag for x in vec])[:, np.newaxis]


def calculate_beta(samples, lambda_function, derivative_lambda_function, curr_w):
    result = np.zeros((samples.shape[0], 1), float)

    for sample in samples.T:
        res1 = lambda_function(np.dot(curr_w, sample[:, np.newaxis])) * sample[:, np.newaxis]
        res2 = derivative_lambda_function(np.dot(curr_w, sample[:, np.newaxis])) * curr_w.T
        result += (res1 - res2)

    result = (1 / len(samples)) * result
    return result


def calculate_N(samples, lambda_function, derivative_lambda_function, curr_w, curr_beta):
    res = (1/len(samples)) * np.array([math.pow(LA.norm((lambda_function(np.dot(curr_w, sample[:, np.newaxis])) * sample[:, np.newaxis]) - (derivative_lambda_function(np.dot(curr_w, sample[:, np.newaxis])) * curr_w.T[:, np.newaxis])), 2) for sample in samples.T]).sum()
    return res - math.pow(LA.norm(curr_beta), 2)


# TODO - make more efficient: matrix[np.all([LA.norm(x) > epsilon], axis=1)]
def remove_small_vectors(matrix, epsilon):
    result = np.empty((matrix.shape[0], 0), float)
    i = 0
    while i < len(matrix.T):
        if LA.norm(matrix[:, i][np.newaxis]) > epsilon:
            result = np.append(result, matrix[:, i][np.newaxis].T, axis=1)
        i += 1
    return result


def run_pca(V, requested_dimension):
    sklearn_pca = sklearnPCA(n_components=requested_dimension, whiten=True)
    # (n_samples, n_components)
    sklearn_transf = sklearn_pca.fit_transform(V)
    return sklearn_transf


def run_ngca_algorithm(samples, samples_dimension, T, epsilon, num_of_samples_in_range, requested_dimension):
    lambdas = generate_lambdas(num_of_samples_in_range)
    derivative_lambdas = generate_derivative_lambdas(num_of_samples_in_range)
    w = np.empty((samples_dimension, 0), float)
    v = np.empty((samples_dimension, 0), float)

    k = 0
    while k < len(lambdas):
        print('Running {} out of {}'.format(k, len(lambdas)))
        lambda_function = lambdas[k]
        derivative_lambda_function = derivative_lambdas[k]
        w0 = generate_unit_vector(samples_dimension)
        w = np.append(w, w0, axis=1)
        t = 1
        while t <= T:
            curr_beta = calculate_beta(samples, lambda_function, derivative_lambda_function, w[:, t-1][np.newaxis])
            w = np.append(w, np.divide(curr_beta, LA.norm(curr_beta)), axis=1)
            t += 1
        N = calculate_N(samples, lambda_function, derivative_lambda_function, w[:, T-1], curr_beta)
        v = np.append(v, (curr_beta * math.sqrt(len(samples) / (N + 1e-10))), axis=1)
        k += 1

    # Threshold
    V = remove_small_vectors(v, epsilon)

    # PCA
    pca_result = run_pca(V, requested_dimension)

    normalize_pca_result = preprocessing.normalize(pca_result, axis=0, norm='l2')

    result = unwhiten(normalize_pca_result)

    return result


def get_values_list_in_rage(min, max, num_of_samples):
    return np.arange(min, max, (max-min)/num_of_samples)


def whiten_covariance(samples):
    # TODO check why makes matrix complex
    return samples
    # return np.dot(fractional_matrix_power(np.cov(samples), 0.5), samples)


def unwhiten(samples):
    # TODO check why makes matrix complex
    return samples
    # return np.dot(fractional_matrix_power(np.cov(samples), -0.5), samples)


def main():

    # input
    n = 5  # dimension (number of features)
    d = 2  # subspace dimension - requested dimension of NG data
    N = 3  # number of samples to generate
    epsilon = 1.5
    T = 10
    num_of_samples_in_range = 3  # range divided into num

    samples, NG_subspace = generate_synthetic_isotropic_samples(N, n, d)


    whiten_samples = whiten_covariance(samples)

    # Implementation of algorithm in the paper
    approximate_NG_subspace = run_ngca_algorithm(whiten_samples, n, T, epsilon, num_of_samples_in_range, d)

    print('\napproximate_NG_subspace')
    print_matrix(approximate_NG_subspace)

    print('\nNG_subspace')
    print_matrix(NG_subspace)

    return approximate_NG_subspace


if __name__ == "__main__":
    main()
