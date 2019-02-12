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
    print('Covariance matrix:\n', ACov)
    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # fig.set_size_inches(10, 10)
    # # Choosing the colors
    # cmap = sns.color_palette("GnBu", 10)
    # sns.heatmap(ACov, cmap=cmap, vmin=0)
    # ax1 = plt.subplot(2, 2, 2)
    # # data can include the colors
    # if data.shape[1] == 3:
    #     c = data[:, 2]
    # else:
    #     c = "#0A98BE"
    # ax1.scatter(data[:, 0], data[:, 1], c=c, s=40)
    # # Remove the top and right axes from the data plot
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['top'].set_visible(False)

def print_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


# def whiten(X, fudge=1E-18):
#
#    # the matrix X should be observations-by-components
#
#    # get the covariance matrix
#    Xcov = np.dot(X.T,X)
#
#    # eigenvalue decomposition of the covariance matrix
#    d, V = np.linalg.eigh(Xcov)
#
#    # a fudge factor can be used so that eigenvectors associated with
#    # small eigenvalues do not get overamplified.
#    D = np.diag(1. / np.sqrt(d+fudge))
#
#    # whitening matrix
#    W = np.dot(np.dot(V, D), V.T)
#
#    # multiply by the whitening matrix
#    X_white = np.dot(X, W)
#
#    return X_white

def svd_whiten(X):

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white

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
    np.random.seed(1234)
    return np.random.normal(mu, sigma, (rows, cols))


def center(X):
    newX = X - np.mean(X, axis = 0)
    return newX

def standardize(X):
    newX = center(X)/np.std(X, axis = 0)
    return newX


def decorrelate(X):
    newX = center(X)
    cov = X.T.dot(X)/float(X.shape[0])
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigVals, eigVecs = np.linalg.eig(cov)
    # Apply the eigenvectors to X
    decorrelated = X.dot(eigVecs)
    return decorrelated


def whiten(X):
    # newX = center(X)
    cov = X.T.dot(X)/float(X.shape[0])
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigVals, eigVecs = np.linalg.eig(cov)
    # Apply the eigenvectors to X
    decorrelated = X.dot(eigVecs)
    # Rescale the decorrelated data
    whitened = decorrelated / np.sqrt(eigVals + 1e-5)
    return whitened


def generate_isotropic_samples(G, N, n, d):
    Q, _ = np.linalg.qr(G)  # QR decomposition from Gaussian Matrix size: n X (n-d)
    Q_orthogonal = orthogonal_complement(Q, normalize=True)  # subspace orthogonal to Q - non gaussian Matrix size: n X d

    samples = np.empty((n, 0), float)
    samples_copy = np.empty((n, 0), float)

    # Samples should be of the isotripic model

    for _ in range(N):
        # each sample should have mean zero
        sample = np.dot(Q, np.random.rand(n - d, 1)) + np.dot(Q_orthogonal, np.random.rand(d, 1))
        samples = np.append(samples, sample, axis=1)

    whiten_samples = whiten(center(samples))
    
    # # verify that covariance is I
    # print('samples matrix cov')
    # plotDataAndCov(samples)
    # whiten_samples = whiten(center(samples))
    # print('whiten sample covariance matrix')
    # plotDataAndCov(whiten_samples)

    for _ in range(N):
        sample = np.dot(Q, np.random.rand(n - d, 1)) + np.dot(Q_orthogonal, np.random.rand(d, 1))
        samples_copy = np.append(samples_copy, sample, axis=1)

    whiten_samples_copy = whiten(center(samples_copy))

    return whiten_samples, whiten_samples_copy


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

    samples, samples_copy = generate_isotropic_samples(G, N, n, d)

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
