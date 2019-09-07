import numpy as np
from numpy import linalg as LA
import math
# from scipy import linalg
from scipy.linalg import hadamard
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.cluster import adjusted_rand_score


def download_data(file_name, separate_data=False):
    result = np.loadtxt(fname="datasets/{}.txt".format(file_name))
    return (result[:500], result[500:]) if separate_data else result


def download_labels(file_name):
    int_labels = []
    labels = np.loadtxt(fname="datasets/{}Lbls.txt".format(file_name))
    for label in labels:
        if np.array_equal(label.astype(int), [1, 0, 0]):
            int_labels = np.append(int_labels, 0)
        if np.array_equal(label.astype(int), [0, 1, 0]):
            int_labels = np.append(int_labels, 1)
        if np.array_equal(label.astype(int), [0, 0, 1]):
            int_labels = np.append(int_labels, 2)
    return int_labels.astype(int)


def get_result_score(proj_data, labels_true):
    kmeans = KMeans(n_clusters=3, random_state=0).fit(proj_data)
    labels_pred = kmeans.labels_
    # 0.0 for random labeling and samples and exactly 1.0 when the clusterings are identical
    score = adjusted_rand_score(labels_true, labels_pred)
    return 1 - score  # for the purpose of minimization of the score


def compare_labels_for_blanchard_result(file_name):
    labels_pred = np.loadtxt(fname="datasets/blanchard_kmeans_labels_{}.txt".format(file_name))
    labels_true = download_labels(file_name)
    # 0.0 for random labeling and samples and exactly 1.0 when the clusterings are identical
    score = adjusted_rand_score(labels_true, labels_pred)
    print(score)


def calculate_centers_by_labels(X, labels):
    res = np.concatenate((X[labels == 0, :].mean(axis=0)[np.newaxis], X[labels == 1, :].mean(axis=0)[np.newaxis]), axis=0)
    res = np.concatenate((res, X[labels == 2, :].mean(axis=0)[np.newaxis]), axis=0)
    return res


def algorithm_params_to_print(params):
    return 'alpha1={}|alpha2={}|beta1={}|beta1={}'.format(round(params['alpha1'], 2), round(params['alpha2'], 2),
                                                          round(params['beta1'], 2), round(params['beta2'], 2))


def print_score(score):
    print('Score is {}% match between expected labels clusters and kmeans clusters'.format(round((1 - score)*100, 2)))


def assert_isotropic_model(X):
    assert (np.allclose(np.mean(X, axis=0), np.zeros(X.shape[1]), rtol=1.e-1,
                        atol=1.e-1))  # each column vector should have mean zero
    cov_X = np.cov(X, rowvar=False, bias=True)
    assert (cov_X.shape[0] == cov_X.shape[1]) and np.allclose(cov_X, np.eye(cov_X.shape[0]), rtol=5.e-1,
                                                              atol=5.e-1)  # covariance matrix should by identity


def all_zeros(arr):
    return np.count_nonzero(arr) == 0


def subspace_distance(u1, u2):
    return LA.norm(np.dot(u1, u1.T) - np.dot(u2, u2.T), ord='fro')


def subspace_distance_by_angle(v1, v2):
    return math.acos(LA.norm(abs(np.dot(v1.T, v2)), ord='fro'))


def blanchard_subspace_distance(u1, u2):
    return LA.norm(np.dot(u1, u1.T) - np.dot(u2.T, u2), ord='fro')


def generate_gaussian_subspace(rows, cols):
    mu, sigma = 0, 1  # mean and standard deviation
    # np.random.seed(1234)
    return np.random.normal(mu, sigma, (rows, cols))


def get_values_list_in_rage(min, max, num_of_samples):
    return np.arange(min, max, (max-min)/num_of_samples)


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
