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
