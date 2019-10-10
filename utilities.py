import numpy as np
from numpy import linalg as LA
import math
import matplotlib
import constant
from matplotlib import pyplot as plt
import time
# from scipy import linalg
from scipy.linalg import hadamard
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def download(file_name):
    return np.loadtxt(fname="datasets/{}.txt".format(file_name))


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


def generate_clover_data(n):
    X = np.empty((2, 0), float)
    count = 0

    while count < n:
        d = 2 * np.random.rand(2, n) - 1
        index = np.argwhere(np.sum(np.power(d, 2), axis=0) < np.sqrt(np.abs(d[0, :] * d[1, :])))
        index = list(itertools.chain(*index))
        count = count + np.size(index)
        X = np.append(X, np.take(d, index, axis=1), axis=1)

    result = math.sqrt(10) * X[:, :n]
    return result


def generate_shuffled_data(data):
    scaling = np.power(10, np.arange(-1, 1.2, 0.2))

    shuffled_data = data
    s = 0
    while s < np.size(scaling):
        shuffled_data = scaling[s] * np.append(shuffled_data, np.random.randn(1, np.shape(shuffled_data)[1]), axis=0)
        s = s + 1

    Q, R = np.linalg.qr(np.random.randn(13, 13))

    shuffled_data = np.dot(Q, shuffled_data)
    return shuffled_data.T


def perform_pca(data):
    scaler = StandardScaler()
    scaler.fit(data)
    feature_scaled = scaler.transform(data)

    pca = PCA(n_components=3, svd_solver='full')
    principal_components = pca.fit_transform(feature_scaled)

    return principal_components


def PCA_SVM_optimal(find_best_params=False):

    # get data
    train_data = download_data('DataTrn')
    train_labels = download_labels('DataTrn')
    validation_data = download_data('DataVdn')
    validation_labels = download_labels('DataVdn')
    test_data = download_data('DataTst')
    test_labels = download_labels('DataTst')

    #TODO replace with perform_PCA function
    # perform PCA
    scaler = StandardScaler()
    scaler.fit(test_data)
    test_data_scaled = scaler.transform(test_data)
    pca = PCA(n_components=3)
    test_data_scaled_reduced = pca.fit_transform(test_data_scaled)

    scaler1 = StandardScaler()
    scaler1.fit(train_data)
    train_data_scaled = scaler.transform(train_data)
    pca1 = PCA(n_components=3)
    train_data_scaled_reduced = pca1.fit_transform(train_data_scaled)

    if find_best_params:
        # get optimal params
        pipe_steps = [('scaler', StandardScaler()), ('pca', PCA()), ('SupVM', SVC(kernel='rbf'))]
        pipeline = Pipeline(pipe_steps)
        check_params = {
            'pca__n_components': [3],
            'SupVM__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000],
            'SupVM__gamma': [0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
        }

        create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=5)
        create_grid.fit(train_data, train_labels)
        print("score for 5 fold CV is %3.2f" % (create_grid.score(test_data, test_labels)))
        print('best params')
        print(create_grid.best_params_)


    # build SVM model
    if find_best_params:
        svm_model = SVC(kernel='rbf', C=float(create_grid.best_params_['SupVM__C']), gamma=float(create_grid.best_params_['SupVM__gamma']))
    else:
        svm_model = SVC(kernel='rbf', C=500, gamma=0.1)  #found the optimal once

    # svm_model.fit(test_data_scaled_reduced, test_labels)
    svm_model.fit(train_data_scaled_reduced, train_labels)
    test_score = svm_model.score(test_data_scaled_reduced, test_labels)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot first three dimensions of data
    ax.scatter(test_data_scaled_reduced[:, 0], test_data_scaled_reduced[:, 1], test_data_scaled_reduced[:, 2], c=test_labels,
               cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_3_COLORS))

    plt.show()
    # plt.savefig('results/test_oil_data_after_pca.png')

    print('end')


# score initial data (with or without PCA run) by SVM model
def score_initial_data_by_svm(under_pca=False):

    # get samples and labels from train and validation data
    train_data = download_data('DataTrn')
    train_labels = download_labels('DataTrn')
    validation_data = download_data('DataVdn')
    validation_labels = download_labels('DataVdn')
    test_data = download_data('DataTst')
    test_labels = download_labels('DataTst')

    # build SVM classifier - fit by train data and check predication of validation and test data
    clf = SVC(gamma='auto')
    if under_pca:
        train_data = perform_pca(train_data)
        validation_data = perform_pca(validation_data)
        test_data = perform_pca(test_data)
        clf.fit(train_data, train_labels)
        train_score = clf.score(train_data, train_labels)
        validation_score = clf.score(validation_data, validation_labels)
        test_score = clf.score(test_data, test_labels)
    else:
        clf.fit(train_data, train_labels)
        # predicted_validation_labels = clf.predict(validation_data)
        train_score = clf.score(train_data, train_labels)
        validation_score = clf.score(validation_data, validation_labels)
        test_score = clf.score(test_data, test_labels)

    print('Train score is {}\nValidation score is {}\nTest score is {}'.format(train_score, validation_score, test_score))
    return train_score, validation_score, test_score


def get_result_score_by_kmeans(proj_data, labels_true, components_num):
    kmeans = KMeans(n_clusters=components_num, random_state=0).fit(proj_data)
    labels_pred = kmeans.labels_
    return score_labels(labels_true, labels_pred)


def score_labels(labels_true, labels_pred):
    # 0.0 for random labeling and samples and exactly 1.0 when the clusterings are identical
    score = adjusted_rand_score(labels_true, labels_pred)
    return 1 - score  # for the purpose of minimization of the score


def compare_labels_for_blanchard_result(file_name):
    labels_pred = np.loadtxt(fname="datasets/blanchard_kmeans_labels_{}.txt".format(file_name))
    labels_true = download_labels(file_name)
    # 0.0 for random labeling and samples and exactly 1.0 when the clusterings are identical
    score = adjusted_rand_score(labels_true, labels_pred)
    print_score(score)


def blanchard_scoring_by_svm():

    # get samples and labels from train and validation data
    train_data = download_data('DataTrn')
    train_labels = download_labels('DataTrn')
    validation_data = download_data('DataVdn')
    validation_labels = download_labels('DataVdn')
    test_data = download_data('DataTst')
    test_labels = download_labels('DataTst')

    # Run blanchard algorithm on train data and get ngspace from matlab script
    approx_ng_subspace = np.loadtxt(fname="datasets/blanchard_ngspace.txt")

    # Project train and validation data on the result subspace
    proj_train_data = np.dot(train_data, approx_ng_subspace)
    proj_validation_data = np.dot(validation_data, approx_ng_subspace)
    proj_test_data = np.dot(test_data, approx_ng_subspace)

    # build SVM classifier - fit by train data
    clf = SVC(gamma='auto')
    clf.fit(proj_train_data, train_labels)
    train_score = clf.score(proj_train_data, train_labels)
    validation_score = clf.score(proj_validation_data, validation_labels)
    test_score = clf.score(proj_test_data, test_labels)

    print('Train score is {}\nValidation score is {}\nTest score is {}'.format(train_score, validation_score, test_score))
    return train_score, validation_score, test_score


def calculate_centers_by_labels(X, labels):
    res = np.concatenate((X[labels == 0, :].mean(axis=0)[np.newaxis], X[labels == 1, :].mean(axis=0)[np.newaxis]), axis=0)
    res = np.concatenate((res, X[labels == 2, :].mean(axis=0)[np.newaxis]), axis=0)
    return res


def algorithm_params_to_print(params):
    if params:
        return 'alpha1={}|alpha2={}|beta1={}|beta1={}'.format(round(params['alpha1'], 2), round(params['alpha2'], 2),
                                                          round(params['beta1'][0], 2), round(params['beta2'][0], 2))
    else:
        return 'blanchard'


def print_score_fixed(score):
    print('Score is {}% match between predicted labels and result labels'.format(round((1 - score)*100, 2)))


def print_score(score):
    print('Score is {}% match between predicted labels and result labels'.format(round(score*100, 2)))


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
