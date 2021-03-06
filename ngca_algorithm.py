import numpy as np
import math
from numpy import linalg as LA
import utilities
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score


# Features as columns
# Samples as rows


def compute_matrix_phi(samples, samples_copy, alpha):
    result = 0
    z_phi_coeficient = (1 / compute_z_phi(samples, alpha))
    for sample in samples:
        result += math.exp((-1) * alpha * math.pow(LA.norm(sample), 2)) * \
                  np.dot(sample[:, np.newaxis], sample[np.newaxis, :])

    return z_phi_coeficient * result


def compute_z_phi(samples, alpha):
    return np.array([(math.exp((-1) * alpha * math.pow(LA.norm(sample), 2))) for sample in samples]).sum()


def compute_matrix_psi(samples, samples_copy, alpha):
    return_val = 0
    i = 0
    while i < samples.shape[0]:
        return_val += math.exp((-1) * alpha * np.dot(samples[i][np.newaxis, :], samples_copy[i][:, np.newaxis])) * \
                      ((np.dot(samples[i][:, np.newaxis], samples_copy[i][np.newaxis, :])) +
                       (np.dot(samples_copy[i][:, np.newaxis], samples[i][np.newaxis, :])))
        i += 1

    return (1 / compute_z_psi(samples, samples_copy, alpha)) * return_val


def compute_z_psi(samples, samples_copy, alpha):
    return_val = 0
    i = 0
    while i < samples.shape[0]:
        return_val += math.exp((-1) * alpha * np.dot(samples[i][np.newaxis, :], samples_copy[i][:, np.newaxis]))
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


def center(X):
    c = np.mean(X, axis=0)
    Xw = X - c
    return Xw, c


def whiten(X):
    X_c, c = center(X)
    U, s, VT = np.linalg.svd(X_c, full_matrices=False)
    W = VT.T * (math.sqrt(X.shape[0]) / s)
    X_w = U * math.sqrt(X.shape[0])
    return X_w, c, W


def assert_all_columns_unit_vectors(matrix):
    i = 0
    while i < matrix.shape[1]:
        assert (is_unit_vector(matrix[:, i][:, np.newaxis]))
        i += 1


def is_unit_vector(vector):
    return math.isclose(LA.norm(vector),  1.0, rel_tol=1e-2)


def run_ngca_algorithm(samples, samples_copy, alpha1, alpha2, beta1, beta2):

    # Whiten samples
    whiten_samples, _, _ = whiten(samples)
    whiten_samples_copy, _, _ = whiten(samples_copy)

    utilities.assert_isotropic_model(whiten_samples)
    utilities.assert_isotropic_model(whiten_samples_copy)

    # Calculate matrices
    matrix_phi = compute_matrix_phi(whiten_samples, whiten_samples_copy, alpha1)

    matrix_psi = compute_matrix_psi(whiten_samples, whiten_samples_copy, alpha2)

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
    result_space = union_subspace(matrix_phi_relevant_eigenvectors, matrix_psi_relevant_eigenvectors)

    whiten_result_space, _, _ = whiten(result_space)  # try whiten the subspace before return due to matlab code

    return whiten_result_space
    # return result_space


def run(samples, samples_copy, params):
    # Implementation of algorithm in the paper
    approx_ng_subspace = run_ngca_algorithm(samples, samples_copy,
                                            params['alpha1'], params['alpha2'], params['beta1'], params['beta2'])
    return approx_ng_subspace


def score_ngca_on_oil_data_by_kmeans(alpha1, alpha2, beta1, beta2):

    # get samples from train data
    train_samples, train_samples_copy = utilities.download_data('DataTrn', separate_data=True)

    # get samples and labels from validation data
    validation_data = utilities.download_data('DataVdn')
    validation_labels = utilities.download_labels('DataVdn')

    # run NGCA on train data
    approx_ng_subspace = run_ngca_algorithm(train_samples, train_samples_copy, alpha1, alpha2, beta1, beta2)

    # Project validation data on the result subspace
    proj_data = np.dot(validation_data, approx_ng_subspace)

    # evaluate data clustering by algorithm
    score = utilities.get_result_score_by_kmeans(proj_data, validation_labels, 3)

    return score


def score_ngca_algorithm_on_clover_data_by_kmeans(alpha1, alpha2, beta1, beta2):

    samples, samples_copy = utilities.download_data('blanchard_clover_shuffled_full', separate_data=True)
    shuffled_data_full = utilities.download_data('blanchard_clover_shuffled_full')
    clover_data = utilities.download('blanchard_clover_data')

    kmeans_clover = KMeans(n_clusters=4, random_state=0).fit(clover_data)  # Get 4 clusters labels
    clover_kmeans_labels = kmeans_clover.labels_

    # run NGCA on shuffled data
    approx_ng_subspace = run_ngca_algorithm(samples, samples_copy, alpha1, alpha2, beta1, beta2)

    # Project shuffled_data on the result subspace
    proj_data = np.dot(shuffled_data_full, approx_ng_subspace)

    # evaluate result data by KMEANS
    kmeans_clover = KMeans(n_clusters=4, random_state=0).fit(proj_data)  # Get 4 clusters labels
    predicted_result_labels = kmeans_clover.labels_

    score = utilities.score_labels(clover_kmeans_labels, predicted_result_labels)

    return score


def score_ngca_on_oil_data_by_svm(alpha1, alpha2, beta1, beta2):

    # get samples and labels from train and validation data
    train_data = utilities.download_data('DataTrn')
    train_labels = utilities.download_labels('DataTrn')
    validation_data = utilities.download_data('DataVdn')
    validation_labels = utilities.download_labels('DataVdn')

    # Run algorithm on samples from train data
    train_samples, train_samples_copy = utilities.download_data('DataTrn', separate_data=True)
    approx_train_ng_subspace = run_ngca_algorithm(train_samples, train_samples_copy, alpha1, alpha2, beta1, beta2)
    print('reduced 12 dimensions to {} dimensions'.format(approx_train_ng_subspace.shape[1]))

    # in case subspace is not of dimension between 3 and 6 return the worst score - invalid dimensions
    if approx_train_ng_subspace.shape[1] < 3 or approx_train_ng_subspace.shape[1] > 6:
        print('Train NG subspace dimension should be between 3 and 6')
        return 1

    # Project train data on the result subspace to extract the NG components
    proj_train_data = np.dot(train_data, approx_train_ng_subspace)

    # Run algorithm on samples from validation data
    validation_samples, validation_samples_copy = utilities.download_data('DataVdn', separate_data=True)
    approx_validation_ng_subspace = run_ngca_algorithm(validation_samples, validation_samples_copy, alpha1, alpha2, beta1, beta2)
    print('reduced 12 dimensions to {} dimensions'.format(approx_validation_ng_subspace.shape[1]))

    # in case subspace is not of dimension between 3 and 6 return the worst score - invalid dimensions
    if approx_validation_ng_subspace.shape[1] < 3 or approx_validation_ng_subspace.shape[1] > 6:
        print('Validation NG subspace dimension should be between 3 and 6')
        return 1

    if approx_train_ng_subspace.shape[1] != approx_validation_ng_subspace.shape[1]:
        print('Validation and Train NG subspace dimensions are different')
        return 1

    # Project validation data on the result subspace to extract the NG components
    proj_validation_data = np.dot(validation_data, approx_validation_ng_subspace)

    # build SVM classifier - fit by train data and check prediction of validation data
    # clf = SVC(gamma='auto')
    clf = SVC(kernel='rbf', C=500, gamma=0.1)
    clf.fit(proj_train_data, train_labels)

    # assign score
    score = clf.score(proj_validation_data, validation_labels)  # score by SVM model
    train_score = clf.score(proj_train_data, train_labels)  # score by SVM model
    print('train score: {}'.format(train_score))
    return 1 - score  # we want to minimize score

    # predicted_validation_labels = clf.predict(proj_validation_data)
    # score = utilities.score_labels(validation_labels, predicted_validation_labels)  # we want to minimize score


def score_ngca_on_clover_data_by_svm(alpha1, alpha2, beta1, beta2):

    # get samples and labels from train and validation data
    train_shuffled_data = utilities.download_data('cloverDataShuffledTrn')
    train_data = utilities.download_data('cloverDataTrn')
    kmeans_train_data = KMeans(n_clusters=4, random_state=0).fit(train_data)  # Get 4 clusters labels
    train_labels = kmeans_train_data.labels_

    # validation_shuffled_data = utilities.download_data('cloverDataShuffledVdn')
    # validation_data = utilities.download_data('cloverDataVdn')
    # kmeans_validation_data = KMeans(n_clusters=4, random_state=0).fit(validation_data)  # Get 4 clusters labels
    # validation_labels = kmeans_validation_data.labels_

    # Run algorithm on samples from train data
    train_samples, train_samples_copy = utilities.download_data('cloverDataShuffledTrn', separate_data=True)
    approx_ng_subspace = run_ngca_algorithm(train_samples, train_samples_copy, alpha1, alpha2, beta1, beta2)

    # in case subspace is not of dimension 2 then return the worst score
    if approx_ng_subspace.shape[1] != 2:
        print('subspace dimension is not 2')
        return 1

    # Project train data on the result subspace
    proj_train_shuffled_data = np.dot(train_shuffled_data, approx_ng_subspace)

    # build SVM classifier - fit by train data and check predication of validation data
    clf = SVC(kernel='rbf', C=500, gamma=0.1)
    clf.fit(proj_train_shuffled_data, train_labels)
    predicted_train_labels = clf.predict(proj_train_shuffled_data)

    # assign score
    score = clf.score(proj_train_shuffled_data, train_labels)  # score by SVM model
    # score = adjusted_rand_score(train_labels, predicted_train_labels)
    return 1 - score  # we want to minimize score

