import numpy as np
import math
from numpy import linalg as LA
import utilities


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


def assert_isotropic_model(X):
    assert (np.allclose(np.mean(X, axis=0), np.zeros(X.shape[1]), rtol=1.e-1,
                        atol=1.e-1))  # each column vector should have mean zero
    cov_X = np.cov(X, rowvar=False, bias=True)
    assert (cov_X.shape[0] == cov_X.shape[1]) and np.allclose(cov_X, np.eye(cov_X.shape[0]), rtol=1.e-1,
                                                              atol=1.e-1)  # covariance matrix should by identity


def assert_all_columns_unit_vectors(matrix):
    i = 0
    while i < matrix.shape[1]:
        assert (is_unit_vector(matrix[:, i][:, np.newaxis]))
        i += 1


def is_unit_vector(vector):
    return math.isclose(LA.norm(vector),  1.0, rel_tol=1e-2)


def run_ngca_algorithm(samples, samples_copy, alpha1, alpha2, beta1, beta2):

    # Whiten samples
    whiten_samples, c_samples, _ = whiten(samples)
    whiten_samples_copy, c_samples_copy, _ = whiten(samples_copy)

    assert_isotropic_model(whiten_samples)
    assert_isotropic_model(whiten_samples_copy)

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

    assert_all_columns_unit_vectors(result_space)

    return result_space


def run(samples, samples_copy, params):
    # Implementation of algorithm in the paper
    approx_ng_subspace = run_ngca_algorithm(samples, samples_copy,
                                            params['alpha1'], params['alpha2'], params['beta1'], params['beta2'])
    return approx_ng_subspace


def score_ngca_algorithm_on_oil_dataset(alpha1, alpha2, beta1, beta2):

    # get samples from train data
    train_samples, train_samples_copy = utilities.download_data('DataTrn', separate_data=True)

    # get samples and labels from validation data
    validation_data = utilities.download_data('DataVdn')
    validation_labels = utilities.download_labels('DataVdn')

    # Whiten samples
    whiten_samples, c_samples, _ = whiten(train_samples)
    whiten_samples_copy, c_samples_copy, _ = whiten(train_samples_copy)

    assert_isotropic_model(whiten_samples)
    assert_isotropic_model(whiten_samples_copy)

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

    assert_all_columns_unit_vectors(result_space)

    approx_ng_subspace = result_space

    # Project validation data on the result subspace
    proj_data = np.dot(validation_data, approx_ng_subspace)

    # evaluate data clustering by algorithm
    score = utilities.get_result_score(proj_data, validation_labels)
    return score
