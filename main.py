import numpy as np
import math
from numpy import linalg as LA


def print_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def generate_samples(X, N):
    for x in range(N):
        print(x)
    return X


def generate_copy_samples(X, N):
    for x in range(N):
        print(x)
    return X


def compute_matrix_phi(samples, copy_samples, alpha):
    return (1 / compute_z_phi(samples, alpha)) * \
           (np.array([(math.exp((-1) * alpha * math.pow(LA.norm(sample), 2)) * np.dot(sample, sample)) for sample in samples]).sum())


def compute_z_phi(samples, alpha):
    return np.array([(math.exp((-1) * alpha * math.pow(LA.norm(sample), 2))) for sample in samples]).sum()


def compute_matrix_psi(samples, copy_samples, alpha):
    return X * X


def compute_z_psi(samples, samples_copy, alpha):
    samples_tuple_array = np.array((samples, samples_copy)).T
    return 2 * np.array([(math.exp((-1) * alpha * np.dot(tuple[0], tuple[1]))) for tuple in samples_tuple_array]).sum()


def get_eigenvalues(matrix):
    return []


def get_matrix_relevant_eigenvalues(matrix_eigenvalues, gaussian_eigenvalue, threshold):
    return np.where(math.fabs(matrix_eigenvalues - gaussian_eigenvalue) > threshold)


def calculate_gaussian_phi_eigenvalue(alpha):
    return math.pow(2 * alpha + 1, -1)


def calculate_gaussian_psi_eigenvalue(alpha):
    return alpha * math.pow(alpha * alpha - 1, -1)


def get_matrix_corresponding_eigenvectors(matrix, relevant_eigenvalues):
    return []


def calculate_approximate_non_gaussian_space(e1, e2):
    intersection =  np.intersect1d(e1, e2)
    return np.where(LA.norm(intersection) == 1)


def main():


    # # input
    n = 3 # dimension
    N = 8 # number of samples
    X = np.zeros((n, 1)) # random input vector

    alpha1 = 0.3
    alpha2 = 0.4
    beta1 = 0.5
    beta2 = 0.6

    samples = generate_samples(X, N)
    copy_samples = generate_copy_samples(X, N)

    # Calculate matrices
    matrix_phi = compute_matrix_phi(samples, copy_samples, alpha1)
    matrix_psi = compute_matrix_psi(samples, copy_samples, alpha2)

    # Calculate eigenvalues for each matrix
    matrix_phi_eigenvalues = get_eigenvalues(matrix_phi)
    matrix_psi_eigenvalues = get_eigenvalues(matrix_psi)

    # Calculate the gaussian eigenvalue for each matrix
    gaussian_phi_eigenvalue = calculate_gaussian_phi_eigenvalue(alpha1)
    gaussian_psi_eigenvalue = calculate_gaussian_psi_eigenvalue(alpha2)

    # Calculate relevant eigenvalues of each matrix - those which are far away beta from the gaussian eigenvalue
    matrix_phi_relevant_eigenvalues = get_matrix_relevant_eigenvalues(matrix_phi_eigenvalues,
                                                                      gaussian_phi_eigenvalue, beta1)
    matrix_psi_relevant_eigenvalues = get_matrix_relevant_eigenvalues(matrix_psi_eigenvalues,
                                                                      gaussian_psi_eigenvalue, beta2)

    # Calculate corresponding eigenvectors for the relevant eigenvalues
    matrix_phi_relevant_eigenvectors = get_matrix_corresponding_eigenvectors(matrix_phi,
                                                                             matrix_phi_relevant_eigenvalues)
    matrix_psi_relevant_eigenvectors = get_matrix_corresponding_eigenvectors(matrix_psi,
                                                                             matrix_psi_relevant_eigenvalues)

    # Calculate E space - non gaussian space
    approximate_non_gaussian_space = calculate_approximate_non_gaussian_space(matrix_phi_relevant_eigenvectors,
                                                                   matrix_psi_relevant_eigenvectors)

    return approximate_non_gaussian_space



if __name__ == "__main__":
    main()
