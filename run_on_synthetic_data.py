from ngca_algorithm import run as run_ngca_algorithm
import numpy as np


def generate_synthetic_samples(number_of_samples, n, k, m, type_of_requested_subspace):

    samples = np.empty((n, 0), float)
    samples_copy = np.empty((n, 0), float)

    B = np.random.rand(n, n)
    Q, _ = np.linalg.qr(B)
    Q_E = Q[:, 0:k]
    Q_E_orth = Q[:, k:]

    # Gaussian subspace to sample - projection of Q_E_orth
    G_initial = np.random.randn(n-k, m)
    G_to_sample = np.dot(Q_E_orth, G_initial)

    # subspace to sample - projection of Q_E
    if type_of_requested_subspace == 'gaussian':
        G_X_initial = np.random.randn(k, m)
        X_to_sample = np.dot(Q_E, G_X_initial)
    # elif type_of_requested_subspace == 'sub_gaussian':
    #     # TODO check
    #     U_X_initial = np.random.rand(k, m)
    #     X_to_sample = np.dot(Q_E, U_X_initial)
    # elif type_of_requested_subspace == 'super_gaussian':
    #     # TODO implement
    #     X_to_sample = []
    else:
        raise ValueError('unsupported type of requested subspace', type_of_requested_subspace)

    # samples generated as direct sum of range(G_to_sample) and range(X_to_sample)
    for _ in range(number_of_samples):
        # each sample should have mean zero
        # TODO  - check why do we project G_to_sample onto a random noraml vector and we project X_to_sample onto a random vector
        sample = np.dot(G_to_sample, np.random.randn(m, 1)) + np.dot(X_to_sample, (np.random.randn(m, 1)))
        samples = np.append(samples, sample, axis=1)

    for _ in range(number_of_samples):
        # each sample should have mean zero
        sample_copy = np.dot(G_to_sample, np.random.randn(m, 1)) + np.dot(X_to_sample, (np.random.randn(m, 1)))
        samples_copy = np.append(samples_copy, sample_copy, axis=1)

    return samples, samples_copy, Q_E


def main():

    m = 5
    n = 10  # dimension - number of features
    k = 3  # NG subspace dimension
    number_of_samples = 1000  # number of samples
    type_of_requested_subspace = 'gaussian'  # sub_gaussian | gaussian | super_gaussian

    algorithm_params = {
        'alpha1': 0.7,
        'alpha2': 0.3,
        'beta1': 0.34,
        'beta2': 0.64,
    }

    samples, samples_copy, Q_E = generate_synthetic_samples(number_of_samples, n, k, m, type_of_requested_subspace)

    # E is the range of Q_E

    # Implementation of algorithm in the paper
    approx_ng_subspace = run_ngca_algorithm(samples.T, samples_copy.T, algorithm_params)

    # TODO compare Q_E and approx_ng_subspace

    return approx_ng_subspace


if __name__ == "__main__":
    main()
