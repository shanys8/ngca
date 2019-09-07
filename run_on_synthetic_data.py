from ngca_algorithm import run as run_ngca_algorithm
import numpy as np
import utilities
import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
    elif type_of_requested_subspace == 'sub_gaussian':
        U_X_initial = 2 * np.random.rand(k, m) - 1
        X_to_sample = np.dot(Q_E, U_X_initial)
    # elif type_of_requested_subspace == 'super_gaussian':
    #     # TODO implement
    #     X_to_sample = []
    else:
        raise ValueError('unsupported type of requested subspace', type_of_requested_subspace)

    # TODO  - check on which type of vector do we project G_to_sample and on which do we project X_to_sample
    # samples generated as direct sum of range(G_to_sample) and range(X_to_sample)
    for _ in range(number_of_samples):
        sample = np.dot(G_to_sample, np.random.randn(m, 1)) + np.dot(X_to_sample, (np.random.randn(m, 1)))
        samples = np.append(samples, sample, axis=1)

    for _ in range(number_of_samples):
        sample_copy = np.dot(G_to_sample, np.random.randn(m, 1)) + np.dot(X_to_sample, (np.random.randn(m, 1)))
        samples_copy = np.append(samples_copy, sample_copy, axis=1)

    return samples.T, samples_copy.T, Q_E


def plot_2d_data(data, synthetic_subspace, approx_ng_subspace, params, type_of_requested_subspace):


    pca_data = get_PCA_data_for_plot(data)
    # Project samples on the result NG subspace
    proj_data_on_result_subspace = np.dot(data, approx_ng_subspace)
    # Project samples on the known synthetic NG subspace
    proj_data_on_synthetic_subspace = np.dot(data, synthetic_subspace)

    f = plt.figure()
    f, axes = plt.subplots(ncols=3)
    sc = axes[0].scatter(pca_data[:, 0], pca_data[:, 1], c='green', alpha=0.5)
    axes[0].set_xlabel('PCA initial data', labelpad=5)

    axes[1].scatter(proj_data_on_synthetic_subspace[:, 0], proj_data_on_synthetic_subspace[:, 1], c='blue', alpha=0.5)
    axes[1].set_xlabel('Projected data on \nknown NG subspace', labelpad=5)

    axes[2].scatter(proj_data_on_result_subspace[:, 0], proj_data_on_result_subspace[:, 1], c='red', alpha=0.5)
    axes[2].set_xlabel('Projected data on \nresult NG subspace', labelpad=5)

    plt.savefig('results/synthetic_data_2D_{}_{}.png'.format(type_of_requested_subspace, utilities.algorithm_params_to_print(params)))


def get_PCA_data_for_plot(data):
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2, svd_solver='full')
    principalComponents = pca.fit_transform(data)
    return principalComponents


def main():

    m = 5
    n = 10  # dimension - number of features
    k = 3  # NG subspace dimension
    number_of_samples = 2000  # number of samples
    type_of_requested_subspace = 'sub_gaussian'  # sub_gaussian | gaussian | super_gaussian

    algorithm_params = {
        'alpha1': 0.7,
        'alpha2': 0.3,
        'beta1': 0.34,
        'beta2': 0.64,
    }

    samples, samples_copy, Q_E = generate_synthetic_samples(number_of_samples, n, k, m, type_of_requested_subspace)

    all_samples = np.concatenate((samples, samples_copy), axis=0)  # E is the range of Q_E

    approx_ng_subspace = run_ngca_algorithm(samples, samples_copy, algorithm_params)

    plot_2d_data(all_samples, Q_E, approx_ng_subspace, algorithm_params, type_of_requested_subspace)

    return approx_ng_subspace


if __name__ == "__main__":
    main()
