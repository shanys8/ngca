import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from ngca_algorithm import run as run_ngca_algorithm
from ngca_algorithm import score_ngca_on_oil_data_by_svm as score_ngca_on_oil_data_by_svm
from mpl_toolkits.mplot3d import Axes3D
import constant
import nevergrad as ng
from concurrent import futures
import utilities
from sklearn.svm import SVC


def evaluate_test_data_by_svm(algorithm_params):

    # get samples from test data
    test_samples, test_samples_copy = utilities.download_data('DataTst', separate_data=True)
    train_samples, train_samples_copy = utilities.download_data('DataTrn', separate_data=True)

    # get samples and labels from train and test data
    train_data = utilities.download_data('DataTrn')
    train_labels = utilities.download_labels('DataTrn')
    test_data = utilities.download_data('DataTst')
    test_labels = utilities.download_labels('DataTst')

    # Run algorithm on samples from test data
    approx_train_ng_subspace = run_ngca_algorithm(train_samples, train_samples_copy, algorithm_params)
    approx_test_ng_subspace = run_ngca_algorithm(test_samples, test_samples_copy, algorithm_params)

    # Project train data on the result subspace
    proj_train_data = np.dot(train_data, approx_train_ng_subspace)
    proj_test_data = np.dot(test_data, approx_test_ng_subspace)

    # build SVM classifier - fit by train data and check predication (score) of test data
    # clf = SVC(gamma='auto')
    clf = SVC(kernel='rbf', C=500, gamma=0.1)
    clf.fit(proj_train_data, train_labels)

    # assign score
    score = clf.score(proj_test_data, test_labels)
    print('Score on test data:')
    utilities.print_score(score)

    plot_2d_data(proj_test_data, test_labels, algorithm_params)
    plot_3d_data(proj_test_data, test_labels, algorithm_params)

    return


def plot_2d_data(proj_data, labels, params=None):
    # plot first two dimensions of data
    plt.scatter(proj_data[:, 0], proj_data[:, 1], c=labels, cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_3_COLORS))
    plt.savefig('results/oil_data/params_optimization/svm_2D_{}.png'.format(utilities.algorithm_params_to_print(params)))


def plot_3d_data(proj_data, labels, params=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot first three dimensions of data
    ax.scatter(proj_data[:, 0], proj_data[:, 1], proj_data[:, 2], c=labels,
               cmap=matplotlib.colors.ListedColormap(constant.CLUSTERS_3_COLORS))
    plt.savefig('results/oil_data/params_optimization/svm_3D_{}.png'.format(utilities.algorithm_params_to_print(params)))


def main():

    # Optimize params on test and validation datasets
    instrum = ng.Instrumentation(alpha1=ng.var.Array(1).asscalar(),
                                 alpha2=ng.var.Array(1).asscalar(),
                                 beta1=ng.var.Array(1).bounded(a_min=0, a_max=constant.MAX_BETA_VALUE),
                                 beta2=ng.var.Array(1).bounded(a_min=0, a_max=constant.MAX_BETA_VALUE))
    optimizer = ng.optimizers.DiagonalCMA(instrumentation=instrum, budget=100)

    # ask and tell
    for i in range(optimizer.budget):
        try:
            x = optimizer.ask()
            value = score_ngca_on_oil_data_by_svm(*x.args, **x.kwargs)
            print('{} out of {} - value {}'.format(i, optimizer.budget, value))
            optimizer.tell(x, value)
        except:
            print('{} out of {} - error'.format(i, optimizer.budget))

    recommendation = optimizer.provide_recommendation()

    print('Optimal params:')
    print(recommendation.kwargs)

    print('Optimal Score on train and validation data:')
    # score result
    score = score_ngca_on_oil_data_by_svm(recommendation.kwargs['alpha1'],
                                                recommendation.kwargs['alpha2'],
                                                recommendation.kwargs['beta1'],
                                                recommendation.kwargs['beta2'])
    utilities.print_score_fixed(score)

    # Run algorithm with optimal params on test data and evaluate score
    # Plot projected test data on the result NG subspace
    evaluate_test_data_by_svm(recommendation.kwargs)

    return


if __name__ == "__main__":
    main()
