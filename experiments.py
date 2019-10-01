import numpy as np
from optimize_oil_data_by_kmeans import scoring_by_kmeans as run_on_oil_data_scoring
from ngca_algorithm import score_ngca_on_clover_data_by_svm
import utilities


def main():
    score_ngca_on_clover_data_by_svm(0.1, 0.1, 0.1, 0.1)
    # utilities.score_initial_data_by_svm(under_pca=True)
    # utilities.compare_labels_for_blanchard_result('DataTst')
    # run_on_oil_data_scoring()
    return


if __name__ == "__main__":
    main()
