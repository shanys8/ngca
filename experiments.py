import numpy as np
from optimize_oil_data_by_kmeans import scoring_by_kmeans as run_on_oil_data_scoring
from optimize_oil_data_by_svm import evaluate_test_data_by_svm_on_blanchard as run_blanchard_scoring_by_svm
from run_on_clover_data import scoring as run_on_clover_data_scoring
import utilities


def main():
    utilities.compare_labels_for_blanchard_result('DataTst')
    # run_on_oil_data_scoring()
    # run_blanchard_scoring_by_svm()
    return


if __name__ == "__main__":
    main()
