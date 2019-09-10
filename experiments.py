import numpy as np
from optimize_oil_data_by_kmeans import scoring_by_kmeans as run_on_oil_data_scoring
from run_on_clover_data import scoring as run_on_clover_data_scoring
import utilities


def main():
    # utilities.compare_labels_for_blanchard_result('DataTst')
    run_on_oil_data_scoring()
    return


if __name__ == "__main__":
    main()
