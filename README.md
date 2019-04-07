# NGCA PROJECT

NGCA project includes
- matlab code - which I ran on the RDP
- Python code - algorithms implementations + running on different data
- Oil Flow dataset - real dataset taken for experiments
- Tests results - most relevant results are presented in the project report

# Algorithms Implementations
  - ngca_theoretical.py - implementation of the main paper Reweighted PCA algorithm
  - ngca_blanchard.py - implementation of the comparison paper (blanchard's algorithm)

### Experiments
  - algorithm_tuning.py - try grid of params for the theoretical algorithm implementation and pick the ones which gives the smallest error
  - compare_synthetic_data_errors.py - run both algorithms (use blanchard's my python implementation) on same set of samples and compare errors
  - ngca_theoretical_oil_dataset.py - downloads the oil dataset and run theoretical algorithm on it and plot results on 2 dimenensional space

Code and implementations are based on two papers for the purpose of the project experiments

> Polynomial Time and Sample Complexity for
> Non-Gaussian Component Analysis: Spectral Methods

> In Search of Non-Gaussian Components of a
> High-Dimensional Distribution

### Running Instructions

Project requires Python 3.6.8 to run.


```sh
$ cd ngca
$ python <file_to_run>.py
```


License
----

MIT


