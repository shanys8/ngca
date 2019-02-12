import numpy as np
import pandas as pd
import math
from numpy import linalg as LA
from scipy import linalg
from scipy import stats
# from scipy.cluster.vq import whiten
import seaborn as sns

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel


def plotDataAndCov(data):
    ACov = np.cov(data, rowvar=False, bias=True)
    print('Covariance matrix:\n', ACov)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 10)
    # Choosing the colors
    cmap = sns.color_palette("GnBu", 10)
    sns.heatmap(ACov, cmap=cmap, vmin=0)
    ax1 = plt.subplot(2, 2, 2)
    # data can include the colors
    if data.shape[1] == 3:
        c = data[:, 2]
    else:
        c = "#0A98BE"
    ax1.scatter(data[:, 0], data[:, 1], c=c, s=40)
    # Remove the top and right axes from the data plot
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

def svd_whiten(X):

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white


def whiten(X, fudge=1E-18):

   # the matrix X should be observations-by-components

   # get the covariance matrix
   Xcov = np.dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d, V = np.linalg.eigh(Xcov)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = np.diag(1. / np.sqrt(d+fudge))

   # whitening matrix
   W = np.dot(np.dot(V, D), V.T)

   # multiply by the whitening matrix
   X_white = np.dot(X, W)

   return X_white


def standardize(X):
    newX = center(X)/np.std(X, axis = 0)
    return newX


def decorrelate(X):
    newX = center(X)
    cov = X.T.dot(X)/float(X.shape[0])
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigVals, eigVecs = np.linalg.eig(cov)
    # Apply the eigenvectors to X
    decorrelated = X.dot(eigVecs)
    return decorrelated