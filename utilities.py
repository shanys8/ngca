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


def subspace_distance(u1, u2):
    return LA.norm(np.dot(u1, u1.T) - np.dot(u2, u2.T), ord='fro')
