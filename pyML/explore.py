"""
explore.py

"""

import sys
import numpy as np
import scipy

import numpy as np
import scipy
import pandas as pd

def standardize(X, ax=0):
    """
    :parameter:
        X (ndarray or df):  Array of values

    :return:
        zscores:            Array of z-scores
        pvalues:            Array of two-tailed p-values
    """

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    zscores = scipy.stats.zscore(X, axis=ax, ddof=1, nan_policy='omit')
    pvalues = scipy.stats.norm.ppf((1 + zscores) / 2)

    return zscores, pvalues

if __name__ == "__main__":
    print("1. Standardize the data")
    zscores, pvalues = standardize(X, ax=0)

    print('\n' + "Z-scores")
    print(zscores)

    print('\n' + "Two-tailed p-values from Z-scores")
    print(pvalues)
