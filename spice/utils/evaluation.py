from sklearn import metrics
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

import warnings
warnings.filterwarnings('ignore')


def calculate_acc(ypred, y, return_idx=False):
    """
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.

    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth
    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.

    """
    assert len(y) > 0
    assert len(np.unique(ypred)) == len(np.unique(y))

    s = np.unique(ypred)
    t = np.unique(y)

    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)

    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    #
    # indices = linear_sum_assignment(C)
    # row = indices[:][:, 0]
    # col = indices[:][:, 1]
    row, col = linear_sum_assignment(C)
    # calculating the accuracy according to the optimal assignment
    count = 0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)

    if return_idx:
        return 1.0 * count / len(y), row, col
    else:
        return 1.0 * count / len(y)


def calculate_nmi(predict_labels, true_labels):
    # NMI
    nmi = metrics.normalized_mutual_info_score(true_labels, predict_labels, average_method='geometric')
    return nmi


def calculate_ari(predict_labels, true_labels):
    # ARI
    ari = metrics.adjusted_rand_score(true_labels, predict_labels)
    return ari

