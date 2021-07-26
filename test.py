import numpy as np
import time
import CPM_Cythoned
import scipy.stats as scistats
import scipy.io as sio

d = 268

def np_pearson_cor(x, y):
    """
    Return the pearson corr of two vectors using only numpy
    :param x: 1-d vector
    :param y: 1-d vector
    :return: float
    """
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)[0,0]

def get_r_matrix(FC, target):
    """
    :param FC: n * d * d matrix, d nodes, n subjects
    :param target: vector of continuous outcome
    :return: d * d matrix with entry i,j as the r value across the sample, correlated with the target
    """
    # Initialize with zeros
    R = np.zeros(shape=(d, d))

    # populate the r matrix, only going through lower triangle
    for i in range(d):
        for j in range(i):
            subjects = FC[:, i, j]
            r = np_pearson_cor(subjects, target)
            R[i, j] = r
    return R


def get_p_matrix(FC, target):
    """
    :param FC: n * d * d matrix, d nodes, n subjects
    :param target: vector of continuous outcome
    :return: a d * d matrix with entry i,j as the p value across the sample, correlated with the target
    """
    # Initialize with zeros
    P = np.zeros(shape=(d, d))

    # populate the r matrix, only going through lower triangle
    for i in range(d):
        for j in range(i):
            subjects = FC[:, i, j]
            p = scistats.pearsonr(subjects, target)[1]
            P[i, j] = p
    return P


def mask(threshold, M):
    """
    :param M: matrix to mask
              only lower triangular, row i, column j, i < j
    :param threshold: [l, u) interval of desired edges
    :return: vector of tuples (i,j) entry that should count as a feature
    """
    entries = []
    for i in range(len(M)):
        for j in range(i):
            if threshold[0] <= M[i, j] < threshold[1]: entries.append((i, j))
    return entries



FC = sio.loadmat(r"268_108.mat")['all_mats']
target = sio.loadmat(r"unStdAge.mat")['age']
FC = np.swapaxes(FC, 0, 2)

# Flattening FC to be 2D matrix not 3D
FC_flat = []
for x in FC:
    sbj = []
    for i in range(len(x)):
        for j in range(i):
            sbj.append(x[i,j])
    FC_flat.append(sbj)

FC_flat = np.array(FC_flat, dtype=np.float64)

start = time.time()
# Cython implementation
inds, dim = CPM_Cythoned.get_Xtr(FC_flat, target, 0.00, 0.01, True )
FC_flat = FC_flat[:,inds]
# -----------------
end = time.time()
print(FC_flat.shape)
print("Time elapsed Cython: " + str(end - start))


start = time.time()
# Regular python with masking and list comprehension
R = get_p_matrix(FC, target)
pick_edges = mask((0.00, 0.01), R)
X_tr = []
for x in range(108):  X_tr.append([FC[x, i, j] for i, j in pick_edges])
X_tr = np.array(X_tr)
# -----------------
end = time.time()
print(X_tr.shape)
print("Time elapsed Python List comprehension: " + str(end - start))

EQ = X_tr == FC_flat
print()
print(False in EQ)
