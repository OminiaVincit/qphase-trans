import numpy as np
import scipy.stats as stats
import networkx as nx

def shortest(matrix, inv=True):
    """
    Create distance matrix fro weighted matrix
    """
    sz = matrix.shape[0]
    matrix = np.abs(matrix)
    a = np.max(matrix)
    if a > 0:
       matrix = 1.0 - matrix / a
    G = nx.from_numpy_matrix(matrix)
    dist = nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(G)
    print(dist.shape)
    return np.array(dist)
    #return matrix

def pearson(matrix):
    """
    Calculate the Pearsons correlation coefficient
    and the 2-tailed p-value (see scipy.stats.pearsonr)
    Argument: 2d numpy array (e.g. mutual information matrix)
    Return: a tuple with matrix containing Pearson correlation
    coefficients and a second matrix with the 2-tailed p-values
    """
    ll = matrix.shape[0]
    pearsonR = np.zeros((ll, ll))
    pvalues = np.zeros((ll, ll))

    # Pearson coefficients should be symmetric
    for ii in range(ll):
        for jj in range(ii+1, ll):
            r, p = stats.pearsonr(matrix[ii, :], matrix[jj, :])
            pearsonR[ii][jj] = r
            pearsonR[jj][ii] = r
            pvalues[ii][jj] = p
            pvalues[jj][ii] = p

    return (pearsonR, pvalues)
