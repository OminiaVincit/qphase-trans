import numpy as np
import scipy.stats as stats
import networkx as nx
import distance_inner as dm

def checkdist(dist):
    N = dist.shape[0]
    for i in range(N):
        for j in range(i):
            for k in range(j):
                a, b, c = dist[i, j], dist[j, k], dist[k, i]
                if (a + b < c or a + c < b or b + c < a):
                    return False
    return True

def compute_distance(rholist, tlabel):
    """
    Create distrance matrix from trace distance
    """
    N = len(rholist)
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            ri, rj = rholist[i], rholist[j]
            tmp = 0
            if (tlabel == 'trace'):
                tmp = dm.trace_distance(ri, rj)
            elif tlabel == 'bures':
                tmp = dm.bures_distance(ri, rj)
            elif tlabel == 'angle':
                tmp = dm.bures_angle(ri, rj)
            else:
                print('Not found type of ditance {}'.format(tlabel))
                return dist
            dist[i, j] = tmp
    for i in range(N):
        for j in range(i):
            dist[i, j] = dist[j, i]
    
    # Check distance condition
    print('Check dist', checkdist(dist))

    return dist

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
