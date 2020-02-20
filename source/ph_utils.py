import numpy as np
import os
import re
from ripser import ripser
from visual_utils import plot_diagrams_with_mat
from collections import defaultdict


def compute_ph_unit(dmat, max_dim=1):
    dgms = ripser(dmat, maxdim=1, distance_matrix=True)['dgms']
    return dgms

def compute_ph_distlist(dmat_list, max_dim=1):
    dgms_list = []
    for dmat in dmat_list:
        dgms = ripser(dmat, maxdim=1, distance_matrix=True)['dgms']
        dgms_list.append(dgms)
    return dgms_list

def compute_ph(out_path, fig_path, basename, dmat_list, plot, pdim):
    if os.path.isdir(out_path) == False:
        os.mkdir(out_path)
    if os.path.isdir(fig_path) == False:
        os.mkdir(fig_path)

    N = len(dmat_list)
    outfile = os.path.join(out_path, basename)
    outstr = defaultdict(list)
    npent_list, pent_list, pnorm_list, maxnorm_list = [], [], [], []
    for i in range(N):
        dmat = dmat_list[i]
        dgms = ripser(dmat, maxdim=1, distance_matrix = True)['dgms']
        for j, dgm in enumerate(dgms):
            for pt in dgm:
                outstr[j].append('{} {} {} {}'.format(pt[0], pt[1], 1, i))
        npent, pent, pnorm, maxnorm = measure_diagrams(dgms, dim=pdim, p=2)
        print(npent, pent, pnorm, maxnorm)
        npent_list.append(npent)
        pent_list.append(pent)
        pnorm_list.append(pnorm)
        maxnorm_list.append(maxnorm)

        if plot:
            # plot diagrams
            for j in [0, 1]:
                if len(dgms[j]) > 0:
                    tstr = '{}_i={}_dim={}'.format(basename, i, j)
                    print(tstr)
                    plot_diagrams_with_mat(fig_path, tstr, dmat, dgms[j])
    # Save diagrams to file txt
    for k, ostr in outstr.items():
        with open('{}_dim_{}.txt'.format(outfile, k), 'w') as file_hdl:
            file_hdl.write('\n'.join(ostr))
    return npent_list, pent_list, pnorm_list, maxnorm_list

def measure_diagram(dgm, p=2):
    npent, pent, pnorm, maxnorm = 0, 0, 0, 0
    arr = np.array(dgm)
    if arr.shape[0] > 0 and arr.shape[1] > 1:
        tmp = arr[:,1] - arr[:,0]
        tmp = tmp[np.isfinite(tmp)]
        if tmp.size > 0:
            tmp = abs(tmp)
            # Calculate p-norm
            maxnorm = np.max(tmp)
            pnorm = (tmp**p).sum() ** (1/p)
            # Calculate p-entropy
            stmp = np.sum(tmp)
            tmp = tmp/stmp
            for a in tmp:
                if (a > 0):
                    pent += -a*np.log(a)
            if (stmp != 1):
                npent = pent / np.log(stmp)

    return npent, pent, pnorm, maxnorm

def measure_diagrams(dgms, dim, p=2):
    return measure_diagram(dgms[dim], p)
