import sys
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import argparse
import scipy.optimize
from visual_utils import generate_listcol, generate_cmap

# function to fit
def func(x, a, b, c):
    return a + b * np.power(x, c)

if __name__ == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--basename', type=str, default='exp_20200220_bosehubbard')
    parser.add_argument('--res', type=str, default='bosehubbard2')
    parser.add_argument('--dim', type=int, default=0)
    parser.add_argument('--stats', type=int, default=1)
    args = parser.parse_args()
    print(args)
    resname, basename, d = args.res, args.basename, args.dim
    typestats = args.stats
    labels = ['Pentropy', 'Pnorm', 'Diff']
        
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size'] = 20
    cols = generate_listcol(option=1)

    def minmax_norm(a):
        return (a - np.min(a))/(np.max(a) - np.min(a))

    mk = 'o'
    lstyle = 'dashed'
    sz=80
    alpha=0.7
    Ls = [12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700]
    
    xs, qls =[],  []
    
    for L in Ls:
        statsfile = '{}_L_{}_stats_dim_{}.txt'.format(basename, L, d)
        statsfile = os.path.join(resname, statsfile)
        print(statsfile)
        if os.path.isfile(statsfile):
            arr = np.loadtxt(statsfile)
            print(arr.shape)
            glist, npent_list, pnorm_list = arr[:, 0], arr[:, 1], arr[:, 3]
            
            npent_list = minmax_norm(npent_list)
            pnorm_list = minmax_norm(pnorm_list)
            
            vals_list = abs(npent_list - pnorm_list)
            idx = np.argmin(vals_list)
            xs.append(L)
            qls.append(glist[idx])
    print(xs)
    print(qls)
    xs, qls = np.array(xs), np.array(qls) 
    iniparams = np.array([0.3, 1.0, -0.5])
    optparams, covariant = scipy.optimize.curve_fit(func, xs, qls, p0 = iniparams)
    print(optparams)
    print(covariant)
    # Plot the fitting curve
    #fig, ax = plt.subplots()
    #ax.set_xlabel(r"Tunneling " r"$J/U$", fontsize=28)
    #ax.set_ylabel(labels[typestats], fontsize=28)
    #ax.plot(glist, vals_list, linestyle=lstyle, markersize=8, color=c, alpha=alpha, linewidth=4.0, label = 'L-{}'.format(L))
            
    #ax.scatter(glist, vals_list, s=sz, cmap=cm, alpha=alpha, edgecolor='k', linewidths='1', label = 'L-{}'.format(L))
    #ax.tick_params(direction='in', length=8)
    #ax.legend(fontsize=18)
        
    #for figtype in ['png', 'pdf', 'svg']:
    #    fig_ofile = os.path.join(resname, '{}_{}_agg_{}_fig_dim_{}.{}'.format(basename, lb, typestats, d, figtype))
    #    plt.savefig(fig_ofile, bbox_inches='tight', format=figtype)
    
