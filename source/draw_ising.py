import sys
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import argparse

if __name__ == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--basename', type=str, default='exp_20200217_ising')
    parser.add_argument('--res', type=str, required=True)
    parser.add_argument('--dim', type=int, default=0)
    parser.add_argument('--stats', type=int, default=0)
    parser.add_argument('--figtype', type=str, default='pdf')
    args = parser.parse_args()
    print(args)
    resname, basename, d = args.res, args.basename, args.dim
    typestats, figtype = args.stats, args.figtype

    plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')

    fig, ax = plt.subplots()
    ax.set_xlabel(r"Transverse Field " r"$g$", fontsize=16)
    if typestats == 0:
        ax.set_ylabel(r"Persistent Measures (entropy)", fontsize=16)
    else:
        ax.set_ylabel(r"Persistent Measures (pnorm)", fontsize=16)

    def minmax_norm(a):
        return (a - np.min(a))/(np.max(a) - np.min(a))

    mk = '_'
    lstyle = 'dashed'
    sz=60
    alpha=0.7

    for L in [32, 64, 128, 256, 512, 1024]:
        statsfile = '{}_L_{}_stats_dim_{}.txt'.format(basename, L, d)
        statsfile = os.path.join(resname, statsfile)
        print(statsfile)
        if os.path.isfile(statsfile):
            arr = np.loadtxt(statsfile)
            print(arr.shape)
            glist, npent_list, pnorm_list = arr[:, 0], arr[:, 1], arr[:, 3]
            
            # normalize
            if typestats == 0:
                vals_list = minmax_norm(npent_list)
            else:
                vals_list = minmax_norm(pnorm_list)

            #ax.plot(glist, npent_list, linestyle=lstyle, label = 'e-{}'.format(L))
            #ax.plot(glist, pnorm_list, linestyle=lstyle, label = 'p-{}'.format(L))
            
            ax.scatter(glist, vals_list, s=sz, alpha=alpha, edgecolor='k', linewidths='1', label = 'L-{}'.format(L))
            #ax.scatter(glist, pnorm_list, s=sz, alpha=alpha, label = 'p-{}'.format(L))
    ax.legend(fontsize=12)
    fig_ofile = os.path.join(resname, '{}_agg_{}_fig_dim_{}.{}'.format(basename, typestats, d, figtype))
    plt.savefig(fig_ofile, bbox_inches='tight', format=figtype)
    
