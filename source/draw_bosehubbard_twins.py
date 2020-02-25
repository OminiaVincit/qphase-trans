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
    parser.add_argument('--basename', type=str, default='exp_20200220_bosehubbard')
    parser.add_argument('--res', type=str, default='results')
    parser.add_argument('--dim', type=int, default=0)
    args = parser.parse_args()
    print(args)
    resname, basename, d = args.res, args.basename, args.dim

    plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


    
    def minmax_norm(a):
        return (a - np.min(a))/(np.max(a) - np.min(a))

    mk = '_'
    lstyle = 'dashed'
    sz=60
    alpha=0.7
    Ls = [10, 20, 30, 40, 50, 60]
    for i in range(len(Ls)):
        c1 = cycle[0]
        c2 = cycle[2]
        L = Ls[i]
        statsfile = '{}_L_{}_stats_dim_{}.txt'.format(basename, L, d)
        statsfile = os.path.join(resname, statsfile)
        print(statsfile)
        if os.path.isfile(statsfile):
            fig, ax1 = plt.subplots()
            ax1.set_xlabel(r"Tunneling " r"$J/U$", fontsize=16)
            ax1.set_ylabel(r"Normalize persistent entropy", fontsize=16)
            ax2 = ax1.twinx()
            ax2.set_ylabel(r"p-norm persistence", fontsize=16)
            
            arr = np.loadtxt(statsfile)
            print(arr.shape)
            glist, npent_list, pnorm_list = arr[:, 0], arr[:, 1], arr[:, 3]
            npent_list = minmax_norm(npent_list)
            pnorm_list = minmax_norm(pnorm_list)

            #ax.plot(glist, npent_list, linestyle=lstyle, label = 'e-{}'.format(L))
            #ax.plot(glist, pnorm_list, linestyle=lstyle, label = 'p-{}'.format(L))
            
            ax1.scatter(glist, npent_list, s=sz, color=c1, alpha=alpha, edgecolor='k', linewidths='1', label = 'ent-L-{}'.format(L))
            ax2.scatter(glist, pnorm_list, s=sz, color=c2, alpha=alpha, edgecolor='k', linewidths='1', label = 'pnorm-L-{}'.format(L))
    

            ax1.legend(bbox_to_anchor=(1, 0.6), loc='lower right', borderaxespad=1, fontsize=16)
            ax2.legend(bbox_to_anchor=(1, 0.4), loc='upper right', borderaxespad=1, fontsize=16)
            for figtype in ['png', 'pdf', 'svg']:
                fig_ofile = os.path.join(resname, '{}_join_L_{}_fig_dim_{}.{}'.format(basename, L, d, figtype))
                fig.savefig(fig_ofile, bbox_inches='tight', format=figtype)
                plt.close(fig) 
