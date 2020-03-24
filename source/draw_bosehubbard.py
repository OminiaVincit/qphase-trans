import sys
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import argparse
from visual_utils import generate_listcol, generate_cmap

if __name__ == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--basename', type=str, default='exp_20200220_bosehubbard')
    parser.add_argument('--res', type=str, default='results')
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
    Ls = dict()
    Ls['high'] = [300, 400, 500, 600, 700]
    Ls['mid']  = [30, 40, 50, 60, 70]
    Ls['low']  = [12, 14, 16, 18, 20]
    Ls['all']  = [30, 40, 50, 60, 70, 80, 200, 300, 400, 500, 600, 700]
    Ls['all2'] = [12, 14, 16, 18, 20, 40, 50, 60, 70, 80, 300, 400, 500, 600, 700]
    for lb in ['all', 'all2']:
        fig, ax = plt.subplots()
        ax.set_xlabel(r"Tunneling " r"$J/U$", fontsize=28)
        ax.set_ylabel(labels[typestats], fontsize=28)
        
        for i in range(len(Ls[lb])):
            L = Ls[lb][i]
            if lb == 'all':
                if L < 100:
                    c = cols[0]
                else:
                    c = cols[1]
            else:
                c = cols[int(i/5)]
            statsfile = '{}_L_{}_stats_dim_{}.txt'.format(basename, L, d)
            statsfile = os.path.join(resname, statsfile)
            print(statsfile)
            if os.path.isfile(statsfile):
                arr = np.loadtxt(statsfile)
                print(arr.shape)
                glist, npent_list, pnorm_list = arr[:, 0], arr[:, 1], arr[:, 3]
            
                npent_list = minmax_norm(npent_list)
                pnorm_list = minmax_norm(pnorm_list)
            
                # normalize
                if typestats == 0:
                    vals_list = npent_list
                elif typestats == 1:
                    vals_list = pnorm_list
                else:
                    vals_list = abs(npent_list - pnorm_list)

                ax.plot(glist, vals_list, linestyle=lstyle, markersize=8, color=c, alpha=alpha, linewidth=4.0, label = 'L-{}'.format(L))
            
                #ax.scatter(glist, vals_list, s=sz, cmap=cm, alpha=alpha, edgecolor='k', linewidths='1', label = 'L-{}'.format(L))
                ax.tick_params(direction='in', length=8)
                #ax.legend(fontsize=18)
        for figtype in ['png', 'pdf', 'svg']:
            fig_ofile = os.path.join(resname, '{}_{}_agg_{}_fig_dim_{}.{}'.format(basename, lb, typestats, d, figtype))
            plt.savefig(fig_ofile, bbox_inches='tight', format=figtype)
    
