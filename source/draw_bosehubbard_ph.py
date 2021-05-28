import sys
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import argparse
from visual_utils import generate_listcol
import seaborn as sns

def calculate_npent(death_scales):
    sd = np.sum(death_scales)
    npent = 0
    for d in death_scales:
        dr = d/sd
        npent -= dr*np.log(dr)
    npent = npent/np.log(sd)
    return npent

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
    #cycles = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cols = generate_listcol(option=1)

    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size'] = 16
    

    gs = [0.20, 0.28, 0.30, 0.32, 0.4]
    N = len(gs)
    fig, axs = plt.subplots(1, N, figsize=(3*N, 2.8), squeeze=False, sharey=True)
    axs = axs.ravel()
    #ax.set_xlabel(r"Transverse Field " r"$g$", fontsize=24)

    mk = '_'
    lstyle = 'dashed'
    sz=80
    alpha=1.0
    Ls = [30, 40, 50, 60, 70, 80, 200, 300, 400, 500, 600, 700]
    
    for j in range(len(gs)):
        ax = axs[j]
        g = gs[j]
        gidx = int((g - 0.01) / 0.01)
        for i in range(len(Ls)):
            L = Ls[i]
            if L < 100:
                c = cols[0]
            else:
                c = cols[1]

            phfile = '{}_L_{}_ph_dim_{}.txt'.format(basename, L, d)
            phfile = os.path.join(resname, phfile)
            print(phfile)
            if os.path.isfile(phfile):
                arr = np.loadtxt(phfile)
                death_scales, nlist = arr[:, 1], arr[:, 3]
                ids1 = (death_scales != np.inf)
                ids2 = (nlist == gidx)
                ids = ids1 * ids2
                death_scales = death_scales[ids]
                npent = calculate_npent(death_scales)
                print(arr.shape, gidx, len(death_scales), npent)
                # if L == 30:
                #     sns.kdeplot(death_scales, legend=False, shade=True, color=c, ax=ax, label='L=30~80')
                # elif L == 200:
                #     sns.kdeplot(death_scales, legend=False, shade=True, color=c, ax=ax, label='L=200~700')
                # else:
                sns.kdeplot(death_scales, legend=False, shade=True, color=c, ax=ax)

                #sns.displot(death_scales[ids], bins=20, ax=ax)
                
                #ax.plot(glist, npent_list, linestyle=lstyle, label = 'e-{}'.format(L))
                #ax.plot(glist, pnorm_list, linestyle=lstyle, label = 'p-{}'.format(L))
                
                #ax.plot(glist, vals_list, linestyle='solid', marker='o', color=cols[i], alpha=alpha, linewidth=1.0, markersize=8, label='L={}'.format(L))
                #ax.scatter(glist, vals_list, s=sz, alpha=alpha, edgecolor='k', linewidths='1', label = 'L-{}'.format(L))
                #ax.scatter(glist, pnorm_list, s=sz, alpha=alpha, label = 'p-{}'.format(L))
        #ax.set_xlabel('Birth-scale')
        ax.set_ylabel('')
        ax.tick_params(direction='out', length=8)
        ax.set_xlim([0.1, 0.5])
        ax.set_ylim([0, 60])
        ax.set_title('np={:.5f}'.format(npent))

    axs[-1].legend(fontsize=16)
    axs[-1].set_ylabel('Density')
    
    for figtype in ['png', 'pdf', 'svg']:
        fig_ofile = os.path.join(resname, '{}_diagram_d_{}.{}'.format(basename,d, figtype))
        plt.savefig(fig_ofile, bbox_inches='tight', format=figtype)
    #plt.show()
    
