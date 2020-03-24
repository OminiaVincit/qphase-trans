import MPSPyLib as mps
import networkmeasures as nm
import ph_utils as ph
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path
import time
import argparse
from threading import Thread
from multiprocessing import Process
from collections import defaultdict
#import tempfile as tp
import shutil

def simulate(L, ExpName, PostProcess=False, ShowPlot=True):
    """
    PostProcess = False:
        Runs the MPSFortLib to simulate the finite size statics of the BoseHubbard model
        using number conservation
    PostProcess = True:
        Obtain the simulated results from the already run simulation
    """

    # Build operators
    Operators = mps.BuildBoseOperators(6)
    Operators['interaction'] = 0.5 * (np.dot(Operators['nbtotal'], Operators['nbtotal']) - Operators['nbtotal'])
    # Define Hamiltonian of transverse Ising model
    H = mps.MPO(Operators)
    H.AddMPOTerm('bond', ['bdagger', 'b'], hparam='t', weight=-1.0)
    H.AddMPOTerm('site', 'interaction', hparam='U', weight=1.0)
    #H.AddMPOTerm('bond', ['nbtotal', 'nbtotal'], hparam='V', weight=1.0)

    # ground state observables
    myObservables = mps.Observables(Operators)
    # site terms
    myObservables.AddObservable('site', 'nbtotal', name='n')
    myObservables.AddObservable('DensityMatrix_i', [])
    myObservables.AddObservable('DensityMatrix_ij', [])
    myObservables.AddObservable('MI', True)
    myConv = mps.MPSConvParam(max_bond_dimension=200, 
                variance_tol = 1E-8,
                local_tol = 1E-8,
                max_num_sweeps = 7)
    #modparam = ['max_bond_dimension', 'max_num_sweeps']
    #myConv.AddModifiedConvergenceParameters(0, modparam, [80, 4])

    # Specify constants and parameter lists
    U = 1.0
    tlist = np.linspace(0.02, 1.00, 50)
    #tlist = np.linspace(0.01, 0.41, 41)
    #tlist = np.linspace(0.1, 0.4, 5)
    parameters = []
    N = L

    for t in tlist:
        parameters.append({
            'simtype'                   : 'Finite',
            # Directories
            'job_ID'                    : 'Bose_Hubbard_Statics',
            'unique_ID'                 : 't_' + str(t) + 'N_' + str(N),
            'Write_Directory'           : '{}/TMP_BoseHubbard_L_{}/'.format(ExpName, L),
            'Output_Directory'          : '{}/OUTPUTS_BoseHubbard_L_{}/'.format(ExpName, L),
            # System size and Hamiltonian parameters
            'L'                         : L,
            't'                         : t,
            'U'                         : U,
            # Specification of symmetries and good quantum numbers
            'Abelian_generators'        : ['nbtotal'],
            # Define filling
            'Abelian_quantum_numbers'   : [N],
            # Convergence parameters
            'verbose'                   : 1,
            'logfile'                   : True,
            'MPSObservables'            : myObservables,
            'MPSConvergenceParameters'  : myConv
        })

    # Write Fortran-readable main files
    MainFiles = mps.WriteFiles(parameters, Operators, H,
                               PostProcess=PostProcess)

    # Run the simulations and quit if we are not just Post
    if(not PostProcess):
        if os.path.isfile('./Execute_MPSMain'):
            RunDir = './'
        else:
            RunDir = None
        mps.runMPS(MainFiles, RunDir=RunDir)
        return None, tlist

    # PostProcess
    # -----------
    MI_list = []
    Outputs = mps.ReadStaticObservables(parameters)
    M = int(L/2)
    for Output in Outputs:
        print(Output['converged'])
        MI_list.append(Output['MI'][3][M])
    
    timestamp = int(time.time() * 1000.0)
    if ShowPlot:
        plt.style.use('seaborn-colorblind')
        #plt.style.use('seaborn-whitegrid')
        plt.rc('font', family='serif')
        plt.rc('mathtext', fontset='cm')
        plt.scatter(tlist, MI_list)
        plt.xlabel(r"Tunneling " r"$J/U$", fontsize=16)
        plt.ylabel(r"Mutual information (4,L/2+1)", fontsize=16)
        plt.savefig('{}/BoseHubbard_MI_L_{}_{}.pdf'.format(exp_name, L, timestamp), bbox_inches='tight')
        plt.show()
    
    return Outputs, tlist

def persistent_compute(mi_mats, max_dim, bg, time_stamp, basename, tmpdir, process_id):
    # Compute persistent entropy and save to temporary file
    N = len(mi_mats)
    if os.path.isdir(tmpdir) == False:
        os.mkdir(tmpdir)
    outfile = os.path.join(tmpdir, basename) 

    outstr = defaultdict(list)
    stats = defaultdict(list)

    for i in range(N):
        mimat = mi_mats[i]
        idx = bg + i
        #print(process_id, i, bg, idx, mimat[3][26])
        netmeasures = nm.pearson(mimat)
        dist = np.sqrt(1.0 - netmeasures[0]**2)
        for j in range(dist.shape[0]):
            dist[j, j] = 0
         
        dgms = ph.compute_ph_unit(dist, max_dim=max_dim)
        for j, dgm in enumerate(dgms):
            for pt in dgm:
                outstr[j].append('{} {} {} {}'.format(pt[0], pt[1], 1, idx))
            npent, pent, pnorm, maxnorm = ph.measure_diagram(dgm, p=2)
            stats[j].append('{} {} {} {} {}'.format(idx, npent, pent, pnorm, maxnorm))

    # Save diagrams to file txt
    for j, ostr in outstr.items():
        with open('{}_ph_dim_{}_{}.txt'.format(outfile, j, process_id), 'w') as file_hdl:
            file_hdl.write('\n'.join(ostr))
    
    # Save stats to file txt
    for j, ostr in stats.items():
        with open('{}_stats_dim_{}_{}.txt'.format(outfile, j, process_id), 'w') as file_hdl:
            file_hdl.write('\n'.join(ostr))
    print('Finished process {} with bg={}, ed={}, outfile={}'.format(process_id, bg, bg + N, outfile))

if __name__ == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--postprocess', type=int, default=0)
    parser.add_argument('--plot', type=int, default=1)
    parser.add_argument('--persistent', type=int, default=0)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--exp', type=str, default='exp_test')
    parser.add_argument('--odir', type=str, default='results')
    parser.add_argument('--nproc', type=int, default=16)
    parser.add_argument('--maxdim', type=int, default=1)
    parser.add_argument('--base', type=str, default='bosehubbard')
    args = parser.parse_args()
    print(args)
    post_flag, plot_flag = (args.postprocess > 0), (args.plot > 0)
    persis_flag, L = (args.persistent > 0), args.size
    exp_name, out_dir, nproc = args.exp, args.odir, args.nproc
    max_dim = args.maxdim

    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)
    # Run main function
    sim_outs, glist = simulate(L=L, ExpName=exp_name, PostProcess=post_flag, ShowPlot=plot_flag)
    if sim_outs is not None:
        print('Num outputs', len(sim_outs))
    else:
       exit(1) 
    # Multi-processing
    processes = []
    N = len(sim_outs)
    lst = np.array_split(range(0, N), nproc)
    time_stamp = int(time.time() * 1000.0)
    basename = '{}_L_{}'.format(args.base, L)
    tmpdir = os.path.join(args.odir, '{}_{}'.format(exp_name, time_stamp))

    #with tp.TemporaryDirectory() as tmpdir:
    #print(os.path.exists(tmpdir), tmpdir)
    for proc_id in range(nproc):
        outlist = lst[proc_id]
        if outlist.size == 0:
            continue
        print(outlist)
        bg = outlist[0]
        mils = [sim_outs[j]['MI'] for j in outlist]
        p = Process(target=persistent_compute, args=(mils, max_dim, bg, time_stamp, basename, tmpdir, proc_id))
        processes.append(p)
    #print(os.path.exists(tmpdir), tmpdir)

    # Start the processes
    for p in processes:
        p.start()
    
    # Ensure all processes have finished execution
    for p in processes:
        p.join()

    # Sleep 5s
    time.sleep(5)

    plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Concat files and remove tmpdir
    def arr_reshape(arr):
        if arr is not None and arr.ndim == 1:
            return arr.reshape([1, arr.size])
        return arr

    for d in range(max_dim+1):
        phlist, statslist = [], []
        for i in range(nproc):
            filebase = os.path.join(tmpdir, basename)
            phfile = '{}_ph_dim_{}_{}.txt'.format(filebase, d, i)
            statfile = '{}_stats_dim_{}_{}.txt'.format(filebase, d, i)
            if os.path.isfile(phfile):
                arr = np.loadtxt(phfile)
                arr = arr_reshape(arr)
                if arr is not None:
                    phlist.append(arr)
            if os.path.isfile(statfile):
                sarr = np.loadtxt(statfile)
                sarr = arr_reshape(sarr)
                if sarr is not None:
                    statslist.append(sarr)
        
        pharr = np.concatenate(phlist, axis=0)
        statsarr = np.concatenate(statslist, axis=0)
        statsarr[:, 0] = glist
        print(d, pharr.shape, statsarr.shape)
        # Save diagrams and statistic file to txt file
        ph_ofile = os.path.join(out_dir, '{}_{}_ph_dim_{}.txt'.format(exp_name, basename, d))
        stats_ofile = os.path.join(out_dir, '{}_{}_stats_dim_{}.txt'.format(exp_name, basename, d))
        np.savetxt(ph_ofile, pharr, delimiter='\t', fmt=['%.18e', '%.18e', '%d', '%d'])
        np.savetxt(stats_ofile, statsarr, delimiter='\t', fmt=['%.18e', '%.18e', '%.18e', '%.18e', '%.18e'])

        # Plot stats
        fig, ax1 = plt.subplots()
        xlist, npent_list, pnorm_list, maxnorm_list = statsarr[1:,0], statsarr[1:, 1], statsarr[1:, 3], statsarr[1:, 4]
        ax1.set_ylabel('Normalize persistent entropy', fontsize=16)
        ax1.scatter(xlist, npent_list, c=cycle[0], label='norm-ent')
        ax1.set_xlabel(r"Tunneling " r"$J/U$", fontsize=16)
        ax2 = ax1.twinx()
        ax2.scatter(xlist, pnorm_list, c=cycle[2], label='p-norm')
        ##ax2.scatter(xlist, maxnorm_list, c = 'green', label = 'inf-norm')
        ax2.set_ylabel(r"p-norm persistence", fontsize=16)
        ax2.legend(bbox_to_anchor=(1, 0.4), loc='upper right', borderaxespad=1, fontsize=16)
        ax1.legend(bbox_to_anchor=(1, 0.6), loc='lower right', borderaxespad=1, fontsize=16)
        fig_ofile = os.path.join(args.odir, '{}_{}_fig_dim_{}.pdf'.format(exp_name, basename, d))
        plt.savefig(fig_ofile, bbox_inches='tight')
    # remove tmpdir
    import shutil
    shutil.rmtree(tmpdir)

