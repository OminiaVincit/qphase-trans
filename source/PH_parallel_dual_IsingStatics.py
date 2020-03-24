import MPSPyLib as mps
import networkmeasures as nm
import distancemeasures as dm
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
        Runs the MPSFortLib to determine the ground state statics of the Ising model
    PostProcess = True:
        Obtain the simulated results from the already run simulation
    """

    # Build spin operators for spin-1/2 system
    # Transform to perserve Z2 symmetry
    Operators = mps.BuildSpinOperators(0.5)
    Operators['sigmax'] = 2 * Operators['sz']
    Operators['sigmaz'] = (Operators['splus'] + Operators['sminus'])
    Operators['gen'] = np.array([[0, 0], [0, 1.]])
    # Define Hamiltonian of transverse Ising model
    H = mps.MPO(Operators)
    # Note the J parameter in the transverse Ising Hamiltonian
    # has been set to 1, we are modelling a ferromagnetic chain.
    H.AddMPOTerm('bond', ['sigmaz', 'sigmaz'], hparam='J', weight=-1.0)
    H.AddMPOTerm('site', 'sigmax', hparam='g', weight=-1.0)

    # Observables and convergence parameters
    myObservables = mps.Observables(Operators)
    #myObservables.AddObservable('corr', ['sigmaz', 'sigmaz'], 'zz')
    myObservables.AddObservable('DensityMatrix_i', [])
    myObservables.AddObservable('DensityMatrix_ij', [])
    myObservables.AddObservable('MI', True)
    myConv = mps.MPSConvParam(max_bond_dimension=200, 
                variance_tol = 1E-10,
                local_tol = 1E-10,
                max_num_sweeps = 6)
    #modparam = ['max_bond_dimension', 'max_num_sweeps']
    #myConv.AddModifiedConvergenceParameters(0, modparam, [80, 4])

    # Specify constants and parameter lists
    J = 1.0
    glist = np.linspace(0.1, 2.1, 81)
    parameters = []

    for g in glist:
        parameters.append({
            'simtype'                   : 'Finite',
            # Directories
            'job_ID'                    : 'Ising_Statics',
            'unique_ID'                 : 'g_' + str(g),
            'Write_Directory'           : '{}/TMP_ISING_01_L_{}/'.format(ExpName, L),
            'Output_Directory'          : '{}/OUTPUTS_ISING_01_L_{}/'.format(ExpName, L),
            # System size and Hamiltonian parameters
            'L'                         : L,
            'J'                         : J,
            'g'                         : g,
            # ObservablesConvergence parameters
            'verbose'                   : 1,
            'MPSObservables'            : myObservables,
            'MPSConvergenceParameters'  : myConv,
            'logfile'                   : True,
            # Z2 symmetry
            'Discrete_generators': ['gen'],
            'Discrete_quantum_numbers': [0]
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
        return None, None

    # PostProcess
    # -----------
    MI_list = []
    Outputs = mps.ReadStaticObservables(parameters)
    for Output in Outputs:
        print(Output['converged'])
        MI_list.append(Output['MI'][3][26])
    
    timestamp = int(time.time() * 1000.0)
    if ShowPlot:
        plt.style.use('seaborn-colorblind')
        #plt.style.use('seaborn-whitegrid')
        plt.rc('font', family='serif')
        plt.rc('mathtext', fontset='cm')
        plt.scatter(glist, MI_list)
        plt.xlabel(r"transverse field coupling  " r"$g$", fontsize=16)
        plt.ylabel(r"Mutual information (4,27)", fontsize=16)
        plt.savefig('{}/MI_L_{}_{}.pdf'.format(exp_name, L, timestamp), bbox_inches='tight')
        plt.show()
    
    return Outputs, glist

def persistent_compute(mi_mats, max_dim, bg, time_stamp, basename, tmpdir, process_id, tlabel):
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

        # Calculate distance matrix
        #print(process_id, i, bg, idx, mimat[3][26])
        if tlabel == 'MI':
            netmeasures = nm.pearson(mimat)
            dist = np.sqrt(1.0 - netmeasures[0]**2)
            for j in range(dist.shape[0]):
                dist[j, j] = 0
        else:
            # trace dist, bures_dist, bures_angle
            dist = dm.compute_distance(mimat, tlabel)         
                    
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
    parser.add_argument('--odir', type=str, default='ising3')
    parser.add_argument('--nproc', type=int, default=16)
    parser.add_argument('--maxdim', type=int, default=1)
    parser.add_argument('--dist', type=str, default='MI')
    args = parser.parse_args()
    print(args)
    post_flag, plot_flag = (args.postprocess > 0), (args.plot > 0)
    persis_flag, L = (args.persistent > 0), args.size
    exp_name, out_dir, nproc = args.exp, args.odir, args.nproc
    max_dim = args.maxdim
    tlabel = args.dist

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
    basename = 'ising_L_{}_{}'.format(L, tlabel)
    tmpdir = os.path.join(args.odir, '{}_{}'.format(exp_name, time_stamp))

    #with tp.TemporaryDirectory() as tmpdir:
    #print(os.path.exists(tmpdir), tmpdir)
    for proc_id in range(nproc):
        outlist = lst[proc_id]
        print(outlist)
        bg = outlist[0]
        
        # create density matrix
        mils = []
        if tlabel == 'MI':
            mils = [sim_outs[j]['MI'] for j in outlist]
        else:
            for j in outlist:
                sout = sim_outs[j]
                rhols = [sout['rho_{}'.format(i+1)] for i in range(L)]
                mils.append(rhols)

        #print(sim_outs[bg].keys()) 
        
        p = Process(target=persistent_compute, args=(mils, max_dim, bg, time_stamp, basename, tmpdir, proc_id, tlabel))
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
            sarr = np.loadtxt(statfile)
            sarr = arr_reshape(sarr)
            if sarr is not None:
                statslist.append(sarr)
        
        pharr = np.concatenate(phlist, axis=0)
        #print(statslist)
        #print(len(statslist), statslist[0].shape)
        statsarr = np.concatenate(statslist, axis=0)
        statsarr[:, 0] = glist
        print(d, pharr.shape, statsarr.shape)
        # Save diagrams and statistic file to txt file
        ph_ofile = os.path.join(args.odir, '{}_{}_ph_dim_{}.txt'.format(exp_name, basename, d))
        stats_ofile = os.path.join(args.odir, '{}_{}_stats_dim_{}.txt'.format(exp_name, basename, d))
        np.savetxt(ph_ofile, pharr, delimiter='\t', fmt=['%.18e', '%.18e', '%d', '%d'])
        np.savetxt(stats_ofile, statsarr, delimiter='\t', fmt=['%.18e', '%.18e', '%.18e', '%.18e', '%.18e'])

        # Plot stats
        fig, ax1 = plt.subplots()
        npent_list, pnorm_list, maxnorm_list = statsarr[:, 1], statsarr[:, 3], statsarr[:, 4]
        ax1.set_ylabel('Normalize persistent entropy', fontsize=16)
        ax1.scatter(glist, npent_list, c = 'blue', label='norm-ent')
        ax1.set_xlabel(r"Transverse field coupling " r"$g$", fontsize=16)
        ax2 = ax1.twinx()
        ax2.scatter(glist, pnorm_list, c = 'red', label='p-norm')
        ax2.scatter(glist, maxnorm_list, c = 'green', label = 'inf-norm')
        ax2.set_ylabel(r"p-norm persistence", fontsize=16)
        ax2.legend(bbox_to_anchor=(1, 0.9), loc='upper right', borderaxespad=1, fontsize=16)
        ax1.legend(bbox_to_anchor=(1, 0.3), loc='lower right', borderaxespad=1, fontsize=16)
        fig_ofile = os.path.join(args.odir, '{}_{}_fig_dim_{}.pdf'.format(exp_name, basename, d))
        plt.savefig(fig_ofile, bbox_inches='tight')
    # remove tmpdir
    import shutil
    shutil.rmtree(tmpdir)

