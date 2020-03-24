import MPSPyLib as mps
import networkmeasures as nm
import ph_utils as ph
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path
import time

def main(PostProcess=False, ShowPlots=True, pdim=0):
    """
    Introductory example for openMPS to simulate the finite size statics of
    the transverse Ising model. Two modes are available when running the
    example from command line:

    * ``python IsingStatics.py --PostProcess=F`` : runs the MPSFortLib to
      determine the ground state statics of the Ising model. (default if
      ``--PostProcess`` not present.)
    * ``python IsingStatics.py --PostProcess=T`` : plotting the results of the
      simulation run before.
    * The option ``--ShowPlots=T`` (``--ShowPlots=F``) lets you turn on (off)
      the GUI windows with the plots. Default to on if not passed to the
      command line.
    """

    # Build spin operators for spin-1/2 system
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
    #myObservables.AddObservable('DensityMatrix_i', [])
    #myObservables.AddObservable('DensityMatrix_ij', [])
    myObservables.AddObservable('MI', True)
    myConv = mps.MPSConvParam(max_bond_dimension=20, max_num_sweeps=8,
                              local_tol=1E-14)

    # Specify constants and parameter lists
    J = 1.0
    glist = np.linspace(0.1, 2.1, 41)
    parameters = []
    L = 64 

    for g in glist:
        parameters.append({
            'simtype'                   : 'Finite',
            # Directories
            'job_ID'                    : 'Ising_Statics',
            'unique_ID'                 : 'g_' + str(g),
            'Write_Directory'           : 'TMP_ISING_01_L_{}/'.format(L),
            'Output_Directory'          : 'OUTPUTS_ISING_01_L_{}/'.format(L),
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
        return

    # PostProcess
    # -----------
    dmat_list = []
    MI_list = []
    Outputs = mps.ReadStaticObservables(parameters)
    for Output in Outputs:
        print(Output['converged'])
        #print(Output.keys())

        #MI_list.append(Output['MI'][3][26])
        #for i in range(L):
        #    for j in range(i+1, L):
        #        rho1 = Output['rho_{}'.format(i+1)]
        #        rho2 = Output['rho_{}'.format(j+1)]
        #        rho12 = Output['rho_{}_{}'.format(i+1, j+1)]
        #        rho = np.kron(rho1, rho2)
        #        dist[i, j] = np.linalg.norm(rho12 - rho, ord = 'nuc')
        #        dist[j, i] = dist[i, j]
        netmeasures = nm.pearson(Output['MI'])
        dist = 1.0 - np.abs(netmeasures[0])
        for i in range(L):
            dist[i, i] = 0.0
        dmat_list.append(dist)
        #print(dist.dtype)
        #pent_list.append(dist[3, 26])
        # Check distance metric
        #for i in range(L):
        #    for j in range(i+1, L):
        #        for k in range(j+1, L):
        #            if (dist[i, j] + dist[i, k] < dist[j, k]) or (dist[i, k] + dist[j, k] < dist[i, j]) or (dist[j, i] + dist[j, k] < dist[i, k]):
        #                print(i, j, k, dist[i, j], dist[i, k], dist[j, k])
        #                #return
    
        #MI_list.append(np.linalg.norm(rho12 - rho, ord='nuc'))
    npent_list, pent_list, pnorm_list = ph.compute_ph(out_path = './diagrams', fig_path = './figs', basename = 'ising', dmat_list = dmat_list, plot=True, pdim=pdim)
    plt.style.use('seaborn-colorblind')
    #plt.style.use('seaborn-whitegrid')

    fig, ax1 = plt.subplots()
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    ax1.scatter(glist, pnorm_list, c = 'red')
    ax1.set_xlabel(r"transverse field coupling  " r"$g$", fontsize=16)
    ax1.set_ylabel(r"p-norm persistence", fontsize=16)
    ax2 = ax1.twinx()
    ax2.set_ylabel('normalize persistent entropy', fontsize=16)
    ax2.scatter(glist, npent_list, c = 'blue')
    #plt.xlim((0, 2))
    #plt.ylim((0, 1))
    if(ShowPlots):
      timestamp = int(time.time() * 1000.0)
      plt.savefig('IsingStatic_L_{}_ph_dim_{}_{}.pdf'.format(L, pdim, timestamp), bbox_inches='tight')
      plt.show()

    return


if(__name__ == '__main__'):
    # Check for command line arguments
    Post = False
    Plot = True

    for arg in sys.argv[1:]:
        key, val = arg.split('=')
        if(key == '--PostProcess'): Post = (val == 'T') or (val == 'True')
        if(key == '--ShowPlots'): Plot = (val == 'T') or (val == 'True')
        if(key == '--pdim'): pdim = int(val) 
    
    print('pdim={}'.format(pdim))

    # Run main function
    main(PostProcess=Post, ShowPlots=Plot, pdim=pdim)
