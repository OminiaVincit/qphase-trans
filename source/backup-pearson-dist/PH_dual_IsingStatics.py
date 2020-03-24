import MPSPyLib as mps
import networkmeasures as nm
import ph_utils as ph
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path
import time

def main(PostProcess=False, ShowPlots=True, Persistent=False, pdim=0, L=30):
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
    #myObservables.AddObservable('DensityMatrix_i', [])
    #myObservables.AddObservable('DensityMatrix_ij', [])
    myObservables.AddObservable('MI', True)
    myConv = mps.MPSConvParam(max_bond_dimension=200, 
                variance_tol = 1E-10,
                local_tol = 1E-10,
                max_num_sweeps = 6)
    #modparam = ['max_bond_dimension', 'max_num_sweeps']
    #myConv.AddModifiedConvergenceParameters(0, modparam, [80, 4])

    # Specify constants and parameter lists
    J = 1.0
    glist = np.linspace(0.1, 2.1, 41)
    parameters = []

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
        if Persistent:
            netmeasures = nm.pearson(Output['MI'])
            dist = 1.0 - np.abs(netmeasures[0])
            for i in range(L):
                dist[i, i] = 0.0
            dmat_list.append(dist)
        MI_list.append(Output['MI'][3][26])
    plt.style.use('seaborn-colorblind')
    #plt.style.use('seaborn-whitegrid')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    
    timestamp = int(time.time() * 1000.0)
    
    if (not Persistent):
        plt.scatter(glist, MI_list)
        plt.xlabel(r"transverse field coupling  " r"$g$", fontsize=16)
        plt.ylabel(r"Mutual information (4,27)", fontsize=16)
        plt.savefig('MI_L_{}_{}.pdf'.format(L, timestamp), bbox_inches='tight')
        plt.show()
        return
    
    npent_list, pent_list, pnorm_list = ph.compute_ph(out_path = './diagrams_{}_{}'.format(L, timestamp), fig_path = './figs_{}_{}'.format(L, timestamp), basename = 'ising', dmat_list = dmat_list, plot=True, pdim=pdim)
    
    # Save txt
    np.savetxt('stats_L_{}_{}.txt'.format(L, timestamp), list(zip(glist, npent_list, pent_list, pnorm_list)), fmt='%.18g')    

    # Plot
    fig, ax1 = plt.subplots()
    ax1.scatter(glist, pnorm_list, c = 'red')
    ax1.set_xlabel(r"transverse field coupling  " r"$g$", fontsize=16)
    ax1.set_ylabel(r"p-norm persistence", fontsize=16)
    ax2 = ax1.twinx()
    ax2.set_ylabel('normalize persistent entropy', fontsize=16)
    ax2.scatter(glist, npent_list, c = 'blue')
    #plt.xlim((0, 2))
    #plt.ylim((0, 1))
    if(ShowPlots):
      plt.savefig('IsingStatic_L_{}_ph_dim_{}_{}.pdf'.format(L, pdim, timestamp), bbox_inches='tight')
      plt.show()

    return

if(__name__ == '__main__'):
    # Check for command line arguments
    Post = False
    Plot = True
    pdim = 0
    Persistent = False
    L = 30

    for arg in sys.argv[1:]:
        key, val = arg.split('=')
        if(key == '--PostProcess'): Post = (val == 'T') or (val == 'True')
        if(key == '--ShowPlots'): Plot = (val == 'T') or (val == 'True')
        if(key == '--Persistent'): Persistent = (val == 'T') or (val == 'True')
        if(key == '--pdim'): pdim = int(val) 
        if(key == '--Size'): L = int(val)

    print('Post={}, Plot={}, Persistent={}, pdim={}, L={}'.format(Post, Plot, Persistent, pdim, L))

    # Run main function
    main(PostProcess=Post, ShowPlots=Plot, Persistent=Persistent, pdim=pdim, L=L)
