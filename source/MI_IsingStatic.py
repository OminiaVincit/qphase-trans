import MPSPyLib as mps
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path

def main(PostProcess=False, ShowPlots=True):
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
    Operators['sigmaz'] = 2 * Operators['sz']
    Operators['sigmax'] = (Operators['splus'] + Operators['sminus'])
    Operators['gen'] = np.array([[0,0], [0, 1.]])
    # Define Hamiltonian of transverse Ising model
    H = mps.MPO(Operators)
    # Note the J parameter in the transverse Ising Hamiltonian
    # has been set to 1, we are modelling a ferromagnetic chain.
    H.AddMPOTerm('bond', ['sigmaz', 'sigmaz'], hparam='J', weight=-1.0)
    H.AddMPOTerm('site', 'sigmax', hparam='g', weight=-1.0)

    # Observables and convergence parameters
    myObservables = mps.Observables(Operators)
    myObservables.AddObservable('MI', True)

    myConv = mps.MPSConvParam(max_bond_dimension=60, max_num_sweeps=6,
                              local_tol=1E-14)

    # Specify constants and parameter lists
    J = 1.0
    glist = np.linspace(0.1, 2.1, 21)
    parameters = []
    L = 128 

    for g in glist:
        parameters.append({
            'simtype'                   : 'Finite',
            # Directories
            'job_ID'                    : 'Ising_Statics',
            'unique_ID'                 : 'g_' + str(g),
            'Write_Directory'           : 'MI_TMP_01_L_{}/'.format(L),
            'Output_Directory'          : 'MI_OUTPUTS_01_L_{}/'.format(L),
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

    MI_list = []
    Outputs = mps.ReadStaticObservables(parameters)
    for Output in Outputs:
        print(Output['converged'])

        # Get mutual information 
        MI_list.append(Output['MI'][3][26])

    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.scatter(glist, MI_list)
    plt.xlabel(r"transverse field coupling  " r"$g$", fontsize=16)
    plt.ylabel(r"Mutual information "
               r"(4, 27)",
               fontsize=16)
    if(ShowPlots):
      plt.savefig('MI_check_IsingStatics_L_{}.pdf'.format(L), bbox_inches='tight')
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

    # Run main function
    main(PostProcess=Post, ShowPlots=Plot)
