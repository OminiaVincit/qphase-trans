import MPSPyLib as mps
import numpy as np
import sys
import os.path

def main(PostProcess=False):
    """
    Introductory example for openMPS to simulate a fermionic system with
    long-range interactions. Two modes are available when running the
    example from command line:

    * ``python LongRangeTunneling.py --PostProcess=F`` : runs the MPSFortLib to
      determine the ground state statics (initial state).
      (default if ``--PostProcess`` not present.)
    * ``python LongRangeTunneling.py --PostProcess=T`` : printing the results
      of the simulation run before.
    """

    # Build operators
    Operators = mps.BuildFermiOperators()

    # Define Hamiltonian MPO
    H = mps.MPO(Operators)
    H.AddMPOTerm('FiniteFunction', ['fdagger','f'], f=[1.0, -0.2],
                 hparam='t', weight=-1.0, Phase=True)

    # Observables
    myObservables = mps.Observables(Operators)
    # Site terms
    myObservables.AddObservable('site', 'nftotal', 'n')
    # Correlation functions
    myObservables.AddObservable('corr', ['fdagger', 'f'], 'spdm', Phase=True)

    # Convergence parameters
    myConv = mps.MPSConvParam(max_bond_dimension=30, max_num_sweeps=2)
    myConv.AddModifiedConvergenceParameters(0, ['max_bond_dimension',
                                                'local_tol'], [50, 1E-14])

    # Specify constants and parameter list
    t = 1.0
    L = 10
    N = 5

    parameters = [{
        'simtype'                   : 'Finite',
        # Directories
        'job_ID'                    : 'LongRangeTunneling_',
        'unique_ID'                 : 'L_' + str(L) + 'N' + str(N),
        'Write_Directory'           : 'TMP_05/',
        'Output_Directory'          : 'OUTPUTS_05/',
        # System size and Hamiltonian parameters
        'L'                         : L,
        't'                         : t,
        # Specification of symmetries and good quantum numbers
        'Abelian_generators'        : ['nftotal'],
        'Abelian_quantum_numbers'   : [N],
        'MPSObservables'            : myObservables,
        'MPSConvergenceParameters'  : myConv,
        'logfile'                   : True
    }]

    # Write Fortran-readable main files
    MainFiles = mps.WriteFiles(parameters, Operators, H,
                               PostProcess=PostProcess)

    # Run the simulations and quit if not just post processing
    if(not PostProcess):
        if os.path.isfile('./Execute_MPSMain'):
            RunDir = './'
        else:
            RunDir = None
        mps.runMPS(MainFiles, RunDir=RunDir)
        return

    # Postprocessing
    # --------------

    Outputs = mps.ReadStaticObservables(parameters)

    # Get observables of state computed with most stringent convergence criteria
    fullyconvergedOutputs = mps.GetObservables(Outputs,
                                               'convergence_parameter', 2)
    spdm = fullyconvergedOutputs[0]['spdm']
    spdmeigs, U = np.linalg.eigh(spdm)
    print('Eigenvalues of <f^{\dagger}_i f_j>', spdmeigs)

    return


if(__name__ == '__main__'):
    # Check for command line arguments
    Post = False
    for arg in sys.argv[1:]:
        key, val = arg.split('=')
        if(key == '--PostProcess'): Post = (val == 'T') or (val == 'True')

    # Run main function
    main(PostProcess=Post)
