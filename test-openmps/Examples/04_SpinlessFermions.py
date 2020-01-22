import MPSPyLib as mps
import numpy as np
import sys
import os.path

def main(PostProcess=False):
    """
    Introductory example for spinless fermions showing the effect of the phase
    term in the correlation measurement of the single particle density matrix.
    Two modes are available when running the example from command line:

    * ``python IsingStatics.py --PostProcess=F`` : runs the MPSFortLib to
      determine the ground state statics of the Ising model and print them.
      (default if ``--PostProcess`` not present.)
    * ``python IsingStatics.py --PostProcess=T`` : printing the results of the
      simulation run before.
    """

    # Build operators
    Operators = mps.BuildFermiOperators()

    # Define Hamiltonian MPO
    H = mps.MPO(Operators)
    H.AddMPOTerm('bond', ['fdagger', 'f'], hparam='t', weight=-1.0, Phase=True)

    # Observables
    myObservables = mps.Observables(Operators)
    # Correlation functions
    myObservables.AddObservable('corr', ['fdagger','f'], 'bspdm')
    myObservables.AddObservable('corr', ['fdagger','f'], 'spdm', Phase=True)

    # Convergence data
    myConv = mps.MPSConvParam(max_bond_dimension=30, max_num_sweeps=2)

    t = 1.0
    L = 10
    N = 5

    # Define statics
    parameters = [{ 
        # Directories
        'simtype'                   : 'Finite',
        'job_ID'                    : 'SpinlessFermions_',
        'unique_ID'                 : 'L_' + str(L) + 'N' + str(N),
        'Write_Directory'           : 'TMP_04/',
        'Output_Directory'          : 'OUTPUTS_04/',
        # System size and Hamiltonian parameters
        'L'                         : L,
        't'                         : t, 
        'verbose'                   : 2,
        'logfile'                   : True,
        # Specification of symmetries and good quantum numbers
        'Abelian_generators'        : ['nftotal'],
        'Abelian_quantum_numbers'   : [N],
        'MPSObservables'            : myObservables,
        'MPSConvergenceParameters'  : myConv
    }]

    # Write Fortran-readable main files
    MainFiles = mps.WriteFiles(parameters, Operators, H,
                               PostProcess=PostProcess)

    # Run the simulations
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

    spdm = Outputs[0]['spdm']
    spdmeigs, U = np.linalg.eigh(spdm)
    bspdm = Outputs[0]['bspdm']
    bspdmeigs, U = np.linalg.eigh(bspdm)

    print('Eigenvalues of <f^{\dagger}_i f_j> with Fermi phases', spdmeigs)
    print('Eigenvalues of <f^{\dagger}_i f_j> without Fermi phases', bspdmeigs)

    return


if(__name__ == '__main__'):
    # Check for command line arguments
    Post = False
    for arg in sys.argv[1:]:
        key, val = arg.split('=')
        if(key == '--PostProcess'): Post = (val == 'T') or (val == 'True')

    # Run main function
    main(PostProcess=Post)
