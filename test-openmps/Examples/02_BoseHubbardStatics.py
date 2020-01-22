import MPSPyLib as mps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
import sys
import os.path


def main(PostProcess=False, ShowPlots=True):
    """
    Introductory example for openMPS to simulate the finite size statics of
    the Bose-Hubbard model using number conservation. Two modes are available
    when running the example from command line:

    * ``python BoseHubbardStatics.py --PostProcess=F`` : runs the MPSFortLib to
      determine the ground state statics (initial state) and then the dynamics.
      (default if ``--PostProcess`` not present.)
    * ``python BoseHubbardStatics.py --PostProcess=T`` : plotting the results
      of the simulation run before.
    * The option ``--ShowPlots=T`` (``--ShowPlots=F``) lets you turn on (off)
      the GUI windows with the plots. Default to on if not passed to the
      command line.
    """
    # Cannot use time.clock as it does not capture subprocesses
    t0 = time()

    # Build operators
    Operators = mps.BuildBoseOperators(6)
    Operators['interaction'] = 0.5 * (np.dot(Operators['nbtotal'],
                                             Operators['nbtotal'])
                                      - Operators['nbtotal'])
    # Define Hamiltonian MPO
    H = mps.MPO(Operators)
    H.AddMPOTerm('bond', ['bdagger','b'], hparam='t', weight=-1.0)
    H.AddMPOTerm('site', 'interaction', hparam='U', weight=1.0)

    # ground state observables
    myObservables = mps.Observables(Operators)
    # Site terms
    myObservables.AddObservable('site', 'nbtotal', 'n')
    # correlation functions
    myObservables.AddObservable('corr', ['nbtotal', 'nbtotal'], 'nn')
    myObservables.AddObservable('corr', ['bdagger', 'b'], 'spdm')

    myConv = mps.MPSConvParam(max_num_sweeps=7)

    U = 1.0
    tlist = np.linspace(0, 0.4, 21)
    parameters = []
    L = 10
    Nlist = np.linspace(1, 11, 11)

    for t in tlist:
        for N in Nlist:
            parameters.append({
                'simtype'                   : 'Finite',
                # Directories
                'job_ID'                    : 'Bose_Hubbard_statics',
                'unique_ID'                 : 't_' + str(t) + 'N_' + str(N),
                'Write_Directory'           : 'TMP_02/',
                'Output_Directory'          : 'OUTPUTS_02/',
                # System size and Hamiltonian parameters
                'L'                         : L,
                't'                         : t,
                'U'                         : U,
                # Specification of symmetries and good quantum numbers
                'Abelian_generators'        : ['nbtotal'],
                # Define Filling
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
        return

    # Postprocessing and plotting
    # ---------------------------

    Outputs = mps.ReadStaticObservables(parameters)

    depletion = np.zeros((Nlist.shape[0], tlist.shape[0]))
    energies = np.zeros((Nlist.shape[0], tlist.shape[0]))
    tinternal = np.zeros((Nlist.shape[0], tlist.shape[0]))
    chempotmu = np.zeros((Nlist.shape[0], tlist.shape[0]))

    kk = -1
    for ii in range(tlist.shape[0]):
        tinternal[:, ii] = tlist[ii]

        for jj in range(Nlist.shape[0]):
            kk += 1

            spdm = np.linalg.eigh(Outputs[kk]['spdm'])[0]
            depletion[jj, ii] = 1.0 - np.max(spdm) / np.sum(spdm)

            energies[jj, ii] = Outputs[kk]['energy']

        chempotmu[0, ii] = energies[0, ii]
        chempotmu[1:, ii] = energies[1:, ii] - energies[:-1, ii]

    if(ShowPlots):
        plotIt(tinternal, chempotmu, depletion)

    return


def plotIt(jvalues, muvalues, dependentvalues):
    """
    Scatter plot for the phase diagram of the Bose Hubbard model.

    **Arguments**

    jvalues : floats
        x-values of points

    muvalues : floats
        y-values of points

    dependentvalues : floats
        value for coloring points
    """
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.scatter(jvalues, muvalues, c=dependentvalues, cmap=cm.jet)
    plt.xlim((np.min(jvalues), np.max(jvalues)))
    plt.ylim((np.min(muvalues), np.max(muvalues)))
    plt.xlabel(r"tunneling  " r"$t/U$", fontsize=16)
    plt.ylabel(r"chemical potential  " r"$\mu/U$", fontsize=16)
    cbar = plt.colorbar()
    cbar.set_label(r"Quantum Depletion", fontsize=16)
    plt.savefig('02_BoseHubbardStatics.pdf', bbox_inches='tight')
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
