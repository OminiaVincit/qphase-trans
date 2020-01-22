import MPSPyLib as mps
import numpy as np
from sys import version_info
import matplotlib.pyplot as plt
import sys
import os.path

def main(PostProcess=False, ShowPlots=True):
    """
    Introductory example for openMPS to simulate the finite size dynamics of
    the Bose-Hubbard model with number conservation. Two modes are available
    when running the example from command line:

    * ``python BoseHubbardDynamics.py --PostProcess=F`` : runs the MPSFortLib to
      determine the ground state statics (initial state) and then the dynamics.
      (default if ``--PostProcess`` not present.)
    * ``python BoseHubbardDynamics.py --PostProcess=T`` : plotting the results
      of the simulation run before.
    * The option ``--ShowPlots=T`` (``--ShowPlots=F``) lets you turn on (off)
      the GUI windows with the plots. Default to on if not passed to the
      command line.
    """

    # Build operators
    Operators = mps.BuildBoseOperators(5)
    Operators['interaction'] = 0.5 * (np.dot(Operators['nbtotal'],
                                             Operators['nbtotal'])
                                      - Operators['nbtotal'])

    # Define Hamiltonian MPO
    H = mps.MPO(Operators)
    H.AddMPOTerm('bond', ['bdagger', 'b'], hparam='t', weight=-1.0)
    H.AddMPOTerm('site', 'interaction', hparam='U', weight=1.0)

    # Ground state observables
    myObservables = mps.Observables(Operators)
    # Site terms
    myObservables.AddObservable('site', 'nbtotal', 'n')
    # Correlation functions
    myObservables.AddObservable('corr', ['nbtotal','nbtotal'], 'nn')
    myObservables.AddObservable('corr', ['bdagger','b'], 'spdm')

    # Convergence parameters
    myConv = mps.MPSConvParam(max_bond_dimension=20, max_num_sweeps=6)
    myConv.AddModifiedConvergenceParameters(0, ['max_bond_dimension',
                                                'local_tol'], [50, 1E-14])

    myDynConv = mps.TDVPConvParam(max_num_lanczos_iter=20,
                                       lanczos_tol=1E-6)

    # Dynamics time evolution observables
    dynObservables = mps.Observables(Operators)
    # Site terms
    dynObservables.AddObservable('site', 'nbtotal', name='n')

    # Specify constants and parameter lists
    U = 10.0
    t = 1.0
    staticsParameters = []
    L = 6
    tlist = [5.0, 20.0]

    for tau in tlist:
        # Quench function ramping down
        def Ufuncdown(t, tau=tau):
            return 10.0 + 2.0 * (1.0 - 10.0) * t / tau
        # Quench function ramping back up
        def Ufuncup(t, tau=tau):
            return 1.0 + 2.0 * (10.0 - 1.0) * (t - 0.5 * tau) / tau

        Quenches = mps.QuenchList(H)
        Quenches.AddQuench(['U'], 0.5 * tau, min(0.5 * tau / 100.0, 0.1),
                           [Ufuncdown], ConvergenceParameters=myDynConv)
        Quenches.AddQuench(['U'], 0.5 * tau, min(0.5 * tau / 100.0, 0.1),
                           [Ufuncup], ConvergenceParameters=myDynConv)

        staticsParameters.append({
            'simtype'                   : 'Finite',
            # Directories
            'job_ID'                    : 'Bose_Hubbard_mod',
            'unique_ID'                 : 'tau_'+str(tau),
            'Write_Directory'           : 'TMP_03/',
            'Output_Directory'          : 'OUTPUTS_03/',
            # System size and Hamiltonian parameters
            'L'                         : L,
            't'                         : t, 
            'U'                         : U, 
            # Specification of symmetries and good quantum numbers
            'Abelian_generators'        : ['nbtotal'],
            'Abelian_quantum_numbers'   : [L],
            # Convergence parameters
            'verbose'                   : 1,
            'logfile'                   : True,
            'MPSObservables'            : myObservables,
            'MPSConvergenceParameters'  : myConv,
            'Quenches'                  : Quenches,
            'DynamicsObservables'       : dynObservables
        })

    # Write Fortran-readable main files
    MainFiles = mps.WriteFiles(staticsParameters, Operators, H,
                               PostProcess=PostProcess)

    # Run the simulations and quit if not just post processing
    if(not PostProcess):
        if os.path.isfile('./Execute_MPSMain'):
            RunDir = './'
        else:
            RunDir = None
        mps.runMPS(MainFiles, RunDir=RunDir)
        return

    # Postprocessing and plotting
    # ---------------------------

    Outputs = mps.ReadStaticObservables(staticsParameters)
    DynOutputs = mps.ReadDynamicObservables(staticsParameters)

    ii = 0
    le_list = [[], []]

    for t in tlist:
        myfile = 'outmod' + str(t) + '.dat'
        hfile = open(myfile, 'w')
        for p in DynOutputs[ii]:
            hfile.write('%30.15E'%(p['time']) + '%30.15E'%(p['U']) +
                        '%30.15E'%(p['Loschmidt_Echo'].real) +
                        '%30.15E'%(p['Loschmidt_Echo'].imag) +
                        '%30.15E'%(p['bond_dimension']) + '\n')
            le_list[ii].append(abs(p['Loschmidt_Echo'])**2)

        hfile.close()
        ii += 1

    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.plot(np.linspace(0, 1, len(le_list[0])), le_list[0], 'r-',
             label="tq=5")
    plt.plot(np.linspace(0, 1, len(le_list[1])), le_list[1], 'g-',
             label="tq=20")
    plt.xlabel(r"Time in percent " r"$t / t_q$", fontsize=16)
    plt.ylabel(r"Loschmidt Echo "
               r"$| \langle \psi(t) | \psi(0) \rangle |^2$",
               fontsize=16)
    plt.legend(loc="lower left")
    if(ShowPlots):
        plt.savefig('03_BoseHubbardDynamics.pdf', bbox_inches='tight')
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

