import MPSPyLib as mps
import numpy as np
from scipy.special import sici
from math import pi
import sys
import os.path

def main(PostProcess=False):
    """
    Introductory example for openMPS to simulate the Haldana Shastry model with
    infinite MPS. Two modes are available when running the example from command
    line:

    * ``python IsingStatics.py --PostProcess=F`` : runs the MPSFortLib to
      determine the ground state statics of the Ising model. (default if
      ``--PostProcess`` not present.)
    * ``python IsingStatics.py --PostProcess=T`` : plotting the results of the
      simulation run before.
    """

    # Build operators
    Operators = mps.BuildSpinOperators(spin=0.5)

    # Define Hamiltonian MPO
    H = mps.MPO(Operators, PostProcess=PostProcess)
    invrsq = lambda x: 1.0 / (x**2)
    H.AddMPOTerm('InfiniteFunction', ['splus', 'sminus'], hparam='J_xy',
                 weight=0.5, func=invrsq, L=1000, tol=1e-9)
    H.AddMPOTerm('InfiniteFunction', ['sz', 'sz'], hparam='J_z',
                 weight=1.0, func=invrsq, L=1000, tol=1e-9)

    # Ground state observables
    myObservables = mps.Observables(Operators)
    # Site terms
    myObservables.AddObservable('site', 'sz', 'z')
    # Correlation functions
    myObservables.AddObservable('corr', ['sz', 'sz'], 'zz')
    myObservables.AddObservable('corr', ['splus', 'sminus'], 'pm')
    # Get correlation functions out to a distance of 1000
    myObservables.SpecifyCorrelationRange(1000)

    # Convergence parameters
    myConv = mps.iMPSConvParam(max_bond_dimension=12, variance_tol=-1.0,
                               max_num_imps_iter=1000)

    mod_list = ['max_bond_dimension', 'max_num_imps_iter']
    myConv.AddModifiedConvergenceParameters(0, mod_list, [20, 500])
    myConv.AddModifiedConvergenceParameters(0, mod_list, [40, 250])

    # Long run time (Enable if you prefer)
    #myConv.AddModifiedConvergenceParameters(0, mod_list, [60, 250])
    #myConv.AddModifiedConvergenceParameters(0, mod_list, [80, 250])

    L = 2
    parameters = [{ 
        'simtype'                   : 'Infinite',
        # Directories
        'job_ID'                    : 'HaldaneShastry',
        'Write_Directory'           : 'TMP_06/',
        'Output_Directory'          : 'OUTPUTS_06/', 
        # System size and Hamiltonian parameters
        'L'                         : L,
        'J_z'                       : 1.0, 
        'J_xy'                      : 1.0, 
        # Convergence parameters
        'MPSObservables'            : myObservables,
        'MPSConvergenceParameters'  : myConv,
        'logfile'                   : True
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

    # Postprocessing and plotting
    Outputs = mps.ReadStaticObservables(parameters)
    clfilename = parameters[0]['job_ID'] + 'correlationLength.dat'
    clfile = open(clfilename, 'w')

    for Output in Outputs:
        chi = Output['bond_dimension']
        print('Chi', chi)
        print('energy density', Output['energy_density'])

        corrfilename = parameters[0]['job_ID'] + 'chi' + str(chi) + 'corr.dat'
        corrfile = open(corrfilename, 'w')

        for ii in range(1, myObservables.correlation_range):
            (tmp, ci) = sici((ii) * pi)
            ex = tmp / (4.0 * pi * (ii))
            corrfile.write('%16i'%(ii) + '%30.15E'%(Output['zz'][ii])
                           +'%30.15E'%(Output['pm'][ii]) + '%30.15E'%(ex)
                           + '\n')
        corrfile.close()
        clfile.write('%16i'%(chi) + '%30.15E'%(Output['Correlation_length']) + '\n')
        (tmp, ci) = sici((1.0) * pi)
        ex = tmp / (4.0 * pi * (1.0))

        print(sum(Output['z']), Output['zz'][1], Output['zz'][2], ex)

    clfile.close()

    return


if(__name__ == '__main__'):
    # Check for command line arguments
    Post = False
    for arg in sys.argv[1:]:
        key, val = arg.split('=')
        if(key == '--PostProcess'): Post = (val == 'T') or (val == 'True')

    # Run main function
    main(PostProcess=Post)
